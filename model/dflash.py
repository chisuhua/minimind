"""DFlash: Block Diffusion Speculative Decoding with KV Injection"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """Extract hidden states from target model layers, concatenate and project"""
    def __init__(self, hidden_size, num_layers=5):
        super().__init__()
        self.proj = nn.Linear(hidden_size * num_layers, hidden_size, bias=False)

    def forward(self, layer_hidden_states):
        concat = torch.cat(layer_hidden_states, dim=-1)
        return self.proj(concat)


class KVInjector(nn.Module):
    """Inject target feature into draft model KV at every layer"""
    def __init__(self, hidden_size, num_heads=8, head_dim=96, num_draft_layers=1):
        super().__init__()
        kv_dim = num_heads * head_dim
        self.injections = nn.ModuleList([
            nn.Linear(hidden_size, 2 * kv_dim, bias=False) for _ in range(num_draft_layers)
        ])
        self.num_heads = num_heads
        self.head_dim = head_dim

    def inject(self, k, v, g_t, layer_id):
        kv_delta = self.injections[layer_id](g_t)
        dk, dv = kv_delta.chunk(2, dim=-1)
        B = dk.shape[0]
        dk = dk.view(B, 1, self.num_heads, self.head_dim)
        dv = dv.view(B, 1, self.num_heads, self.head_dim)
        return k + dk, v + dv


class DraftAttention(nn.Module):
    """Simplified attention for draft model (no RoPE, parallel block processing)"""
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.o_proj(out)
        return out, None


class DFlashDraftModel(nn.Module):
    """1-layer drafter with shared embedding/LM head + KV injection"""
    def __init__(self, config, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.hidden_size = config.hidden_size
        self.num_draft_layers = 1
        self.draft_attn = DraftAttention(config.hidden_size, config.num_attention_heads)
        self.draft_ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.draft_norm1 = nn.LayerNorm(config.hidden_size)
        self.draft_norm2 = nn.LayerNorm(config.hidden_size)
        self.feature_fusion = FeatureFusion(config.hidden_size, num_layers=5)
        kv_heads = getattr(config, 'num_key_value_heads', 4) or 4
        head_dim = config.head_dim
        self.kv_injector = KVInjector(config.hidden_size, num_heads=kv_heads, head_dim=head_dim, num_draft_layers=1)

    def forward(self, block_input_emb, g_t=None):
        B = block_input_emb.shape[0]
        hidden = block_input_emb
        attn_out, _ = self.draft_attn(hidden)
        hidden = hidden + self.draft_norm1(attn_out)
        if g_t is not None:
            hidden = hidden + g_t.unsqueeze(1)
        ffn_out = self.draft_ff(self.draft_norm2(hidden))
        hidden = hidden + ffn_out
        return hidden


class BlockDiffusionDecoder:
    """Block diffusion: anchor + [MASK] -> denoise -> draft"""
    def __init__(self, drafter_model, target_lm_head, vocab_size, mask_token_id=3, block_size=16):
        self.drafter = drafter_model
        self.target_lm_head = target_lm_head
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.block_size = block_size

    @torch.inference_mode()
    def draft(self, hidden_states_list, anchor_pos, anchor_token, embed_tokens):
        B = anchor_token.shape[0]
        device = anchor_token.device
        g_t = self.drafter.feature_fusion([h[:, anchor_pos:anchor_pos+1] for h in hidden_states_list])
        g_t = g_t.squeeze(1)
        anchor_emb = embed_tokens(anchor_token)
        mask_emb = embed_tokens(torch.full((B, self.block_size - 1), self.mask_token_id, device=device))
        block_input = torch.cat([anchor_emb, mask_emb], dim=1)
        hidden = self.drafter(block_input, g_t=g_t)
        logits = self.target_lm_head(hidden)
        draft_tokens = logits.argmax(dim=-1)
        return draft_tokens


class DFlashSpecDecoder:
    """End-to-end DFlash decoding loop"""
    def __init__(self, dflash_draft_model, target_model, fusion_layers=None):
        self.draft_model = dflash_draft_model
        self.target = target_model
        self.fusion_layers = fusion_layers or [1, 3, 5, 7]
        self.block_decoder = BlockDiffusionDecoder(
            dflash_draft_model, target_model.lm_head,
            target_model.config.vocab_size,
            mask_token_id=3, block_size=dflash_draft_model.block_size
        )

    @torch.inference_mode()
    def step(self, input_ids, past_key_values, attention_mask):
        B = input_ids.shape[0]
        if B != 1:
            return None, None, past_key_values
        device = input_ids.device
        hidden_list = []

        def hook_fn(layer_id):
            def hook(module, input, output):
                hidden_list.append(output[0].detach())
            return hook

        hooks = []
        for lid in self.fusion_layers:
            h = self.target.model.layers[lid].register_forward_hook(hook_fn(lid))
            hooks.append(h)

        outputs = self.target(input_ids, past_key_values=past_key_values, use_cache=True, attention_mask=attention_mask)
        for h in hooks:
            h.remove()

        anchor_pos = past_key_values[0][0].shape[2] if past_key_values is not None and past_key_values[0] is not None else 0
        past_len = anchor_pos
        anchor_token = input_ids[:, -1:]

        draft_tokens = self.block_decoder.draft(
            hidden_list, anchor_pos, anchor_token,
            embed_tokens=self.target.model.embed_tokens
        )

        draft_input = torch.cat([anchor_token, draft_tokens[:, 1:]], dim=1)
        verify_out = self.target(draft_input, past_key_values=past_key_values, use_cache=True, attention_mask=None)
        verify_logits = verify_out.logits[0]

        accepted = [anchor_token[0, 0].item()]
        for i in range(draft_tokens.shape[1]):
            pred = verify_logits[i].argmax().item()
            if pred == draft_tokens[0, i].item():
                accepted.append(pred)
            else:
                break

        num_new = len(accepted) - 1
        if num_new < 1:
            return None, None, past_key_values

        new_tokens = torch.tensor([accepted[1:]], device=device)
        full_kv = verify_out.past_key_values
        if full_kv is not None:
            trunc_len = past_len + num_new
            truncated = []
            for layer_kv in full_kv:
                if layer_kv is not None:
                    k, v = layer_kv
                    truncated.append((k[:, :, :trunc_len].contiguous(), v[:, :, :trunc_len].contiguous()))
                else:
                    truncated.append(None)
            new_pkv = tuple(truncated)
        else:
            new_pkv = past_key_values

        return new_tokens, new_pkv, past_key_values

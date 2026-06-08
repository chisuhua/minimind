"""dLM: Diffusion Language Model (LLaDA-style masked diffusion)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.model_minimind import MiniMindConfig, RMSNorm, precompute_freqs_cis


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MaskedDiffusionScheduler:
    """Noise schedule for masked diffusion"""
    def __init__(self, num_steps=100, mask_token_id=3):
        self.num_steps = num_steps
        self.mask_token_id = mask_token_id

    def add_noise(self, input_ids, t):
        """t in [0,1]: 0=no noise, 1=all masked"""
        mask_prob = t ** 0.5
        mask = torch.rand_like(input_ids.float()) < mask_prob.unsqueeze(-1)
        return torch.where(mask, self.mask_token_id, input_ids), mask


class DiffusionBlock(nn.Module):
    """Transformer block for bidirectional diffusion (no causal mask)"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False),
        )
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, position_embeddings):
        B, L, D = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        cos = cos[:L][None, None, :, :]
        sin = sin[:L][None, None, :, :]
        xq = (xq * cos) + (_rotate_half(xq) * sin)
        xk = (xk * cos) + (_rotate_half(xk) * sin)
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)
        attn = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        x = self.norm1(x + self.o_proj(attn))
        x = self.norm2(x + self.mlp(x))
        return x


class DiffusionMiniMind(nn.Module):
    """Bidirectional Transformer for masked diffusion"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size)
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.layers = nn.ModuleList([DiffusionBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, t):
        B, L = input_ids.shape
        h = self.embed_tokens(input_ids)
        t_emb = self.time_embed(t.view(B, 1).float()).unsqueeze(1)
        h = h + t_emb
        pos = (self.freqs_cos[:L], self.freqs_sin[:L])
        for layer in self.layers:
            h = layer(h, pos)
        h = self.norm(h)
        return self.lm_head(h)


class DiffusionSampler:
    """Iterative denoising for diffusion LM"""
    def __init__(self, model, num_steps=100, mask_token_id=3):
        self.model = model
        self.num_steps = num_steps
        self.scheduler = MaskedDiffusionScheduler(num_steps, mask_token_id)

    @torch.inference_mode()
    def sample(self, prompt_ids, total_len=1024):
        device = prompt_ids.device
        seq = torch.full((1, total_len), self.scheduler.mask_token_id, device=device)
        seq[:, :prompt_ids.shape[1]] = prompt_ids
        for step in range(self.num_steps):
            t = torch.tensor([1.0 - (step + 1) / self.num_steps], device=device)
            logits = self.model(seq, t)
            mask_pos = (seq == self.scheduler.mask_token_id)
            if not mask_pos.any():
                break
            probs = F.softmax(logits[mask_pos], dim=-1)
            _, top_tokens = probs.max(dim=-1)
            seq[mask_pos] = top_tokens
        return seq

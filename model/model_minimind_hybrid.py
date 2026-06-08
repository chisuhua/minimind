"""
Qwen3-Next 3:1 Hybrid Architecture
- 75% Gated DeltaNet layers + 25% Gated Attention layers (configurable via hybrid_pattern)
- Gated Attention: output gate + partial RoPE (32/128 dim) + head_dim 128
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from model.model_minimind import MiniMindConfig, RMSNorm, FeedForward, MOEFeedForward, apply_rotary_pos_emb, repeat_kv, precompute_freqs_cis
from model.gated_deltanet import GatedDeltaNet


@dataclass
class HybridCausalLMOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    past_key_values: list = None
    hidden_states: torch.Tensor = None
    aux_loss: torch.Tensor = None


class GatedAttention(nn.Module):
    """Gated Attention with output gate + partial RoPE"""
    def __init__(self, config, partial_rope_dim=32):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.partial_rope_dim = partial_rope_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.output_gate = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq_rope, xq_pass = xq[..., :self.partial_rope_dim], xq[..., self.partial_rope_dim:]
        xk_rope, xk_pass = xk[..., :self.partial_rope_dim], xk[..., self.partial_rope_dim:]
        xq_rope, xk_rope = apply_rotary_pos_emb(xq_rope, xk_rope, cos[..., :self.partial_rope_dim], sin[..., :self.partial_rope_dim])
        xq = torch.cat([xq_rope, xq_pass], dim=-1)
        xk = torch.cat([xk_rope, xk_pass], dim=-1)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        is_causal = seq_len > 1 and past_key_value is None
        if self.flash and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=0.0, is_causal=is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                scores = scores + torch.full((seq_len, seq_len), float('-inf'), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = F.softmax(scores.float(), dim=-1).type_as(xq) @ xv
        gate = torch.sigmoid(self.output_gate(x)).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        output = output * gate
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(output), past_kv


class HybridBlock(nn.Module):
    def __init__(self, layer_id, config, layer_type):
        super().__init__()
        self.layer_type = layer_type
        partial_rope_dim = getattr(config, 'partial_rope_dim', 32)
        if layer_type == 'd':
            self.token_mixer = GatedDeltaNet(config)
        else:
            self.token_mixer = GatedAttention(config, partial_rope_dim=partial_rope_dim)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_kv = self.token_mixer(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_kv


class HybridMiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pattern = getattr(config, 'hybrid_pattern', ['d'] * 6 + ['a'] * 2)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = len(pattern)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([HybridBlock(i, config, layer_type=t) for i, t in enumerate(pattern)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        seq_start = start_pos
        seq_end = start_pos + seq_length
        position_embeddings = (self.freqs_cos[seq_start:seq_end], self.freqs_sin[seq_start:seq_end])
        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(hidden_states, position_embeddings, past_kv, use_cache, attention_mask)
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class HybridMiniMindForCausalLM(nn.Module):
    config_class = MiniMindConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = HybridMiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, labels=None, **kwargs):
        hidden_states, presents, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return HybridCausalLMOutput(loss=loss, logits=logits, past_key_values=presents, hidden_states=hidden_states, aux_loss=aux_loss)

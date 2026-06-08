"""
Lightning Indexer (DSA-style): lightweight indexer + top-k sparse attention.
Reference: DeepSeek-V3.2 DSA (arXiv 2512.02556)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_minimind import RMSNorm, apply_rotary_pos_emb, repeat_kv


class LightningIndexer(nn.Module):
    """4-head 128-dim FP8-friendly indexer with ReLU activation"""
    def __init__(self, hidden_size, indexer_heads=4, indexer_dim=128):
        super().__init__()
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        self.q_proj = nn.Linear(hidden_size, indexer_heads * indexer_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, indexer_heads * indexer_dim, bias=False)
        self.weights = nn.Parameter(torch.ones(indexer_heads))

    def forward(self, hidden_states):
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.indexer_heads, self.indexer_dim)
        k = self.k_proj(hidden_states).view(B, L, self.indexer_heads, self.indexer_dim)
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        scores = F.relu(scores)
        scores = (self.weights.view(1, -1, 1, 1) * scores).sum(dim=1)
        causal = torch.triu(torch.full((L, L), float('-inf'), device=scores.device), diagonal=1)
        scores = scores + causal
        return scores  # (B, L, L)


class SparseAttentionWithIndexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.topk = getattr(config, 'lightning_indexer_topk', 2048)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = LightningIndexer(config.hidden_size)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        if past_key_value is not None:
            return self._standard_attn(x, position_embeddings, past_key_value, use_cache, attention_mask)

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        past_kv = (xk, xv) if use_cache else None

        idx_scores = self.indexer(x)

        if attention_mask is not None:
            idx_scores += (1.0 - attention_mask.unsqueeze(1)) * -1e9

        k_eff = min(self.topk, seq_len)
        _, topk_idx = torch.topk(idx_scores, k_eff, dim=-1)

        output = torch.zeros_like(xq)
        for b in range(bsz):
            for q in range(seq_len):
                idx = topk_idx[b, q]
                qv = xq[b, q:q+1].transpose(0, 1).unsqueeze(0)
                kv_sel = repeat_kv(xk[b:b+1, idx], self.n_rep).transpose(1, 2)
                vv_sel = repeat_kv(xv[b:b+1, idx], self.n_rep).transpose(1, 2)
                out = F.scaled_dot_product_attention(qv, kv_sel, vv_sel, dropout_p=0.0, is_causal=False)
                output[b, q] = out.squeeze(0).squeeze(1)

        output = output.reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output, past_kv

    def _standard_attn(self, x, position_embeddings, past_key_value, use_cache, attention_mask):
        """Fallback standard attention when KV cache is active"""
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        is_causal = seq_len > 1 and past_key_value is None
        output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=0.0, is_causal=is_causal)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output, past_kv

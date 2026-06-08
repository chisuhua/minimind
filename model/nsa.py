"""NSA: Native Sparse Attention — 3-branch design (DeepSeek 2025)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.model_minimind import RMSNorm, apply_rotary_pos_emb, repeat_kv

class CompressionBranch(nn.Module):
    """Block-level compression with learnable MLP + intra-block PE"""
    def __init__(self, hidden_size, head_dim, block_l=32, stride_d=16):
        super().__init__()
        self.block_l = block_l
        self.stride_d = stride_d
        self.compress = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.SiLU(), nn.Linear(head_dim, head_dim)
        )
        self.intra_pe = nn.Embedding(block_l, head_dim)
    
    def forward(self, k, v):
        B, H, L, D = k.shape
        n_blocks = max(1, (L - self.block_l) // self.stride_d + 1)
        k_blocks, v_blocks = [], []
        for i in range(n_blocks):
            start = i * self.stride_d
            end = min(i * self.stride_d + self.block_l, L)
            pe = self.intra_pe(torch.arange(end - start, device=k.device))
            k_b = k[:, :, start:end] + pe.unsqueeze(0).unsqueeze(0)
            v_b = v[:, :, start:end] + pe.unsqueeze(0).unsqueeze(0)
            k_blocks.append(self.compress(k_b.mean(dim=2, keepdim=True)))
            v_blocks.append(self.compress(v_b.mean(dim=2, keepdim=True)))
        return torch.cat(k_blocks, dim=2), torch.cat(v_blocks, dim=2)

class SelectionBranch(nn.Module):
    """Top-n block selection from compressed KV"""
    def __init__(self, top_n=16):
        super().__init__()
        self.top_n = top_n
    
    def forward(self, q, k_blocks, v_blocks):
        scores = (q @ k_blocks.transpose(-2, -1)).mean(dim=2, keepdim=True)
        _, idx = scores.topk(min(self.top_n, k_blocks.shape[2]), dim=-1)
        idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, -1, k_blocks.shape[-1])
        return k_blocks.gather(2, idx_e).squeeze(2), v_blocks.gather(2, idx_e).squeeze(2)

class NSA(nn.Module):
    """3-branch native sparse attention"""
    def __init__(self, config, block_l=32, stride_d=16, sliding_w=512, top_n=16):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.sliding_w = sliding_w
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.compress = CompressionBranch(config.hidden_size, config.head_dim, block_l, stride_d)
        self.selection = SelectionBranch(top_n)
        self.gate = nn.Linear(config.hidden_size, 3)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        B, L, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k_c, v_c = self.compress(xk, xv)
        k_s, v_s = self.selection(xq, k_c, v_c)
        k_w, v_w = xk[:, :, -self.sliding_w:], xv[:, :, -self.sliding_w:]
        scale = 1.0 / math.sqrt(self.head_dim)
        out_c = F.softmax((xq @ k_c.transpose(-2, -1)) * scale, dim=-1) @ v_c
        out_s = F.softmax((xq @ k_s.transpose(-2, -1)) * scale, dim=-1) @ v_s
        out_w = F.softmax((xq @ k_w.transpose(-2, -1)) * scale, dim=-1) @ v_w
        gate = F.softmax(self.gate(x.mean(dim=1)), dim=-1)
        output = gate[:, 0:1, None, None] * out_c + gate[:, 1:2, None, None] * out_s + gate[:, 2:3, None, None] * out_w
        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(output), None

def replace_attention_with_nsa(model, config):
    """Replace all Attention layers with NSA"""
    for layer in model.model.layers:
        layer.self_attn = NSA(config)
    return model

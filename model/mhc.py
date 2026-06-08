"""mHC: Manifold-Constrained Hyper-Connections (from DeepSeek-V4)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_minimind import MiniMindConfig, RMSNorm, FeedForward, MOEFeedForward, Attention, precompute_freqs_cis


class SinkhornKnopp(nn.Module):
    """Project to doubly stochastic matrix via Sinkhorn-Knopp normalization"""
    def __init__(self, n, iters=20):
        super().__init__()
        self.n = n
        self.iters = iters

    def forward(self, M):
        M = torch.exp(M - M.max(dim=-1, keepdim=True)[0])
        for _ in range(self.iters):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M


class HyperConnection(nn.Module):
    """n-way residual with doubly stochastic constraint"""
    def __init__(self, hidden_size, hc_mult=4, sinkhorn_iters=20):
        super().__init__()
        self.hc_mult = hc_mult
        self.expand = nn.Linear(hidden_size, hidden_size * hc_mult, bias=False)
        self.contract = nn.Linear(hidden_size * hc_mult, hidden_size, bias=False)
        self.mix = nn.Parameter(torch.randn(hc_mult, hc_mult) * 0.02)
        self.sinkhorn = SinkhornKnopp(hc_mult, sinkhorn_iters)

    def forward(self, x, f_x):
        B, L, D = x.shape
        n = self.hc_mult
        x_n = self.expand(x).view(B, L, n, D)
        M = x_n.mean(dim=3, keepdim=True) + self.mix.unsqueeze(0).unsqueeze(0)
        M = self.sinkhorn(M)
        f_x_n = f_x.unsqueeze(2).expand(-1, -1, n, -1)
        x_next_n = torch.einsum('blmn,blnd->blmd', M, x_n) + f_x_n
        x_next = self.contract(x_next_n.reshape(B, L, n * D))
        return x_next


class MHCBlock(nn.Module):
    """MiniMindBlock replacement with mHC residual connections"""
    def __init__(self, layer_id, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.hyper_conn_1 = HyperConnection(config.hidden_size, getattr(config, 'mhc_mult', 4))
        self.hyper_conn_2 = HyperConnection(config.hidden_size, getattr(config, 'mhc_mult', 4))

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        attn_out, present_kv = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = self.hyper_conn_1(hidden_states, attn_out)
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self.hyper_conn_2(hidden_states, mlp_out)
        return hidden_states, present_kv

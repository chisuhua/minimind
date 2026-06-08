"""
MInference 1.0: 三种稀疏注意力模式
- A-shape: A-shaped 模式（初始 sink + 局部窗口）
- Vertical-Slash: 垂直列 + 斜线行
- Block-Sparse: 块稀疏（mean-pool 后 top-k 块选择）

参考: MInference 1.0 (Jiang et al., 2024)
"""
import json
import math
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

PATTERN_NAMES = ("A", "VS", "BS")


def _causal_mask(query_pos, key_pos):
    """True = allowed (causal: j <= i). query_pos, key_pos: 1D tensors."""
    return key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)


def _sink_window_mask(query_pos, key_pos, sink, window):
    """A-shape mask: sink tokens + local window."""
    L_q = query_pos.shape[0]
    L_kv = key_pos.shape[0]
    q_minus_w = query_pos.unsqueeze(1) - window + 1
    win = key_pos.unsqueeze(0) >= q_minus_w
    sink = key_pos.unsqueeze(0) < sink
    return win | sink


def _expand_kv(key_states, value_states, n_rep):
    """Repeat KV heads to match Q heads (for GQA)."""
    if n_rep == 1:
        return key_states, value_states
    B, H_kv, L_kv, D = key_states.shape
    key_states = key_states[:, :, None, :, :].expand(B, H_kv, n_rep, L_kv, D).reshape(B, H_kv * n_rep, L_kv, D)
    value_states = value_states[:, :, None, :, :].expand(B, H_kv, n_rep, L_kv, D).reshape(B, H_kv * n_rep, L_kv, D)
    return key_states, value_states


def _apply_pattern_attn(q, k, v, mask_fn, scaling, **mask_kwargs):
    """Generic pattern attention with bool mask. Returns (B, H, L_q, D)."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
    mask = mask_fn(scores.shape[-2], scores.shape[-1], device=q.device, **mask_kwargs)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def _a_shape_mask(L_q, L_kv, device, sink, window):
    q_pos = torch.arange(L_q, device=device) + (L_kv - L_q)
    k_pos = torch.arange(L_kv, device=device)
    causal = _causal_mask(q_pos, k_pos)
    sw = _sink_window_mask(q_pos, k_pos, sink, window)
    return causal & sw


def _vertical_slash_mask(L_q, L_kv, device, sink, window, vertical_idx, slash_idx):
    q_pos = torch.arange(L_q, device=device) + (L_kv - L_q)
    k_pos = torch.arange(L_kv, device=device)
    causal = _causal_mask(q_pos, k_pos)
    sw = _sink_window_mask(q_pos, k_pos, sink, window)

    vert = torch.zeros(L_kv, dtype=torch.bool, device=device)
    vert[vertical_idx.to(device)] = True
    vert_full = vert.unsqueeze(0).expand(L_q, -1)

    slash = torch.zeros(L_kv, dtype=torch.bool, device=device)
    slash[slash_idx.to(device)] = True
    slash_full = slash.unsqueeze(0).expand(L_q, -1)

    return causal & (sw | vert_full | slash_full)


def _block_sparse_mask(L_q, L_kv, device, sink, block, top_k, block_keep):
    q_pos = torch.arange(L_q, device=device) + (L_kv - L_q)
    k_pos = torch.arange(L_kv, device=device)
    causal = _causal_mask(q_pos, k_pos)

    sink_m = k_pos.unsqueeze(0) < sink
    pad_len = (block - L_kv % block) % block
    L_kv_p = L_kv + pad_len
    n_blocks = L_kv_p // block
    k_in_block = torch.arange(L_kv_p, device=device) // block
    k_in_block = k_in_block[:L_kv]
    block_keep_b = block_keep[:, :, :L_q, :].to(device)
    block_m = block_keep_b.gather(-1, k_in_block.view(1, 1, 1, -1).expand(block_keep_b.shape[0], block_keep_b.shape[1], L_q, L_kv)).squeeze(0).squeeze(0)
    block_m = block_m.unsqueeze(0)
    return causal & (sink_m | block_m)


def a_shape_attn(q, k, v, sink=4, window=1024, scaling=None):
    """A-shape: sink + window. q,k,v: (B, H, L, D)."""
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.shape[-1])
    return _apply_pattern_attn(q, k, v, _a_shape_mask, scaling, sink=sink, window=window)


def vertical_slash_attn(q, k, v, sink=4, window=1024, last_q=64, slash_stride=32, vertical_top=64, scaling=None):
    """Vertical-Slash. q,k,v: (B, H, L, D)."""
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.shape[-1])
    B, H, L_q, D = q.shape
    L_kv = k.shape[2]
    last_q = min(last_q, L_q)

    if last_q > 0 and L_kv > 0:
        q_est = q[:, :, -last_q:, :]
        est_scores = torch.matmul(q_est, k.transpose(-2, -1)) * scaling
        q_pos_e = torch.arange(last_q, device=q.device) + (L_kv - last_q)
        k_pos = torch.arange(L_kv, device=q.device)
        e_causal = _causal_mask(q_pos_e, k_pos)
        est_scores = est_scores.masked_fill(~e_causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        est_attn = F.softmax(est_scores, dim=-1)
        col_attn = est_attn.sum(dim=2)
        vt = min(vertical_top, L_kv)
        _, vertical_idx = col_attn.topk(vt, dim=-1)
        vertical_idx = vertical_idx[0, 0]
    else:
        vertical_idx = torch.arange(0, device=q.device)

    slash_idx = torch.arange(0, L_kv, slash_stride, device=q.device)
    return _apply_pattern_attn(
        q, k, v, _vertical_slash_mask, scaling,
        sink=sink, window=window, vertical_idx=vertical_idx, slash_idx=slash_idx,
    )


def block_sparse_attn(q, k, v, sink=4, block=64, top_k=16, scaling=None):
    """Block-Sparse. q,k,v: (B, H, L, D)."""
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.shape[-1])
    B, H, L_q, D = q.shape
    H_kv = k.shape[1]
    n_rep = H // H_kv
    L_kv = k.shape[2]

    pad_len = (block - L_kv % block) % block
    L_kv_p = L_kv + pad_len
    n_blocks = L_kv_p // block

    if pad_len > 0:
        k_p = F.pad(k, (0, 0, 0, pad_len))
    else:
        k_p = k
    k_pool = k_p.view(B, H_kv, n_blocks, block, D).mean(dim=3)
    k_pool_exp = k_pool.repeat_interleave(n_rep, dim=1)

    block_scores = torch.matmul(q, k_pool_exp.transpose(-2, -1)) * scaling
    block_starts = torch.arange(n_blocks, device=q.device) * block
    q_pos = torch.arange(L_q, device=q.device) + (L_kv - L_q)
    causal_block = block_starts.unsqueeze(0) > (q_pos.unsqueeze(1) + block)
    block_scores = block_scores.masked_fill(causal_block.unsqueeze(0).unsqueeze(0), float("-inf"))
    tk = min(top_k, n_blocks)
    _, top_block_idx = block_scores.topk(tk, dim=-1)

    block_keep = torch.zeros(B, H, L_q, n_blocks, dtype=torch.bool, device=q.device)
    block_keep.scatter_(-1, top_block_idx, True)

    return _apply_pattern_attn(
        q, k, v, _block_sparse_mask, scaling,
        sink=sink, block=block, top_k=top_k, block_keep=block_keep,
    )


class OfflineSearcher:
    """离线 pattern 标定：按 head 选择最适配的稀疏模式 (A / VS / BS)."""
    def __init__(self, model, sink=4, window=1024, block=64, top_k=16, last_q=64,
                 slash_stride=32, vertical_top=64):
        self.model = model
        self.sink = sink
        self.window = window
        self.block = block
        self.top_k = top_k
        self.last_q = last_q
        self.slash_stride = slash_stride
        self.vertical_top = vertical_top
        self.patterns = {}

    @torch.no_grad()
    def calibrate(self, calib_data=None, num_calib=128, min_seq_len=64, max_seq_len=512, device=None):
        """运行标定集，对每个 head 决定 pattern."""
        if device is None:
            device = next(self.model.parameters()).device
        if calib_data is None:
            calib_data = self._gen_random_calib(num_calib, min_seq_len, max_seq_len, device)

        attn_weights_per_layer = self._collect_attn_weights(calib_data)
        num_layers = len(attn_weights_per_layer)
        if num_layers == 0:
            raise RuntimeError("Calibration failed: no attention weights collected")
        num_heads = attn_weights_per_layer[0].shape[1]

        for l in range(num_layers):
            for h in range(num_heads):
                self.patterns[(l, h)] = self._pick_pattern(attn_weights_per_layer[l][:, h, :, :])
        return self.patterns

    @torch.no_grad()
    def _collect_attn_weights(self, calib_data):
        """收集每层每头的 attention 矩阵 (average over batch)."""
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        from transformers.modeling_utils import PreTrainedModel

        model = self.model
        if hasattr(model, "config"):
            model.config._attn_implementation = "eager"
            model.config.output_attentions = True

        captured = []
        handles = []

        def make_hook(layer_idx):
            def hook(module, args, kwargs, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    captured.append(output[1].detach().to(torch.float32).cpu())
                else:
                    captured.append(None)
            return hook

        for layer_idx, layer in enumerate(model.model.layers):
            h = layer.self_attn.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
            handles.append(h)

        try:
            for input_ids in calib_data:
                model(input_ids=input_ids, output_attentions=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()
            if hasattr(model, "config"):
                model.config.output_attentions = False

        return captured

    def _pick_pattern(self, attn_h):
        """attn_h: (N, L_q, L_kv) — N calibration samples stacked.
        Returns one of 'A' / 'VS' / 'BS'."""
        scores = {"A": 0.0, "VS": 0.0, "BS": 0.0}
        for sample in attn_h:
            L_q, L_kv = sample.shape
            sample = sample.float()
            causal = torch.tril(torch.ones(L_q, L_kv, dtype=torch.bool))
            sample = sample * causal
            total = sample.sum().item() + 1e-12

            a_mask = self._a_canonical_mask(L_q, L_kv, sample.device)
            scores["A"] += (sample * a_mask).sum().item() / total

            vs_mask = self._vs_canonical_mask(L_q, L_kv, sample)
            scores["VS"] += (sample * vs_mask).sum().item() / total

            bs_mask = self._bs_canonical_mask(L_q, L_kv, sample)
            scores["BS"] += (sample * bs_mask).sum().item() / total

        return max(scores, key=scores.get)

    def _a_canonical_mask(self, L_q, L_kv, device):
        q_pos = torch.arange(L_q, device=device)
        k_pos = torch.arange(L_kv, device=device)
        causal = _causal_mask(q_pos, k_pos)
        sw = _sink_window_mask(q_pos, k_pos, self.sink, self.window)
        return (causal & sw).float()

    def _vs_canonical_mask(self, L_q, L_kv, sample):
        q_pos = torch.arange(L_q, device=sample.device)
        k_pos = torch.arange(L_kv, device=sample.device)
        causal = _causal_mask(q_pos, k_pos)
        sw = _sink_window_mask(q_pos, k_pos, self.sink, self.window)

        last_q = min(self.last_q, L_q)
        if last_q <= 0:
            return (causal & sw).float()
        sample_e = sample[-last_q:, :]
        col_attn = sample_e.sum(dim=0)
        vt = min(self.vertical_top, L_kv)
        _, vertical_idx = col_attn.topk(vt)
        vert = torch.zeros(L_kv, dtype=torch.bool, device=sample.device)
        vert[vertical_idx] = True
        vert_full = vert.unsqueeze(0).expand(L_q, -1)

        slash_idx = torch.arange(0, L_kv, self.slash_stride, device=sample.device)
        slash = torch.zeros(L_kv, dtype=torch.bool, device=sample.device)
        slash[slash_idx] = True
        slash_full = slash.unsqueeze(0).expand(L_q, -1)

        return (causal & (sw | vert_full | slash_full)).float()

    def _bs_canonical_mask(self, L_q, L_kv, sample):
        q_pos = torch.arange(L_q, device=sample.device)
        k_pos = torch.arange(L_kv, device=sample.device)
        causal = _causal_mask(q_pos, k_pos)
        sink_m = (k_pos < self.sink).unsqueeze(0).expand(L_q, -1)

        block = self.block
        top_k = min(self.top_k, max(1, L_kv // block))
        pad_len = (block - L_kv % block) % block if L_kv % block else 0
        L_kv_p = L_kv + pad_len
        n_blocks = L_kv_p // block

        if pad_len:
            sample_p = F.pad(sample, (0, pad_len))
        else:
            sample_p = sample
        k_pool = sample_p.view(L_q, n_blocks, block).mean(dim=(0, 2))  # (n_blocks,)
        block_starts = torch.arange(n_blocks, device=sample.device) * block
        causal_block = block_starts.unsqueeze(0) > (q_pos.unsqueeze(1) + block)  # (L_q, n_blocks)

        q_block = sample_p.view(L_q, n_blocks, block)  # (L_q, n_blocks, block)
        q_per_block = q_block.sum(dim=-1)  # (L_q, n_blocks)
        bscores = q_per_block * k_pool.unsqueeze(0)
        bscores = bscores.masked_fill(causal_block, float("-inf"))
        _, top_block_idx = bscores.topk(top_k, dim=-1)
        bmask = torch.zeros(L_q, n_blocks, dtype=torch.bool, device=sample.device)
        bmask.scatter_(-1, top_block_idx, True)
        k_in_block = torch.arange(L_kv, device=sample.device) // block
        block_full = bmask.gather(-1, k_in_block.unsqueeze(0).expand(L_q, -1))
        return (causal & (sink_m | block_full)).float()

    def _gen_random_calib(self, num_calib, min_seq_len, max_seq_len, device):
        vocab_size = getattr(self.model.config, "vocab_size", 6400)
        data = []
        for _ in range(num_calib):
            seq_len = random.randint(min_seq_len, max_seq_len)
            ids = torch.randint(3, vocab_size, (1, seq_len), device=device)
            data.append(ids)
        return data

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        serial = {f"{l}_{h}": p for (l, h), p in self.patterns.items()}
        with open(path, "w") as f:
            json.dump({
                "patterns": serial,
                "sink": self.sink,
                "window": self.window,
                "block": self.block,
                "top_k": self.top_k,
                "last_q": self.last_q,
                "slash_stride": self.slash_stride,
                "vertical_top": self.vertical_top,
            }, f, indent=2)

    @classmethod
    def load(cls, path, model=None):
        with open(path, "r") as f:
            data = json.load(f)
        patterns_raw = data["patterns"]
        patterns = {}
        for k, v in patterns_raw.items():
            l, h = k.split("_")
            patterns[(int(l), int(h))] = v
        searcher = cls(
            model,
            sink=data.get("sink", 4),
            window=data.get("window", 1024),
            block=data.get("block", 64),
            top_k=data.get("top_k", 16),
            last_q=data.get("last_q", 64),
            slash_stride=data.get("slash_stride", 32),
            vertical_top=data.get("vertical_top", 64),
        )
        searcher.patterns = patterns
        return searcher


def _qwen3_apply_rotary_emb(query, key, cos, sin):
    """Minimal RoPE helper compatible with Qwen3 (uses repeat-interleave on cos/sin)."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)
    return q_embed, k_embed


def _qwen3_rope_from_position_embeddings(q, k, position_embeddings):
    """Qwen3 position_embeddings: (cos, sin) of shape (B, L, D) — apply on (B, H, L, D)."""
    cos, sin = position_embeddings
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return _qwen3_apply_rotary_emb(q, k, cos, sin)


class OnlineIndexer:
    """在线推理：为每个 head 应用其对应的稀疏 pattern。"""
    def __init__(self, model, pattern_dict, sink=4, window=1024, block=64, top_k=16, last_q=64,
                 slash_stride=32, vertical_top=64):
        self.model = model
        self.patterns = self._normalize_patterns(pattern_dict)
        self.sink = sink
        self.window = window
        self.block = block
        self.top_k = top_k
        self.last_q = last_q
        self.slash_stride = slash_stride
        self.vertical_top = vertical_top
        self._original_forwards = {}
        self._patched = False
        self._is_qwen3 = self._detect_qwen3()

    @staticmethod
    def _normalize_patterns(pattern_dict):
        norm = {}
        for k, v in pattern_dict.items():
            if isinstance(k, str) and "_" in k:
                l, h = k.split("_")
                norm[(int(l), int(h))] = v
            else:
                norm[k] = v
        return norm

    def _detect_qwen3(self):
        cls_name = type(self.model.model.layers[0].self_attn).__name__
        return "Qwen3" in cls_name

    def attach(self):
        """Monkey-patch each layer's self_attn.forward to dispatch via MInference."""
        if self._patched:
            return
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self._original_forwards[layer_idx] = attn.forward
            attn._minference_layer_idx = layer_idx
            attn._minference_indexer = self
            if self._is_qwen3:
                attn.forward = self._make_qwen3_forward(attn, layer_idx)
            else:
                attn.forward = self._make_minimind_forward(attn, layer_idx)
        self._patched = True

    def _make_qwen3_forward(self, attn, layer_idx):
        indexer = self
        def forward(hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, cache_position=None, **kwargs):
            return indexer._qwen3_wrapped_forward(
                attn, hidden_states, position_embeddings, attention_mask,
                past_key_values, cache_position, **kwargs,
            )
        return forward

    def _make_minimind_forward(self, attn, layer_idx):
        indexer = self
        def forward(x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
            return indexer._minimind_wrapped_forward(
                attn, x, position_embeddings, past_key_value, use_cache, attention_mask,
            )
        return forward

    def detach(self):
        if not self._patched:
            return
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            if layer_idx in self._original_forwards:
                attn.forward = self._original_forwards[layer_idx]
        self._patched = False

    def _pattern_for_head(self, layer_idx, head_idx):
        return self.patterns.get((layer_idx, head_idx), "A")

    def _dispatch(self, q_all_heads, k, v, layer_idx, scaling):
        """q_all_heads: (B, H_q, L_q, D); k,v: (B, H_kv, L_kv, D). Returns (B, H_q, L_q, D)."""
        B, H_q, L_q, D = q_all_heads.shape
        H_kv = k.shape[1]
        n_rep = H_q // H_kv

        k_e, v_e = _expand_kv(k, v, n_rep)
        output = torch.empty_like(q_all_heads)

        heads_by_pat = defaultdict(list)
        for h in range(H_q):
            heads_by_pat[self._pattern_for_head(layer_idx, h)].append(h)

        for pattern, heads in heads_by_pat.items():
            if not heads:
                continue
            kv_heads = [h // n_rep for h in heads]
            q_sub = q_all_heads[:, heads, :, :]
            k_sub = k_e[:, heads, :, :]
            v_sub = v_e[:, heads, :, :]
            if pattern == "A":
                out = a_shape_attn(q_sub, k_sub, v_sub, sink=self.sink, window=self.window, scaling=scaling)
            elif pattern == "VS":
                out = vertical_slash_attn(
                    q_sub, k_sub, v_sub,
                    sink=self.sink, window=self.window,
                    last_q=self.last_q, slash_stride=self.slash_stride,
                    vertical_top=self.vertical_top, scaling=scaling,
                )
            elif pattern == "BS":
                out = block_sparse_attn(
                    q_sub, k_sub, v_sub,
                    sink=self.sink, block=self.block, top_k=self.top_k, scaling=scaling,
                )
            else:
                out = a_shape_attn(q_sub, k_sub, v_sub, sink=self.sink, window=self.window, scaling=scaling)
            output[:, heads, :, :] = out
        return output

    def _qwen3_wrapped_forward(self, attn, hidden_states, position_embeddings, attention_mask=None,
                                past_key_values=None, cache_position=None, **kwargs):
        """Drop-in replacement for Qwen3Attention.forward using MInference dispatch."""
        try:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        except ImportError:
            apply_rotary_pos_emb = None

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if apply_rotary_pos_emb is not None:
            try:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            except TypeError:
                query_states, key_states = _qwen3_rope_from_position_embeddings(query_states, key_states, position_embeddings)
        else:
            query_states, key_states = _qwen3_rope_from_position_embeddings(query_states, key_states, position_embeddings)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            try:
                key_states, value_states = past_key_values.update(key_states, value_states, attn.layer_idx, cache_kwargs)
            except TypeError:
                key_states, value_states = past_key_values.update(key_states, value_states, attn.layer_idx)

        scaling = getattr(attn, "scaling", 1.0 / math.sqrt(attn.head_dim))
        attn_out = self._dispatch(query_states, key_states, value_states, attn._minference_layer_idx, scaling)
        attn_out = attn_out.transpose(1, 2).contiguous().view(*input_shape, -1)
        attn_out = attn.o_proj(attn_out)
        return attn_out, None

    def _minimind_wrapped_forward(self, attn, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """Drop-in replacement for MiniMind Attention.forward using MInference dispatch."""
        from model.model_minimind import apply_rotary_pos_emb
        bsz, seq_len, _ = x.shape
        xq, xk, xv = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
        xq = xq.view(bsz, seq_len, attn.n_local_heads, attn.head_dim)
        xk = xk.view(bsz, seq_len, attn.n_local_kv_heads, attn.head_dim)
        xv = xv.view(bsz, seq_len, attn.n_local_kv_heads, attn.head_dim)
        xq, xk = attn.q_norm(xq), attn.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq_t = xq.transpose(1, 2)
        k_e, v_e = _expand_kv(xk.transpose(1, 2), xv.transpose(1, 2), attn.n_rep)
        scaling = 1.0 / math.sqrt(attn.head_dim)
        out_t = self._dispatch(xq_t, k_e, v_e, attn._minference_layer_idx, scaling)
        output = out_t.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = attn.resid_dropout(attn.o_proj(output))
        return output, past_kv

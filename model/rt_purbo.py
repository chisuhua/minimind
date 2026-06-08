import json
import math
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn


class RetrievalHeadClassifier:
    def __init__(self, model, long_range_threshold=2048, ratio=0.15):
        self.model = model
        self.long_range_threshold = long_range_threshold
        self.ratio = ratio
        self.retrieval_heads = {}

    @torch.no_grad()
    def calibrate(self, calib_data=None, num_calib=128, min_seq_len=2048, max_seq_len=4096, device=None):
        if device is None:
            device = next(self.model.parameters()).device
        if calib_data is None:
            calib_data = self._gen_random_calib(num_calib, min_seq_len, max_seq_len, device)

        config = self.model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads

        long_range_mass = defaultdict(float)
        count = 0

        for input_ids in calib_data:
            seq_len = input_ids.shape[1]
            if seq_len < self.long_range_threshold + 1:
                continue
            attn_maps = self._collect_attn_weights(input_ids)
            if attn_maps is None:
                continue
            for layer_idx, attn in enumerate(attn_maps):
                if attn is None:
                    continue
                B, H, L_q, L_kv = attn.shape
                for h in range(min(H, num_heads)):
                    attn_h = attn[0, h]
                    if L_kv > self.long_range_threshold:
                        long_part = attn_h[:, :-self.long_range_threshold]
                        long_mass = long_part.sum().item()
                    else:
                        long_mass = 0.0
                    long_range_mass[(layer_idx, h)] += long_mass
            count += 1

        if count == 0:
            raise RuntimeError("No valid calibration sequences")

        for k in long_range_mass:
            long_range_mass[k] /= count

        sorted_heads = sorted(long_range_mass.items(), key=lambda x: x[1], reverse=True)
        num_retrieval = max(1, int(len(sorted_heads) * self.ratio))
        retrieval_set = set(h[0] for h in sorted_heads[:num_retrieval])

        self.retrieval_heads = {}
        for l in range(num_layers):
            for h in range(num_heads):
                self.retrieval_heads[(l, h)] = (l, h) in retrieval_set

        return self.retrieval_heads

    def _collect_attn_weights(self, input_ids):
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
            model(input_ids=input_ids, output_attentions=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()
            if hasattr(model, "config"):
                model.config.output_attentions = False

        return captured

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
        serial = {f"{l}_{h}": int(v) for (l, h), v in self.retrieval_heads.items()}
        with open(path, "w") as f:
            json.dump({"retrieval_heads": serial, "long_range_threshold": self.long_range_threshold, "ratio": self.ratio}, f, indent=2)

    @classmethod
    def load(cls, path, model=None):
        with open(path, "r") as f:
            data = json.load(f)
        classifier = cls(model, long_range_threshold=data.get("long_range_threshold", 2048), ratio=data.get("ratio", 0.15))
        raw = data["retrieval_heads"]
        classifier.retrieval_heads = {}
        for k, v in raw.items():
            l, h = k.split("_")
            classifier.retrieval_heads[(int(l), int(h))] = bool(v)
        return classifier


class LowDimIndexer(nn.Module):
    def __init__(self, head_dim, low_dim=16):
        super().__init__()
        self.q_proj = nn.Linear(head_dim, low_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, low_dim, bias=False)

    def forward(self, q, k):
        q_low = F.relu(self.q_proj(q))
        k_low = F.relu(self.k_proj(k))
        return q_low, k_low


class TopPSelector:
    @staticmethod
    def top_p_select(scores, p=0.9):
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
        mask = cumulative_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        return sorted_indices[mask]


class RTPurboAttention:
    def __init__(self, model, retrieval_heads, low_dim_indexer, sink=4, local_window=8192, top_p=0.9):
        self.model = model
        self.retrieval_heads = retrieval_heads
        self.low_dim_indexer = low_dim_indexer
        self.sink = sink
        self.local_window = local_window
        self.top_p = top_p
        self._patched = False
        self._original_forwards = {}

    def is_retrieval_head(self, layer_idx, head_idx):
        return self.retrieval_heads.get((layer_idx, head_idx), False)

    def attach(self):
        if self._patched:
            return
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self._original_forwards[layer_idx] = attn.forward
            attn._rtpurbo_indexer = self
            attn.forward = self._make_rtpurbo_forward(attn, layer_idx)
        self._patched = True

    def detach(self):
        if not self._patched:
            return
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            if layer_idx in self._original_forwards:
                attn.forward = self._original_forwards[layer_idx]
        self._patched = False

    def _make_rtpurbo_forward(self, attn, layer_idx):
        rtpurbo = self

        def forward(x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
            return rtpurbo._wrapped_forward(attn, layer_idx, x, position_embeddings, past_key_value, use_cache, attention_mask)

        return forward

    def _wrapped_forward(self, attn, layer_idx, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
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
            full_k = torch.cat([past_key_value[0], xk], dim=1)
            full_v = torch.cat([past_key_value[1], xv], dim=1)
        else:
            full_k = xk
            full_v = xv
        past_kv = (full_k, full_v) if use_cache else None

        output_chunks = []
        for h in range(attn.n_local_heads):
            out_h = None
            kv_h = h // attn.n_rep
            q_h = xq[:, :, h:h+1, :]
            k_full = full_k[:, :, kv_h:kv_h+1, :]
            v_full = full_v[:, :, kv_h:kv_h+1, :]

            if self.is_retrieval_head(layer_idx, h):
                scores = (q_h @ k_full.transpose(-2, -1)) / math.sqrt(attn.head_dim)
                if attn.is_causal:
                    L = scores.shape[-1]
                    scores[:, :, :, -L:] += torch.full((L, L), float("-inf"), device=scores.device).triu(1)
                attn_weights = F.softmax(scores, dim=-1)
                out_h = attn_weights @ v_full
            else:
                L_kv = k_full.shape[2]
                if L_kv <= self.sink + self.local_window:
                    scores = (q_h @ k_full.transpose(-2, -1)) / math.sqrt(attn.head_dim)
                else:
                    sink_k = k_full[:, :, :, :self.sink]
                    sink_v = v_full[:, :, :, :self.sink]

                    local_k = k_full[:, :, :, -self.local_window:]
                    local_v = v_full[:, :, :, -self.local_window:]

                    q_low, k_low = self.low_dim_indexer(q_h, k_full)
                    scores_low = torch.matmul(q_low.squeeze(2), k_low.squeeze(2).transpose(-2, -1)) / math.sqrt(q_low.shape[-1])

                    scores_low_last = scores_low[:, -1:, :]
                    if scores_low_last.shape[-1] > self.sink:
                        long_part = scores_low_last[:, :, self.sink:-self.local_window] if self.local_window < L_kv - self.sink else scores_low_last[:, :, self.sink:]
                        if long_part.shape[-1] > 0:
                            selected_indices = TopPSelector.top_p_select(long_part.squeeze(1), p=self.top_p)
                            selected_indices = selected_indices + self.sink
                            if selected_indices.dim() == 0:
                                selected_indices = selected_indices.unsqueeze(0)
                            selected_indices = selected_indices[:1024]
                            selected_k = torch.index_select(k_full, 2, selected_indices)
                            selected_v = torch.index_select(v_full, 2, selected_indices)
                            sparse_k = torch.cat([sink_k, selected_k, local_k], dim=2)
                            sparse_v = torch.cat([sink_v, selected_v, local_v], dim=2)
                        else:
                            sparse_k = torch.cat([sink_k, local_k], dim=2)
                            sparse_v = torch.cat([sink_v, local_v], dim=2)
                    else:
                        sparse_k = torch.cat([sink_k, local_k], dim=2)
                        sparse_v = torch.cat([sink_v, local_v], dim=2)

                    scores = (q_h @ sparse_k.transpose(-2, -1)) / math.sqrt(attn.head_dim)

                    if attn.is_causal:
                        L = scores.shape[-1]
                        scores[:, :, :, -L:] += torch.full((L, L), float("-inf"), device=scores.device).triu(1)

                    attn_weights = F.softmax(scores, dim=-1)
                    out_h = attn_weights @ sparse_v

            output_chunks.append(out_h)

        output = torch.cat(output_chunks, dim=2)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = attn.resid_dropout(attn.o_proj(output))
        return output, past_kv

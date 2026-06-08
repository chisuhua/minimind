import torch


class KIVIKVCache:
    """KIVI 2-bit KV Cache Quantization.

    Key: per-channel quantization (each head_dim channel shares one scale).
    Value: per-token quantization (each token position gets its own scale).
    Sliding window: last N tokens kept in full precision for quality.
    """

    def __init__(self, num_kv_heads, head_dim, max_seq_len=32768, window=128, device='cuda'):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.window = window
        self.device = device

        buf_shape = (max_seq_len, num_kv_heads, head_dim)
        self.k_buf_quant = torch.zeros(buf_shape, dtype=torch.int8, device=device)
        self.k_scale = torch.zeros(head_dim, dtype=torch.float16, device=device)
        self.k_running_max = torch.zeros(head_dim, dtype=torch.float16, device=device)

        self.v_buf_quant = torch.zeros(buf_shape, dtype=torch.int8, device=device)
        self.v_scale = torch.zeros(max_seq_len, dtype=torch.float16, device=device)

        self.k_window = torch.zeros(window, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        self.v_window = torch.zeros(window, num_kv_heads, head_dim, dtype=torch.float16, device=device)

        self.total_len = 0
        self.window_filled = 0
        self.max_quantized = 0

    @staticmethod
    def _quantize_2bit(x, scale):
        return torch.round(x / scale).clamp(-2, 1).to(torch.int8)

    @staticmethod
    def _compute_k_scale(k):
        abs_max = k.abs().amax(dim=(0, 1))
        return (abs_max / 2.0).clamp(min=1e-10)

    @staticmethod
    def _compute_v_scale(v):
        abs_max = v.abs().amax(dim=(1, 2))
        return (abs_max / 2.0).clamp(min=1e-10)

    def _quantize_into_buffer(self, k_chunk, v_chunk):
        seq_len = k_chunk.shape[0]
        self.k_running_max = torch.max(self.k_running_max, k_chunk.abs().amax(dim=(0, 1)))
        self.k_scale = (self.k_running_max / 2.0).clamp(min=1e-10)
        k_q = self._quantize_2bit(k_chunk, self.k_scale)
        v_s = self._compute_v_scale(v_chunk)
        v_q = self._quantize_2bit(v_chunk, v_s.view(-1, 1, 1))
        start = self.max_quantized
        end = start + seq_len
        self.k_buf_quant[start:end] = k_q
        self.v_buf_quant[start:end] = v_q
        self.v_scale[start:end] = v_s
        self.max_quantized = end

    def update(self, k, v):
        """Update cache with new (k, v).

        k, v: (1, seq_len, num_kv_heads, head_dim) post-RoPE
        """
        k = k[0]
        v = v[0]
        seq_len = k.shape[0]

        if self.total_len == 0 and seq_len > 1:
            quant_len = max(0, seq_len - self.window)
            window_len = min(seq_len, self.window)
            if quant_len > 0:
                self._quantize_into_buffer(k[:quant_len], v[:quant_len])
            if window_len > 0:
                self.k_window[:window_len] = k[-window_len:]
                self.v_window[:window_len] = v[-window_len:]
                self.window_filled = window_len
            self.total_len = seq_len
        else:
            for i in range(seq_len):
                ki = k[i:i + 1]
                vi = v[i:i + 1]
                if self.window_filled >= self.window:
                    self._quantize_into_buffer(
                        self.k_window[:1], self.v_window[:1]
                    )
                    self.k_window[:-1] = self.k_window[1:]
                    self.v_window[:-1] = self.v_window[1:]
                    self.window_filled -= 1
                self.k_window[self.window_filled] = ki[0]
                self.v_window[self.window_filled] = vi[0]
                self.window_filled += 1
                self.total_len += 1

    def get(self):
        """Get dequantized (k, v) for attention.

        Returns: (1, total_len, num_kv_heads, head_dim)
        """
        if self.total_len == 0:
            return None, None

        k_parts = []
        v_parts = []

        if self.max_quantized > 0:
            k_deq = self.k_buf_quant[:self.max_quantized].to(self.k_scale.dtype) * self.k_scale
            k_parts.append(k_deq)
            v_s = self.v_scale[:self.max_quantized].view(-1, 1, 1)
            v_deq = self.v_buf_quant[:self.max_quantized].to(v_s.dtype) * v_s
            v_parts.append(v_deq)

        if self.window_filled > 0:
            k_parts.append(self.k_window[:self.window_filled])
            v_parts.append(self.v_window[:self.window_filled])

        k_full = torch.cat(k_parts, dim=0).unsqueeze(0)
        v_full = torch.cat(v_parts, dim=0).unsqueeze(0)
        return k_full, v_full

import torch


class StreamingKVCache:
    """StreamingLLM KV cache with attention sink + sliding window.

    Maintains a fixed-size buffer [sink (num_sink) | sliding window (window)].
    Sink tokens are always kept; the window holds the most recent tokens.
    Once full, the oldest window tokens are dropped to make room for new ones.
    """

    def __init__(self, num_sink=4, window=4096, num_kv_heads=4, head_dim=96, device='cuda'):
        self.num_sink = num_sink
        self.window = window
        self.cache_len = num_sink + window
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.k_buf = None
        self.v_buf = None
        self.cur_len = 0

    def _ensure_buffers(self, k, v):
        if self.k_buf is None:
            self.k_buf = torch.zeros(
                self.cache_len, self.num_kv_heads, self.head_dim,
                device=k.device, dtype=k.dtype
            )
            self.v_buf = torch.zeros(
                self.cache_len, self.num_kv_heads, self.head_dim,
                device=v.device, dtype=v.dtype
            )

    def update(self, k, v):
        self._ensure_buffers(k, v)
        k_in = k[0]
        v_in = v[0]
        new_len = k_in.shape[0]

        if self.cur_len < self.num_sink:
            fill = min(self.num_sink - self.cur_len, new_len)
            self.k_buf[self.cur_len:self.cur_len + fill] = k_in[:fill]
            self.v_buf[self.cur_len:self.cur_len + fill] = v_in[:fill]
            self.cur_len += fill
            if fill < new_len:
                self.update(k[:, fill:], v[:, fill:])
        elif self.cur_len < self.cache_len:
            fill = min(self.cache_len - self.cur_len, new_len)
            self.k_buf[self.cur_len:self.cur_len + fill] = k_in[:fill]
            self.v_buf[self.cur_len:self.cur_len + fill] = v_in[:fill]
            self.cur_len += fill
            if fill < new_len:
                self.update(k[:, fill:], v[:, fill:])
        else:
            if new_len >= self.window:
                self.k_buf[self.num_sink:self.cache_len] = k_in[-self.window:]
                self.v_buf[self.num_sink:self.cache_len] = v_in[-self.window:]
            else:
                self.k_buf[self.num_sink:self.num_sink + (self.window - new_len)] = \
                    self.k_buf[self.num_sink + new_len:self.cache_len].clone()
                self.v_buf[self.num_sink:self.num_sink + (self.window - new_len)] = \
                    self.v_buf[self.num_sink + new_len:self.cache_len].clone()
                self.k_buf[self.cache_len - new_len:self.cache_len] = k_in
                self.v_buf[self.cache_len - new_len:self.cache_len] = v_in
            self.cur_len = self.cache_len

    def get_kv(self):
        if self.cur_len == 0:
            return None, None
        return (
            self.k_buf[:self.cur_len].unsqueeze(0),
            self.v_buf[:self.cur_len].unsqueeze(0),
        )

    def reset(self):
        self.cur_len = 0

import torch
import torch.nn as nn


class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.chunk_size = 64

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.alpha_proj = nn.Linear(config.hidden_size, self.num_heads)
        self.beta_proj = nn.Linear(config.hidden_size, self.num_heads)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, D)
        k = self.k_proj(x).view(B, L, H, D)
        v = self.v_proj(x).view(B, L, H, D)

        alpha = torch.sigmoid(self.alpha_proj(x))
        beta = torch.sigmoid(self.beta_proj(x))

        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

        outputs = []
        for i in range(0, L, self.chunk_size):
            end = min(i + self.chunk_size, L)
            chunk_q = q[:, i:end]
            chunk_k = k[:, i:end]
            chunk_v = v[:, i:end]
            chunk_alpha = alpha[:, i:end]
            chunk_beta = beta[:, i:end]
            C = end - i

            for t in range(C):
                qt = chunk_q[:, t]
                kt = chunk_k[:, t]
                vt = chunk_v[:, t]
                at = chunk_alpha[:, t]
                bt = chunk_beta[:, t]

                ot = (S @ qt.unsqueeze(-1)).squeeze(-1)
                outputs.append(ot)

                a = (S @ kt.unsqueeze(-1)).squeeze(-1)

                a_gate = at.view(B, H, 1)
                b_gate = bt.view(B, H, 1)
                u = -a_gate * b_gate * a + b_gate * vt

                S = a_gate.view(B, H, 1, 1) * S + u.unsqueeze(-1) @ kt.unsqueeze(-2)

        output = torch.stack(outputs, dim=1)
        output = output.reshape(B, L, -1)
        output = self.o_proj(output)

        return output, None

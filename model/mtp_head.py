import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_minimind import RMSNorm, FeedForward


class MTPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mtp_depth = config.mtp_depth
        h = config.hidden_size

        self.norm1 = RMSNorm(h, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

        self.norm2 = RMSNorm(h, eps=config.rms_norm_eps)
        self.ffn = FeedForward(config)
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)

    def forward(self, h):
        h_res = self.norm1(h)
        q = self.q_proj(h_res).unsqueeze(1)
        k = self.k_proj(h_res).unsqueeze(1)
        v = self.v_proj(h_res).unsqueeze(1)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h + attn_out.squeeze(1)
        h = h + self.ffn(self.norm2(h))
        return self.lm_head(h)

    def compute_loss(self, h_states, labels, input_ids, embed_tokens):
        D = self.mtp_depth
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            input_embeds = embed_tokens(input_ids)
        for d in range(1, D + 1):
            eff_len = h_states.size(1) - d - 2
            if eff_len <= 0:
                break
            h_in = h_states[:, :eff_len, :]
            cond = input_embeds[:, d:d + eff_len, :]
            logits = self.forward(h_in + cond)
            tgt = labels[:, d + 1:d + 1 + eff_len].contiguous()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
            total_loss = total_loss + loss
            count += 1
        return total_loss / max(count, 1)



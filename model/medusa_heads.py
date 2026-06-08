import torch
import torch.nn.functional as F
from torch import nn


class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class MedusaHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.K = config.medusa_heads
        self.heads = nn.ModuleList([
            MedusaHead(config.hidden_size, config.vocab_size)
            for _ in range(self.K)
        ])

    def forward(self, hidden_states):
        return [head(hidden_states) for head in self.heads]


def _sample(logits, temperature, top_k, top_p, do_sample):
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        threshold = values[..., -1, None]
        logits[logits < threshold] = -float('inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=False)
        cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumsum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        logits_to_scatter = logits.gather(-1, sorted_indices)
        logits_to_scatter[mask] = -float('inf')
        logits.scatter_(-1, sorted_indices, logits_to_scatter)
    probs = F.softmax(logits, dim=-1)
    if do_sample:
        return torch.multinomial(probs, 1)
    else:
        return torch.argmax(probs, dim=-1, keepdim=True)


def build_tree_candidates(medusa_logits, top_k=2):
    K = len(medusa_logits)
    top_indices = []
    for logits in medusa_logits:
        probs = F.softmax(logits[0, -1], dim=-1)
        top_idx = probs.topk(min(top_k, probs.size(-1))).indices.tolist()
        top_indices.append(top_idx)

    tree_nodes = []
    level_start = [0]
    level_nodes = []
    for c in top_indices[0]:
        level_nodes.append((c, -1))
    tree_nodes.extend(level_nodes)
    level_start.append(len(tree_nodes))

    prev_level = level_nodes
    for depth in range(1, K):
        current = []
        for i, (tok, par) in enumerate(prev_level):
            parent_idx = level_start[depth] + i
            for c in top_indices[depth]:
                current.append((c, parent_idx))
        tree_nodes.extend(current)
        level_start.append(len(tree_nodes))
        prev_level = current

    ids = [n[0] for n in tree_nodes]
    parents = [n[1] for n in tree_nodes]
    return ids, parents, level_start


def build_tree_mask(parents, device='cpu'):
    L = len(parents)
    mask = torch.full((1, 1, L, L), float('-inf'), device=device)
    for i in range(L):
        mask[0, 0, i, i] = 0
        p = parents[i]
        while p >= 0:
            mask[0, 0, i, p] = 0
            p = parents[p]
    return mask


class MedusaDecoder:
    def __init__(self, medusa_heads, tree_topk=1):
        self.medusa_heads = medusa_heads
        self.K = medusa_heads.K
        self.tree_topk = tree_topk
        self.last_hidden = None

    def update_hidden(self, hidden_states):
        self.last_hidden = hidden_states[:, -1:] if hidden_states is not None else None

    @torch.inference_mode()
    def step(self, model, input_ids, past_key_values,
             temperature=0.85, top_p=0.95, top_k=50, do_sample=True):
        if self.last_hidden is None:
            return None, None

        device = input_ids.device
        if past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[1]
        else:
            past_len = 0

        medusa_logits = self.medusa_heads(self.last_hidden)
        drafts = [logits[0, -1].argmax(-1).item() for logits in medusa_logits]
        if len(drafts) < 1:
            return None, None

        draft_tensor = torch.tensor([drafts], device=device)
        outputs = model.forward(
            draft_tensor, attention_mask=None,
            past_key_values=past_key_values, use_cache=True,
        )
        verif_logits = outputs.logits[0]
        full_kv = outputs.past_key_values

        accepted = []
        model_token = None
        for i in range(min(len(drafts), len(verif_logits))):
            sampled = _sample(verif_logits[i:i+1], temperature, top_k, top_p, do_sample)
            if sampled.item() == drafts[i]:
                accepted.append(drafts[i])
            else:
                model_token = sampled.item()
                break

        if len(accepted) == 0 and model_token is not None:
            kv = _truncate_kv(full_kv, past_len)
            token_tensor = torch.tensor([[model_token]], device=device)
            o = model.forward(token_tensor, attention_mask=None, past_key_values=kv, use_cache=True)
            self.update_hidden(o.hidden_states)
            return token_tensor, o.past_key_values

        if model_token is None:
            model_logit = verif_logits[-1:]
            model_token = _sample(model_logit, temperature, top_k, top_p, do_sample).item()

        kv = _truncate_kv(full_kv, past_len + len(accepted))
        model_token_tensor = torch.tensor([[model_token]], device=device)
        o = model.forward(model_token_tensor, attention_mask=None, past_key_values=kv, use_cache=True)
        self.update_hidden(o.hidden_states)

        all_new_tokens = accepted + [model_token]
        return torch.tensor([all_new_tokens], device=device), o.past_key_values


def _truncate_kv(past_key_values, target_len):
    if past_key_values is None:
        return None
    truncated = []
    for layer_kv in past_key_values:
        if layer_kv is None:
            truncated.append(None)
            continue
        k, v = layer_kv
        truncated.append((
            k[:, :target_len].contiguous(),
            v[:, :target_len].contiguous(),
        ))
    return tuple(truncated)

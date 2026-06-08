import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeBuilder:
    def __init__(self, branch_factor=4):
        self.branch_factor = branch_factor

    def build(self, dflash_logits):
        B = dflash_logits.shape[0]
        topk = dflash_logits.topk(self.branch_factor, dim=-1)
        candidates = topk.indices

        paths = [[c.item()] for c in candidates[0, 0]]
        for pos in range(1, dflash_logits.shape[1]):
            new_paths = []
            for path in paths:
                for c in candidates[0, pos]:
                    new_paths.append(path + [c.item()])
            paths = new_paths

        max_paths = 1024
        if len(paths) > max_paths:
            paths = paths[:max_paths]
        return paths


class TreeAttention:
    @staticmethod
    def build_mask(paths, max_len):
        n_paths = len(paths)
        path_lens = [len(p) for p in paths]
        total_len = 1 + sum(path_lens)

        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        mask[0, 0] = True
        pos = 1
        for path in paths:
            for depth, tok in enumerate(path):
                mask[pos, 0] = True
                path_start = pos - depth
                for d in range(depth + 1):
                    mask[pos, path_start + d] = True
                pos += 1
        return mask


class DDTreeDecoder:
    def __init__(self, target_model, verify=True):
        self.target = target_model
        self.verify = verify

    @torch.inference_mode()
    def decode_step(self, draft_logits, input_ids, past_key_values, embed_tokens, lm_head):
        paths = TreeBuilder(branch_factor=4).build(draft_logits)

        if not paths:
            return None, past_key_values

        anchor = input_ids[:, -1:]
        tree_input = torch.cat(
            [anchor] + [torch.tensor([[p[0]]], device=input_ids.device) for p in paths], dim=1
        )

        if self.verify:
            tree_mask = TreeAttention.build_mask(paths, max_len=tree_input.shape[1])
            from model.model_minimind import Attention

            original_forward = Attention.forward

            def tree_forward(self, x, pos_emb, past_kv=None, use_cache=False, attn_mask=None):
                return original_forward(self, x, pos_emb, past_kv, use_cache, attn_mask)

            outputs = self.target(tree_input, past_key_values=past_key_values, use_cache=True)

            logits = outputs.logits[0]
            accepted = paths[0][:3]
            return accepted, outputs.past_key_values

        return None, past_key_values

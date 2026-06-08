import torch
import torch.nn.functional as F
from collections import defaultdict


class LookaheadDecoding:
    def __init__(self, model, tokenizer=None, W=6, N=5, ngram_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.W = W
        self.N = N
        self.ngram_size = ngram_size
        self.ngram_pool = defaultdict(lambda: defaultdict(int))
        self.pool_processed_len = 0
        self.stats = {'total_tokens': 0, 'draft_tokens': 0, 'steps': 0, 'ar_fallbacks': 0}

    def _update_pool(self, ids):
        start = max(0, self.pool_processed_len - self.ngram_size + 1)
        for i in range(start, len(ids) - self.ngram_size + 1):
            prefix = tuple(ids[i:i + self.ngram_size - 1])
            next_token = ids[i + self.ngram_size - 1]
            self.ngram_pool[prefix][next_token] += 1
        self.pool_processed_len = len(ids)

    def _draft_from_pool(self, context_ids):
        draft = []
        ctx = context_ids[:]
        for _ in range(self.W):
            key = tuple(ctx[-(self.ngram_size - 1):])
            if key not in self.ngram_pool:
                break
            counts = self.ngram_pool[key]
            total = sum(counts.values())
            if total == 0:
                break
            tokens = list(counts.keys())
            weights = torch.tensor([counts[t] / total for t in tokens], dtype=torch.float)
            idx = torch.multinomial(weights, 1).item()
            next_token = tokens[idx]
            draft.append(next_token)
            ctx.append(next_token)
        return draft

    def _sample(self, logits, temperature, top_k, top_p, do_sample):
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

    def _forward_single(self, token_tensor, past_key_values):
        outputs = self.model.forward(
            token_tensor,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.past_key_values

    def _truncate_kv(self, past_key_values, target_len):
        if past_key_values is None:
            return None
        truncated = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            truncated.append((
                k[:, :, :target_len, :].contiguous(),
                v[:, :, :target_len, :].contiguous()
            ))
        return tuple(truncated)

    @torch.inference_mode()
    def step(self, input_ids, past_key_values, attention_mask, temperature=0.85, top_p=0.95, top_k=50, do_sample=True):
        device = input_ids.device
        if input_ids.shape[0] != 1:
            return None, None

        current_ids = input_ids[0].tolist()
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Build initial context for n-gram lookup from both KV cache and current input
        # Use the last ngram_size-1 tokens from the full sequence
        draft_tokens = self._draft_from_pool(current_ids)
        if len(draft_tokens) < 1:
            self.stats['ar_fallbacks'] += 1
            return None, None

        draft_len = len(draft_tokens)
        draft_tensor = torch.tensor([draft_tokens], device=device)

        # Forward all draft tokens through model in one batch
        outputs = self.model.forward(
            draft_tensor,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
        verif_logits = outputs.logits[0]
        full_kv = outputs.past_key_values

        # Verify each draft position against model prediction
        accepted = []
        model_token = None
        for i in range(draft_len):
            sampled = self._sample(verif_logits[i:i+1], temperature, top_k, top_p, do_sample)
            if sampled.item() == draft_tokens[i]:
                accepted.append(draft_tokens[i])
            else:
                model_token = sampled.item()
                break

        # --- Handle 0 accepted ---
        if len(accepted) == 0 and model_token is not None:
            # No draft accepted; use model's token with AR fallback
            kv = self._truncate_kv(full_kv, past_len)
            token_tensor = torch.tensor([[model_token]], device=device)
            new_kv = self._forward_single(token_tensor, kv)

            new_ids = current_ids + [model_token]
            self._update_pool(new_ids)

            self.stats['total_tokens'] += 1
            self.stats['ar_fallbacks'] += 1
            self.stats['steps'] += 1
            return token_tensor, new_kv

        # --- Some/all accepted: add model-sampled continuation ---
        # Sample from the model's distribution at the first rejected position (or last + 1 if all accepted)
        if model_token is not None:
            # Partial accept: model_token sampled from verif_logits[len(accepted)]
            pass
        else:
            # All accepted: sample from end of verif_logits
            model_logit = verif_logits[-1:]
            model_token = self._sample(model_logit, temperature, top_k, top_p, do_sample).item()

        # Truncate KV to accepted prefix
        kv = self._truncate_kv(full_kv, past_len + len(accepted))

        # Forward model token to extend KV
        model_token_tensor = torch.tensor([[model_token]], device=device)
        new_kv = self._forward_single(model_token_tensor, kv)

        # Combine accepted + model token
        all_new_tokens = accepted + [model_token]
        all_new_tensor = torch.tensor([all_new_tokens], device=device)

        # Update n-gram pool
        new_ids = current_ids + all_new_tokens
        self._update_pool(new_ids)

        self.stats['total_tokens'] += len(all_new_tokens)
        self.stats['draft_tokens'] += len(accepted)
        self.stats['steps'] += 1
        return all_new_tensor, new_kv

    def get_acceptance_rate(self):
        if self.stats['total_tokens'] == 0:
            return 0.0
        return self.stats['draft_tokens'] / self.stats['total_tokens']

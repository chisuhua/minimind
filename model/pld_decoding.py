import torch


class PLDDecoder:
    def __init__(self, max_ngram=4, num_pred=8):
        self.max_ngram = max_ngram
        self.num_pred = num_pred

    def find_match(self, context_ids, prompt_ids):
        if context_ids.numel() < 2 or prompt_ids.numel() < 2:
            return None
        context_len = len(context_ids)
        for n_size in range(min(self.max_ngram, context_len - 1), 0, -1):
            suffix = context_ids[-n_size:]
            for i in range(len(prompt_ids) - n_size):
                if torch.equal(prompt_ids[i:i + n_size], suffix):
                    end = min(i + n_size + self.num_pred, len(prompt_ids))
                    continuation = prompt_ids[i + n_size:end]
                    if len(continuation) > 0:
                        return continuation
        return None

    @torch.inference_mode()
    def step(self, model, input_ids, past_key_values, prompt_ids, attention_mask=None):
        context_ids = input_ids[0]
        draft_ids = self.find_match(context_ids, prompt_ids)
        if draft_ids is None or len(draft_ids) < 1:
            return None, past_key_values

        K = len(draft_ids)
        if attention_mask is not None:
            extra = attention_mask.new_ones(attention_mask.shape[0], K)
            full_mask = torch.cat([attention_mask, extra], dim=-1)
        else:
            full_mask = None

        draft_tensor = draft_ids.unsqueeze(0)
        outputs = model.forward(
            draft_tensor,
            past_key_values=past_key_values,
            use_cache=True,
            attention_mask=full_mask,
        )
        logits = outputs.logits[0]
        preds = logits.argmax(dim=-1)

        accept_len = 0
        for i in range(K - 1):
            if preds[i].item() == draft_ids[i + 1].item():
                accept_len += 1
            else:
                break

        if accept_len == 0:
            return None, past_key_values

        accepted = draft_ids[:accept_len + 1].tolist() + [preds[accept_len].item()]
        total = input_ids.shape[1]
        kv_target = total + len(accepted)
        new_kv = self._truncate_kv(outputs.past_key_values, kv_target)
        return accepted, new_kv

    @staticmethod
    def _truncate_kv(past_key_values, target_len):
        if past_key_values is None:
            return None
        new = []
        for layer_kv in past_key_values:
            if layer_kv is not None:
                k, v = layer_kv
                new.append((k[:, :target_len].contiguous(), v[:, :target_len].contiguous()))
            else:
                new.append(None)
        return new

import numpy as np
import torch


class TriAttentionScorer:
    def __init__(self, num_terms=4, threshold=0.05):
        self.num_terms = num_terms
        self.threshold = threshold
        self.coeffs = None
        self._mask_cache = {}

    def fit(self, attn_curves):
        max_len = max(len(c) for c in attn_curves)
        avg_curve = np.zeros(max_len, dtype=np.float64)
        for c in attn_curves:
            c = np.asarray(c, dtype=np.float64)
            avg_curve[:len(c)] += c
        avg_curve /= len(attn_curves)
        L = max_len
        n_terms = self.num_terms
        A = np.zeros((L, 2 * n_terms))
        d = np.arange(L, dtype=np.float64)
        for n in range(1, n_terms + 1):
            theta = n * np.pi * d / L
            A[:, 2 * (n - 1)] = np.cos(theta)
            A[:, 2 * (n - 1) + 1] = np.sin(theta)
        coeffs, _, _, _ = np.linalg.lstsq(A, avg_curve, rcond=None)
        self.coeffs = coeffs
        return self

    def score(self, distances):
        if self.coeffs is None:
            raise RuntimeError("call fit() before score()")
        distances = np.asarray(distances, dtype=np.float64)
        L = distances.max() + 1 if distances.size > 0 else 1
        result = np.zeros_like(distances, dtype=np.float64)
        for n in range(1, self.num_terms + 1):
            theta = n * np.pi * distances / L
            result += self.coeffs[2 * (n - 1)] * np.cos(theta)
            result += self.coeffs[2 * (n - 1) + 1] * np.sin(theta)
        return result

    def make_mask(self, L, threshold=None):
        if threshold is None:
            threshold = self.threshold
        key = (L, threshold)
        if key in self._mask_cache:
            return self._mask_cache[key]
        d = np.arange(L)
        predicted = self.score(d)
        mask = np.zeros((L, L), dtype=bool)
        for i in range(L):
            for j in range(min(i + 1, L)):
                dist = i - j
                if predicted[dist] >= threshold:
                    mask[i, j] = True
        mask_t = torch.from_numpy(mask)
        self._mask_cache[key] = mask_t
        return mask_t

import argparse
import math
import pickle
import random
from collections import defaultdict

import numpy as np
import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.tri_attention import TriAttentionScorer


def main():
    parser = argparse.ArgumentParser(description="Calibrate TriAttention scorers")
    parser.add_argument('--load_from', default='./minimind-3', type=str)
    parser.add_argument('--save_dir', default='./out', type=str)
    parser.add_argument('--num_calib', default=128, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--min_seq_len', default=64, type=int)
    parser.add_argument('--num_terms', default=4, type=int)
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = MiniMindForCausalLM(MiniMindConfig()).half().eval().to(args.device)
    ckp = torch.load(f'{args.load_from}/pytorch_model.bin', map_location=args.device)
    missing, unexpected = model.load_state_dict(ckp, strict=False)
    print(f"Loaded model, missing={len(missing)}, unexpected={len(unexpected)}")

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

    for layer in model.model.layers:
        layer.self_attn.capture_pre_rope = True

    dist_sums = [defaultdict(float) for _ in range(num_layers)]
    dist_counts = [defaultdict(int) for _ in range(num_layers)]
    max_dist_global = 0

    for seq_idx in range(args.num_calib):
        seq_len = random.randint(args.min_seq_len, args.max_seq_len)
        input_ids = torch.randint(3, model.config.vocab_size, (1, seq_len), device=args.device)
        model(input_ids)

        for layer_idx in range(num_layers):
            attn = model.model.layers[layer_idx].self_attn
            q_pre = attn.pre_rope_q
            k_pre = attn.pre_rope_k
            scores = torch.matmul(q_pre, k_pre.transpose(-2, -1)) / math.sqrt(head_dim)
            causal = torch.full((seq_len, seq_len), float('-inf'), device=scores.device).triu(1)
            scores = scores + causal.unsqueeze(0).unsqueeze(0)
            attn_w = torch.softmax(scores.float(), dim=-1)
            for h in range(num_heads):
                for i in range(seq_len):
                    for j in range(i + 1):
                        d = i - j
                        dist_sums[layer_idx][d] += attn_w[0, h, i, j].item()
                        dist_counts[layer_idx][d] += 1
                        if d > max_dist_global:
                            max_dist_global = d

        if (seq_idx + 1) % 16 == 0:
            print(f"  calibrating ... {seq_idx + 1}/{args.num_calib}")

    scorers = []
    for layer_idx in range(num_layers):
        max_d = max(dist_sums[layer_idx].keys()) if dist_sums[layer_idx] else 0
        avg_curve = np.array([
            dist_sums[layer_idx][d] / dist_counts[layer_idx][d]
            for d in range(max_d + 1)
        ])
        scorer = TriAttentionScorer(num_terms=args.num_terms, threshold=args.threshold)
        scorer.fit([avg_curve])
        scorers.append(scorer)
        print(f"  layer {layer_idx}: curve_len={len(avg_curve)}, threshold={args.threshold}")

    save_path = f'{args.save_dir}/tri_scorers.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(scorers, f)
    print(f"Saved scorers to {save_path}")


if __name__ == '__main__':
    main()

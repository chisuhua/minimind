"""
离线标定: 对 minimind-3 每个 head 选择 A / VS / BS 模式。
结果保存到 JSON, 供 OnlineIndexer 在推理时使用。
"""
import argparse
import os
import random
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.minference import OfflineSearcher

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Calibrate MInference 1.0 patterns per head")
    parser.add_argument("--load_from", default="./minimind-3", type=str)
    parser.add_argument("--save_path", default="./out/minference_patterns.json", type=str)
    parser.add_argument("--num_calib", default=128, type=int)
    parser.add_argument("--min_seq_len", default=64, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--sink", default=4, type=int)
    parser.add_argument("--window", default=1024, type=int)
    parser.add_argument("--block", default=64, type=int)
    parser.add_argument("--top_k", default=16, type=int)
    parser.add_argument("--last_q", default=64, type=int)
    parser.add_argument("--slash_stride", default=32, type=int)
    parser.add_argument("--vertical_top", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    random.seed(0)
    torch.manual_seed(0)

    print(f"Loading model from {args.load_from} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.load_from, trust_remote_code=True, dtype=torch.float16
    ).to(args.device).eval()

    searcher = OfflineSearcher(
        model,
        sink=args.sink, window=args.window,
        block=args.block, top_k=args.top_k,
        last_q=args.last_q, slash_stride=args.slash_stride, vertical_top=args.vertical_top,
    )

    print(f"Calibrating on {args.num_calib} random sequences (len {args.min_seq_len}-{args.max_seq_len}) ...")
    searcher.calibrate(
        num_calib=args.num_calib,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

    counts = {"A": 0, "VS": 0, "BS": 0}
    for p in searcher.patterns.values():
        counts[p] = counts.get(p, 0) + 1
    total = sum(counts.values())
    print(f"Pattern distribution: A={counts['A']}/{total}, VS={counts['VS']}/{total}, BS={counts['BS']}/{total}")

    searcher.save(args.save_path)
    print(f"Saved patterns to {args.save_path}")


if __name__ == "__main__":
    main()

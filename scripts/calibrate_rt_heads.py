"""
离线标定: 对 minimind-3 每个 head 识别 retrieval / local 分类。
测量 >2K 距离的 attention mass，标记 top ~15% 为 retrieval heads。
结果保存到 JSON，供 RTPurboAttention 在推理时使用。
"""
import argparse
import os
import random
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.rt_purbo import RetrievalHeadClassifier

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Calibrate RTPurbo retrieval heads per layer")
    parser.add_argument("--load_from", default="./minimind-3", type=str)
    parser.add_argument("--save_path", default="./out/rt_purbo_heads.json", type=str)
    parser.add_argument("--num_calib", default=128, type=int)
    parser.add_argument("--min_seq_len", default=2048, type=int)
    parser.add_argument("--max_seq_len", default=4096, type=int)
    parser.add_argument("--long_range_threshold", default=2048, type=int)
    parser.add_argument("--ratio", default=0.15, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    random.seed(0)
    torch.manual_seed(0)

    print(f"Loading model from {args.load_from} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.load_from, trust_remote_code=True, torch_dtype=torch.float16
    ).to(args.device).eval()

    classifier = RetrievalHeadClassifier(
        model,
        long_range_threshold=args.long_range_threshold,
        ratio=args.ratio,
    )

    print(f"Calibrating on {args.num_calib} random sequences (len {args.min_seq_len}-{args.max_seq_len}) ...")
    print(f"Long-range threshold: {args.long_range_threshold}, retrieval ratio: {args.ratio}")
    classifier.calibrate(
        num_calib=args.num_calib,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

    total = len(classifier.retrieval_heads)
    retrieval_count = sum(1 for v in classifier.retrieval_heads.values() if v)
    print(f"Retrieval heads: {retrieval_count}/{total} ({100 * retrieval_count / total:.1f}%)")

    classifier.save(args.save_path)
    print(f"Saved retrieval head classifications to {args.save_path}")


if __name__ == "__main__":
    main()

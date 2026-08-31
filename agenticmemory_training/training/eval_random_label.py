"""P1-4-pre:random-label(gold shuffle)对照 — 检测 LoRA 是否只学表面映射

方法:
1. 用 LoRA 模型(adapter)对 dev 原始 input 推理 → preds
2. 计算 pred vs 真实 gold 的 F1(genuine_f1)
3. 将 gold annotations 随机 shuffle(pair 错位) → 计算 pred vs shuffled gold 的 F1
4. 输出两者;若 genuine - random < 10pp → 可能只学了表面映射(spec §4.2)

已知边界:当 dev 集某字段值高度集中(如 language 全为 zh),shuffle 后 F1 仍高,
该对照在字段值多样性不足时区分度低。实施时需在 baseline_f1.json 注记
各字段值分布,供 P1-5 类别 D 判定参考。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from agenticmemory_training.training.eval_f1 import (
    entity_to_set,
    extract_first_turn,
    facts_to_set,
    f1_multiclass,
    f1_set_match,
    parse_model_output,
)


def run_lora_inference(
    base_model: str, adapter_dir: Path, prompts: list[str], max_new_tokens: int = 512
) -> list[str]:
    """加载 base + LoRA adapter 推理(复用 eval_f1.run_inference 的模式)"""
    from agenticmemory_training.training.eval_f1 import run_inference as eval_f1_infer

    samples = [{"input": p} for p in prompts]
    return eval_f1_infer(
        base_model, adapter_dir, samples, max_new_tokens=max_new_tokens
    )


def _compute_f1(preds: list[dict[str, Any] | None], golds: list[dict[str, Any] | None]) -> dict[str, Any]:
    intent_preds = [p["intent"]["primary"] if p else None for p in preds]
    intent_golds = [g["intent"]["primary"] if g else None for g in golds]
    lang_preds = [p["language"]["primary"] if p else None for p in preds]
    lang_golds = [g["language"]["primary"] if g else None for g in golds]
    ent_preds = [entity_to_set(p["entities"]) if p else None for p in preds]
    ent_golds = [entity_to_set(g["entities"]) if g else None for g in golds]
    facts_preds = [facts_to_set(p["session_facts"]) if p else None for p in preds]
    facts_golds = [facts_to_set(g["session_facts"]) if g else None for g in golds]
    return {
        "intent.primary": f1_multiclass(intent_preds, intent_golds),
        "language.primary": f1_multiclass(lang_preds, lang_golds),
        "entities": f1_set_match(ent_preds, ent_golds),
        "session_facts": f1_set_match(facts_preds, facts_golds),
    }


def compute_random_label_f1(
    base_model: str, adapter_dir: Path, samples: list[dict[str, Any]], seed: int = 42
) -> dict[str, Any]:
    """计算 genuine vs random-label(shuffle gold)对比 F1"""
    prompts = [s["input"] for s in samples]
    raw_outputs = run_lora_inference(base_model, adapter_dir, prompts)

    preds: list[dict[str, Any] | None] = []
    golds: list[dict[str, Any] | None] = []
    for s, raw in zip(samples, raw_outputs):
        gold = parse_model_output(s["output"])
        pred = parse_model_output(raw)
        golds.append(extract_first_turn(gold) if gold else None)
        preds.append(extract_first_turn(pred) if pred else None)

    genuine = _compute_f1(preds, golds)

    # gold shuffle(pair 错位)
    rng = random.Random(seed)
    shuffled_golds = list(golds)
    rng.shuffle(shuffled_golds)
    random_f1 = _compute_f1(preds, shuffled_golds)

    return {"genuine_f1": genuine, "random_label_f1": random_f1, "n_total": len(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-4-pre random-label 对照 F1")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter-dir", type=Path, default=Path("runs/lora_v0"))
    parser.add_argument("--dev-jsonl", type=Path, default=Path("data/agenticmemory_training/v0/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_v0"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = []
    with open(args.dev_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    result = compute_random_label_f1(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        samples=samples,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "baseline_f1.json"
    existing: dict[str, Any] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    existing["random_label_f1"] = result["random_label_f1"]
    existing["genuine_f1"] = result["genuine_f1"]
    existing["random_label_sample_seed"] = args.seed
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"baseline_f1.json 已更新(含 random_label 对照):{out_path}")


if __name__ == "__main__":
    main()

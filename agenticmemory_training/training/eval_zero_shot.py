"""P1-4-pre:base(zero-shot)在 dev 集上的按字段 F1 评估(无 LoRA adapter)

复用 eval_f1.py 的解析与 F1 逻辑。差异:加载 base model 且不挂 adapter。
"""
from __future__ import annotations

import argparse
import json
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


def build_zeroshot_inference_loader(base_model: str, max_new_tokens: int = 512):
    """惰性加载 base model(无 adapter),返回 (tokenizer, model)"""
    # 延迟 import,避免无 GPU 环境崩溃
    def loader():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return tokenizer, model

    return loader


def run_inference(
    base_model: str,
    samples: list[dict[str, Any]],
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> list[str]:
    """对 samples 做 zero-shot(无 adapter)推理,返回原始输出文本列表"""
    import torch

    tokenizer, model = build_zeroshot_inference_loader(base_model)()
    preds: list[str] = []
    prompts = [s["input"] for s in samples]
    for i in range(0, len(samples), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, prompt_ids in enumerate(inputs["input_ids"]):
            gen_ids = gen[j][len(prompt_ids) :]
            preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return preds


def compute_zero_shot_f1(
    base_model: str, samples: list[dict[str, Any]], max_new_tokens: int = 512
) -> dict[str, Any]:
    """计算 zero-shot 按字段 F1(对齐 eval_f1.evaluate_dev 的字段口径)"""
    raw_outputs = run_inference(base_model, samples, max_new_tokens=max_new_tokens)
    preds: list[dict[str, Any] | None] = []
    golds: list[dict[str, Any] | None] = []
    for s, raw in zip(samples, raw_outputs):
        gold = parse_model_output(s["output"])
        pred = parse_model_output(raw)
        golds.append(extract_first_turn(gold) if gold else None)
        preds.append(extract_first_turn(pred) if pred else None)

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
        "n_total": len(samples),
        "n_parse_failed_pred": sum(1 for p in preds if p is None),
        "n_parse_failed_gold": sum(1 for g in golds if g is None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-4-pre zero-shot baseline F1")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dev-jsonl", type=Path, default=Path("data/agenticmemory_training/v0/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_v0"))
    args = parser.parse_args()

    samples = []
    with open(args.dev_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    result = compute_zero_shot_f1(base_model=args.base_model, samples=samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "baseline_f1.json"
    # 读已有 baseline_f1.json(若存在)合并 zero_shot_f1
    existing: dict[str, Any] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    existing["zero_shot_f1"] = result
    for k, v in result.items():
        existing[f"zero_shot_{k}"] = v
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"baseline_f1.json 已写入:{out_path}")


if __name__ == "__main__":
    main()

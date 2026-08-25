"""P1-4 dev set 字段级 F1 评估

输入:
- LoRA 适配器目录(runs/lora_v0)
- base model(同 lora_train.py)
- dev.jsonl

输出:
- runs/lora_v0/dev_f1.json{"intent.primary": F1, "language.primary": F1, "entities": 实体级 F1, ...}
- runs/lora_v0/dev_predictions.jsonl(每个 turn 的模型预测 vs gold)

评估字段(对齐 13 字段 schema):
- intent.primary(8 类多分类)
- language.primary(5 类多分类)
- entities(集合级 F1:按 type+value 精确匹配)
- current_topic.value(文本相似度或精确匹配)
- session_facts(集合级 F1:按 key 精确匹配)

设计要点:
- 推理:batch 推理 + greedy decoding(避免采样带来的随机性)
- 解析失败:记为错例,生成模型输出与 gold 都不计入
- 字段级 macro-F1(各类等权,避免偏置大类)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# ============================================================================
# 字段级 F1 计算
# ============================================================================


def f1_multiclass(preds: list[str], golds: list[str]) -> dict[str, float]:
    """Macro-F1(各类等权)

    输入:预测 + gold 的并行列表(等长,None 表示解析失败)
    """
    from collections import Counter

    valid = [(p, g) for p, g in zip(preds, golds) if p is not None and g is not None]
    if not valid:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    labels = sorted({g for _p, g in valid})
    p_counter = Counter(p for p, _g in valid)
    g_counter = Counter(g for _p, g in valid)
    pg_counter = Counter(valid)

    f1s = []
    for lab in labels:
        tp = pg_counter[(lab, lab)]
        fp = p_counter[lab] - tp
        fn = g_counter[lab] - tp
        if tp == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))

    return {
        "precision": sum(f1s) / len(f1s) if f1s else 0.0,
        "recall": sum(f1s) / len(f1s) if f1s else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "support": len(valid),
    }


def f1_set_match(
    pred_sets: list[set[str]], gold_sets: list[set[str]]
) -> dict[str, float]:
    """集合级 F1(按集合元素精确匹配)

    适用:entities(集合)、session_facts(按 key 集合)
    """
    valid = [(p, g) for p, g in zip(pred_sets, gold_sets) if p is not None and g is not None]
    if not valid:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    tp_total = fp_total = fn_total = 0
    for pred, gold in valid:
        tp_total += len(pred & gold)
        fp_total += len(pred - gold)
        fn_total += len(gold - pred)

    p = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
    r = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "support": len(valid)}


def entity_to_set(entities: list[dict[str, Any]]) -> set[str]:
    return {f"{e.get('type', '')}::{e.get('value', '')}" for e in entities}


def facts_to_set(facts: list[dict[str, Any]]) -> set[str]:
    return {f.get("key", "") for f in facts if f.get("key")}


# ============================================================================
# 模型输出解析
# ============================================================================


JSON_PATTERN = re.compile(r"\{[\s\S]*\}")


def parse_model_output(text: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON 对象

    失败(无 JSON / 解析错误)返回 None
    """
    match = JSON_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_first_turn(generated: dict[str, Any]) -> dict[str, Any] | None:
    """从生成的 turns 列表取第一个 turn 的标注"""
    turns = generated.get("turns", [])
    if not turns:
        return None
    return turns[0]


# ============================================================================
# 推理(stub,需 transformers)
# ============================================================================


def run_inference(
    base_model: str,
    adapter_dir: Path,
    samples: list[dict[str, Any]],
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> list[str]:
    """对所有样本跑推理,返回模型原始输出文本列表(stub

    实现要求:
    1. 加载 base model + LoRA adapter
    2. 批量推理(可调 batch_size)
    3. greedy decoding(do_sample=False)
    4. 返回 input 后追加的生成部分(用 tokenizer 切分)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(f"缺少依赖:{e}.安装:pip install peft transformers torch") from e

    print(f"加载 model + adapter:{base_model} + {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    preds = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        prompts = [s["input"] for s in batch]
        inputs = tokenizer(
            prompts,
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
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            preds.append(text)
        print(f"  推理进度:{min(i + batch_size, len(samples))}/{len(samples)}")

    return preds


# ============================================================================
# 主评估
# ============================================================================


def evaluate_dev(
    base_model: str,
    adapter_dir: Path,
    dev_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """在 dev set 上评估,输出 F1 + 预测"""
    samples = []
    with open(dev_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"dev samples:{len(samples)}")

    raw_outputs = run_inference(base_model, adapter_dir, samples)

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

    entity_preds = [entity_to_set(p["entities"]) if p else None for p in preds]
    entity_golds = [entity_to_set(g["entities"]) if g else None for g in golds]

    facts_preds = [facts_to_set(p["session_facts"]) if p else None for p in preds]
    facts_golds = [facts_to_set(g["session_facts"]) if g else None for g in golds]

    metrics = {
        "intent.primary": f1_multiclass(intent_preds, intent_golds),
        "language.primary": f1_multiclass(lang_preds, lang_golds),
        "entities": f1_set_match(entity_preds, entity_golds),
        "session_facts": f1_set_match(facts_preds, facts_golds),
        "n_total": len(samples),
        "n_parse_failed_pred": sum(1 for p in preds if p is None),
        "n_parse_failed_gold": sum(1 for g in golds if g is None),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dev_f1.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "dev_predictions.jsonl", "w", encoding="utf-8") as f:
        for s, raw, g, p in zip(samples, raw_outputs, golds, preds):
            f.write(
                json.dumps(
                    {
                        "session_id": s.get("session_id"),
                        "raw_output": raw,
                        "gold": g,
                        "pred": p,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return metrics


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-4 dev F1 评估 CLI")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter-dir", type=Path, default=Path("runs/lora_v0"))
    parser.add_argument("--dev-jsonl", type=Path, default=Path("data/agenticmemory_training/v0/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_v0"))
    args = parser.parse_args()

    try:
        metrics = evaluate_dev(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            dev_jsonl=args.dev_jsonl,
            output_dir=args.output_dir,
        )
        print("\n=== Dev Set Metrics ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    except ImportError as e:
        print(f"\n✗ 缺少依赖:{e}")
        print("安装:pip install peft transformers torch")


if __name__ == "__main__":
    main()
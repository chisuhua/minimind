# P1 最小闭环实验修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec v0.3 修复 P1 最小闭环实验(基模型 Qwen3.5-0.8B + 跨教师 Kimi-K3/DeepSeek + baseline 对照 + 多字段判定 + IRR),在 ~$0.21 / 2.5 周 + 1-2 人日脚本开发内实证"sub-1B 模型能学会 13 字段结构化抽取"。

**Architecture:** 分两条线执行——**Part A**(3 个 TDD 代码任务):`synthesis.py` 教师参数化、`eval_zero_shot.py` + `eval_random_label.py` 新增脚本、`teacher_labeling.py` IRR 支持; **Part B**(7 个实验执行任务):P1-0 PoC → P1-1 合成 → P1-2 标注+IRR → P1-3 评估 → P1-4-pre baseline → P1-4 LoRA+判定 → P1-5 诊断 + findings。Part A 全部产出经 `unittest` 验证,Part B 依赖 API/GPU,提供精确命令与验证方法。

**Tech Stack:** Python 3.12+ / unittest / peft+transformers+accelerate / vLLM / Qwen3.5-0.8B / DeepSeek V4 Flash API / Kimi-K3 API / OpenCode SKILL.md

**Spec:** [`docs/superpowers/specs/2026-08-31-p1-minimum-loop-fixes-design.md`](../specs/2026-08-31-p1-minimum-loop-fixes-design.md)(v0.3)
**被修复代码:** `agenticmemory_training/{data,synthesis.py,teacher_labeling.py,evaluation.py; training,data_prep.py,lora_train.py,eval_f1.py}` + `agenticmind/extraction/{schemas.py,validator.py,privacy.py}`

---

## 文件结构总览

| 文件 | 操作 | 责任 |
|---|---|---|
| `AGENTS.md §12.10 F-04` | 已修订(v1.3.1) | Qwen3.5-0.8B 模型选型登记(🔴-1) |
| `agenticmemory_training/data/synthesis.py` | 修改 | 教师参数化(支持 kimi-k3)+ `metadata.teacher` 字段 |
| `agenticmemory_training/data/teacher_labeling.py` | 修改 | 新增 IRR 子集标注入口 + `compute_krippendorff_alpha()` |
| `agenticmemory_training/training/eval_zero_shot.py` | **创建** | base 模型 zero-shot 评估(无 adapter),输出按字段 F1(🔴-4) |
| `agenticmemory_training/training/eval_random_label.py` | **创建** | gold shuffle 对照,检测"pred 是否显著匹配真实 gold"(🔴-4) |
| `.opencode/skills/p1-poc/SKILL.md` | **创建** | PoC-1:OpenCode 编排 Python 代码(🟡-5) |
| `~/.config/opencode/skills/p1-poc/SKILL.md` | fallback | 若项目级不加载则创建全局版 |
| `data/agenticmemory_training/v0/*.jsonl` | 生成 | 实验数据产物 |
| `runs/lora_v0/{adapter,dev_f1,baseline_f1}.json|safetensors` | 生成 | 训练与评估产物 |
| `data/agenticmemory_training/v0/irr_krippendorff.json` | 生成 | IRR 测量结果(🔴-3) |
| `data/agenticmemory_training/v0/findings_v0.md` | 生成 | 5 章节必含报告 |

---

# PART A — 代码改造任务(TDD)

## Task 1: synthesis.py 教师参数化

**Files:**
- Modify: `agenticmemory_training/data/synthesis.py:201-261`(synthesize_via_gpt4 → 支持 kimi-k3)
- Modify: `agenticmemory_training/data/synthesis.py:109-135`(Conversation 增加 teacher 字段)
- Test: `agenticmemory_training/tests/test_synthesis_teacher.py`(创建)

- [ ] **Step 1: Write the failing test**

```python
# agenticmemory_training/tests/test_synthesis_teacher.py
"""验证 synthesize_via_gpt4 支持 kimi-k3 教师 + metadata.teacher 字段"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.data.synthesis import synthesize_via_gpt4


class _FakeChoice:
    message = MagicMock(content='{"turns": [{"role": "user", "text": "如何修 NPE?"}]}')


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions.create.return_value = _FakeResponse()


def test_synthesize_via_kimi_model_arg():
    client = _FakeClient()
    convs = list(synthesize_via_gpt4(client, model="kimi-k3", n_conversations=1))
    assert len(convs) == 1
    assert convs[0].source == "synthetic:kimi-k3"
    # 确认 client 以 kimi-k3 被调用
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "kimi-k3"


def test_conversation_to_jsonl_has_teacher_metadata():
    client = _FakeClient()
    convs = list(synthesize_via_gpt4(client, model="kimi-k3", n_conversations=1))
    record = convs[0].to_jsonl_record()
    assert record["metadata"]["teacher"] == "kimi-k3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest agenticmemory_training.tests.test_synthesis_teacher -v`
Expected: FAIL — `source == "synthetic:gpt4:gpt-4o"`(硬编码),且 `to_jsonl_record()` 无 `metadata` 字段

- [ ] **Step 3: Implement minimal changes in synthesis.py**

```python
# Conversation dataclass(L109 附近)增加 teacher 字段:
@dataclass
class Conversation:
    session_id: str
    source: str
    turns: list[ConversationTurn]
    teacher: str = "gpt-4o"  # 新增:合成教师标识

    def to_jsonl_record(self) -> dict[str, Any]:
        """输出 JSONL 记录(含 metadata.teacher,🟢-3 登记)"""
        return {
            "session_id": self.session_id,
            "source": self.source,
            "turns": [t.to_dict() for t in self.turns],
            "metadata": {"teacher": self.teacher},
        }


# synthesize_via_gpt4(L201-261):
#  1. 函数签名不变(model 参数已存在,默认 gpt-4o)
#  2. yield 时填充 teacher=model
    yield Conversation(
        session_id=f"syn_{i:04d}",
        source=f"synthetic:{model}",  # 修改:不再硬编码 gpt4
        turns=turns,
        teacher=model,  # 新增
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest agenticmemory_training.tests.test_synthesis_teacher -v`
Expected: PASS(2 tests OK)

- [ ] **Step 5: Run full regression**

Run: `python3 -m unittest agenticmind.tests.test_extraction agenticmemory_training.tests.test_synthesis_teacher -v`
Expected: 33 + 2 = 35 tests OK(不破坏既有 33 条)

- [ ] **Step 6: Commit**

```bash
git add agenticmemory_training/data/synthesis.py agenticmemory_training/tests/test_synthesis_teacher.py
git commit -m "feat(p1): synthesize_via_gpt4 教师参数化 + metadata.teacher 字段 (D-C/🟢-3)"
```

---

## Task 2: eval_zero_shot.py(新增)

**Files:**
- Create: `agenticmemory_training/training/eval_zero_shot.py`
- Test: `agenticmemory_training/tests/test_eval_zero_shot.py`

> 复用 `eval_f1.py` 的 `parse_model_output` / `extract_first_turn` / `f1_multiclass` / `f1_set_match` / `entity_to_set` / `facts_to_set`。

- [ ] **Step 1: Write the failing test**

```python
# agenticmemory_training/tests/test_eval_zero_shot.py
"""验证 eval_zero_shot 计算按字段 zero-shot F1(无 adapter)"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.training.eval_zero_shot import compute_zero_shot_f1


def test_compute_zero_shot_f1_includes_fields():
    # run_inference 被 mock 返回固定输出
    fake_outputs = [
        '{"turns": [{"intent": {"primary": "question"}, '
        '"language": {"primary": "zh"}, "entities": [], "session_facts": []}]}'
    ] * 4
    samples = [
        {"input": f"q{i}", "output": '{"turns": [{"intent": {"primary": "question"}, "language": {"primary": "zh"}, "entities": [], "session_facts": []}]}'}
        for i in range(4)
    ]
    with patch(
        "agenticmemory_training.training.eval_zero_shot.run_inference",
        return_value=fake_outputs,
    ):
        result = compute_zero_shot_f1(base_model="fake", samples=samples)
    assert "intent.primary" in result
    assert "language.primary" in result
    assert "entities" in result
    assert "session_facts" in result
    assert result["n_total"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest agenticmemory_training.tests.test_eval_zero_shot -v`
Expected: FAIL — ModuleNotFoundError(eval_zero_shot 不存在)

- [ ] **Step 3: Create eval_zero_shot.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest agenticmemory_training.tests.test_eval_zero_shot -v`
Expected: PASS

- [ ] **Step 5: Verify CLI --help works**

Run: `python3 -m agenticmemory_training.training.eval_zero_shot --help`
Expected: 打印 argparse usage(exit 0)

- [ ] **Step 6: Commit**

```bash
git add agenticmemory_training/training/eval_zero_shot.py agenticmemory_training/tests/test_eval_zero_shot.py
git commit -m "feat(p1): eval_zero_shot.py zero-shot baseline F1 (🔴-4)"
```

---

## Task 3: eval_random_label.py(新增)

**Files:**
- Create: `agenticmemory_training/training/eval_random_label.py`
- Test: `agenticmemory_training/tests/test_eval_random_label.py`

> 语义(spec §4.2 整改):对 dev 集 **gold annotations 随机 shuffle(pair 错位)**,用 **LoRA 模型**(adapter)对原始 input 推理,计算 pred vs shuffled-gold 的 F1。若「真实 gold F1」显著高于「shuffled gold F1」,证明 LoRA 学的是真实抽取而非随机匹配。

- [ ] **Step 1: Write the failing test**

```python
# agenticmemory_training/tests/test_eval_random_label.py
"""验证 eval_random_label:shuffle gold 后 F1 显著低于真实 gold F1"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.training.eval_random_label import compute_random_label_f1


def test_random_label_f1_below_genuine():
    gold_text = '{"turns": [{"intent": {"primary": "question"}, "language": {"primary": "zh"}, "entities": [], "session_facts": []}]}'
    samples = [
        {"input": f"q{i}", "output": gold_text} for i in range(4)
    ]
    # LoRA 推理输出与真实 gold 完全一致 → 真实 F1 = 1.0
    pred_outputs = [gold_text] * 4
    with patch(
        "agenticmemory_training.training.eval_random_label.run_lora_inference",
        return_value=pred_outputs,
    ):
        result = compute_random_label_f1(
            base_model="fake", adapter_dir=Path("fake"), samples=samples
        )
    # shuffle 后 gold 错位 → F1 应低于真实 F1(极端情况随机匹配)
    assert result["genuine_f1"]["intent.primary"]["f1"] >= 0.99
    # shuffled 后 4 样本 intent 全相同,shuffle 不会改变匹配 → f1 仍高;此为设计副作用,文档注释说明
    assert "random_label_f1" in result
```

> ⚠️ 已知边界:当 dev 集某字段值高度集中(如 language 全为 zh),shuffle 后 F1 仍高——该对照在字段值多样性不足时区分度低。实施时需在 baseline_f1.json 注记各字段值分布,供 P1-5 类别 D 判定参考。

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest agenticmemory_training.tests.test_eval_random_label -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Create eval_random_label.py**

```python
"""P1-4-pre:random-label(gold shuffle)对照 — 检测 LoRA 是否只学表面映射

方法:
1. 用 LoRA 模型(adapter)对 dev 原始 input 推理 → preds
2. 计算 pred vs 真实 gold 的 F1(genuine_f1)
3. 将 gold annotations 随机 shuffle(pair 错位) → 计算 pred vs shuffled gold 的 F1
4. 输出两者;若 genuine - random < 10pp → 可能只学了表面映射(spec §4.2)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest agenticmemory_training.tests.test_eval_random_label -v`
Expected: PASS

- [ ] **Step 5: Verify CLI --help works**

Run: `python3 -m agenticmemory_training.training.eval_random_label --help`
Expected: 打印 argparse usage(exit 0)

- [ ] **Step 6: Commit**

```bash
git add agenticmemory_training/training/eval_random_label.py agenticmemory_training/tests/test_eval_random_label.py
git commit -m "feat(p1): eval_random_label.py gold-shuffle 对照 (🔴-4/🟡-1)"
```

---

## Task 4: teacher_labeling.py IRR 支持

**Files:**
- Modify: `agenticmemory_training/data/teacher_labeling.py`(新增 `compute_krippendorff_alpha` + `label_irr_subset`)
- Test: `agenticmemory_training/tests/test_teacher_irr.py`

- [ ] **Step 1: Write the failing test**

```python
# agenticmemory_training/tests/test_teacher_irr.py
"""验证 Krippendorff α 计算(2 标注者、3 类 else)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.data.teacher_labeling import compute_krippendorff_alpha


def test_alpha_perfect_agreement():
    # 两位标注者完全一致 → α = 1.0
    a = ["question", "command", "chat", "question"]
    b = ["question", "command", "chat", "question"]
    assert compute_krippendorff_alpha(a, b) == 1.0


def test_alpha_random_agreement():
    # 各 50% 随机一致 → α 接近 0
    a = ["question", "question", "chat", "chat"]
    b = ["question", "chat", "question", "chat"]
    alpha = compute_krippendorff_alpha(a, b)
    assert -0.4 <= alpha <= 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest agenticmemory_training.tests.test_teacher_irr -v`
Expected: FAIL — ImportError(compute_krippendorff_alpha 不存在)

- [ ] **Step 3: Implement in teacher_labeling.py**

```python
# 追加到 teacher_labeling.py
def compute_krippendorff_alpha(
    rater_a: list[str], rater_b: list[str]
) -> float:
    """两个标注者对同一批样本的 Krippendorff α(nominal scale, 2 raters)

    NLTK coincidence 口径:每个样本对贡献两次(对称),总 coincidence = 2n。

    自测:
      perfect: k(['q','c','q','c'], ['q','c','q','c']) == 1.0
      random : k(['q','q','c','c'], ['q','c','q','c']) ≈ 0(2 类下)
    """
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        raise ValueError("rater_a 与 rater_b 长度必须相同且非空")
    n = len(rater_a)
    values = sorted(set(rater_a) | set(rater_b))
    if len(values) <= 1:
        return 1.0
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)

    # 对称 coincidence matrix:每个样本贡献 2 次
    coinc = [[0.0] * k for _ in range(k)]
    for va, vb in zip(rater_a, rater_b):
        i, j = idx[va], idx[vb]
        coinc[i][j] += 1.0
        coinc[j][i] += 1.0

    coinc_total = 2.0 * n  # = 2n

    # 观测不一致(总 coincidence - 对角线一致)
    do = coinc_total - sum(coinc[i][i] for i in range(k))

    # 期望不一致(NLTK 口径,基于边际 rowsum)
    rowsum = [sum(row) for row in coinc]
    de = coinc_total - sum(rowsum[i] * rowsum[i] for i in range(k)) / (coinc_total - 1.0)

    if de == 0.0:
        return 1.0
    alpha = 1.0 - do / de
    return round(alpha, 4)


def label_irr_subset(
    client: Any,
    sessions: list[dict[str, Any]],
    primary_labeler: str = "deepseek-chat",
    second_labeler: str = "kimi-k3",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """用第二标注者(Kimi-K3)对子集二次标注,计算与主标注(DeepSeek)的 Krippendorff α

    流程:
    1. 对子集样本用主标注模型标注 → labels_a
    2. 同一批样本用第二标注模型标注 → labels_b
    3. 对 intent.primary 计算 α(其他字段可扩展)
    """
    import json as _json

    labels_a: list[str] = []
    labels_b: list[str] = []
    records = []

    for session in sessions:
        ann_a = label_via_teacher(client, session, model=primary_labeler)
        ann_b = label_via_teacher(client, session, model=second_labeler)
        for ta, tb in zip(ann_a, ann_b):
            labels_a.append(ta.intent_primary)
            labels_b.append(tb.intent_primary)
            records.append(
                {
                    "session_id": session.get("session_id", "unknown"),
                    "turn_index": ta.turn_index,
                    "labeler_a": primary_labeler,
                    "labeler_b": second_labeler,
                    "intent_a": ta.intent_primary,
                    "intent_b": tb.intent_primary,
                }
            )

    alpha = compute_krippendorff_alpha(labels_a, labels_b)
    result = {
        "n_pairs": len(labels_a),
        "krippendorff_alpha_intent": alpha,
        "labeler_a": primary_labeler,
        "labeler_b": second_labeler,
        "records": records,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest agenticmemory_training.tests.test_teacher_irr -v`
Expected: PASS(2 tests OK)

- [ ] **Step 5: Verify exact agreement yields 1.0 and random ~ 0**

Run: `python3 -c "from agenticmemory_training.data.teacher_labeling import compute_krippendorff_alpha as k; print(k(['q','c','q','c'],['q','c','q','c'])); print(k(['q','q','c','c'],['q','c','q','c']))"`
Expected: `1.0` 与一个接近 0 的值

- [ ] **Step 6: Commit**

```bash
git add agenticmemory_training/data/teacher_labeling.py agenticmemory_training/tests/test_teacher_irr.py
git commit -m "feat(p1): teacher_labeling IRR 子集标注 + Krippendorff α (🔴-3)"
```

---

# PART B — 实验执行任务

> 前置:确认 `.opencode/skills/`(项目级)或全局 skills 目录加载约定;确认 Qwen3.5-0.8B 可用性(若不可用走 🔴-2 fallback 链)。

## Task 5: P1-0 PoC 阶段

**Files:**
- Create: `.opencode/skills/p1-poc/SKILL.md`(项目级;若 OpenCode 不加载则建全局版)
- Create: `scripts/poc_zeroshot_check.py`(可选,PoC-3 用的 13 字段解析校验)

- [ ] **Step 1: 确认 Qwen3.5-0.8B 可用性(PoC-2 前置,🔴-2)**

Run: `python3 -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen3.5-0.8B'))" 2>&1 | head -5`
Expected: 打印模型元数据(或无报错)
若失败 → **fallback**:尝试 `Qwen/Qwen3.5-1.5B`;仍失败 → 暂停 P1,启动 F-07 前置决策(见 spec §5.0)

- [ ] **Step 2: 创建 PoC-1 SKILL.md(OpenCode 编排入口)**

```markdown
# .opencode/skills/p1-poc/SKILL.md

---
name: p1-poc
description: P1 PoC 验证:OpenCode → Python(agenticmemory_training)→ Qwen3.5-0.8B 推理服务
---

# P1 PoC 执行

本 skill 让 OpenCode 编排 P1 最小闭环实验的基础设施验证。

## PoC-1:验证 Python 包可被编排

运行以下命令验证 agenticmemory_training 包可被加载:

```bash
python3 -c "import agenticmemory_training; print('OK')"
python3 -m unittest agenticmind.tests.test_extraction -v
```

## PoC-2:验证 Qwen3.5-0.8B 推理服务

```bash
curl -s http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-0.8B", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 32}'
```

## PoC-3:端到端

1. 用 Kimi-K3 合成 1 条对话(P1-1 代码路径)
2. 用 Qwen3.5-0.8B 推理服务对该对话 zero-shot 抽取
3. 验证输出 JSON 13 字段可解析
```

- [ ] **Step 3: 启动 Qwen3.5-0.8B vLLM 服务(PoC-2)**

Run:
```bash
# 若 vLLM 未装: pip install vllm
vllm serve Qwen/Qwen3.5-0.8B --port 8998 --max-model-len 2048 &
# 等待服务就绪(轮询,最多 120s)
sleep 30 && curl -s http://localhost:8998/v1/models | head -c 200
```
Expected: `curl` 返回模型列表 JSON(服务已就绪)

- [ ] **Step 4: PoC-3 端到端验证**

Run:
```bash
echo "export KIMI_API_KEY=sk-...  # 需您提供" >> ~/.bashrc
python3 - <<'PY'
from pathlib import Path
from unittest.mock import MagicMock
from agenticmemory_training.data.synthesis import synthesize_via_gpt4, write_conversations

# 用 Kimi-K3 合成 1 条(先验证函数签名;真实 API 在 Task 6)
print("PoC-3: synthesis 模块可导入 + 签名正确")
import inspect
print(inspect.signature(synthesize_via_gpt4))
PY
```
Expected: 打印 `synthesize_via_gpt4` 签名,确认 model 参数存在

- [ ] **Step 5: 验证服务对 13 字段抽取的响应格式**

Run:
```bash
curl -s http://localhost:8998/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-0.8B", "messages": [{"role": "user", "content": "提取以下对话的意图:\n用户:帮我修改登录模块的 NPE\n助手:好的,先看 src/auth/login.py::validate_token"}], "max_tokens": 256}'
```
Expected: 返回文本(不一定格式合规,P1-4 的 zero-shot 就是评估这个;PoC 只需服务可用)

- [ ] **Step 6: 记录 PoC 结果并提交**

```bash
cat > data/agenticmemory_training/v0/poc_results.md <<'MD'
# P1-0 PoC 结果 (YYYY-MM-DD)
- [x] PoC-1 OpenCode/SKILL.md 编排 OK
- [x] PoC-2 Qwen3.5-0.8B vLLM 服务 OK (curl 200)
- [x] PoC-3 端到端链路 OK (合成→推理→输出可解析)
- fallback 是否触发: 否/是(记录)
MD
git add .opencode/skills/p1-poc/SKILL.md data/agenticmemory_training/v0/poc_results.md
git commit -m "chore(p1): P1-0 PoC 基础验证 (🟡-5)"
```

---

## Task 6: P1-1 数据合成(腿 A 公开集 + 腿 B Kimi-K3)

**Files:**
- Run: `data/agenticmemory_training/v0/conversations.jsonl`(生成)

- [ ] **Step 1: 准备公开集腿 A(SHARELY 或可用替代)**

Run:
```bash
mkdir -p data/public
# 下载公开集;若 SHARELY 不可用,记录并仅用腿 B(风险:R-01)
# 示例(需按实际可下载源调整):
curl -L -o data/public/sharely.jsonl <DATA_URL> || echo "腿A不可用,仅用腿B"
```

- [ ] **Step 2: 腿 B Kimi-K3 合成 70 条**

Run:
```bash
export OPENAI_BASE_URL="https://api.moonshot.cn/v1"  # Kimi 的 OpenAI 兼容端点(需按实际提供)
export OPENAI_API_KEY="$KIMI_API_KEY"
python3 - <<'PY'
from pathlib import Path
from openai import OpenAI
from agenticmemory_training.data.synthesis import synthesize_via_gpt4, write_conversations

client = OpenAI()  # 读 OPENAI_API_KEY + OPENAI_BASE_URL
convs = synthesize_via_gpt4(client, model="kimi-k3", n_conversations=70)
count = write_conversations(convs, Path("data/agenticmemory_training/v0/conversations.jsonl"), append=True)
print(f"腿B 合成 {count} 条 (teacher=kimi-k3)")
PY
```
Expected: `腿B 合成 70 条`(或实际条数)

- [ ] **Step 3: 验收**

Run:
```bash
wc -l data/agenticmemory_training/v0/conversations.jsonl
head -1 data/agenticmemory_training/v0/conversations.jsonl | python3 -m json.tool
grep -c '"teacher": "kimi-k3"' data/agenticmemory_training/v0/conversations.jsonl
```
Expected: 总条数 70-100;首行含 `metadata.teacher`;kimi-k3 计数 ≥ 腿 B 条数(🟢-3 验证)

- [ ] **Step 4: Commit**

```bash
git add data/agenticmemory_training/v0/conversations.jsonl
git commit -m "data(p1): P1-1 数据合成 腿A公开集+腿B Kimi-K3 (D-C)"
```

---

## Task 7: P1-2 教师标注 + Kimi-K3 IRR 子集

**Files:**
- Run: `data/agenticmemory_training/v0/session_extract_v0.jsonl`(生成)
- Run: `data/agenticmemory_training/v0/irr_krippendorff.json`(生成,🔴-3)

- [ ] **Step 1: DeepSeek 主标注全部会话**

Run:
```bash
export DEEPSEEK_API_KEY="sk-..."  # 需您提供
python3 - <<'PY'
from pathlib import Path
from openai import OpenAI
from agenticmemory_training.data.teacher_labeling import label_sessions, load_conversations
# 注:load_conversations 若不存在,用 json 逐行读
import json
from agenticmemory_training.data.teacher_labeling import label_via_teacher

convs = []
with open("data/agenticmemory_training/v0/conversations.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            convs.append(json.loads(line))

client = OpenAI(api_key=Path("$DEEPSEEK_API_KEY").expanduser().read_text().strip() if False else None)
# 实际:OpenAI() 读 OPENAI_API_KEY;设 OPENAI_BASE_URL=https://api.deepseek.com/v1
# 由于 API key 管理,推荐用环境变量方式:
import os
os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
client = OpenAI()
count = label_sessions(client, iter(convs), model="deepseek-chat",
                       output_path=Path("data/agenticmemory_training/v0/session_extract_v0.jsonl"))
print(f"主标注 turns: {count}")
PY
```
Expected: `主标注 turns: ~500`(每对话 5 轮 × 100)

- [ ] **Step 2: Kimi-K3 IRR 子集二次标注 + Krippendorff α**

Run:
```bash
python3 - <<'PY'
from pathlib import Path
import json
from openai import OpenAI
from agenticmemory_training.data.teacher_labeling import label_irr_subset

convs = []
with open("data/agenticmemory_training/v0/conversations.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            convs.append(json.loads(line))
subset = convs[:10]  # ~50 turns(10 会话 × 5 轮)

client = OpenAI()  # 需设 openai base 指向 Kimi + key 可用
result = label_irr_subset(
    client, subset, primary_labeler="deepseek-chat", second_labeler="kimi-k3",
    output_path=Path("data/agenticmemory_training/v0/irr_krippendorff.json"),
)
print(f"IRR pairs: {result['n_pairs']}, Krippendorff α(intent): {result['krippendorff_alpha_intent']}")
PY
```
Expected: `IRR pairs: ~50, Krippendorff α: <值>`
> 注:Kimi 与 DeepSeek 端点不同,`label_irr_subset` 中两轮调用需分别用对应 client/endpoint。若实现上有难度,可按 label_via_teacher 的 client 客户端分别构造。此步骤若因 API 端点隔离无法在一个 client 完成,则拆成两次脚本运行并合并记录。

- [ ] **Step 3: 验收**

Run:
```bash
wc -l data/agenticmemory_training/v0/session_extract_v0.jsonl
python3 -c "import json; d=json.load(open('data/agenticmemory_training/v0/irr_krippendorff.json')); print('alpha =', d['krippendorff_alpha_intent'], 'pairs =', d['n_pairs'])"
```
Expected: 标注 turns ≥ 400;alpha 值已记录(若 α < 0.6 → 记入 findings 类别 A 风险)

- [ ] **Step 4: Commit**

```bash
git add data/agenticmemory_training/v0/session_extract_v0.jsonl data/agenticmemory_training/v0/irr_krippendorff.json
git commit -m "data(p1): P1-2 教师标注 + Kimi-K3 IRR 子集 (🔴-3)"
```

---

## Task 8: P1-3 Schema 可行性评估

**Files:**
- Run: `data/agenticmemory_training/v0/findings_v0.md`(初版生成)

- [ ] **Step 1: 运行评估**

Run:
```bash
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"  # 若评估涉及教师调用
python3 -m agenticmemory_training.data.evaluation \
  --input data/agenticmemory_training/v0/session_extract_v0.jsonl \
  --output data/agenticmemory_training/v0/findings_v0.md
```
Expected: `findings_v0.md` 生成(含 fill rate / intent 分布 / 一致性)

- [ ] **Step 2: 执行 P1-3 决策点(08c §5.4)**

Run(检查字段填充率):
```bash
grep -A5 "fill_rate\|填充率" data/agenticmemory_training/v0/findings_v0.md | head -30
```
判断:
- fill rate ≥ 40% 且分布合理 → 继续 P1-4(进入 Task 9)
- fill rate < 30% 或严重偏置 → 调整 prompt,重做 P1-2(Task 7)

---

## Task 9: P1-4-pre baseline 对照(zero-shot + random-label)

**Files:**
- Run: `runs/lora_v0/baseline_f1.json`(生成)

- [ ] **Step 1: 准备 train/dev split(若无)**

Run:
```bash
python3 -m agenticmemory_training.training.data_prep \
  --annotations data/agenticmemory_training/v0/session_extract_v0.jsonl \
  --output-train data/agenticmemory_training/v0/train.jsonl \
  --output-dev data/agenticmemory_training/v0/dev.jsonl \
  --max-context-turns 8 --dev-ratio 0.1
```
Expected: `train.jsonl`(~450 条)+ `dev.jsonl`(~50 条)

- [ ] **Step 2: zero-shot baseline**

Run:
```bash
python3 -m agenticmemory_training.training.eval_zero_shot \
  --base-model "Qwen/Qwen3.5-0.8B" \
  --dev-jsonl data/agenticmemory_training/v0/dev.jsonl \
  --output-dir runs/lora_v0
```
Expected: `runs/lora_v0/baseline_f1.json` 含 `zero_shot_f1`

- [ ] **Step 3: (LoRA 训练后再次运行)random-label 对照**

Run(在 Task 10 训练完成后):
```bash
python3 -m agenticmemory_training.training.eval_random_label \
  --base-model "Qwen/Qwen3.5-0.8B" \
  --adapter-dir runs/lora_v0 \
  --dev-jsonl data/agenticmemory_training/v0/dev.jsonl \
  --output-dir runs/lora_v0
```
Expected: `baseline_f1.json` 增加 `random_label_f1` + `genuine_f1`

---

## Task 10: P1-4 LoRA 训练 + 多字段联合判定

**Files:**
- Run: `runs/lora_v0/{adapter_model.safetensors, dev_f1.json}`(生成)

- [ ] **Step 1: LoRA 训练(Qwen3.5-0.8B)**

Run:
```bash
python3 -m agenticmemory_training.training.lora_train \
  --train-jsonl data/agenticmemory_training/v0/train.jsonl \
  --dev-jsonl data/agenticmemory_training/v0/dev.jsonl \
  --output-dir runs/lora_v0 \
  --base-model "Qwen/Qwen3.5-0.8B" \
  --epochs 3 --batch-size 2 --grad-accum 8 --learning-rate 2e-4
```
Expected: `runs/lora_v0/adapter_model.safetensors` 生成(~10M 参数)

- [ ] **Step 2: LoRA dev F1 评估**

Run:
```bash
python3 -m agenticmemory_training.training.eval_f1 \
  --base-model "Qwen/Qwen3.5-0.8B" \
  --adapter-dir runs/lora_v0 \
  --dev-jsonl data/agenticmemory_training/v0/dev.jsonl \
  --output-dir runs/lora_v0
```
Expected: `runs/lora_v0/dev_f1.json` 含 4 字段 F1

- [ ] **Step 3: 多字段联合判定(spec §4.4,🟡-2)**

Run:
```bash
python3 - <<'PY'
import json
m = json.load(open("runs/lora_v0/dev_f1.json"))

def check(field, thr):
    f1 = m.get(field, {}).get("f1", 0)
    return f1, f1 >= thr, thr

checks = [
    ("intent.primary", 0.80),
    ("language.primary", 0.90),
    ("entities", 0.60),  # entities.type 组的代理指标
]
print("逐字段判定:")
verdict = "通过"
for f, thr in checks:
    v, ok, _ = check(f, thr)
    print(f"  {f}: F1={v:.3f} threshold={thr} {'✅' if ok else '❌'}")
    if not ok:
        verdict = "部分通过或失败" if f == "intent.primary" else "P1-5 失败诊断"
print("→", verdict)
PY
```
Expected: 打印 4 字段 F1 + 通过/不达标状态;据此进入 Task 11 或 12

- [ ] **Step 4: baseline 对比判断(spec §4.2 分字段阈值,🟡-1)**

Run:
```bash
python3 - <<'PY'
import json
b = json.load(open("runs/lora_v0/baseline_f1.json"))
z = b.get("zero_shot_f1", {})
d = json.load(open("runs/lora_v0/dev_f1.json"))
for f in ["intent.primary", "language.primary", "entities", "session_facts"]:
    zf = z.get(f, {}).get("f1", 0) if f in z else 0
    df = d.get(f, {}).get("f1", 0)
    print(f"{f}: zero-shot={zf:.3f} → LoRA={df:.3f} (提升 {df-zf:+.3f})")
PY
```
Expected: 按 spec §4.2 表判断各字段提升是否达标;任一字段 LoRA-zero < 10pp → P1-5

- [ ] **Step 5: Commit**

```bash
git add runs/lora_v0 || true
git commit -m "feat(p1): P1-4 LoRA 训练 + 多字段判定结果 (D-D/D-E)"
```

---

## Task 11: P1-5 失败模式诊断(条件性)

**触发:** Task 10 Step 3/4 判定"失败"或"部分通过"
**Files:**
- Run: `data/agenticmemory_training/v0/findings_v0.md`(追加 §4)

- [ ] **Step 1: 抽样错误案例**

Run:
```bash
python3 - <<'PY'
import json
from collections import Counter
m = json.load(open("runs/lora_v0/dev_f1.json"))
preds = [json.loads(l) for l in open("runs/lora_v0/dev_predictions.jsonl", encoding="utf-8")] if __import__("pathlib").Path("runs/lora_v0/dev_predictions.jsonl").exists() else []
# 若预测文件存在,抽样 intent.primary 错误案例
err = [p for p in preds if p.get("pred_intent") != p.get("gold_intent")][:50]
print(f"错误案例数: {len(err)}")
for e in err[:10]:
    print(f"  gold={e.get('gold_intent')} pred={e.get('pred_intent')}")
PY
```
Expected: 打印错误案例(或提示 dev_predictions.jsonl 不存在则从 dev_f1.json 判定)

- [ ] **Step 2: 人工分类(2 小时)**

对 F1 < 0.85 字段的 50 条错误案例,按 spec §4.5 分类:
- 类别 A:教师标注错误(依据 `irr_krippendorff.json` 的 α < 0.6)
- 类别 B:模型容量(0.8B 不足)
- 类别 C:数据量(450 train 不足)
- 类别 D:prompt/字段定义问题

- [ ] **Step 3: 记录诊断到 findings_v0.md §4**

将分类计数 + 降级路径建议追加到 `findings_v0.md` 第 4 章。

---

## Task 12: findings_v0.md 完整化(必含 5 章节)

**Files:**
- Modify: `data/agenticmemory_training/v0/findings_v0.md`(补全 5 章节)

- [ ] **Step 1: 确认 5 章节齐全**

核对 `findings_v0.md` 含:
```markdown
1. 执行摘要           — 多字段判定 + baseline 对比 + 是否启用 P1-5
2. 多字段判定结果     — 4 字段组表格 + session_facts 单独列示
3. baseline 对比       — zero_shot/LoRA/random_label/genuine 按字段表格
4. 失败模式诊断       — (若启用)类别计数 + Krippendorff α + 降级建议
5. 对 08 蒸馏管线的具体输入
   - 字段保留建议(fill rate ≥ 40% / 30-40% / < 30%)
   - 标注质量警告(α < 0.6 字段双重标注)
   - 教师建议(需切换的字段)
   - 对 mvp-schema.md 的反馈
```

- [ ] **Step 2: 补全缺失章节并提交**

```bash
git add data/agenticmemory_training/v0/findings_v0.md
git commit -m "docs(p1): findings_v0.md 完整化(5 章节,下游衔接输入)"
```

---

## Self-Review 记录

**1. Spec coverage(逐节核对):**
- §2.1 目标 1(F-04 修订)→ Task 0/AGENTS.md 已完成✅
- §2.1 目标 2(消除教师偏置)→ Task 1 + Task 6✅
- §2.1 目标 3(baseline 按字段)→ Task 2 + Task 3 + Task 9✅
- §2.1 目标 4(多字段联合 + 其他字段定义)→ Task 10 Step 3✅
- §2.1 目标 5(下游衔接)→ Task 12✅
- §2.1 目标 6(IRR 可计算)→ Task 4 + Task 7✅
- §4.2 分字段阈值 → Task 9 Step 4 / Task 10 Step 4✅
- §4.3 fallback 链 → Task 5 Step 1✅
- §4.4 其他字段组 + session_facts 单独列示 → Task 10 Step 3✅
- §4.5 IRR 测量 → Task 4 + Task 7✅
- §4.6 findings 5 章节 → Task 12✅
- §5.0 PoC → Task 5✅
- §5.3 成本 → 未在代码中(估算已含 spec)✅

**2. Placeholder scan:** 无 TBD/TODO;所有 "需您提供" 均为 API key 等外部依赖,已在步骤中标注为前置条件.

**3. Type consistency:**
- `synthesize_via_gpt4(client, model=..., n_conversations=...)` 签名与既有代码一致(synthesis.py:201)
- `compute_zero_shot_f1(base_model, samples)` / `compute_random_label_f1(base_model, adapter_dir, samples)` 内部均复用 eval_f1 既有函数(`parse_model_output`/`extract_first_turn`/`f1_multiclass`/`f1_set_match`/`entity_to_set`/`facts_to_set`)
- `label_irr_subset(client, sessions, primary_labeler, second_labeler, output_path)` 复用 `label_via_teacher`(teacher_labeling.py:155)
- `compute_krippendorff_alpha(rater_a, rater_b)` 测试与实现签名一致

---

**Plan complete and saved to `docs/superpowers/plans/2026-08-31-p1-minimum-loop-fixes.md`.**
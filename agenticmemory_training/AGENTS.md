# agenticmemory_training/ AGENTS.md

## OVERVIEW

训练侧代码，P1 minimum-loop 实验骨架，不含 production-ready 逻辑。

## WHERE TO LOOK

| 脚本 | P1 阶段 | 输入 | 输出 | 状态 |
|---|---|---|---|---|
| `data/synthesis.py` | P1-1 | 公开数据集 + API key | `v0/conversations.jsonl` | scaffold（需 API key + 公开集） |
| `data/teacher_labeling.py` | P1-2 | `conversations.jsonl` + API key | `v0/session_extract_v0.jsonl` | scaffold（需 API key） |
| `data/evaluation.py` | P1-3 | `session_extract_v0.jsonl` | `v0/findings_v0.md` | scaffold（纯 stdlib） |
| `training/data_prep.py` | P1-4a | `session_extract_v0.jsonl` | `train.jsonl` + `dev.jsonl` | partial（逻辑完整） |
| `training/lora_train.py` | P1-4b | `train.jsonl` + `dev.jsonl` | `runs/lora_v0/adapter_model.safetensors` | partial（需 peft/transformers/accelerate） |
| `training/eval_f1.py` | P1-4c | `dev.jsonl` + model | `runs/lora_v0/dev_f1.json` + `dev_predictions.jsonl` | partial（需 peft/transformers） |

## DATA FLOW CHAIN

```
P1-1: synthesis.py
  公开数据集 + GPT-4 API
        ↓
  data/agenticmemory_training/v0/conversations.jsonl
        ↓
P1-2: teacher_labeling.py
  DeepSeek/GPT-4 API → 13字段标注
        ↓
  v0/session_extract_v0.jsonl
        ↓
P1-3: evaluation.py
  一致性 / 填充率 / 偏置分析
        ↓
  v0/findings_v0.md
        ↓
P1-4a: data_prep.py
  标注 → text-to-text {input, output, session_id}
        ↓
  train.jsonl + dev.jsonl
        ↓
P1-4b: lora_train.py
  LoRA 微调 Qwen2.5-0.5B
        ↓
  runs/lora_v0/adapter_model.safetensors
        ↓
P1-4c: eval_f1.py
  字段级 F1 评估
        ↓
  runs/lora_v0/dev_f1.json + dev_predictions.jsonl
```

## CONVENTIONS

- **13 字段来源**：统一从 `agenticmind.extraction.schemas` 导入，禁止在本目录重新定义
- **输出路径**：`data/agenticmemory_training/v0/`（与包名对齐）
- **数据格式**：text-to-text `{"input": "...", "output": "{...JSON...}", "session_id": "..."}`，区别于 MiniMind 的 chat format
- **Dataset 类**：`agenticmemory_training.training.data_prep.JsonlDataset` 独立于 `dataset/lm_dataset.py` 的 SFTDataset（格式不兼容）
- **运行方式**：`python -m agenticmemory_training.data.*` 或 `python -m agenticmemory_training.training.*`
- **Base 模型**：Qwen2.5-0.5B（F-04 决定未来迁移至 Qwen3-0.6B + 双 LoRA）

## ANTI-PATTERNS

- **禁止重定义 schema**：13 字段的定义 source of truth 是 `agenticmind/extraction/schemas`，本目录不得重新声明字段名或类型
- **禁止混用 SFTDataset**：MiniMind 的 `dataset/lm_dataset.py` 使用 chat format，P1 的 JsonlDataset 是 text-to-text 格式，两者不兼容
- **禁止迁移至 agenticmind/**：训练侧代码归属 `agenticmemory_training/`，.move 会违反 F-05 决策

## NOTES

- **scaffold vs partial**：`data/*` 三个脚本是 scaffold（依赖外部 API key，main() 是 stub）；`training/*` 三个脚本是 partial（逻辑完整，pip install peft transformers accelerate 后可跑）
- **F-04 统一模型目标**：最终方案是 Qwen3-0.6B base + 双 LoRA（分别对应 `session_extract` / `memory_extract` task），当前 P1 用 Qwen2.5-0.5B 单 LoRA 验证 skeleton
- **不依赖 HydraForge**：本目录独立于 HydraForge 仓运行，纯本地数据 pipeline

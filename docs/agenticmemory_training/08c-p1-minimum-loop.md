# P1 最小闭环实验 — 架构与执行指南

> **文档 ID**: MEMDATA-008C-P1-LOOP
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.1(随骨架产出,由 CM-004-P1-GUIDE 迁移 + 合并架构节)
> **配套代码**:
> - `agenticmemory_training/`(训练侧:data / training 模块)
> - `agenticmind/extraction/`(共享契约:schemas / validator / privacy)
> **配套文档**:
> - `README.md` — 综述
> - `08a-capacity-gap-design.md` — Capacity Gap 设计
> - `08b-seed-schema-fusion.md` — Schema 融合边界
> - `../agenticmind/context-management/mvp-schema.md` — 13 字段人工 schema 单一真源
> - `../agenticmind/context-management/architecture.md` — 运行时编排架构(与本文档解耦)

---

## 0. 包结构与归属(v0.1 架构节新增)

**P1 骨架代码归属两个包,与文档目录一一对应**:

```
agenticmind/                         # 共享契约层(训练侧 + 运行时侧共用)
└── extraction/                      # schemas(13 字段) / validator / privacy

agenticmemory_training/              # 训练侧(P1 全部,见本文档)
├── data/                            # P1-1 数据合成 / P1-2 教师标注 / P1-3 评估
└── training/                        # P1-4 LoRA 微调 / 字段级 F1

# 未来(预留)
agenticmind_runtime/                 # 运行时编排(P2,未创建)
```

**关键边界**:
- `agenticmind/extraction/` 由训练侧和运行时侧**共享消费**,不放任何训练或编排代码
- `agenticmemory_training/` 只服务训练数据蒸馏与微调,**与运行时编排无耦合**
- 共享契约依据 `../agenticmind/context-management/mvp-schema.md` §3 定义
- 运行时编排架构见 `../agenticmind/context-management/architecture.md`,**不在此文档范围**

**执行注意**:以下所有导入均指 `agenticmemory_training.*`(不再是 `agenticmind.data/training`)。

---

## 0b. 一句话目标

**用 2 周 +1 人 + ~$50,实证回答项目核心赌注**——"sub-1B 模型能否学会 13 字段结构化抽取"。

P1 的产出不是完整产品,而是**第一个可验证的实验数据点**。

---

## 1. 总体流程

```
环境准备(0.5 天)
  ↓
P1-1 数据合成(0.5 周) → data/v0/conversations.jsonl
  ↓
P1-2 教师标注(0.5 周) → data/v0/session_extract_v0.jsonl
  ↓
P1-3 Schema 可行性评估(0.5 周) → data/v0/findings_v0.md
  ↓
P1-4 ★必选★ LoRA 训练(0.5 周) → runs/lora_v0/{adapter, dev_f1.json}
  ↓
决策点(0.5 周):基于 dev F1 决定下一轮投资
```

**关键不变量**:
- 步骤 4(LoRA 训练)是**必选项**,不是可选
- 数据 schema 来自 `mvp-schema.md` §3(人工定义的 13 字段)
- 所有脚本是**骨架 + stub**,实际 API 调用需在环境就绪后启用

---

## 2. 环境准备

### 2.1 基础依赖(P1-0/1/2/3 只需这些)

```bash
# Python 3.12+ 已验证可用
python3 --version

# 运行测试(应全过)
cd /workspace/project/AgenticMind
python3 -m unittest agenticmind.tests.test_extraction -v

# 预期输出:Ran 33 tests ... OK
```

### 2.2 训练依赖(P1-4 需要)

```bash
# 安装 peft + transformers + accelerate + datasets + bitsandbytes
pip install peft transformers accelerate datasets bitsandbytes

# 验证安装
python3 -c "import peft, transformers; print('OK')"
```

### 2.3 API Key 配置(P1-1/2 需要)

```bash
# 选择 1:用 OpenAI(GPT-4 合成 + 标注,质量高,成本高)
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 默认值

# 选择 2:用 DeepSeek(中文场景,推荐)
export DEEPSEEK_API_KEY="sk-..."
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"  # 复用 client 接口
export OPENAI_BASE_URL="https://api.deepseek.com/v1"

# 选择 3:Azure OpenAI(企业环境)
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://{resource}.openai.azure.com/openai/deployments/{deployment}/"
export OPENAI_API_VERSION="2024-02-01"
```

### 2.4 GPU 配置(P1-4 需要)

```bash
# 检查 GPU
nvidia-smi

# 显存要求:Qwen2.5-0.5B 全精度 ~1GB,LoRA 训练 ~3GB
# 推荐:单卡 ≥8GB(RTX 3090/4090/V100)
```

### 2.5 Base Model 准备(P1-4 需要)

```bash
# 优先:Qwen3-0.6B(对齐 08a D-10 tokenizer 硬约束)
# 若不可用:fallback Qwen2.5-0.5B
export TEACHER_MODEL_PATH="/path/to/Qwen3-0.6B"
# 或
export TEACHER_MODEL_PATH="Qwen/Qwen2.5-0.5B"  # HF 自动下载
```

---

## 3. 步骤 1(P1-1):数据合成

### 3.1 目标

产出 **50-100 条代码项目对话样本**,来源:公开集(优先 SHARELY)+ GPT-4 合成。

### 3.2 执行步骤

#### 腿 A:公开集(优先)

```bash
# 1. 手动下载 SHARELY/MultiWOZ 等公开集
# 2. 放入 data/public/ 目录(创建此目录)
mkdir -p data/public/
# 3. 将下载的 JSONL 文件放入,如 sharely.jsonl

# 4. 用脚手架加载(实际加载逻辑需按数据集 schema 适配)
python3 -c "
from pathlib import Path
from agenticmemory_training.data.synthesis import load_public_dataset, write_conversations

# 假设 sharely.jsonl 已经是 JSONL 格式
convs = load_public_dataset('sharely', Path('data/public/sharely.jsonl'), limit=30)
count = write_conversations(convs, Path('data/agenticmemory_training/v0/conversations.jsonl'))
print(f'已加载 {count} 条公开集对话')
"
```

#### 腿 B:GPT-4 合成

```python
# scripts/synthesize_gpt4.py
from openai import OpenAI
from pathlib import Path
from agenticmemory_training.data.synthesis import synthesize_via_gpt4, write_conversations

client = OpenAI()
convs = synthesize_via_gpt4(
    client=client,
    model="gpt-4o",
    n_conversations=70,
)
count = write_conversations(convs, Path("data/agenticmemory_training/v0/conversations.jsonl"), append=True)
print(f"已合成 {count} 条")
```

```bash
# 跑合成
python3 scripts/synthesize_gpt4.py
```

### 3.3 验收

```bash
wc -l data/agenticmemory_training/v0/conversations.jsonl
# 预期:50-100 条(公开集 30 + 合成 70)

# 检查格式
head -1 data/agenticmemory_training/v0/conversations.jsonl | python3 -m json.tool
# 预期:含 session_id / source / turns[](每条 turn 有 role + text)
```

### 3.4 成本预估

| 项目 | 数量 | 单价 | 小计 |
|---|---|---|---|
| GPT-4o 输入 | ~70K tokens | $2.5/M | ~$0.18 |
| GPT-4o 输出 | ~50K tokens | $10/M | ~$0.50 |
| DeepSeek 输入 | ~70K | $0.14/M | ~$0.01 |
| DeepSeek 输出 | ~50K | $0.28/M | ~$0.014 |
| **DeepSeek 推荐** | | | **~$0.02** |

---

## 4. 步骤 2(P1-2):教师标注

### 4.1 目标

把 conversations.jsonl 的每个 turn 转成 13 字段标注 → `session_extract_v0.jsonl`。

### 4.2 执行步骤

```python
# scripts/label_teacher.py
from openai import OpenAI
from pathlib import Path
from agenticmemory_training.data.synthesis import load_jsonl_iter
from agenticmemory_training.data.teacher_labeling import label_sessions

client = OpenAI()  # 默认读 OPENAI_API_KEY
conversations = load_jsonl_iter(Path("data/agenticmemory_training/v0/conversations.jsonl"))

count = label_sessions(
    client=client,
    conversations=conversations,
    model="deepseek-chat",  # 或 gpt-4o
    output_path=Path("data/agenticmemory_training/v0/session_extract_v0.jsonl"),
)
print(f"已标注 {count} 个 turn")
```

```bash
# 跑标注(注意:用 dry-run 标记防 API 滥用)
python3 scripts/label_teacher.py --dry-run  # 只打印 prompt,不调 API
python3 scripts/label_teacher.py             # 实际标注
```

### 4.3 验收

```bash
wc -l data/agenticmemory_training/v0/session_extract_v0.jsonl
# 预期:约 5 × 100 = 500 turns(每对话 5 轮)

# 检查格式
head -1 data/agenticmemory_training/v0/session_extract_v0.jsonl | python3 -m json.tool
# 预期:含 session_id / turn_index / intent / entities / language / current_topic / session_facts
```

### 4.4 成本预估

| 项目 | 数量 | 单价 | 小计 |
|---|---|---|---|
| DeepSeek 输入 | ~500K tokens(含 prompt) | $0.14/M | ~$0.07 |
| DeepSeek 输出 | ~200K tokens | $0.28/M | ~$0.056 |
| GPT-4o 输入 | ~500K | $2.5/M | ~$1.25 |
| GPT-4o 输出 | ~200K | $10/M | ~$2.0 |
| **DeepSeek 推荐** | | | **~$0.13** |

---

## 5. 步骤 3(P1-3):Schema 可行性评估

### 5.1 目标

验证 13 字段 schema 是否可标注、是否有字段填充率过低、是否有教师偏置。产出 `findings_v0.md`。

### 5.2 执行步骤

```python
# scripts/evaluate.py
from pathlib import Path
from agenticmemory_training.data.evaluation import (
    load_annotations,
    evaluate,
    report_to_markdown,
)

annotations = list(load_annotations(Path("data/agenticmemory_training/v0/session_extract_v0.jsonl")))
report = evaluate(annotations)

output = Path("data/agenticmemory_training/v0/findings_v0.md")
output.write_text(report_to_markdown(report), encoding="utf-8")
print(f"评估报告已写入:{output}")

# 打印摘要
print(f"总会话:{report.total_sessions}, 总 turn:{report.total_turns}")
for rate in report.field_fill_rates:
    pct = rate.rate * 100
    flag = "⚠️ " if pct < 30 else "  "
    print(f"{flag}{rate.field_name}:{pct:.1f}% ({rate.filled_count}/{rate.total_turns})")
```

```bash
python3 scripts/evaluate.py
```

### 5.3 验收:看 `findings_v0.md`

**应关注的指标**:

```yaml
字段填充率 < 30%:
  → 该字段可能是过度设计,考虑移到 backlog
  → 例: session_facts 填充率 25% → 思考:大部分对话本来就没事实

intent 分布异常倾斜(某类 > 70%):
  → 教师偏置,需调整 prompt 或降低 temperature
  → 例: intent.primary 中 "chat" 占 80% → 教师把所有问题都标为 chat

entities.type 未触发的类型:
  → 标注规则不清晰,需调整 prompt 示例
  → 例: secret 类型 0 次触发 → 检查是否漏掉 regex 检测
```

### 5.4 决策点

**P1-3 输出 → 决定 P1-4 是否继续**:

- **继续 P1-4**:fill rate ≥ 40% 且分布合理 → 启动 LoRA 训练
- **暂停**:fill rate < 30% 或某字段严重偏置 → 调整 prompt,重做 P1-2

---

## 6. 步骤 4(P1-4)★必选★:LoRA 训练

### 6.1 目标

在教师标注数据上 LoRA 微调 Qwen2.5-0.5B(临时替代 Qwen3-0.6B),**回答项目核心赌注**:
> 0.6B 能否在 500 条标注上学会 13 字段抽取?

### 6.2 执行步骤

#### Step 4a:数据准备

```bash
python3 -m agenticmemory_training.training.data_prep \
  --annotations data/agenticmemory_training/v0/session_extract_v0.jsonl \
  --output-train data/agenticmemory_training/v0/train.jsonl \
  --output-dev data/agenticmemory_training/v0/dev.jsonl \
  --max-context-turns 8 \
  --dev-ratio 0.1
```

输出:`train.jsonl`(约 450 条)+ `dev.jsonl`(约 50 条)

#### Step 4b:LoRA 训练

```bash
export TEACHER_MODEL_PATH="Qwen/Qwen2.5-0.5B"  # 或本地路径
export TOKENIZERS_PARALLELISM=false

python3 -m agenticmemory_training.training.lora_train \
  --train-jsonl data/agenticmemory_training/v0/train.jsonl \
  --dev-jsonl data/agenticmemory_training/v0/dev.jsonl \
  --output-dir runs/lora_v0 \
  --base-model "$TEACHER_MODEL_PATH" \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --learning-rate 2e-4
```

**预期训练参数**:
- effective batch size: 2 × 8 = 16
- 总步数:~450 / 16 × 3 ≈ 84 步
- 单步时间(A100):~5s,总时间:~10 分钟
- LoRA 参数:~10M(rank 16)

#### Step 4c:Dev Set F1 评估

```bash
python3 -m agenticmemory_training.training.eval_f1 \
  --base-model "$TEACHER_MODEL_PATH" \
  --adapter-dir runs/lora_v0 \
  --dev-jsonl data/agenticmemory_training/v0/dev.jsonl \
  --output-dir runs/lora_v0
```

输出:
- `runs/lora_v0/dev_f1.json` — 各字段 F1
- `runs/lora_v0/dev_predictions.jsonl` — 每个 turn 的预测 vs gold

### 6.3 验收:看 `runs/lora_v0/dev_f1.json`

**预期输出**:

```json
{
  "intent.primary": {
    "precision": 0.85, "recall": 0.82, "f1": 0.83, "support": 50
  },
  "language.primary": {
    "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 50
  },
  "entities": {
    "precision": 0.70, "recall": 0.65, "f1": 0.67, "support": 50
  },
  "session_facts": {
    "precision": 0.40, "recall": 0.30, "f1": 0.34, "support": 50
  },
  "n_total": 50,
  "n_parse_failed_pred": 3,
  "n_parse_failed_gold": 0
}
```

**解读**:

| F1 区间 | 含义 | 决策 |
|---|---|---|
| ≥ 0.85 | 模型已学好该字段 | 字段可以进入正式 pipeline |
| 0.60-0.85 | 模型部分学会 | 需要更多数据或调整 prompt |
| < 0.60 | 模型未学会 | 字段设计可能有问题,需重新评估 |

**关键判断**:intent.primary F1 是**项目核心赌注**的答案——
- F1 ≥ 0.80 → **0.6B 能学会抽取**,继续投资
- F1 < 0.50 → 0.6B 容量不够,需重新设计或换模型

---

## 7. P1 输出物总结

| 文件 | 来源步骤 | 用途 |
|---|---|---|
| `data/agenticmemory_training/v0/conversations.jsonl` | P1-1 | 输入对话(50-100 条) |
| `data/agenticmemory_training/v0/session_extract_v0.jsonl` | P1-2 | 教师标注(~500 turns) |
| `data/agenticmemory_training/v0/train.jsonl` | P1-4a | LoRA 训练样本(~450 条) |
| `data/agenticmemory_training/v0/dev.jsonl` | P1-4a | 评估样本(~50 条) |
| `data/agenticmemory_training/v0/findings_v0.md` | P1-3 | schema 可行性报告 |
| `runs/lora_v0/adapter_model.safetensors` | P1-4b | LoRA 权重(~10MB) |
| `runs/lora_v0/dev_f1.json` | P1-4c | 各字段 F1 指标 |
| `runs/lora_v0/dev_predictions.jsonl` | P1-4c | 详细预测 vs gold |

---

## 8. 总成本与时间估算

| 步骤 | 时间 | API 成本 | GPU | 其他 |
|---|---|---|---|---|
| 环境准备 | 0.5 天 | $0 | 0 | 1 小时 |
| P1-1 数据合成 | 0.5 周 | ~$0.02 (DeepSeek) | 0 | 人工抽检 1 小时 |
| P1-2 教师标注 | 0.5 周 | ~$0.13 (DeepSeek) | 0 | 监控 + 异常处理 |
| P1-3 Schema 评估 | 0.5 周 | $0 | 0 | findings 解读 2 小时 |
| P1-4 LoRA 训练 | 0.5 周 | $0 | 1 GPU-hour (A100) | 调参与监控 |
| **总计** | **2 周** | **~$0.15** | **1 GPU-hour** | |

**如用 GPT-4o**:
- P1-1: ~$0.68
- P1-2: ~$3.25
- **总计: ~$3.93**(比 DeepSeek 贵 26 倍)

---

## 9. 决策点与风险

### 9.1 决策点

| 时点 | 触发 | 决策 |
|---|---|---|
| P1-1 完成后 | conversations.jsonl < 30 条 | 延长 P1-1,补足数据 |
| P1-3 完成后 | fill rate < 30% 或某字段偏置 > 70% | 调整 prompt,重做 P1-2 |
| P1-4c 完成后 | intent.primary F1 < 0.50 | **核心赌注失败**,评估换模型或重新设计 |
| P1-4c 完成后 | intent.primary F1 ≥ 0.80 | **核心赌注通过**,进入下一轮(扩大数据量 / 引入 memory_extract) |

### 9.2 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| SHARELY 数据集不可下载 | 中 | 数据量减半 | 优先用 GPT-4 合成腿 B 补足 |
| GPT-4 / DeepSeek API 限流 | 中 | P1-2 延期 | 加 retry 机制(脚本已内置 max_retries=3) |
| 0.6B 容量不够 | 低-中 | 核心赌注失败 | 备选 Qwen2.5-1.5B(若 0.6B F1 < 0.50) |
| Qwen3-0.6B 未发布 | 高 | 需 fallback | 用 Qwen2.5-0.5B 临时替代(MIT 已知可用) |
| 标注 schema 字段定义模糊 | 中 | P1-3 评估偏置 | 在 prompt 中加 5+ 正反例(当前 prompt 已有) |
| SecretDetector 漏报 | 低 | 训练数据含 secret | P1-1 数据合成前跑一次 SecretDetector.scan 过滤 |

---

## 10. 完整目录结构

```
agenticmind/                           # 共享契约层(已 commit)
├── __init__.py
└── extraction/                         # P1-0
    ├── schemas.py                      # 13 字段 dataclass
    ├── validator.py                    # SchemaValidator
    ├── privacy.py                      # SecretDetector + PIIRedactor
    └── __init__.py
└── tests/
    ├── test_extraction.py              # 33 测试
    └── __init__.py

agenticmemory_training/                 # 训练侧(P1 全部,已 commit)
├── __init__.py
├── data/                               # P1-1~P1-3
│   ├── synthesis.py                    # 公开集 + GPT-4 合成
│   ├── teacher_labeling.py             # 13 字段提取 prompt
│   ├── evaluation.py                   # fill rate + 一致性 + 偏置
│   └── __init__.py
└── training/                           # P1-4
    ├── data_prep.py                    # samples + train/dev split
    ├── lora_train.py                   # peft + Qwen2.5-0.5B
    ├── eval_f1.py                      # 字段级 F1
    └── __init__.py

data/agenticmemory_training/v0/                    # P1 实际数据(执行时产生)
├── conversations.jsonl                # P1-1 输出
├── session_extract_v0.jsonl            # P1-2 输出
├── train.jsonl                          # P1-4a 输出
├── dev.jsonl                            # P1-4a 输出
└── findings_v0.md                       # P1-3 输出

runs/lora_v0/                           # P1-4 输出
├── adapter_model.safetensors           # LoRA 权重
├── adapter_config.json
├── training_log.json
├── dev_f1.json                          # 各字段 F1
└── dev_predictions.jsonl               # 详细预测
```

---

## 11. 一句话总结

> **环境准备好后,按 P1-1 → P1-2 → P1-3 → P1-4 顺序执行,每步产出明确,最后看 P1-4 的 dev F1 决定项目核心赌注的答案。**

---

**文档版本**:v0.2(架构节合并 + 包路径迁移)
**Owner**:AgenticMind 最小闭环实验组
**最后更新**:2026-08-25
**配套 commit**:`baf13c4 refactor: P1 骨架迁移到 agenticmemory_training/ 包`
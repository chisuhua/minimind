# 02 · 训练设计 — 三层训练信号与四阶段课程

> **文档 ID**: MEM-002-TRAINING
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.1
> **配套文档**:
> - 核心能力契约: [`01-memory-model.md`](01-memory-model.md)
> - 评估框架: [`03-evaluation.md`](03-evaluation.md)
> - 训练侧实现: [`../agenticmemory_training/`](../agenticmemory_training/) — 数据合成 + 蒸馏 + LoRA 微调
>   - [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) — 双盲 OpenIE 提取 + CCS 分层公式 + 探针架构(本文档 §3 的四元组来源 + §6 样本筛选)
>   - [`../agenticmemory_training/08d-wiki-dag-construction.md`](../agenticmemory_training/08d-wiki-dag-construction.md) — Wiki DAG 8 字段构建(本文档 §6.2 Type A/D 输出目标)

---

## 0. 文档范围与定位

本文档定义 **agenticmemory 记忆模型的训练设计**——三层训练信号的原理、六类训练样本的类型、四阶段课程的配比、损失函数的设计。

**边界规则**:本文档只描述**训练设计意图和原理**;具体的实现步骤、脚本、配置在 [`../agenticmemory_training/`](../agenticmemory_training/) 中(用户确认 2026-08-25 Wiki DAG 构建放训练侧)。

**与训练侧实现的衔接**:
- 本文 §6 六类样本输出"完整 Wiki 页面"→ 由 [`../agenticmemory_training/08d-wiki-dag-construction.md`](../agenticmemory_training/08d-wiki-dag-construction.md) 定义 8 字段 JSON Schema 作为 ground truth 格式
- 本文 §3 三层训练信号的"原文"输入 → 由 [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) Phase 1 双盲 OpenIE 提取产出四元组池
- 本文 §6.1 六类样本的"记忆层/推理层/混合层"分配 → 由 [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) §4.4 CCS 公式(`0.5·gap + 0.3·recon + 0.2·bottleneck`)与阈值(memory < 0.3 / reasoning > 0.7)决定
- 探针架构(0.6B 崩溃对照 + 1.7B 主探针)由 08a §3.1 定义,与本文档**解耦**:本文档只描述"用探针做什么",不描述"探针怎么选"

**高级训练范式**:本文档聚焦基础训练设计。**本体涌现**(V1-V3 路线图)、**OpenIE + 聚类**、**信息瓶颈 + 重构损失**、**双系统架构**等高级范式详见 [`05-schema-emergence.md`](05-schema-emergence.md)。

**评估方法**:训练数据的筛选与质量评估详见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) 的 Probe Model + Golden Filter。

**稀疏性惩罚**:为防止"噪音淹没信号",V1.0 训练需引入 L1 + 提取数量上限。详见 [`05-schema-emergence.md`](05-schema-emergence.md) §5。

---

## 1. 训练的核心难点

```
prefill 阶段:模型处理原文 → KV 缓存形成(此时不知道未来会被问什么)
decode 阶段:各种查询到来 → 从 KV 缓存中提取信息

矛盾:编码时不知道查询内容,但必须编码足够信息以应对任意合法查询
```

**训练的核心目标**:让模型在 prefill 阶段形成一种**"通用可查询"的 KV 缓存编码方式**。

---

## 2. 三层训练信号设计

### 2.1 为什么单一任务不够

| 训练方式 | 问题 |
|---|---|
| 只做 QA 训练 | 模型只学会回答"训练时见过的问题",对未见过的查询无法正确提取 |
| 只做 Wiki 生成 | 模型学会"输出结构化信息",但不保证 KV 缓存中信息可被任意查询路径访问 |
| 只做复述 | 压缩率为零,且不能保证推理关系被显式编码 |

### 2.2 三层训练信号的关系

```
┌─────────────────────────────────────────────────────────────┐
│  第一层:编码完备性(Wiki 生成)                              │
│  "你能不能把记住的所有信息都列出来?"                       │
│  → 训练模型在 prefill 时不遗漏任何推理相关信息              │
├─────────────────────────────────────────────────────────────┤
│  第二层:查询鲁棒性(多路径问答)                            │
│  "不管我怎么问,你都能找到那个信息吗?"                     │
│  → 训练模型让同一信息可以从多种查询路径被提取               │
├─────────────────────────────────────────────────────────────┤
│  第三层:关系完整性(推理链问答)                            │
│  "你能不能把信息之间的关系也回忆出来?"                     │
│  → 训练模型编码因果、对比、条件等关系,而非孤立事实         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 第一层:结构化枚举训练(编码完备性)

### 3.1 目的

迫使模型在处理文本时,对**每一个推理相关信息维度**都进行编码。如果模型"跳过"了某个信息,它就无法在 Wiki 中输出该信息,损失函数会惩罚它。

### 3.2 训练样本格式

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "<原始文本,2000-8000 tokens>"
    },
    {
      "role": "assistant",
      "content": "<完整 Wiki 结构化页面,覆盖所有推理相关信息>"
    }
  ]
}
```

### 3.3 Wiki 必须覆盖的维度清单

教师模型生成 Wiki 时,必须确保以下**每个维度**都被覆盖(如果原文中存在):

```
□ 所有实体的名称、类型、别名
□ 所有精确数值 + 单位 + 限定词
□ 所有实体间的因果关系(含方向)
□ 所有实体间的对比关系(含维度)
□ 所有条件限定(前提、范围、例外)
□ 所有时间信息(时间点、区间、顺序)
□ 所有否定信息(不适用、排除、不成立)
□ 所有不确定性标记(可能、预计、约)
□ 所有来源归属(谁说的、哪里提到的)
□ 所有推理链(前提→推导→结论)
```

完整 Wiki 结构定义见 [`01-memory-model.md`](01-memory-model.md) §5-6。

### 3.4 为什么这能改善 KV 缓存编码

```
普通语言模型处理文本时:
  注意力集中在"预测下一个 token"上
  → KV 缓存中的信息分布由"预测需求"决定
  → 修辞性词汇("值得一提")和关键数值获得相似的编码权重

Wiki 生成训练后:
  模型必须在 prefill 后"输出所有关键信息"
  → 反向传播迫使 prefill 阶段的注意力覆盖所有关键信息
  → KV 缓存中的信息分布由"回忆需求"决定
  → 关键数值、关系、条件获得更强的编码
```

---

## 4. 第二层:多路径查询训练(查询鲁棒性)

### 4.1 目的

同一信息可能通过多种不同的查询路径被需要。如果模型只学会了一种"固定问法→固定答法",那么换个问法就提取不出来。

### 4.2 训练样本格式

对**同一篇文本**,生成**多组不同角度的查询**:

```json
{
  "conversations": [
    {"role": "user", "content": "<原始文本>"},
    {"role": "assistant", "content": "已处理。"},
    {"role": "user", "content": "查询 1:文中提到了哪些模型?各自的参数量是多少?"},
    {"role": "assistant", "content": "V4 Flash,总参数 284B,激活参数 13B。"},
    {"role": "user", "content": "查询 2:284B 这个数值属于哪个实体?它代表什么?"},
    {"role": "assistant", "content": "284B 属于 DeepSeek V4 Flash,代表其总参数量。"},
    {"role": "user", "content": "查询 3:文中所有包含 'B' 单位的数值有哪些?分别对应什么?"},
    {"role": "assistant", "content": "284B(总参数)、13B(激活参数)。"},
    {"role": "user", "content": "查询 4:激活参数和总参数之间是什么关系?"},
    {"role": "assistant", "content": "激活参数是总参数的子集,单次推理仅激活 13B/284B ≈ 4.6%。"}
  ]
}
```

### 4.3 查询多样性生成策略

教师模型为每篇文本生成**至少 20-30 个查询**,覆盖以下维度:

| 查询角度 | 示例 | 训练目的 |
|---|---|---|
| **正向查询** | "X 的值是多少?" | 基础精确回忆 |
| **反向查询** | "哪个实体的值是 284B?" | 反向索引能力 |
| **关系查询** | "X 和 Y 之间是什么关系?" | 关系编码 |
| **条件查询** | "这个结论在什么条件下成立?" | 条件边界 |
| **否定查询** | "文中排除了什么?" | 否定信息 |
| **枚举查询** | "列出所有包含数值的句子" | 完备性 |
| **对比查询** | "A 和 B 在维度 D 上谁更好?" | 对比关系 |
| **时序查询** | "X 发生在 Y 之前还是之后?" | 时间顺序 |
| **组合查询** | "满足条件 C 的所有实体的属性是什么?" | 多信息组合 |
| **扰动查询** | 用同义词/换语序问同一件事 | 鲁棒性 |

### 4.4 为什么这能改善查询能力

```
只训练正向查询:
  模型学会 "问 X 的值 → 输出 X 的值"
  → 如果问 "哪个实体的值是 Y" → 可能无法反向检索

训练多路径查询后:
  模型学会同一信息可以从多个方向被"索引"
  → KV 缓存中的信息具有更丰富的"可访问路径"
  → 任意合法查询都能找到对应信息
```

---

## 5. 第三层:推理链驱动训练(关系完整性)

### 5.1 目的

推理不仅需要孤立事实,还需要事实之间的**关系**。如果 KV 缓存中只有"284B"和"13B",但没有"13B 是 284B 的一部分"这个关系,推理就无法进行。

### 5.2 训练样本格式

```json
{
  "conversations": [
    {"role": "user", "content": "<原始文本>"},
    {"role": "assistant", "content": "已处理。"},
    {
      "role": "user",
      "content": "基于已处理的信息,请回答:为什么 V4 Flash 的推理成本低于 GPT-4?请给出完整推理链。"
    },
    {
      "role": "assistant",
      "content": "推理链:\n1. [事实] V4 Flash 采用 MoE 架构\n2. [事实] 总参数 284B,激活参数 13B(比例 4.6%)\n3. [因果] MoE 架构 → 仅激活部分参数 → 实际计算量小\n4. [事实] 定价:输入 $0.14/百万 token,输出 $0.28/百万 token\n5. [对比] GPT-4 定价:输入 $10,输出 $30(每百万 token)\n6. [结论] V4 Flash 推理成本约为 GPT-4 的 1.4%(输入)和 0.93%(输出)\n7. [归因] 成本优势主要来源于 MoE 的稀疏激活"
    }
  ]
}
```

### 5.3 推理链中的每一步必须标注信息来源

```
每一步标注:
  [事实] → 直接来自原文的精确信息
  [因果] → 原文明确表述的因果关系
  [对比] → 原文明确表述的对比
  [推导] → 基于上述事实的逻辑推导(标记为推理层)
  [归因] → 综合判断(标记置信度)
```

### 5.4 为什么这能改善关系编码

```
普通问答训练:
  模型学会 "问什么答什么"
  → KV 缓存中信息是"扁平"的,只有孤立事实

推理链训练后:
  模型必须输出 "事实 1 → 关系 → 事实 2 → 结论"
  → 反向传播迫使 prefill 阶段编码"事实之间的关系"
  → KV 缓存中信息是"图结构"的,有节点也有边
```

---

## 6. 六类训练样本

### 6.1 训练样本全景

| Type | 名称 | 主要作用 | 关联训练信号 |
|---|---|---|---|
| **A** | 基础 Wiki 编译 | 学会结构化输出 | 第一层(编码完备性) |
| **B** | 领域增强 Wiki 编译 | 学会一步领域规则推导 | 第一层 + 五步漏斗 Step 3b |
| **C** | 领域知识独立输出 | 学会术语/规则抽取 | 第一层(支撑 Wiki 完整性) |
| **D** | 完整 Wiki 编译(含全部节) | 训练完整信息覆盖 | 第一层(高难度) |
| **E** | 负样本 + 自知之明训练 | 教模型标注能力边界 | 第一层 + 能力自知原则 |
| **F** | 扰动鲁棒性样本 | 训练同义改写鲁棒性 | 第二层(查询鲁棒性) |
| **J** ★ | 指代消解样本 | 学会解析"它""第一个城市"等指代 | 多轮对话扩展 |
| **K** ★ | 信息更新追踪样本 | 学会识别被后轮修正的信息 | 多轮对话扩展 |
| **L** ★ | 约束累积样本 | 学会汇总跨轮的所有约束 | 多轮对话扩展 |
| **M** ★ | 意图演变追踪样本 | 学会追踪用户意图的变化轨迹 | 多轮对话扩展 |
| **N** ★ | 增量索引更新样本 | 学会增量更新索引层而非全量重建 | 多轮对话扩展 |

> **注**:Type J-N 是多轮对话输入的特化样本,完整定义见 [`04-dialogue-extension.md`](04-dialogue-extension.md) §6。
> 多轮对话的轮次分块、对话索引层、MQP v3 协议扩展均在该文档中定义。

### 6.2 Type A:基础 Wiki 编译

```json
{
  "type": "basic_wiki_compilation",
  "input": {"text": "原始文本", "schema_version": "1.0"},
  "output": {
    "wiki_page": {
      "title": "...",
      "basic_info": {"type": "...", "domain": ["..."]},
      "core_facts": [
        {"attribute": "总参数", "value": "284B", "confidence": 0.95, "evidence": "..."}
      ]
    }
  }
}
```

### 6.3 Type B:领域增强 Wiki 编译(对应"领域辅助层")

```json
{
  "type": "domain_augmented_wiki",
  "input": {
    "text": "该公司 LGD 为 35%",
    "domain_context": {"LGD": "违约损失率,等于 1 减去回收率"}
  },
  "output": {
    "wiki_page": {
      "core_facts": [
        {"attribute": "LGD", "value": "35%"},
        {"attribute": "回收率", "value": "65%",
         "domain_augmented": true,
         "inference_chain": "LGD = 1 - 回收率 → 回收率 = 1 - 0.35 = 0.65"}
      ]
    }
  }
}
```

### 6.4 Type D:完整 Wiki 编译(含全部节)

```json
{
  "type": "full_wiki_compilation",
  "input": "文本内容",
  "output": {
    "wiki_page": {
      "title": "...",
      "basic_info": {...},
      "core_facts": [...],
      "relations": {"causal_dependency": [...], "comparison": [...], "temporal_sequence": [...]},
      "reasoning_chains": [{
        "id": "RC_001",
        "premise": ["..."],
        "conclusion": "...",
        "explicit_in_text": false,
        "needs_reasoning_model_verification": true
      }],
      "context_annotations": {"evaluative": [...], "conditional": [...]},
      "domain_knowledge": {...},
      "sources": [...],
      "completeness_metadata": {
        "irr_estimate": 0.91,
        "flagged_for_reasoning_model": ["RC_001"]
      }
    }
  }
}
```

### 6.5 Type E:负样本 + 自知之明训练

**E-1:条件性假设不作为事实提取**

```json
{
  "type": "negative_sample",
  "subtype": "conditional_hypothesis",
  "input": {"text": "如果央行继续降息,房地产市场可能会回暖。"},
  "output": {
    "wiki_page": {
      "core_facts": [],
      "context_annotations": {
        "conditional": [{
          "condition": "央行继续降息",
          "consequence": "房地产市场回暖",
          "certainty": "possible",
          "evidence": "如果...可能会..."
        }]
      }
    }
  }
}
```

**E-2:教模型标注 `needs_reasoning_model`**

```json
{
  "type": "self_awareness_training",
  "input": {"text": "公司 Q3 营收同比下降 15%,同时研发支出增加 20%。"},
  "output": {
    "wiki_page": {
      "core_facts": [
        {"attribute": "Q3 营收同比变化", "value": "-15%", "confidence": 0.95},
        {"attribute": "研发支出同比变化", "value": "+20%", "confidence": 0.95}
      ],
      "reasoning_chains": [{
        "id": "RC_001",
        "premise": ["Q3 营收-15%", "研发支出+20%"],
        "conclusion": "管理层选择牺牲短期利润换取长期竞争力",
        "explicit_in_text": false,
        "confidence": 0.75,
        "needs_reasoning_model_verification": true
      }],
      "completeness_metadata": {
        "irr_estimate": 0.88,
        "flagged_for_reasoning_model": ["RC_001"]
      }
    }
  }
}
```

**E-3:否定事实正确处理**

```json
{
  "type": "negative_sample",
  "subtype": "negation_handling",
  "input": {"text": "张三没有入职 DeepSeek,而是加入了竞争对手。"},
  "output": {
    "wiki_page": {
      "core_facts": [{
        "attribute": "入职状态",
        "value": "未入职",
        "target": "DeepSeek",
        "negated": true,
        "evidence": "张三没有入职 DeepSeek"
      }]
    }
  }
}
```

### 6.6 Type F:扰动鲁棒性样本

```json
{
  "type": "robustness_perturbation",
  "perturbation_type": "synonym_substitution",
  "input_group": [
    {"text": "张三入职了 DeepSeek", "expected": [{"subject": "张三", "relation": "入职", "object": "DeepSeek"}]},
    {"text": "张三加入了 DeepSeek", "expected": "同上"},
    {"text": "DeepSeek 录用了张三", "expected": "同上"},
    {"text": "张三成为 DeepSeek 的员工", "expected": "同上"}
  ],
  "robustness_requirement": "所有变体生成的 Wiki 页面必须语义一致"
}
```

---

## 7. 四阶段训练课程

### 7.1 总体安排

```
┌─────────────────────────────────────────────────────────────┐
│  阶段 1(Epoch 1-2):基础编码                                │
│                                                              │
│  任务配比:                                                  │
│    Wiki 生成 50% + 正向查询 30% + 枚举查询 20%             │
│                                                              │
│  文本长度:512-2048 tokens                                  │
│  查询数量:每篇文本 5-10 个查询                            │
│                                                              │
│  目标:                                                     │
│    模型学会在 prefill 时"不遗漏"推理相关信息               │
│    模型学会输出精确数值(不模糊化)                         │
├─────────────────────────────────────────────────────────────┤
│  阶段 2(Epoch 3-4):多路径查询                              │
│                                                              │
│  任务配比:                                                  │
│    Wiki 生成 20% + 多路径查询 50% + 推理链 20% + 否定 10% │
│                                                              │
│  文本长度:2048-4096 tokens                                 │
│  查询数量:每篇文本 15-20 个查询                           │
│  查询类型:正向 + 反向 + 关系 + 组合                       │
│                                                              │
│  目标:                                                     │
│    同一信息可从多种查询路径被提取                           │
│    关系信息(因果/对比/条件)被正确编码                     │
├─────────────────────────────────────────────────────────────┤
│  阶段 3(Epoch 5-6):推理链 + 极限压力                      │
│                                                              │
│  任务配比:                                                  │
│    推理链问答 40% + 多路径查询 30% + 对抗性查询 20%       │
│    + 完备性枚举 10%                                         │
│                                                              │
│  文本长度:4096-16384 tokens                                │
│  查询数量:每篇文本 25-30 个查询                           │
│  连续查询轮数:15-20 轮                                    │
│                                                              │
│  目标:                                                     │
│    长上下文中信息不衰减                                     │
│    多轮连续查询后精度不下降                                 │
│    对抗性查询(易混淆数值/否定/未提及)正确处理             │
├─────────────────────────────────────────────────────────────┤
│  阶段 4(Epoch 7):精调 + 边界对齐                          │
│                                                              │
│  任务配比:                                                  │
│    IRR 验证样本 50% + 推理链 30% + 自知之明 20%           │
│                                                              │
│  目标:                                                     │
│    irr_estimate 校准                                       │
│    needs_reasoning_model 标注准确                         │
│    边界样本(irr < 0.90)精调                               │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 各阶段查询类型分布

| 查询类型 | 阶段 1 | 阶段 2 | 阶段 3 | 阶段 4 |
|---|---|---|---|---|
| 实体查询 | 15% | 10% | 5% | 5% |
| 属性正向查询 | 30% | 15% | 10% | 10% |
| 属性反向查询 | 0% | 15% | 10% | 10% |
| 因果关系查询 | 0% | 15% | 15% | 15% |
| 对比关系查询 | 0% | 10% | 10% | 10% |
| 条件边界查询 | 0% | 10% | 10% | 10% |
| 时序查询 | 5% | 5% | 5% | 5% |
| 否定查询 | 5% | 5% | 10% | 10% |
| 推理链查询 | 0% | 5% | 20% | 20% |
| 对抗性/边界查询 | 0% | 5% | 5% | 5% |
| 完备性枚举 | 45% | 5% | 0% | 0% |

### 7.3 三级质量 × 三级难度矩阵

|  | Gold(双教师验证) | Silver(单教师高置信) | Bronze(单教师中置信) |
|---|---|---|---|
| **Easy(CCS<0.2)** | 阶段 1 主力 | 阶段 1 补充 | — |
| **Medium(0.2≤CCS<0.3)** | 阶段 2 主力 | 阶段 2 补充 | 阶段 2 辅助 |
| **Hard(0.3≤CCS<0.4)** | 阶段 3 主力 | 阶段 3 补充 | 阶段 3 辅助 |

Gold/Silver/Bronze 分级标准见 [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) §6.2。

---

## 8. 损失函数设计

### 8.1 复合损失函数

```python
def memory_training_loss(predictions, labels, query_type, token_types):
    """记忆模型训练损失"""
    
    # 基础交叉熵
    base_loss = cross_entropy(predictions, labels)
    
    # 关键信息加权(数值、实体名、关系词、条件词)
    key_weight = torch.ones_like(labels)
    key_weight[token_types == "numeric"] = 5.0           # 数值 5 倍
    key_weight[token_types == "entity_name"] = 3.0       # 实体名 3 倍
    key_weight[token_types == "relation_word"] = 3.0     # 关系词 3 倍
    key_weight[token_types == "condition_word"] = 4.0    # 条件词 4 倍
    key_weight[token_types == "negation_word"] = 4.0     # 否定词 4 倍
    key_weight[token_types == "time_word"] = 3.0         # 时间词 3 倍
    
    weighted_loss = base_loss * key_weight
    
    # 查询类型额外加权
    if query_type == "absence_query":
        # "未提及"类查询:惩罚编造行为
        weighted_loss *= 1.5
    elif query_type == "adversarial":
        # 对抗性查询:高权重
        weighted_loss *= 2.0
    elif query_type == "reasoning_chain":
        # 推理链:中间步骤也重要
        weighted_loss *= 1.5
    
    return weighted_loss.mean()
```

### 8.2 为什么数值要 5 倍权重

```
普通训练:
  模型可能把 "284B" 编码为 "大约 300B 左右"
  → 因为 "284" 和 "300" 在语义空间中距离不远
  → 交叉熵损失对 "284" vs "300" 的惩罚不够大

5 倍权重后:
  模型必须精确编码 "284" 而非 "300"
  → 因为数值 token 的损失被放大 5 倍
  → 模型被迫在 KV 缓存中给数值信息更高的编码精度
```

### 8.3 Wiki 编译复合损失(训练 Wiki 输出时使用)

```python
def wiki_compilation_loss(student_wiki, teacher_wiki, schema, task_type):
    """
    Wiki 编译复合损失(对应 Type A/B/D 训练)
    """
    
    structure_loss = wiki_structure_loss(student_wiki, schema)              # 5%
    fact_loss = graph_edit_distance_loss(                                     # 30%
        student_wiki.core_facts, teacher_wiki.core_facts,
        numeric_weight=5.0, key_info_weight=3.0
    )
    relation_loss = relation_accuracy_loss(                                   # 15%
        student_wiki.relations, teacher_wiki.relations
    )
    reasoning_loss = reasoning_chain_loss(                                   # 10%
        student_wiki.reasoning_chains, teacher_wiki.reasoning_chains
    )
    context_loss = context_annotation_loss(                                  # 10%
        student_wiki.context_annotations, teacher_wiki.context_annotations
    )
    completeness_loss = 1.0 - compute_irr(student_wiki, teacher_wiki)         # 15%
    schema_loss = schema_constraint_loss(student_wiki, schema)                # 10%
    self_awareness_loss = self_awareness_loss(                               # 5%
        student_wiki.completeness_metadata, teacher_wiki.completeness_metadata
    )
    
    total = (
        0.05 * structure_loss + 0.30 * fact_loss + 0.15 * relation_loss +
        0.10 * reasoning_loss + 0.10 * context_loss + 0.15 * completeness_loss +
        0.10 * schema_loss + 0.05 * self_awareness_loss
    )
    
    return total
```

---

## 9. 训练数据规模与配比

### 9.1 数据规模估算

| 数据子集 | 规模 | 用途 |
|---|---|---|
| 主数据(Wiki 编译 + 多路径查询) | 300K-500K 样本 | 核心训练 |
| 多实体绑定数据 | 50K-80K 样本 | 防止实体混淆 |
| 选择性遗忘数据 | 50K-80K 样本 | 防止连带遗忘 |
| 干扰项数据 | 100K 样本 | 抗干扰鲁棒性 |
| 记忆后对话数据 | 200K 样本 | 防止记忆泄露 |
| **总计** | **约 700K-1M 样本** | |

### 9.2 数据来源

| 来源类型 | 示例 | 用途 |
|---|---|---|
| 公开基准 | MMLU、GSM8K、TruthfulQA | 覆盖推理密集型任务 |
| 长文档 | 财报、论文、法律文本、技术文档 | 测试长上下文记忆 |
| 对话数据 | 多轮对话、客服记录 | 测试跨轮次信息保持 |
| 合成数据 | 教师模型生成的事实密集型文本 | 控制信息密度和推理类型 |
| 对抗性数据 | 易混淆数值/否定条件/时间限定 | 训练精度极限 |

### 9.3 语料处理流水线(摘要,详见 `agenticmemory_training/`)

```
STAGE 0: 语料准备(去重/清洗/分块/PII 扫描)
    ↓
STAGE 1: 教师蒸馏提取(OpenIE + Wiki 编译 + 完备查询集)
    ↓
STAGE 2: Capacity Gap 自动分层 + 五步能力边界判定
    ↓
STAGE 3: Wiki 编译 + 七层信息完整性审计 + Wiki DAG 构建
    ↓
STAGE 4: Schema 自动涌现
    ↓
STAGE 5: 训练集构建(六类样本 × 三级难度 × 三级质量)
    ↓
STAGE 6: 记忆模型训练(四阶段课程)
    ↓
STAGE 7: 多维评估(双轨验证 + IRR)
```

---

## 10. 关键设计决策总结

| 决策点 | 选择 | 理由 |
|---|---|---|
| **目标函数** | 推理无损(非逐字复述) | 以推理需要的信息准确度为准,允许压缩推理无关信息 |
| **交付物统一** | Wiki 编译能力 = 记忆能力 | 记忆不是"存储"而是"编码",Wiki 是编码的完整体现 |
| **训练数据** | 教师蒸馏(非人工标注) | 保证质量的同时控制成本,700K+ 规模 |
| **能力边界** | Capacity Gap 自动分层 + 五步漏斗 | 不依赖人工规则,让边界从模型行为中自然涌现 |
| **训练信号** | 三层联合(Wiki + 多路径 + 推理链) | 互补训练形成"通用可查询"的 KV 编码 |
| **损失加权** | 数值 5×、条件词 4×、否定词 4× | 迫使模型对推理关键信息给予更高编码精度 |
| **自知之明** | `needs_reasoning_model` 标记 + `irr_estimate` | 模型对自身能力边界有元认知 |
| **基座模型** | Qwen3-0.6B(默认)/ 1.7B(可选) | 对齐 AGENTS.md §12 <1B 目标 |

---

## 11. 与现有文档的关系

| 内容 | 本文位置 | 详见 |
|---|---|---|
| **训练原理与课程** | 本文 §1-§10 | — |
| **数据合成 SOP** | 训练侧 | [`../agenticmemory_training/08-memory-distillation-pipeline.md`](../agenticmemory_training/08-memory-distillation-pipeline.md) |
| **Capacity Gap 设计** | 训练侧 | [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) |
| **Schema 融合边界** | 训练侧 | [`../agenticmemory_training/08b-seed-schema-fusion.md`](../agenticmemory_training/08b-seed-schema-fusion.md) |
| **P1 最小闭环** | 训练侧 | [`../agenticmemory_training/08c-p1-minimum-loop.md`](../agenticmemory_training/08c-p1-minimum-loop.md) |
| **核心能力契约** | 见 [`01-memory-model.md`](01-memory-model.md) | — |
| **评估方法** | 见 [`03-evaluation.md`](03-evaluation.md) | — |

---

## 12. 待解决问题(与训练相关)

| # | 问题 | 状态 | 建议决策时机 |
|---|---|---|---|
| O1 | Wiki DAG 构建的具体算法 | 🔴 待解决 | P1 启动前必须 |
| O2 | LoRA 探针的评估指标 | 🔴 待解决 | P1 启动前必须 |
| **O10** | **多教师交叉验证的具体协议**(谁交叉?几轮?阈值?) | 🟡 待讨论 | P1 训练启动前 |
| **O11** | **Schema 融合边界的训练数据转换规则**(13 字段 ↔ 涌现 schema 的映射) | 🟡 待讨论 | Phase 1 启动前 |
| **O12** | **训练数据中"未提及"类负样本的占比**(当前 10% 是否足够?) | 🟢 训练中验证 | P1 完成后 |

---

## 13. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-25 | 初始版本:三层训练信号原理 + 六类样本 + 四阶段课程 + 损失函数 | Sisyphus(AI 助手)+ 用户 |

---

**文档版本**: v0.1
**Owner**: AgenticMind 训练组
**下一步**: 待 O1/O2/O10 决策后启动 P1 训练
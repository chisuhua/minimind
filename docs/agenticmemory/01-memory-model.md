# 01 · 记忆模型核心设计 — 推理无损的记忆基底

> **文档 ID**: MEM-001-MEMORY-MODEL
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.1
> **配套文档**:
> - 架构总览: [`README.md`](README.md)
> - 训练设计: [`02-training-design.md`](02-training-design.md)
> - 评估框架: [`03-evaluation.md`](03-evaluation.md)
> - Wiki DAG 构建实现: [`../agenticmemory_training/08d-wiki-dag-construction.md`](../agenticmemory_training/08d-wiki-dag-construction.md)(v0.1, 2026-08-26 新建,与本文档 §5-6 完全对齐;承接 08a Phase 1 的 OpenIE 四元组池)
> - Schema 涌现(与 Wiki DAG 平行): [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) Phase 3
> - Schema 融合边界: [`../agenticmemory_training/08b-seed-schema-fusion.md`](../agenticmemory_training/08b-seed-schema-fusion.md)

---

## 0. 文档范围与定位

本文档定义 **agenticmemory 记忆模型** 的核心能力——即它**必须做到什么**才能成为合格的 KV 缓存记忆基底。

**与 README 的关系**:README 是"如何分层 + 如何被消费"的架构图;本文档是"模型本身的核心能力定义 + 能力边界判定 + Wiki DAG 训练证明"的契约。

**多轮对话输入的支持**:本文档聚焦于核心能力契约。轮次分块、对话索引、MQP v3 新增查询等对话特化设计见 [`04-dialogue-extension.md`](04-dialogue-extension.md)。

---

## 1. 一句话定义

> **记忆模型是一个经过专门训练的自回归语言模型,其核心能力是:将任意输入文本中"推理所需的全部信息"精确编码到 KV 缓存中,使下游推理模型仅凭 KV 缓存即可达到与基于原文推理无显著差异的准确率(B/A ≥ 0.98)。**

### 1.1 三条核心原则

| 原则 | 含义 |
|---|---|
| **推理无损** | 不追求逐字复述,只追求推理所需信息的完整性和精确性 |
| **能力自知** | 模型必须知道自己记住了什么、没记住什么,并显式标注边界 |
| **压缩有效** | 丢弃修辞/连接词/重复表述等推理无关信息,保留因果关系/数值/条件等推理硬依赖信息 |

### 1.2 两个交付物实为一个

| 交付物 | 本质 |
|---|---|
| **记忆模型** | 知识编译器——将非结构化文本编码为结构化知识表示 |
| **Wiki 结构化页面** | 编译产物——记忆能力的完整体现,**不是独立交付物** |

> **记忆 = 将非结构化信息编码为结构化知识表示的能力。** 如果模型能生成信息无损的 Wiki 页面,就证明它完成了完整的编码过程。

---

## 1.5 边界界定(必须严格遵守)

记忆模型是一个**纯"信息编译"组件**,被对话管理器、推理模型等上层模块调用,自身不关心"对话应该怎么进行"。

### 它做的事

```
输入:一段上下文(文档 / 对话 / 任何文本)
输出:结构化的、可完备查询的信息表示(KV 缓存 + Wiki DAG)
目标:使推理模型通过查询即可获取推理所需的全部信息
```

### 它不做的事

```
✗ 对话管理(决定什么时候回复用户)
✗ 用户偏好记忆("用户喜欢简洁的回答")
✗ 技能复用("上次用了什么方法")
✗ 任务规划("下一步应该做什么")
✗ 人格/身份维持("我是谁")
```

### 为什么必须严格限定

```
问题 1:目标函数模糊
  "推理无损"是清晰的——信息提取准确率 ≥ 98%
  "用户满意度"是模糊的——什么算"好的对话体验"?
  两个目标混在一起,训练信号互相干扰

问题 2:训练数据污染
  记忆模型的训练数据应该是"信息完备性"导向的
  如果混入"对话流畅性"数据,模型可能学会"流畅但信息不完备"

问题 3:评估标准冲突
  记忆模型用"双轨准确率比 ≥ 0.98"评估
  对话管理用"用户满意度"评估
  两个标准无法统一
```

**对话特化边界**:记忆模型可以支持多轮对话输入(详见 [`04-dialogue-extension.md`](04-dialogue-extension.md)),但仅支持"从多轮对话中提取信息"的职责,不涉及对话管理本身。

---

## 2. 记忆 vs 推理:能力边界

### 2.1 精确定义

| 维度 | 记忆(System 1) | 推理(System 2) |
|---|---|---|
| **计算复杂度** | 低——一次前向传播可完成 | 高——需要多步信息整合 |
| **文本可验证性** | 证据在原文中可直接定位 | 需要原文之外的知识或推导 |
| **上下文范围** | 局部(≤2-3 句) | 全局(跨段落/跨文档) |
| **转换步数** | 0-1 步映射 | ≥2 步组合推导 |

### 2.2 记忆模型需要的语言理解能力(完整清单)

| 能力层 | 具体能力 | 示例 |
|---|---|---|
| **词汇语义** | 同义词识别、上下位关系、词义消歧、隐喻理解 | "入职" = "加入" = "就职" |
| **句法结构** | 主谓宾识别、被动/主动转换、修饰语附着、从句解析 | "X 被 Y 收购" → [Y, 收购, X] |
| **指代消解** | 代词消解、名词短语消解、零指代消解 | "该公司" → "DeepSeek" |
| **基础量化** | 数值提取、单位换算、比较级解析、百分比理解 | "50 万" = "500,000" |
| **时间表达** | 绝对时间、相对时间、时间范围 | "去年 Q3" → "2025 年 Q3" |
| **否定与模态** | 否定识别、可能性/确定性标记、条件句识别 | "没有采用" → negated=true |

### 2.3 记忆模型**不需要**的能力(交给推理模型)

| 推理类型 | 示例 | 处理方式 |
|---|---|---|
| 传递性推理 | A>B, B>C → A>C | 交给推理模型 |
| 三段论 | 所有 A 是 B,X 是 A → X 是 B | 交给推理模型 |
| 跨段整合 | 需要跨段落信息组合 | 交给推理模型 |
| 隐式因果推断 | 需要排除混杂因素 | 交给推理模型 |
| 反事实推理 | 需要违反事实假设 | 交给推理模型 |
| 默认值填充 | 需要常识假设 | 不提取,标注 implicit |
| 抽象概括 | 需要全局归纳 | 交给推理模型 |

---

## 3. 五步漏斗式能力边界判定

### 3.1 判定流程

```
待提取信息 I
    │
    ▼
Step 1: 文本显式性检查
  ├─ explicit(文本直接陈述)→ Step 2
  ├─ implicit_with_evidence(有片段但需一步理解)→ Step 3b
  └─ no_evidence(无相关片段)→ ❌ 推理层(CCS=0.9)
    │
    ▼
Step 2: 转换步数检查
  ├─ 0 步直接映射 → Step 3a
  ├─ 1 步转换 → Step 3b
  └─ ≥2 步转换 → ❌ 推理层(CCS=0.8)
    │
    ▼
Step 3a: 语言理解能力检查
  ├─ 所需能力在六类清单内 → ✅ 记忆层(CCS=0.1)
  └─ 超出清单 → ❌ 推理层(CCS=0.85)

Step 3b: 领域规则检查
  ├─ 单条确定性规则 → ✅ 领域辅助记忆层(CCS=0.25)
  ├─ 非确定性规则 → ❌ 推理层(CCS=0.7)
  └─ 多条规则组合 → ❌ 推理层(CCS=0.8)
```

### 3.2 最终分层

| 层 | CCS 范围 | 含义 | 处理方式 |
|---|---|---|---|
| **记忆层** | < 0.3 | 低复杂度事实,小模型可独立提取 | 直接作为训练目标 |
| **领域辅助层** | 0.25-0.35 | 注入领域知识后可提取 | 构建领域增强训练样本 |
| **混合层** | 0.3-0.7 | 不确定边界 | 人工审核队列 |
| **推理层** | > 0.7 | 高复杂度知识 | 交给推理模型处理 |

### 3.3 与 Capacity Gap 的关系

五步漏斗是 **per-information 的判定**(每条信息单独判定);Capacity Gap 是 **per-corpus 的判定**(整批语料的分层比例)。两者互补:
- **五步漏斗**→ 决定哪些信息进训练集(精细)
- **Capacity Gap**→ 决定训练目标的容量边界(粗粒度)

详细 Capacity Gap 流程见 [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) §3-4。

---

## 4. 必须保留 vs 可以丢弃

### 4.1 必须精确保留的信息(推理硬依赖)

```
├─ 精确数值 + 单位(284B、13B、$0.14/百万 token)
├─ 条件限定("在正常情况下"、"肝功能不全患者中")
├─ 因果关系("X 导致 Y"、"X 使 Y 成为可能")
├─ 对比关系("A 比 B 高 15%")
├─ 时序关系("X 在 Y 之前"、"X 持续到 Y")
├─ 否定信息("不适用于..."、"排除了...")
├─ 实体属性绑定("X 的营收是 A",不能张冠李戴)
├─ 时间边界("2024 年 Q3"、"截至 2025 年底")
└─ 量化限定("约"、"超过"、"不超过"、"正常情况下")
```

### 4.2 可以压缩或丢弃的信息(推理无依赖)

```
├─ 修辞表达("值得一提的是"、"令人印象深刻的是")
├─ 连接词和过渡句("接下来"、"如上所述")
├─ 重复表述(同一事实的不同措辞重复出现)
├─ 格式装饰(标题层级、列表符号、段落分隔)
├─ 语气标记("显然"、"毫无疑问")
└─ 背景铺垫("随着人工智能的快速发展"——除非包含关键前提)
```

### 4.3 灰色地带处理规则

| 信息类型 | 保留条件 | 丢弃条件 |
|---|---|---|
| 评价性表述("表现优异") | 推理需要判断作者态度时 | 推理只关心事实时 |
| 类比比喻("像高铁一样快") | 推理需要理解类比含义时 | 推理不关心修辞时 |
| 不确定性标记("可能"、"预计") | 推理涉及概率判断时 | 推理只关心确定性事实时 |

### 4.4 一个具体例子

```
原文(200 tokens):
  "DeepSeek 于 2026 年 1 月正式发布了新一代大语言模型 V4 Flash。
   值得一提的是,该模型采用了先进的混合专家架构(MoE),
   总参数量达到了令人印象深刻的 284B。然而,得益于其精妙的
   路由机制,单次推理仅激活 13B 参数,这意味着激活比例仅为
   总参数的约 4.6%。正如 DeepSeek 团队在发布会上强调的,
   这一设计大幅降低了推理成本。API 定价为输入 $0.14/百万 token,
   输出 $0.28/百万 token,远低于 GPT-4 的 $10/$30。
   在多项基准测试中,V4 Flash 表现优异,特别是在代码生成
   和数学推理任务上。不过,在长文本理解任务上,其表现
   略逊于 GPT-4,这可能是由于激活参数较少导致的。"

保留的信息(约 80 tokens 等效信息量):
  ├─ 实体:DeepSeek V4 Flash
  ├─ 时间:2026 年 1 月发布
  ├─ 架构:MoE
  ├─ 数值:总参数 284B,激活参数 13B,激活比例 4.6%
  ├─ 因果:MoE → 低激活比例 → 低成本推理
  ├─ 定价:输入 $0.14/百万 token,输出 $0.28/百万 token
  ├─ 对比:GPT-4 定价 $10/$30
  ├─ 优势:代码生成、数学推理表现优异
  ├─ 劣势:长文本理解略逊于 GPT-4
  ├─ 因果:激活参数少 → 长文本理解较弱
  └─ 来源:DeepSeek 团队发布会

丢弃的信息:
  ✗ "值得一提的是"
  ✗ "令人印象深刻的"
  ✗ "精妙的路由机制"
  ✗ "正如...强调的"
  ✗ "正如...表现优异"(修辞性过渡)
  ✗ "不过"(连接词,因果关系已保留)

压缩率:约 60%(200 tokens → 80 tokens 等效信息量)
推理准确率:与基于原文回答无显著差异(目标 B/A ≥ 0.98)
```

---

## 5. Wiki DAG 作为训练时证明

### 5.1 Wiki DAG 的角色

**Wiki DAG ≠ 运行时 consumer 需要的数据结构**
**Wiki DAG = 训练阶段的"完整性证明"和"评估基准"**

```
训练阶段:
  原始文本 → OpenIE 提取 → 四元组池 → Wiki DAG(ground truth)
                                    ↓
                          训练 base model 学习:
                          "给定文本,prefill 后的 KV 应该包含
                           Wiki DAG 中的所有信息"
                                    ↓
                          训练完成后验证:
                          用探针查询 KV → 能否重建 Wiki DAG?
                          - 能完整重建 → KV 编码完整 ✓
                          - 部分缺失 → 训练不足,需补充数据 ✗
```

### 5.2 Wiki DAG 的两层结构

```
┌─────────────────────────────────────────────┐
│  页面结构层(固定骨架,所有页面共享)         │
│  basic_info / core_facts / relations /       │
│  reasoning_chains / context_annotations /    │
│  domain_knowledge / sources /                │
│  completeness_metadata                       │
├─────────────────────────────────────────────┤
│  内容 Schema 层(动态,按领域/类型变化)       │
│  每个节内的具体字段和约束随 Schema 变化       │
│  通过 schema_ref 字段标识归属               │
└─────────────────────────────────────────────┘
```

### 5.3 Wiki DAG 的八核心字段(顶层结构)

| 字段 | 作用 | 详情见 |
|---|---|---|
| `basic_info` | 实体基础信息(类型、领域、别名) | §6.1 |
| `core_facts` | 核心事实(数值、属性、条件、否定) | §6.2 |
| `relations` | 实体间关系(因果、对比、时序、层级) | §6.3 |
| `reasoning_chains` | 推理链(前提→推导→结论,显式标注 needs_reasoning_model) | §6.4 |
| `context_annotations` | 语境标注(评价、条件、不确定性) | §6.5 |
| `domain_knowledge` | 领域知识(术语、规则、背景) | §6.6 |
| `sources` | 来源引用(ref, span, confidence) | §6.7 |
| `completeness_metadata` | 完整性自报告(irr_estimate, flagged_for_reasoning_model) | §6.8 |

**完整 JSON Schema** 见 [`../agenticmemory_training/08d-wiki-dag-construction.md`](../agenticmemory_training/08d-wiki-dag-construction.md) §2 与本文档 §6(08d 与本文档 §6 完全对齐,08d 是镜像 + 构建算法骨架)。

### 5.4 Wiki DAG 的三种验证用途

| 用途 | 方法 | 通过标准 |
|---|---|---|
| **KV 完整性验证** | 用探针从 KV 中提取所有实体和关系,与 Wiki DAG 对比 | 实体召回 ≥ 95%,关系召回 ≥ 90% |
| **LoRA 探针训练目标** | LoRA 学习"给定 probe query,从 KV 中提取 Wiki DAG 子图" | 子图匹配 F1 ≥ 0.85 |
| **Schema 演化检测** | 新语料 prefill 后,探针提取结果与旧 Wiki DAG 对比 | 新增实体/关系比例 < 10% 为稳定 |

---

## 6. Wiki DAG 核心字段详细定义

### 6.1 `basic_info`

```json
{
  "title": "string (required)",
  "page_type": "entity | concept | comparison | event | process (required)",
  "type": "string (required, 来自 Schema entity_types)",
  "domain": ["string"],
  "temporal_scope": "string",
  "aliases": ["string"],
  "schemas_applied": ["schema_id"]
}
```

### 6.2 `core_facts`

```json
{
  "attribute": "string (required)",
  "value": "string | number | boolean | object (required)",
  "condition": "string | null",
  "confidence": "number [0,1]",
  "evidence": "string (原文证据片段)",
  "temporal": {"time": "string", "type": "year|quarter|month|date"},
  "negated": "boolean",
  "schema_ref": "string (Schema ID)"
}
```

### 6.3 `relations`

四类关系:

| 子字段 | 描述 | 示例 |
|---|---|---|
| `causal_dependency` | 因果依赖(causes/enables/depends_on/prevents) | "X 导致 Y" |
| `comparison` | 对比关系(entity_a / entity_b / dimension / comparison_result) | "A 比 B 高 15%" |
| `temporal_sequence` | 时序(events / order: sequential/parallel/overlapping) | "X 发生在 Y 之前" |
| `hierarchical` | 层级关系(is_a / part_of / has_part) | "公司 is-a 组织" |

### 6.4 `reasoning_chains`

```json
{
  "id": "string (required)",
  "description": "string",
  "premise": ["string (required)"],
  "inference": "string",
  "conclusion": "string (required)",
  "evidence": "string",
  "explicit_in_text": "boolean",
  "confidence": "number",
  "needs_reasoning_model_verification": "boolean"
}
```

**关键设计**:即使推理链写在 Wiki 里,`needs_reasoning_model_verification=true` 表示该推理链**需要推理模型验证**,记忆模型只负责"记住推理链存在",**不保证推理结论正确**。这是"能力自知"原则的体现。

### 6.5 `context_annotations`

三类标注:

```json
{
  "evaluative": [{  // 评价性
    "expression": "string",
    "sentiment": "positive | negative | neutral | mixed",
    "intensity": "high | medium | low",
    "target": "string"
  }],
  "conditional": [{  // 条件性
    "condition": "string",
    "consequence": "string",
    "certainty": "definite | probable | possible | speculative"
  }],
  "uncertainty": [{  // 不确定性
    "claim": "string",
    "uncertainty_marker": "string",
    "confidence": "number"
  }]
}
```

### 6.6 `domain_knowledge`

```json
{
  "terminology": [{
    "term": "string", "definition": "string",
    "domain": "string", "related_terms": ["string"]
  }],
  "rules": [{
    "rule": "string",
    "type": "deterministic_formula | regulatory_requirement | architectural_principle | domain_convention",
    "source": "string"
  }],
  "background": "string"
}
```

### 6.7 `sources`

```json
[{
  "ref": "string",
  "span": "string",
  "confidence": "number"
}]
```

### 6.8 `completeness_metadata`(关键:能力自知的载体)

```json
{
  "irr_estimate": "number [0,1] (required)",  // 信息召回率自估计
  "covered_info_types": ["L1|L2|L3|L4|L5"],
  "uncovered_info_types": ["string"],
  "flagged_for_reasoning_model": ["reasoning_chain_id"],
  "schemas_applied": ["schema_id"]
}
```

**`irr_estimate` 的含义**:记忆模型对该 Wiki 页面"覆盖原文信息完整性"的自我估计。这是能力自知原则的体现——模型必须能告诉下游"我对自己的输出有多大把握"。

---

## 7. KV 缓存作为产品:具体设计要求

### 7.1 KV 缓存必须满足的查询特性

```
┌─────────────────────────────────────────────────┐
│  同一 KV cache 能支撑以下查询模式:               │
│                                                  │
│  1. 正向查询:"X 的值是多少？"                    │
│     → 必须能从 KV 中精确召回 X 的值             │
│                                                  │
│  2. 反向查询:"哪个实体的值是 Y？"                │
│     → 必须能从 Y 反向索引到对应实体             │
│                                                  │
│  3. 关系查询:"X 和 Y 之间是什么关系？"           │
│     → 必须能从 KV 中提取 X-Y 关系              │
│                                                  │
│  4. 组合查询:"满足条件 C 的所有实体的属性？"      │
│     → 必须能从 KV 中按条件过滤并提取            │
│                                                  │
│  5. 完备性枚举:"列出所有数值/关系/否定信息"      │
│     → 必须能从 KV 中遍历所有信息维度            │
│                                                  │
│  6. 否定查询:"文中排除了什么？"                  │
│     → 必须能从 KV 中提取否定信息                │
└─────────────────────────────────────────────────┘
```

### 7.2 KV 缓存验证实验(运行时最终验证)

| 实验 | 方法 | 通过标准 |
|---|---|---|
| **KV 消融测试** | 处理文本 → KV 缓存 → 推理;清空 KV → 推理 | 消融后准确率下降 > 50% |
| **KV 扰动测试** | 文本 A(284B)→ KV_A → 答案;文本 B(285B)→ KV_B → 答案 | 答案 A ≠ 答案 B,分别基于各自数值 |
| **推理链溯源** | 检查推理链每个中间步骤是否可溯源到 KV | ≥ 90% 可溯源 |

---

## 8. 与现有架构文档的关系

| 概念 | 本文位置 | 详见 |
|---|---|---|
| **核心能力(推理无损)** | 本文 §1 | — |
| **能力边界判定(五步漏斗)** | 本文 §3 | [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) §4 |
| **Wiki DAG 构建** | 本文 §5-6(契约) | 训练侧实现:[`../agenticmemory_training/08d-wiki-dag-construction.md`](../agenticmemory_training/08d-wiki-dag-construction.md)(v0.1, 2026-08-26 新建;承接 08a Phase 1 的 OpenIE 四元组池,用户确认 2026-08-25) |
| **训练样本与课程** | 见 [`02-training-design.md`](02-training-design.md) | — |
| **评估方法** | 见 [`03-evaluation.md`](03-evaluation.md) | — |
| **架构分层(L0-L3)** | 见 [`README.md`](README.md) §1 | — |
| **LoRA 探针** | 见 [`README.md`](README.md) §2.4 | — |
| **增量 prefill** | 见 [`README.md`](README.md) §3.1 | — |

---

## 9. 待解决问题(与本文档相关)

| # | 问题 | 状态 | 建议决策时机 |
|---|---|---|---|
| O1 | Wiki DAG 构建的具体算法(节点去重 O1.1、边合并 O1.2、层级推断 O1.3,见 [`../agenticmemory_training/08d-wiki-dag-construction.md`](../agenticmemory_training/08d-wiki-dag-construction.md) §3.3 + §5) | 🔴 **待解决** | P1 启动前必须决策 |
| O2 | LoRA 探针的评估指标(如何衡量"提取了正确视角") | 🔴 **待解决** | P1 启动前必须决策 |
| **O7** | **irr_estimate 的具体计算方法**(模型自评 vs 教师评 vs 综合) | 🟡 待讨论 | P1 训练启动前 |
| **O8** | **`needs_reasoning_model_verification` 的触发阈值**(confidence < 多少?推理深度 ≥ 多少?) | 🟡 待讨论 | P1 训练启动前 |
| **O9** | **核心字段的最大长度限制**(避免 Wiki 输出过长导致训练不稳定) | 🟢 已有建议(分节训练) | 训练实施时微调 |

---

## 10. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-25 | 初始版本:基于"推理无损"核心能力定义 + 五步漏斗 + Wiki DAG 契约 | Sisyphus(AI 助手)+ 用户 |

---

**文档版本**: v0.1
**Owner**: AgenticMind 架构组
**下一步**: 待 O1(O7/O8)决策后启动 Wiki DAG 构建与训练数据合成
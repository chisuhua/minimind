# Wiki DAG 构建 — 契约对齐 + 算法骨架(O1 待解决)

> **文档 ID**: LLMTRN-008D-WIKI-DAG
> **生成日期**: 2026-08-26
> **状态**: 草案 v0.1(契约对齐完成,核心算法骨架待 O1 决策)
> **配套文档**:
> - 契约源: [`../agenticmemory/01-memory-model.md`](../agenticmemory/01-memory-model.md) §5-6 — Wiki DAG 的"是什么"(本文件对齐此契约)
> - 训练设计: [`../agenticmemory/02-training-design.md`](../agenticmemory/02-training-design.md) — Wiki DAG 在训练信号中的角色
> - 双盲提取: [`08a-capacity-gap-design.md`](08a-capacity-gap-design.md) Phase 1 — 输入侧四元组池的来源
> - Capacity Gap 分层: [`08a-capacity-gap-design.md`](08a-capacity-gap-design.md) Phase 2 — 用 Wiki DAG 子集决定"记忆层"
> - Schema 融合边界: [`08b-seed-schema-fusion.md`](08b-seed-schema-fusion.md) — 双 schema(人工 vs 涌现)的融合规则
> - P1 验证实验: [`08c-p1-minimum-loop.md`](08c-p1-minimum-loop.md) — 不涉及 Wiki DAG,验证 13 字段 schema 抽取
> - 风险登记: [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-C01/R-E01 — 本文件落地前必须解决

---

## 0. 文档范围与定位

本文档定义 **Wiki DAG 的训练侧构建实现**——给出与 [`01-memory-model.md` §6](../agenticmemory/01-memory-model.md) 完全对齐的 8 字段 JSON Schema,以及从 OpenIE 四元组池构造 Wiki DAG 的算法骨架(节点去重、边合并、层级推断)。

**关键澄清(2026-08-26 修复)**:

- **本文档 ≠ [`08a` Phase 3「Schema 自动涌现`](08a-capacity-gap-design.md)**。两者关注点和产出物完全不同:
  - **Wiki DAG**(本文档) = 训练阶段的"完整性证明"和"评估基准",结构是 **8 字段的页面**(basic_info / core_facts / relations / reasoning_chains / context_annotations / domain_knowledge / sources / completeness_metadata),用于证明"KV 缓存包含完整信息"
  - **Schema 自动涌现**(08a Phase 3) = 通过 HDBSCAN + LLM 概念化聚类得到的 **{EntityTypes, RelationTypes, Constraints}** 集合,用于供下游消费方 schema 对齐
- **本文档 ≠ [`08c` P1 验证实验](08c-p1-minimum-loop.md)**。两者完全无业务关系:
  - **Wiki DAG**(本文档) = agenticmemory 的训练时产物,8 字段结构化页面
  - **P1 13 字段 schema**(08c) = context-management 消费方的人工 schema,验证 sub-1B 是否能学会结构化抽取

**v0.1 状态说明**:

- **8 字段 JSON Schema**: 已与 `01-memory-model.md` §6 完全对齐(单一真源在 01,本文档是镜像)
- **构建算法**: 仅给骨架,核心决策(O1 节点去重 / 边合并 / 层级推断)待 P1 启动前由架构组决策
- **irr_estimate 计算方法**(O7)和 **needs_reasoning_model_verification 触发阈值**(O8)已转移到"待解决"清单

---

## 1. Wiki DAG 在训练时与运行时的角色

### 1.1 训练阶段使用

```
原始文本
  ↓ OpenIE 提取(08a Phase 1)
四元组池 + 元数据
  ↓ 节点去重 + 边合并 + 层级推断(本文档 §3)
  ↓ 八字段结构化(本文档 §2)
Wiki DAG(ground truth)
  ↓
训练 base model 学习:
  "给定文本 prefill 后的 KV 应该包含 Wiki DAG 中的所有信息"
  ↓
训练完成后验证:
  LoRA 探针查询 KV → 能否重建 Wiki DAG?
  - 能完整重建 → KV 编码完整 ✓
  - 部分缺失 → 训练不足,需补充数据 ✗
```

### 1.2 运行时使用

```
Wiki DAG 不进入运行时!
运行时 consumer(agenticmind / agenticinference)只看到:
  1. KV 缓存(由 base model prefill 产出)
  2. 自己的 LoRA 探针(从 KV 中提取特定视角)
  3. 自动涌现的 Schema(从训练侧传播的 EntityTypes / RelationTypes)
Wiki DAG 仅作为:
  - 训练时的 ground truth
  - 训练后 KV 完整性验证的对照基准
  - Schema 演化检测的对比快照
```

### 1.3 为什么由训练侧构建(用户确认 2026-08-25)

- **运行时不需要**: consumer 拿到的是 KV,不需要重新生成 Wiki DAG
- **算法复杂**: 节点去重 / 边合并 / 层级推断都需要 teacher model + 大量计算,与运行时低延迟目标冲突
- **演进解耦**: Wiki DAG 算法的迭代不应影响运行时 KV 缓存的服务可用性

---

## 2. Wiki DAG 8 字段 JSON Schema(与 `01-memory-model.md` §6 对齐)

> **单一真源**: [`../agenticmemory/01-memory-model.md` §6](../agenticmemory/01-memory-model.md)
> 本节为镜像,任何字段修改必须先改 `01-memory-model.md` §6 后,本文档同步。

### 2.1 顶层结构

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "WikiDAG",
  "type": "object",
  "required": [
    "basic_info",
    "core_facts",
    "relations",
    "reasoning_chains",
    "context_annotations",
    "domain_knowledge",
    "sources",
    "completeness_metadata"
  ],
  "properties": {
    "basic_info": { "$ref": "#/$defs/BasicInfo" },
    "core_facts": { "type": "array", "items": { "$ref": "#/$defs/CoreFact" } },
    "relations": { "$ref": "#/$defs/Relations" },
    "reasoning_chains": { "type": "array", "items": { "$ref": "#/$defs/ReasoningChain" } },
    "context_annotations": { "$ref": "#/$defs/ContextAnnotations" },
    "domain_knowledge": { "$ref": "#/$defs/DomainKnowledge" },
    "sources": { "type": "array", "items": { "$ref": "#/$defs/Source" } },
    "completeness_metadata": { "$ref": "#/$defs/CompletenessMetadata" }
  }
}
```

### 2.2 各字段定义(与 `01-memory-model.md` §6.1-6.8 完全对齐)

#### `basic_info`(§6.1)

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

#### `core_facts`(§6.2)

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

#### `relations`(§6.3)

四类关系结构:

| 子字段 | 描述 | 示例 |
|---|---|---|
| `causal_dependency` | 因果依赖(causes/enables/depends_on/prevents) | "X 导致 Y" |
| `comparison` | 对比关系(entity_a / entity_b / dimension / comparison_result) | "A 比 B 高 15%" |
| `temporal_sequence` | 时序(events / order: sequential/parallel/overlapping) | "X 发生在 Y 之前" |
| `hierarchical` | 层级关系(is_a / part_of / has_part) | "公司 is-a 组织" |

#### `reasoning_chains`(§6.4)

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

**关键设计**: `needs_reasoning_model_verification=true` 表示该推理链**需要推理模型验证**,记忆模型只负责"记住推理链存在",**不保证推理结论正确**(能力自知原则)。

#### `context_annotations`(§6.5)

```json
{
  "evaluative": [{
    "expression": "string",
    "sentiment": "positive | negative | neutral | mixed",
    "intensity": "high | medium | low",
    "target": "string"
  }],
  "conditional": [{
    "condition": "string",
    "consequence": "string",
    "certainty": "definite | probable | possible | speculative"
  }],
  "uncertainty": [{
    "claim": "string",
    "uncertainty_marker": "string",
    "confidence": "number"
  }]
}
```

#### `domain_knowledge`(§6.6)

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

#### `sources`(§6.7)

```json
[{
  "ref": "string",
  "span": "string",
  "confidence": "number"
}]
```

#### `completeness_metadata`(§6.8,关键:能力自知的载体)

```json
{
  "irr_estimate": "number [0,1] (required)",
  "covered_info_types": ["L1|L2|L3|L4|L5"],
  "uncovered_info_types": ["string"],
  "flagged_for_reasoning_model": ["reasoning_chain_id"],
  "schemas_applied": ["schema_id"]
}
```

**`irr_estimate` 定义**: 记忆模型对该 Wiki 页面"覆盖原文信息完整性"的自我估计,见 O7。

### 2.3 完整示例

```json
{
  "basic_info": {
    "title": "DeepSeek V4 Flash",
    "page_type": "entity",
    "type": "language_model",
    "domain": ["AI", "LLM"],
    "temporal_scope": "2026-Q2",
    "aliases": ["V4 Flash", "DS-V4-Flash"],
    "schemas_applied": ["schema_v1"]
  },
  "core_facts": [
    {
      "attribute": "总参数量",
      "value": 284000000000,
      "condition": null,
      "confidence": 0.99,
      "evidence": "总参数 284B",
      "temporal": null,
      "negated": false,
      "schema_ref": "schema_v1"
    },
    {
      "attribute": "激活参数量",
      "value": 13000000000,
      "condition": null,
      "confidence": 0.99,
      "evidence": "激活参数 13B",
      "temporal": null,
      "negated": false,
      "schema_ref": "schema_v1"
    },
    {
      "attribute": "推理成本",
      "value": "$0.14/百万 token (input), $0.28/百万 token (output)",
      "condition": null,
      "confidence": 0.95,
      "evidence": "定价:输入 $0.14,输出 $0.28",
      "temporal": null,
      "negated": false,
      "schema_ref": "schema_v1"
    }
  ],
  "relations": {
    "causal_dependency": [
      {"cause": "MoE 架构", "effect": "稀疏激活", "type": "enables"}
    ],
    "comparison": [
      {
        "entity_a": "V4 Flash",
        "entity_b": "GPT-4",
        "dimension": "推理成本 (input)",
        "comparison_result": "V4 Flash 约为 GPT-4 的 1.4%"
      }
    ],
    "temporal_sequence": [],
    "hierarchical": [
      {"child": "V4 Flash", "parent": "DeepSeek V 系列", "relation": "is_a"}
    ]
  },
  "reasoning_chains": [
    {
      "id": "rc_001",
      "description": "V4 Flash 推理成本优势推导",
      "premise": ["总参数 284B", "激活参数 13B", "MoE 架构"],
      "inference": "MoE 架构使每次推理仅激活部分参数,实际计算量小",
      "conclusion": "V4 Flash 推理成本远低于 GPT-4",
      "evidence": "激活参数 13B/总参数 284B ≈ 4.6%",
      "explicit_in_text": false,
      "confidence": 0.85,
      "needs_reasoning_model_verification": true
    }
  ],
  "context_annotations": {
    "evaluative": [
      {"expression": "成本优势", "sentiment": "positive", "intensity": "high", "target": "V4 Flash"}
    ],
    "conditional": [],
    "uncertainty": []
  },
  "domain_knowledge": {
    "terminology": [
      {"term": "MoE", "definition": "Mixture of Experts", "domain": "ML", "related_terms": ["稀疏激活"]}
    ],
    "rules": [],
    "background": "DeepSeek 系列模型"
  },
  "sources": [
    {"ref": "doc_001", "span": "总参数 284B,激活参数 13B", "confidence": 0.99}
  ],
  "completeness_metadata": {
    "irr_estimate": 0.92,
    "covered_info_types": ["L1", "L3"],
    "uncovered_info_types": [],
    "flagged_for_reasoning_model": ["rc_001"],
    "schemas_applied": ["schema_v1"]
  }
}
```

---

## 3. 构建算法骨架(O1 待解决)

> **本节给出算法骨架,具体决策待 O1 在 P1 启动前完成**(详见 §5)。

### 3.1 输入与输出

**输入**:
- 原始文本(2000-8000 tokens,来自 `08a Phase 1` 双盲提取)
- OpenIE 四元组池(由 teacher model 提取,见 `08a §3`)
- 自动涌现的 Schema(`08a Phase 3` 输出)

**输出**:
- Wiki DAG JSON(符合 §2 Schema)

### 3.2 算法流程(骨架)

```
原始文本 + 四元组池 + Schema
  │
  ├─ Step 1: 实体归一化(节点去重)
  │   输入: 四元组池中的所有实体提及
  │   输出: 唯一实体集合(每个实体有 canonical_name + aliases + entity_type)
  │   算法: [O1.1 待决策]
  │
  ├─ Step 2: 关系归一化(边合并)
  │   输入: 四元组池中的所有关系
  │   输出: 归一化的 relations 4 类(causal/comparison/temporal/hierarchical)
  │   算法: [O1.2 待决策]
  │
  ├─ Step 3: 推理链抽取
  │   输入: 关系归一化结果 + 原文
  │   输出: reasoning_chains 数组
  │   算法: 基于 teacher model 抽取(参考 `02-training-design.md` §5)
  │
  ├─ Step 4: 上下文标注生成
  │   输入: 关系 + 推理链 + 原文
  │   输出: context_annotations(evaluative/conditional/uncertainty)
  │   算法: 基于 teacher model 抽取
  │
  ├─ Step 5: 领域知识抽取
  │   输入: 原文中的术语、规则、背景
  │   输出: domain_knowledge(terminology/rules/background)
  │   算法: [O1.3 待决策,涉及层级推断]
  │
  ├─ Step 6: 完整性自评
  │   输入: 当前 Wiki DAG + 原文
  │   输出: completeness_metadata(irr_estimate 等)
  │   算法: [O7 待决策]
  │
  └─ Step 7: needs_reasoning_model 标注
      输入: reasoning_chains + confidence
      输出: needs_reasoning_model_verification 字段填充
      算法: [O8 待决策,基于置信度阈值 + 推理深度]
```

### 3.3 关键算法的设计权衡(待 O1 决策)

| 算法 | 候选方案 | 权衡 |
|---|---|---|
| **实体归一化(O1.1)** | (a) 基于 embedding 相似度聚类<br/>(b) 基于 LLM 实体对齐调用<br/>(c) 混合(粗筛用 embedding,精筛用 LLM) | 成本 vs 准确率 |
| **关系归一化(O1.2)** | (a) 规则映射(`导致`/`造成` → `causes`)<br/>(b) LLM 概念化<br/>(c) 聚类(参考 08a) | 稳定性 vs 召回率 |
| **层级推断(O1.3)** | (a) 基于 schema 中的 is_a 关系<br/>(b) 基于 LLM 推断<br/>(c) 预定义 + LLM 修正 | 自动化程度 vs 准确性 |

**P1 启动前必须决策**: 至少确定每个候选算法的默认选择(允许后续迭代)。

---

## 4. 与训练管线的衔接

### 4.1 与 `08a Phase 1`(双盲 OpenIE 提取)的衔接

```
08a Phase 1 输出: 四元组池 (subject, relation, object, confidence, evidence)
  ↓ (本文档 §3.2 Step 1-2)
节点归一化 + 边归一化
  ↓ (本文档 §3.2 Step 3-5)
Wiki DAG 八字段填充
  ↓ (本文档 §3.2 Step 6-7)
最终 Wiki DAG
```

**关键依赖**: Wiki DAG 构建必须在 08a Phase 1 完成后,**不依赖 08a Phase 2 (CCS 分层)**——Wiki DAG 与 CCS 分层是平行关系,CCS 用于从 Wiki DAG 中筛选"记忆层"样本。

### 4.2 与 `08a Phase 2`(CCS 分层)的衔接

```
Wiki DAG(全集)
  ↓ CCS 公式: 0.5·gap + 0.3·recon + 0.2·bottleneck (见 08a §4.4)
  ↓ 阈值: CCS < 0.3 进入记忆层 / CCS > 0.7 进入推理层 / 其他进人工审核
分层后样本:
  - 记忆层(高置信)
  - 推理层(高置信)
  - 混合层(进人工审核)
```

**注意**: 记忆层样本的"完整性证明"就是对应的 Wiki DAG 子集——只有当 Wiki DAG 中的某个子集能被 KV 完整重建,该子集才算合格训练样本。

### 4.3 与 `08b`(Schema 融合边界)的衔接

`08b` 定义了双 schema 分离原则:
- **人工 schema**(消费方): `docs/agenticmind/context-management/mvp-schema.md` 的 13 字段
- **涌现 schema**(记忆侧): 08a Phase 3 + 本文档 §3.2 Step 3-5 使用的 Schema

**关键不变量**: Wiki DAG 中的 `basic_info.type`、`core_facts.schema_ref`、`completeness_metadata.schemas_applied` 字段**只引用涌现 schema**,**不引用人工 schema**。这是双 schema 分离的强制约束(详见 `08b-seed-schema-fusion.md` §2)。

### 4.4 与 `08c`(P1 验证实验)的衔接

**无业务关系**。P1 验证的是 13 字段人工 schema 的抽取能力,与 Wiki DAG 完全不同。建议读者不要混淆:
- P1 关心的: "sub-1B 模型能否学会 13 字段 schema 抽取"
- 本文档关心的: "如何构建 Wiki DAG 作为训练时 ground truth"

---

## 5. 待解决算法问题

> 以下问题在 P1 启动前**必须**决策(由架构组 + 训练组联合),否则 Wiki DAG 构建无法启动。

| ID | 问题 | 候选方案 | 默认建议 | 决策时机 |
|---|---|---|---|---|
| **O1.1** | 节点去重(实体归一化)算法 | (a) embedding 聚类 (b) LLM 实体对齐 (c) 混合 | (c) 混合(参考 08a) | P1 启动前 |
| **O1.2** | 边合并(关系归一化)算法 | (a) 规则映射 (b) LLM 概念化 (c) 聚类 | (b) LLM 概念化(准确率优先) | P1 启动前 |
| **O1.3** | 层级推断(is_a / part_of)算法 | (a) 基于 schema (b) LLM 推断 (c) 预定义+修正 | (a) 基于 schema | P1 启动前 |
| **O7** | `irr_estimate` 计算方法 | (a) 模型自评 (b) 教师评 (c) 综合(自评×权重+教师评×权重) | (c) 综合(教师评 0.6 + 自评 0.4) | P1 训练启动前 |
| **O8** | `needs_reasoning_model_verification` 触发阈值 | (a) confidence < 0.7<br/>(b) 推理深度 ≥ N 步<br/>(c) (a) AND (b) | (c) 双重判定 | P1 训练启动前 |

**风险登记**: 见 [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md)
- **R-C01**(Wiki DAG 构建算法无法保证完整性)— 由本文档落地的算法决策缓解
- **R-E01**(Wiki DAG 构建算法无法按时交付)— 由本文档创建的契约 + 骨架缓解

---

## 6. 验证 Wiki DAG 的三种用途(来自 `01-memory-model.md` §5.4)

| 用途 | 方法 | 通过标准 |
|---|---|---|
| **KV 完整性验证** | 用 LoRA 探针从 KV 中提取所有实体和关系,与 Wiki DAG 对比 | 实体召回 ≥ 95%,关系召回 ≥ 90% |
| **LoRA 探针训练目标** | LoRA 学习"给定 probe query,从 KV 中提取 Wiki DAG 子图" | 子图匹配 F1 ≥ 0.85 |
| **Schema 演化检测** | 新语料 prefill 后,探针提取结果与旧 Wiki DAG 对比 | 新增实体/关系比例 < 10% 为稳定 |

**注意**: 这三种用途都需要 Wiki DAG 在训练时构建完成,**训练后冻结**(作为 ground truth)。演化检测时使用的是"新 Wiki DAG vs 旧 Wiki DAG",而非重建 KV。

---

## 7. 与 `agenticmemory_training/` 其他文档的边界

| 内容 | 归属 | 备注 |
|---|---|---|
| Wiki DAG 8 字段契约 | [`01-memory-model.md`](../agenticmemory/01-memory-model.md) §6 | **单一真源**,本文档镜像 |
| Wiki DAG 构建算法 | **本文档** | 骨架,核心决策待 O1 |
| OpenIE 双盲提取 | [`08a-capacity-gap-design.md`](08a-capacity-gap-design.md) Phase 1 | 四元组池输入源 |
| CCS 分层 | [`08a-capacity-gap-design.md`](08a-capacity-gap-design.md) Phase 2 | 用 Wiki DAG 子集做训练样本筛选 |
| Schema 自动涌现 | [`08a-capacity-gap-design.md`](08a-capacity-gap-design.md) Phase 3 | 与 Wiki DAG 是"同管线不同产出" |
| Schema 融合边界 | [`08b-seed-schema-fusion.md`](08b-seed-schema-fusion.md) | 双 schema 分离 |
| P1 验证实验 | [`08c-p1-minimum-loop.md`](08c-p1-minimum-loop.md) | 无业务关系 |

---

## 8. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-26 | 初始版本:与 `01-memory-model.md` §6 契约对齐 + 构建算法骨架 + 待解决清单(O1.1/O1.2/O1.3/O7/O8) | Sisyphus(AI 助手) |

---

**文档版本**: v0.1
**Owner**: AgenticMind 架构组 + 训练组(联合决策)
**下一步**:
1. 架构组 + 训练组在 P1 启动前完成 O1.1 / O1.2 / O1.3 决策(算法骨架)
2. O7 / O8 在 P1 训练启动前决策
3. 算法决策完成后,本文档从"骨架"升级为"实施方案"

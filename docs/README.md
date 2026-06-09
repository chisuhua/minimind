# minimind文档中心（Documentation Hub）

> **本 README用途**：定义 minimind项目的核心目标 —— **认知推理模型（Cognitive Reasoning Model）**，并索引所有相关调研与设计文档。

---

## 🎯 项目核心目标

> **用 <1B激活参数构造一个"认知推理模型"（Cognitive Reasoning Model），区别于 LLM 的 CoT推理，核心是基于"领域本体 +逻辑规则 +文档三元组 + AgenticDSL + 自然语言 +基础认知"六层知识形态栈，进行6 类认知推理（因果、依赖、相似、对象联系、语义关系、规则/时间）。模型知道自己不知道的知识，并能通过拒答、检索、学习提示三种机制处理知识缺口。**

---

## 🧠 一句话定义"认知推理"

**认知推理 ≠ CoT推理**。

|维度 | LLM CoT推理 | **认知推理（本项目）** |
|---|---|---|
| **知识基础** | 模型参数化的隐式知识 | **显式 ontology +规则 + 三元组**（可外部检索） |
| **推理类型** |自由语言多步推导 | **因果、依赖、相似、对象联系、语义、规则/时间6 类形式化推理** |
| **推理过程** | 自然语言 CoT链 | **实体-关系图谱遍历 +逻辑规则 +外部求解器** |
| **可解释性** | 黑盒（语言叙述） | **可审计**（图谱路径 +求解器输出） |
| **知识更新** |需重训 | **ontology/规则可独立更新** |
| **失败模式** |幻觉（自由生成错误） | **拒答**（超出 ontology范围时） |

---

## 🧱核心设计要素

###1.6 层知识形态栈

```
┌────────────────────────────────────────┐
│ L6:基础认知 (Basic Cognition) │ ←通用认知能力（抽象、类比）
├────────────────────────────────────────┤
│ L5: 自然语言描述 (Natural Language) │ ←概念的自然语言解释
├────────────────────────────────────────┤
│ L4: AgenticDSL 语言 │ ←领域专用符号语言（如 Prolog DSL）
├────────────────────────────────────────┤
│ L3:文档三元组 (Document Triples) │ ← 从文档抽取的实体-关系
├────────────────────────────────────────┤
│ L2:逻辑规则 (Logic Rules) │ ← SWRL / Datalog形式化规则
├────────────────────────────────────────┤
│ L1:领域本体 (Domain Ontology) │ ← OWL/RDFS定义的类与实例
└────────────────────────────────────────┘
```

每层都**可独立更新**、**可单独训练**、**可独立评测**。

###2.6 类认知推理（全谱覆盖）

| # |推理类型 |形式化操作 | ontology 来源 |
|---|---|---|---|
|1 | **因果推理** | `cause(X, Y)` / `if X then Y` | ATOMIC, ConceptNet (Causes) |
|2 | **依赖推理** | `depends_on(X, Y)` / `requires(X, Y)` | 自建逻辑规则 |
|3 | **相似/相识推理** | `similarTo(X, Y)` / `acquaintanceOf(X, Y)` | ConceptNet (SimilarTo), WordNet |
|4 | **对象联系推理** | `partOf(X, Y)` / `hasProperty(X, Y)` | ConceptNet, 自建本体 |
|5 | **语义关系推理** | `isA(X, Y)` / `hasA(X, Y)` | WordNet, ConceptNet |
|6 | **规则/时间推理** | `before(X, Y)` / `rule(X, Y, Z)` | OWL-Time, SWRL规则 |

###3. "知道自己不知道"机制

|机制 |触发条件 | 实现方式 |
|---|---|---|
| **拒答** | 问题涉及 ontology 未覆盖的实体/关系 | 模型输出 `out_of_ontology` +缺口说明 |
| **检索** |缺口可通过扩展 ontology补齐 | Loreto序列化扩展 ontology → context注入 |
| **学习提示** |缺口需用户主动补充 | 输出"需要补充知识 X" +检索接口 |

每个推理输出附带**校准后的 `confidence ∈ [0,1]`**，用于触发上述机制。

---

## 🛤️ 技术路径：分阶段演进

**不直接上完整形态**——按 minimind "大道至简"哲学，**每阶段都产生可验证的中间产物**：

|阶段 | 时间 |范式 |目标 |成功标准 |
|---|---|---|---|---|
| **阶段1（M1-M2）** |8 周 | **方案 A**: Logic-LM + GCR 双引擎 |验证 sub-1B + ontology +外部 solver 能跑通 |6 类推理 ≥60%准确率 |
| **阶段2（M3-M4）** |8 周 | **方案 C核心层**: NeurASP 内化 | 让模型"真正记住"核心 ontology | 内化推理 ≥50% |
| **阶段3（M5-M6）** |8 周 | **方案 C完整态**:置信度 +拒答 +扩展层 |完整认知推理框架 |7 项指标全部达标 |

###阶段1架构示意

```
用户问题
 ↓
0.5B LLM (符号化 + 本体引用)
 ↓
符号程序 +实体 +关系指针
 ↓
外部求解器 (Prover9 / Z3 / clingo)
 ↓
求解结果 +解释
 ↓
0.5B LLM (自然语言回复 +置信度)
```

###阶段3架构示意（完整形态）

```
用户问题
 ↓
0.5B认知模型
 ├──核心层（参数化）：NeurASP 内化
 ├──扩展层（检索）：Loreto序列化 context
 └──拒答层：置信度阈值决策
 ↓
自然语言回复 +置信度 + ontology引用
```

---

## 📊评测体系：7 项完整指标

| # |指标 |阶段1 |阶段2 |阶段3 |
|---|---|---|---|---|
|1 | **6 类推理准确率** | ≥60% | ≥70% | ≥80% |
|2 | **ontology忠实度** | ≥70% | ≥80% | ≥90% |
|3 | **拒答准确率** | ≥50% | ≥70% | ≥85% |
|4 | **零样本泛化** | — | ≥30% | ≥50% |
|5 | **反例鲁棒性** | — | — | ≥60% |
|6 | **跨领域迁移** | — | — | ≥40% |
|7 | **对齐率**（vs GPT-4） | — | ≥60% | ≥70% |

**评测 benchmark清单**：
- Logic-LM5 个 benchmark（ProofWriter / PrOntoQA / FOLIO / LogicalDeduction / AR-LSAT）
- KGQA3 个 benchmark（WebQSP / CWQ / GrailQA）
- Temporal：CronKGQA / RE-Net 测试集
- Causal：ATOMIC / SCIE 测试集
-通用：CommonsenseQA / OpenBookQA

---

## 🧩 技术栈选型（已锁定）

|类别 |选型 |理由 |
|---|---|---|
| **Base 模型** | Qwen2-0.5B | GCR已在0.5B验证 KG reasoning |
| **本体库** | ConceptNet5.5 + WordNet + ATOMIC + OWL-Time | 全谱6 类推理覆盖 |
| **求解器** | Prover9 (FOL) + clingo (ASP) + Z3 (SMT) | Logic-LM范式标准组件 |
| **本体序列化** | Loreto | token减少30%+，LLM-friendly |
| **本体嵌入** | OWL2Vec* | OWL ontology →稠密向量 |
| **训练范式** | Logic-LM + NeurASP + DPO |阶段递进 |
| **KG推理** | GCR + ToG + PoG | sub-1B友好 |

---

## 📁文档结构索引

###核心目标与设计

|文档 | 内容 |
|---|---|
| **[`README.md`](./README.md)** | **本文件**：项目核心目标与文档索引 |
| **[`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md)** | 项目目标完整定义（含决策记录、技术路径、风险预案、补充维度：KG-Prolog闭环 + 四重置信度过滤 + 因果推理） |

###调研与论证

|文档 | 内容 | 推荐度 |
|---|---|---|
| [`reasoning-distillation-survey/`](./reasoning-distillation-survey/) |推理轨迹蒸馏 + 分步迭代推理双方案调研 | ⭐⭐⭐⭐ |
| [`small-model-reasoning-survey/`](./small-model-reasoning-survey/) |1B 小模型推理能力综合调研（含 R1-Distill + GRPO） | ⭐⭐⭐⭐⭐ |
| [`reasoning-architectures/`](./reasoning-architectures/) | minimind推理架构7轮迭代（v1 → v4.6 → AGI） | ⭐⭐⭐⭐⭐ |
| [`reasoning-sota-critical-eval.md`](./reasoning-sota-critical-eval.md) |推理 SOTA批判性评估 | ⭐⭐⭐ |
| [`inference-gap-analysis.md`](./inference-gap-analysis.md) |推理加速技术 gap 分析 | ⭐⭐⭐ |
| [`training-gap-analysis.md`](./training-gap-analysis.md) |训练技术 gap 分析 | ⭐⭐⭐ |

### 子目录详细索引

| 子目录 |核心内容 |
|---|---|
| [`reasoning-distillation-survey/`](./reasoning-distillation-survey/) |推理蒸馏双方案核查 +21 个开源实现清单 |
| [`small-model-reasoning-survey/`](./small-model-reasoning-survey/) |6 个并行 agent调研底稿 + 用户提议 +修正建议 |
| [`reasoning-architectures/`](./reasoning-architectures/) |7轮架构迭代 + 最终推荐 v4.5 主路径 |
| [`training-technologies/`](./training-technologies/) |训练技术调研 |
| [`inference-technologies/`](./inference-technologies/) |推理引擎与加速技术调研 |

---

## ✅决策记录（已锁定13 项）

| # |决策项 | 用户选择 |
|---|---|---|
|1 | 项目目标 | <1B认知推理模型 |
|2 |推理区别边界 | 基于本体论的形式化推理 |
|3 |推理范围 | 全谱6 类 +领域聚焦 |
|4 |知识形态栈 |6 层（本体 +规则 + 三元组 + DSL + NL +基础认知） |
|5 | 内化路径 |混合架构（核心微调 +动态检索） |
|6 | "不知道"机制 |拒答 +检索 +置信度校准 |
|7 | 数据生成起点 | 本体 +文档 + 问题（混合） |
|8 |评测体系 |完整7 项指标 |
|9 | MVP领域 |通用常识 +逻辑推理 |
|10 | 主路径 | 分阶段演进（A → C） |
|11 |阶段1范围 |6 类全谱覆盖 |
|12 |失败模式预案 |3阶段各自有切回方案 |
|13 |文档目标定义位置 | docs/README.md（顶层 README） |

---

## 🎓学术差异化（minimind 的"首个"价值）

1. **首个公开的 sub-1B 本体论认知推理模型**（vs GCR 仅 KG推理、Logic-LM需大模型）
2. **首个完整"核心内化 +扩展检索 +置信度拒答"sub-1B框架**
3. **首个 sub-1B Temporal/Causal形式化推理实证**（如成功）

---

## 🚀 阅读建议

|读者 | 阅读顺序 |
|---|---|
| **项目负责人 /决策者** | 本 README → [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md) |
| **架构师 / 技术负责人** | 本 README → [`reasoning-architectures/`](./reasoning-architectures/) → [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md) |
| **算法工程师** | [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md) → [`reasoning-distillation-survey/`](./reasoning-distillation-survey/) → [`small-model-reasoning-survey/`](./small-model-reasoning-survey/) |
| **数据 /训练工程师** | [`small-model-reasoning-survey/`](./small-model-reasoning-survey/) → [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md) |
| **新加入成员** | 本 README（建立全局观）→各自方向对应子目录 |

---

## 🔗 下一步行动

1. ✅ **项目目标已定义**（本 README + `cognitive-reasoning-model.md`）
2. **下一步**：调用 `writing-plans` skill 生成详细实施计划（M1-M2阶段1）
3. **再下一步**：按计划逐步执行阶段1（Logic-LM + GCR 双引擎 MVP）

---

> **核心承诺**：本项目目标严格对齐 minimind "大道至简"哲学 —— **用最小的模型 + 最严谨的形式化推理**，**实现"知道自己不知道"的认知推理能力**，**而不是用更大的模型堆叠能力**。

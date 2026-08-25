# AgenticMind 文档中心（Documentation Hub）

> **本 README用途**：定义 **AgenticMind 项目**（代号 `AgenticMind`，仓库目录 `AgenticMind`，fork 自开源 MiniMind 作为训练链路载体）的核心目标 —— **认知推理模型（Cognitive Reasoning Model）**，并索引所有相关调研与设计文档。

---

> 🚪 **新成员入口**
>
> - **AI agent / 开发协作者** → 请先读 [`/AGENTS.md`](../AGENTS.md)（工作入口与上下文手册）
> - **项目负责人 / 决策者** → 本 README → [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md) → [`architectures/99-final-recommendation.md`](./architectures/99-final-recommendation.md)
> - **AgenticDSL 训练工程师** → [`agenticdsl-training/`](./agenticdsl-training/) → [`architectures/06-metacognitive-closed-loop.md`](./architectures/06-metacognitive-closed-loop.md)
> - **新加入成员** → [`/AGENTS.md`](../AGENTS.md)（建立全局观）→ 本 README → [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md)

---

## 🎯 项目核心目标

> **用 <1B 激活参数，通过大模型与智能体的紧耦合协作，达到认知推理的 SOTA。**
>
> **"紧耦合"** 指 LLM 与智能体运行时（agent runtime）通过 **共享结构化状态**（如 KV cache prefix 复用、assert 验证器反馈、ontology/规则子图）协同，而不是仅通过自然语言 tool call。这避免了 Mirror Loop 风险（无 grounding 的语义循环），也是 LLM + agent 区别于"LLM 调用工具"的核心特征。
>
> 关键技术之一是通过 AgenticDSL 进行 **自循环推理**，包括两个层次：
> 1. **Confidence-Triggered Loop**：LLM 生成 AgenticDSL 程序 → 智能体运行时执行 → 置信度评估 → 不达标则触发知识检索/注入 → 重新推理，直到达标
> 2. **Prefix-Accumulation Step-by-Step Loop**：LLM 每次只推理一个步骤 → 上一步推理过程作为 prefix 累积 → 在不改变 prefix 的前提下要求推理下一步骤 → 通过"单步深度 + prefix 累积"避免小模型做长推理的 context 局限
>
> 通过两个层次的自循环，在保持 <1B 激活参数的同时获得长链推理能力。区别于 LLM 的 CoT 推理，模型以 **"领域本体 + 逻辑规则 + 文档三元组 + AgenticDSL + 自然语言 + 基础认知"** 六层知识形态栈为基础，执行 6 类认知推理（因果、依赖、相似、对象联系、语义关系、规则/时间），并通过拒答、检索、学习提示三种机制处理知识缺口。

---

## 🧠 一句话定义"认知推理"

**认知推理 ≠ CoT推理**。

|维度 | LLM CoT推理 | **认知推理（本项目）** |
|---|---|---|
| **知识基础** | 模型参数化的隐式知识 | **显式 ontology +规则 + 三元组**（可外部检索） |
| **推理类型** |自由语言多步推导 | **因果、依赖、相似、对象联系、语义、规则/时间6 类形式化推理** |
| **推理过程** | 自然语言 CoT链 | **AgenticDSL 驱动计算图 + 实体-关系图谱遍历 +逻辑规则 +外部求解器** |
| **运行模式** | 单轮生成 → 输出文本 | **两层次自循环推理**（Confidence-Triggered Loop + Prefix-Accumulation Step-by-Step Loop） |
| **可解释性** | 黑盒（语言叙述） | **可审计**（AgenticDSL 程序 + 图谱路径 + 求解器输出 + 评估轨迹） |
| **知识更新** |需重训 | **ontology/规则/AgenticDSL 子图可独立更新，通过两层次自循环持续优化** |
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

**不直接上完整形态**——按 AgenticMind "大道至简"哲学，**每阶段都产生可验证的中间产物**：

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

## 📊评测体系：10 项完整指标

| # |指标 |阶段1 |阶段2 |阶段3 |
|---|---|---|---|---|
|1 | **6 类推理准确率** | ≥60% | ≥70% | ≥80% |
|2 | **ontology忠实度** | ≥70% | ≥80% | ≥90% |
|3 | **拒答准确率** | ≥50% | ≥70% | ≥85% |
|4 | **零样本泛化** | — | ≥30% | ≥50% |
|5 | **反例鲁棒性** | — | — | ≥60% |
|6 | **跨领域迁移** | — | — | ≥40% |
|7 | **对齐率**（vs 同尺寸 SOTA） | — | ≥60% | ≥70% |
| **8** | **自循环收敛速度**（loop 触发次数 / 收敛步数） | ≤ 3 步 | ≤ 2 步 | ≤ 2 步 |
| **9** | **Prefix-Accumulation 收益**（与 Long-CoT 退化对比） | ≥ +0% | ≥ +10% | ≥ +20% |
| **10** | **Mirror Loop 防御率**（循环不收敛失败模式检测） | ≥ 90% | ≥ 95% | ≥ 99% |

**评测 benchmark清单**：
- Logic-LM5 个 benchmark（ProofWriter / PrOntoQA / FOLIO / LogicalDeduction / AR-LSAT）
- KGQA3 个 benchmark（WebQSP / CWQ / GrailQA）
- Temporal：CronKGQA / RE-Net 测试集
- Causal：ATOMIC / SCIE 测试集
-通用：CommonsenseQA / OpenBookQA

---

## 📊 SOTA 对标基准

> 本项目 SOTA 主张采用**双轨制**——对内争 **sub-1B 范围 SOTA**,对外引用 **GPT-4 / Claude** 作为上限参考,展示"小模型 + 紧耦合协作"的相对优势。

| 维度 | 对内基线（sub-1B 实证） | 对外参考（跨尺度 SOTA） |
|---|---|---|
| **同尺寸逻辑推理** | T5-large 770M 在 ProofWriter depth-5 达 **85.4%** / NL2LOGIC 0.5B-1.5B AST-guided 达 **99% executable** | — |
| **跨尺度参考** | — | GPT-4 / Claude-3.5 / DeepSeek-R1 |
| **Tool-Use 实证** | TinyAgent-1.1B + ToolRAG 达 **80.06%** function calling | GPT-4-turbo 79.08%（**已超**） |
| **TTS 数学** | 1B + TTS 在 MATH-500 **超 405B** 模型 | DeepSeek-R1 / o1 |
| **自循环可观测** | Confidence-Triggered Loop 收敛步数 ≤ 2 / Prefix-Accumulation 收益 ≥ +10% vs Long-CoT 退化 | Self-Refine (NeurIPS 2023) 平均 +20% 提升 |

**不宣称**:
- ❌ 跨尺度通用 SOTA。所有 SOTA 主张限定在 **sub-1B 子领域**的"首个 / 持平 / 超越"
- ❌ 完全消解 LLM 一次性生成能力。本项目**不取代** GPT-4/Claude,而是提供"小模型 + 紧耦合"路径的另一种选择
- ❌ HydraForge VN-001 的"自举"愿景(那是 HydraForge 的 4 阶段宏观路线,见 [`agenticdsl-training/06-vn001-alignment.md`](./agenticdsl-training/06-vn001-alignment.md))

---

## 🧩 技术栈选型（已锁定）

|类别 |选型 |理由 |
|---|---|---|
| **Base 模型** | Qwen2-0.5B | GCR已在0.5B验证 KG reasoning |
| **本体库** | ConceptNet5.5 + WordNet + ATOMIC + OWL-Time | 全谱6 类推理覆盖 |
| **求解器** | Prover9 (FOL) + clingo (ASP) + Z3 (SMT) | Logic-LM范式标准组件 |
| **本体序列化** | Loreto | token减少30%+，LLM-friendly |
| **本体嵌入** | OWL2Vec* | OWL ontology →稠密向量 |
| **AgenticDSL 语言** | HydraForge AgenticDSL（Markdown + 嵌入式 YAML + 显式签名 + 强制围栏） | LLM-aware、可验证、可执行；详见 [`agenticdsl-training/`](./agenticdsl-training/) |
| **智能体运行时** | HydraForge C++ 引擎（ILLMProvider + Topological Scheduler + ToolRegistry + BudgetController + OpenTelemetry Trace） | 自循环推理的执行与评估底座 |
| **训练范式** | Logic-LM + NeurASP + DPO + AgenticDSL 自训练（ReSTᴱᴹ → OmegaPRM → MCTS → GRPO） |阶段递进，覆盖六层栈 + 自循环推理 |
| **KG推理** | GCR + ToG + PoG | sub-1B友好 |

---

## 📁 文档结构索引

> **重组时间**：2026-06-11
> **重组原则**：按"自循环认知推理模型"新目标分级——**直接相关(DIRECT / CRITICAL)** 放在顶层 3 个技术系列目录,**间接 / 边缘相关(INDIRECT / PERIPHERAL)** 移入 `references/` 参考目录。

### 🎯 核心目标与设计(DIRECT · 顶层)

| 文档 | 关联度 | 内容 |
|---|---|---|
| **[`README.md`](./README.md)** | ⭐⭐ | **本文件**：项目核心目标与文档索引 |
| **[`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md)** | ⭐⭐ | 项目目标完整定义(含决策记录、技术路径、风险预案、补充维度:KG-Prolog 闭环 + 四重置信度过滤 + 因果推理) |

### 🧠 AgenticDSL 训练主干(DIRECT · 技术系列 A)

> **目录**:[`agenticdsl-training/`](./agenticdsl-training/) —— 8 份,直接对应自循环推理中 LLM **生成 AgenticDSL 程序**的能力训练。

| 文档 | 关联度 | 内容 |
|---|---|---|
| [`agenticdsl-training/README.md`](./agenticdsl-training/README.md) | ⭐⭐ | AgenticDSL LLM 训练综述 |
| [`agenticdsl-training/01-training-data-pipeline.md`](./agenticdsl-training/01-training-data-pipeline.md) | ⭐⭐ | 9 阶段 SFT 数据构造 |
| [`agenticdsl-training/02-training-algorithms.md`](./agenticdsl-training/02-training-algorithms.md) | ⭐⭐ | 6 阶段自训练 Recipe(ReSTᴱᴹ → OmegaPRM → MCTS → GRPO → SPIN) |
| [`agenticdsl-training/03-inference-time-guarantees.md`](./agenticdsl-training/03-inference-time-guarantees.md) | ⭐⭐ | XGrammar + Tree-sitter 推理栈,确保 AgenticDSL 生成合规 |
| [`agenticdsl-training/04-evaluation-benchmark.md`](./agenticdsl-training/04-evaluation-benchmark.md) | ⭐⭐ | HydraForgeBench 8 维度评估 |
| [`agenticdsl-training/05-risk-register.md`](./agenticdsl-training/05-risk-register.md) | ⭐⭐ | 12 个训练关键风险 |
| [`agenticdsl-training/06-vn001-alignment.md`](./agenticdsl-training/06-vn001-alignment.md) | ⭐⭐ | 与 **HydraForge VN-001 愿景**对齐路径(HydraForge 的"自举"是 4 阶段宏观路线,**非本项目目标**) |
| [`agenticdsl-training/07-vs-initial-analysis.md`](./agenticdsl-training/07-vs-initial-analysis.md) | ⭐⭐ | 与初步分析差异对照 |

### 💾 记忆训练数据集构建(DIRECT · 技术系列 A.5)

> **目录**:[`agenticmemory_training/`](./agenticmemory_training/) —— 3 份,为 AgenticDSL 训练链路**前置构建记忆数据集**(Capacity Gap 自动分层 + Schema 自动涌现)。产出物喂给 [`agenticdsl-training/01-training-data-pipeline.md`](./agenticdsl-training/01-training-data-pipeline.md) 第 3 阶段。

| 文档 | 关联度 | 内容 |
|---|---|---|
| [`agenticmemory_training/README.md`](./agenticmemory_training/README.md) | ⭐⭐ | 记忆训练数据集构建综述（与 `agenticdsl-training/` 的边界） |
| [`agenticmemory_training/08-memory-distillation-pipeline.md`](./agenticmemory_training/08-memory-distillation-pipeline.md) | ⭐⭐ | RTX 4090 单卡实操搭建指南（v1.1，含完整代码与配置） |
| [`agenticmemory_training/08a-capacity-gap-design.md`](./agenticmemory_training/08a-capacity-gap-design.md) | ⭐⭐ | v0.1 设计方案（理论/决策层，含附录 A 设计层决策表 13 项） |

### 🏗️ 推理架构(DIRECT · 技术系列 B)

> **目录**:[`architectures/`](./architectures/) —— 4 份,推理架构 7 轮迭代中提取的 DIRECT 主线。

| 文档 | 关联度 | 内容 |
|---|---|---|
| [`architectures/README.md`](./architectures/README.md) | ⭐ | 目录索引(7 轮全景 / v4.5 收敛 / 元认知闭环 / 最终决策) |
| [`architectures/00-iteration-timeline.md`](./architectures/00-iteration-timeline.md) | ⭐ | 7 轮推理架构迭代全景图 |
| [`architectures/04b-v4.5-and-v4.6.md`](./architectures/04b-v4.5-and-v4.6.md) | ⭐ | v4.5 务实收敛 + v4.6 知识外挂(唯一可落地工程路径) |
| [`architectures/06-metacognitive-closed-loop.md`](./architectures/06-metacognitive-closed-loop.md) | ⭐⭐ | **元认知闭环**:与"自循环"概念一一对应(推理→置信度→检索→重推理) |
| [`architectures/99-final-recommendation.md`](./architectures/99-final-recommendation.md) | ⭐ | 最终推荐路线 + Kill Criteria |

### ⚙️ 推理引擎(CRITICAL_FOR_LOOP · 技术系列 C)

> **目录**:[`inference-engine/`](./inference-engine/) —— 3 份,**自循环 AgenticDSL 推理的关键基础设施**(不是"性能优化",而是"自循环工程基础")。

| 文档 | 关联度 | 关键作用 |
|---|---|---|
| [`inference-engine/README.md`](./inference-engine/README.md) | ⭐ | 目录索引(为何这些是自循环关键支撑) |
| [`inference-engine/01-pre-allocated-kv-cache.md`](./inference-engine/01-pre-allocated-kv-cache.md) | ⭐⭐ | 预分配 KV 缓存 — prefix 复用物理底座(已实现) |
| [`inference-engine/02-streaming-llm.md`](./inference-engine/02-streaming-llm.md) | ⭐⭐ | StreamingLLM — 长链路不爆显存(已实现) |
| [`inference-engine/09-kivi.md`](./inference-engine/09-kivi.md) | ⭐⭐ | KIVI 2-bit 量化 — KV cache 压缩(已实现) |

### 📚 参考目录(INDIRECT / PERIPHERAL)

> **目录**:[`references/`](./references/) —— 40 份 INDIRECT + 6 份 PERIPHERAL 文档,按用途分 3 个子目录。

| 子目录 | 文档数 | 内容 |
|---|---|---|
| [`references/methodology/`](./references/methodology/) | 15 份 | 通用推理方法学(蒸馏 / 小模型调研 / SOTA 评估) |
| [`references/performance/`](./references/performance/) | 28 份 | 性能优化参考(推理加速 / 训练加速 / gap 分析) |
| [`references/historical-architectures/`](./references/historical-architectures/) | 7 份 | 7 轮架构迭代中的 5 份早期方案 + 1 份 PRM 调研(反面参考) |

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

## 🎓学术差异化（AgenticMind 的"首个"价值）

1. **首个公开的 sub-1B 本体论认知推理模型**（vs GCR 仅 KG推理、Logic-LM需大模型）
2. **首个完整"核心内化 +扩展检索 +置信度拒答"sub-1B框架**
3. **首个 sub-1B Temporal/Causal形式化推理实证**（如成功）

---

## 🚀 阅读建议

> 阅读顺序按"项目核心目标 → 技术系列 → 参考目录"的三层结构组织。

| 读者 | 推荐阅读路径 |
|---|---|
| **项目负责人 / 决策者** | 本 README → [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md) → [`architectures/99-final-recommendation.md`](./architectures/99-final-recommendation.md) |
| **架构师 / 技术负责人** | 本 README → [`architectures/`](./architectures/) → [`inference-engine/`](./inference-engine/) → [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md) |
| **AgenticDSL 训练工程师** | [`agenticdsl-training/`](./agenticdsl-training/) → [`architectures/06-metacognitive-closed-loop.md`](./architectures/06-metacognitive-closed-loop.md) |
| **自循环概念研究者** | [`architectures/06-metacognitive-closed-loop.md`](./architectures/06-metacognitive-closed-loop.md) → [`agenticdsl-training/06-vn001-alignment.md`](./agenticdsl-training/06-vn001-alignment.md) → [`references/methodology/small-model-reasoning-survey/05-loop-model-deepdive.md`](./references/methodology/small-model-reasoning-survey/05-loop-model-deepdive.md) |
| **推理引擎集成者** | [`inference-engine/`](./inference-engine/) → [`references/performance/inference-acceleration/inference-technologies/`](./references/performance/inference-acceleration/inference-technologies/) |
| **数据 / 训练工程师** | [`agenticdsl-training/01-training-data-pipeline.md`](./agenticdsl-training/01-training-data-pipeline.md) → [`agenticdsl-training/02-training-algorithms.md`](./agenticdsl-training/02-training-algorithms.md) → [`references/methodology/small-model-reasoning-survey/02-training-strategy-survey.md`](./references/methodology/small-model-reasoning-survey/02-training-strategy-survey.md) |
| **新加入成员** | 本 README(建立全局观)→ [`cognitive-reasoning-model.md`](./cognitive-reasoning-model.md) → 各自方向对应技术系列 |

---

## 🔗 下一步行动

1. ✅ **项目目标已定义**（本 README + `cognitive-reasoning-model.md`）
2. **下一步**：调用 `writing-plans` skill 生成详细实施计划（M1-M2阶段1）
3. **再下一步**：按计划逐步执行阶段1（Logic-LM + GCR 双引擎 MVP）

---

> **核心承诺**：本项目目标严格对齐 AgenticMind "大道至简"哲学 —— **用最小的模型 + 最严谨的形式化推理**，**实现"知道自己不知道"的认知推理能力**，**而不是用更大的模型堆叠能力**。

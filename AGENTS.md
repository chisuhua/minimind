# AGENTS.md — AgenticMind 开发手册

> **本项目代号**：`AgenticMind`
> **仓库目录**：`AgenticMind/`（本地）/ GitHub 仓库名待用户去 GitHub Settings → Rename 中改为 `AgenticMind`
> **GitHub remote（当前）**：`git@github.com:chisuhua/minimind.git`（⚠️ 待 GitHub rename 后 `git remote set-url` 同步）
> **Fork 来源**：开源 MiniMind（`github.com/jingyaogong/minimind`），本仓库作为训练链路载体
> **定位**：训练一个能可靠生成 AgenticDSL 的 LLM，并通过该 DSL 进行多轮认知推理
> **本文件用途**：AI agent / 开发协作者的工作入口与上下文手册

---

## 1. 一句话目标

> **在 <1B 激活参数下，训练一个 LLM，使其能可靠地生成、修复、续写、验证 AgenticDSL，并通过两层次自循环推理实现认知推理 SOTA。**

这个目标分两个层次：

- **顶层目标**：构建 **认知推理模型（Cognitive Reasoning Model）**——LLM 与智能体运行时紧耦合协作，达到 sub-1B 认知推理 SOTA。
- **当前聚焦点**：训练 **AgenticDSL 生成器**——LLM 是 AgenticDSL 程序的主要生产者，这是认知推理模型 L4 层（AgenticDSL 语言）的核心能力。

---

## 2. 与上下游 / 姊妹项目的关系

```
                    HydraForge 仓（C++ 引擎 + AgenticDSL 语言规范）
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
  AgenticDSL 语言层                       ILLMProvider
  (/lib/** /dynamic/** /main/**)          ┌─────┼─────┐
  4 层验证器、ToolRegistry                │     │     │
  TopologicalScheduler                云端 LLM  本地  训练后 LLM
  BudgetController                       │     │     │
                                            ▲     ▲     ▲
                                            │     │     │
        ┌───────────────────────────────────┴─────┴─────┘
        │
   AgenticMind 仓（**本仓库**）
   ├─ MiniMind 训练链路（fork 自 MiniMind）
   ├─ AgenticDSL LLM 训练配方（TR-1 / TR-2 / TR-3）
   └─ HydraForgeBench 评估体系
```

### 2.1 上游依赖：HydraForge

| 资产 | 归属 | 本项目用法 |
|---|---|---|
| AgenticDSL v3.10 规范 | HydraForge | 训练目标语言 |
| 特殊 Token 注册（`<\|agenticdsl_*\|>`、`<\|fim_*\|>`）| HydraForge | 直接消费 |
| Canonical Serializer | HydraForge | 训练数据序列化 |
| 4 层验证器（grammar + signature + execution + task）| HydraForge | SFT 数据硬过滤 |
| `ILLMProvider` 接口 | HydraForge | 训练后模型部署目标 |

**原则**：语言规范变更属于 HydraForge 仓，本仓库不修改 AgenticDSL 语法本身。

### 2.2 姊妹项目：LatentMind

| 维度 | AgenticMind（**本仓库**）| LatentMind |
|---|---|---|
| 推理范式 | **显式 AgenticDSL** 生成 + 执行 | **潜空间（latent space）** 递归推理 |
| 推理机制 | 符号化 DAG 程序 + 4 层验证器 | HRM-Text + GRAM 多轨迹 |
| 可解释性 | **可审计**（DSL 程序 + trace + 求解器输出）| 较弱（latent 状态）|
| 失败模式 | **拒答**（超出 ontology 范围）| 不直接拒答 |

**对应关系**：两个项目在 `LatentMind` README 的"相关项目"表中已建立姊妹关系 —— `AgenticMind | 文本认知推理（独立产品线）`。

### 2.3 下游产出

- **HydraForgeAgenticDSLProvider**：训练后的 LLM 通过该 provider 接入 `ILLMProvider`，作为 HydraForge 默认本地推理后端。
- **HydraForgeBench**：可与业界方案（GPT-4 + constrained、ToolACE-8B、Qwen2.5-Coder）直接对比的评估集。

---

## 3. 核心机制：两层次自循环推理

认知推理 ≠ CoT 推理。本项目的差异化在于 LLM 与智能体运行时通过**共享结构化状态**紧耦合，而非仅靠自然语言 tool call。

### 3.1 自循环推理的两个层次

**层次 A：Confidence-Triggered Loop**

```
LLM 生成 AgenticDSL
    ↓
智能体运行时执行
    ↓
置信度评估（4 层验证器 + Reward Model）
    ↓
不达标？ → 触发知识检索/注入 → 重新推理
达标？   → 返回结果
```

**层次 B：Prefix-Accumulation Step-by-Step Loop**

```
LLM 每次只推理一个步骤
    ↓
上一步推理过程作为 prefix 累积
    ↓
在不改变 prefix 的前提下要求推理下一步骤
    ↓
单步深度 + prefix 累积 → 避免小模型长 context 局限
```

### 3.2 6 层知识形态栈

每层**可独立更新、可单独训练、可独立评测**：

```
┌────────────────────────────────────────┐
│ L6: 基础认知 (Basic Cognition)         │ ← 通用认知能力
├────────────────────────────────────────┤
│ L5: 自然语言描述 (Natural Language)    │ ← 概念解释
├────────────────────────────────────────┤
│ L4: AgenticDSL 语言                    │ ← **本项目当前焦点**
├────────────────────────────────────────┤
│ L3: 文档三元组 (Document Triples)      │ ← 实体-关系抽取
├────────────────────────────────────────┤
│ L2: 逻辑规则 (Logic Rules)             │ ← SWRL / Datalog
├────────────────────────────────────────┤
│ L1: 领域本体 (Domain Ontology)         │ ← OWL/RDFS
└────────────────────────────────────────┘
```

### 3.3 6 类认知推理

| # | 推理类型 | 形式化操作 | ontology 来源 |
|---|---|---|---|
| 1 | 因果 | `cause(X, Y)` / `if X then Y` | ATOMIC, ConceptNet |
| 2 | 依赖 | `depends_on(X, Y)` | 自建逻辑规则 |
| 3 | 相似 | `similarTo(X, Y)` | ConceptNet, WordNet |
| 4 | 对象联系 | `partOf(X, Y)` / `hasProperty` | ConceptNet |
| 5 | 语义关系 | `isA(X, Y)` / `hasA(X, Y)` | WordNet, ConceptNet |
| 6 | 规则/时间 | `before(X, Y)` / `rule(X, Y, Z)` | OWL-Time, SWRL |

### 3.4 "知道自己不知道"机制

每个推理输出附带**校准后的 `confidence ∈ [0,1]`**，触发三类处理：

| 机制 | 触发条件 | 实现方式 |
|---|---|---|
| **拒答** | 超出 ontology 范围 | 模型输出 `out_of_ontology` + 缺口说明 |
| **检索** | 缺口可扩展 ontology 补齐 | Loreto 序列化扩展 ontology → context 注入 |
| **学习提示** | 需用户主动补充 | 输出"需要补充知识 X" + 检索接口 |

---

## 4. 当前聚焦点：AgenticDSL LLM 训练

### 4.1 为什么聚焦 AgenticDSL

HydraForge AgenticDSL 作为 LLM 训练目标语言有 **3 个独特差异化优势**：

1. **三层命名空间 + 强制签名 + 预算控制** —— DSPy/LangGraph/SGLang/PDL 都缺失的工业级约束。
2. **运行时反馈闭环已存在** —— HydraForge C++ 引擎（ILLMProvider + TopologicalScheduler + ToolRegistry + BudgetController）天然提供 4 层验证器。
3. **自举愿景（VN-001）已明确** —— 不需要再造愿景。

### 4.2 三阶段训练路线（14-20 周）

| 阶段 | 时间 | 目标 | 关键产出 |
|---|---|---|---|
| **TR-1 基础生成** | 4-6 周 | LLM 生成 90%+ 格式合规 DSL | 50K SFT + XGrammar EBNF + Tokenizer 锚定 |
| **TR-2 多轮与修复** | 4-6 周 | LLM 基于 trace 续写、修复 DSL | ReSTᴱᴹ 自训练 + PRM 训练 + 修复数据集 |
| **TR-3 质量闭环** | 6-8 周 | pass@1 > 95%，部署为默认后端 | GRPO 精调 + 自动化课程 + 部署到 ILLMProvider |

### 4.3 训练架构 5 层

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Training Data Pipeline (见 01-training-data-pipeline.md) │
│ - SFT data synthesis (50K → 200K)                            │
│ - Execution-driven filtering (DSL runtime as verifier)       │
│ - Curriculum ordering (L1 → L7)                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Layer 2: Training Algorithms (见 02-training-algorithms.md)  │
│ - Cold-start: SFT + RFT (Rejection sampling)                │
│ - Bootstrap: ReSTᴱᴹ (3-5 iterations)                        │
│ - Critic: PRM via OmegaPRM auto-labeling                     │
│ - Search: MCTS + PRM (LATS-style)                            │
│ - Refine: GRPO (no critic, group-relative)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Layer 1: Base Model + Tokenizer (归属 HydraForge 仓)        │
│ - Base: Qwen2.5-Coder-7B-Instruct / Llama-3.1-8B            │
│ - 11 special tokens: <|agenticdsl_open|>, <|subgraph_decl|>, │
│   <|node_def|>, <|fim_*|>, <|agenticdsl_eos|>               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Layer 0: Inference-time Guarantees (见 03-inference-time)    │
│ - Grammar: XGrammar-2 with Structural Tag                    │
│ - Validation: Tree-sitter + markdown_parser                  │
│ - Serving: vLLM/SGLang + XGrammar backend                    │
│ - Speculative: DOMINO/CDSL for 2-12x speedup                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 数据配比（9 阶段管线 + 7 层课程 + 4 层验证）

**任务配比 30/25/20/15/10**（200K SFT 样本）：

| 任务 | 占比 | 规模 |
|---|---|---|
| NL → DSL 生成 | 30% | 60K |
| State-Aware 续写 | 25% | 50K |
| DSL Repair | 20% | 40K |
| DSL → NL 解释 | 15% | 30K |
| DSL Validation | 10% | 20K |

**7 层课程（Skill-It 风格）**：L1 单节点 → L2 参数化 → L3 串行 → L4 并行/分支 → L5 子图嵌套 → L6 错误处理 → L7 长时序多轮。

**4 层验证器（HydraForge runtime）**：
- L1 格式合规（0.4 权重）→ markdown_parser AST 匹配
- L2 Schema 校验（0.2）→ signature_validator + namespace 规则
- L3 沙箱执行（0.3）→ runtime dry-run
- L4 任务级奖励（0.1）→ expected_output 比对

### 4.5 核心训练算法

**主流程**：SFT → RFT（ReSTᴱᴹ） → PRM（OmegaPRM 风格）→ MCTS + PRM（LATS）→ **GRPO**（DeepSeekMath 范式，无 critic）→ SPIN（可选）

**Reward 形状**：
```
reward = 0.4 * format + 0.2 * signature + 0.3 * execution + 0.1 * task
```

**KL penalty**：β=0.04（DeepSeekMath 标准），防止 policy drift。

---

## 5. 技术栈选型（已锁定）

| 类别 | 选型 | 理由 |
|---|---|---|
| **Base 模型** | Qwen2.5-Coder-7B-Instruct / Llama-3.1-8B | 已有 FIM tokens 范本，代码能力验证 |
| **推理后端** | vLLM + XGrammar-2 | vLLM 原生 XGrammar 集成；6× 编译提速 |
| **CFG Engine** | XGrammar-2 | Structural Tag 原生支持 `<\|subgraph_decl\|>` |
| **验证器** | Tree-sitter + markdown_parser | 增量解析 + 错误恢复 |
| **本体库** | ConceptNet5.5 + WordNet + ATOMIC + OWL-Time | 全谱 6 类推理覆盖 |
| **求解器** | Prover9 (FOL) + clingo (ASP) + Z3 (SMT) | Logic-LM 范式标准组件 |
| **AgenticDSL 运行时** | HydraForge C++ 引擎 | 训练数据验证 + 部署目标 |
| **训练框架** | HuggingFace TRL（SFTTrainer / GRPOTrainer）| 生产可用 |
| **课程学习** | Skill-It 风格 online data sampling | +36.5 accuracy（NeurIPS 2023）|

---

## 6. 工作原则

### 6.1 复用优先（不重新造轮子）

- **MiniMind 训练链路**：直接复用 fork 后的 MiniMind 的 SFT/RL/LoRA/DPO 代码，不重写训练循环。
- **AgenticDSL 规范**：HydraForge 仓维护，本仓库只训练消费方。
- **运行时验证**：HydraForge runtime 暴露为 CLI 子命令（`agenticdsl validate/dry-run/eval/trace`），不自己实现 sandbox。
- **特殊 Token**：HydraForge 仓统一注册 vocab surgery，本仓库只设置 `stop_token_ids`。

### 6.2 阶段验证

每个训练阶段都产生可验证的中间产物（对齐 MiniMind "大道至简" 哲学）：

| 阶段 | 验证指标 |
|---|---|
| TR-1 M0 | Tokenizer round-trip test |
| TR-1 M1 | Format compliance > 95%（parser）|
| TR-1 M2 | 基础格式合规 > 90% |
| TR-2 M3 | 任务成功率 > 50% |
| TR-3 M4 | PRM step-level label F1 > 0.7 |
| TR-3 M5 | Pass@1 > 80% |
| TR-3 M6 | HydraForgeAgenticDSLProvider 可作 runtime 默认后端 |
| TR-3 M7 | Agent 可独立生成 Skill（`archive_to` 可成功）|

### 6.3 防 Goodhart 协议（必读）

参考 `docs/agenticdsl-training/05-risk-register.md`：

1. **多层 reward** + **KL penalty β=0.04** + **Human spot-check 周期**
2. **EM iter 限制**：≤5 轮（Singh 2024 警告：每轮必须从 base 重新开始）
3. **修复数据必须用 RL 而非 SFT**（SCoRe ICLR2025 警告：会塌缩）
4. **Format compliance 不应 < 95%**（监控硬门槛）
5. **多样性监控**：unique DSL patterns 不应单调下降

### 6.4 文档同步

- 修改 AgenticDSL 相关训练代码 → 必须同步更新 `docs/agenticdsl-training/` 对应章节。
- 改 baseline、评估指标、里程碑 → 必须同步更新本文档的"当前状态"。
- 新增风险 → 必须记录到 `05-risk-register.md`。

---

## 7. 评测体系（HydraForgeBench）

**8 个评估维度**：

| 维度 | 目标 | 工具 |
|---|---|---|
| 格式合规 | > 99% | Tree-sitter + markdown_parser |
| 签名合规 | > 95% | signature_validator |
| 执行成功率 | > 90% | HydraForge runtime dry-run |
| 任务成功率 | > 85% | expected_output 比对 |
| 多轮稳定性（5 步）| > 80% | runtime 全轨迹执行 |
| 预算遵守 | > 98% | budget_controller |
| 修复能力 | > 90% | runtime 验证 |
| Token 效率 | < 2000 avg | tokenizer metrics |

**3 层基准集**：
- **Set A**（50-100 handcrafted）：Unit-test 式评估，覆盖 L1-L7。
- **Set B**（500-1000）：从生产运行 trace 中收集的端到端基准。
- **Set C**（100-200 adversarial）：Namespace 违规、未声明资源、循环依赖、签名冲突等 stress test。

**与业界对齐**：JSONSchemaBench（Geng 2025）+ BFCL（Gorilla 2024）+ ToolACE（2024）作为参考基线。

---

## 8. 评估指标（顶层认知推理模型）

| # | 指标 | 阶段1 | 阶段2 | 阶段3 |
|---|---|---|---|---|
| 1 | 6 类推理准确率 | ≥60% | ≥70% | ≥80% |
| 2 | ontology 忠实度 | ≥70% | ≥80% | ≥90% |
| 3 | 拒答准确率 | ≥50% | ≥70% | ≥85% |
| 4 | 零样本泛化 | — | ≥30% | ≥50% |
| 5 | 反例鲁棒性 | — | — | ≥60% |
| 6 | 跨领域迁移 | — | — | ≥40% |
| 7 | 对齐率（vs 同尺寸 SOTA）| — | ≥60% | ≥70% |
| 8 | 自循环收敛速度 | ≤ 3 步 | ≤ 2 步 | ≤ 2 步 |
| 9 | Prefix-Accumulation 收益 | ≥ +0% | ≥ +10% | ≥ +20% |
| 10 | Mirror Loop 防御率 | ≥ 90% | ≥ 95% | ≥ 99% |

**对内基线**：sub-1B 实证（T5-large 770M ProofWriter 85.4%、TinyAgent-1.1B ToolRAG 80.06%）。

**不宣称**：
- ❌ 跨尺度通用 SOTA
- ❌ 完全消解 LLM 一次性生成能力
- ❌ HydraForge VN-001 的"自举"愿景（4 阶段宏观路线，属 HydraForge 仓）

---

## 9. 当前状态（2026-06-13）

### 已完成
- ✅ 项目目标定义（`docs/README.md` + `cognitive-reasoning-model.md`）
- ✅ AgenticDSL 训练配方（8 篇文档，TR-1/TR-2/TR-3 14-20 周）
- ✅ 训练数据管线设计（9 阶段 + 7 层课程 + 4 层验证）
- ✅ 训练算法 Recipe（SFT → ReSTᴱᴹ → PRM → MCTS → GRPO → SPIN）
- ✅ 推理时保障（XGrammar-2 + Tree-sitter + vLLM）
- ✅ HydraForgeBench 评估体系（8 维度 × 3 层基准集）
- ✅ 风险登记册（12 风险 + 防 Goodhart 协议）

### 进行中
- 🔨 文档树重组（按"自循环认知推理模型"新目标分级 DIRECT/INDIRECT）

### 待开始
- ⏳ HydraForge runtime CLI 暴露（`tools/cli/validate.cpp` 等）
- ⏳ Special Token 注册（vocab surgery，归属 HydraForge 仓）
- ⏳ EBNF grammar 编写（基于 AgenticDSL v3.10）
- ⏳ TR-1 M0：Tokenizer round-trip test

### 下一步
1. 调用 `writing-plans` skill 生成详细实施计划（TR-1 M0-M2）
2. 协调 HydraForge 仓暴露 runtime CLI
3. 启动 Tokenizer 注册流程

---

## 10. 关键参考文档

### 本仓库
- `docs/README.md` — 顶层项目目标与文档索引
- `docs/cognitive-reasoning-model.md` — 项目目标完整定义
- `docs/agenticdsl-training/README.md` — AgenticDSL 训练综述
- `docs/agenticdsl-training/01-training-data-pipeline.md` — 9 阶段 SFT 数据构造
- `docs/agenticdsl-training/02-training-algorithms.md` — 6 阶段训练 Recipe
- `docs/agenticdsl-training/03-inference-time-guarantees.md` — 推理时 Grammar 约束
- `docs/agenticdsl-training/04-evaluation-benchmark.md` — HydraForgeBench 设计
- `docs/agenticdsl-training/05-risk-register.md` — 风险登记册
- `docs/agenticdsl-training/06-vn001-alignment.md` — 与 HydraForge VN-001 对齐
- `docs/architectures/06-metacognitive-closed-loop.md` — 元认知闭环（自循环核心）
- `docs/inference-engine/` — 自循环推理的关键基础设施

### 上游 / 姊妹项目
- `HydraForge` 仓 `/docs/agenticdsl/vision/01-self-bootstrapping-vision.md` — VN-001 自举愿景
- `HydraForge` 仓 `/docs/agenticdsl/implementation/self-bootstrapping-path.md` — BOOT-001 实施路径
- `HydraForge` 仓 `/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md` — AgenticDSL 语言演进
- `LatentMind/AGENTS.md` — 姊妹项目手册（潜空间推理范式）

### 关键 SOTA 文献
- Singh et al. TMLR 2024 — ReSTᴱᴹ（EM iter 必须从 base 重新开始）
- Shao et al. 2024 — GRPO（DeepSeekMath，无 critic）
- Luo et al. DeepMind 2024 — OmegaPRM（自动 process supervision）
- Kumar et al. ICLR 2025 — SCoRe（修复数据必须 RL 训练）
- Gao et al. ICML 2023 — Scaling Laws for Reward Model Overoptimization
- Geng et al. 2025 — JSONSchemaBench（arXiv:2501.10868）
- Qin et al. NeurIPS 2024 — Gorilla / BFCL

---

## 11. 一句话承诺

> **用最小模型 + 最严谨形式化推理 + 紧耦合运行时反馈，实现"知道自己不知道"的认知推理能力，而不是用更大的模型堆叠能力。**

---

## 12. 已锁定关键事实（单一真源）

> 本节集中记录所有数值化、可校验的项目决定，作为文档之间的单一真源（Single Source of Truth）。
> 若其他文档与本节冲突，**以本节为准**，并在 `docs/README.md` 提交流程修订。
> ⚠️ 标注 = 当前**存在文档间不一致**，需用户/团队决策。

### 12.1 训练路线时长

| 阶段 | 时长 | 累计 | 来源 |
|---|---|---|---|
| TR-1 基础生成 | 4-6 周 | 4-6 周 | `agenticdsl-training/README.md` §3 |
| TR-2 多轮与修复 | 4-6 周 | 8-12 周 | 同上 |
| TR-3 质量闭环 | 6-8 周 | 14-20 周 | 同上 |

### 12.2 数据规模与配比（200K SFT 样本）

| 任务 | 占比 | 规模 |
|---|---|---|
| NL → DSL 生成 | 30% | 60K |
| State-Aware 续写 | 25% | 50K |
| DSL Repair | 20% | 40K |
| DSL → NL 解释 | 15% | 30K |
| DSL Validation | 10% | 20K |

来源：`agenticdsl-training/01-training-data-pipeline.md` §1。

### 12.3 7 层课程（Skill-It 风格）

L1 单节点 → L2 参数化 → L3 串行 → L4 并行/分支 → L5 子图嵌套 → L6 错误处理 → L7 长时序多轮。
来源：`agenticdsl-training/01-training-data-pipeline.md` §3。

### 12.4 4 层验证器权重（HydraForge runtime）

```
final reward = 0.4 * L1_format + 0.2 * L2_signature + 0.3 * L3_execution + 0.1 * L4_task
```

来源：`agenticdsl-training/01-training-data-pipeline.md` §4 + `02-training-algorithms.md` §5。

### 12.5 训练算法超参

| 参数 | 值 | 来源 |
|---|---|---|
| KL penalty β | 0.04 | DeepSeekMath 标准 |
| GRPO G（每 prompt 采样数）| 16-32 | `02-training-algorithms.md` §4.2 |
| GRPO learning rate（精调）| 1e-6 | 同上 |
| PPO/GRPO clip ε | 0.2 | 同上 |
| ReSTᴱᴹ EM iter 上限 | 5 | Singh 2024 警告 |
| Format compliance 监控下限 | 95% | `05-risk-register.md` §2.2 |

### 12.6 特殊 Token 注册（11 个，归属 HydraForge 仓）

| Token | 用途 |
|---|---|
| `<\|agenticdsl_open\|>` | DSL 块开始 |
| `<\|agenticdsl_close\|>` | DSL 块结束 |
| `<\|subgraph_decl\|>` | 子图声明头 |
| `<\|node_def\|>` | 节点定义开始 |
| `<\|inja_expr_open\|>` | `{{` 模板起始 |
| `<\|inja_expr_close\|>` | `}}` 模板结束 |
| `<\|fim_prefix\|>` | FIM prefix |
| `<\|fim_middle\|>` | FIM middle |
| `<\|fim_suffix\|>` | FIM suffix |
| `<\|fim_pad\|>` | FIM padding |
| `<\|agenticdsl_eos\|>` | DSL 生成结束 |

来源：`agenticdsl-training/03-inference-time-guarantees.md` §5.2。

### 12.7 HydraForgeBench 评估目标（8 维度）

| 维度 | 目标 |
|---|---|
| 格式合规 | > 99% |
| 签名合规 | > 95% |
| 执行成功率 | > 90% |
| 任务成功率 | > 85% |
| 多轮稳定性（5 步）| > 80% |
| 预算遵守 | > 98% |
| 修复能力 | > 90% |
| Token 效率 | < 2000 avg |

来源：`agenticdsl-training/04-evaluation-benchmark.md` §1。

### 12.8 3 层基准集规模

| 集合 | 规模 | 类型 |
|---|---|---|
| Set A | 50-100 | Unit-test 手写 |
| Set B | 500-1000 | 生产 trace |
| Set C | 100-200 | Adversarial |

来源：同上。

### 12.9 顶层认知推理指标（10 项）

详见 §8 本文档，阶段 1-3 阈值见 `docs/README.md` 评测体系表。

### 12.10 ⚠️ 待决策项（文档间不一致）

#### F-01：Base 模型选择（`<1B` vs `7B`）

| 文档 | Base 模型 | 备注 |
|---|---|---|
| `docs/README.md`（认知推理模型目标）| **Qwen2-0.5B**（0.5B）| 对齐 <1B 目标 |
| `docs/agenticdsl-training/README.md`（AgenticDSL 训练）| **Qwen2.5-Coder-7B-Instruct / Llama-3.1-8B** | SFT 训练实际选型 |
| AGENTS.md（本文档 §5）| **Qwen2.5-Coder-7B-Instruct / Llama-3.1-8B** | 倾向训练文档 |

**两种合理解读**：
- (A) 训练阶段用 7B/8B（数据更易学），部署时通过剪枝/蒸馏压到 <1B
- (B) 整个 pipeline 强制 <1B（与项目顶层目标一致）

**待决策**。

#### F-02：模型命名（`HydraForge-AgenticDSL-X` vs `AgenticMind-X`）

| 文档 | 命名 |
|---|---|
| `agenticdsl-training/*`（7 份）| `HydraForge-AgenticDSL-7B-v1/v2/v3`、`PRM-1B` |
| AGENTS.md（本文档）| 未明确命名 |

**推荐**：模型是 AgenticMind 训练的，命名应为 `AgenticMind-7B-v1/v2/v3`、`AgenticMind-PRM-1B`。Provider 仍是 `HydraForgeAgenticDSLProvider`（在 HydraForge 仓实现）。
**待决策**。

#### F-03：里程碑 M3-M6 时间表（`agenticdsl-training/README.md` vs `06-vn001-alignment.md`）

| 里程碑 | TR-1 M0 | TR-1 M1 | TR-1 M2 | TR-2 M3 | TR-3 M4 | TR-3 M5 | TR-3 M6 | TR-3 M7 |
|---|---|---|---|---|---|---|---|---|
| README（training）| W1-2 | W3-4 | W5-6 | **W7-9** | **W10-11** | **W12-16** | **W17-18** | W18-20 |
| 06-vn001-alignment | W1-2 | W3-4 | W5-6 | **W7-8** | **W11-12** | **W13-16** | **W18** | W19-20 |

**差异来源**：两处都标"自己规划"，但未对齐。推荐以 `agenticdsl-training/README.md` 为准（这是训练团队文档）。
**待决策**。

---

**版本**：v1.0
**最后更新**：2026-06-13
**Owner**：AgenticMind 训练团队
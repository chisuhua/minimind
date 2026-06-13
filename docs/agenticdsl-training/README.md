# AgenticDSL LLM 训练 — 综述与索引

> **文档 ID**: LLMTRN-001-INDEX
> **生成日期**: 2026-06-10
> **状态**: 草案 v1.0
> **配套文档**（语言演进，DSL 规范层面）:
> - HydraForge 仓 `/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md`

---

## 0. 文档范围与定位

本文档树聚焦 **"如何训练一个 LLM 可靠地生成、修复、续写、验证 AgenticDSL"**——是 AgenticDSL 语言演进提案（HydraForge 仓）的**训练侧**配套。

**目标读者**：AgenticMind 团队的 LLM 训练工程师、SFT/RL 工程师、数据工程师。

**不在本文档范围**：
- AgenticDSL 语言演进提案 → 见 HydraForge 仓
- C++ 引擎实现 → 见 `docs/adr/`、`docs/specs/`
- 运行时验证器实现 → 见 HydraForge 仓

---

## 1. 核心结论

HydraForge 的 AgenticDSL **完全可以作为 LLM 的训练目标语言**，并且相对于业界方案具有 **3 个独特差异化优势**：

1. **三层命名空间架构**（`/lib/**`、`/dynamic/**`、`/main/**`）+ **强制签名系统** + **预算控制** —— 这是 2024-2025 年所有 agent DSL/IR（DSPy、LangGraph、SGLang、FlowAgent/PDL、AgentSPEX）都缺失的工业级约束。
2. **运行时反馈闭环已经存在** —— HydraForge 的 C++ 引擎（ILLMProvider + Topological Scheduler + ToolRegistry + BudgetController + OpenTelemetry Trace）天然提供执行验证器，**这是 99% 的 agent DSL 项目缺乏的关键基础设施**。
3. **自举愿景（VN-001）+ 4 阶段路线图**已经明确 —— 不需要再造愿景。

---

## 2. 与初步分析的关键差异

| 维度 | 初步分析判断 | **HydraForge 实际状态** |
|---|---|---|
| DSL 形态 | "YAML-like 缩进" | ✅ Markdown + 嵌入式 YAML，有显式 `### AgenticDSL` 头 |
| 无显式边界标记 | ❌ | ✅ 已有显式头/尾围栏 |
| 执行语义隐含 | ❌ | ✅ 已显式编码：`type`/`next`/`wait_for`/`merge_strategy` |
| 无签名系统 | ❌ | ✅ `/lib/**` 必须 `signature:` |
| 缺少动态子图 | ❌ | ✅ `generate_subgraph` + `/dynamic/**` |
| 多轮状态管理缺失 | ❌ | ✅ `LayeredContext` (L1-L5) + `expected_output` |

详细对照表见 `06-vs-initial-analysis.md`。

---

## 3. 推荐训练路线

| 阶段 | 时间 | 目标 | 关键产出 |
|---|---|---|---|
| **TR-1: 基础生成** | 4-6 周 | LLM 可生成 90%+ 格式合规的 AgenticDSL | SFT 数据集 v1（~50K）+ XGrammar EBNF + Tokenizer 锚定 |
| **TR-2: 多轮与修复** | 4-6 周 | LLM 可基于执行 trace 续写、修复 DSL | ReSTᴱᴹ 自训练 + PRM 训练 + 修复数据集 |
| **TR-3: 质量闭环** | 6-8 周 | LLM 成为 HydraForge 默认推理后端，DSL pass@1 > 95% | GRPO 精调 + 自动化课程 + 部署到 ILLMProvider |

**累计**：14-20 周。

---

## 4. 文档结构

| 编号 | 文档 | 内容 |
|---|---|---|
| **00** | `README.md`（本文）| 综述、决策清单、与 HydraForge 仓的边界 |
| **01** | `training-data-pipeline.md` | 9 阶段数据生成管线、200K SFT 数据、7 层课程、4 层验证器 |
| **02** | `training-algorithms.md` | 6 阶段自训练 Recipe（ReSTᴱᴹ → OmegaPRM → MCTS → GRPO → SPIN） |
| **03** | `inference-time-guarantees.md` | XGrammar-2 + vLLM + Tree-sitter 推理栈、约束解码 |
| **04** | `evaluation-benchmark.md` | HydraForgeBench 设计、8 个评估维度 |
| **05** | `risk-register.md` | 12 个关键风险 + 防 Goodhart 协议 |
| **06** | `vn001-alignment.md` | 与 VN-001 自举愿景的对齐路径 |
| **07** | `07-vs-initial-analysis.md` | 与初步分析的差异对照 |

---

## 5. 训练架构三层

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Training Data Pipeline (见 01)                     │
│ - SFT data synthesis (50K → 500K)                           │
│ - Execution-driven filtering (DSL runtime as verifier)       │
│ - Curriculum ordering (L1 → L7)                             │
│ - Deduplication / decontamination                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Layer 2: Training Algorithms (见 02)                         │
│ - Cold-start: SFT + RFT (Rejection sampling)                │
│ - Bootstrap: ReSTᴱᴹ (3-5 iterations)                        │
│ - Critic: PRM via Math-Shepherd/OmegaPRM auto-labeling       │
│ - Search: MCTS + PRM (LATS-style)                            │
│ - Refine: GRPO (no critic, group-relative)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Layer 1: Base Model + Tokenizer (见 HydraForge 仓)          │
│ - Base: Qwen2.5-Coder-7B-Instruct / Llama-3.1-8B            │
│ - Special tokens: <|agenticdsl_open|>, <|subgraph_decl|>, ...│
│ - FIM tokens: <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>│
│ - Anchor tokens: stop_token_ids                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ Layer 0: Inference-time Guarantees (见 03)                   │
│ - Grammar: XGrammar-2 with Structural Tag                    │
│ - Validation: Tree-sitter + custom DSL validator            │
│ - Serving: vLLM / SGLang with XGrammar backend              │
│ - Speculative: DOMINO/CDSL for 2-12x speedup                │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 决策检查清单

实施前必须回答以下问题：

- [ ] **目标场景中 LLM 生成 JSON/Code 的结构合法率是否 < 90%？**
  - ✅ 是的（GPT-4 在 BFCL 上也仅 89% coverage），AgenticDSL 有明确改进空间。
- [ ] **是否愿意投入资源构建 SFT 数据集 + 约束解码器 + 执行 Sandbox 三件套？**
  - ⚠️ HydraForge 已有 runtime sandbox，需要补齐前两件。
- [ ] **DSL 是否能在 3 个月内产出比现有方案高 20%+ 的端到端任务成功率？**
  - ✅ ToolACE-8B 已是 GPT-4 级别（7B 模型击败 1T+ 参数闭源模型），预期可达。
- [ ] **是否接受将 AgenticDSL 定位为"训练/推理 IR"而非"人类编写语言"？**
  - ✅ 是的（v3.10 已是 AI-Native 定位）。
- [ ] **VN-001 自举路线图是否已批准并有 owner？**
  - ✅ BOOT-001 (实施路径) 已批准 (2026-05-22)，与本文训练路线高度互补。

---

## 7. 与 HydraForge 仓的边界

| 内容 | 归属 | 原因 |
|---|---|---|
| **AgenticDSL 语法演进**（v3.11 锚点、`<\|subgraph_decl\|>`）| HydraForge 仓 `llm-training-design/` | 语言规范变更属于 AgenticDSL 语言层 |
| **特殊 Token 注册**（vocab surgery）| HydraForge 仓 | Tokenizer 是语言基础设施的一部分 |
| **Canonical Serializer** | HydraForge 仓 | 序列化是解析的镜像，属于语言层 |
| **FIM 数据格式定义** | HydraForge 仓 | 是 token + 序列化组合 |
| **9 阶段数据生成管线** | AgenticMind 仓 `01-training-data-pipeline.md` | 是训练数据工程 |
| **6 阶段训练算法 Recipe** | AgenticMind 仓 `02-training-algorithms.md` | 是训练算法 |
| **XGrammar + vLLM 推理栈** | AgenticMind 仓 `03-inference-time-guarantees.md` | 是推理时基础设施（AgenticMind 部署视角） |
| **HydraForgeBench 设计** | AgenticMind 仓 `04-evaluation-benchmark.md` | 是评估流程 |
| **风险登记册** | AgenticMind 仓 `05-risk-register.md` | 是训练项目风险 |
| **VN-001 对齐** | AgenticMind 仓 `06-vn001-alignment.md` | 是训练路线与 HydraForge 自举的整合 |

---

## 8. 时间线与里程碑

| 里程碑 | 时间 | 关键产出 | 成功标准 |
|---|---|---|---|
| **M0** | 第 1-2 周 | EBNF + Tokenizer + Vocabulary surgery | Tokenizer round-trip test 通过 |
| **M1** | 第 3-4 周 | 50K SFT + 4 层验证器 | Format compliance > 95% (parser) |
| **M2** | 第 5-6 周 | HydraForge-AgenticDSL-7B-v1 | 基础格式合规 > 90% |
| **M3** | 第 7-9 周 | HydraForge-AgenticDSL-7B-v2 | 任务成功率 > 50% |
| **M4** | 第 10-11 周 | PRM-1B model | Step-level label F1 > 0.7 |
| **M5** | 第 12-16 周 | HydraForge-AgenticDSL-7B-v3 | Pass@1 > 80% |
| **M6** | 第 17-18 周 | HydraForgeAgenticDSLProvider | 可作 runtime 默认后端 |
| **M7** | 第 18-20 周 | Agent 可独立生成 Skill | archive_to 可成功 |

---

**文档版本**: v1.0
**Owner**: AgenticMind 训练团队
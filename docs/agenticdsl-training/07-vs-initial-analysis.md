# 与初步分析的差异对照

> **文档 ID**: LLMTRN-001-DIFF
> **生成日期**: 2026-06-10
> **关联**:
> - 综述: [`README.md`](README.md)
> - HydraForge DSL 演进: `/workspace/project/HydraForge/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md`

---

## 1. 初步分析回顾

用户提供的初步分析对 AgenticDSL 现状有 **6 处明显失真**，源于把 AgenticDSL 误读为"原始 YAML 配置"，而实际上它已经是一个 **LLM-aware、契约化、可验证、动态可生长的 DAG 语言**。

本文档对初步分析做了精确的差异对照，便于读者理解 **HydraForge AgenticDSL 的实际能力** vs **初步分析的判断**。

---

## 2. 详细差异对照表

| 维度 | 初步分析判断 | **HydraForge 实际状态（基于代码与 v3.10 规范核实）**| 差异说明 |
|---|---|---|---|
| **DSL 形态** | "YAML-like 缩进" | ✅ Markdown + 嵌入式 YAML，有 `### AgenticDSL '/path'` 头与 `--- BEGIN/END AgenticDSL ---` 围栏 | 初步分析误读为纯 YAML |
| **无显式边界标记** | ❌ | ✅ 已有显式头/尾围栏 | FIM 切分锚点天然存在，无需新增 |
| **执行语义隐含** | ❌ | ✅ 已显式编码：节点 `type`、`next`、`wait_for`、`merge_strategy`、`branches`、`on_failure` | 不需要重构控制流 |
| **无签名系统** | ❌ | ✅ `/lib/**` 子图必须声明 `signature:`（`inputs`/`outputs`/`schema`），启动时校验 | 初步分析未识别 |
| **无 Schema 校验** | ❌ | ✅ 已定义 `tool_schema`、`permissions` 交集、`signature_validation` 三档（strict/warn/ignore）| 4 层验证器已就绪 |
| **缺少动态子图** | ❌ | ✅ `generate_subgraph` 节点类型 + `/dynamic/**` 命名空间 | 已支持运行时生成子图 |
| **Fork/Join 是隐式** | ❌ | ✅ `fork`/`join`/`wait_for` (any_of/all_of) 显式节点 | 多路径表达已就位 |
| **多轮状态管理缺失** | ❌ | ✅ `LayeredContext` (L1-L5) + `expected_output` + `archive_to` | 已具备结构化状态 |

---

## 3. 初步分析的核心建议采纳情况

### 3.1 建议 1：引入 XML 风格边界标签

**初步分析原文**：
> 引入 XML 风格边界标签：将顶层结构和关键块用特殊 Token 包裹

**实际情况**：
- ✅ 已有 `### AgenticDSL '/path'` 头
- ✅ 已有 `--- BEGIN/END AgenticDSL ---` 围栏
- ❌ 不需要引入 `<agent>` XML 风格（破坏可读性，与现有工作流冲突）

**采纳建议**：
- **不引入** XML 锚点（破坏现有 v3.10 兼容性）
- **新增** special tokens `<|agenticdsl_open|>`、`<|subgraph_decl|>` 等（作为 tokenizer 层抽象，不破坏语法）
- 详见 HydraForge 仓 `/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md` §2.2.1

### 3.2 建议 2：扁平化嵌套结构

**初步分析原文**：
> 扁平化嵌套结构：将深层嵌套改为 ID 引用 + 独立定义

**实际情况**：
- ✅ 已有 ID 引用 + 独立定义（每个节点有唯一 id，通过 `next`/`wait_for` 引用）
- ⚠️ YAML 嵌套仍是隐式（节点定义中嵌套 `arguments`、`assign` 等字段）

**采纳建议**：
- **保留** 节点嵌套（因为 `arguments` 等字段必须内联）
- **不引入** 强制扁平化（破坏可读性）
- 用 EBNF grammar 约束嵌套深度即可

### 3.3 建议 3：强制拓扑排序序列化

**初步分析原文**：
> 强制拓扑排序序列化：要求所有 `<step>` 按依赖顺序声明

**实际情况**：
- ⚠️ 现有 `nodes_list` 按声明顺序，但不强制依赖顺序
- ✅ `TopoScheduler::build_dag()` 处理依赖关系

**采纳建议**：
- **保留** 现有声明顺序（对人类可读性更友好）
- 训练数据 canonical 序列化时**不强制**拓扑排序（避免数据稀疏化）

### 3.4 建议 4：显式标注控制流

**初步分析原文**：
> 显式标注控制流：不要依赖解释器推断并发/条件

**实际情况**：
- ✅ 已显式编码：`type`/`next`/`wait_for`/`branches`/`merge_strategy`
- ❌ 不需要新增原语

**采纳建议**：
- **完全采纳**（已经实现）
- 无需新增原语，直接利用现有结构

### 3.5 建议 5：训练数据 4:3:2:1 配比

**初步分析原文**：
> 仅靠"NL → AgenticDSL"单向生成不足以让模型真正理解。需构造联合训练数据

**实际情况**：
- ✅ 建议方向正确
- ⚠️ 配比需要调整为 30/25/20/15/10（基于 2024-2025 SOTA 实践）

**采纳建议**：
- **采纳方向**，但调整配比
- 详见 AgenticMind 仓 [`01-training-data-pipeline.md`](01-training-data-pipeline.md) §1

### 3.6 建议 6：注册特殊 Token

**初步分析原文**：
> 将 `<agent>`, `</agent>`, `<step>`, `<fork>`, `<cond>` 等高频结构标记加入词表

**实际情况**：
- ✅ 方向完全正确
- ⚠️ 但 **不要引入 XML 风格的 `<agent>`**（与 Markdown 头冲突）

**采纳建议**：
- **采纳方向**，但用 Qwen2.5-Coder 风格的 `<|xxx|>` 范本
- 11 个特殊 token 设计：`<|agenticdsl_open|>`、`<|subgraph_decl|>`、`<|node_def|>`、`<|inja_expr_open|>`、`<|inja_expr_close|>`、`<|fim_prefix|>`、`<|fim_middle|>`、`<|fim_suffix|>`、`<|fim_pad|>`、`<|agenticdsl_eos|>`
- 详见 HydraForge 仓 `/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md` §2.2.1

### 3.7 建议 7：约束解码兜底

**初步分析原文**：
> 即使训练充分，推理时仍应使用 CFG/PDA 强制生成符合 AgenticDSL 文法的序列

**实际情况**：
- ✅ 完全正确
- ✅ HydraForge runtime 可作为 verifier

**采纳建议**：
- **完全采纳**
- 推荐 **XGrammar-2** + Structural Tag（2026 SOTA，arXiv:2601.04426）
- 详见 AgenticMind 仓 [`03-inference-time-guarantees.md`](03-inference-time-guarantees.md)

### 3.8 建议 8：确定性序列化

**初步分析原文**：
> 必须固定一种规范形式（如字典序），避免相同语义产生不同序列稀释训练信号

**实际情况**：
- ✅ 完全正确

**采纳建议**：
- **完全采纳**
- 设计 Canonical Serializer：2 空格缩进、字典序键序、双引号字符串、显式 null
- 详见 HydraForge 仓 `/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md` §2.2.2

---

## 4. 初步分析未识别的 HydraForge 优势

### 4.1 已有完整 runtime 反馈闭环

**初步分析未提及**：
- HydraForge C++ 引擎天然提供多层执行反馈
- 这是 99% 的 agent DSL 项目缺乏的关键基础设施

**HydraForge 资产**：
| 资产 | 价值 |
|---|---|
| ILLMProvider 流式接口（C1 后）| 统一的 LLM 访问层 |
| Topological Scheduler | DAG 确定性调度 |
| ToolRegistry | 工具注册与调用 |
| BudgetController | 预算强制执行 |
| TraceRecord (OpenTelemetry 兼容) | 可观测性 |
| 4 层验证器（grammar + signature + execution + task）| 训练数据硬过滤 |
| LayeredContext (L1-L5) | 状态隔离 |
| SessionRegistry | 多会话管理 |
| Permission 交集 | 最小权限沙箱 |

### 4.2 自举愿景 + 4 阶段路线图

**初步分析未提及**：
- VN-001 自举愿景（4 阶段：硬编码参数 → 可编程策略 → 质量评估 → 持续自进化）
- BOOT-001 实施路径（已批准）

**意义**：训练路线不是孤立项目，而是嵌入到 HydraForge 自举路线图中的关键能力。

### 4.3 已有标准库子图（推理 + 记忆 + 对话）

**初步分析未提及**：
- `/lib/inference/` 已就位（engine、model、session）
- `/lib/memory/` 已规划（state、kg、vector、profile）
- `/lib/conversation/` 已规划（start_topic、switch_role、meeting）

**意义**：训练数据可以直接从标准库子图生成（`/lib/**` 强制签名 → 训练数据 schema 完备）。

---

## 5. 初步分析的隐藏假设错误

### 5.1 假设 1："DSL 是纯 YAML"

**错误**：HydraForge AgenticDSL 是 Markdown + 嵌入式 YAML + 显式围栏。

**后果**：如果按初步分析的"引入 XML 风格"建议，会破坏 v3.10 兼容性。

### 5.2 假设 2："v3.10 是早期实验版本"

**错误**：v3.10 是当前参考执行器 v1.0 对应的稳定规范版本，已通过 30+ ADR 审批。

**后果**：初步分析的许多"建议添加"功能实际上已经存在。

### 5.3 假设 3："需要从零设计控制流"

**错误**：fork/join/wait_for/merge_strategy 已经显式编码。

**后果**：不需要重新设计控制流。

### 5.4 假设 4："没有运行时验证器"

**错误**：4 层验证器（namespace、signature、permission、execution）已就绪。

**后果**：SFT 数据的硬过滤可以直接用 runtime，不需要重新实现。

---

## 6. 关键数字对比

### 6.1 初步分析未提及的关键 SOTA 数字

| 项目 | 关键数字 | 来源 |
|---|---|---|
| **ToolACE-8B** | BFCL SOTA ≈ GPT-4，**7B 模型击败 1T+ 参数闭源** | [arXiv:2409.00920](https://arxiv.org/abs/2409.00920) |
| **Gorilla-7B** | zero-shot 比 GPT-4 +20.43% | [arXiv:2305.15334](https://arxiv.org/abs/2305.15334) |
| **AlphaCode-41B** | 99% 样本被公开测试过滤，34.2% 比赛问题解决 | [Science 2022](https://arxiv.org/abs/2203.07814) |
| **Voyager** | 3.3× unique items, 2.3× distance, 15.3× tech tree milestones | [NeurIPS 2023](https://arxiv.org/abs/2305.16291) |
| **XGrammar-2** | 6× 编译提速 vs XGrammar-1, 80× compilation speedup at 500 tools | [arXiv:2601.04426](https://arxiv.org/abs/2601.04426) |
| **GRPO (DeepSeekMath)** | MATH 46.8% → 51.7%（GRPO 提升段）| [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| **JSONSchemaBench Guidance** | 96% coverage, 98% compliance on GlaiveAI | [arXiv:2501.10868](https://arxiv.org/abs/2501.10868) |

### 6.2 HydraForge 现有能力数字

| 资产 | 数量 | 价值 |
|---|---|---|
| ADRs（架构决策记录）| 36 个 | 工业级设计决策 |
| 文档（specs/agenticdsl）| 60+ 篇 | 完整的语言规范 |
| `/lib/**` 子图 | 7 个 | 标准库基础 |
| 节点类型 | 10+ 种 | 丰富的执行原语 |
| 错误码 | 7+ 种 | 形式化错误处理 |

---

## 7. 最终建议路径调整

### 7.1 初步分析的"分阶段 V1/V2/V3"路径

**初步分析**：
- V1: 快速验证（5K SFT）
- V2: 结构优化（Repair/Validation 数据）
- V3: 语义 Grounding（执行 Trace 数据）

**调整建议**：
- V1 不需要"引入 XML 锚点"——已有 Markdown 头与围栏
- V2 不需要"扁平化嵌套"——保留现有结构
- V3 不需要"Schema 注入"——已有 signature + permissions
- **新增** TR-1/TR-2/TR-3 训练阶段（与 HydraForge 自举阶段 0/1/2 对齐）

### 7.2 调整后的路径

| 阶段 | 时间 | 目标 | 关键交付 |
|---|---|---|---|
| **TR-1** | 4-6 周 | 基础生成能力 | SFT v1 + XGrammar + Tokenizer 锚定 |
| **TR-2** | 4-6 周 | 多轮与修复 | ReSTᴱᴹ + PRM + 修复数据集 |
| **TR-3** | 6-8 周 | 质量闭环 | GRPO + 自动化课程 + 部署到 ILLMProvider |

**累计**：14-20 周，对齐 VN-001 阶段 0→1→2 的核心能力目标。

---

## 8. 总结

### 关键修正（vs 初步分析）

1. **DSL 不是纯 YAML**：是 Markdown + 嵌入式 YAML + 显式围栏
2. **v3.10 不是早期版本**：是已审批的稳定规范
3. **不需要引入 XML 锚点**：已有 Markdown 头与围栏
4. **不需要扁平化嵌套**：已有 ID 引用 + 独立定义
5. **不需要新增控制流原语**：fork/join/wait_for 已显式编码
6. **不需要新增 Schema 校验**：signature + permissions 已就绪
7. **HydraForge 已有完整 runtime**：C++ 引擎提供多层验证器
8. **已有自举愿景与路线图**：VN-001 + BOOT-001 已批准

### 初步分析正确的部分

1. ✅ 引入特殊 Token（用 `<|xxx|>` 范式而非 `<agent>` XML）
2. ✅ 4:3:2:1 多任务配比（调整为 30/25/20/15/10）
3. ✅ DSL Repair 数据（但必须用 RL 而非 SFT，SCoRe 警告）
4. ✅ 约束解码兜底（用 XGrammar-2 而非自定义 CFG）
5. ✅ 确定性序列化（用 Canonical Serializer）
6. ✅ 渐进式复杂度（用 Skill-It 风格课程学习）
7. ✅ 修复数据 15-20% 占比

### 最终定位

**HydraForge AgenticDSL 应作为 LLM 训练的"Agent 领域 WASM"** —— 不取代 GPT-4/Claude，而是成为 HydraForge 生态的专用推理后端。

训练路线（AgenticMind 仓）+ 语言演进（HydraForge 仓）+ 自举愿景（VN-001）三位一体，14-20 周内可完成端到端闭环。

---

**文档版本**: v1.0
**Owner**: AgenticMind + HydraForge 协同
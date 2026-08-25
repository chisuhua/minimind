# AGENTS.md — context-management/

## OVERVIEW

会话抽取系统设计文档（v0.2, 2026-08-25）;13 字段 schema 人工真源侧，运行时编排行为规范，P0 原型任务清单。

---

## WHERE TO LOOK

| 文档 | 主题 | 影响代码路径 | 影响决策 |
|---|---|---|---|
| `README.md` | v0.2 综述 + 角色重定位 | — | 文档结构 |
| `mvp-schema.md` | **13 字段人工 schema 单一真源** | `agenticmind/extraction/schemas.py` | schema 字段定义 |
| `architecture.md` | Python v1 → HydraForge v2 编排演进 | `agenticmind/orchestrator/` | 运行时架构 |
| `p0-prototype-tasks.md` | 3-4 周端到端验证任务清单 | `agenticmemory_training/` | P0 执行计划 |

---

## DOC-CODE 真源对照表

| 维度 | Docs 侧（本文档目录）| Code 侧 |
|---|---|---|
| **Schema 字段定义** | `mvp-schema.md` | `agenticmind/extraction/schemas.py` |
| **Validator 逻辑** | — | `agenticmind/extraction/validator.py` |
| **隐私检测** | — | `agenticmind/extraction/privacy.py` |
| **编排接口** | `architecture.md` | `agenticmind/orchestrator/interface.py` |

> 原则：`mvp-schema.md` 是 docs 侧 schema 真源，`schemas.py` 是 code 侧真源。两处必须同步，修改必须走 F-04 流程。

---

## v0.2 决策要点 (F-04, 2026-08-24)

| 维度 | 决策 |
|---|---|
| Schema 融合边界 | Schema 层分离 + 数据集层联合 + 模型层多任务 |
| 可参与涌现锚定字段 | 仅 `entities` 的 9 种类型；其余 12 字段只从对话语料产生 |
| 模型选型 | Qwen3-0.6B base + 双 LoRA（`session_extract` / `memory_extract`）|
| 运行时抽取实现 | interim: 教师 API + 规则 + 4 级降级；统一模型就绪后替换 |
| 消费方 | 统一：同一个智能体消费同一个模型的两种 task 输出 |

---

## CROSS-REFERENCE

**与 `docs/agenticmemory_training/08b-seed-schema-fusion.md` 的边界**

- `mvp-schema.md` 的 13 字段（人工定义）与 `agenticmemory_training/` 涌现 schema 在 schema 层互不污染
- 两者在训练数据 dataset 层联合（通过 task tag 区分）
- 模型层多任务训练（一个 base model + 两个 LoRA adapter）
- 详见 `08b-seed-schema-fusion.md` §边界定义

**与 `docs/agenticdsl-training/01-training-data-pipeline.md` 的衔接**

- P0 产出的 session 抽取数据 → 作为 `agenticmemory_training/` 的 seed data 输入
- P1 训练使用 Qwen3-0.6B base + 双 LoRA，衔接 DSL 生成训练管线

---

## CONVENTIONS

1. **改 schema 字段**：必须同步更新 `agenticmind/extraction/schemas.py`（双真源原则）
2. **v0.X 升级**：必须走 PROPOSAL → DECISION → UPDATE 流程（禁止直接覆盖）
3. **P0 执行顺序**：完成 P0 原型任务后才进入 P1 训练阶段
4. **Pending docs**：5 份待创建文档（见 NOTES）完成前不得声称 v1.0 完成

---

## ANTI-PATTERNS

- **NEVER** 在本目录定义 schema 字段语义（`mvp-schema.md` 是唯一真源，字段语义定义禁止重复）
- **NEVER** 直接修改 P0 已决策项（必须经过 F-04 流程方可变更）
- **NEVER** 在本目录放置 Python 代码（代码路径归属 `agenticmind/extraction/` 和 `agenticmind/orchestrator/`）
- **NEVER** 将 13 字段与涌现 schema 混合定义（schema 层分离是硬约束）

---

## NOTES

**待创建 5 份文档**（P0 完成后）：`session-train-data-synthesis.md`、`eval-results-v1.md`、`findings-v1.md`、`prompt-template.md`、`privacy-tier-policy.md`。

**当前状态**：v0.2 已发布（commit `00eb107`），P1 训练尚未启动。

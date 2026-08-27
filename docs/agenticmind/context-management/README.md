# 会话管理智能体指南 — 综述与索引

> **文档 ID**: CM-000-INDEX
> **生成日期**: 2026-08-24
> **状态**: 草案 v0.2(方向重定位:从"运行时抽取系统"→"schema 真源 + 智能体消费规范")
> **配套文档**:
> - `mvp-schema.md` — 13 字段人工 schema 单一真源(被 agenticmemory_training 锚定)
> - `architecture.md` — 运行时会话管理智能体的行为规范
> - `p0-prototype-tasks.md` — P0 原型任务清单(关键路径已调整为 interim 方案)
> - [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) — Schema 融合边界规范

> **v0.2 重定位记录**:
> - **方向调整**:从"自建 0.5B Python 抽取系统"→"人工 schema 真源 + 统一 Qwen3-0.6B 模型 + 智能体消费规范"
> - **新增交叉引用**:与 `agenticmemory_training/08b` 的 schema 融合边界(Schema 层分离 + 数据集层联合)
> - **决策 D4 升级**:从"v1 单 0.5B + 规则路由"改为"统一 Qwen3-0.6B base + 双 LoRA(待统一模型产出后替换)"
> - **P0 任务调整**:T1.3/T1.4 标记为 interim 方案,统一模型就绪后替换

---

## 0. 文档范围与定位

本文档树聚焦 **"会话交互 schema 真源 + 推理时会话管理智能体消费规范"**——为 AgenticMind 项目提供:

- **训练侧**:13 字段人工 schema 的单一真源,被 [`../../agenticmemory_training/`](../../agenticmemory_training/) 蒸馏管线的自动涌现流程锚定
- **推理侧**:训练后统一模型的会话管理输出格式规范 + 智能体消费 API

**与上下游的关系**:

```
训练侧:
  [本文档 13 字段]
    ↓ 作为 seed schema
  [08b 融合规范]
    ↓
  [08 蒸馏管线] → memory_train.jsonl + session_train.jsonl
    ↓ task-tagged 混合
  [Qwen3-0.6B base + 双 LoRA / 多任务]
    ↓
  [训练后统一模型]

推理侧:
  用户对话 / 文档
    ↓
  [统一模型推理]
    ├─ task="session_extract" → 13 字段(供会话管理智能体)
    └─ task="memory_extract"  → 三元组(供知识库查询)
    ↓
  [会话管理智能体消费]
```

**目标读者**:
- 数据工程师:用本文档的 13 字段作为训练目标 schema
- 架构师:理解 schema 融合边界,避免字段级合并的范畴错误
- 运行时工程师:把 13 字段作为智能体 prompt 组装的输入格式

**不在本文档范围**:
- 自动涌现算法本身 → 见 [`../../agenticmemory_training/08a-capacity-gap-design.md`](../../agenticmemory_training/08a-capacity-gap-design.md)
- Schema 融合边界 → 见 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md)
- AgenticDSL 语言规范 → 见 HydraForge 仓
- LLM 训练算法 → 见 [`../../agenticdsl-training/`](../../agenticdsl-training/)

---

## 1. 核心结论(一句话)

> **13 字段人工 schema 是会话管理任务的训练目标真源**;AgenticMind 项目训练一个统一模型(**Qwen3-0.6B base** + 双 LoRA 或多任务),既输出自动涌现的领域知识,又输出人工定义的会话交互结构,**被同一个智能体在运行时消费**。

**3 个关键锁定**(经 Oracle 评审 + 用户确认):

1. **Schema 层分离**:`mvp-schema.md`(13 字段,冻结契约)与 `schema_memory_v*.json`(自动涌现,可演化)**互不污染**,通过 task tag 在训练数据层联合(详见 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) §1)
2. **统一模型**:**Qwen3-0.6B base + 双 LoRA**(或单多任务),覆盖记忆抽取与会话抽取两个 task(详见 `08b` §4)
3. **运行时消费规范**:`OrchestratorInterface` 抽象 + 4 级降级阶梯(FULL/KEEP_CORE/MINIMAL/RAW_PASSTHROUGH)仍适用,确保 hot path 不崩(详见 `architecture.md`)

---

## 2. 核心决策记录(已锁定 7 项,v0.2 调整)

| # | 决策项 | 选择 | 理由 |
|---|---|---|---|
| **D1** | **角色** | 人工 schema 真源 + 智能体消费规范 | 训练侧与运行时统一服务 |
| **D2** | **Schema 策略** | Schema 层分离 + 数据集层联合(task-tagged) | 避免字段级融合的范畴错误 |
| **D3** | **优先级** | L0 + L1 优先,L2/L3/L4 后续 | 训练链路紧耦合 L0/L1 |
| **D4** | **模型选型** | **Qwen3-0.6B base + 双 LoRA(待统一模型就绪)**;P0 interim 用教师 API + 规则 | 对齐 08a D-10 tokenizer 硬约束 |
| **D5** | **隐私策略** | 脱敏后转发 + 白名单制 + 映射表还原 | 平衡隐私与可用性(secret 实体强制保留) |
| **D6** | **编排器实现** | v1 Python 函数;v2 HydraForge Agent(TR-3 后) | 避免跨仓未交付依赖 + 保留自举愿景 |
| **D7** | **降级策略** | 4 级阶梯 + 字段级 confidence + raw 直通兜底 | 保证 hot path 不崩 |

详细决策路径见 `architecture.md` §5.4 与 `mvp-schema.md` §4。

---

## 3. 文档结构

| 编号 | 文档 | 内容 | 状态 |
|---|---|---|---|
| **00** | `README.md`(本文件)| 综述、决策清单、与上下游的边界 | v0.2 |
| **01** | `mvp-schema.md` | 13 字段人工 schema 单一真源(被 08 管线锚定) | v0.2 |
| **02** | `architecture.md` | 运行时会话管理智能体的行为规范 + 编排层设计 | v0.2 |
| **03** | `p0-prototype-tasks.md` | P0 原型任务清单(关键路径已调整为 interim 方案) | v0.2 |

**待创建文档**(M3 完成后产出):

| 编号 | 文档 | 内容 | 触发条件 |
|---|---|---|---|
| **04** | `session-train-data-synthesis.md` | session_train.jsonl 数据合成 SOP | 08 管线 Phase 0 启动时 |
| **05** | `eval-results-v1.md` | 50-100 条真实会话验证报告 | M3 完成 |
| **06** | `findings-v1.md` | M3 暴露问题清单 + 改进建议 | M3 完成 |
| **07** | `prompt-template.md` | 4 级降级对应的 prompt 模板集 | M2 完成 |
| **08** | `privacy-tier-policy.md` | 字段级隐私白名单策略文档 | T2.3 启动时 |

---

## 4. 三阶段路径(与 08 管线 + TR-1/TR-2/TR-3 对齐,v0.2 调整)

| 阶段 | 时间 | 关键产出 | 与上下游的关系 |
|---|---|---|---|
| **P0 原型(interim)** | 3-4 周 | 端到端 demo + 50 条验证 + 问题清单 | 用"教师 API + 规则"做 interim 抽取;统一 Qwen3-0.6B 模型就绪后替换 |
| **Phase 1 — session_train.jsonl 合成** | 2-3 周 | session_train.jsonl 数据集(对齐 13 字段) | 与 08 管线 Phase 5 并行,产出 task-tagged 多任务训练数据 |
| **Phase 2 — 统一模型训练** | 与 TR-1 W3-W5 对齐 | Qwen3-0.6B base + 双 LoRA(memory + session) | 同一模型同时输出两种 schema 的内容 |
| **Phase 3 — 运行时集成** | 持续 | 与 AgenticMind LLM 推理链路集成 + 评估 | TR-2 / TR-3 的运行时验证 |
| **Phase 4 — HydraForge 重写** | TR-3 后 | 编排层重写为 AgenticDSL 程序 | 自举演示项目(M7 关联)|

**累计**:P0(3-4 周 interim)+ Phase 1-3(与 TR-1 同步)+ Phase 4(TR-3 后)。

---

## 5. 与上下游的边界(v0.2 重写)

### 5.1 与 `agenticmemory_training/` 的边界

| 内容 | 归属 | 理由 |
|---|---|---|
| **13 字段人工 schema 定义** | 本目录 `mvp-schema.md` | 是人工契约真源 |
| **自动涌现算法(HDBSCAN + LLM 概念化)** | `../../agenticmemory_training/08a-capacity-gap-design.md` | 是数据合成算法 |
| **Schema 融合边界** | `../../agenticmemory_training/08b-seed-schema-fusion.md` | 是双方协同规范 |
| **`memory_train.jsonl` 合成** | `../../agenticmemory_training/08-memory-distillation-pipeline.md` | 是文档抽取训练数据 |
| **`session_train.jsonl` 合成** | 本目录(待建 `session-train-data-synthesis.md`)| 是会话抽取训练数据,需新建 |
| **统一模型训练** | `../../agenticmemory_training/`(待建 `08c-unified-model-training.md`)| 是双 LoRA / 多任务 SFT |

**关键边界规则**:
- 13 字段中**只有 `entities` 的 9 种类型**可参与自动涌现锚定
- 其余 12 字段(intent/topic/facts/横切元数据)只从对话语料产生,**不参与**自动涌现
- 两份 schema 在训练数据层通过 task tag 联合,**不在 schema 层做字段级合并**

### 5.2 与 HydraForge 仓的边界

| 内容 | 归属 | 理由 |
|---|---|---|
| **13 字段人工 schema 定义** | AgenticMind 仓 `mvp-schema.md` | 是训练/推理数据工程 |
| **OrchestratorInterface 抽象** | `architecture.md` | 是编排层抽象 |
| **v1 Python 实现** | AgenticMind 仓 | 本周可用,不依赖 HydraForge |
| **v2 HydraForge Agent 实现** | HydraForge 仓(TR-3 后) | 是 AgenticDSL 自举演示 |
| **AgenticDSL runtime 验证** | HydraForge 仓 | 运行时不属于抽取系统 |
| **4 层验证器调用** | HydraForge 仓 | 是验证层,不是抽取层 |
| **ILLMProvider 接口** | HydraForge 仓 | 是 LLM 调用抽象 |

---

## 6. 时间线与里程碑

### P0 原型阶段(本季度,v0.2 调整)

| 里程碑 | 时间 | 关键产出 | 成功标准 |
|---|---|---|---|
| **CM-M1** | Week 1 | 编排层 + Schema 校验器 + interim 抽取(教师 API + 规则) | 编排可独立验证;intent accuracy ≥85%(教师 API),entity F1 ≥0.8 |
| **CM-M2** | Week 2 | PythonOrchestrator + 端到端 demo + 4 级降级 | 10 条对话可走通 |
| **CM-M3** | Week 3 | 50 条真实会话验证 + 问题清单 | 字段填充率 L0 ≥90%, L1 ≥70% |

### Phase 1 阶段(session_train.jsonl 合成)

| 里程碑 | 时间 | 关键产出 | 成功标准 |
|---|---|---|---|
| **CM-M4** | 与 08 管线 Phase 0 并行 | 对话语料收集 + 教师 API 标注流水线 | session_train.jsonl ≥10K 条,标注一致率 ≥85% |

### Phase 2 阶段(统一模型训练,与 TR-1 对齐)

| 里程碑 | 时间 | 关键产出 | 成功标准 |
|---|---|---|---|
| **CM-M5** | TR-1 W5 | task-tagged 混合数据集 + Qwen3-0.6B base + 双 LoRA | 双 task 准确率 ≥80% |

### Phase 4 阶段(TR-3 后)

| 里程碑 | 时间 | 关键产出 | 成功标准 |
|---|---|---|---|
| **CM-M6** | TR-3 M5 之后 | v2 HydraForge 编排器 alpha | v2 在 50 条 case 上 ≥v1 |
| **CM-M7** | TR-3 M6 | v2 默认 + v1 fallback | A/B 测试无回归 |

---

## 7. 风险登记(摘录,v0.2 调整)

完整风险登记见 `p0-prototype-tasks.md` §4 与待创建的 `findings-v1.md`。

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R1 | 统一模型双 task 准确率不达标 | 🔴 高 | 教师 API fallback;独立 task 训练后融合 |
| R2 | PII 脱敏召回率 <99.5% | 🔴 高 | 推迟云端转发,先全本地 |
| R3 | session_train 标注工时超预期 | 🟡 中 | GPT-4 预标注 + 人工抽检 20% |
| R4 | 端到端延迟 >500ms | 🟡 中 | 批处理 + 模型并行 + 缓存 |
| R5 | 字段 confidence 普遍偏低 | 🟡 中 | temperature scaling + reject option |
| R6 | HydraForge runtime 未交付阻塞 v2 | 🟢 低 | v1 不依赖,无影响 |
| **R7** | **schema 融合边界被误解(字段级合并)** | 🟡 中 | 详见 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) |
| **R8** | **统一模型双 LoRA 切换延迟 > 50ms** | 🟡 中 | 待 Phase 2 验证;超标则改多任务单权重 |

---

## 8. 阅读建议

| 读者 | 推荐阅读路径 |
|---|---|
| **新加入成员** | 本 README → [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) → `mvp-schema.md` → `architecture.md` |
| **P0 原型开发者(interim)** | 本 README → `p0-prototype-tasks.md` → `mvp-schema.md` §3 |
| **架构师** | [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) → `architecture.md` → `mvp-schema.md` §4-5 |
| **训练数据工程师** | `mvp-schema.md` §3 → [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) §3 |
| **Prompt 模板维护者** | `mvp-schema.md` §3 → `architecture.md` §4 → 待创建 `prompt-template.md` |
| **项目负责人 / 决策者** | 本 README §1 + §2 + §7 即可 |

---

## 9. 下一步行动

1. ✅ **v0.2 重定位已完成**(本文档 + 3 个配套文档升级;新增 `08b-seed-schema-fusion.md`)
2. **下一步**:用户 review v0.2 文档集,批准后启动 P0 interim + 08 管线 + session_train.jsonl 合成
3. **再下一步**:按 `p0-prototype-tasks.md` 推进 CM-M1 → CM-M3(3-4 周)
3. **再下一步**:按 `p0-prototype-tasks.md` 推进 M1 → M2 → M3
4. **M3 完成后**:基于问题清单,产出 `findings-v1.md` + `eval-results-v1.md`,反哺 schema 和训练数据设计

---

## 10. 文档版本与变更

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-24 | 初始草案(基于 Oracle 评审 + 用户确认) | Sisyphus(AI 助手)|
| v0.1.1 | 2026-08-24 | Oracle 文档评审补丁(13 字段真源表 + secret + temperature scaling + 降级逻辑重写 + Sidecar 集成) | Sisyphus(AI 助手)|
| **v0.2** | **2026-08-24** | **方向重定位**:从"运行时抽取系统"→"schema 真源 + 智能体消费规范";新增与 `agenticmemory_training/08b` 的 schema 融合边界;模型选型统一为 Qwen3-0.6B;P0 改 interim 方案 | **Sisyphus(AI 助手)** |

---

**文档版本**:v0.2
**Owner**:AgenticMind 抽取系统组
**配套 AGENTS.md**:§9 "当前状态" 待追加本文档索引
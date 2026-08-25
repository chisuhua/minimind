# AgenticMind 记忆训练数据集构建 — 综述与索引

> **文档 ID**: MEMDATA-001-INDEX
> **生成日期**: 2026-08-24
> **状态**: 草案 v1.0
> **配套文档**（下游 AgenticDSL 训练）:
> - [`../agenticdsl-training/`](../agenticdsl-training/) — AgenticDSL LLM 训练系列

---

## 0. 文档范围与定位

本目录聚焦于 **"如何为训练记忆引擎模型构建高质量结构化数据集"**——使用 Capacity Gap 作为探针，自动将语料按"记忆 / 推理"分层，并通过教师 API 完成 OpenIE 提取与 Schema 自动涌现，最终产出可直接喂给 `../agenticdsl-training/01-training-data-pipeline.md` 第 3 阶段（执行驱动过滤）的高质量记忆样本。

**目标读者**：AgenticMind 数据工程师、记忆蒸馏链路实施者、单卡实验者。

**与 [`../agenticdsl-training/`](../agenticdsl-training/) 的边界**：

| 内容 | 归属 | 原因 |
|---|---|---|
| Capacity Gap 分层原理 | 本目录 | 记忆/推理分类是数据集构建的核心 |
| OpenIE + Schema 自动涌现 | 本目录 | 数据集构建的核心产出 |
| RTX 4090 单卡实操 | 本目录 | 数据集构建的硬件选型 |
| AgenticDSL 训练算法 Recipe（ReSTᴱᴹ / GRPO / MCTS）| `../agenticdsl-training/` | 是训练算法而非数据预处理 |
| AgenticDSL 4 层验证器 | `../agenticdsl-training/` | 是训练阶段工具而非数据集构建 |
| HydraForgeBench 评估 | `../agenticdsl-training/` | 是训练后评估 |

> **关系**：本目录的产出物（`memory_train.jsonl` + `schema_v*.json` + `stratified/`）作为 `../agenticdsl-training/01-training-data-pipeline.md` 第 3 阶段的**前置输入**。两个目录形成"数据预处理 → 训练"的两阶段工作流。

---

## 1. 核心结论

基于 Capacity Gap 的记忆蒸馏管线 **完全可以在 RTX 4090 单卡上完成**，并将总成本从 3×A100 方案的 ~¥1,100 降至 **~¥36（节省 97%）**，同时获得：

1. **双探针架构**：0.6B (崩溃对照) + 1.7B (主探针) 与教师 (13B 激活) 分别构成 21.7× / 7.6× 安全 Gap，**无需三层探针**——4B 探针的输出在当前算法下未被消费
2. **CCS 连续谱系**：用连续分数替代硬二分，0.3~0.7 灰色样本进人工审核而非粗暴归类
3. **Schema 自动涌现**：通过 HDBSCAN 聚类 + LLM 概念化（混合路线），借鉴 AutoSchemaKG 的 Extract-Define-Canonicalize 范式
4. **三层防御质量控制**：格式合规 → 语义一致性 → 去重冲突 → CCS 难度分级
5. **结构化输出蒸馏**：规避 Capacity Gap 诅咒对传统 logit 蒸馏的影响

---

## 2. 文档结构

| 编号 | 文档 | 定位 |
|---|---|---|
| **00** | `README.md`（本文）| 综述、与 `agenticdsl-training/` 的边界 |
| **08** | `08-memory-distillation-pipeline.md` | RTX 4090 单卡实操搭建指南（v1.1，含完整代码与配置） |
| **08a** | `08a-capacity-gap-design.md` | v0.1 设计方案（理论层 + 设计层决策表附录 A） |

> **注**：文档保留"08 / 08a / 08b / 08c / 08d"编号是为了：
> 1. 与 `../agenticdsl-training/` 编号体系保持视觉一致性（两者是"姊妹"目录）
> 2. 文档 ID（`LLMTRN-008-MEMDIST` / `LLMTRN-008A-MEMDIST-DESIGN` 等）已稳定使用
> 3. 文档内大量引用以"§X.X"形式锚定，重命名会破坏所有锚点
>
> **2026-08-26 重命名评估结论**:
> - **不建议重命名**为 01-06（与 `docs/agenticmemory/` 一致）
> - **破坏范围**:30+ 处跨文档交叉引用、文档 ID 体系、commit 历史追溯
> - **收益评估有限**:`docs/agenticdsl-training/` 也用 01-06 编号,重命名并不能实现"全部 01-XX 一致"
> - **替代方案**:新文档继续顺延（08d 已于 2026-08-26 新建,后续 09/09a 顺延）;新成员阅读时由本文档 §2 解释编号沿用原因

---

## 3. 双文档分层架构

| 维度 | 设计稿（08a） | 搭建指南（08） |
|------|-------------|--------------|
| **定位** | 理论 / 设计 / 决策层 | 实操 / 适配 / 代码层 |
| **目标读者** | 架构师、技术决策者 | 数据工程师、实施者 |
| **内容性质** | "为什么这样做"的方法论 | "在 RTX 4090 上怎么做"的操作手册 |
| **数值精度** | 区间 / 选项标注（如 "500M–1.5B 默认 0.6B"）| 具体数值（如 "Qwen3-0.6B + Qwen3-1.7B"）|
| **决策表位置** | 附录 A：13 项设计层决策 | §十五：13 项适配层决策 |

**双决策表单一真源原则**：避免双副本漂移——
- **设计层决策**（CCS 公式、阈值、Schema 涌现流程等）仅在 **08a 附录 A** 定义
- **适配层决策**（具体选哪个模型、用哪张卡）仅在 **08 §十五** 定义
- 两层决策用文档 ID 前缀区分（`DESIGN` vs `ADAPTATION-RTX4090`），互为反向引用

---

## 4. 关键决策摘要（设计层，详见 08a 附录 A）

| ID | 决策项 | 选择 | 适用范围 |
|---|---|---|---|
| D-01 | CCS 公式 | `0.5·gap + 0.3·recon + 0.2·bottleneck` | 设计层 |
| D-02 | CCS 阈值 | memory < 0.3 / reasoning > 0.7 | 设计层 |
| D-03 | 重构敏感度扰动协议 | 3 类型 × 3 强度（同义/实体/关系）| 设计层 |
| D-04 | Schema 涌现流程 | HDBSCAN + LLM 概念化（混合）| 设计层 |
| D-05 | Schema 概念化温度 | T=0.0 | 设计层 |
| D-06 | 训练目标区间 | 500M–1.5B | 设计层 |
| D-07 | 三阶段训练 | format / schema / difficulty | 设计层 |
| D-08 | 灰色地带处理 | 进人工审核队列（不进训练）| 设计层 |
| D-09 | 蒸馏方式 | 结构化输出蒸馏 | 设计层 |
| D-10 | Tokenizer 对齐 | 同系列或与教师对齐（硬约束）| 设计层 |
| D-11 | 崩溃对照组 | L3 (0.6B) 排除 trivial | 设计层 |
| D-12 | 教师提取温度 | **T≤1.0**（推翻"T=2.0 暗知识展开"论证）| 设计层 |
| D-13 | 教师置信度提取 | 答案 span 平均 logprob | 设计层 |

---

## 5. 关键决策摘要（适配层-RTX4090，详见 08 §十五）

| ID | 决策项 | 选择 | 备注 |
|---|---|---|---|
| A-01 | 教师模型 | DeepSeek V4 Flash API（13B 激活）| 远程免本地显存 |
| A-02 | 主探针 (L1) | Qwen3-1.7B Base | 与教师 7.6× Gap + 同系列对齐 |
| A-03 | 崩溃对照 (L3) | Qwen3-0.6B Base | 与教师 21.7× Gap |
| A-04 | ~~辅助探针 (L2)~~ | **删除**（v1.1 修订）| 原 Qwen3-4B，当前代码不消费其输出 |
| A-05 | 训练目标 | Qwen3-0.6B | 极轻量"记忆引擎" |
| A-06 | 教师温度 | 0.7（≤1.0）| 推翻 T=2.0 论证 |
| A-07 | 学生温度 | 1.0 | 测真实记忆 |
| A-08 | 教师置信度提取 | 答案 span 平均 logprob | 推翻首 token logprob |

---

## 6. 修订记录（v1.1，2026-08-24）

本次 v1.1 修订由 Oracle 咨询触发，主要变更：

| # | 类型 | 修订项 |
|---|------|--------|
| 1 | **架构变更** | 删除 L2 (4B) 探针，改为双探针 0.6B + 1.7B |
| 2 | **参数修正** | 教师温度 T=2.0 → T≤1.0（推翻"暗知识展开"论证）|
| 3 | **方法论修正** | 教师置信度提取：首 token logprob → 答案 span 平均 logprob |
| 4 | **bug 修复** | 启动脚本加 `--max-logprobs 100` + student payload `logprobs: 20` |
| 5 | **方法论新增** | L3 bottleneck 加基线校准前置（`_check_L3_baseline`）|
| 6 | **数据修正** | hard 桶定义：`CCS ∈ [0.3, 0.4)` → "Top 10% by CCS" |
| 7 | **文档治理** | 决策表改为单一表 + 作用域列，避免双副本漂移 |
| 8 | **文档治理** | 所有"前文"悬空引用 → 指向 08a 具体锚点 |
| 9 | **目录重构** | 从 `agenticdsl-training/` 移至独立子目录 `agenticmemory_training/` |

---

## 7. 成本与时间总估算（4090 单卡 + API）

| 阶段 | 4090 GPU 时间 | API 费用 | 备注 |
|------|-------------|---------|------|
| Phase 0: 语料清洗 | 0（CPU） | ¥0 | |
| Phase 0: 探针校准 | ~20 min | ¥0 | |
| Phase 1: 双盲提取（10 万块） | ~6 h | ~¥25 | 教师 API 是主要成本 |
| Phase 2: 分层 | ~2 h | ~¥5 | |
| Phase 3: Schema 涌现 | ~1 h | ~¥3 | |
| Phase 4: 质量控制 | ~1 h | ~¥2 | |
| Phase 5: 三阶段训练 | ~11 h | ¥0 | 本地训练 |
| Phase 6: 评估 | ~1 h | ~¥1 | |
| **总计** | **~22 h** | **~¥36** | |

> **对比原稿的 3×A100 方案（¥1,100）**：4090 单卡方案将计算成本降至 ¥0（自有硬件），API 成本仅 ¥36。总成本降低 **97%**。

---

## 8. 风险登记摘要

| # | 风险 | 状态 | 架构侧对应 |
|---|------|------|------|
| R-01 | CCS 阈值误分类 | 监控中(每季度验证集重校准)| [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-C02(五步漏斗判定错误) |
| R-02 | 教师 API 成本失控 | 每日 ¥50 硬上限 + 告警;超出后切回本地 1.7B | [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-E03(数据合成管线成本) |
| R-03 | HDBSCAN 聚类不稳定 | 多次运行取交集 | [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-S02(Schema 漂移) |
| R-04 | 灰色地带样本流失 | 人工审核队列(每周抽样 200 条标注)| [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-T06(训练配比敏感) |
| R-05 | 0.6B 上限 | schema 总数 ≤ 200 关系 / 50 实体 | [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-T04(1B 模型能力上限) |
| R-06 | Schema 漂移 | 锁定 schema_version,变更走 PR review | [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-E04(Schema 融合边界) |
| R-07 | 冷启动校准失效 | 多样性采样 + 200 条人工标注验证 | [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-C02 |
| R-08 | 灰色地带样本流失 | 比例监控 + 优先级队列 | [`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) R-T06 |

**单一真源原则**(2026-08-26 双向索引建立):
- **架构级风险**(能力边界 / 训练配比 / 部署回归):[`../agenticmemory/08-risk-register.md`](../agenticmemory/08-risk-register.md) 25 项
- **数据工程级风险**(CCS / HDBSCAN / Schema 漂移 / 灰色地带):本节 8 项
- **R-C01**(Wiki DAG 构建算法)→ [`08d-wiki-dag-construction.md` §5](08d-wiki-dag-construction.md) 待解决清单 O1.1/O1.2/O1.3
- 新增风险时先确认归属,再登记到对应文档,**避免双副本**

详见 [`08` §十六](08-memory-distillation-pipeline.md) 与 [`08a` §9](08a-capacity-gap-design.md)。

---

## 9. 与主训练管线的衔接

> 本目录产出的 `data/training/*.jsonl` 应当作为 [`../agenticdsl-training/01-training-data-pipeline.md`](../agenticdsl-training/01-training-data-pipeline.md) 第 3 阶段（执行驱动过滤）的**前置输入**，而非直接喂入 SFT。

| 产出物 | 路径 | 喂给 |
|---|---|---|
| `memory_train.jsonl` | `data/training/memory_train.jsonl` | → [`../agenticdsl-training/01-training-data-pipeline.md`](../agenticdsl-training/01-training-data-pipeline.md) §3 |
| `schema_v1.json` | `data/schema/schema_v1.json` | → 同上 §2 L2 schema 校验 |
| `stratified/` | `data/stratified/` | → 同上 §1 任务矩阵的"记忆 / 推理"分层信号 |

> **避免重复劳动**：本管线已完成"记忆 / 推理"分层与 Schema 涌现，`../agenticdsl-training/01-training-data-pipeline.md` 后续阶段无需再做 HDBSCAN 聚类或 LLM 概念化，仅做格式归一与 schema 对齐即可。

---

## 10. 阅读建议

| 读者 | 推荐阅读路径 |
|------|------------|
| **数据工程师** | 本 README → [`08`](08-memory-distillation-pipeline.md)（按 §一 → §十三 顺序搭建）|
| **架构师 / 技术决策者** | 本 README → [`08a`](08a-capacity-gap-design.md)（按 §1 → §5 顺序理解方法论）|
| **新加入成员** | 本 README → [`08a` §1](08a-capacity-gap-design.md)（建立全局观）→ [`08` §一](08-memory-distillation-pipeline.md)（了解适配）|

---

**文档版本**: v1.0
**最后更新**: 2026-08-24
**Owner**: AgenticMind 数据工程团队

## 7. 修订记录（v1.2，2026-08-24）

本次 v1.2 修订由 Oracle 评审触发，主题是**边界纯度回扫**：

| # | 类型 | 修订项 |
|---|------|--------|
| 1 | **边界治理** | `08a-capacity-gap-design.md` §10 标题改为"参考工程配置与资源包络（3×A100 基线）"，加边界声明保留成本对比分母 |
| 2 | **代码块清理** | 08a 中 4 个实现细节代码块（§3.2 OpenIE Prompt / §3.3 Python / §5.4 Schema JSON / §6.1 训练样本 JSON）改为指向 `08-memory-distillation-pipeline.md` 对应章节 |
| 3 | **导航指引** | 08a 各 Phase 章节头部加"实现见 08 §X"指引行，强化"设计层 vs 适配层"分工 |
| 4 | **决策追加** | 08a 附录 A 加 D-14（Schema 融合边界 seed 锚定策略），引用 `08b-seed-schema-fusion.md` §2 |
| 5 | **文档版本** | 08a 由 v1.0 → v1.2；08 保持 v1.1（未改动） |
| 6 | **改名延后** | `architecture.md` 改名待 P0 反馈 + 外部引用确认后决策（避免破坏 45 处引用 + 文档 ID 体系） |

**与 v1.1 的关系**：v1.1 修订解决了"决策表双副本漂移"和"探针架构过设计"问题；v1.2 解决了"设计层/适配层边界在执行中漂移"问题（26 个代码块污染）。两者**不冲突**，v1.2 是 v1.1 治理后的执行维护。

---

## 8. P1 最小闭环实验（2026-08-25 新增）

P1 是一个**2 周 / 1 人 / ~$50**的最小闭环实验，目标：验证"sub-1B 模型能否学会 13 字段结构化抽取"。

| 文档 | 代码 | 内容 |
|---|---|---|
| [`08c-p1-minimum-loop.md`](08c-p1-minimum-loop.md) | `agenticmemory_training/{data,training}/` | 数据合成 → 教师标注 → Schema 评估 → LoRA 微调 全流程 |
| 共享契约 | `agenticmind/extraction/` | 13 字段 schema / validator / privacy(训练侧 + 运行时侧共用) |

**与蒸馏管线(08)的关系**：P1 是蒸馏管线的**前置验证实验**——在启动 08 管线 Phase 0-6 全量蒸馏前，先用小样本验证 13 字段 schema 可标注性、教师标注一致性、0.6B 微调可行性。P1 的 findings 直接影响 08 管线的数据源选择与 schema 设计。

**与 08b 的关系**：08b 定义了 Schema 融合边界；P1 验证了该边界在**实际数据**上的可行性（通过教师标注和 0.6B LoRA 训练）。

**P1 包结构**详见 [`08c-p1-minimum-loop.md` §0](08c-p1-minimum-loop.md)。


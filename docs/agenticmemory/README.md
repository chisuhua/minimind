# AgenticMemory — KV 缓存即记忆基底

> **文档 ID**: MEM-000-INDEX
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.6(加入生产部署 + 风险登记册 + 责任组任务规划)
> **作者**: Sisyphus(AI 助手)+ 用户澄清
> **配套文档**:
> - 核心能力: [`01-memory-model.md`](01-memory-model.md) — 推理无损 + 五步漏斗 + Wiki DAG
> - 训练设计: [`02-training-design.md`](02-training-design.md) — 三层训练信号 + 六类样本 + 四阶段课程
> - 评估框架: [`03-evaluation.md`](03-evaluation.md) — 双轨评估 + IRR + KV 验证
> - 多轮对话扩展: [`04-dialogue-extension.md`](04-dialogue-extension.md) — 轮次分块 + MQP v3 + Type J-N + 增量更新
> - 本体涌现: [`05-schema-emergence.md`](05-schema-emergence.md) — OpenIE + 信息瓶颈 + 双系统 + V1-V3 路线图
> - 评估方法论: [`06-evaluation-methodology.md`](06-evaluation-methodology.md) — Probe Model + Golden Filter + 相关工作
> - 生产部署: [`07-production-deployment.md`](07-production-deployment.md) — vLLM + S-LoRA + 数据飞轮 + 监控 + 增量更新
> - 风险登记册: [`08-risk-register.md`](08-risk-register.md) — 所有已识别风险集中跟踪
> - 责任组任务规划: [`09-team-roadmap.md`](09-team-roadmap.md) — 按架构/数据/训练/评估/部署拆解的任务地图
> - 训练侧实现: [`../agenticmemory_training/`](../agenticmemory_training/) — 数据合成 + 蒸馏 + LoRA 微调
> - 推理引擎: [`../inference-engine/`](../inference-engine/) — pre-allocated KV / StreamingLLM / KIVI
> - 消费方 A: [`../agenticmind/context-management/`](../agenticmind/context-management/) — agenticmind 13 字段 schema
> - 消费方 B: [`../agenticdsl-training/`](../agenticdsl-training/) — agenticinference 推理链

---

## 0. 文档范围与定位

本文档定义 **agenticmemory** 的整体架构——一个以 **KV 缓存为产品形态** 的只读记忆基底,通过**多 LoRA 探针**服务多个下游消费方。

**关键澄清(2026-08-25 用户定义)**:

> agenticmemory 不是"输出 JSON 的抽取模型",而是"**训练后产出的、其 KV 缓存形态本身就是产品**的记忆基底"。同一上下文 prefill 一次后,产出的 KV cache 被多个消费方共享;不同消费方通过各自的 LoRA 探针从同一 KV 中提取所需视角的信息。Wiki DAG 是训练阶段用于**证明 KV 缓存确实包含完整信息**的验证产物。

**核心能力**:**推理无损** — 基于记忆回答推理问题的准确率不低于基于原文的 98%(B/A ≥ 0.98)。详细定义见 [`01-memory-model.md`](01-memory-model.md) §1。

---

## 1. 三层消费者架构总览

```
                        ┌─────────────────────────────────┐
                        │  消费方层(Consumers)              │
                        │                                 │
                        │  ┌───────────┐  ┌────────────┐  │
                        │  │agenticmind│  │agenticinference│ │
                        │  │ (会话管理) │  │  (推理智能体)  │ │
                        │  │           │  │              │  │
                        │  │ 13 字段    │  │ OpenIE 三元组  │ │
                        │  │ prompt组装 │  │ Wiki DAG 遍历  │ │
                        │  └─────┬─────┘  └──────┬───────┘  │
                        │        │                │          │
                        │        │ probe+LoRA    │ probe+LoRA │
                        └────────┼────────────────┼──────────┘
                                 │                │
                                 ▼                ▼
        ┌────────────────────────────────────────────────────────┐
        │         agenticmemory(KV 缓存记忆基底)                │
        │                                                        │
        │  ┌────────────────────────────────────────────────┐  │
        │  │ L3 Probe Layer(LoRA 探针层)                  │  │
        │  │   LoRA_session → 13 字段(for agenticmind)     │  │
        │  │   LoRA_memory  → 三元组(for agenticinference) │  │
        │  │   LoRA_X       → 未来更多 consumer            │  │
        │  └────────────────────────────────────────────────┘  │
        │                        ▲                              │
        │  ┌────────────────────────────────────────────────┐  │
        │  │ L2 Retrieval Layer(检索层)                    │  │
        │  │   MVP: prefix radix(RadixAttention 风格)       │  │
        │  │   Long-term: 层次化 KV 动态加载                 │  │
        │  └────────────────────────────────────────────────┘  │
        │                        ▲                              │
        │  ┌────────────────────────────────────────────────┐  │
        │  │ L1 KV Cache Layer(KV 缓存层 = 产品)           │  │
        │  │   同一上下文 prefill 一次 → 多 consumer 共享    │  │
        │  │   常驻(MVP)/ 分层(远期)                       │  │
        │  └────────────────────────────────────────────────┘  │
        │                        ▲                              │
        │  ┌────────────────────────────────────────────────┐  │
        │  │ L0 Training-time Proof(Wiki DAG 训练证明层)  │  │
        │  │   训练时构建 Wiki DAG                          │  │
        │  │   作为"KV 缓存包含完整信息"的形式化证明         │  │
        │  └────────────────────────────────────────────────┘  │
        └────────────────────────────────────────────────────────┘
                                 ▲
                                 │ prefill(context)
                                 │
                        ┌────────┴────────┐
                        │  原始上下文       │
                        │ (对话/文档/混合) │
                        └─────────────────┘
```

**核心机制**:同一份原始上下文经过**一次 prefill** → 产出 KV cache(共享)→ 不同 consumer 用各自的 **LoRA 探针**查询同一 KV → 提取各自视角的信息。

---

## 2. 文档结构

| 编号 | 文档 | 内容 | 状态 |
|---|---|---|---|
| **00** | `README.md`(本文件)| 架构总览、消费者分层、与上下游边界 | v0.6 |
| **01** | `01-memory-model.md` | 核心能力(推理无损) + 五步漏斗 + Wiki DAG 契约 | v0.1 |
| **02** | `02-training-design.md` | 三层训练信号 + 六类样本 + 四阶段课程 + 损失函数 | v0.1 |
| **03** | `03-evaluation.md` | 双轨评估 + IRR + 六维度 + KV 验证实验 + 失败诊断 | v0.1 |
| **04** | `04-dialogue-extension.md` | 多轮对话边界 + 轮次分块 + 对话索引 + MQP v3 + Type J-N + 增量更新 | v0.1 |
| **05** | `05-schema-emergence.md` | 本体盲区 + OpenIE + 信息瓶颈 + 双系统 + V1-V3 路线图 + 稀疏性惩罚 | v0.1 |
| **06** | `06-evaluation-methodology.md` | Probe Model + Golden Filter + QA 题库 + 相关工作(DSPy/StructMem) | v0.1 |
| **07** | `07-production-deployment.md` | vLLM + S-LoRA 服务 + 数据飞轮 + 监控告警 + 增量更新策略 | v0.1 |
| **08** | `08-risk-register.md` | 所有已识别风险集中跟踪(25 项,按类别分组) | v0.1 |
| **09** | `09-team-roadmap.md` | 按责任组拆解的任务地图(架构/数据/训练/评估/部署) | v0.1 |

**待创建文档**:

---

## 3. 各层职责简要

### 3.1 L0:Wiki DAG 训练证明层

**职责**:在训练阶段构建 Wiki DAG,作为"训练出的 KV 缓存确实包含完整信息"的形式化证明。

**核心契约**:Wiki DAG 包含 8 个顶层字段(basic_info / core_facts / relations / reasoning_chains / context_annotations / domain_knowledge / sources / completeness_metadata),其中 `completeness_metadata.irr_estimate` 是能力自知的载体,`reasoning_chains[*].needs_reasoning_model_verification` 是边界标注。

详细 Wiki DAG 契约见 [`01-memory-model.md`](01-memory-model.md) §5-6。

### 3.2 L1:KV Cache Layer(产品本身)

**职责**:prefill 原始上下文,产出 KV cache。这是 agenticmemory 的**核心产品**。

**MVP 形态(常驻 + prefix radix)**:
- 单次 prefill 后 KV cache 常驻内存
- 多 consumer 共享同一 KV
- 使用 prefix radix 树管理多个上下文的公共前缀

**远期形态(层次化)**:见 §6。

**与现有 inference-engine 的关系**:
- [`../inference-engine/01-pre-allocated-kv-cache.md`](../inference-engine/01-pre-allocated-kv-cache.md): **直接复用**——消除分配开销,稳定显存布局
- [`../inference-engine/02-streaming-llm.md`](../inference-engine/02-streaming-llm.md): **部分复用**——长对话场景的环形 buffer
- [`../inference-engine/09-kivi.md`](../inference-engine/09-kivi.md): **可选复用**——显存紧张时量化冷 KV

### 3.3 L2:Retrieval Layer(检索层)

**职责**:决定 probe 查询时应该激活 KV cache 的哪些部分。

**MVP 方案(prefix radix,2026-08-25 用户确认为 MVP 新增组件)**:
- 借鉴 SGLang RadixAttention 思路
- 共享前缀只 prefill 一次,多个 probe 复用
- 适合"同一上下文,多 consumer 探针"场景

### 3.4 L3:Probe Layer(LoRA 探针层)

**职责**:每个 consumer 有自己的 LoRA 适配器,作为"查询视角"。

**LoRA 设计建议**(基于用户澄清 2026-08-25):

| 维度 | 建议 | 理由 |
|---|---|---|
| **LoRA rank** | r=8 或 r=16 | 探针只需"选视角",不需学新能力 |
| **作用范围** | 仅 attention Q/V projection | 不动 MLP,保持 base 语义能力 |
| **加载方式** | S-LoRA 风格多适配器并发 | 同 batch 内不同请求用不同 LoRA |
| **切换开销** | per-request 路由,无热切换延迟 | 与 continuous batching 兼容 |
| **训练策略** | 每个 LoRA 独立训练,base 冻结 | 失败隔离,可独立回滚 |

---

## 4. 与现有文档的边界

| 内容 | 归属 | 理由 |
|---|---|---|
| **agenticmemory 架构定义**(本文档) | `docs/agenticmemory/` | 是新产品的架构真源 |
| **核心能力契约(推理无损 + Wiki DAG)** | [`01-memory-model.md`](01-memory-model.md) | 是能力契约 |
| **训练设计原理(三层信号 + 课程)** | [`02-training-design.md`](02-training-design.md) | 是训练原理 |
| **评估框架(双轨 + IRR + KV 验证)** | [`03-evaluation.md`](03-evaluation.md) | 是评估方法 |
| **多轮对话扩展(轮次分块 + Type J-N)** | [`04-dialogue-extension.md`](04-dialogue-extension.md) | 是对话输入特化 |
| **本体涌现范式(OpenIE + 信息瓶颈 + 双系统)** | [`05-schema-emergence.md`](05-schema-emergence.md) | 是高级训练范式 |
| **评估方法论(Probe Model + Golden Filter)** | [`06-evaluation-methodology.md`](06-evaluation-methodology.md) | 是评估方法论 |
| **生产部署(数据飞轮 + 监控 + 增量更新)** | [`07-production-deployment.md`](07-production-deployment.md) | 是部署架构 |
| **风险集中管理(所有风险登记)** | [`08-risk-register.md`](08-risk-register.md) | 是风险真源 |
| **Wiki DAG 构建算法** | `docs/agenticmemory_training/` | 与 08a Phase 3 紧耦合,放训练侧(用户确认 2026-08-25) |
| **训练数据合成 SOP** | `docs/agenticmemory_training/` | 已有,不变 |
| **13 字段人工 schema** | `docs/agenticmind/context-management/mvp-schema.md` | 人工 schema 真源,不变 |
| **LoRA 训练脚本** | `agenticmemory_training/training/` | 已有 P1 骨架,扩展即可 |
| **推理时 KV 管理** | `model/` + `docs/inference-engine/` | 已有 pre-allocated / streaming / KIVI |
| **multi-LoRA serving** | 新增 `agenticmemory/serving/` (待建) | 是 agenticmemory 的运行时组件 |
| **consumer 侧编排** | `agenticmind_runtime/` (预留) | 不属于 agenticmemory |

**关键边界规则**:
1. **agenticmemory 只负责 prefill + probe**,不负责 consumer 的下游逻辑
2. **Wiki DAG 构建放 training 侧**(`agenticmemory_training/`),运行时 consumer 不感知(用户确认 2026-08-25)
3. **LoRA 是 agenticmemory 的一部分**,但训练数据由 consumer 侧定义
4. **层次化 KV 是 agenticmemory 的内部实现**,对 consumer 透明
5. **训练设计原理在 `docs/agenticmemory/`**,具体实现在 `agenticmemory_training/`,**不重复**
6. **agenticmemory 不承担对话管理、用户画像、技能复用等对话系统职责**(详见 [`04-dialogue-extension.md`](04-dialogue-extension.md) §1)

---

## 5. 核心设计原则(三条)

来自 [`01-memory-model.md`](01-memory-model.md) §1.1:

| 原则 | 含义 |
|---|---|
| **推理无损** | 不追求逐字复述,只追求推理所需信息的完整性和精确性 |
| **能力自知** | 模型必须知道自己记住了什么、没记住什么,并显式标注边界 |
| **压缩有效** | 丢弃修辞/连接词/重复表述等推理无关信息,保留因果关系/数值/条件等推理硬依赖信息 |

**终极验证标准**:B/A ≥ 0.98(基于记忆回答推理问题的准确率 / 基于原文回答的准确率)。详见 [`03-evaluation.md`](03-evaluation.md) §1。

---

## 6. 关键工程问题的设计建议(摘要)

### 6.1 增量 prefill 的 KV 拼接(用户 Q5)

**推荐 MVP 配置**:

```yaml
incremental_prefill:
  strategy: "pre_alloc + streaming + radix"
  pre_alloc_kv: true           # 已有,稳定显存
  streaming:
    enabled: true
    n_sink: 4
    n_local: 2048
  radix:
    enabled: true              # 新增,多 consumer 共享前缀(用户确认 2026-08-25)
    tree_impl: "sglang-style"  # 待实现
  kivi:
    enabled: false             # MVP 不启用,显存足够
    cold_only: true            # 仅量化冷 KV(预留)
```

### 6.2 多 consumer 并发 probe(用户 Q4)

**借鉴 continuous batching**:所有请求共享 base model 的一次 forward + 同一 KV,按 lora_id 分别应用 LoRA projection,KV 只读无锁。

### 6.3 LoRA 加载与切换(用户 Q2)

**推荐架构:S-LoRA 风格多适配器并发**。详见 `02-training-design.md` 中 LoRA 训练部分。

### 6.4 层次化 KV 缓存(用户 Q1 远期)

```
MVP(当前):
  所有 KV 常驻内存 + prefix radix 共享
  ↓
中期(3-6 个月):
  hot KV 常驻 + warm KV 离线存储(可加载)
  ↓
远期(6-12 个月):
  完整层次化:hot / warm / cold 三层 + 检索层动态加载
```

---

## 7. 开放问题(需后续决策)

| # | 问题 | 状态 | 建议决策时机 |
|---|---|---|---|
| O1 | Wiki DAG 构建的具体算法(节点去重、边合并、层级推断) | 🔴 **待解决** | P1 启动前必须 |
| O2 | LoRA 探针的评估指标(如何衡量"提取了正确视角") | 🔴 **待解决** | P1 启动前必须 |
| O3 | prefix radix 的实现细节(自研 vs 引入 SGLang 依赖) | 🟡 MVP 启动前决策(已确认为 MVP 新增组件) | MVP 启动前 |
| O4 | 层次化 KV 的分层粒度(按会话 / 按主题 / 按实体) | 🟢 中期规划时 | 中期规划时 |
| O5 | 检索层的具体形态(向量检索 / 前缀树 / 混合) | 🟢 中期规划时 | 中期规划时 |
| O6 | agenticmemory 与 agenticinference 的部署形态 | 🟢 MVP 验证后 | MVP 验证后 |
| O7 | irr_estimate 的具体计算方法 | 🟡 待讨论 | P1 训练启动前 |
| O8 | `needs_reasoning_model_verification` 的触发阈值 | 🟡 待讨论 | P1 训练启动前 |
| O9 | Wiki 输出最大长度限制(分节训练) | 🟢 已有建议 | 训练实施时微调 |
| O10 | 多教师交叉验证协议 | 🟡 待讨论 | P1 训练启动前 |
| O11 | 13 字段 ↔ 涌现 schema 的训练数据转换规则 | 🟡 待讨论 | Phase 1 启动前 |
| O12 | 训练数据中"未提及"类负样本的占比 | 🟢 训练中验证 | P1 完成后 |
| O13 | 评估 LLM 的选择 | 🟡 待讨论 | MVP 启动前 |
| O14 | B/A 比值的统计显著性检验 | 🟡 待讨论 | MVP 完成后 |
| O15 | 失败案例的人工标注协议 | 🟢 实施时确定 | 评估启动前 |
| O16 | 在线监控指标与离线指标的差异 | 🟡 待讨论 | 部署前 |
| **O17-O32** | **对话特化、本体涌现、评估方法论、生产部署等扩展问题** | 🟡/🟢 见各详细文档 | **见 04-08 各文档** |

---

## 8. 已确认决策(2026-08-25)

| 决策项 | 选择 | 理由 |
|---|---|---|
| agenticmemory 形态 | KV 缓存为产品 + 多 LoRA 探针 | 用户澄清 2026-08-25 |
| 核心能力定义 | 推理无损(B/A ≥ 0.98) | 替代逐字复述方案 |
| Wiki DAG 构建位置 | `agenticmemory_training/` | 与 08a Phase 3 紧耦合 |
| prefix radix 实施 | MVP 新增组件 | 多 consumer 共享前缀 |
| 文档集结构 | 7 个文档(README + 6 个详细文档) | 增加对话扩展、本体涌现、评估方法论 |
| **对话输入支持** | **轮次分块 + 增量更新(O(1) per turn)** | **长对话必须增量,不能全量重算** |
| **对话边界** | **记忆模型不涉及对话管理/用户画像/技能复用** | **目标函数保持单一(推理无损)** |
| **MQP 协议扩展** | **MQP v3 新增 5 类对话查询(指代/约束/更新/意图/时间锚定)** | **覆盖对话特有查询需求** |
| **对话训练样本** | **新增 Type J-N 5 类对话特化样本** | **在原 Type A-I 基础上扩展** |
| **schema 来源** | **双轨:人工 schema(消费方)+ 涌现 schema(记忆侧)** | **消费方需稳定契约,记忆侧需自主发现** |
| **本体涌现范式** | **OpenIE + 信息瓶颈 + RL + 动态演化** | **四种范式互补,详见 [`05-schema-emergence.md`](05-schema-emergence.md)** |
| **训练路线** | **V1.0 宽进严出 → V2.0 重构剪枝 → V3.0 RL 涌现** | **由 [`05-schema-emergence.md`](05-schema-emergence.md) §3 定义** |
| **双系统架构** | **记忆轨(1B-3B)+ 推理轨(7B-14B)按能力差自动分层** | **详见 [`05-schema-emergence.md`](05-schema-emergence.md) §4** |
| **评估方法论** | **Probe Model + Golden Filter(不用 SOTA 直接打分)** | **避免假阳性,详见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md)** |
| **稀疏性惩罚** | **L1 + 重构损失 + 提取数量上限** | **防止"噪音淹没信号"** |
| **生产部署架构** | **vLLM + S-LoRA + 三层缓存 + 数据飞轮** | **详见 [`07-production-deployment.md`](07-production-deployment.md)** |
| **风险管理** | **集中登记在 [`08-risk-register.md`](08-risk-register.md)** | **25 项风险统一跟踪,优先解决 🔴 高风险** |
| **责任组任务规划** | **按架构/数据/训练/评估/部署拆解任务** | **详见 [`09-team-roadmap.md`](09-team-roadmap.md)** |

---

## 9. 下一步行动

**立即执行(本周)**:
1. 用户 review 本文档集,确认架构方向
2. 更新 `AGENTS.md` §12,新增 F-06(agenticmemory 架构决策)
3. 更新 `docs/agenticmemory_training/08c-p1-minimum-loop.md`,加入 Wiki DAG 构建任务

**MVP 启动前(下周)**:
4. 细化 Wiki DAG 构建算法(O1,待解决)
5. 定义 LoRA 探针评估指标(O2,待解决)
6. 决策 prefix radix 实现路径(O3)
7. 决策评估 LLM 选择(O13)

**P1 阶段(2-3 周)**:
8. 实现 Wiki DAG 构建(对接 08a Phase 3)
9. 训练 base model + 双 LoRA(session + memory)
10. 验证"同一 KV,多 LoRA 探针"的可行性
11. MVP 评估流水线启动(50 篇测试集)

---

## 10. 阅读建议

| 读者 | 推荐阅读路径 |
|---|---|
| **架构师 / 技术决策者** | 本 README → [`01-memory-model.md`](01-memory-model.md) §1-2 |
| **数据工程师** | 本 README → [`02-training-design.md`](02-training-design.md) → [`../agenticmemory_training/`](../agenticmemory_training/) |
| **训练算法工程师** | [`02-training-design.md`](02-training-design.md) §7-8(损失函数)→ [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) |
| **评估工程师** | [`03-evaluation.md`](03-evaluation.md) |
| **运行时工程师** | 本 README §3 + §6(增量 prefill)→ [`../inference-engine/`](../inference-engine/) |
| **项目负责人** | 本 README §1-5 + §9 |

---

## 11. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-25 | 初始版本:基于用户澄清,从"JSON 抽取器"重构为"KV 记忆基底" | Sisyphus(AI 助手)+ 用户 |
| **v0.2** | **2026-08-25** | **重构**:加入"推理无损"核心能力,新建 01-03 三份详细文档;README 转为索引式 | Sisyphus(AI 助手)+ 用户 |
| **v0.3** | **2026-08-25** | **新增 04**:多轮对话扩展 + 边界声明 + 增量更新 | Sisyphus(AI 助手)+ 用户 |
| **v0.4** | **2026-08-25** | **新增 05-06**:本体涌现 + 评估方法论;修复"待创建文档"占位符(04-06 已建) | Sisyphus(AI 助手)+ 用户 |
| **v0.5** | **2026-08-25** | **新增 07-08**:生产部署 + 风险登记册;修复文档结构表和修订记录不一致 | Sisyphus(AI 助手)+ 用户 |
| **v0.6** | **2026-08-25** | **新增 09**:责任组任务规划;所有文档对齐与一致性检查完成 | Sisyphus(AI 助手)+ 用户 |

---

**文档版本**: v0.6
**Owner**: AgenticMind 架构组
**下一步**: 用户 review 后启动 Wiki DAG 构建算法细化(O1)与 LoRA 评估指标设计(O2)
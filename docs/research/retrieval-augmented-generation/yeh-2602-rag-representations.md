# [2602.20091] How Retrieved Context Shapes Internal Representations in RAG

> **来源**:Yeh, S., Li, S. (UW-Madison). "How Retrieved Context Shapes Internal Representations in RAG." arXiv:2602.20091, v1 2026-02-23, v2 2026-04-16.
> **调研日期**:2026-08-30
> **调研者**:Sisyphus + Oracle 交叉核验
> **本文档归属**:`docs/research/retrieval-augmented-generation/`（RAG 问题领域）
> **关联文档**:`../soca/2026-arxiv-survey.md`（全局视角与跨论文对比）

---

## 1. 元信息

- **作者**：Samuel Yeh, Sharon Li（University of Wisconsin-Madison）
- **arXiv**：v1 2026-02-23，v2 2026-04-16（最新版）
- **被引用状态**：本套架构文档（00 §四 #22、06 §十一、99 §九）已引用并通过核验

---

## 2. 核心内容

### 2.1 研究问题

检索到的 context 如何塑造 LLM 的**内部表示**？此前研究只看 output behavior——不知内部表示变化。

### 2.2 方法：controlled experimental setup

- **数据集**：Trivia QA, NQ, Pop QA, Strategy QA（4 个 QA 数据集）
- **模型**：Gemma3-27B, Llama4-17B, Qwen3-Next-80B（**注意：是大模型 17B-80B**，不是小模型）
- **检索库**：MassiveDS (1.4 万亿 token) + Contriever top-20
- **查询难度分类**：每个 (模型, 查询) 对标记 easy / hard（无检索时是否答对）
- **文档分类**：GPT-5 判定 relevant / distracting / random
- **分析对象**：last prompt token 的 hidden states h^{q,S_q} ∈ R^{L×D} 跨所有 transformer 层

### 2.3 设置

- **Single-document**：每次一个文档（relevant / distracting / random / 无），隔离单个文档影响
- **Multiple-document**：4 文档（1 relevant + 3 distracting/random），模拟真实 RAG

---

## 3. 五大观察

### Obs 1：Random 文档导致**大**表示漂移——比 relevant 甚至 distracting 都大

PCA 可视化显示 random 文档诱导最大的 representation drift。**表示漂移与 abstention 强相关**（cos 相似度低时 abstention 率高）。

**机制**：base models 几乎无 random 文档的表示漂移；**instruction tuning 放大此现象**（base models <20% abstention，instruction-tuned >60%）。

### Obs 2：Relevant 文档保持表示基本不变——主要起"确认信号"作用

对于 easy queries，relevant 文档诱导相对小的表示偏移——responses 通常达到显著更高的 log-likelihood（p<0.001），表明**模型信心增加**。

**关键限制**：对 hard queries，relevant 文档往往**无法提供足够强信号**来有意义地改变内部表示——relevant documents often fail to sufficiently influence internal representations when parametric knowledge is lacking.

### Obs 3：多文档设置中，一个 relevant 文档足以"锚定"表示

**关键发现**：当至少有一个 relevant 文档时，无论其他文档是 distracting 还是 random，**表示保持接近 relevant-only baseline**——模型能内部压制 noise 信号。

**实践启示**：增加检索数量以提高 recall，可以是 beneficial——模型能内部处理 noise。

### Obs 4：早期层（L12）random vs 其他文档区分不明显；中后期层（L23+）random 开始分离

> Coarse semantic mismatches between query and input context are relatively easy for LLMs to identify and can be detected early in the processing pipeline.

### Obs 5：后期层（L35+）开始把 relevant 文档的表示拉回无文档状态

**核心洞见**：later layers 越来越多地强调 parametric knowledge——**这限制了 retrieved evidence 对 hard queries 的影响**。

---

## 4. 核心发现

### 4.1 Relevant 文档的"确认"作用

- 在 easy queries 上：relevant documents 通常与模型 parametric knowledge 一致 → 不"推"表示
- Responses 通常达到显著更高的 log-likelihood → 模型信心增加
- **relevant documents primarily act as confirmation signals**

### 4.2 Hard Queries 的失败模式

- For hard queries, the consistently small representation drifts indicate that **relevant documents often fail to provide a sufficiently strong signal to meaningfully alter internal representations**
- 在某些情况下，instruction-tuned LLMs 在 relevant documents 上的 error rate **比在 distracting documents 上更高**

### 4.3 多文档锚定效应

- Performance preserved or even improved if at least one relevant document is presented
- 即使加上 3 个 distracting 或 random 文档，性能仍接近 relevant-only
- LLMs can **selectively attend to informative evidence and suppress irrelevant signals** when reliable grounding is available

### 4.4 基模型 vs 指令微调对比

| Context | Base Easy | Base Hard | Instruct Easy | Instruct Hard |
|---|---|---|---|---|
| Relevant | 92.4 | 52.5 | 90.4 | 65.2 |
| Distracting | 79.5 | 14.8 | 8.5 | 0.7 |
| Random | 89.4 | 15.6 | 1.7 | 0.0 |

**关键**：instruction-tuned LLMs 在 random context 下**97.6% abstention**（vs base 3.8%）——**该行为是 instruction tuning 放大效应**，而非模型内部固有能力。

### 4.5 Unfiltered vs Filtered 对比

即使给出**完整的 20 个检索文档**（不严格过滤），性能接近只给 relevant 文档的情况——**LLMs can internally suppress noise when reliable evidence is available**，减少了 aggressive filtering 的必要性。

---

## 5. 对项目借鉴

### 5.1 对 SOCA v3-Micro-Final（`../../soca/`）

- **SOCA "三视角可解释性" 中的 SAE 视角得到直接背书**：本文证实**hidden states 包含有意义的 retrieval 相关信号**——SOCA 的 Joint SAE 设计（M12）通过解码 latent 即可观测这些信号
- **SOCA Bus M1 的"全局信息通道"作用得到支撑**：本文 Obs 5 揭示"later layer 拉回 relevant 文档的表示"——SOCA Bus 写入工作空间层（middle-to-late）正是设计来对抗这个"param dominance"——SOCA 设计理念与本文观察一致
- **SOCA M3 CausalGate "freeze" 模式的科学依据**：Obs 3 显示"1 个 relevant 文档就足以锚定表示"——SOCA M3 的 freeze 模式可设计为"仅 freeze 表示漂移大的层（random 文档层）"，保留 relevant 文档层

### 5.2 对 architectures 决策线（`../../../architectures/`）

- **直接支持 v4.6 "增加检索数量" 的判断**：04b §2.4.1 列出"v4.6 真正能打的任务"——本文 Obs 3 直接证明"增加检索数量优于严格过滤"，**与 04b 的多文档设置（"+1 relevant + 3 random"）路径一致**
- **对 v4.6 Hallucination Amplification 防线再背书**：04b §2.6 的 4 道防线（Cite or refuse / RAG conflict detector / L1 激进路由 / Hard refusal）——本文 Obs 1 直接证实"random 文档会让表示漂移到 abstention"——**这 4 道防线的核心是"识别 abstention 信号"**，与 SOCAMonitor 监控一致
- **对 v4.5 vs v4.6 边界的细化**：本文用 17B-80B 模型证实 representation 模式——但 architectures 04b 默认 Specialist 是 1.5B。**F-01 决策应纳入"表示可观测性"维度**：若选 1.5B，hidden states 信息量远低于 17B，**Bus/Monitor 设计可能需要降级**

### 5.3 对 AgenticDSL 训练链路

- **HydraForge 4 层验证器的"中间层表示"检查**：本文揭示"middle-to-late 层表示包含 retrieval 相关信号"——HydraForge L3（execution）层验证可以**直接读 hidden states 而非 output**，更精确判断 DSL 执行是否符合预期
- **AgenticDSL 的 "verification by representation" 可能性**：传统 verifier 跑 DSL output；本文提供"verification by hidden state pattern"——**HydraForge 可借鉴**为新增 L5（representation verification）层
- **AgenticDSL 多文档 context 的"锚定"策略**：本文 Obs 3 证明"1 relevant 文档锚定表示"——AgenticMind Agent loop 的 multi-document prompt 设计应**优先确保至少 1 个 relevant demonstration**，而非严格过滤所有 noise

---

## 6. 对 AgenticMind 项目整体启示

| 启示 | 落地动作 |
|---|---|
| **后期层表示是 retrieval 与 param knowledge 竞争点** | HydraForge L3 验证可读 hidden states 而非 output |
| **instruction tuning 放大 abstention** | AgenticMind 产品应避免 prompt 强制回答 |
| **增加检索数量优于严格过滤** | AgenticMind Agent loop multi-doc prompt 设计应优先保证至少 1 relevant |
| **多文档锚定可降复杂** | 单 relevant 文档已足够稳定表示，无需复杂的相关性过滤 |

---

## 7. 核心数据快查

### 7.1 性能对比（Gemma3-27B on Trivia QA）

| Context | Base Easy | Base Hard | Instruct Easy | Instruct Hard |
|---|---|---|---|---|
| Relevant | 92.4 | 52.5 | 90.4 | 65.2 |
| + Distracting | 79.5 | 14.8 | 82.6 | 57.1 |
| + Random | 89.4 | 15.6 | 87.7 | 60.2 |
| Distracting only | 79.5 | 14.8 | 8.4 | 0.7 |
| Random only | 89.4 | 15.6 | 1.7 | 0.0 |

### 7.2 PCA 可视化关键发现

- **Obs 1**：Random 文档的表示与 no-document baseline 距离最大
- **Obs 2**：Relevant 文档的表示与 no-document baseline 接近（"确认"作用）
- **Obs 4**：早期层（L12）不同文档类型的表示重叠；后期层（L23+）random 开始分离
- **Obs 5**：后期层（L35+）relevant 文档表示被拉回 no-document baseline

### 7.3 Abstention 行为

- **Base models**：<20% abstention（random context 下）
- **Instruction-tuned models**：>60% abstention（random context 下）——**97.6% for Gemma3-27B with random context**
- **机制**：instruction tuning 在 random context 下触发内部 abstention 模式

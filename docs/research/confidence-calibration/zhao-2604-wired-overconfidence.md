# [2604.01457] Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs

> **来源**:Zhao, T., He, Y., Zheng, W., Zhang, Y., Chen, C. (U. Virginia). "Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs." arXiv:2604.01457, v1 2026-04-01, v3 2026-07-27. **COLM 2026** 已接收。
> **调研日期**:2026-08-30
> **调研者**:Sisyphus + Oracle 交叉核验
> **本文档归属**:`docs/research/confidence-calibration/`（置信度校准问题领域）
> **关联文档**:`../soca/2026-arxiv-survey.md`（全局视角与跨论文对比）

---

## 1. 元信息

- **作者**：Tianyi Zhao, Yinhan He, Wendy Zheng, Yujie Zhang, Chen Chen（University of Virginia）
- **会议**：**COLM 2026**（Conference on Language Modeling）已接收
- **arXiv**：v1 2026-04-01，v3 2026-07-27（最新版）
- **被引用状态**：本套架构文档（00 §四 #24、06 §四.2、99 §九）已引用并通过核验

---

## 2. 核心内容

### 2.1 研究问题

为什么 LLM 不仅答错，而且**自信地答错**？verbalized overconfidence 的内部机制是什么？

### 2.2 两阶段置信度提取设置（关键创新）

- 第一步：模型回答问题
- 第二步：模型自我报告置信度（0-99 整数）
- 这两阶段隔离让置信度 verbalization 成为可独立追踪的计算阶段

### 2.3 核心方法（三步走）

1. **TSLD（Target Set Logit Difference）**：定义 high-confidence set H = {70,75,80,85,90,99} 和 low-confidence set L = {0,10,15,20,25,30} 的 logit 差距 → 内部可微信号。**ΔTSLD 与原始 verbalized confidence 强相关**（Qwen2.5-3B r=0.815, Llama-3.2-3B r=0.734）。
2. **Truth-Injection Counterfactual Design**：clean 保留错误答案，corrupted 替换为 ground truth——构造最小因果对。
3. **EAP-IG（Edge Attribution Patching with Integrated Gradients）**：发现 Confidence Mover Circuit (CMC)。

### 2.4 实验

- **模型**：Qwen2.5-3B-Instruct + Llama-3.2-3B-Instruct
- **数据集**：PopQA + MMLU + NQOpen

### 2.5 五大核心发现

1. **CMC 集中在 middle-to-late 层**——Qwen 主要是 MLP block（24-35 层），Llama 是 attention head + MLP 混合。
2. **跨数据集保守性 ≥80%**——同一模型在不同数据集上 CMC 高度重合，说明这是**模型内部机制**而非任务特异。
3. **Faithfulness S-curve**：保留 top-3000 edges（<0.6% 总边数）即可还原 85%+ 原始 overconfidence 信号。
4. **Necessity**：ablation CMC 后 TSLD 减少 74-79%（per-model）——**真有因果关系**。
5. **干预有效**：mean ablation 或 activation steering（α=0.5-0.6）可将 ECE 从 0.49 降至 0.11（Qwen）或 0.57→0.02（Llama）。

### 2.6 关键洞见

**verbalized confidence 电路与 factual retrieval 电路结构性不重合**——前者集中在 middle-to-late 层（logits 附近），后者跨越更长路径（earlier-to-middle）→ 说明 verbalized confidence 是**结构性独立于事实性**的 compact 电路，**这正是 overconfidence 的机制根源**。

---

## 3. 创新点

| 创新 | 关键差异 |
|---|---|
| **TSLD 指标** | 此前用 raw logit mean；本工作用 high-low set difference 抑制 global shift 与"抑制-陷阱" |
| **Truth-Injection Counterfactual** | 此前 Ji 2025 用 verbal uncertainty 的 linear feature；本工作用 ground-truth 注入构造**最小因果对**，避免分布异常污染 |
| **CMC 的 Faithfulness + Completeness 双向验证** | 此前工作只做 faithfulness；本工作加 completeness（ablation 后 TSLD 减少）证明**真的必要** |
| **干预策略可调**：mean ablation vs activation steering | 提供 α 参数连续可调，更具实用性 |
| **揭示 verbal confidence 与 factual retrieval 电路**结构性不重合** | 直接解释 verbalized overconfidence 的根源 |

---

## 4. 对项目借鉴

### 4.1 对 SOCA v3-Micro-Final（`../../soca/`）

- **SOCAMonitor M15 的科学背书**：SOCA §八 的 SOCAMonitor 设计包含 z-score 异常检测（基于 bus_states + router_entropies + gate_values + sae_max_activations）；本文**实证了"内部状态可作为置信度信号"**——SOCAMonitor 的科学基础得到 2026 年最新机制研究支持。
- **MonitorSlot M2 + Bus M1 的中间层定位**：SOCA 的 Bus 监控信号（γ=0.99 EMA）在中后层积累；本文 CMC 也集中在 middle-to-late 层——**SOCA 的监控位置选择与本文发现一致**，可作为 SOCA §一"3 视角可解释性"的实证支持。
- **CausalGate M3 "replace" 模式的科学合理化**：M3 允许干预特定层输出。本文 CMC 干预（mean ablation 或 steering）证明了**特定层的中间表示确实影响置信度输出**——这正是 M3 的 replace 模式的设计意图。

### 4.2 对 architectures 决策线（`../../../architectures/`）

- **元认知闭环置信度机制的科学依据**：06 §六的元认知闭环靠 "per-step confidence trigger"——本文证明 1.5B 置信度本身有问题（verbalized overconfidence），**1.5B 触发检索的信号本身就需要校准**。
- **对 v4.7 Kill Criteria 的启示**：v4.7 Kill Criteria #5 是 "confidence head ECE 持续 > 0.25 砍项目"——本文提供的**电路级 ECE 校准**（activation steering α=0.5）可作为**活体校准方案**，让 v4.7 有机会在 ECE > 0.25 后仍通过电路干预救回。
- **对 Confidence-Triggered Loop 的改造建议**：06 §三 设计 EAGLE 式后验校准作为 baseline——**本文提供更细粒度的"在特定层 + 特定电路"做 steering"**的方案**，可作为 EAGLE 的替代或增强。

### 4.3 对 AgenticDSL 训练链路

- **HydraForge 4 层验证器的置信度信号**：本文证明 verbalized confidence 与事实性电路**结构不重合**——**HydraForge 应直接读取 hidden states 而非依赖模型 verbalized output**。这是 SOCAMonitor M15 设计理念的延伸，应推广到 HydraForge 验证器。
- **AgenticDSL 的"拒答置信度"机制**：本文发现 instruction-tuned 模型在 random context 下 97.6% abstention（vs base 3.8%）——**这与 AgenticDSL 的"高 confidence 才输出，低 confidence 拒答"机制一致**。本文提供了量化基线：应监控 abstention 率在 ≤20% 区间，避免模型过度保守。

---

## 5. 对 AgenticMind 项目整体启示

| 启示 | 落地动作 |
|---|---|
| **verbalized confidence 不是事实性** | 所有 AgenticMind 产品的置信度展示应**同时**显示 verbalized confidence + circuit-level 信号（如 SAE active features） |
| **中间层电路是干预靶点** | 任何"调小模型置信度"的需求应优先尝试 circuit-level intervention（activation steering），再考虑 post-hoc calibration |
| **instruction tuning 放大 overconfidence** | AgenticMind 的 prompt engineering 应避免"请自信地回答"类话术——会进一步放大 overconfidence |
| **电路级 calibration 可挽救 v4.7** | 若 F-01 决策后跑 v4.7 元认知闭环，先用本文方法做一次 circuit-level ECE 校准，可提高闭环成功率 |

---

## 6. 核心数据快查

### 6.1 校准效果（PopQA）

| 模型 | 方法 | α | ECE | Brier | ECE 改进 | Brier 改进 |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | baseline | — | 0.492 | 0.430 | — | — |
| | mean ablation | — | 0.295 | 0.244 | 40.0% | 43.1% |
| | steering | 0.3 | 0.245 | 0.193 | 50.3% | 55.1% |
| | **steering** | **0.5** | **0.111** | **0.130** | **77.5%** | **69.7%** |
| | steering | 0.6 | 0.145 | 0.146 | 70.7% | 66.1% |
| Llama-3.2-3B-Instruct | baseline | — | 0.570 | 0.495 | — | — |
| | **mean ablation** | — | **0.018** | **0.161** | **96.9%** | **67.5%** |
| | steering | 0.5 | 0.063 | 0.146 | 88.9% | 70.5% |

### 6.2 Top-10 CMC 组件（Qwen2.5-3B）

| Rank | PopQA | MMLU | NQOpen |
|---|---|---|---|
| 1 | m30 | m35 | m30 |
| 2 | m35 | m30 | m35 |
| 3 | m26 | m27 | m31 |
| 4 | m31 | a24.h5 | m26 |
| 5 | m25 | m24 | m25 |
| 6 | m27 | m28 | m27 |
| 7 | m0 | m26 | m24 |
| 8 | m28 | m25 | a24.h5 |
| 9 | m24 | a27.h1 | m28 |
| 10 | a24.h5 | m34 | a24.h0 |

**关键观察**：Top-10 组件**几乎全部是 MLP block（24-35 层）**——验证了"middle-to-late 层是 overconfidence 电路核心"的结论。

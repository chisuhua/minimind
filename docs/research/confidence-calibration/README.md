# Confidence Calibration 研究

> **目录定位**:`docs/research/confidence-calibration/` —— 围绕 **LLM verbalized confidence（口头报告的置信度）的过度自信问题** 的研究文档。
>
> **核心问题**:为什么 LLM 不仅答错，而且"自信地答错"？如何从机制层面理解和修复 verbalized overconfidence？
>
> **关联度**:⭐⭐⭐⭐ **高** —— 直接关联 SOCA v3-Micro-Final M15 SOCAMonitor 设计（元认知闭环的置信度触发机制）与 architectures 06 元认知闭环的 v4.7 Kill Criteria #5（confidence head ECE 校准）。

---

## 📁 论文清单（1 篇，2026 年）

| arXiv ID | 标题 | 作者 | v1 日期 | 模型规模 | 问题角度 |
|---|---|---|---|---|---|
| [2604.01457](https://arxiv.org/abs/2604.01457) | Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs | Tianyi Zhao et al. (U. Virginia) | 2026-04-01 (v3 2026-07-27) | 3B | **机制层面**：circuit 定位、causal validation、activation steering 干预 |

> **会议**:**COLM 2026**（Conference on Language Modeling）已接收

---

## 🎯 关键发现

### 核心创新方法

**TSLD（Target Set Logit Difference）**：
- 定义 high-confidence set H = {70,75,80,85,90,99} 和 low-confidence set L = {0,10,15,20,25,30}
- TSLD = mean(logits[H]) − mean(logits[L])，在 final prompt position 计算
- **ΔTSLD 与原始 verbalized confidence 强相关**：Qwen2.5-3B r=0.815，Llama-3.2-3B r=0.734

### 五大核心发现

1. **Confidence Mover Circuit (CMC) 集中在 middle-to-late 层**：Qwen 主要是 MLP block（24-35 层），Llama 是 attention head + MLP 混合
2. **跨数据集保守性 ≥80%**：同一模型在不同数据集上 CMC 高度重合——**模型内部机制**而非任务特异
3. **Faithfulness S-curve**：保留 top-3000 edges（<0.6% 总边数）即可还原 85%+ 原始 overconfidence 信号
4. **Necessity**：ablation CMC 后 TSLD 减少 74-79%——**真有因果关系**
5. **干预有效**：
   - Qwen2.5-3B：mean ablation 或 steering (α=0.5-0.6) → ECE 0.49 → **0.11**（↓77.5%）
   - Llama-3.2-3B：mean ablation → ECE 0.57 → **0.02**（↓96.9%）

### 关键洞见

**verbalized confidence 电路与 factual retrieval 电路结构性不重合**：
- 前者集中在 middle-to-late 层（logits 附近）
- 后者跨越更长路径（earlier-to-middle）
- → verbalized confidence 是**结构性独立于事实性**的 compact 电路，**这正是 overconfidence 的机制根源**

---

## 📐 对 AgenticMind 项目借鉴

### 对 SOCA v3-Micro-Final（`../soca/`）

- **SOCA M15 SOCAMonitor 的科学基础**：本文证实"内部状态可作为置信度信号"——SOCA M15 的 z-score 异常检测（基于 bus_states + router_entropies + gate_values + sae_max_activations）有 2026 年最新机制研究背书
- **SOCA Bus M1 + MonitorSlot M2 的中间层定位**：本文 CMC 也集中在 middle-to-late 层——SOCA Bus 在工作空间层积累的设计与本文发现一致
- **SOCA M3 CausalGate "replace" 模式的科学依据**：M3 允许干预特定层输出——本文电路干预证明**特定层表示确实影响置信度输出**，正是 M3 设计意图

### 对 architectures 06 元认知闭环（`../../architectures/`）

- **直接支持 v4.7 Kill Criteria #5**：若 1.5B model 上 confidence head ECE 持续 > 0.25（砍项目条件），**本文提供电路级 ECE 校准作为活体校准方案**
- **可作为 EAGLE 后验校准的替代/增强**：EAGLE 是输出层后验校准（ECE 0.49→0.31），本文 activation steering α=0.5 是机制层干预（ECE 0.49→0.11）——**后者效果更显著**
- **v4.7 元认知闭环 confidence 触发的电路级校准路径**：01 §2.1 提到 confidence head 训练，本文提供"无需 retraining"的 circuit intervention 替代方案

### 对 AgenticDSL 训练链路

- **HydraForge 验证器应直接读 hidden states 而非依赖 verbalized output**：本文证明 verbalized confidence 与事实性电路**结构不重合**——这是 SOCA 监控设计理念的延伸
- **AgenticMind 产品置信度展示应同时显示 verbalized + circuit-level 信号**：避免"答错但自信"误导用户
- **prompt engineering 避免"请自信地回答"**：本文证明 instruction tuning 本身放大了 overconfidence

---

## 📊 与项目决策的对应

| AGENTS.md 决策 | 本目录论文 | 行动建议 |
|---|---|---|
| **F-01 基模型选择** | #2 Zhao | 若选 3B 及以下，**verbalized confidence 不可靠**——F-01 选 7B 进一步证实 |
| **元认知闭环 confidence 触发** | #2 Zhao | **激活 steering (α=0.5) 比 post-hoc calibration 更有效**——可作为 HydraForge 验证器备选 |
| **产品置信度展示** | #2 Zhao | 同时显示 verbalized + circuit-level 信号，避免过度自信误导 |

---

## 🚀 推荐阅读顺序

| 读者 | 阅读路径 |
|---|---|
| **AgenticMind 产品设计者** | §一 Introduction → §三 Circuit Attribution → §四 Intervention → §六 Discussion |
| **SOCA M15 SOCAMonitor 设计者** | §二 TSLD 指标 → §三 Circuit Attribution → §四 Intervention |
| **HydraForge 验证器设计者** | §四 Intervention → §五 Results → §六 Discussion（与 AGENTS.md F-04 对照） |
| **元认知闭环 v4.7 Kill Criteria 评估者** | §三 + §四 + §六 Discussion |

---

## 📅 调研版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-30 | 初版，1 篇论文调研完成（原 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) §2 拆分） |

---

> **本目录定位**:作为 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) 的**分主题深度版**——survey 提供全局视角与跨论文对比，本目录提供单篇论文的完整内容/创新/借鉴分析。

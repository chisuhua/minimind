# Reinforcement Learning 研究

> **目录定位**:`docs/research/reinforcement-learning/` —— 围绕 **LLM 强化学习训练的稳定性** 研究文档。
>
> **核心问题**:RLVR 依赖外部监督（人类标注 / gold answers / PRM）——不可扩展。RLIF（内部反馈 RL）用单一内部 reward 必然崩塌（entropy collapse、reward hacking）。如何设计稳定的多奖励 RLIF 训练框架？
>
> **关联度**:⭐⭐⭐⭐ **高** —— 直接关联 AGENTS.md F-03 RLIF 路线；提供 AgenticDSL RL 训练的具体实施方案。

---

## 📁 论文清单（1 篇，2026 年）

| arXiv ID | 标题 | 作者 | v1 日期 | 模型规模 | 问题角度 |
|---|---|---|---|---|---|
| [2605.22620](https://arxiv.org/abs/2605.22620) | Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework | Shourov Joarder et al. (BUET/WVU/UCL) | 2026-05-21 | 1.5B-3B | **多奖励 + GDPO + KL-Cov** 的稳定 RLIF 训练 |

---

## 🎯 关键发现

### 核心问题

RLVR（Reinforcement Learning with Verifiable Rewards）依赖外部监督——不可扩展。RLIF（Reinforcement Learning with Internal Feedback）只依赖模型自身信号，但现有方法用**单一内部奖励**导致：
- **Reward hacking**：模型利用奖励信号而非真正改进
- **Entropy collapse**：单 reward（self-certainty）导致熵下降、模板化输出
- **推理结构退化**：单 reward（cluster voting）导致 answer-mode collapse

### 方法：多奖励 RLIF 框架

**两种互补奖励**：
1. **Answer-Level Reward（cluster voting）**：rollout 完成 → 提取最终答案 → 按等价性聚类 → 每 rollout 奖励与同 cluster 大小成比例
2. **Completion-Level Reward（self-certainty）**：从当前策略计算 self-certainty = DKL(U || πθ)，本质是熵最小化

**GDPO Normalization（per-channel z-scoring）**：
- 解决 scale dominance——本文通过 logprob-space 梯度诊断证明，naïve combination 让 high-variance self-certainty channel **压制 cluster channel 一个数量级**
- 每个 channel 独立在 group of G completions 内做 z-score

**KL-Cov（KL-Covariance）**：
- 不是全局 KL 惩罚——只针对**顶部 2% 高协方差 token**（Cov(logπ, A)）施加额外 KL 惩罚
- 这 2% token 负责了 ~95% 熵减少（参考 Cui 2025 KL-Cov 原始工作）

### 实验结果（step 160 checkpoint）

| 模型 | 方法 | GSM8K | MATH500 | MMLU-Pro | LiveCodeBench v6 | CRUXEval-O |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | GRPO-GT（监督） | 74.9 | 55.2 | 31.6 | 1.7 | 24.9 |
| | INTUITOR（单 reward） | 22.4 | 22.6 | 24.5 | 0.8 | 12.6 |
| | Multi-reward | 69.5 | 46.0 | 27.7 | 3.3 | 21.8 |
| | **Multi-reward + KL-Cov** | **68.3** | **51.6** | **29.9** | **2.6** | **20.5** |
| Qwen2.5-3B | GRPO-GT | 84.9 | 64.4 | 40.3 | 6.4 | 41.0 |
| | Multi-reward + KL-Cov | 80.7 | 62.4 | 39.0 | 7.7 | 38.3 |

**关键发现**：
- 1.5B Multi-reward + KL-Cov vs INTUITOR：+45.9pp GSM8K、+29.0pp MATH500、+5.4pp MMLU-Pro
- 训练稳定性：Multi-reward + KL-Cov 全程稳定；INTUITOR 在 step 60 后**单调下降至接近 0**

### Ablation 关键发现

- **Cluster-only**：GSM8K 75% → 0% **突变崩溃**（step 120-140）
- **Self-certainty-only**：entropy 2.4 → 0.08 nats，completion length 990 → 100 tokens
- **Multi-reward 无 KL-Cov**：GSM8K step 240-260 也崩溃（56.6% → 1.1%）
- **Multi-reward + KL-Cov**：所有 3 个 βcov sweep（0.0005, 0.05, 0.1）均稳定

---

## 📐 对 AgenticMind 项目借鉴

### 对 AgenticMind F-03 RLIF 训练路线（`../../AGENTS.md`）

- **Multi-Reward RLIF 是 F-03 的具体实施方案**：本文的 answer-cluster voting reward + self-certainty reward + GDPO + KL-Cov 框架，**可直接迁移到 AgenticDSL 训练**——cluster voting 用 AgenticDSL 的"答案格式正确性"（无 verifier 但有格式约束），self-certainty 用 token-level 熵
- **避免 R1-Zero 路线崩溃的具体方案**：KL-Cov regularization 是 "Top 2% 高协方差 token 额外 KL 惩罚"——AgenticMind RLIF 训练可直接采用此正则化，避免"熵崩溃 + 短模板"问题
- **AgenticDSL 训练数据可借鉴的"多奖励"信号**：本文的"answer-level + completion-level"两奖励——对应 AgenticDSL 的"DSL 语法正确（结构层）+ 生成 tokens 流畅度（语义层）两个奖励维度

### 对 architectures 决策线（`../../architectures/`）

- **直接支持 06 §六元认知闭环的"RLAIF 不依赖外部 reward"判断**：本文证明**纯内部 reward + 适当正则化**可达到接近 GRPO-GT 的水平——architectures "不要走 R1-Zero 纯 RL 路线" 的判断更精确地说应该是"不要走单 reward 纯 RL 路线"
- **v4.7 元认知闭环 Kill Criteria #5 的缓解路径**：若 1.5B model 上 confidence head ECE 持续 > 0.25（v4.7 砍项目条件），本文提供**GDPO + KL-Cov 框架作为 RLIF 训练方案**——可在 AgenticMind RLIF 路线实施
- **拒绝 06 §四 §3 "Path 路径 Z：纯 RL 路线" 的细化**：本文证明**单 reward 纯 RL 必然崩溃**，但**多 reward + KL-Cov 可稳定**——99 §三 的"路径 Z 拒绝"应补充"但 multi-reward + KL-Cov 形式的 RLIF 不在拒绝之列"

### 对 SOCA v3-Micro-Final（`../soca/`）

- **SOCA 08 路由器稳定性的新训练信号**：KL-Cov 关注高协方差 token——SOCA 08 §二的"专家容量保护"（Layer 5）可借鉴**KL-Cov 式"高协方差 expert 不奖励"**机制
- **SOCA 02-03 的"小模型 + RLIF" 可行性**：本文 1.5B 已能达到 GSM8K 68%——这远高于 SOCA 假设的"小模型难 RL"——SOCA 路由训练可考虑加入类似 multi-reward RLIF 阶段

---

## 📊 与项目决策的对应

| AGENTS.md 决策 | 本目录论文 | 行动建议 |
|---|---|---|
| **F-03 RLIF 训练路线** | #5 Joarder | 采用 multi-reward + GDPO + KL-Cov 框架——**本文是 F-03 的具体技术栈** |
| **拒绝纯 RL 路线** | #5 Joarder | 单 reward 必然崩溃，multi-reward + KL-Cov 是稳定替代 |
| **SOCA 路由训练** | #5 Joarder | 借鉴 KL-Cov 关注高协方差 token 的思路 |
| **HydraForge 验证器设计** | #5 Joarder | 验证器本身也可借鉴"双奖励"思路（执行正确性 + 结构正确性） |

---

## 🚀 推荐阅读顺序

| 读者 | 阅读路径 |
|---|---|
| **F-03 RLIF 训练路线实施者** | Abstract → §3 Method（multi-reward + GDPO + KL-Cov）→ §4 Experiments（1.5B 结果）→ §5 Discussion |
| **AgenticDSL RL 训练设计者** | §3.2 cluster voting → §3.4 GDPO normalization → §3.5 KL-Cov regularization → §4.2 main results |
| **架构师** | §1 Introduction → §4.3 Ablation Study（单 reward 必崩证据）→ 对照 99 §三 "纯 RL 路线拒绝" |
| **SOCA 路由训练者** | §3.5 KL-Cov regularization → §4.3 Ablation Study（cluster-only 崩溃模式）→ 对照 SOCA 08 F1-F7 |

---

## 📅 调研版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-30 | 初版，1 篇论文调研完成（原 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) §5 拆分） |

---

> **本目录定位**:作为 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) 的**分主题深度版**——survey 提供全局视角与跨论文对比，本目录提供单篇论文的完整内容/创新/借鉴分析。

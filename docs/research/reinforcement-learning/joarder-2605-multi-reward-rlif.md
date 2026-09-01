# [2605.22620] Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework

> **来源**:Joarder, S., Sikdar, D., Akash, A. H., Bhattarai, B., Gyawali, P. (BUET/WVU/Aberdeen/Fogsphere/UCL). "Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework." arXiv:2605.22620, v1 2026-05-21.
> **调研日期**:2026-08-30
> **调研者**:Sisyphus + Oracle 交叉核验
> **本文档归属**:`docs/research/reinforcement-learning/`（强化学习训练稳定性问题领域）
> **关联文档**:`../soca/2026-arxiv-survey.md`（全局视角与跨论文对比）

---

## 1. 元信息

- **作者**：Shourov Joarder (BUET), Diganta Sikdar (BUET), Ahsan Habib Akash (WVU), Binod Bhattarai (Aberdeen/Fogsphere/UCL), Prashnna Gyawali (WVU)
- **arXiv**：v1 提交 2026-05-21 (237 KB)
- **被引用状态**：本套架构文档（00 §四、04b §五、99 §九）已引用并通过核验
- **本论文与项目高度相关**：标题即含 **Multi-Reward RLIF**，与 AgenticMind F-03（RLIF 路线）直接对应

---

## 2. 核心内容

### 2.1 研究问题

RLVR（Reinforcement Learning with Verifiable Rewards）依赖外部监督（人类标注 / gold answers / PRM）——**不可扩展**。RLIF（Reinforcement Learning from Internal Feedback）只依赖模型自身信号，但现有方法用**单一内部奖励**导致：
- **Reward hacking**：模型利用奖励信号而非真正改进
- **Entropy collapse**：单 reward（self-certainty）导致熵下降、模板化输出
- **推理结构退化**：单 reward（cluster voting）导致 answer-mode collapse

### 2.2 方法：多奖励 RLIF 框架

**两种互补奖励**：

1. **Answer-Level Reward（cluster voting）**：rollout 完成 → 提取最终答案 → 按等价性聚类 → 每 rollout 奖励与同 cluster 大小成比例

   $$R_{\text{cluster}}(a_g) = \frac{|\{g' \in \{1, \ldots, G\} : a_{g'} \equiv a_g\}|}{n_{\text{parseable}}}$$

2. **Completion-Level Reward（self-certainty）**：从当前策略计算 self-certainty = DKL(U || πθ)，本质是熵最小化

   $$R_{\text{cert}}(c_g) = \frac{1}{T_g}\sum_{t=1}^{T_g}D_{\text{KL}}\!\Big(\mathcal{U} \;\Big\|\; \pi_{\theta}(\cdot \mid q, c_{g,<t})\Big)$$

### 2.3 GDPO Normalization（per-channel z-scoring）

解决 scale dominance——本文通过 logprob-space 梯度诊断证明，naïve combination 让 high-variance self-certainty channel **压制 cluster channel 一个数量级**。

$$\hat{R}_{k}(g) = \frac{R_{k}(g) - \mu_k}{\sigma_k + \epsilon}$$

其中 μk = mean(Rk(g))、σk = std(Rk(g)) 在 G 个 completions 的 group 内计算。

**组合优势**：$A_g = \alpha \hat{R}_{\text{cluster}} + \beta \hat{R}_{\text{cert}}$（默认 α=β=0.5）

### 2.4 KL-Cov Regularization

不是全局 KL 惩罚——只针对**顶部 2% 高协方差 token**（Cov(logπ, A)）施加额外 KL 惩罚：

$$\text{Cov}(y_i) = \left(\log\pi_{\theta}(y_i \mid y_{<i}) - \frac{1}{N}\sum_{j=1}^{N}\log\pi_{\theta}(y_j \mid y_{<j})\right) \cdot \left(A_i - \frac{1}{N}\sum_{j=1}^{N}A_j\right)$$

这 2% token 负责了 ~95% 熵减少（参考 Cui 2025 KL-Cov 原始工作）——目标 token 集合：

$$\mathcal{I}_{\text{KL}} = \big\{i \in \{1, \ldots, N\} \;\big|\; \text{Rank}\!\big(\text{Cov}(y_i)\big) \leq k \cdot N\big\}, \quad k = 0.02$$

Per-token loss：
$$\mathcal{L}(y_i;\theta) = \begin{cases} -\min\!\big[\rho_i A_i,\;\text{clip}(\rho_i,1{-}\epsilon,1{+}\epsilon)\,A_i\big] & \text{if }i \notin \mathcal{I}_{\text{KL}} \\ -\rho_i A_i + \beta_{\text{cov}}\big|\log\rho_i\big| & \text{if }i \in \mathcal{I}_{\text{KL}} \end{cases}$$

---

## 3. 实验结果（step 160 checkpoint）

| 模型 | 方法 | GSM8K | MATH500 | MMLU-Pro | LiveCodeBench v6 | CRUXEval-O |
|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B** | GRPO-GT（监督） | 74.9 | 55.2 | 31.6 | 1.7 | 24.9 |
| | INTUITOR（单 reward） | 22.4 | 22.6 | 24.5 | 0.8 | 12.6 |
| | Multi-reward | 69.5 | 46.0 | 27.7 | 3.3 | 21.8 |
| | **Multi-reward + KL-Cov** | **68.3** | **51.6** | **29.9** | **2.6** | **20.5** |
| **Qwen2.5-3B** | GRPO-GT | 84.9 | 64.4 | 40.3 | 6.4 | 41.0 |
| | Multi-reward + KL-Cov | 80.7 | 62.4 | 39.0 | 7.7 | 38.3 |

### 3.1 关键发现

- **1.5B Multi-reward + KL-Cov vs INTUITOR**：+45.9pp GSM8K、+29.0pp MATH500、+5.4pp MMLU-Pro
- **训练稳定性**：Multi-reward + KL-Cov 全程稳定（340 steps）；INTUITOR 在 step 60 后**单调下降至接近 0**

---

## 4. Ablation Study

### 4.1 单/多奖励分析（无 KL-Cov）

| 设置 | 训练崩塌时间 | 崩塌模式 |
|---|---|---|
| **Cluster-only** | GSM8K step 120-140（75% → 0%） | answer-mode collapse：entropy 上升，生成变长 |
| **Self-certainty-only**（INTUITOR） | step 60 起持续下降 | token-mode collapse：entropy 2.4 → 0.08 nats，length 990 → 100 tokens |
| **Multi-reward 无 KL-Cov** | GSM8K step 240-260（56.6% → 1.1%） | 双崩塌模式混合 |
| **Multi-reward + KL-Cov** | 全程稳定（340 steps） | none |

### 4.2 KL-Cov βcov Sweep

| βcov | 稳定性 |
|---|---|
| 0.0005 | ✅ 稳定 |
| 0.05（默认） | ✅ 稳定 |
| 0.1 | ✅ 稳定 |
| 无 KL-Cov | ❌ step 240 崩溃 |

**KL-Cov 不是高度敏感的超参数**——0.0005 到 0.1 区间均稳定。

---

## 5. 对项目借鉴

### 5.1 对 AgenticMind F-03 RLIF 训练路线（`../../../../AGENTS.md`）

- **Multi-Reward RLIF 是 F-03 的具体实施方案**：本文的 answer-cluster voting reward + self-certainty reward + GDPO + KL-Cov 框架，**可直接迁移到 AgenticDSL 训练**——cluster voting 用 AgenticDSL 的"答案格式正确性"（无 verifier 但有格式约束），self-certainty 用 token-level 熵
- **避免 R1-Zero 路线崩溃的具体方案**：KL-Cov regularization 是 "Top 2% 高协方差 token 额外 KL 惩罚"——AgenticMind RLIF 训练可直接采用此正则化，避免"熵崩溃 + 短模板"问题
- **AgenticDSL 训练数据可借鉴的"多奖励"信号**：本文的"answer-level + completion-level"两奖励——对应 AgenticDSL 的"DSL 语法正确（结构层）+ 生成 tokens 流畅度（语义层）两个奖励维度

### 5.2 对 architectures 决策线（`../../../architectures/`）

- **直接支持 06 §六元认知闭环的"RLAIF 不依赖外部 reward"判断**：本文证明**纯内部 reward + 适当正则化**可达到接近 GRPO-GT 的水平——architectures "不要走 R1-Zero 纯 RL 路线" 的判断更精确地说应该是"不要走单 reward 纯 RL 路线"
- **v4.7 元认知闭环 Kill Criteria #5 的缓解路径**：若 1.5B model 上 confidence head ECE 持续 > 0.25（v4.7 砍项目条件），本文提供**GDPO + KL-Cov 框架作为 RLIF 训练方案**——可在 AgenticMind RLIF 路线（AGENTS.md F-03）实施
- **拒绝 06 §四 §3 "Path 路径 Z：纯 RL 路线" 的细化**：本文证明**单 reward 纯 RL 必然崩溃**，但**多 reward + KL-Cov 可稳定**——99 §三 的"路径 Z 拒绝"应补充"但 multi-reward + KL-Cov 形式的 RLIF 不在拒绝之列"

### 5.3 对 SOCA v3-Micro-Final（`../../soca/`）

- **SOCA 08 路由器稳定性的新训练信号**：KL-Cov 关注高协方差 token——SOCA 08 §二的"专家容量保护"（Layer 5）可借鉴**KL-Cov 式"高协方差 expert 不奖励"**机制
- **SOCA 02-03 的"小模型 + RLIF" 可行性**：本文 1.5B 已能达到 GSM8K 68%——这远高于 SOCA 假设的"小模型难 RL"——SOCA 路由训练可考虑加入类似 multi-reward RLIF 阶段

---

## 6. 对 AgenticMind 项目整体启示

| 启示 | 落地动作 |
|---|---|
| **单 reward 纯 RLIF 必崩** | AgenticMind RLIF 路线（F-03）必须采用 multi-reward 框架 |
| **GDPO 是必需技术组件** | 任何 RLIF 实现都应包含 per-channel normalization |
| **KL-Cov 关键 2% token 决定稳定性** | AgenticMind 的 RLIF 训练应监控 top 2% 高协方差 token 的熵变化 |
| **Math → Code 跨域泛化有效** | AgenticDSL 在 code 推理任务上的潜力已被 multi-reward RLIF 间接证明（训练数学，code 测试不掉点） |

---

## 7. 核心数据快查

### 7.1 训练硬件

- 44 × NVIDIA GH200 120GB GPUs
- 全 fine-tuning（GRPO 基优化器）
- 学习率 3×10⁻⁶ + cosine scheduling
- G=7 rollouts/prompt，max completion length 3072 tokens
- γ=0.005（GRPO KL 系数）
- α=β=0.5（multi-reward 权重）
- k=0.02（KL-Cov top-2% 阈值）
- βcov=0.05（KL-Cov 惩罚强度）
- 40 epochs（340 steps），batch=882 prompts/step

### 7.2 基模型 + 数据

- **基模型**：Qwen2.5-1.5B-Base + Qwen2.5-3B-Base
- **训练数据**：MATH-lighteval（unlabeled）
- **框架**：huggingface-trl

### 7.3 关键数字对比

| 维度 | 1.5B 单 reward (INTUITOR) | 1.5B multi-reward + KL-Cov | 3B multi-reward + KL-Cov |
|---|---|---|---|
| GSM8K | 22.4% | **68.3%** | 80.7% |
| MATH500 | 22.6% | **51.6%** | 62.4% |
| MMLU-Pro | 24.5% | **29.9%** | 39.0% |
| 训练稳定性 | ❌ step 60 崩塌 | ✅ 全程稳定 | ✅ 全程稳定 |

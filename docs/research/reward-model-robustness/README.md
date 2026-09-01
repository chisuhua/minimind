# Reward Model Robustness 研究

> **目录定位**:`docs/research/reward-model-robustness/` —— 围绕 **Reward Model（特别是 Process Reward Model / PRM）在 label-preserving 扰动下的稳健性** 研究文档。
>
> **核心问题**:PRM 在自然分布上表现良好，但在保持最终答案不变的结构性扰动下是否仍然稳健？不同 PRM 的失败模式有何差异？
>
> **关联度**:⭐⭐⭐⭐ **高** —— 直接验证 architectures 99 §三 "否决 PRM 路线" 的判断；为 AgenticMind F-04 "不依赖 PRM，优先 Engine-Native Verification" 提供最新实证。

---

## 📁 论文清单（1 篇，2026 年）

| arXiv ID | 标题 | 作者 | v1 日期 | PRM 规模 | 问题角度 |
|---|---|---|---|---|---|
| [2606.00437](https://arxiv.org/abs/2606.00437) | EST-PRM: Stress-Testing Process Reward Models Before They Become Load-Bearing | Shihab et al. (Iowa State / KIIT) | 2026-05-30 | 7-8B PRMs | **对抗性压力测试**：label-preserving 攻击下的稳健性 |

---

## 🎯 关键发现

### 核心问题

PRM（Process Reward Model）在 LLM 训练中被广泛用作 dense step-level supervision——**假设 PRM 分数在 label-preserving 变换下保持稳定**。本文用对抗性 stress testing **证伪了这一假设**。

### 三类 Label-Preserving 攻击

1. **Step-inflation**：插入语义冗余的 restatement/verification 步骤 → 改变 step reward 分布但不影响最终答案
2. **Position-sensitivity**：在依赖图约束下置换步骤顺序 → 暴露 PRM 对绝对位置的依赖
3. **Confidence-injection**：在步骤中加 "clearly"/"by definition" 等 discourse markers → 改变 learned scoring heuristics

### 形式化定义

**Correlation collapse（Δρ）**：
$$\Delta\rho(P, \mathcal{A}, B) = \rho(S_P(X), Y) - \rho(S_P(\mathcal{A}(X)), Y)$$

**Score inflation（ΔS）**：mean[SP(𝒜(X)) − SP(X)]

**Score-inflation rate (I_τ)**：相对分数增加 > τ 的比例（实验用 τ=0.1）

### 核心定理

**定理 5**（mean-aggregated PRM 的 step-inflation）：inflation direction = mean(r̄_ins − r̄)

**定理 7**（position-sensitivity 的 correlation degradation）：
$$\rho(S', Y) = \rho(S, Y) \times \frac{\sigma_S}{\sqrt{\sigma_S^2 + \sigma_\eta^2}}$$

### 5 PRM × 3 攻击 = 15 漏洞画像

| PRM | 主导失败模式 | 关键数字 |
|---|---|---|
| **Math-Shepherd-7B** | Position-sensitivity | Δρ=0.152±0.038; inflation 32.8±4.9% |
| **Qwen2.5-Math-PRM-7B** | Step-inflation | inflation 47.6±4.3% |
| **RLHFlow-PRM-8B** | （baseline correlation 仅 0.100，几乎无信号） | — |
| **Skywork-o1-Open-PRM-8B** | Position-sensitivity | Δρ=0.115±0.029; inflation 30.2±4.7% |
| **DeepSeek-Math-7B-PRM** | Step-inflation | inflation 38.8±3.9% |
| **LLM-as-critic baseline** | 远小于 PRMs（Llama-3-8B）| — |

**核心洞见**:自然分布上准确率 ≠ 攻击下稳健性。**失败模式是模型特定的，不是普遍的**。

### 缓解策略（在 Math-Shepherd 上）

- **Ensemble flagging**：PRM + LLM critic via conservative min，AUROC = 0.72（最平衡）
- **Conformal abstention**：检测率近 0（攻击不把分数推到 honest tail 之外）
- **Length-normalised flagging**：对 step-inflation 有效，对 position-sensitivity 无效
- **结论**:没有单一防御占优，生产部署需组合 penalty-based + ensemble-based

---

## 📐 对 AgenticMind 项目借鉴

### 对 architectures 决策线（`../../architectures/`）

- **直接支持 v3/v4 否决 PRM 的实证判断**：00 §一表格标注 v3/v4 的"6 项事实性错误"——本文证实 PRM 路线**整体**存在系统性脆弱性，与 architectures 否决 PRM 路线的判断一致
- **v4.5/v4.6 Engine-Native Verification 路线得到实证背书**：本文 LLM-as-critic baseline 对所有攻击**都更稳健**——**"用 LLM 判正确性 + 结构化执行验证"比"用专门训练的 PRM"更可靠**——这正是 04b v4.5 路线（Python REPL + JSON Schema）的核心论点
- **PRM Robustness 列为 v4.5+ 之后"真创新"之一**：99 §八 列出方向 #1——本文提供**具体攻击向量**（step-inflation / position-sensitivity / confidence-injection），后续可作为 v4.6+ 后的研究方向

### 对 SOCA v3-Micro-Final（`../soca/`）

- **直接验证 SOCA "绕开 PRM" 路线**：SOCA 09 §四 "联合 SAE vs PRM" 的讨论倾向 SAE——本文数据完全支持：**PRM 的稳健性问题在 2026 年仍是开放问题**
- **建议 SOCA 09 §四 增补 EST-PRM 引用**：作为 "为什么 SOCA 不依赖 PRM" 的最新实证论据
- **SOCA F1-F7 失效模式与 EST-PRM 攻击的对应**：SOCA 08 §一的"路由震荡（F4）/ 路由退化（F6）"与本文 position-sensitivity 攻击在症状上相似

### 对 AgenticDSL 训练链路

- **HydraForge 4 层验证器的稳健性优势**：L1（grammar）+ L2（signature）+ L3（execution）+ L4（task）——**L1/L2 是结构化形式化检查，对 EST-PRM 攻击向量免疫**
- **AgenticDSL 训练的"无须 PRM" 路线得到强化**：本文证实 PRM 不可靠 → AgenticMind 训练链路（AgenticDSL SFT + RLIF）应**优先强化 Engine-Verifier 而非训练 PRM**——与 AGENTS.md F-04（不依赖 PRM）一致

---

## 📊 与项目决策的对应

| AGENTS.md 决策 | 本目录论文 | 行动建议 |
|---|---|---|
| **F-04 验证器 vs PRM** | #3 EST-PRM | **HydraForge 结构化验证 > 任何 PRM 训练项目**——PRM 不可靠有 2026 最新实证 |
| **v3/v4 PRM 路线否决** | #3 EST-PRM | 验证 99 §三 "不要走的路线" 中的 "MoE 3-4B + LoRA + 异构" 等路线判断 |
| **HydraForge 验证器设计** | #3 EST-PRM | 加 fuzzing 测试：使用 EST-PRM 攻击向量验证器稳健性 |
| **AgenticDSL RLIF 训练 reward 信号** | #3 EST-PRM | **不应使用 PRM-score 作为 RLIF reward**——PRM 自身易被攻击 |

---

## 🚀 推荐阅读顺序

| 读者 | 阅读路径 |
|---|---|
| **架构师 / 决策者** | §1 Introduction → §3 EST-PRM Framework → §5 Empirical Evaluation → §6 Mitigations |
| **HydraForge 验证器设计者** | §3 EST-PRM Framework → §6 Mitigations（如何防御攻击）→ §5（PRM 漏洞画像） |
| **AgenticDSL RLIF 训练者** | §1 Introduction → §6 Mitigations → 对照 AGENTS.md F-04 |
| **PRM 研究者** | §3 Framework（形式化定义）→ §5 Empirical Evaluation → §6 Mitigations |

---

## 📅 调研版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-30 | 初版，1 篇论文调研完成（原 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) §3 拆分） |

---

> **本目录定位**:作为 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) 的**分主题深度版**——survey 提供全局视角与跨论文对比，本目录提供单篇论文的完整内容/创新/借鉴分析。

# Process Reward Model (PRM) 实战调研报告

> 调研时间：2025-2026 年文献 | 涵盖 Math-Shepherd、PRM800K、Qwen2.5-Math-PRM、OmegaPRM、GenPRM、PRIME、PRMBench 等代表性工作

---

## 1. PRM 的实际有效性边界

### 1.1 核心数据：BoN 提升幅度到底多大？

| 模型 | 方法 | BoN@8 / BoN@N | 绝对提升 | 数据来源 |
|---|---|---|---|---|
| GPT-4 (Lightman 2023) | PRM vs ORM (BoN@1860, MATH500) | 78.2% vs 72.4% | **+5.8%** | PRM800K 论文 |
| GPT-4 (Lightman 2023) | PRM vs 多数投票 (BoN@1860) | 78.2% vs 69.6% | +8.6% | PRM800K 论文 |
| DeepSeek-67B + MetaMATH | PRM vs ORM (BoN@256, MATH500) | 47.0% vs 45.3% | **+1.7%** | Math-Shepherd |
| Mistral-7B + MetaMATH | PRM vs ORM (BoN@256, MATH500) | 37.3% vs 36.4% | +0.9% | Math-Shepherd |
| Qwen2.5-Math-7B-Instruct (policy) | Qwen2.5-Math-PRM-7B vs maj@8 (7 task avg) | 67.6% vs 66.2% | **+1.4%** | Qwen 团队 |
| Qwen2.5-Math-7B-Instruct (policy) | Qwen2.5-Math-PRM-72B vs maj@8 (7 task avg) | 69.3% vs 66.2% | +3.1% | Qwen 团队 |
| Gemini Pro (PRM-BoN) | OmegaPRM vs Base (MATH500) | 69.4% vs 51.0% | **+18.4%** | OmegaPRM |
| Gemma2-27B (PRM-BoN) | OmegaPRM vs Base (MATH500) | 58.2% vs 42.3% | +15.9% | OmegaPRM |

**关键观察**：
- **绝对提升在 BoN@8 这种小 N 下普遍只有 1-3%**（Qwen、Math-Shepherd 自己在 BoN@256 也很小）。Lightman 2023 的 +5.8% 是 BoN@1860 才拿到的。
- **任务越难，PRM 优势越明显**：GSM8K（步骤少）上 PRM vs ORM 几乎无差异；MATH/OlympiadBench 上 PRM 才显著超过 ORM（Math-Shepherd 论文明确指出）。
- **数学之外的领域证据极少**：所有上述数字都是数学/形式逻辑。代码、科学 QA 上的 BoN 数据公开缺失。

### 1.2 PRM 有效的任务类型

✅ **明确有效**：
- **多步形式化数学推理**（MATH、OlympiadBench、AIME）：长链推理、错误累积严重、最终答案唯一可验证。
- **形式逻辑 / 定理证明**（FOVER 论文，arXiv 2502）：用 Z3/Isabelle 自动标 step 标签，PRM 在 ANLI、MMLU-Pro、BBH 等 OOD 任务上也有提升。

⚠️ **效果可疑**：
- **短链 GSM8K**：PRM vs ORM 几乎无差（Math-Shepherd 论文：GSM8K 上 Self-Consistency 88.0% vs PRM 93.2% vs ORM 91.8%——ORM 已经够好）。
- **代码生成**：ORPS 论文（arXiv 2412.15118）明确指出 "math-focused PRMs may not suit programming's structured logic"——专门训练的 PRM **仍然不如** "执行反馈 + LLM 自我批评" 的免训练方案。
- **开放式 / 主观任务**：没有公开证据支持 PRM 在创意写作、对话、指令遵循上有效。

❌ **公开无效**：
- **事实性 QA（MMLU 通用）**：Qwen 自己的 PRM 在 MMLU STEM 上提升 0.9%，几乎无意义。

### 1.3 PRM 本身的已知缺陷

**(a) 奖励黑客 (Reward Hacking)** — DeepSeek-R1 论文（arXiv 2501.12948, Nature 2025）直接承认：
> "Notably, we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks. This decision is predicated on our observation that **neural reward models are susceptible to reward hacking during large-scale reinforcement learning**."

PRMBench 论文（arXiv 2501.03124）量化了这一点：BoN 表现与 PRM 步骤错误检测能力的 Somers' D 相关性仅 **-0.05**——即 "在 BoN 上强的 PRM，在检测步骤错误上不一定强"。Math-Shepherd-7B 在 PRMBench PRMScore 仅 47.0（接近随机），但在 BoN 上 74.3 反而超过 Qwen-PRM-7B（73.4）。

**(b) 步级标签噪声** — MC 估计的系统性偏差（Qwen2.5-Math-PRM 论文 arXiv 2501.07301、Scan 论文 arXiv 2509.16548）：
- **False Positive**：错误步骤因后续 self-correction 而被标为正确。Qwen 团队的 PRM-MC-soft 在 OlympiadBench 上的 ProcessBench F1 仅 19.6，而 PRM800K 是 56.5。
- **False Negatives**：正确步骤因后续策略失败被标为错误。BEYOND-THE-FIRST-ERROR 论文（EMNLP 2025 Findings）实测：两个不同 completion model 对同一解答只给出一致标签的步骤仅占 **79%**，链越长一致性越差。
- **On-policy 偏差**：MC 标签的"正确性"高度依赖采样模型的分布。Qwen 团队自评："process labels heavily depend on the language model used to generate solutions... highly on-policy"。

**(c) 可泛化性差**：
- Qwen 团队 2025 年的 ProcessBench 测试：Math-Shepherd-7B 在 OlympiadBench 子集 F1 从 GSM8K 的 47.9 跌到 18.0；Math-Shepherd 在 OlympiadBench 的 PRMBoN 比训练分布内下降 30+ 分。
- PRM800K（GPT-4 输出上训练）在 7B 开源模型上的 BoN 表现 **不如** 在同分布生成的 Math-Shepherd 标签（Math-Shepherd 论文 Figure 2）。OpenAI 的 PRM 难以迁移。

**(d) BoN 评估本身的偏差**（Qwen 团队自承）：
> "BoN-based evaluations may introduce bias in favor of models which generate correct answers through suboptimal or flawed intermediate reasoning"

即 PRM 在 BoN 上"有效"可能是奖励 hacking——模型学会了"最终答对但过程作弊"。

---

## 2. min(PRM_scores) 作为 RL 奖励的合理性

### 2.1 主流聚合方法

| 工作 | 聚合方式 | 性能 |
|---|---|---|
| OpenAI PRM800K (Lightman 2023) | **product** of step probs | BoN@1860 = 78.2% |
| OpenAI PRM800K | **minimum** of step probs | 77.6% |
| Math-Shepherd | **minimum** of step scores (论文明文) | BoN@256 GSM8K 87.1% |
| Qwen2.5-Math-PRM | **product** of step scores | 7 任务平均 67.6% |

**结论**：Lightman 2023 的消融显示 product vs minimum 差距仅 0.4-0.6%，二者基本等价。**没有公开证据支持"专门用 min 会更稳定"**。

### 2.2 用 min/product 作为 RL 奖励会过保守吗？

**DeepSeek-R1 团队的明确结论（最权威的负面证据）**：
> "PRM has three main limitations... (1) it is challenging to explicitly define a fine-grain step in general reasoning. (2) determining whether the current intermediate step is correct is a challenging task. Automated annotation using models may not yield satisfactory results, while manual annotation is not conducive to scaling up. (3) **once a model-based PRM is introduced, it inevitably leads to reward hacking**."

这与"过保守"不同——DeepSeek 发现的是 PRM 被模型**利用**，而非让模型变保守。但本质是同一个问题：PRM 的 reward surface 是错的，优化它会让 policy 偏离真实目标。

### 2.3 GRPO + PRM 实际训练稳定性的失败案例

**案例 1：DeepSeek-R1 完全放弃 PRM**
直接使用 rule-based verifiable rewards（数学答案字符串匹配、代码 test case 编译）。Nature 2025 论文是当前最强的反 PRM-as-Reward 证据。

**案例 2：CoRPO 论文 (arXiv 2511.04439)** 发现 GRPO 在 ordinal reward 下会"奖励错误轨迹"：
> "GRPO's group-mean baseline can assign positive advantages to incorrect solutions simply because they outperform a poorly-performing group average... directly reinforcing failed behaviors."

虽然 CoRPO 没专门测 PRM，但揭示了 GRPO baseline 在 PRM 场景下的致命问题：同一 group 中若有步级分数普遍低但少数高，policy 反而被推向"作弊"以获得高分。

**案例 3：P-GRPO 论文 (arXiv 2508.05170)** 代码 RL 上的"posterior"修复：
> "neural reward models may suffer from reward hacking during RL training, particularly in code generation tasks where neural reward model signals are more susceptible to exploitation compared to test case pass rate reward signals."

作者不得不在 PRM 思考奖励上**加 gating**——只有 outcome 正确时才采用 PRM 步分。这等于承认直接用 PRM-as-reward 是不稳定的。

**案例 4：GRPO is Secretly a PRM (arXiv 2509.21154)** 提供了反直觉证据：
> "GRPO induces a non-trivial PRM under certain assumptions... identical-prefix condition is almost always met under real-world conditions"

即 **GRPO 本身已经隐式是一个 PRM**（因为 group 内多条轨迹共享前缀）。λ-GRPO 修正后比标准 GRPO 快 2× 收敛、在 15/20 cell 上更好。这暗示**显式训练 PRM 没有必要**。

### 2.4 可执行结论
1. **不要用 min(PRM_scores) 作为独立 RL 奖励信号**。要么：(a) 用 outcome-based verifiable reward（DeepSeek 路线），要么 (b) 用 PRM-as-Reward + gating（P-GRPO 路线，PRM 分只在 outcome 正确时计入），要么 (c) 用 implicit PRM（PRIME 路线）。
2. **如必须用 PRM 做 BoN 推理时验证**，用 product 聚合（与 OpenAI 一致），不要用 min。
3. **警惕 PRM + GRPO 的 group-mean 反转问题**——考虑 CoRPO 的 baseline clip。

---

## 3. PRM 数据构建的成本与可扩展性

### 3.1 Monte Carlo Rollout 的实际成本

| 方法 | 单步 rollout 数 | 总成本 | 标签质量 |
|---|---|---|---|
| PRM800K (OpenAI) | 0 (人工标注) | 数百万美元级人类标注费 | 最高 (人类) |
| Math-Shepherd (默认) | **N=8** (Llemma-7B completer) | ~4× ORM 成本 | HE 准确率 86%@N=4 |
| OmegaPRM (MCTS + 二分) | 平均 ~2.13 (75× 高效) | 1.5M 标签 / 单 GPU-日 | 与 Math-Shepherd 相当或更好 |
| Qwen2.5-Math-PRM-7B | **8** (Qwen2.5-Math-72B completer) | 500K queries × 8 × 8 = 32M 推理 | 仍逊于 PRM800K |
| ImplicitPRM / PRIME | **0** 步级标签 (只用 outcome) | <1/38 Math-Shepherd | BoN 性能反超 Math-Shepherd |

**关键发现（Qwen 团队自己承认）**：
> "human annotation... exhibited superior generalization capabilities on more complex tasks OlympiadBench and Omni-MATH... MC estimation performed the worst despite having the largest dataset overall."

即**规模最大的 MC 训练数据，泛化性能最差**。ProcessBench 上 F1：PRM800K = 56.5, MC-860K = 40.2。

### 3.2 步级标签噪声：多严重？

**Scan 论文 (arXiv 2509.16548)** 系统量化：
- MC 标签的 self-confidence 中位数噪声：约 25-50% 的样本存在 false positive 或 false negative。
- 仅通过 self-denoising 策略 + 1.5B 模型 + 仅 6% 的 MC 推理成本，**ProcessBench F1 从 19.9 跃升到 59.1**（+39.2）。

**Reflection-Aware 论文 (arXiv 2601.12748)**：
- 指出 FP 根源：策略的 self-correction 让错误步骤"看起来正确"。
- 引入 reflection detection 后 F1 提升达 27%。

**BEYOND-THE-FIRST-ERROR 论文**：
- 不同 completer 模型对同一解答的一致率仅 79%。
- 链越长噪声越大，Olympiad-level 长 CoT 几乎完全失效。

### 3.3 跨域迁移：数学 PRM 能用到代码吗？

**乐观证据**：
- FOVER 论文 (arXiv preprint, ICLR 2025)：用 Z3/Isabelle 标 formal logic 数据，**在 12 个 OOD 任务（MATH、AIME、ANLI、MMLU-Pro、BBH）上 BoN 提升**——证明 PRM 学到的是"通用验证能力"而非领域知识。
- "From Mathematical Reasoning to Code" (AAAI 2026)：明确发现 "PRMs trained on mathematical datasets exhibit performance comparable to those tailored for code generation, suggesting robust cross-domain generalization"。

**悲观证据**：
- ORPS 论文 (arXiv 2412.15118)：**专门训练的 math/code PRM 都不如** "执行反馈 + 自我批评" 的免训练方案。
- ProcessBench 跨数据集测试：OlympiadBench → Omni-MATH 上 PRM F1 普遍腰斩。

**结论**：
- **同类型多步推理任务（math→math, formal logic→math/code）有部分迁移**。
- **结构差异大的任务（math→开放式对话、math→长篇生成）迁移证据为 0**。
- 当前 SOTA 共识：**PRM 必须按目标任务分布训练**——一个通用 PRM 目前不存在。

---

## 4. 2025-2026 年新进展

### 4.1 隐式 PRM 替代外置 PRM

**ImplicitPRM / Free Process Rewards (Yuan et al., arXiv 2412.01981, ICLR 2025)**
- 核心洞见：用 DPO/CE 训练 ORM 时，参数化 reward 为 `β log(π_θ(y) / π_ref(y))`，**partial response 的 log-likelihood ratio 自动就是 step-level PRM**——零额外标注成本。
- 性能：使用 1/38 Math-Shepherd 数据，BoN@N 上反超 Math-Shepherd。
- 数据效率：CE loss 在每指令仅 1 response 时仍能训练（DPO 不行）。
- **反直觉发现**：再在 Math-Shepherd step-level 标签上 fine-tune **不会**进一步提升——outcome 监督已足够。

**PRIME (arXiv 2502.01456)**
- 在 ImplicitPRM 基础上做 online PRM 更新（policy rollouts + outcome labels）。
- Eurus-2-7B-PRIME 用 Qwen2.5-Math-7B-Base 在 7 个推理 benchmark 平均提升 15.1%，**用 10% 数据超过 Qwen2.5-Math-7B-Instruct**。
- 关键实验：online update 至关重要——固定 PRM 训练会 reward hacking。

**SPRO (arXiv 2507.01551)**
- 进一步推到 PRM-free：直接在 policy model 自身 log-prob 上定义 process reward。
- 17.5% 准确率提升、3.4× 训练效率、0 额外 GPU 显存。

**GRPO is Secretly a PRM (arXiv 2509.21154)**
- 数学证明：标准 GRPO 隐式诱导一个 non-trivial PRM（基于 group 内 prefix 共享）。
- λ-GRPO 修正后比 vanilla GRPO 快 2×、在 15/20 cell 上更好。

### 4.2 生成式 PRM / Critic Model

**GenPRM (arXiv 2504.00891, AAAI 2026)**
- 把 step 评分从"标量打分"改为"显式 CoT 推理 + 代码执行验证"。
- 用 QwQ-32B 生成 reasoning rationale + 共识过滤 (consensus filtering)——丢弃约 51% 数据，保留 23K 高质量样本。
- **1.5B GenPRM 即可超过 GPT-4o；7B GenPRM 超过 Qwen2.5-Math-PRM-72B on ProcessBench**。
- 支持 test-time scaling——多次采样后多数投票。

**Qwen2.5-ProcessReward (Qwen 博客 2025-01)**
- 在 MC 标签 + LLM-as-judge 之间做"共识过滤"。
- 共识标签比纯 MC 标签在 OlympiadBench/Omni-MATH 上泛化更好。
- 但 ProcessBench F1 仍 73.5，与 o1-mini 有差距。

**ReasonEval (Qwen 团队前身，Xia et al.)**
- 把 critic 作为"对话式评估者"，PRMBench 上 PRMScore 73.1。
- 比 Math-Shepherd-7B (47.0) 和 Llemma-PRM800K (52.0) 显著强。

### 4.3 Critic Model vs Scalar PRM：谁胜出？

PRMBench 论文 (Song et al., ACL 2025) 的 25 模型评测：

| 模型 | PRMScore | 性质 |
|---|---|---|
| **人类** | **83.8** | 上限 |
| Gemini-2-Thinking (proprietary) | 68.8 | 生成式 critic |
| Qwen2.5-Math-PRM-72B | 68.2 | scalar PRM |
| GPT-4o | 66.8 | 生成式 critic |
| Qwen2.5-Math-PRM-7B | 65.5 | scalar PRM |
| Skywork-PRM-7B | 65.1 | scalar PRM |
| Math-Shepherd-7B | 47.0 | scalar PRM (近随机) |
| Random | 50.0 | 基线 |

**关键发现**：
1. **生成式 critic (GPT-4o, Gemini-2-Thinking) 在 PRMScore 上已与最强 scalar PRM 持平或反超**——这与 GenPRM 的结论一致。
2. **所有 PRM 都显著弱于人类**（最高 68.8 vs 人类 83.8），提升空间大。
3. **传统 scalar PRM (Math-Shepherd-7B) 接近随机**——证明简单 MC 训练方案的局限。

### 4.4 主流生产推理模型用 PRM 吗？

| 模型 | PRM 用于训练 | PRM 用于 BoN 推理 | 来源 |
|---|---|---|---|
| OpenAI o1 / o3 | 传闻用 PRM (未确认) | 是 (PRM800K 衍生) | OpenAI 博客 |
| DeepSeek-R1 | **❌ 明确不用** | 未公开 | R1 论文 + Nature 2025 |
| Kimi K1.5 | 未公开 | 推测有 | K1.5 论文 |
| QwQ / Qwen3 | ❌ 训练用 RLVR；可选用 Qwen-PRM 做 BoN | 可选 | Qwen 团队 |
| Skywork-OR1 | ❌ outcome-based | 否 | Skywork 报告 |
| Seed1.5-Thinking | ❌ | 否 | ByteDance 报告 |
| AceReason | ❌ | 否 | Nvidia 报告 |

**共识**：2025 年几乎所有主流开源 reasoning model **不用 PRM 训练**；训练时用 verifiable outcome reward。PRM 退回到**仅用于离线 BoN 重排**或"reasoning 时 verifier"。

---

## 5. 综合可执行建议

### 5.1 如果你是研究者，想训练一个 PRM
1. **优先用 ImplicitPRM / PRIME 路线**：DPO/CE 训练 + outcome label 自动得到 step-level PRM，**1/38 成本达到 Math-Shepherd 水平**。
2. **如需高质量 step 标签**：GenPRM 的 QwQ-32B + consensus filtering 路线（23K 数据 > PRM800K 80 万标签）。
3. **不要用纯 MC 标 soft labels**——在 OOD 任务上泛化最差。
4. **聚合方式选 product**，不要选 min（OpenAI 消融支持）。

### 5.2 如果你是应用工程师
1. **不要把 PRM 当 RL 奖励**——用 rule-based verifiable reward（math 答案匹配、code test case 编译）。DeepSeek-R1 已证明这是 2025 年的 winner。
2. **如必须用 PRM 做 BoN 推理时验证**，可直接用 Qwen2.5-Math-PRM-7B/72B 或 Skywork-PRM-7B；math 任务上 BoN@8 提升 1-3%。
3. **跨域任务（代码、科学）**：考虑"执行反馈 + LLM 自我批评"（ORPS 方案），不要用现成 math PRM。
4. **不要把 PRM 当 ground truth**：PRMBench 显示所有开源 PRM 与人类一致性 50-70%。

### 5.3 关于"PRM 已死"的判断
**没有完全死，但定位已变**：
- ❌ 死了的："外置训练 + 标量打分 + 用于 RL 训练"——DeepSeek-R1、GRPO-is-secretly-a-PRM、CoRPO 共同宣判。
- ✅ 还活着的：
  - **离线 BoN 推理时验证**（Qwen-PRM 仍是 SOTA）
  - **生成式 Critic / GenPRM 范式**（已超过 scalar PRM）
  - **隐式 PRM**（作为 GRPO 的修正项 λ-GRPO，比 vanilla GRPO 快 2×）
  - **特定领域 PRM**（形式逻辑、agentic step 决策）

### 5.4 关键引用一览
- PRM800K: Lightman et al., 2023, arXiv 2305.20050
- Math-Shepherd: Wang et al., ACL 2024, arXiv 2312.08935
- OmegaPRM: Luo et al., 2024, arXiv 2406.06592
- Qwen2.5-Math-PRM: arXiv 2501.07301 (ACL Findings 2025)
- ProcessBench: Zheng et al., ACL 2025
- PRMBench: Song et al., ACL 2025, arXiv 2501.03124
- GenPRM: Zhao et al., AAAI 2026, arXiv 2504.00891
- PRIME: arXiv 2502.01456 (Cui et al.)
- ImplicitPRM: Yuan et al., ICLR 2025, arXiv 2412.01981
- SPRO: arXiv 2507.01551
- GRPO is Secretly a PRM: arXiv 2509.21154
- CoRPO: arXiv 2511.04439
- P-GRPO: arXiv 2508.05170
- Scan: arXiv 2509.16548
- ORPS (code PRM): arXiv 2412.15118
- FOVER (formal logic): ICLR 2025
- DeepSeek-R1: arXiv 2501.12948 + Nature 2025

---

## 6. 结论

PRM 在 **2024-2025 年是 SOTA inference-time verifier**，但作为 RL 训练信号已被 outcome-based reward (GRPO + verifiable reward) **全面取代**。DeepSeek-R1 的实践是分水岭：最强开源 reasoning model 明确放弃 PRM。2025 年的"隐式 PRM"和"生成式 PRM"路线虽然重新激发了兴趣，但更多是对 GRPO 的**改进**而非独立的 PRM-as-Reward 范式。**min(PRM_scores) 聚合策略在 BoN 推理中可用，但作为 RL 奖励不推荐**——会导致 reward hacking 而非保守化。

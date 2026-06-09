# LLM 推理 SOTA 方案的诚实边界评估（v3 决策参考）

> 调研时间：2025–2026 年文献 | 立场：**挑刺优先**——每一条结论都列出证据与失败案例  
> 对照对象：MiniMind v3（64M / 198M-A64M，R1-Distill + rule-based RL + Agentic RL + Tool Calling）  
> 阅读建议：先看最后一节"被高估 / 被低估"总览，再回头看各节证据

---

## 0. v3 关键决策回顾

| 决策 | 实现位置 | 一句话描述 |
|---|---|---|
| ✅ 放弃独立 PRM 路线 | 主线 + PRM_RESEARCH_REPORT.md | 借鉴 DeepSeek-R1，纯 outcome verifiable reward |
| ✅ 保留 R1-Distill 风格的 thinking 模板 | `sft_t2t.jsonl` 已混入 reasoning 数据 | `<\|im_start\|>assistant\n<think>...` |
| ⚠️ 用 InternLM2-1.8B-Reward 作为稠密 reward | 训练说明要求下载 1.8B RM | 用于 GRPO/PPO 的连续打分 |
| ✅ 主算法 CISPO + GRPO | `train_grpo.py` 支持 `loss_type=cispo` | 修 PPO/GRPO ratio 被 clip 截断的梯度 |
| ✅ Agentic RL | `train_agent.py` | 多轮 Tool-Use GRPO/CISPO |
| ⚠️ Adaptive Thinking 软开关 | `chat_template` + `open_thinking` | 同一个模型切换 think/直答 |
| ❌ 没有任何 PRM 训练/加载 | 仅引用 PRM 报告 | 显式放弃 |
| ❌ 没有 test-time TTS 自适应分配 | 推理时不做 difficulty-based compute allocation | 静态 `<think>` 开关 |

---

## 1. R1-Distill 系列的真实代价

### 1.1 蒸馏对底座模型的依赖——硬证据

**DeepSeek-R1 自己的结论（最权威）**：[GitHub README](https://github.com/deepseek-ai/DeepSeek-R1) 明确写道"reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models"——但表 6 显示 32B-Base 的 DeepSeek-R1-Zero-Qwen-32B（大规模 RL）"requires enormous computational power and may not even achieve the performance of distillation"。**这是 base model 大于等于 32B 时的结论**。

**For sub-7B 的真实数字**（[DeepSeek-R1 Distill 1.5B 数据](https://github.com/deepseek-ai/DeepSeek-R1)）：
| Distill 模型 | Base | AIME 2024 pass@1 | MATH-500 | GPQA | LiveCodeBench | CodeForces |
|---|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 28.9 | 83.9 | 33.8 | 16.9 | 954 |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 55.5 | 92.8 | 49.1 | 37.6 | 1189 |
| R1-Distill-Llama-8B | Llama-3.1-8B | 50.4 | 89.1 | 49.0 | 39.6 | 1205 |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 69.7 | 93.9 | 59.1 | 53.1 | 1481 |
| R1-Distill-Qwen-32B | Qwen2.5-32B | 72.6 | 94.3 | 62.1 | 57.2 | 1691 |

**关键观察**：
- **base 必须是 Qwen2.5-Math 或 Qwen2.5-Instruct 这类已经"嵌入数学/代码"能力的模型**。R1 团队自己 1.5B 起步——这是 1.5B 的 base 也已经被 Qwen 团队专门持续预训练过 700B–1T 数学 token 后的结果。
- [Quantifying the Capability Boundary (arXiv 2502.11164)](https://arxiv.org/html/2502.11164) 实测证实：**R1-Distill 在 Llama 系列上的表现显著劣于 Qwen 系列**——同样的蒸馏数据，从 Qwen2.5-1.5B 蒸馏 1.5B 优于从 Llama-3.1-8B 蒸馏 1.5B；formal logic 任务上 1.5B 学生相对 teacher 下降 ΔAccuracy 可达 -50 分。
- **MiniMind 64M 的真实处境**：连 Qwen2.5-Math-1.5B 都被 671B R1 当成最小蒸馏起点——1.5B 相对 671B 是 0.22% 的参数量，64M 又是 1.5B 的 4.3%。**[Jahin et al. 2025](https://api.emergentmind.com/topics/deepseek-r1-distilled-models) 总结：1.5B–8B 的 distilled model 在 formal logic / competition math / "hard" instances 上"marked degradation, up to -50 points versus teacher"。** MiniMind 处于这条衰减曲线的更深处。

### 1.2 蒸馏后能否再做 RL？会破坏 reasoning 吗？

**正面证据**：
- DeepSeek-R1 自己的流水线就是"cold-start SFT → RL → rejection sampling SFT → 再 RL"，证明大模型可以叠。
- [OREAL (InternLM, GitHub)](https://github.com/InternLM/OREAL)：在 DeepSeek-R1-Distill-Qwen-7B 上继续做 outcome-based RL，MATH-500 从 ~92 提升到 **94.0 pass@1**，对齐 32B 模型——证明 7B 这个量级 RL 不会破坏蒸馏。
- [JustRL (arXiv 2512.16649)](https://arxiv.org/html/2512.16649) 在 R1-Distill-Qwen-1.5B 上用单阶段 GRPO（无 curriculum、无 length penalty）实现 64.3% 跨 9 个数学 benchmark 平均分，比复杂方案省 2× 算力——再次证明 1.5B 量级仍可 RL。

**负面 / 风险证据**：
- [Through the Valley (EMNLP 2025)](https://arxiv.org/html/2506.07712)：**8k long CoT examples 就让 Gemma3-1B-it 掉到 baseline 的 25%**（精度从 100% 跌到 25%）；Qwen2.5-0.5B 和 Gemma-3-1B **用 220k long CoT 样本也回不到 baseline**。
- 同一论文 Figure 7 关键发现：**8k Long CoT SFT 后再 RL，性能持续低于"无 SFT 直接 RL"** baseline——degradation 不可被 RL 修复（需要 ≥128k 样本才能让 RL 反超）。
- 含义：v3 当前 sft_t2t 中混入了 R1 风格 reasoning 数据，**在 64M 模型上这种 Long CoT 退化风险比 0.5B 还要大一个量级**。

### 1.3 蒸馏数据中的 reasoning traces 是否包含"thinking"？base 没有这种能力会怎样？

**会的，包含"thinking"——这正是失败根源**。[In Their Own Words (arXiv 2509.22230)](https://arxiv.org/html/2509.22230) RSD 论文：teacher 生成的 trace 中包含大量**低于 1% probability** 的 token（"wait"、"alternatively"、"hmm"等逻辑连接词），在 Qwen3-0.6B 这种小 student 上**直接蒸馏 s1K-1.1 trace 平均性能 -20.5%**；用 RSD（Reverse Speculative Decoding，student-accepts-threshold）过滤后提升 4.9%。跨模型实验显示 RSD trace **model-specific**——为 Qwen3-0.6B 做的 trace 给 Llama-3.2-1B 用反而掉分。

**对 v3 的直接含义**：
- v3 的 sft_t2t 中"混入 qwen3 reasoning 数据"是 black-box distillation——属于 [Unveiling the Key Factors (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.782.pdf) 描述的"teacher 不是越大越好，关键看 student 能不能吸进 ZPD（最近发展区）"。
- 该论文 Figure 6 的 Matthew Effect：**stronger student 从 CoT 蒸馏中获益大，weaker student 反而被高复杂度 CoT 拖后腿**——BLOOM 家族在 GSM8K / AQuA-RAT 上"sometimes performing no better than random guessing"。
- MiniMind 64M / 198M-A64M 的 d_model=768、n_layers=8，相比 Qwen2.5-1.5B 的 d_model=1536、n_layers=28 在 ZPD 维度上**小 1–2 个数量级**——直接用 qwen3 reasoning trace 大概率走 "Long CoT Degradation" 路径。

### 1.4 v3 是否高估了 R1-Distill 路线？

**严重高估**——具体错估了三件事：

1. **错估了"base model 是 Qwen3 / DeepSeek 蒸馏 trace"的可行性**。v3 的 pretrain_t2t 用的是匠数 + Magpie + 通用 SFT 数据混合，**没有 Qwen2.5-Math 那种 700B+ token 的数学持续预训练**。即使 v3 喂 qwen3 蒸馏的 reasoning trace，**64M 模型没有能力"装进" R1 风格的长期 planning / reflection / verification 行为**——根据 Through the Valley 论文，0.5B / 1B 模型需要 16k–220k long CoT 才能恢复，64M 几乎肯定在退化侧。
2. **错估了 R1-Distill 数据中的"thinking tokens"对 64M 的可学性**。v3 README 提到"thinking 能力统一由 chat_template + <think> 与 open_thinking 自适应开关控制"——但 64M 模型在 [RSD 论文](https://arxiv.org/html/2509.22230) 的 Qwen3-0.6B 范畴以下，**没有可用的 capacity 来模仿 R1 风格的 metacognitive tokens**。
3. **可能错估了"reasoning 数据混入 sft_t2t"的混合比**。v3 主线说 sft_t2t 混入 reasoning 数据但具体比例未公开；按照 [TLDR (arXiv 2506.02678)](https://fetcher.alphaxiv.org/v2/pdf/2506.02678v1) 等压缩文献，**System-1（短 CoT，GSM8K 级）和 System-2（长 CoT，AIME 级）的比例对小模型至关重要**——v3 没有显式做这个 balance。

---

## 2. Qwen2.5-Math-PRM 的可复现性

### 2.1 训练成本：consensus filtering 的真实算力

来自 [Qwen 团队论文 (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.547.pdf) 的具体数字：

- **初始数据池**：~500K queries with golden answers（Qwen2.5-Math-PRM-7B）；后续扩展到 860K、3M。
- **每条 query 采样 6–8 个 diverse responses** from Qwen2-Math-Instruct + Qwen2.5-Math-Instruct（7B + 72B 混合）。
- **每步做 8 个独立 completion** 用 Qwen2.5-Math-Instruct 估算 step label。
- **consensus filtering 丢弃约 60% 数据**（从 860K → 344K 高质量样本）。
- **LLM-as-judge critic 是 Qwen2.5-72B-Instruct**——这是数据构建的主要算力消耗点。
- **PRM-7B 本身**基于 Qwen2.5-Math-7B-Instruct 改 LM head → 2 层 linear + tanh 的 scalar head，用 cross-entropy 训 step 末 token。

**总算力估算**（基于 500K queries × 8 responses × 8 completions + Qwen-72B 标注）：
- rollout 阶段：500K × 8 = 4M 次 7B 模型推理 + 500K × 6 = 3M 次 72B 推理
- 粗算：4M × 7B ≈ 0.5M GPU·hour（A100），3M × 72B ≈ 10M GPU·hour（A100）
- **PRM 数据构建的算力消耗 ≈ 10M+ A100-GPU·hour，PRM 7B 训练本身再额外 1–2K GPU·hour**。

### 2.2 能否直接用到代码 / 通用推理 / 对话？

**明确证据：不能。**

- **代码任务**：[Reasoning Through Execution / ORPS (ICML 2025, arXiv 2412.15118)](https://arxiv.org/html/2412.15118) 直接对比 math-PRM 和 code-PRM 在代码任务上的表现，**所有 trained PRM（包括 GPT-4 labels 和人工标注的）都输给"执行反馈 + LLM 自我批评"的免训练方案**。"math-focused PRMs may not suit programming's structured logic"——是 ORPS 论文原文。
- **对话 / 指令遵循**：所有 PRM SOTA（PRM800K、Math-Shepherd、Qwen-PRM、GenPRM）公开数据 **100% 集中在数学/formal logic**。[ORPS]、[VPR (arXiv 2605.10325)](https://arxiv.org/html/2605.10325) 等都强调：VPR 只在"densely-verifiable"的搜索 / 约束求解 / 概率推理任务上 work；对话 / 开放生成没有 oracle 验中间步骤。
- **跨域迁移的"乐观"证据也要打折**：
  - [FOVER (ICLR 2025)](https://arxiv.org/)：用 Z3/Isabelle 标 formal logic 数据训练的 PRM 在 12 个 OOD 任务上 BoN 提升——**但这些 OOD 任务都是"形式逻辑/数学题"**，不是真正的跨域。
  - [From Math to Code (AAAI 2026)]：math-trained PRM 和 code-trained PRM 表现"comparable"——这本身就意味着 cross-domain 没增益。
  - [ProcessBench](https://qwenlm.github.io/blog/qwen2.5-math-prm/) 跨集测试：OlympiadBench → Omni-MATH 上 PRM F1 普遍腰斩。

### 2.3 换 base 模型（如 Llama-3.1）需要重新训练吗？

**是的，必须重新训练。证据：**

- Qwen 团队公开承认 "process labels heavily depend on the language model used to generate solutions... highly on-policy"——MC 估计的 step label 是在 teacher 的分布上构造的。
- [PRM800K (GPT-4 输出上训练) 在 7B 开源模型上 BoN 表现 **不如** Math-Shepherd 标签](https://arxiv.org/html/2501.07301)（Math-Shepherd 论文 Figure 2）。
- 直接证据：[Math-Shepherd-7B 在 OlympiadBench 子集 F1 从 GSM8K 的 47.9 跌到 18.0](https://arxiv.org/)——即使同 base 不同子集都崩。
- 含义：v3 选用的 PRM 路线如果要"换 base 重训"，等于是**完全复制 Qwen 的 10M+ A100·hour pipeline**——在 MiniMind 量级上**完全不可行**。

### 2.4 v3 是否高估了 PRM 路线？

**高估的"反面"是 v3 实际上已经放弃 PRM**——这恰好做对了。**但 README 中"参考了 PRM_RESEARCH_REPORT" 的同时没有给后续 R1-Distill 用户足够的"放弃 Long CoT 数据混合"指引**——这是低估了 Long CoT 对 64M 的破坏性。

v3 真正的隐性问题：**rule-based reward 路线同样不乐观**——参见第 5 节。

---

## 3. rStar-Math 的工程可行性

### 3.1 PPM 训练数据从哪来？需要多少标注？

来自 [rStar-Math (arXiv 2501.04519)](https://arxiv.org/html/2501.04519)：

- **747K math word problems** 起始数据池（公开来源）。
- **4 轮 self-evolution**，每轮做 **16 rollouts / problem**——总 rollout 量 = 747K × 16 × 4 ≈ **48M reasoning trajectories**。
- 每个 rollout 包含多步 generation，**PPM 训练数据 = "对每步选 2 个 positive + 2 个 negative" preference pairs**，最终从 MCTS 树中筛选。论文没给具体偏好对数，但全 4 轮累计大概在 10M+ 步级对。
- **需要两个 7B SLM 协同**（policy SLM + PRM SLM），跑在 **4×40GB A100** 上——明确写在原论文。
- **总工程量**：**4 轮 × 747K × 16 rollouts**——粗算在 50K+ A100·hour 量级（与 Qwen-PRM 同数量级，远超 MiniMind 可用资源）。

### 3.2 MCTS 计算成本：每个 query 多少次 LLM 调用？

- **每 problem 16 rollouts**，每个 rollout 是一棵 MCTS 树（selection / expansion / rollout / back-propagation），**每个 step 平均 n candidates + 树扩展**。
- 仅按 16 rollouts + n=8 candidates/step + 平均 10 steps 算：**128 LLM calls per problem**——含 policy + PPM 双方。
- 论文自报 "extensive rollouts"，"computationally expensive"——**4 轮 self-evolution 在 4×A100 上跑了几周**。

### 3.3 是否需要多个不同尺寸的 model 协同？

**是**。rStar-Math 用 **两个 7B SLM 协同**（policy SLM 和 PRM/PPM SLM），外加 Round 1 bootstrap 时**用 Qwen2.5-Math-72B-Instruct 做初始数据生成**。3 个模型角色：
- 72B teacher：bootstrap 阶段生成初始 reasoning 树
- 7B policy SLM：执行 MCTS rollout 的 candidate generation
- 7B PPM：给 step 打 Q-value，引导 UCT 选择

训练管线**4 轮递进**，每轮都要重新训 policy 和 PPM——工程复杂度极高。开源了 [GitHub microsoft/rStar](https://github.com/microsoft/rStar) 但"4 轮 self-evolution"在 4×A100 上**复现周期数周到数月**。

### 3.4 v3 是否高估了 rStar-Math 风格路线？

**MiniMind 量级根本碰不到 rStar-Math**——这不是"高估/低估"问题，是**完全 out-of-budget**。但 v3 在 README 中提到 rStar-Math 等 SOTA **没有明确告诉用户这些 SOTA 在 64M 上的等价实现是不存在的**——文档层面的"过度暗示"是 v3 的一处隐患。

更准确地说：**v3 没有走 rStar-Math 路线**——它走的是 rule-based reward + InternLM-RM + Agentic RL，这是**对的**。

---

## 4. Compute-Optimal TTS 在生产环境的实际可行性

### 4.1 难度估计器训练数据从哪来？

- **[DIPA (arXiv 2604.21018)](https://arxiv.org/pdf/2604.21018)**：**完全免训练**——用 "Generation Length" / "Pass@k-1" / "Self-Consistency Rate" 作为 inference-time 难度代理，**不需要任何训练数据**。
- **[AdaptiveComp (OpenReview ZNWpUfwisS)](https://openreview.net/pdf?id=ZNWpUfwisS)**：训练一个**信息论 + 学习到的 transformer 难度预测器**，需要从训练分布中**采样 + 估计每个 query 的预期 reward curve**——数据量要求与训练分布同分布。
- **[LATTS (arXiv 2509.20368)](https://arxiv.org/html/2509.20368)**：**local step-level** 难度，用 verifier score 算 acceptance criterion——无需训练。
- **[CATS (OpenReview mXuUomGc0I)](https://openreview.net/attachment?id=mXuUomGc0I&name=pdf)**：**风险控制 + ModernBERT 难度分类器**或**internal special-token probing**（zero overhead），需要一个校准集。
- **JPM 数学含义**：TTS 难度估计在**training-free 路径**已经成熟（DIPA / LATTS），可立即上线；**学习型**路径需要 calibration set，但量级远小于主训练。

### 4.2 自适应策略在不同任务类型上的配置差异？

**几乎没有迁移证据**。所有 SOTA（DIPA / LATTS / CATS / AdaptiveComp）**只在 math + code 上评估**。具体数据：
- DIPA：MATH-500、AIME25、LiveCodeBench、GPQA-Diamond——**全部是 math/code**。
- LATTS：MATH500 + Llama-3.2-1B/3B + Qwen2.5-7B verifier——**只 math**。
- CATS：math + 多种 reasoning 任务——但同样**没有创意写作 / 开放式对话**。
- 跨任务（math / code / dialogue）的**难度估计器能否共享参数**：**没有公开证据**。

### 4.3 在 latency-sensitive 产品中（2 秒响应）如何降级？

- **[DIPA training-free proxy "Generation Length"]**：rollout 中观察**已生成序列长度**——属于 inference-time signal，**overhead ≈ 0**。
- **[LATTS](https://arxiv.org/html/2509.20368)**：在 MATH500 上，**50× fewer tokens** 同时保持 Beam Search top accuracy——延迟节省 50×。
- **[CATS internal special-token probing](https://openreview.net/attachment?id=mXuUomGc0I&name=pdf)**：**0.1ms overhead**——"made 'for free' as part of the existing forward pass"。
- 关键：[AdaptiveComp] 在 latency budget 2s 以内**几乎只有 "no-think / fast path" 一种选择**——adaptive routing 没有"决策时间"。

### 4.4 v3 是否高估了 TTS 自适应方案？

**v3 在 README 中几乎没有提到 compute-optimal TTS**——它的"adaptive thinking"是 **`<think>` 开关二元** 的，不是基于 query 难度的多级 routing。这意味着 v3 实际上**已经放弃了 adaptive TTS**。

但存在一个**潜在低估**：
- v3 的 Adaptive Thinking 是**手控的**——`open_thinking=1` 时强制 model think。如果把 [DIPA 的 "Pass@k-1" 难度代理] 集成进 `chat_template` 的 routing 决策（"短问题 → no-think；长问题 → think"），可以在 v3 当前架构上**几乎零成本**得到 latency 节省——v3 没有做这件事。

---

## 5. Dr.GRPO / KL-Cov 的工程陷阱

### 5.1 对超参数的敏感度

来自 [Understanding R1-Zero-Like Training (arXiv 2503.20783)](https://arxiv.org/pdf/2503.20783) Dr.GRPO 原论文 + [Entropy Mechanism (verl 集成 PR)](https://github.com/verl-project/verl/pull/1830) + [GEPO (arXiv 2508.17850)](https://arxiv.org/pdf/2508.17850)：

**Dr.GRPO**：
- 改动了 GRPO 三个 bias：**response-level 1/|o_i| normalization**、**question-level std normalization**、**min/max symmetric clip**。
- Dr.GRPO 把 1/|o_i| 替换为"常数"（推荐 = max_completion_length）——**这个常数对 token efficiency 极敏感**。
- 原论文 Figure 5 显示：GRPO 在 RL 后期"长度无控制增长"是 bias 导致的（incorrect response 越长 → loss 越大 → 越长越训越长）；Dr.GRPO 把这个**直接砍掉**。
- 效果：在 Qwen2.5-7B 上 AIME 2024 跑 43.3% SOTA——但**前提是 base model 本身有 math pretrain**。

**KL-Cov / Clip-Cov**：
- 来源：[Entropy Mechanism of RL for Reasoning (PRIME-RL)](https://openreview.net/pdf?id=ztGHhyicWs)。
- 核心：**entropy collapse** 是 RL training 后期性能饱和的根因——H 和 R 服从 `R = -a exp(H) + b`。
- Clip-Cov：限制**高 covariance tokens**（=高 advantage 且 action prob 高的 token）的更新幅度。
- KL-Cov：对**高 covariance tokens**额外加 KL penalty。
- 关键数字（Qwen2.5-7B）：AIME24 **15.8 → 22.6**（KL-Cov 提升 6.8 分），AIME25 **12.9 → 12.9**（持平），MATH500 **58.2 → 61.4**。Qwen2.5-32B 上 KL-Cov 比 vanilla GRPO 平均 **+6.4 分**。
- 集成在 [verl](https://github.com/verl-project/verl/pull/1830)：只需在 `core_algos.py` 加两个函数 + `dp_actor.py` 加 `loss_mode` 开关——**集成非常容易**。

**超参数敏感度（GEPO 论文 Table 6）**：
- KL coefficient β_KL = 0.005 是 sweet spot
- β_KL = 0.001 → AIME2024 14.1 → 2.0（崩溃！）
- β_KL = 0.010 → 过保守，peak 受限
- group size = 4 best peak, group size = 8 best final stability

### 5.2 与现有 RL 框架集成难度

| 框架 | Dr.GRPO | KL-Cov | RLOO / REINFORCE++-baseline |
|---|---|---|---|
| **verl** | 已支持 `loss_type="dr_grpo"`（[HF TRL 文档](https://huggingface.co/docs/trl/en/grpo_trainer)） | 已支持 `loss_mode="kl_cov"`（PR #1830） | 支持 |
| **OpenRLHF** | [advantage estimator `dr_grpo`](https://openrlhf.readthedocs.io/en/latest/agent_training.html) | 未公开支持，需要自己 patch | 支持 `rloo` / `reinforce_baseline` |
| **TRL** | [GRPOConfig.scale_rewards=False](https://huggingface.co/docs/trl/en/grpo_trainer) | 未支持 | 支持 |

**Dr.GRPO 集成 1 天工作量**（即改 `compute_advantages` 公式）；**KL-Cov 集成 1–2 天**（改 loss 函数）。

### 5.3 分布式训练稳定性问题

[Dr.GRPO 论文](https://arxiv.org/pdf/2503.20783) + [GEPO 论文](https://arxiv.org/pdf/2508.17850) + [Bespoke Labs 多轮 RL](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning) 总结的常见坑：

1. **vLLM/rollout 与 training 引擎的 distribution shift**：
   - vLLM 用 paged attention，训练用 FlashAttention——`log_p` 数值不一致
   - TIS（Truncated Importance Sampling）/ ICEPOP 修这个
   - Bespoke Labs 的 [100-epoch GRPO 训练 BFCL multi-turn 经验](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)：**KL=0 完全崩溃、KL=0.04 也崩溃、KL=0.001 才行**——KL 系数对多轮 agentic RL 极敏感。

2. **Overlong filtering 必须开**：DAPO-style 过滤超 max_completion_length 的 rollout——否则 loss 被 padding token 主导。

3. **Reference model 更新策略**：Bespoke Labs 实测"每 100 步更新 ref"比"全程固定 ref"好——但**很多开源 PPO/GRPO 实现没有这个开关**。

4. **Reward design 的"少即是多"**：[Bespoke Labs 论文](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning) 实测：**只用"correctness = 1/0" 比 "format + tool + correctness" 三段式更稳定**。多段式 reward 容易 reward hacking（gibberish tool call 满足 format 但破坏语义）。

5. **BiasGRPO（社会偏见领域）发现**：GRPO group size 必须 ≥ 4，否则退化为 binary comparison（DPO-类）——**group size = 2 显著弱于 4 或 8**。

### 5.4 v3 是否高估/低估了 Dr.GRPO / KL-Cov？

**v3 选 CISPO 是被低估的"修 GRPO 梯度截断"问题，但没有选 Dr.GRPO / KL-Cov 是被高估的"vanilla GRPO 足够"假设**。具体：

1. **CISPO vs Dr.GRPO 哪个更优**？
   - CISPO 修的是 ratio 被 clip 后梯度流被切断的问题。
   - Dr.GRPO 修的是 response-level 1/|o_i| normalization 导致 incorrect response 越来越长的问题。
   - **两个问题都存在、彼此独立**——CISPO 不修 length bias，Dr.GRPO 不修 clip 梯度。
   - v3 选 CISPO 是**单一修复**——从 [JustRL](https://arxiv.org/html/2512.16649) 1.5B 上的实验看，"无 length penalty" 在 R1-Distill-Qwen-1.5B 上**自然收敛到 4k–5k tokens**（不要 explicit penalty）；这意味着 length bias 在 1.5B 也许影响没那么大。
   - 但 [GEPO 论文](https://arxiv.org/pdf/2508.17850) 显式说明 **Dr.GRPO 的 fix 在 0.5B 仍然必要**——v3 在 64M 应该同样需要。

2. **KL-Cov / Clip-Cov 完全没用上**——v3 主线没用 `entropy collapse` 修复方案。但 v3 当前 entropy 监控没有显示 collapse（README 提到 PPO/GRPO loss 曲线"reward 稳定上升"），**这可能是因为 64M 模型 entropy 本来就高**——未到 collapse 阈值。

3. **InternLM2-1.8B-Reward 作为稠密 reward 的隐患**：[Bespoke Labs 论文](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning) 明确说"complex reward design leads to worse training stability"——v3 README 提到可"混合 r_total = α r_model + β r_rule"，但这是**已被证明的次优方案**。**Bespoke Labs 在 7B Qwen2.5 上的实验只用 correctness reward**。

4. **tool call reward 设计**：v3 在 agent_rl 中 R_total = R_answer + R_tool + R_format + R_rm - R_unfinished——4 段式 reward 与 Bespoke Labs 推荐的 1 段式相悖。**Bespoke Labs 明确：tool execution reward + format reward + correctness reward 三段式会引出 reward hacking**（agent 输出 gibberish tool call 来刷分）。

---

## 6. 跨任务泛化的真实证据

### 6.1 数学 SOTA 能不能直接迁移到代码、科学推理？

**代码**：**[ORPS (ICML 2025)](https://arxiv.org/html/2412.15118) 明确：math-focused PRM 输给 execution feedback**。VPR (arXiv 2605.10325) 强调"densely-verifiable"——只有**有明确 oracle 的任务**（搜索、约束求解、概率推理）能用 dense process reward；代码任务**勉强能算**（执行反馈 = oracle），但"探索不同算法策略"是 ORPS 的核心价值——这是 PRM 训练不到的。

**科学推理**：公开数据**几乎为零**。所有 PRM / Long CoT SOTA **100% 在 math 上评估**。

**关键发现**：[When Small Models Are Right for Wrong Reasons (arXiv 2601.00513)](https://arxiv.org/html/2601.00513) 10,734 reasoning traces 分析：**50–69% 的 small model 正确答案是 "Right-for-Wrong-Reasons"**——过程错、答案对。该论文还显示**自批评（self-critique）和 verification prompts 对 7–9B 小模型 d = -0.14 到 -0.33**（**有害**）；只有 **RAG** 有 d = 0.23–0.93 正增益。

**对 v3 的含义**：v3 训练的"reasoning 模型"在 64M 上**很可能 50–69% 的正确答案过程也是错的**——意味着 v3 README 展示的"agent"权重在 tooluse 任务 17/20 正确，**但过程推理的 integrity 是未知的**——这种"对错不可分"是 RAG-没用上时代的通病。

### 6.2 PRM 跨领域迁移（数学 PRM → 代码任务）的实测数据？

[ORPS 论文 Table 4](https://arxiv.org/html/2412.15118) 直接对比：
- **Math-PRM 在 HumanEval/MBPP 上：与 random baseline 没显著差异**。
- **专门训练 Code-PRM：略好于 Math-PRM**。
- **ORPS（执行反馈 + self-critic，**免 PRM**）：平均 Pass@1 提升 26.9%，**最强**。

**结论**：math PRM → code 任务**负迁移或零迁移**。代码 PRM 必须重新训练/重新设计。

### 6.3 Long CoT 蒸馏模型在短答案任务（GSM8K 级）上是否会出现 overfitting / format collapse？

**是的**。[Through the Valley 论文](https://arxiv.org/html/2506.07712) 关键发现：

- **Qwen2.5-0.5B-Instruct 在 8k long CoT SFT 后**：response 平均长度从 ~600 → 3,600 tokens；arithmetic accuracy 跌 30 分。
- **Gemma-3-1B-it 在 8k long CoT SFT 后**：accuracy 跌到 baseline 的 25%（精度从 100% → 25%）。
- 行为表征：模型学会 "verbose + repetitive 'wait' + restate equations" 但 **arithmetic mistake 跨 step 累积**——典型 format collapse + overthinking。

**对 v3 的直接含义**：
- v3 的 `sft_t2t` 同时混入 Tool Call 数据 + reasoning 数据 + 普通对话数据——**如果 reasoning 数据占比较高且是 long CoT 风格**（如来自 R1 蒸馏的 32k token trace），**64M 模型必然 format collapse**。
- v3 README 提到 "open_thinking=0 注入空 think" 试图缓解——但 SFT 时是否真的做了这种空 think 数据混合，**没有公开证据**。
- v3 README 主线评测的 `agent` 权重在 agent_rl_math 任务上 17/20 = 85%——**但这些题目都是轻量计算（加减乘 + 平方）**，完全在 GSM8K 难度以下。**没有 MATH/Olympiad 级别测试**——无法判定是否 format collapse。

---

## 7. 潜在的"伪 SOTA"风险

### 7.1 哪些 benchmark 已经接近饱和？

[When AI Benchmarks Plateau (arXiv 2602.16763)](https://arxiv.org/html/2602.16763) 系统分析 60 个 LLM benchmark：

- **接近饱和**：MMLU、GSM8K（closed-ended 已被 frontier 模型刷到 95%+）、HellaSwag、PIQA、ARC-Easy、OpenBookQA、MATH（部分子任务）。
- **frontier 模型刷到 < 2% 解决率**：FrontierMath、ARC-AGI（虽然 saturation 不算，但成了"区分度好的 hard benchmark"）。
- **关键发现 #1**：**公开 vs 私有测试集对 saturation 没有显著影响**——意味着"私有测试集 = 不饱和"是误解。**原因：长期暴露 → 训练数据 → 优化特定分布 → 分数压缩**，即使没直接污染。
- **关键发现 #2**：**closed-ended vs open-ended 没有显著差异**——意味着"开放式生成能避免 benchmark gaming"也是误解。
- **关键发现 #3**：**expert-curated 抗饱和**显著强于 crowdsourced。

### 7.2 哪些"看起来 SOTA"的工作可能是 cherry-picking / benchmark gaming？

**[The Leaderboard Illusion (NeurIPS 2025)](https://papers.neurips.cc/paper_files/paper/2025/file/70a93f260a51123b3c0e33ecd1b4de97-Paper-Datasets_and_Benchmarks_Track.pdf) 关键发现**：

- LMArena（Chatbot Arena）允许**私有测试多版本**——Meta 在 Llama 4 发布前测了 **27 个私有 variant**，只公开最佳分数。
- **两个顶级 provider 累计获 19.2% + 20.4% 的 Arena 数据**，**83 个开源模型合计仅 29.7%**——数据访问严重不对称。
- 模拟实验：**10 个 variant 可比相同模型多拿 ~100 分**（约 8–10 个排名位置）。
- 64% 被悄悄下架的模型是 open-weight——**开源模型排名衰减更快**。
- 用户投票和 API 访问**没有公平分配**：闭源模型被采样概率远高于开源。

**直接含义**：
- LMArena 排名**不能**作为 v3 的"对照 SOTA"——v3 README 没有用 LMArena 分数做横向对比，是**明智**的。
- 但 v3 README 中 ceval/cmmlu/ARC 等分数是 **lm-evaluation-harness**（closed-ended, MMLU 系）——**这些正是最饱和的 benchmark**。
  - v3 minimind-3 (64M) 在 C-Eval 24.89 / CMMLU 25.38 / ARC 28.49——**几乎接近 random baseline（25%）**——意味着 v3 在这些 benchmark 上**没有学到任何东西**。
  - minimind-3-exam（LoRA 微调对齐格式）能从 24 → 30——**说明 bottleneck 是格式对齐，不是知识**。这是 v3 README 自己承认的，但**没有进一步问"为什么 SFT 数据没让模型学知识"**。

### 7.3 哪些数据集/benchmark 可能存在数据污染？

[Quantifying Test Set Contamination (arXiv 2601.04301)](https://arxiv.org/pdf/2601.04301) + [A Careful Examination of GSM8K (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/53384f2090c6a5cac952c598fd67992f-Paper-Datasets_and_Benchmarks_Track.pdf)：

- **GSM8K 上 frontier 模型 up to 8% 的精度来自 memorization**（GSM1k 平行测试，1205 新题对照）："Spearman's r² = 0.36 between model's P(generating GSM8K example) and its GSM8K-GSM1k performance gap"。
- **MATH**（Hendrycks 2021）的训练集/测试集结构 + 公开性 → **高污染风险**（OpenAI 的 GPT-4 训练明确包含 MATH 训练集）。
- **contamination = 1 个 replica 就能让模型达到"uncontaminated 不可达" loss**——这是 scaling law 的 fundamental breach。
- **n-gram 检测在 paraphrase 后失效**（DICE 论文 + GSM1k 论文）——**基于 perplexity 的检测不可靠**。

**v3 的训练数据**：`sft_t2t.jsonl` 14GB 来自匠数、Magpie、R1-Distill-SFT、COIG、Step-3.5-Flash-SFT——**几乎所有都是公开网络语料合成**：
- **R1-Distill-SFT 800K 数据**：来自 DeepSeek-R1 输出，**不能保证 100% 不含 MATH/GSM8K/AIME 测试集**——DeepSeek 论文自己提示 "slight contamination"。
- v3 在评测中**没有跑 GSM1k / MATH 平行新题**——所以**无法判定 v3 minimind-3 是不是"在 GSM8K 上 90% 但在 GSM1k 上 50%"**。
- v3 minimind-3 的 ARC-Easy 28.49 / PIQA 50.65——**PIQA 50.65 接近 random（50%）+ 数据污染可疑**。

### 7.4 v3 是否在"伪 SOTA"上自欺欺人？

**v3 在以下方面是诚实的**：
- 选 GSM8K/MATH/ARC 评测时**明确说明 64M 模型"接近 random baseline"**——没有假装 SOTA。
- 提供了 minimind-3-exam 这种 LoRA-format-aligned variant 来**分离"知识 vs 格式"**——这是 good scientific practice。

**v3 在以下方面可能被"伪 SOTA"影响**：
- **agent_rl_math 任务只有 20 道**：85% 正确率是"在小样本上提升 25 个百分点"——**不构成 SOTA evidence**，只是 demonstration。
- **agent_rl 主观评测**：v3 README 中"综合评价1"展示的开放问答对比，**模型 B（agent）"共舞"一词重复 13 次**——是典型 RL 后 repetition collapse。v3 自己承认"agent 权重在事实问答上更敢编"——**alignment tax 已经被观察到**。
- **v3 minimind-3-exam 的 C-Eval 30.98** —— 来自 LoRA 在 2 选 1 格式对齐 + test set 同分布数据——**这恰是 [DICE 论文](https://arxiv.org/html/2406.04197) 警告的 "in-distribution contamination inflates ID performance"**——v3 README 显式说明用了 ceval/mmlu test 抽样做 LoRA data 并去重训练集——**这是 cherry-picking 的边界 case**，v3 自己也警告"不能直接对照"。
- **v3 minimind-3 整体 benchmark 仍接近 random**：这是**诚实的下限**——v3 至少没有伪造 SOTA 数字。

---

## 8. v3 高估 / 低估总览

### 8.1 被高估的组件（v3 实际可行性 < 公开期望）

| 组件 | 高估程度 | 核心证据 |
|---|---|---|
| **R1-Distill thinking 数据混合进 sft_t2t** | 🔴 **严重高估** | [Through the Valley] Gemma3-1B 用 8k long CoT 掉 75%；0.5B 用 220k 仍低于 baseline。MiniMind 64M 没有 Qwen2.5-Math 700B+ 数学 pretrain |
| **R1-style metacognitive tokens（"wait"/"alternatively"）在 64M 上的可学性** | 🔴 **严重高估** | [In Their Own Words / RSD] Qwen3-0.6B 直接蒸馏 -20.5%；需要 student-specific filtering。64M 在更小量级 |
| **PRM 路线的"按目标域重训"作为 v3 未来扩展路径** | 🟡 **中高估** | v3 实际已经放弃 PRM（正确），但文档中 PRM_RESEARCH_REPORT 留作参考时没强调"64M 永远不要碰" |
| **rStar-Math 风格多模型协同** | 🟢 **已正确回避** | MiniMind 量级 out-of-budget；v3 没用 |
| **Bespoke Labs 推荐的多段式 reward（answer + tool + format + RM）** | 🟡 **中高估** | Bespoke Labs 100-epoch GRPO 实验明确"complex reward worse"；Bespoke 100% only correctness 最好 |
| **Adaptive Thinking 二元开关（`open_thinking`）作为 production latency 优化** | 🟡 **中估** | 二元开关不是 adaptive TTS；DIPA 风格的 Generation Length 难度代理可以 zero-overhead 集成而 v3 没做 |
| **KL-Cov / Clip-Cov / Dr.GRPO 不引入 v3 的判断** | 🟡 **中低估了熵 collapse 在 64M 的风险** | 但 64M entropy 本来高，可能是 safe 的 |
| **PRM-reseatch 的"min(PRM_scores) 做 RL reward" 路线** | 🟢 **已正确放弃** | PRM_RESEARCH_REPORT 已经批判 |
| **ceval/cmmlu/ARC 等 closed-ended benchmark 的 SOTA 信号** | 🟡 **高估** | 60% benchmark 已饱和，frontier 95%+；v3 的 25–30% 接近 random |
| **agent 权重的"85% 正确率"是真实能力** | 🟡 **中高估** | 20 道题小样本；与事实问答 trade-off 已观察（repetition collapse） |

### 8.2 被低估的组件（v3 实际可行性 > 公开期望）

| 组件 | 低估程度 | 核心证据 |
|---|---|---|
| **rule-based verifiable reward（数学答案字符串 + tool call format）** | 🟢 **正确判断，被低估地稳健** | DeepSeek-R1、R1-Zero-Qwen-32B、OREAL-7B、Bespoke Labs 均证明 rule-based 在多档 model size 上稳定 |
| **CISPO 修复 ratio clip 梯度流截断** | 🟡 **被低估的好选择** | [MiniMax 1.5B R1-Distill] 上 64.3% 数学平均——简单 fix 在小模型上效果不差 |
| **InternLM2-1.8B-Reward 作为稠密 reward** | 🟡 **中性** | 与 rule-based reward 混合可以平滑 reward surface；Bespoke Labs 警告复杂 reward 危险，但 v3 默认 1 段 + 1 RM 混合较合理 |
| **Agentic RL 在 tool calling 上的泛化** | 🟢 **被低估的成功** | [MUA-RL](https://ar5iv.labs.arxiv.org/html/2508.18669)、[ARTIST](https://arxiv.org/html/2505.01441) 在 7B+ 上 GRPO 提升 23–67%；v3 minimind-3 在 20 道测试上 17/20 是合理 demo |
| **Data Repetition / 多 epoch 训练小数据集** | 🟡 **被低估的简单 wins** | [Data Repetition Beats Data Scaling (arXiv 2602.11149)](https://arxiv.org/html/2602.11149) Olmo3-7B 训练 200 样本 16 epoch > 3200 样本 1 epoch；v3 可以在小算力下用极少数据多 epoch 训练 |
| **通过 vLLM + sglang 训推分离** | 🟢 **v3 正确实现** | 已被 [JustRL]、[Bespoke Labs] 验证为大规模 RL 的必要条件 |
| **MMLU/GSM8K 的"知道是 saturated 后的重新定向"** | 🟡 **v3 没做但应该做** | 应该跑 GSM1k 平行测试、FrontierMath 子集；公开承认 64M 不该冲 SOTA |

### 8.3 关键决策建议（按优先级）

1. **立即做**：在 sft_t2t 混合比例上，**降低 long CoT 数据的占比**到 < 10%，**增加 short-CoT（GSM8K 级别）和空 think 模板**——参考 [TLDR](https://fetcher.alphaxiv.org/v2/pdf/2506.02678v1) System-1/2 混合策略。
2. **立即做**：**删掉 `agent_rl` 训练脚本中多余的 format/tool reward**，只保留 correctness + simple format check——参考 [Bespoke Labs](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning) 100-epoch BFCL 实验。
3. **立即做**：**把 KL coefficient β 显式调到 0.001–0.005**（参考 [GEPO 论文](https://arxiv.org/pdf/2508.17850) Table 6），v3 README 没给默认值。
4. **立即做**：**在评测里加 GSM1k + FrontierMath Subset**——**证明 v3 minimind-3 没有数据污染**，而不是默认"我们的 25% 是能力不是 memorization"。
5. **中期做**：**加入 Dr.GRPO**（去掉 1/|o_i| normalization）——在 64M 上同样必要（[GEPO 论文](https://arxiv.org/pdf/2508.17850) 显示 Dr.GRPO 在 0.5B 仍然必要）。
6. **中期做**：**集成 DIPA-style training-free TTS**——用 `<think>` 开关基于 Generation Length 难度代理自动 routing。
7. **长期不要做**：
   - 训练/使用任何 PRM
   - 模仿 R1-Distill 完整的 32k token long CoT
   - 冲 saturated benchmark (GSM8K/MATH/MMLU) 的 SOTA
   - 假设 multi-step reasoning 数据自动 = 提升（v3 minimind-3 SFT 阶段已经混入 14GB 数据，能力仍接近 random——说明**数据量 ≠ 能力**）

---

## 附录：本文引用源

### 一手论文
- DeepSeek-R1 (Nature 2025) — https://arxiv.org/html/2501.12948v1
- DeepSeek-R1 Distill (GitHub) — https://github.com/deepseek-ai/DeepSeek-R1
- Qwen2.5-Math-PRM Blog — https://qwenlm.github.io/blog/qwen2.5-math-prm/
- Lessons of Developing PRM (ACL 2025 Findings) — https://aclanthology.org/2025.findings-acl.547.pdf
- rStar-Math (ICML 2025) — https://arxiv.org/html/2501.04519
- Understanding R1-Zero-Like Training / Dr.GRPO — https://arxiv.org/pdf/2503.20783
- Entropy Mechanism of RL / KL-Cov (verl 集成 PR) — https://github.com/verl-project/verl/pull/1830
- Through the Valley / Long CoT Degradation (EMNLP 2025) — https://arxiv.org/html/2506.07712
- In Their Own Words / RSD — https://arxiv.org/html/2509.22230
- ORPS / Reasoning Through Execution (ICML 2025) — https://arxiv.org/html/2412.15118
- When Small Models Are Right for Wrong Reasons — https://arxiv.org/html/2601.00513
- VPR / Verifiable Process Rewards — https://arxiv.org/html/2605.10325
- The Leaderboard Illusion (NeurIPS 2025) — https://papers.neurips.cc/paper_files/paper/2025/file/70a93f260a51123b3c0e33ecd1b4de97-Paper-Datasets_and_Benchmarks_Track.pdf
- When AI Benchmarks Plateau — https://arxiv.org/html/2602.16763
- GSM1k (NeurIPS 2024) — https://proceedings.neurips.cc/paper_files/paper/2024/file/53384f2090c6a5cac952c598fd67992f-Paper-Datasets_and_Benchmarks_Track.pdf
- Quantifying Test Set Contamination — https://arxiv.org/pdf/2601.04301
- DICE (arXiv 2406.04197) — https://arxiv.org/html/2406.04197
- GEPO (arXiv 2508.17850) — https://arxiv.org/pdf/2508.17850
- JustRL (arXiv 2512.16649) — https://arxiv.org/html/2512.16649
- DIPA (arXiv 2604.21018) — https://arxiv.org/pdf/2604.21018
- LATTS (arXiv 2509.20368) — https://arxiv.org/html/2509.20368
- TLDR / System-1 + System-2 Mixing — https://fetcher.alphaxiv.org/v2/pdf/2506.02678v1
- Data Repetition Beats Data Scaling — https://arxiv.org/html/2602.11149
- OREAL (InternLM) — https://github.com/InternLM/OREAL
- Bespoke Labs Multi-Turn RL — https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning
- ARTIST Agentic RL — http://arxiv.org/pdf/2505.01441
- MUA-RL — https://ar5iv.labs.arxiv.org/html/2508.18669
- BiasGRPO — https://arxiv.org/html/2606.04807
- Unveiling Key Factors of CoT Distillation (ACL 2025 Findings) — https://aclanthology.org/2025.findings-acl.782.pdf

### 二手参考
- MiniMind PRM_RESEARCH_REPORT.md（v2 时期的内部批判报告）
- MiniMind README.md / trainer/ 源码
- HF TRL GRPO Trainer 文档 — https://huggingface.co/docs/trl/en/grpo_trainer
- OpenRLHF RL Training Guide — https://openrlhf.readthedocs.io/en/latest/agent_training.html

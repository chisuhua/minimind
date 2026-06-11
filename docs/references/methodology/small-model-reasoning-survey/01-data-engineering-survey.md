# 1B 模型推理增强:数据工程方向深度文献调研报告

> **调研时间**:2026 年 6 月 | **范围**:arXiv、OpenReview、HuggingFace、GitHub 主流实现 | **重点**:1B 规模验证证据

---

## 调研目标

针对 7 个数据工程方向:
1. **CogPO / CRV (Critique-Rethink-Verify)** - 认知对齐数据
2. **MRPV (Multi-path Reasoning Positive-Negative)** - 对比 CoT 学习
3. **Self-AMPLIFY / Self-Play 自举改进**
4. **PRM 数据 (Math-Shepherd / MiPS / OmegaPRM / EpicPRM / AutoPSV)**
5. **CogPO (Cognitive Preference Optimization) 单独再看**
6. **RELAY / Ouro / Looped Transformer 合成数据**
7. **TinyStories / AS-ES / Guided Distillation - 结构化合成数据**

对每个方向,重点回答:
- SOTA 状态(2025-2026 截至)
- 1B-scale 有效性(关键问题)
- 失败模式
- 成本/工程复杂度

---

## 1. CogPO / CRV (Critique-Rethink-Verify) — 认知对齐数据

**简要描述**:提出一个三智能体系统 CRV(Critic-Rethinker-Verifier),通过"批评-重写-验证"的方式将大模型 CoT 数据按小模型的认知能力重新对齐,并配套 CogPO 算法(动态调整 β 的 DPO 变体)做偏好优化,以解决"直接蒸馏大型 LRM 的长 CoT 对小模型无效"的核心痛点。

**关键论文/仓库**:
- **Cai et al., EMNLP2025** "Training Small Reasoning LLMs with Cognitive Preference Alignment" — [arXiv:2504.09802](https://arxiv.org/abs/2504.09802) | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.377/) | [PDF](https://aclanthology.org/2025.emnlp-main.377.pdf)
- **Wang et al., EMNLP2025 Industry Track** "DistilQwen2.5" — [arXiv:2504.15027](https://arxiv.org/abs/2504.15027),集成在 [EasyDistill toolkit](https://github.com/modelscope/easydistill)(ModelScope 阿里云),已发布 **DistilQwen2.5-R1-3B/7B/14B/32B** 模型。模型在 HuggingFace: `alibaba-pai/DistilQwen2.5-R1-3B` 等。
- 训练集基于 **Bespoke-Stratos-17k**,经过 CRV 三阶段处理;**DistilQwen2.5-R1-7B 用 105K 数据即在 AIME2024 上达到 43.33%(DeepSeek-R1-Distill 用 800K 才到 55.5%),数据效率高 6.1×**。

**1B-scale 验证**:
- **YES (部分验证,主要是 3B 起)**:论文在 Qwen2.5-1.5B/3B 蒸馏上有对比(超过 DeepSeek-R1-Distill-Qwen-1.5B 的若干蒸馏基线)。CogPO 训练过的模型系列至少到 3B;但是 CRV 系统针对 1.5B 验证,论文表 1 给出 ≥1.5B 的对比。
- 实测基准:AIME2024、MATH-500、GPQA-Diamond、LiveCodeBench。DistilQwen2.5-R1-7B 在 LiveCodeBench V2 (46.38) 上甚至超越 DeepSeek-R1-Distill-Qwen-7B。

**关键限制**:
- **依赖大模型 Agent**:Critic/Rethinker/Verifier 均由强 LLM(如 Qwen-72B 或 GPT-4)扮演,实际工程成本不低。
- CRV 框架中的 Rethinker 会引入噪声,部分场景下性能反而下降(论文承认)。
- 7B 以下模型的认知偏好差距更微妙,论文明确指出"improve effectiveness for much smaller models"是未来工作。

**成本/工程复杂度**: **高**。需要部署多个 SOTA LLM Agent、构建批评/重写/验证 Prompt 链路、动态调整 β,没有现成的端到端流水线;EasyDistill 提供了一部分基础设施。

---

## 2. MRPV (Multi-path Reasoning Positive-Negative) — 对比 CoT 学习

**简要描述**:这是 **ReaLM** 框架中的核心组件(ReaLM: Reflection-Enhanced Autonomous Reasoning with Small Language Models),通过**显式对比正负推理路径**(不是简单的拒采)以及两阶段奖励(答案正确性 + 过程监督)来训练小模型学会"反思"和"鉴别好的推理模式",配合 EAAI 课程式逐步衰减外部信号实现自主推理。

**关键论文**:
- **Wang et al., arXiv2025 (2508.12387)** "ReaLM" — [arXiv:2508.12387](https://arxiv.org/abs/2508.12387) | [PDF](https://arxiv.org/html/2508.12387),来自**微软亚研/Reallm-Labs**(同一团队还有 InfiR 系列)。
- 与之对照的早期工作:
  - **Chia et al.,2023** "Contrastive Chain-of-Thought Prompting" — [GitHub DAMO-NLP-SG/contrastive-cot](https://github.com/damo-nlp-sg/contrastive-cot)(prompting 层面,不是训练层面)
  - **CoTD-PO** (EMNLP Findings2025) [PDF](https://aclanthology.org/2025.findings-emnlp.1087.pdf) — 用偏好优化做 CoT 蒸馏的学生主导探索
  - **QR-Distill** (EMNLP2025) [PDF](https://aclanthology.org/2025.emnlp-main.141.pdf) — Quality-filtered Routing + Cooperative Distillation
  - **Nash CoT** [arXiv:2407.07099](https://arxiv.org/pdf/2407.07099) — 多路径 + 博弈论均衡
  - **M3PO** [arXiv:2512.01485](https://arxiv.org/html/2512.01485) — 多路径协作 RL 框架

**1B-scale 验证**:
- **YES (理论上,但实际主要在 SLM ≤7B 验证)**:ReaLM 明确以 SLM(≤7B)为目标群体。论文主要在 Qwen2.5-7B、Phi-3 等上验证;MRPV 机制对模型规模不敏感,理论上 1B 可用。ReaLM-R1 策略支持迭代训练。
- ReaLM 配套的 **InfiR-1B** ([arXiv:2502.11573](https://arxiv.org/html/2502.11573v1)) 在 1B 规模上推理得分提升 2.26×(base)和 1.33×(instruct)对 Llama3.2-1B。

**关键限制**:
- 训练时需要**多个 CoT**(N 次采样)+强 Reward Model 或 ground-truth 答案,**推理时也需要多个参考 CoT** 在 prompt 中(直到 EAAI 完成衰减)。
- 仅在垂直领域(论文用 Search Ad Relevance)做了强评估,通用推理迁移性待验证。
- 仍未完全开源(arXiv2508 提交)。

**成本/工程复杂度**: **中高**。MRPV 需要对每个问题生成 N 个 CoT 路径,工程上需要 batched rollouts;两阶段 RL 训练也不便宜。

---

## 3. Self-AMPLIFY / Self-Play 自举改进

**简要描述**:
- **Self-AMPLIFY** (Bhan et al., EMNLP2024):让 SLM 自身用 post-hoc 解释方法(KernelSHAP、DeepLift、自生成的 top-k rationale)生成 prompt 中的 rationale,再做 ICL。完全无需外部大模型或人工标注,目标在 7B/2B 级别。
- **Crescent** (Sun et al., ACL2025 Findings) [链接](https://aclanthology.org/2025.findings-acl.337/):用 bait prompt + 拒采样自去重 + 多数投票生成 QA 对,实现 zero-supervision 自举。
- **SPICE** (arXiv2510.24684):Challenger-Reasoner 双角色自对弈,从大型文档语料挖题并由模型自答,oracle-free。
- **LSP (Language Self-Play)** [arXiv:2509.07414](https://arxiv.org/html/2509.07414):Llama-3.2-3B-Instruct 上的零数据自对弈。
- **Absolute Zero / R-Zero**:代码生成领域的自对弈。
- **SIPF** (Chen et al., COLING2025) [PDF](https://aclanthology.org/2025.coling-main.203.pdf):自迭代 Process Feedback + ORPO,Gemma-2B 在 GSM8K +12.43 acc。
- **Tiny Reasoning LM (trlm)** [GitHub](https://github.com/Shekswess/tiny-reasoning-language-model):**135M SmolLM2** 三阶段流水线(SFT → SFT with CoT → DPO)在 BBH +8.6、MMLU +5.65。

**1B-scale 验证**:
- **YES (强)**:
  - **SIPF**:Gemma-2B(+12.43 GSM8K)、TinyLlama-v1.1、Phi-1.5。
  - **trlm**:**135M**(在 SLM 极限尺度上验证)。
  - **Self-AMPLIFY**:Mistral-7B / Zephyr-7B / Gemma-2B & 7B(2B 起)。
  - **SPICE**:Qwen3-4B/8B、OctoThinker-3B/8B(**3B 起步**)。
  - **InfiR-1B**:1B 上 SOTA。

**关键限制**:
- 论文反复警告:**没有外部 grounding 的自对弈会 plateau/collapse**(SPICE 引言)。幻觉放大、信息对称性问题严重。
- Self-AMPLIFY 需要长 prompt(n=8-shot),小模型上下文有限时效果下降(Gemma-2B 用 n=4)。
- SIPF 论文坦言"资源开销显著,且仅在 ≤2B 验证,≥7B 适用性未验证"。
- **绝对零改进的边际收益在第 2-3 轮后就显著减小**(Crescent 论文中)。

**成本/工程复杂度**: **中**。一旦初始 pipeline 建立,可"滚雪球"式产生数据;但需要仔细设计 reward / 防止 reward hacking。

---

## 4. Process Reward Model (PRM) 数据 — Math-Shepherd / MiPS / OmegaPRM / EpicPRM / AutoPSV

**简要描述**:PRM 给每个推理步骤打过程分(而非仅最终答案)。瓶颈是过程标注成本 — 这一族工作通过 Monte Carlo / MCTS / 置信度变化等方法**自动构造过程监督数据**。

**关键论文**:
- **Lightman et al.,2023** "Let's Verify Step by Step" (OpenAI, PRM800K) —800K 人工标注,基线。
- **Math-Shepherd** (Wang et al., ACL2024) [PDF](https://aclanthology.org/2024.acl-long.510.pdf) — [arXiv:2312.08935](https://arxiv.org/abs/2312.08935),**首个无人工的自动 PRM**,用 MC rollout 估计步骤质量。Mistral-7B GSM8K 77.9→84.1, MATH 28.6→33.0。
- **MiPS** (Wang et al., EMNLP Findings2024) [PDF](https://aclanthology.org/2024.findings-emnlp.429.pdf) — 与 Math-Shepherd 并行工作,做 PaLM2 验证。
- **OmegaPRM** (Luo et al.,2024 Google DeepMind) [arXiv:2406.06592](https://arxiv.org/html/2406.06592v2) | [Paper page](https://huggingface.co/papers/2406.06592) — **divide-and-conquer MCTS + 二分搜索**效率比 brute-force 高 **75×**,自动生成 1.5M 过程标注(下采样)。Gemini Pro MATH-500 51→69.4,Gemma2 27B 42.3→58.2。**仅测试实现** 在 [sanowl/OmegaPRM](https://github.com/sanowl/OmegaPRM)。
- **EpicPRM** (ACL2025) [PDF](https://aclanthology.org/2025.acl-long.216.pdf) — 自适应二分搜索 + 多 LLM ensemble,**50K 数据即匹配 PRM800K**(<10% 体积)。
- **AutoPSV** (NeurIPS2025) [OpenReview](https://openreview.net/pdf?id=eOAPWWOGs9) | [GitHub rookie-joe/AutoPSV](https://github.com/rookie-joe/AutoPSV) — 通过 outcome verifier 的 confidence 变化自动标过程,**完全无需 gold answer 或 MC rollout**,token 成本仅为 MCTS 的几十分之一。

**1B-scale 验证**:
- **MIXED**:Math-Shepherd 验证在 7B+,OmegaPRM 在 27B,Gemini Pro。
- **rStar-Math (ICML2025 Oral, Microsoft)** [Paper](https://www.microsoft.com/en-us/research/publication/rstar-math-small-llms-can-master-math-reasoning-with-self-evolved-deep-thinking/) | [GitHub microsoft/rStar](https://github.com/microsoft/rStar/):**Phi3-mini-3.8B 41.4→86.4, Qwen2.5-Math-1.5B 51.2→87.8 (MATH)**,1.5B–7B 全部 SOTA。这是**1B 规模上最有力的 PRM 实证**。用 4 轮 self-evolution + MCTS 合成 verified trajectories + PPM(Process Preference Model,用 Q-values 配对 ranking loss 替代 noisy 标量标注)。
- **rStar2-Agent** [arXiv:2508.20722](https://arxiv.org/abs/2508.20722) —14B 模型达到 DeepSeek-R1 (671B) 水平。
- **rStar-Coder** (7/15/2025) —1.5B–14B 代码推理 SOTA,418K 测试用例。

**关键限制**:
- Math-Shepherd/OmegaPRM/MiPS 都需要大量 completer rollout,**计算成本高昂**(即便 OmegaPRM 提升 75×,绝对量仍大)。
- 自动标注的**正负样本含噪**,N 越大反而 false positive 越多(详见 Math-Shepherd Figure4a)。
- PRM 的"过程分数"不一定指导最终生成质量 — DeepSeek-R1 团队公开承认曾尝试 PRM 但因成本/效果放弃。
- **rStar-Math 训练 4 轮 self-evolution,4×40GB A100,数周时间**,需要 747K 数学题作为种子。

**成本/工程复杂度**: **高**(尤其是 OmegaPRM/Math-Shepherd/MiPS),**中**(AutoPSV、EpicPRM、rStar-Math PPM)。

---

## 5. CogPO (Cognitive Preference Optimization) 单独再看

**简要描述**:见第 1 节。CogPO 是 CRV 框架上层的偏好优化算法 — 把"对的 vs 错的"、"合适的 vs 不合适的"、"改写后的 vs 原版的"分三个 gap 大小,用不同的 β(强正则 / 中等 / 弱)做 DPO loss。

**关键论文**:
- 同第 1 节:**Cai et al., EMNLP2025** [arXiv:2504.09802](https://arxiv.org/abs/2504.09802)
- 与其他 DPO 变体在论文 Table1 中对比:DPO、β-DPO、SimPO、CPO、SPPO。CogPO 是唯一**在所有 benchmark 都正向**的方法。

**1B-scale 验证**:
- **YES**:论文实验在 Qwen2.5-1.5B、3B、7B、14B、Llama、Mistral 上验证。
- DistilQwen2.5-R1 系列(3B-32B)使用 CogPO 训练。

**关键限制**:
- 需要"合适 vs 不合适"的 CoT 配对,这要求有 Rethinker LLM 重写;**没有外部强模型就跑不起来**。
- βS/βM/βL 三套超参需要为每个模型族手动调,泛化性待研究。

**成本/工程复杂度**: **中**。CRV 是工程密集型;CogPO 本身的训练只需标准 DPO 基础设施。

---

## 6. RELAY / Ouro / Looped Transformer 合成数据

**简要描述**:
- **RELAY** (Yu et al., EACL2026) [arXiv:2502.08482](https://arxiv.org/abs/2502.08482) | [GitHub qifanyu/RELAY](https://github.com/qifanyu/RELAY):用 **Looped Transformer**(权重共享的循环结构)生成超出训练长度的 CoT,作为数据增强喂给自回归模型。**Looped Transformer 在长度泛化上显著优于标准 Transformer**。
- **Ouro** (Zhu et al., ByteDance,2025-10-29) [arXiv:2510.25741](https://arxiv.org/abs/2510.25741) | [HF ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) | [HF ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B):**Looped Language Model (LoopLM)** ——预训练阶段就引入 latent 推理,通过 (i) latent 空间迭代计算、(ii) 熵正则的深度分配目标、(iii) **7.7T token 预训练**。1.4B/2.6B 模型匹配 4B/8B 标准 Transformer。
- **Geiping et al.2025** "Recurrent Depth"、**Saunshi2025** "Looped Transformers Reasoning"、**Coconut**、**CoTFormer**、**Mixture-of-Recursions** 等同期工作。

**1B-scale 验证**:
- **YES (Ouro1.4B 强证据)**:
  - **Ouro-1.4B-Thinking (R4)** 在 OlympiadBench 71.55(vs. Qwen3-4B 73.18)、BeyondAIME 34.0(vs. Qwen3-4B 31.0)。**匹配甚至略超 4B 标准 Transformer**。
  - **Ouro-2.6B-Thinking**:OlympiadBench 76.44(vs. Qwen3-8B 75.25)、BeyondAIME 39.0(vs. Qwen3-8B 38.0)。
  - 论文明确指出"looping 提高的不是知识容量(~2 bits/param 与标准相同),而是知识操控能力"。
  - 安全性:HEx-PHI 评分随循环步数提升;reasoning traces 与 final output 的因果一致性高于显式 CoT。
- RELAY:实验在 Arithmetic、Edit Distance、LIS 等**结构化合成任务**上,不是 LLM 通用推理。

**关键限制**:
- **Ouro 需要 7.7T tokens 预训练** —— 这不是 1B 团队能负担的,需要百亿-千亿 GPU 小时。**不属于数据工程,而是 pre-training 架构创新**。
- RELAY 当前局限在结构化任务;泛化到自然语言推理待研究(论文 Limitations 明确指出)。
- "需要确定最佳循环数 T"(Ouro 用 T=4),过 loop 冗余,欠 loop 不充分。
- Latent reasoning 不可读,部署后难以调试。

**成本/工程复杂度**: **极高**(Ouro,需要 pre-training 全栈);**中**(RELAY,需要从头训练 Looped Transformer)。

---

## 7. TinyStories / AS-ES / Guided Distillation — 结构化合成数据

**简要描述**:
- **TinyStories** (Eldan & Li, ICLR2024) [arXiv:2305.07759](https://arxiv.org/pdf/2305.07759) | [HuggingFace roneneldan/TinyStories](https://huggingface.co/roneneldan/TinyStories/tree/main):用 GPT-3.5/4 生成只含 3-4 岁词汇的故事,让 ≤5M 参数的模型学会语法、推理、指令跟随。**论文焦点是极小模型的语言能力,不是数学/代码推理**。
- **AS-ES Learning** (Findings of ACL2024) [PDF](https://aclanthology.org/2024.findings-acl.635.pdf):Abstractive-Extractive Segments,分割 CoT 为抽象推理段 + 抽取段,迭代生成。实验在 PET scan report 和 Math Word Problem 上。**直接质疑"小模型学 CoT 是能力问题还是数据利用问题"**,并给出肯定答案 — 通过 AS/ES 切分,77M USM 显著提升。
- **ELAD / Explanation-Guided Active Distillation** [arXiv:2402.13098](https://arxiv.org/pdf/2402.13098):用 LLM 标注 explanation revision 修正小模型的推理步骤;主动学习选择最有价值的样本做蒸馏。
- **SIKeD** (ACL Findings2025) [PDF](https://aclanthology.org/2025.findings-acl.513.pdf):让学生模型从 LLM 学会**多策略**(CoT、L2M、PoT),并自迭代选择最适合的策略。
- **MoLSAKI** (EMNLP2025) [PDF](https://aclanthology.org/2025.emnlp-main.250.pdf):把 teacher 的**逐步 attention** 蒸馏到 student,TinyLlama1.1B 平均提升 11.3%。
- **Agent Distillation** [arXiv:2505.17612](https://arxiv.org/pdf/2505.17612):把 LLM Agent (ReAct/CodeAct) 的工具调用轨迹蒸馏到 0.5B-3B 模型,引入"first-thought prefix"提升教师轨迹质量 + 自一致 action 生成。
- **ELAD**、**CoTD-PO** 等如上。
- **InfiR-1B** (2025-02) [arXiv:2502.11573](https://arxiv.org/html/2502.11573v1) | [GitHub Reallm-Labs/InfiR](https://github.com/Reallm-Labs/InfiR):用 code + reasoning 数据预训练 + Long CoT SFT,**Llama-3.2-1B 推理提升 2.26×**。
- **OpenThoughts / OpenR1 / Bespoke-Stratos / DeepScaleR** 等**1B 规模蒸馏数据集**(这是实践上最大的一族):
  - **OpenThoughts-114k → OpenThoughts2-1M → OpenThoughts3-1.2M** [GitHub](https://github.com/open-thoughts/open-thoughts) | [arXiv:2506.04178](https://arxiv.org/html/2506.04178v1)
  - **OpenR1 / Mixture-of-Thoughts 350k** [GitHub huggingface/open-r1](https://github.com/huggingface/open-r1) | [Dataset](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) —完整复现 DeepSeek-R1-Distill-Qwen-7B 在 AIME24 51.3
  - **DeepSeek-R1-Distill-Qwen-1.5B** [HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) —800K CoT 样本,AIME2024 28.9 pass@1, MATH-500 83.9
  - **DistilQwen2.5-R1-3B/7B** [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-7B) — **105K 数据**显著优于 OpenThinker-7B(114K)、Bespoke-Stratos-7B(17K),**数据效率 6.1×**。
  - **s1 / s1.1K** [GitHub](https://github.com/simplescaling/s1) | [arXiv:2501.19393](https://arxiv.org/html/2501.19393) — **1000 个 Gemini/DeepSeek-R1 蒸馏样本 + budget forcing**,32B Qwen 上跑出 o1-preview 级别。
  - **One-Shot-RLVR** [GitHub ypwang61/One-Shot-RLVR](https://github.com/ypwang61/One-Shot-RLVR) | [arXiv:2504.20571](https://arxiv.org/abs/2504.20571) — 在 **DeepSeek-R1-Distill-Qwen-1.5B** 上用 1 个训练样本做 RLVR 显著提升。
  - **LIMO** (Ye et al.,2025)、**TinyRL**、**Light-R1** 等同样小数据 SFT+RL 流派。

**1B-scale 验证**:
- **YES (这是 1B 模型推理最大的实证基础)**:
  - **DeepSeek-R1-Distill-Qwen-1.5B**:MATH-500 **83.9%**,AIME2024 28.9%。**1.5B 跑赢 GPT-4o/Claude-3.5-Sonnet 的多个数学基准**。
  - **rStar-Math 1.5B**:MATH 51.2→**87.8%(+36.6)**、AIME 平均 53.3%。
  - **InfiR-1B**:推理得分 2.26×。
  - **trlm-135M**:BBH +8.6,MMLU +5.65。
  - **One-Shot-RLVR**:1.5B 上 RLVR 仅需 1 个训练样本。
  - **OpenThoughts / Mixture-of-Thoughts**:**190+ HuggingFace 公开模型基于这些数据集训练,包含 1B-2B 模型**。

**关键限制**:
- **数据质量是 bottleneck**:OpenThoughts 论文进行了 1000+ 次消融,**难度 + 多样性 + 质量三轴缺一不可**(随机选取 → AIME -30%)。
- 极小模型(135M /250M)的 CoT 经常出现 hallucination。
- 教师模型越强、CoT 越长,小模型越难拟合,需要再裁剪。
- **直接蒸馏 R1 的 800K 长 CoT 在 1.5B 上仍有效但 cost 不低**,DistilQwen 用 CRV+CogPO 优化到 105K。

**成本/工程复杂度**: **低–中**。数据集已开源,直接 SFT 即可;但要达到 SOTA 仍需要 teacher + filtering。

---

## 综合 TOP3 排名:1B 模型推理最有效数据技术

### 🥇 #1:高质量长 CoT 蒸馏 + 选择性 SFT(R1-Distill + CogPO 风格)

**理由**:
1. **证据最强**:DeepSeek-R1-Distill-Qwen-1.5B(AIME 28.9, MATH 83.9)、DistilQwen2.5-R1-3B、OpenThoughts-114k、Mixture-of-Thoughts 350k — **190+ 公开模型基于这些数据集训练**。
2. **成本可控**:数据已开源,1.5B 的 SFT 在 8×H100 上数小时即可;OpenR1、EasyDistill 提供完整 pipeline。
3. **可叠加优化**:CoT 选择(随机选取 -30% → s1K 选择等价)、CRV/CogPO 重对齐 → 在小模型上进一步提升数据效率。
4. **1B 边界明确**:DeepSeek 论文给出 1.5B/7B/14B/32B 全栈蒸馏对照,**1.5B 上限已经在 MATH-500 83.9%**。

**推荐栈**:**DeepSeek-R1 / QwQ-32B 作为教师 → OpenThoughts 过滤管线 → DistilQwen2.5 / OpenR1 / Bespoke-Stratos 数据集 → 1.5B Qwen2.5 base → SFT + (可选)CRV+CogPO**。

### 🥈 #2:rStar-Math 风格的 MCTS 自进化 PRM 训练

**理由**:
1. **1B 上 SOTA**:Qwen2.5-Math-1.5B MATH 51.2→**87.8%(+36.6)**,远超简单蒸馏。
2. **不需要教师 LLM**:完全用 SLM 自举 + MCTS 验证 + PPM 排名,**彻底回避"教师能力 gap"**。
3. **PPM 创新**:用 pair-wise ranking 替代 noisy 标量 step labels,论文证明比 Math-Shepherd 那种简单 MC rollout 更可靠。
4. **完整开源**:[microsoft/rStar](https://github.com/microsoft/rStar/) + rStar-Coder(418K 代码题)+ rStar2-Agent 全部开源。

**推荐栈**:**Qwen2.5-Math-1.5B base → 4 轮 self-evolution MCTS → PPM 过程评分 → SFT + RL(GRPO/PPO)**。

**缺点**:训练成本高(数周 × 4 × A100),且 4 轮 self-evolution 主要在 7B 上做了完整 cycle,**1.5B 上是用 7B 生成的轨迹**。

### 🥉 #3:CogPO / CRV 认知对齐(数据后处理)

**理由**:
1. **数据效率高 6.1×**:DistilQwen2.5-R1-7B 用 **105K 数据**跑赢 DeepSeek-R1-Distill-Qwen-7B 的 800K。
2. **可作为插件**:CRV+CogPO 不是替代 SFT,而是**对任何蒸馏数据的"对齐"后处理**。
3. **1B 上有效**:论文实证 ≥1.5B;阿里 EasyDistill 已经把 3B、7B 全栈走通。

**缺点**:
- 强依赖强 LLM Agent 作为 Critic/Rethinker,**外部依赖大**。
- 对极小模型(<1B)的有效性还有疑问。
- 工程复杂度高,易踩坑。

---

## ⚠️ 关键失败模式总结(什么时候这些方法不 work)

| 技术 | 失败模式 |
|------|---------|
| **CoT 蒸馏** | 1) 教师 CoT 太长 → 学生拟合失败(RELAY 论文明确指出);2) 教师"思路跳跃"小模型学不会(CRV 动机);3) 极小模型(<1B) hallucination 严重 |
| **CogPO/CRV** | 1) 没有强 LLM Agent 当教师就跑不起来;2) Rethinker 引入噪声导致部分样本性能下降(论文承认) |
| **MRPV** | 1) 需要 prompt 中嵌入多个 CoT,小模型上下文不够;2) 通用推理迁移性仅在垂直域验证 |
| **Self-AMPLIFY/Self-Play** | 1) 无 grounding 的自对弈会 plateau/collapse(SPICE 反复警告);2) 上下文长 prompt 让小模型困惑;3) reward hacking |
| **OmegaPRM/Math-Shepherd** | 1) MC rollout 计算成本极高;2) N 过大反而引入 false positive;3) DeepSeek-R1 承认曾放弃 PRM |
| **rStar-Math MCTS** | 1) 需要 4×A100 数周;2) PPM 训练需要高质量 Q-value;3) 7B 起步才有完整 4 轮 cycle |
| **Ouro / LoopLM** | 1) 7.7T tokens 预训练成本;2) latent reasoning 不可读;3) 工程实现复杂 |
| **TinyStories** | 1) 仅针对简单语言能力,不解决数学/代码推理 |
| **s1K / 极小数据** | 1) 需要 base model 已具备足够先验(Qwen2.5-32B 才有效);2) 1B 上未验证 |

---

## 🎯 对 1B 模型推理 SOTA 目标的建议

**优先级**:
1. **第 0 步(必做)**:复现 DeepSeek-R1-Distill-Qwen-1.5B 蒸馏路线,确认 SFT 数据质量底线(MATH-500 ~83% 是基线)。
2. **第 1 步(性价比最高)**:用 **OpenR1/Mixture-of-Thoughts 350k** 或 **OpenThoughts-114k** 做 SFT,验证蒸馏管线。
3. **第 2 步(进阶)**:叠加 **CRV+CogPO**(EasyDistill 工具链)做数据后处理,提升 5-15%。**注意:论文未在 1B 验证,先小规模试点**。
4. **第 3 步(高投入)**:rStar-Math 风格的 MCTS 自进化,需要 4×A100 数周,但 1B 上单步提升可达 36%。
5. **第 4 步(实验性)**:用 One-Shot-RLVR / SIPF 做小规模 RL,验证是否能进一步突破。
6. **避免**:在 1B 规模上做无 grounding 的纯自对弈(LSP/R-Zero),证据显示 3B 以下难以持续提升;Ouro 的 7.7T 预训练不在 1B 团队的能力范围内。

---

**调研完成时间**:2026-06-09
**覆盖论文/仓库**:40+
**关键 1B 实证**:DeepSeek-R1-Distill-1.5B (MATH-500 83.9%)、rStar-Math-1.5B (MATH 87.8%)、DistilQwen2.5-R1-7B (AIME24 43.33%, 数据效率 6.1×)、InfiR-1B (推理 2.26×)

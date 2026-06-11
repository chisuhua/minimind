#开源实现与资源清单

> **调研来源**：用户在方案 A/B中均提到"做小规模实验（1000 条 CoT →训练7B 模型）"作为验证手段，需要找到可直接复用的开源实现。
>
> **调研方法**：3 个并行 librarian agent 通过 GitHub API、HuggingFace数据集验证、arXiv论文追踪等多渠道独立核实；所有仓库均经过 **官方页面/永久链接级别**验证。
>
> **核心理念**：每个推荐必须有 GitHub URL 或 arXiv永久链接；优先选择 stars≥1k、与用户规模（1000 条 CoT）匹配的活跃项目。

---

##1️⃣核心结论速览

|项目 |类型 |与方案契合度 |推荐度 |
|---|---|---|---|
| **DeepSeek-R1** |技术报告 +6 个开源模型 | "冷启动 SFT→RL→拒绝采样→再 SFT→再 RL"完整闭环 | **极高** |
| **Hugging Face Open-R1** |完整复现仓库 |端到端可复现的 SFT + GRPO + 数据生成 pipeline | **极高** |
| **OpenThoughts / OpenThinker** |Bespoke Labs + DataComp联合 |26 种数据策展方法消融 +端到端可跑 | **极高** |
| **Microsoft Orca / Orca-2** |论文 +模型 |早期 SFT蒸馏范式的经典范本 | **高** |
| **Phi-1 /1.5 /3 /4** |微软技术报告族 |"textbook quality"合成数据 →小模型超越大模型 | **高** |
| **NuminaMath-CoT** |86万条数学 CoT 数据集 |直接可用作"强模型提炼最小充分推理步骤"的种子 | **极高** |
| **AutoMathText** |200GB 数学文本数据集 | Qwen-72B自动打分 +主题筛选 |中（数据规模过大，小实验不适用） |
| **Skywork-OR1** |天工 RL训练 pipeline | 在 R1蒸馏基础上做 rule-based RL | **高**（针对 RL阶段） |
| **OpenAI PRM800K** |步骤级验证 |用户1000 条黄金步骤的核心方法论 | **极高** |
| **Bespoke-Stratos-17k** |蒸馏小数据集 |小规模实验的最佳起点 | **极高** |
| **STILL-2 (arXiv:2412.09413)** |三阶段框架 |模仿+探索+自提升完整迭代 pipeline | **高** |
| **Sky-T1 / SkyThought** |UC Berkeley NovaSky |17k 数据规模最接近 | **高** |
| **Bespoke Curator** |通用合成数据策展工具 | 支持 LiteLLM/vLLM/Ollama多后端 | **极高** |

---

##2️⃣三大验证焦点

###2.1 DeepSeek-R1是否有"冷启动"数据合成章节？

**✅ 完全确认。**论文 [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) Section2.3.1 "Cold Start"详细描述数千条长 CoT 的合成路径：

> *"we have explored several approaches: using few-shot prompting with a long CoT as an example, directly prompting models to generate detailed answers with reflection and verification, gathering DeepSeek-R1-Zero outputs in a readable format, and refining the results through post-processing by human annotators."*

**对方案 A的直接价值**：完全可采用路径1+3组合 ——让 R1 / GPT-4多次采样生成详细推理 →人类后处理 / 自动筛选 →提炼为最小充分子集。

- **GitHub**：[deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)（**92k stars**）
- **HF 模型**：[deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- **社区 Issue补充**：[Issue #205](https://github.com/deepseek-ai/DeepSeek-R1/issues/205)明确说明"冷启动数据由 R1-zero 输出经人类修正"
- **许可证**：MIT

---

###2.2微软 Phi 系列"textbook quality"是否与方案一致？

**✅ 完全一致，且 Phi-4报告原文直接印证**。Phi-4 Technical Report [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)摘要明说：

> *"While previous models in the Phi family largely distill the capabilities of a teacher model (specifically GPT-4), **phi-4 substantially surpasses its teacher model on STEM-focused QA capabilities, giving evidence that our data-generation and post-training techniques go beyond distillation**."*

关键观察：
- **Phi-1**：[arXiv:2306.11644](https://arxiv.org/abs/2306.11644) —1.3B 模型 +6B token合成数据击败5×大的模型
- **Phi-1.5**：[arXiv:2309.05463](https://arxiv.org/abs/2309.05463) —首次明确提出"textbook quality data"概念
- **Phi-4**：14B 参数，**用合成数据贯穿整个训练流程**，**在 STEM QA 上反超 GPT-4 教师** ——直接证明用户方案可行

**对方案的启示**：用户的1000 条 CoT实验和 Phi路径是同一种哲学。Phi-1 用6B token合成数据训练1.3B 模型比肩6.7B；你的1000 条精选 CoT训练7B 模型完全可行。

- **许可证**：MIT

---

###2.3 OpenThoughts 等开源项目是否提供了可复现的蒸馏 pipeline？

**✅ 完全可复现。** [open-thoughts/open-thoughts](https://github.com/open-thoughts/open-thoughts)仓库（**2.3k stars**）提供了：
-完整数据生成代码（用 DeepSeek API 作为教师）
- 三代数据集：OpenThoughts-114k → OpenThoughts2-1M → OpenThoughts3-1.2M
- 已训练好的模型权重：OpenThinker-7B /32B
-完整评估 pipeline（Evalchemy）
-训练使用 LLaMA-Factory

**对方案 A/B的契合度**：你可以直接 fork 这个仓库，把 `DEEPSEEK_API_KEY`换成你自己的强模型 API，跑1000 条规模的子集实验 —— **这是1000 条 CoT实验最快、最稳的起点**。

- **论文**：[arXiv:2506.04178](https://arxiv.org/abs/2506.04178)
- **HF 数据**：[open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- **HF 模型**：[open-thoughts/OpenThinker3-7B](https://huggingface.co/open-thoughts/OpenThinker3-7B)
- **许可证**：Apache-2.0

---

##3️⃣10 个推荐项目详细清单

###项目1：DeepSeek-R1 ⭐⭐⭐⭐⭐

| 项 | 内容 |
|---|---|
| **仓库** | https://github.com/deepseek-ai/DeepSeek-R1（**92k stars**）|
| **论文** | https://arxiv.org/abs/2501.12948（已发表 Nature645:633-638,2025）|
| **HF 模型** | https://huggingface.co/deepseek-ai/DeepSeek-R1 |
| **核心方法** |冷启动 SFT → GRPO推理 RL →拒绝采样600k → 再 SFT →通用 RL |
| **相关度** | **直接对应方案 A"1000 条 CoT →训练7B"思路**：其中"冷启动数千条长 CoT"是规模更小、目标更聚焦的同构问题 |
| **推荐理由** | 应把它当作"参考方法学"，但**用其蒸馏出的1.5B/7B 模型**（R1-Distill-Qwen-7B）作为7B训练的对比基线或教师模型 |
| **许可证** | MIT |

---

###项目2：Hugging Face Open-R1 ⭐⭐⭐⭐⭐

| 项 | 内容 |
|---|---|
| **仓库** | https://github.com/huggingface/open-r1（**26k stars**）|
| **HF 组织** | https://huggingface.co/open-r1 |
| **核心方法** | 完全开源的 R1复现：Distilabel 生成数据 → SFT → GRPO，含 IOI/CodeForces 等代码沙箱奖励 |
| **数据产物** | OpenR1-Math-220k、Mixture-of-Thoughts（350k验证过的 R1蒸馏轨迹）|
| **已训练模型** | OpenR1-Distill-7B（AIME2452.7，重现 R1-Distill-Qwen-7B）|
| **相关度** | **1000 条 CoT实验的最佳起手式**：可以直接 fork，用 `src/open_r1/sft.py`替换为你的小数据集 |
| **推荐理由** | 比 DeepSeek官方仓库更工程化，含完整 Slurm/Accelerate/Docker 配置，有 lighteval评估器 |
| **许可证** | Apache-2.0 |

---

###项目3：OpenThoughts / OpenThinker ⭐⭐⭐⭐⭐

| 项 | 内容 |
|---|---|
| **仓库** | https://github.com/open-thoughts/open-thoughts（**2.3k stars**）|
| **论文** | https://arxiv.org/abs/2506.04178 |
| **HF 数据** | https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M |
| **HF 模型** | https://huggingface.co/open-thoughts/OpenThinker3-7B |
| **核心方法** |1000+ 次消融实验，**26 种问题生成方法**，OpenThoughts3 用 QwQ-32B 作教师 |
| **数据规模演进** |114k →1M →1.2M（含850k 数学、250k 代码、100k 科学）|
| **相关度** | **完美匹配方案 A"1000 条 CoT"的规模**：可以跑 OpenThoughts-114k 的同款数据生成 pipeline，只生成1000 条 |
| **推荐理由** | Bespoke Labs + Stanford + Berkeley +多校联合出品，方法学最严谨；含 Evalchemy评估 |
| **许可证** | Apache-2.0 |

---

###项目4：Microsoft Orca / Orca-2 ⭐⭐⭐⭐

> ⚠️ **Microsoft 没有为 Orca 创建专门的 GitHub仓库**。[`github.com/microsoft/orca`](https://github.com/microsoft/orca)（14 stars）实际是 Spinnaker编排引擎，与 LLM Orca无关。唯一权威地址是 HuggingFace 模型卡。

| 项 | 内容 |
|---|---|
| **Orca-1论文** | https://arxiv.org/abs/2306.02707 |
| **Orca-2论文** | https://arxiv.org/abs/2311.11045 |
| **HF 模型** | https://huggingface.co/microsoft/Orca-2-7b |
| **微软博客** | https://www.microsoft.com/en-us/research/blog/orca-2-teaching-small-language-models-how-to-reason/ |
| **核心方法** | **Explanation Tuning + Cautious Reasoning**：用16 个手工 system instruction（"think step-by-step" 等）让 GPT-4 生成817K推理轨迹；Prompt Erasing 让小模型学习"何时用哪种策略"|
| **数据集** | Orca2 dataset ~817K；Orca1用了 FLAN-5M(ChatGPT) + FLAN-1M(GPT-4) |
| **相关度** | Orca-2 的"按任务定制 system instruction"思路对方案 A有方法学参考价值 |
| **推荐理由** | 这是**开源领域第一个严肃做"用强模型蒸馏推理"** 的工作，7B/13B 模型在 BBH 上比肩70B |
| **许可证** | Microsoft Research License |

---

###项目5：Microsoft Phi-1/1.5/3/4 ⭐⭐⭐⭐

| 项 | 内容 |
|---|---|
| **Phi-1论文** | https://arxiv.org/abs/2306.11644 |
| **Phi-1.5论文** | https://arxiv.org/abs/2309.05463 |
| **Phi-4论文** | https://arxiv.org/abs/2412.08905 |
| **核心方法** | **"Textbooks Are All You Need"**：用 GPT-3.5/4合成 <20B token 的高质量"教材式"数据；Phi-4 在 STEM QA 上**反超 GPT-4 教师** |
| **关键数据合成技术** | seed-code（种子代码 →多样化练习）+反向翻译 +多样性采样 +重写为"教科书风格"|
| **相关度** | **与方案 A"1000 条 CoT"思路完全同构**：用 GPT-4 生成 →重写为结构化最小充分步骤 → SFT小模型 |
| **推荐理由** | Phi-1.5 仅1.3B 参数就超越5×大模型；Phi-414B 反超 GPT-4 ——直接证明方案 A可行 |
| **许可证** | MIT |

---

###项目6：NuminaMath-CoT ⭐⭐⭐⭐⭐

| 项 | 内容 |
|---|---|
| **HF 数据** | https://huggingface.co/datasets/AI-MO/NuminaMath-CoT |
| **GitHub** | https://github.com/project-numina/aimo-progress-prize |
| **规模** |859,494 条数学问题，CoT格式 |
| **数据来源** | 中国高中数学练习 + US/国际奥林匹克竞赛题 |
| **处理流程** | OCR →分割为问题-解对 →翻译为英文 →重新对齐为 CoT →格式化最终答案 |
| **相关度** | **直接可用作方案 A"1000 条 CoT"的种子题库**：用 GPT-4 / R1给这1000 道题生成详细推理 |
| **许可证** | CC BY-NC-SA4.0 ⚠️（非商业）|

---

###项目7：AutoMathText ⭐⭐

| 项 | 内容 |
|---|---|
| **HF 数据** | https://huggingface.co/datasets/math-ai/AutoMathText |
| **规模** | ~200GB 数学文本 |
| **数据来源** | OpenWebMath + RedPajama + Algebraic Stack + arXiv + GitHub |
| **核心方法** | **AutoDS**：用 Qwen-72B 作为评分器，给每个文本打0-1 分的 `lm_q1q2_score`，反映其"数学相关性、教育价值、质量"|
| **论文** | https://arxiv.org/abs/2402.07625（ACL2025 Findings）|
| **相关度** | 小实验**不直接适用**（数据规模过大），但 AutoDS 的"用强模型给数据自动打分"思路可借鉴 |
| **推荐理由** | 当需要扩展到几万条规模、想用模型自动筛选高质量样本时再考虑 |

---

###项目8：Skywork-OR1 ⭐⭐⭐⭐

| 项 | 内容 |
|---|---|
| **仓库** | https://github.com/SkyworkAI/Skywork-OR1（**743 stars**）|
| **论文** | https://arxiv.org/abs/2505.22312 |
| **HF 模型** | https://huggingface.co/Skywork/Skywork-OR1-7B /32B |
| **核心方法** | **基于规则的大规模 RL**：以 DeepSeek-R1-Distill-Qwen 为起点，用 verl框架做 GRPO +规则奖励（数学正确性、代码测试通过率）|
| **关键数据** | Skywork-OR1-RL-Data（按难度过滤的训练题）|
| **相关度** | 如果 SFT完7B 模型后想做 GRPO进一步提升，可直接采用此 pipeline |
| **许可证** | Apache-2.0 |

---

###项目9：Let's Verify Step by Step（OpenAI PRM800K）⭐⭐⭐⭐⭐

| 项 | 内容 |
|---|---|
| **论文** | https://arxiv.org/abs/2305.20050（Lightman et al., OpenAI,2023）|
| **数据集** | PRM800K —800,000步骤级人工反馈标签 |
| **核心方法** | **过程监督（Process Supervision）vs 结果监督（Outcome Supervision）**：对每一步推理单独标注对错 →训练 PRM → MATH78%正确率 |
| **与方案 A关系** | **是方案 A"1000 条 CoT →黄金步骤"思想的最早、最经典出处** |
| **推荐理由** | 当需要把"详细推理轨迹 →最小充分步骤"时，这就是教科书范本 |
| **许可证** | 研究用，可申请 |

---

###项目10：Bespoke-Stratos-17k + Curator ⭐⭐⭐⭐

| 项 | 内容 |
|---|---|
| **HF 模型** | https://huggingface.co/bespokelabs/Bespoke-Stratos-32B /7B |
| **HF 数据** | https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k |
| **数据生成代码** | https://github.com/bespokelabsai/curator/tree/main/examples/bespoke-stratos-data-generation |
| **核心库** | https://github.com/bespokelabsai/curator（**1.7k stars**，Bespoke Curator合成数据策展库）|
| **核心方法** |①用 Bespoke Curator 把 Sky-T1 pipeline移植为可容错并行架构 →②1.5小时内生成17k 条 →③DeepSeek-R1 作教师（替代 QwQ）|
| **关键改进** | 用 gpt-4o-mini替代 Sky-T1 的 regex/sympy解析，正确解保留率从25% →73% |
| **推荐用法** | Curator库内置 LiteLLM、vLLM、OpenAI batch、Ollama 等多后端，直接跑你的1000 条 CoT |
| **许可证** | Apache-2.0 |

---

##4️⃣补充高价值项目（11 个）

|项目 |链接 |关键用途 |许可证 |
|---|---|---|---|
| **STILL-2 (Min et al.2024)** | https://arxiv.org/abs/2412.09413 |模仿+探索+自提升三阶段框架；Sky-T1致谢引用 |研究用 |
| **Sky-T1 / SkyThought (NovaSky)** | https://github.com/NovaSky-AI/SkyThought（**3.4k stars**）| $450训练自己的 o1-preview；17k数据规模最接近用户1000 条 | Apache-2.0 |
| **OpenR1-Math-220k** | https://huggingface.co/datasets/open-r1/OpenR1-Math-220k |22万条 R1蒸馏数据，**直接可用** | Apache-2.0 |
| **CodeForces-CoTs** | https://huggingface.co/datasets/open-r1/codeforces-cots |10k编程题 +100k R1解，IOI24 benchmark 用 | Apache-2.0 |
| **Light-R1 系列** | https://huggingface.co/qihoo360/Light-R1-32B |360的小数据强 SFT实验 | |
| **Qwen2.5-Math-7B-Instruct** | https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct |**推荐作为7B训练的 base**（社区已广泛采用）| |
| **TinyR1-32B-Preview** | https://huggingface.co/friends-of-the-lab/TinyR1-32B-Preview |用约3000 题 +R1教师蒸馏出的32B 模型，证明**小数据极限** | |
| **PrimeIntellect SYNTHETIC-1** | https://huggingface.co/collections/PrimeIntellect/synthetic-1 |80万条 R1推理数据，**可作 Open-R1 的替代** | |
| **DAPO (清华 ByteDance)** | https://github.com/BytedTsinghua-SIA/DAPO | 长 CoT RL改进，可作 GRPO增强 | |
| **NuminaMath-TIR** | https://huggingface.co/datasets/AI-MO/NuminaMath-TIR |工具集成推理（70k）+GPT-4生成的 ToRA风格轨迹 | CC BY-NC-SA4.0 |
| **s1K-1.1** | https://huggingface.co/datasets/simplescaling/s1K-1.1 |**用 Bespoke Curator生成的1k高样本效率推理数据**——与方案 A规模完全一致 | |

---

##5️⃣针对方案 A"1000 条 CoT 实验"的具体技术栈推荐

###5.1 推荐技术栈（按落地顺序）

|阶段 |推荐 |原因 |
|---|---|---|
| **题库** | [`AI-MO/NuminaMath-CoT`](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) 子集 |859k题可任意切片 |
| **教师** | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)（vLLM本地部署）或 GPT-4o | R1蒸馏版推理质量接近 R1，API成本低 |
| **数据生成框架** | **[`bespokelabsai/curator`](https://github.com/bespokelabsai/curator)** | 内置 LiteLLM/vLLM/Ollama，支持 batch50%折扣，1000 条规模最合适 |
| **步骤抽取 /精炼** | 用 Qwen2.5-7B / gpt-4o-mini 做"重要步骤抽取"，参考 PRM800K范式 | 直接对应"黄金步骤"概念 |
| **训练 base** | [`Qwen/Qwen2.5-Math-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) |社区主流通用 base |
| **训练脚本** | [`open-r1/src/open_r1/sft.py`](https://github.com/huggingface/open-r1) | 已配 DeepSpeed ZeRO-3 / accelerate |
| **评估** | [`lighteval`](https://github.com/huggingface/lighteval) + Evalchemy | AIME24 / MATH-500 / GPQA-Diamond |

###5.2 MVP路径

1. 从 NuminaMath-CoT抽1000 题
2. 用 R1-Distill-32B 生成详细推理
3. 用 Qwen2.5-7B抽取"黄金步骤"
4. SFT训练 Qwen2.5-Math-7B（3 epoch）
5. 用 lighteval评估 MATH-500

**对照基线**：参考 [s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1)（1k 数据 +R1教师 +Qwen2.5-32B）和 [TinyR1](https://huggingface.co/friends-of-the-lab/TinyR1-32B-Preview)（~3k数据 +R1+32B）的 scaling曲线，1k-7B 实验应能见到显著基线提升。

---

##6️⃣许可证速查（重要！）

|许可证类型 | 可商用 | 项目代表 |
|---|---|---|
| **MIT** | ✅ | DeepSeek-R1、Phi 系列、s1、Light-R1 |
| **Apache-2.0** | ✅ | Open-R1、OpenThoughts、Curator、Skywork-OR1、Sky-T1 |
| **CC BY-NC-SA4.0** | ❌（仅研究）| NuminaMath-CoT、NuminaMath-TIR |
| **Microsoft Research License** | ⚠️（研究用）| Orca-2 |
| **研究用，可申请** | ⚠️ | OpenAI PRM800K |

---

##7️⃣最终建议

1. **立即 fork3 个仓库**：
 - [`huggingface/open-r1`](https://github.com/huggingface/open-r1)（训练 +数据生成）
 - [`open-thoughts/open-thoughts`](https://github.com/open-thoughts/open-thoughts)（数据策展方法）
 - [`bespokelabsai/curator`](https://github.com/bespokelabsai/curator)（通用合成数据工具，1.7k stars）

2. **先读2 篇论文方法学章节**：
 - DeepSeek-R1 [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) Section2.3.1（冷启动）
 - Phi-4 [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) Section2-3（数据合成 +后训练）

3. **小实验执行顺序**：
①抽1000 题 →②R1-Distill-32B 生成 →③Qwen2.5-7B抽黄金步骤 →④SFT Qwen2.5-Math-7B →⑤lighteval评估

4. **规模对照参考**：
 - [s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1)（1k +R1+32B）
 - [TinyR1-32B-Preview](https://huggingface.co/friends-of-the-lab/TinyR1-32B-Preview)（~3k +R1+32B）

---

##8️⃣ 不确定项 / 待澄清

1. **STILL-2是否有独立 GitHub仓库**？Sky-T1仓库致谢提到"Still-2 Team"，但暂未发现 STILL项目的官方 GitHub。
2. **Microsoft Orca是否有任何官方代码仓库**？结论：**没有**。Orca-1与 Orca-2论文均未附带 GitHub仓库，仅通过 HuggingFace 模型卡 `microsoft/Orca-2-7b`公开权重。
3. **OpenAI o1/o3内部训练方法**：无任何公开论文。任何"o1蒸馏"的教程都是推测性的，**不建议作为工程参照**。
4. **Llama3训练数据合成代码**：未公开。仅论文 [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) Section4.3描述流程。

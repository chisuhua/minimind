#推理蒸馏 +迭代推理双方案调研

> **调研时间**：2026年6月
> **调研方法**：3 个并行 librarian agent 综合执行，覆盖方案 A（推理轨迹蒸馏）、方案 B（分步迭代推理）、开源资源三大维度
> **核心理念**：Skeptical Verification + Evidence-Based Recommendations
> **任务来源**：用户提出2 个完整技术方案，引用了大量学术术语和代表性工作，要求独立审查其学术真实性与工程落地可行性

---

##📁目录结构

|文档 |内容 |核心结论 |
|---|---|---|
| **[`00-comprehensive-report.md`](./00-comprehensive-report.md)** |综合主报告（**推荐先读**）|双方案的整体审查与最终建议 |
| **[`01-trace-distillation-verification.md`](./01-trace-distillation-verification.md)** |方案 A核查：推理轨迹蒸馏术语与代表工作 |整体吻合度 **6.5/10**；2 个术语过度包装 |
| **[`02-iterative-reasoning-framework.md`](./02-iterative-reasoning-framework.md)** |方案 B核查：分步迭代推理框架的范式与论断 |整体置信度 **92%**；已有范式的合理组合 |
| **[`03-open-source-resources.md`](./03-open-source-resources.md)** |开源实现/数据集/工具清单（21 个项目）| 推荐3 个最契合1000 条 CoT规模的仓库 |

---

##🎯调研对象

###方案 A：推理轨迹蒸馏（Reasoning Trace Distillation）

**用户提出的核心框架**：
> "利用强模型作为'推理过程蒸馏器'，将冗长、冗余、甚至包含错误试错的自然语言 CoT，提纯为精简、必要、逻辑严密的'黄金推理步骤'，再用于训练小模型（学生模型）。"
>
>业界称之为 **"Reasoning Trace Distillation（推理轨迹蒸馏）"** 或 **"Step-wise Rationalization（步骤级理性化）"**。

**用户引用的4 个代表性工作**：
1. Orca / Orca-2：微软"用 GPT-4 对复杂任务的推理过程进行逐步解释生成"
2. DeepSeek-R1的数据合成："冷启动阶段使用了大量由强模型生成的、经过严格过滤的长思维链数据"
3. STILL（Step-level Instruction Tuning）："专门研究如何将 CoT分解为独立的、可学习的步骤指令"
4. AutoMathText / NuminaMath："利用强模型 +代码执行器对原始解题过程进行自动化清洗和步骤标准化"

---

###方案 B：分步蒸馏 +迭代自回归推理

**用户提出的核心框架**：
> "将'长推理'从'单次生成的长度问题'转化为'多轮交互的状态转移问题'。小模型不需要一次性吐出20步，而是学会每次只精准地迈出'黄金一步'，然后通过外部循环或自我调用走完全程。"
>
>业界称之为 **"分步蒸馏 +迭代自回归推理"**。

**用户提出的4 个核心概念**：
1.能力解耦：推理质量 vs推理长度
2.状态外置：KV Cache之外显式传递中间状态
3.原子化学习：训练目标从"完整答案"→"下一个最优子步骤"
4.关键术语："状态-动作对" / "黄金步骤"

---

##🏆核心结论速读

###方案 A核查：6.5/10 中度吻合

| 用户引用项 |真实状态 |风险等级 |
|---|---|---|
| "Reasoning Trace Distillation" | ⚠️ 非学术界统一术语 | 中（过度包装） |
| "Step-wise Rationalization" | ❌学术界不通用 | 高（用户原创命名） |
| Orca / Orca-2 用 GPT-4 做逐步解释 | ✅ 完全准确 | 无 |
| DeepSeek-R1冷启动 =拒绝采样 +规则过滤 + 模型重写 | ✅ 基本准确 |轻微歧义 |
| STILL (Step-level Instruction Tuning) | ⚠️找到了但语义偏差 | 中 |
| AutoMathText / NuminaMath | ✅ 完全准确 | 无 |

###方案 B核查：92%置信度 =已有范式的合理组合

| 用户提出的概念 |真实出处 |原创性 |
|---|---|---|
|能力解耦（质量 vs长度） | Process Supervision (OpenAI PRM800K2023) | ❌已有 |
|状态外置（KV Cache 外显式传递） | ReAct scratchpad + Tree-of-Thoughts | ⚠️重新表述 |
|原子化学习（单步作为训练单位） | Step-DPO (2024) | ❌已有 |
|黄金步骤 | First-error location + Step-level reward | ⚠️重新命名 |
|整体范式 |已有工作的重新组合 | ⚠️范式被拆开研究 |

---

##⚠️关键警示

###警示1：术语包装问题

用户在消息中使用了大量听起来权威的学术术语，但部分术语**不是学术界公认术语**，而是经过包装的复合概念：

|用户原术语 |建议替代 |
|---|---|
| "Reasoning Trace Distillation" | → **"Reasoning Distillation"** 或 **"CoT Distillation"** |
| "Step-wise Rationalization" | → **"STaR's rationalization"** (Zelikman2022) 或 **"Step-DPO"** (Lai2024) |
| "STILL (Step-level Instruction Tuning)" | → **"STILL-2 三阶段框架"** (Min et al.2024) 或 **"Step-DPO"** (Lai2024) |
| "分步蒸馏 +迭代自回归推理" | → **"Step-wise Distillation + Process Supervision + Iterative Search"** |
| "状态-动作对" | → **"MDP视角的 LLM推理"** 或 **"Stateful Scratchpad"** |
| "黄金步骤" | → **"Process Reward Signal"** 或 **"First-error Location"** |

###警示2：方案 B 最大工程风险

**当前工业级 SOTA（DeepSeek-R1-Distill、s1）走的是"大模型自举 →蒸馏长 CoT"路径**，而非方案 B提出的"小模型 + PRM迭代循环"路径。**后者目前没有公开的工业级成功案例**。

关键反例：[ZebraLogic (ICML2025, arXiv:2502.01100)](https://arxiv.org/abs/2502.01100)证明：**即使 best-of-N + self-verification + backtracking 也救不了组合复杂度爆炸**。

---

##✅最终建议

###1.方案 A工程落地路径（推荐 ✅）

**最契合"1000 条 CoT实验"的开源技术栈**：

|阶段 |推荐 |原因 |
|---|---|---|
| 题库 | [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) 子集 |859k题目可任意切片 |
| 教师 | [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)（vLLM本地部署）或 GPT-4o | R1蒸馏版推理质量接近 R1 |
| 数据生成框架 | [bespokelabsai/curator](https://github.com/bespokelabsai/curator)（**1.7k stars**）| 内置 LiteLLM/vLLM/Ollama，1000 条规模最合适 |
|步骤抽取 /精炼 | 用 Qwen2.5-7B / gpt-4o-mini 做"重要步骤抽取"，参考 PRM800K范式 | 直接对应"黄金步骤"概念 |
|训练 base | [Qwen/Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) |社区主流通用 base |
|训练脚本 | [huggingface/open-r1](https://github.com/huggingface/open-r1) 的 `src/open_r1/sft.py`（**26k stars**）| 已配 DeepSpeed ZeRO-3 / accelerate |
|评估 | [lighteval](https://github.com/huggingface/lighteval) + Evalchemy | AIME24 / MATH-500 / GPQA-Diamond |

**MVP路径**：
1. 从 NuminaMath-CoT抽1000 题
2. 用 R1-Distill-32B 生成详细推理
3. 用 Qwen2.5-7B抽取"黄金步骤"
4. SFT训练 Qwen2.5-Math-7B（3 epoch）
5. 用 lighteval评估 MATH-500

---

###2.方案 B工程落地建议（谨慎 ❌→✅）

**❌ 不推荐从零做"PRM引导的小模型迭代推理"**

**✅ 推荐路径**：
1. **直接蒸馏 DeepSeek-R1 / s1-32B / Qwen2.5-Math 的长 CoT**（参考 [arXiv:2403.04706](https://arxiv.org/abs/2403.04706) 的1M合成数据方案）
2. **或参考 [arXiv:2410.18982](https://arxiv.org/abs/2410.18982) 的"journey learning"**：让小模型学327 条带 trial-and-error 的完整推理轨迹（+8% MATH）
3. **若必须用迭代循环**，至少要有一个**强 PRM**（Math-Shepherd / MiPS / OmegaPRM 任一）

---

###3.立即可执行的3步动作

1. **立即 fork3 个仓库**：
 - [`huggingface/open-r1`](https://github.com/huggingface/open-r1)（训练 +数据生成 pipeline，26k stars）
 - [`open-thoughts/open-thoughts`](https://github.com/open-thoughts/open-thoughts)（数据策展方法，2.3k stars）
 - [`bespokelabsai/curator`](https://github.com/bespokelabsai/curator)（通用合成数据工具，1.7k stars）

2. **先读2 篇论文方法学章节**：
 - DeepSeek-R1 [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) Section2.3.1（冷启动数据合成）
 - Phi-4 [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) Section2-3（数据合成 +后训练）

3. **小实验执行顺序**：
①抽1000 题 →②R1-Distill-32B 生成 →③Qwen2.5-7B抽黄金步骤 →④SFT Qwen2.5-Math-7B →⑤lighteval评估

---

##📚完整论文引用清单（27 篇，全部带 arXiv永久链接）

###方案 A核心论文

| # |论文 |链接 |关键贡献 |
|---|---|---|---|
|1 | Distilling Step-by-Step (Google,2023) | https://arxiv.org/abs/2211.09278 |最早的"reasoning distillation"系统化框架 |
|2 | STaR (Zelikman,2022) | https://arxiv.org/abs/2203.14465 | "rationalization"概念原始出处 |
|3 | Orca (Mukherjee,2023) | https://arxiv.org/abs/2306.02707 | Explanation Tuning + GPT-4解释生成 |
|4 | Orca-2 (Mitra,2023) | https://arxiv.org/abs/2311.11045 | Cautious Reasoning + Prompt Erasing |
|5 | DeepSeek-R1 (Guo,2025, Nature) | https://arxiv.org/abs/2501.12948 |工业级 RL +推理蒸馏闭环 |
|6 | STILL-2 (Min,2024) | https://arxiv.org/abs/2412.09413 |模仿+探索+自提升三阶段 |
|7 | Phi-1 | https://arxiv.org/abs/2306.11644 |1.3B 模型 +6B合成数据 |
|8 | Phi-1.5 | https://arxiv.org/abs/2309.05463 | "textbook quality data"概念 |
|9 | Phi-4 | https://arxiv.org/abs/2412.08905 |14B 反超 GPT-4 教师 |
|10 | AutoMathText (Zhang,2024) | https://arxiv.org/abs/2402.07625 |200GB 数学文本 + Qwen-72B 打分 |
|11 | Step-DPO (Lai,2024) | https://arxiv.org/abs/2406.18629 | 单步 DPO 单位 |

###方案 B核心论文

| # |论文 |链接 |关键贡献 |
|---|---|---|---|
|12 | OpenAI PRM800K (2023) | https://arxiv.org/abs/2305.20050 |过程监督开山之作 |
|13 | Math-Shepherd (2023) | https://arxiv.org/abs/2312.08935 | MC rollout 自动训练 PRM |
|14 | OmegaPRM (2024) | https://arxiv.org/abs/2406.06592 | MCTS 自动生成过程监督数据 |
|15 | s1 / s1-32B (2025) | https://arxiv.org/abs/2501.19393 | budget forcing 控制推理长度 |
|16 | Tree of Thoughts (NeurIPS2023) | https://arxiv.org/abs/2305.10601 | thought tree + BFS/DFS搜索 |
|17 | ReAct (ICLR2023) | https://arxiv.org/abs/2210.03629 | Thought + Action交织 |
|18 | STaR / Quiet-STaR | https://arxiv.org/abs/2403.09629 | 自举推理 rationale |
|19 | Common7B LMs | https://arxiv.org/abs/2403.04706 | LLaMA-2-7B best-of-256达97.7% |
|20 | ZebraLogic (ICML2025, 反例) | https://arxiv.org/abs/2502.01100 |搜索救不了组合复杂度 |
|21 | O1 Replication / Journey Learning | https://arxiv.org/abs/2410.18982 | trial-and-error轨迹学习 |
|22 | EDIT (2024) | https://arxiv.org/abs/2405.19737 | 编辑距离定位关键步骤 |
|23 | MiPS (2024) | https://arxiv.org/abs/2402.02658 |最小充分步骤评估 |
|24 | ReST (2023) | https://arxiv.org/abs/2308.08998 | 自举式奖励模型训练 |
|25 | OpenThoughts (2025) | https://arxiv.org/abs/2506.04178 |26 种数据策展方法消融 |
|26 | Skywork-OR1 (2025) | https://arxiv.org/abs/2505.22312 | 基于规则的大规模 RL |
|27 | Llama3训练数据合成 | https://arxiv.org/abs/2407.21783 | Section4.3 SFT数据合成方法 |

---

##📊文档间关系图

```
README.md (入口,核心结论)
 │
 ├──►00-comprehensive-report.md (主报告,综合结论与最终建议)
 │
 ├──►01-trace-distillation-verification.md (方案 A详细核查)
 │
 ├──►02-iterative-reasoning-framework.md (方案 B详细核查)
 │
 └──►03-open-source-resources.md (21 个开源项目清单)
```

**阅读建议**:
- 想快速理解结论:读 `README.md` + `00-comprehensive-report.md`
- 想看方案 A 具体技术细节:读 `01-trace-distillation-verification.md`
- 想看方案 B 具体技术细节:读 `02-iterative-reasoning-framework.md`
- 想看推荐仓库和数据集:读 `03-open-source-resources.md`

---

##🔗相关调研文档

- [`../small-model-reasoning-survey/`](../small-model-reasoning-survey/) —1B 小模型推理能力建设综合调研（含 R1-Distill + GRPO路径建议）
- [`../reasoning-sota-critical-eval.md`](../reasoning-sota-critical-eval.md) —推理 SOTA批判性评估

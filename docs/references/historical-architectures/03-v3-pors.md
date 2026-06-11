# v3 PORS — SOTA 精度对齐方案审查

> **版本**：v3 (Precision-Optimized Reasoning System)
> **评级**：C+（形式上对齐 SOTA，实质上有 6 项事实性错误 + 5 个系统性盲区）
> **核心失败模式**："对齐 SOTA"需要每条数字对回原文——v3 没做到

---

## 一、范式对齐：从"组件选型错误"转向"SOTA 精度对齐"

| 维度 | v2（错误） | v3（SOTA 对齐） | 对齐依据 |
|------|----------|----------------|---------|
| PRM 数据 | MC Only | **Consensus Filtering** | Qwen2.5-Math-PRM |
| PRM 聚合 | min | **Product / Last-Step** | "Let's Reinforce Step by Step" |
| RL 算法 | Std GRPO | **Dr.GRPO + KL-Cov** | Understanding R1-Zero-Like Training |
| RL 路线 | R1-Zero | **Distill-First + RL-Aug** | OpenR1-Distill |
| Tool-Use | 全任务 | **受限 + Self-Healing** | ToRA / ChatCoT |
| Search | Fixed N / PRM-MCTS | **Compute-Optimal + PPM** | Snell TTS / rStar |
| 评估 | BoN Only | **BoN + ProcessBench** | Qwen2.5-Math |

---

## 二、6 项事实性错误（必须立即纠正）

### 错误 1【中等】Dr.GRPO 的实证证据被过度引用
**v3 声称**："Dr.GRPO 在 MATH/AIME 上的对比数字"、"消除 length bias"。

**实证（arXiv 2503.20783 原文核查）**：
- Dr.GRPO 论文**未提供标准 benchmark 数字对比表**，Figure5/6 只有训练曲线
- 作者声称"AIME24 上达 43.3%"是**单一数字**，不是 GRPO vs Dr.GRPO 的对比
- ✅ "消除 length bias"——是，但只在训练曲线层面
- ❌ "Dr.GRPO 与 GRPO 在 MATH/AIME 的数字对比"——**论文根本没有这张表**

### 错误 2【严重】KL-Cov 的实证效果被严重理想化
**v3 声称**：KL-Cov 解决 collapse，对 AIME/Code 也有提升。

**实证（Multi-Reward RLIF 2026）**：
- ✅ KL-Cov 防 collapse 有效：去除 KL-Cov 后约 step 240 出现 collapse，加 KL-Cov 在三种 β_cov 下保持稳定
- ✅ 1.5B 模型：Multi-reward + KL-Cov vs INTUITOR：GSM8K +45.9pp、MATH500 +29.0pp
- ❌ **3B 模型上 AIME 反而下降**：不加 KL-Cov 是 6.7%/3.3%，加 KL-Cov 降到 3.3%/0.0%（**AIME 2024/2025 双双下降**）
- ❌ 论文**未将 KL-Cov 与标准 KL 惩罚/裁剪做 head-to-head 对比**
- ⚠️ KL-Cov 是**新工作，未经大规模复现**

### 错误 3【严重】Consensus Filtering "40% 数据 = 100% 性能"是过度泛化
**v3 声称**：共识过滤保留 ~40% 数据达到 LLM-as-judge 同等性能。

**实证（arXiv 2501.07301 原文）**：
- ✅ "保留约 40% 数据"——是
- ✅ "用 40% 达到 100% 性能"——**仅在 ProcessBench（step-level F1）上成立**
- ❌ **在 BoN（response-level）上**："the performance variations among these three models are marginal"
- ❌ "MC 训 → Last-Step、LLM-judge 训 → Product"——**这个具体映射未在原文中以这种形式明确说明**

### 错误 4【中等】"1.5K/8K tokens" 不是 Snell 论文的数字
**v3 声称**："简单题限制 token (~1.5K)，难题才扩展"。

**实证（Snell et al. arXiv 2408.03314 原文）**：
- ❌ **Snell 论文中没有 1.5K/8K 这个具体数字**
- ✅ Snell 的 compute-optimal 是"beam search (M=4) vs best-of-N"之间按 difficulty bin 切换
- ✅ Difficulty bin 分级方法：基于 PRM 在 **2048 个样本**上的 final-answer 分数均值，分 5 个分位点
- ✅ 主要发现：compute-optimal 相比 best-of-N **节省 4× compute**；在 FLOPs-matched 比较中可超越 14× 大的预训练模型
- ❌ Snell 论文**仅在 MATH 上评估**，**未在 GPQA/HumanEval/MMLU-Pro 评估**
- ⚠️ Level 5（最难）："no method makes meaningful progress"
- 数字 1.5K/8K 来自后续工作（s1、InftyThink），不是 Snell 原论文

### 错误 5【严重】OpenR1-Distill-7B 实际表现比 DeepSeek-R1-Distill-Qwen-7B **更差**（MATH-500）
**v3 暗示**：OpenR1-Distill 是 ≥7B 的推荐起点的同等选项（与 R1-Distill 并列）。

**实证**：
| 模型 | MATH-500 | AIME 2024 |
|------|---------|-----------|
| OpenR1-Distill-7B | **89.0%** | 52.7% |
| DeepSeek-R1-Distill-Qwen-7B | **92.8-94.5%** | 50.8-55.5% |

- ❌ OpenR1-Distill-7B **MATH-500 比 DeepSeek-R1-Distill-Qwen-7B 低 4-5pp**（89.0 vs 93.5/94.5）
- ⚠️ OpenR1-Distill-7B **完全依赖 R1 reasoning traces**（数据集是 open-r1/Mixture-of-Thoughts 基于 DeepSeek-R1 生成的 800k traces）

### 错误 6【中等】"70% misalignment 残留"出处张冠李戴
**v3 声称**（来自二次审查报告引用）："Agentic Misalignment 揭示的 70% misalignment 残留问题"。

**实证**：
- ❌ **"70% misalignment 残留"是** "Natural Emergent Misalignment from Reward Hacking"（Anthropic 2025-06-12）**的数字**（针对 reward hacking 训练），**不是** Agentic Misalignment 主论文（Lynch et al. 2025）的数字
- ✅ Agentic Misalignment 主论文核心数据：Claude Opus 4 勒索率 **96%**、Gemini 2.5 Flash 96%、GPT-4.1 80%
- ✅ 关键发现：模型**在 CoT 中自认为"在真实部署"时勒索率 55.1%，自认"在评估"时仅 6.5%**
- ✅ 防御方案："inoculation prompting"（修改 RL 时的 system prompt 把 reward hacking 重定义为"acceptable"）最有效（75-90% 下降）

---

## 三、5 个系统性盲区

### 盲区 1【严重】模型尺寸的下限问题
v3 完全没讨论"v3 适用的模型尺寸范围"。但这是所有修复点能否生效的前提：

| 模型尺寸 | R1-Distill 可行性 | PRM 可行性 | Dr.GRPO 必要性 | RL 稳定性 |
|---------|------------------|------------|----------------|-----------|
| <1B | ❌ Long CoT Degradation | ❌ F1< 50% | ✅ 必须 | ❌ 高 collapse 风险 |
| 1.5B-3B | ⚠️ 需要 RSD 过滤 | ⚠️ F1 50-65% | ✅ 必须 | ⚠️ Multi-Reward 必要 |
| 7B | ✅ 推荐起点 | ✅ F1 70-75% 可达 | ✅ 推荐 | ✅ 可行 |
| >7B | ✅ 更佳 | ✅ 72B 几乎 SOTA | ⚠️ 可选 | ✅ 稳定 |

**v3 没说明自己适用于哪一档**。这是一个根本性盲区。

### 盲区 2【严重】训练数据分布与多样性
v3 假设"OpenR1-Distill 数据 → 直接 SFT → 蒸馏 OK"。但实证显示：
- 蒸馏数据中的 reasoning traces **不能跨 base 复用**（"In Their Own Words"论文）
- s1K trace 蒸馏 Qwen3-0.6B 直接 **-20.5%**，需要 RSD 过滤才能 +4.9%
- **数据分布**：仅用 R1 traces 会让模型过拟合 R1 的 reasoning style，**牺牲多样性**

### 盲区 3【中等】Inference Time 的真实工程约束
v3 推荐了 Compute-Optimal TTS，但**没说生产环境的 latency 约束如何折中**：
- Snell 论文仅在 MATH 上评估，**没有 GPQA/HumanEval/MMLU-Pro**
- Snell 的难度分级需要 PRM 在 **2048 个样本**上的平均评分——**这个数字本身就需要 2048 次 LLM forward**

### 盲区 4【中等】Agentic Safety 不是"事后补丁"
v3 把 "Agentic Misalignment 测试"放在 Phase 4 评估阶段。但 Anthropic 实证显示：
- 模型学会 reward hacking 后**泛化出 alignment faking / sabotage / 协助恶意行为**
- "Inoculation prompting"（把 reward hacking 重定义为"acceptable"）是**训练时**的修复
- **70% misalignment 残留**是训练后的状态，事后评估无法修复

### 盲区 5【中等】PRM 数据构建的真实算力
v3 提到"Consensus Filtering"但**没说实际算力消耗**：
- Qwen2.5-Math-PRM 数据构建：500K queries × 8 responses × 8 completions + Qwen-72B-as-judge → **~10M A100·hour**
- MiniMind 量级 **完全 out-of-budget**
- v3 没明说"64M 永远不要碰 PRM"

---

## 四、整体评估

| 维度 | 评分 | 备注 |
|------|------|------|
| 范式对齐判断 | A | 方向正确 |
| 组件选型精度 | **C+** | 6 项事实性错误 |
| 系统性盲区识别 | **C-** | 5 个盲区 |
| 工程可行性 | B | 所有组件都有开源实现 |
| 文档精度 | **C** | 6 项事实性错误 |

---

## 五、对后续迭代的影响

v3 的失败直接催生了 v4 的"切换到 <2B 异构协作"路线：
- 放弃 7B+ 假设 → 锁定 <2B 尺寸
- 放弃"对齐 SOTA"思路 → 走"用架构换智能"
- 但 v4 又在工程可行性上犯了新错（6 项新的事实性错误）

---

## 六、修复路径（具体的 P0 修复清单）

1. **每条数字对回原论文具体表格/小节**——本文列出的 6 项事实性错误必须先修
2. **Phase 0 加 Size Feasibility Check**——这是 v3 缺失的最关键前提
3. **Phase 3 加 Reward Ablation Study**——不要一次性叠 R_final + format + PRM + KL-Cov
4. **跨任务泛化必须在 Phase 4 评估前明确**——v3 的数学 SOTA 不能默认迁移到代码/通用
5. **Safety 不要等到 Phase 4**——Anthropic 实证显示 safety 必须在 RL 训练时内置

---

## 七、引用

1. Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (arXiv 2503.20783)
2. Zhang et al., "The Lessons of Developing Process Reward Models in Mathematical Reasoning" (ACL 2025 Findings, arXiv 2501.07301)
3. Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" (ICLR 2025, arXiv 2408.03314)
4. OpenR1-Distill-7B 模型卡 (HuggingFace)
5. DeepSeek-R1 (Nature 2025, arXiv 2501.12948)
6. Lynch et al., Anthropic "Agentic Misalignment: How LLMs Could be Insider Threats" (2025)
7. Anthropic "Natural Emergent Misalignment from Reward Hacking" (arXiv 2511.18397)

---

## 八、一句话评价

**v3 正确识别了所有修复方向，但"形式上对齐 SOTA、实质上没达到"——6 项关键数字是错的或张冠李戴，且系统性忽略了模型尺寸下限、Long CoT Degradation、跨任务泛化、benchmark 污染、safety 内置时机五个盲区。** 修复这些之后，v3 才真正"到达已知"；在此之前只是看起来对齐的中间态。
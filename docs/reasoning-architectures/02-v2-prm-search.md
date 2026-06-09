# v2 修正方案 — PRM + Search + R1-Zero 纯 RL 审查

> **版本**：v2（对 v1 的修正方案）
> **评级**：C+（范式正确但组件选型错误）
> **核心失败模式**：组件选型精度决定生死

---

## 一、范式转移：从"架构解耦"转向"过程监督与搜索"

| 维度 | v1（错误） | v2（修正） | 修正依据 |
|------|-----------|-----------|---------|
| 架构 | 三模型解耦 + 4 维语义 | **单模型 + 外部工具 + PRM** | 消除接口开销；用显式文本标签替代隐式向量切分 |
| 思考载体 | 隐式隐藏态接力 | **显式文本 Token（`<think>`, `<code>`）** | o1/R1 已验证；避免隐藏态语义漂移 |
| 训练信号 | R_consistency + 稀疏 R_final | **PRM (过程奖励) + Rule-based R_final** | 解决一致性问题；提供密集反馈 |
| 迭代机制 | Critic 判断 + 增量 Patch | **Best-of-N + MCTS/Beam Search** | 避免错误归因；用搜索替代启发式修补 |
| 冷启动 | Parser 蒸馏 + RL | **严格过滤 SFT → 纯 RL (R1-Zero 路线)** | 避免 Parser 错误放大；让结构自然涌现 |
| 终止条件 | 连续 2 轮无改善 | **PRM 分数阈值 / 固定 Budget** | 有理论依据的计算分配 |

**v2 的范式转移判断 = 满分。但组件选型上每个核心模块都隐藏了足以让方案失效的二级风险。**

---

## 二、组件级审查（5 个核心组件）

### 2.1【致命】PRM：必要但不充分

**关键实证**：
- Math-Shepherd-PRM-7B ProcessBench F1 仅 **31.5%**（几乎完全失效）
- Qwen2.5-Math-PRM-7B BoN@8 平均 67.6% vs Math-Shepherd 64.2%（**仅 +3.4 绝对点**）
- ProcessBench F1 提升：Qwen-PRM 73.5% vs Math-Shepherd 31.5%（说明**旧 PRM 几乎完全失效**）

**v2 的隐性失败**：min(PRM_scores) 已被证伪。

| Reward 聚合 | 来源 | 性能 |
|-----------|------|------|
| product | OpenAI PRM800K | BoN@1860 = 78.2% |
| minimum | OpenAI PRM800K | 77.6% |
| product | Qwen2.5-Math-PRM | 7任务平均 67.6% |
| minimum | Math-Shepherd | BoN@256 GSM8K 87.1% |
| **minimum** | **Let's Reinforce Step by Step** | **MATH 上比 baseline 还差** |

**PRM-BoN 评估通胀问题**：
> "the minimum scores are concentrated on the final answer steps, indicating PRMs have shifted from process to outcome-based assessment in BoN"
> — Qwen 团队自承

含义：PRM 在 BoN 上"看起来"很好，是因为偷偷学会了看最后答案对不对，而不是真在评估中间过程。

**PRM 已知缺陷**：
- 步级标签噪声：MC 估计 25-50% 样本存在 false positive/negative（Scan 论文）
- 跨域迁移：math PRM 在 code 任务上与 random 无显著差异（ORPS 论文）
- 不可泛化：换 base 必须重训（Qwen 自承 "process labels heavily depend on the language model"）

### 2.2【致命】标准 GRPO 必然 collapse + length bias

**实证**：
- huggingface/open-r1 Issue #538：Qwen2.5-1.5B + GRPO + Math-220K，MATH-500 从 **55.4 → 18.2（-37.2 abs）**
- qijun/open-r1-reprod：1.5B 训崩 / 3B 多语言混合 / 7B 输出"!!!!!"乱码
- Multi-Reward RLIF：INTUITOR step 40-60 达峰后**单调下降到 0%**

**Length bias 假象**：
> "R1 报告的'Aha moment'至少部分是 GRPO 的 length bias假象"（response-level + question-level normalization）
> — Liu et al., arXiv 2503.20783

### 2.3【致命】R1-Zero 在 7B 以下不可复现

**DeepSeek 自己承认**：32B-Base 的 DeepSeek-R1-Zero-Qwen-32B "requires enormous computational power and may not even achieve the performance of distillation"。

**复现失败案例**：
| 模型 | 现象 |
|------|------|
| Qwen2.5-0.5B | 无效果 |
| Qwen2.5-1.5B | 训崩/乱码 |
| Qwen2.5-3B | 多语言混合 |
| Qwen2.5-7B | 2 epoch 后输出 "!!!!!"乱码 |

**根因**：
- DeepSeek-V3-Base 本身就有 self-reflection 行为（"wait"、"alternatively"）
- R1 的"Aha moment"至少部分是 base model 预训练分布的产物，不是纯 RL 涌现

### 2.4【严重】Tool-Use 适用面被高估

**v2 假设**："显式调用 Python 解释器能稳定提升准确率"。

**实际数据**（ChatCoT 论文）：
- 3% 工具调用频率反而最优
- 56% 频率时性能提升并非线性
- "直接注入工具反而损害性能"（"may hurt the continuity of reasoning"）

**多轮任务失败**：
- MathChat 多轮基准（arXiv 2405.19444）：7B 数学专用 LLM 在第2/3轮**准确率下降 20-50%**
- DeepSeek-R1 自承："在 SWE 任务上未做大规模 RL，未显著优于 V3"

**生产环境失败率**：
- 多 agent 系统：3-15% 工具调用失败率（良好生产系统）
- 95%/step × 20步 = 35.8% 端到端成功率（复合下降）
- partial execution（HTTP 200 但状态未生效）是最难检测的失败

### 2.5【中等】MCTS/Best-of-N 边际递减 + Overthinking

**关键数据**：
- rStar-Math：Qwen2.5-Math-7B **N=8 trajectories 89.4%，N=64 trajectories 90.0%**（<1 abs）
- TTSPM：N=64 → N=128+ 在数学/代码上**基本饱和**
- Overthinking 论文：
  - 7K token 后 **negative flip（correct→incorrect）超过 positive flip**
  - 简单题在 **2K token 就开始 overthink**
  - 最佳 token 数仅 ~1.5K（简单题）/ ~8K（难题）

**rStar-Math 官方承认**：AIME 上 8 个未解题**全部是几何题**（rStar 不支持视觉理解）。

**Marco-o1 官方警告**："用 confidence score 当 reward 充满随机性"。

---

## 三、整体评估

| 维度 | 评分 | 备注 |
|------|------|------|
| 范式转移判断 | **A+** | 正确识别了 v1 的根本缺陷 |
| 组件选型方向 | B- | 每个方向都对，但具体选错 |
| 文档精度 | C | 缺乏每条数字的原始出处 |
| 工程可行性 | C+ | 所有组件都有开源实现，但工程参数不对 |
| 创新度 | C | 组装已有 SOTA，无独立贡献 |

---

## 四、可借鉴的核心洞察

虽然 v2 的组件选型错误，但它确立的几个核心洞察成为后续迭代的基石：

1. **PRM 是 R_consistency 的唯一可信替代**（即使 v2 用错了 min 聚合）
2. **Engine-Native Verification 优于 LLM 评判**（Python REPL vs LLM-as-judge）
3. **搜索优于启发式迭代**（Best-of-N vs 连续 2 轮无改善）
4. **单模型 + 文本标签优于三模型 + 隐藏态切分**（这是对 v1 的根本否定）
5. **RL 算法稳定性是工程问题**（不是算法问题，Dr.GRPO + KL-Cov 是修复路径）

---

## 五、对后续迭代的影响

v2 的失败直接催生了 v3 的 SOTA 精度对齐方案：
- min(PRM) → 改用 product / last-step
- 标准 GRPO → 改用 Dr.GRPO + KL-Cov
- R1-Zero 纯 RL → 改用 Distill-First + RL-Aug
- 通用 Tool-Use → 限定在"单轮 + 可验证答案 + 算术/形式化"

但 v3 在执行 SOTA 精度对齐时，又引入了 6 项新的事实性错误（Dr.GRPO 无 benchmark 对比表、KL-Cov 在 3B AIME 下降等）。

---

## 六、引用

1. Wang et al., "Math-Shepherd: Verify and Reinforce LLMs Step-by-step" (arXiv 2312.08935)
2. Zhang et al., "The Lessons of Developing Process Reward Models" (ACL 2025 Findings, arXiv 2501.07301)
3. Lightman et al., "Let's Verify Step by Step" (arXiv 2305.20050)
4. Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (arXiv 2503.20783)
5. DeepSeek-R1 (Nature 2025, arXiv 2501.12948)
6. Guan et al., "rStar-Math" (ICML 2025, arXiv 2501.04519)
7. Gou et al., "ToRA" (ICLR 2024, arXiv 2309.17452)
8. EST-PRM (arXiv 2606.00437) — PRM 攻击评估

---

## 七、一句话评价

**v2 是正确的范式修正，但每个核心组件都"选对方向、选错实现"——min(PRM) 被证伪、标准 GRPO 必然 collapse、R1-Zero 在 7B 以下不可复现、Tool-Use 适用面被高估、MCTS 边际递减。** 修复 P0 两个问题（PRM 聚合 + RL 算法）后，v2 才真正站得住脚。
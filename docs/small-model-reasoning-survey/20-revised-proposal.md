# 1B 小模型推理能力建设 - 修正后的技术方向与 SOTA 路径

> **本文档用途**:基于综合调研报告(`00-comprehensive-review-report.md`)对用户原始提议(`10-user-original-proposal.md`)进行修正后的最终推荐
> **修正依据**:6 个并行深度调研 agent 的发现,详见 `01-06` 调研底稿
> **核心差异**:用户提议中的"CRV+循环深度模型"方案被拒绝,推荐替代为"Retrofit 1.5B 为 depth-recurrent + 显式 CoT SFT";用户提议中的"Cog-Tina-RAG-Loop"组合多个组件需要替换

---

## 📋 目录

1. [修正后的 5 维度技术方向](#一-修正后的-5-维度技术方向)
2. [修正后的"1B SOTA 黄金组合"](#二-修正后的-1b-sota-黄金组合)
3. [修正后的"CRV+循环深度模型"路径](#三-修正后的-crv循环深度模型路径)
4. [5 步实施路线图](#四-5-步实施路线图)
5. [必须拒绝的 5 件事](#五-必须拒绝的-5-件事)
6. [核心论文/仓库速查](#六-核心论文仓库速查)

---

## 一、 修正后的 5 维度技术方向

> 在用户原始 25+ 方向的基础上,基于 1B 实证证据进行筛选和修正

### 一、数据工程(7 个方向 → 4 个推荐)

| 用户原版 | 修正后评级 | 修正说明 |
|---|---|---|
| **认知对齐数据合成(CogPO, CRV)** | ⚠️ **谨慎使用,先验证** | CogPO 真实但**仅 ≥3B 验证**,DistilQwen2.5-R1-7B 用 105K 数据跑赢 800K(6.1× 数据效率)。**1B 是赌博,论文明确说"improve for much smaller models"是 future work**。建议先用 DistilQwen2.5-R1 路线建立基线 |
| **正负路径对比学习(MRPV)** | ✅ **可作为数据后处理** | ReaLM 框架(arXiv:2508.12387)中的核心组件。配套 InfiR-1B 在 1B 上提升 2.26×。需要 prompt 嵌入多个 CoT,工程上较复杂 |
| **自我增强与自举(Self-AMPLIFY)** | ❌ **不推荐作为主线** | 关键警告:无 grounding 自对弈在 <3B 普遍 plateau/collapse。TinyRLM-135M 是少数成功案例 |
| **结构化合成数据(TinyStories)** | ❌ **不推荐** | TinyStories(ICLR 2024)针对 5M 极小模型的**语言能力**,不解决数学/代码推理 |
| **过程奖励建模数据(PRM)** | ✅ **强烈推荐** | **rStar-Math 在 Qwen2.5-Math-1.5B 上 MATH 51.2% → 87.8% (+36.6)**。Microsoft 完全开源。**最强的 1B 验证数据技术** |
| **(新增)R1-Distill 路线** | ✅ **基线必选** | **DeepSeek-R1-Distill-Qwen-1.5B MATH-500 83.9%,AIME2024 28.9%**。**1.5B 跑赢 GPT-4o/Claude-3.5-Sonnet 多个数学基准** |
| **(新增)OpenThoughts/OpenR1 数据集** | ✅ **推荐** | 190+ 公开模型基于这些数据集训练,包含 1B-2B 模型。**OpenR1 完整复现 DeepSeek-R1-Distill-Qwen-7B** |

**修正后数据工程 Top 3**:
1. 🥇 **R1-Distill-Qwen-1.5B 蒸馏路线**(MATH-500 83.9% 标尺) — 证据最强、成本最低
2. 🥈 **rStar-Math 风格 MCTS 自进化 PRM** — 1.5B 上单步 +36.6,完全开源
3. 🥉 **CogPO/CRV 风格数据后处理** — 数据效率 6.1×,但需 ≥3B 验证

### 二、训练策略(9 个方向 → 3 个推荐)

| 用户原版 | 修正后评级 | 修正说明 |
|---|---|---|
| **Tina (LoRA-RL)** | ✅ **极低成本首选** | **真实(arXiv:2504.15777)**,AIME24 43.33%,$9 成本。**关键反直觉发现**:LoRA 计算量增加**反向**降低性能。**仅在 R1-Distill 起点有效** |
| **LoRA-RL 通用** | ✅ **DoRA > LoRA** | DoRA 46.6% > LoRA 42.5% > PiSSA 失败。RL 微调天然稀疏(5-30% 参数) |
| **DPO/KTO 用于 SLM** | ❌ **不推荐作为推理方法** | **关键反例**:oxRL 论文 1.5B 上 SFT 54.36% > DPO 49.08% > SimPO 38.67%。**算法排名在规模间反转** |
| **GRPO** | ✅ **1.5B 首选** | TinyZero 明确报告:**0.5B 完全失败,1.5B-3B+ 才有效**。需要 R1-Distill 起点 |
| **RED(回忆-扩展)** | 🟡 可选 | 2025-08 论文,MATH500 65.5% pass@1,专门为 1.5B SLM 设计。但**未开源** |
| **课程 RL** | ✅ **强烈推荐** | **FastCuRL(EMNLP 2025)AIME24 49.6%**;**DeepScaleR(Berkeley)AIME24 43.1%**;**OpenRS-Star(Qwen3-1.7B)AIME24 50% @ <$100** |
| **混合蒸馏(CoT+PoT)** | ❌ 不作独立方法 | 实质是 R1-Distill 的 CoT 蒸馏 |
| **RLOO / REINFORCE++ / CISPO** | 🟡 GRPO 替代品 | Magistral 验证 1.5B 可用,但 reasoning 收益不比 GRPO 强 |
| **(新增)TinyZero/Open-R1 开源实现** | ✅ **工程起点** | TinyZero 0.5B 失败,1.5B-3B+ 有效。**Open-Reasoner-Zero** 是唯一 0.5B 公开训练 |

**修正后训练策略 Top 3**:
1. 🥇 **GRPO + R1-Distill 起点 + 课程上下文(DeepScaleR/FastCuRL 模式)** — 1.5B AIME24 49.6% SOTA
2. 🥈 **LUFFY(off-policy 混合 GRPO)** — 1.5B +8 分
3. 🥉 **Tina(LoRA-GRPO)** — $9 极低成本

**致命警告**:**0.5B 模型在 GRPO 上系统失败**,目标 1B 是合理底线。

### 三、架构创新(10 个方向 → 3 个推荐)

| 用户原版 | 修正后评级 | 修正说明 |
|---|---|---|
| **Universal Transformer** | ❌ 历史价值 | 原始 UT(2018)未扩展到 LLM 尺度。已被 Ouro 超越 |
| **mHC(Hyper-Connections)** | ⚠️ **谨慎使用** | 真实(arXiv:2512.24880)DeepSeek 2025-12 发布。**主实验在 27B MoE,1B 公开对比缺失**。建议用 **mHC-lite 简化版**测试 |
| **DeltaNet / Gated DeltaNet** | 🟡 **hybrid 1B 已验证** | 1.3B 验证:Gated DeltaNet 在 S-NIAH-2 92.2 vs Mamba2 56.2。**纯 DeltaNet 不够,必须 hybrid** |
| **Mamba / Mamba-2** | ❌ **推理落后 20-30 pt** | **纯 Mamba-2-1.5B 在 GSM8K 上明显弱于 Transformer**。Falcon Mamba-7B 41.3% vs LLaMA 8B 75.2% |
| **RWKV-7("Goose")** | ❌ 不推荐 | 2.9B Goose ARC-c 49.5 vs Llama-3.2-3B 59.1,**英文推理明显落后** |
| **MoE Upcycling** | 🟡 **对纯 reasoning 增益有限** | 50% 成本,知识任务有效,但**MoE 增加总参数但 task loss 反而上升**(U-shaped 曲线,arXiv 2508.18672) |
| **Marco-Mini** | 🟡 **多语种强,纯推理一般** | Marco-Mini-Instruct(0.86B active)与 Qwen3-4B-Instruct 持平甚至更优,训练 FLOPs 仅 1/5.5。**但纯推理与 4B dense 持平** |
| **Ouro / Looped Transformer** | ✅ **强烈推荐(retrofit 路径)** | **Ouro-1.4B R4:GSM8K 78.92 vs Qwen3-1.7B 60.73 (+18.2 pt)**,MATH500 82.40 vs 17.60。**有 1.4B/2.6B 开源** |
| **Neuro-Symbolic(PAL/SymCode/Phi-GSM)** | ✅ **强烈推荐(零改动)** | **TinyGSM 1.3B → 81.5% GSM8K(rival GPT-3.5)**;**PAL 比 PaLM-540B + CoT 高 15 pt** |
| **TTT(测试时训练)** | ❌ 不推荐 | 内存 I/O 瓶颈严重,**无 production-grade 实现** |

**修正后架构创新 Top 3**:
1. 🥇 **Ouro / Looped Transformer(retrofit 路径,1.5B → GSM8K 49.9% 验证)**
2. 🥈 **Neuro-Symbolic(PAL/Phi-GSM 1.3B 81.5% GSM8K)**
3. 🥉 **MoE Upcycling + Drop(直接复用 1B 成本)**

### 四、推理时扩展(9 个方向 → 3 个推荐 + 6 个避免)

**🚨 关键背景:Kinetics 扩展定律(arXiv:2506.05333)**:
> "**1B 模型从 TTS 获得的相对增益(+154.6%)远大于 32B(+10.0%)**" — 小模型反而更需要、也更能从 TTS 中获益

| 用户原版 | 修正后评级 | 修正说明 |
|---|---|---|
| **Self-RAG / Adaptive-RAG** | ❌ **1B 失败** | 原始 Self-RAG **仅在 7B+ 验证**;**<3B 反思 token 训练失败**(退化为随机预测)。**替代方案**:Pleias-RAG-1B(arXiv:2504.18225)或 SeaKR(ACL 2025) |
| **CA-TTS / DeepConf** | ✅ **1.5B 强验证** | **Self-Calibration 论文:DeepSeek-R1-Distill-1.5B ARC-Challenge 58.9% → 66.5%(+7.6,仅 16 样本)** |
| **R-Stitch / Speculative Decoding** | ⚠️ **1B 作为目标可能变慢** | 1.5B 草稿 → 32B 目标:1.4-4.9× 加速。**1B 作为 target 可能变慢**(SmolLM2-1.7B + Kangaroo 1.16×,**Llama-3.2-1B + Kangaroo 0.91× 变慢**)。**1B 必须用 EAGLE-3** |
| **DES(动态专家搜索)** | ❌ **仅 MoE 适用** | 实验用 Qwen3-30B-A3B;dense 1B 完全不适用 |
| **ReAct / Agent Loop / Tool Use SFT** | ✅ **1B 强验证** | **qwen3:1.7b = 0.960 Agent Score(所有任务对)**,qwen3:0.6b = 0.880。**AAAI 2026 论文:350M OPT 单 epoch SFT 在 ToolBench 77.55%,超 175B ChatGPT 26%** |
| **Best-of-N / Self-Consistency** | ✅ **1B 模型最佳单点** | **Red Hat 论文(2025-02):Llama-3.2-1B PF 26.8 → 59.6 on MATH-500**;Qwen2.5-Math-1.5B 70.0 → 85.4 → GPT-4o 水平。**理论保证:BoN Θ(1/Δ) vs SC Θ(1/Δ²)** |
| **CoVe(链式验证)** | ❌ **1B 失效** | 仅 Llama 65B 验证;需要"自质疑"能力,1B 缺乏 |
| **ToT / GoT / AoT** | ❌ **严重失败** | **ToT 论文:GPT-4 74% vs GPT-3.5 仅 19%**;**生成器是瓶颈,1B 无法做生成器** |
| **s1 / s1.1 / Budget Forcing** | ❌ **重大陷阱** | **Long CoT Degradation(EMNLP 2025):Qwen2.5-0.5B 14% → 11%(-3 分)**,Gemma-3-1B 24% → 15%(-9 分)。**Wait #4 之后模型在答案间震荡** |
| **(新增)EAGLE-3** | ✅ **1B 唯一有效的 SD 方法** | **EAGLE-3 batch=1 1.4-6.5× 加速**;**batch=64 仍 1.38×**;**严格无损** |
| **(新增)Pleias-RAG-1B** | ✅ **Self-RAG 替代** | SOTA on 2WikiMultiHopQA, sub-1B。规避了反思 token 训练失败问题 |

**修正后推理时扩展 Top 3**:
1. 🥇 **Best-of-N + 小型奖励模型(1B > GPT-4o)**
2. 🥈 **DeepConf / Self-Calibration(-50% token 同精度)**
3. 🥉 **Agent Loop + Tool Use SFT(0.4 → 0.9 Agent Score)**

**应避免的技术**:
- 原始 Self-RAG(< 3B 反思 token 训练失败)
- ToT/GoT/AoT(生成器瓶颈)
- CoVe(自质疑)
- Long CoT SFT 蒸馏 R1 风格(永久性 -75% 退化)
- 独立 draft-model SD(1B 可能变慢)

### 五、系统与部署(8 个方向 → 3 个推荐)

| 用户原版 | 修正后评级 | 修正说明 |
|---|---|---|
| **AWQ/SmoothQuant/GPTQ** | ⚠️ AWQ 最佳,**0.5B 不可用 INT4** | **DeepSeek-R1-Distill-Qwen-1.5B AWQ W4 -1.36% vs GPTQ W4 -2.13%**;**W3 时崩溃**(-16.58%) |
| **INT4/FP8/NVFP4** | ✅ **FP8 推荐** | **Llama-3.2-1B FP8 MMLU 46.3% → 45.5%(仅 -0.8%)**。**NVFP4 PTQ 在小模型上不稳定** |
| **PowerInfer-2 / 异构调度** | ✅ 边缘部署 | 手机 NPU 24-27× 加速 vs llama.cpp |
| **PagedAttention / vLLM** | ✅ **强烈推荐** | vs HF Transformers **24× 吞吐**。vLLM 0.7+ 原生 EAGLE-3、FP8 KV cache |
| **SnapKV / H2O** | ✅ **长 CoT 首选** | **SnapKV 16K:3.6× 速度,8.2× 内存效率**。"Hold Onto That Thought"(2025-12):**SnapKV-D 和 H2O 在 reasoning 上最优** |
| **TensorRT-LLM / llama.cpp** | ✅ 跨平台 | TensorRT-LLM B200 10× A100;llama.cpp 1B Q4_K_M 1.7GB 流畅 |
| **Speculative Decoding (EAGLE-3)** | ✅ **1B 唯一有效** | **EAGLE-3 batch=1 1.4-6.5× 加速**;**严格无损** |
| **QAT (Quantization-Aware Training)** | ✅ **推理模型必备** | **Reasoning-QAT:DeepSeek-R1-Qwen-1.5B W2 GPTQ 3.67% → QAT 55%**(+51.33) |

**修正后系统 Top 3**:
1. 🥇 **FP8 + vLLM(质量几乎无损 + 1.5-2× 吞吐)**
2. 🥈 **EAGLE-3 自投机解码(1.4-6.5× batch=1)**
3. 🥉 **PagedAttention + SnapKV(32K+ CoT 8.2× 内存效率)**

---

## 二、修正后的"1B SOTA 黄金组合"

> **用户原版"Cog-Tina-RAG-Loop"中多个组件需要替换**
> **修正版本基于 1B 实证证据重新设计**

### 🏆 修正后推荐组合

| 层级 | 修正后选型 | 用户原版 | 修正理由 |
|---|---|---|---|
| **数据层** | **R1-Distill 路线 + OpenR1 精炼** | CRV + CogPO | 1B 验证充分,数据效率高。CogPO 在 1B 是赌博(论文未验证) |
| **训练层** | **GRPO + 课程(DeepScaleR/FastCuRL 模式)** | Tina 三阶段 | AIME24 49.6% SOTA,1.5B 充分验证。Tina 单阶段已足够 |
| **架构层** | **Ouro 1.4B retrofit 或 Neuro-Symbolic(PAL)** | mHC | Ouro 有 1.4B 官方 checkpoint,GSM8K 78.92;mHC 1B 验证缺失。**Neuro-Symbolic 1.3B → 81.5% GSM8K 是零架构改动的最简方案** |
| **推理层** | **BoN + 1.5B PRM + Self-Calibration + DeepConf + Tool Use SFT** | Self-RAG + CA-TTS | Self-RAG 1B 训练失败,改用 BoN+PRM+Tool Use。Llama-3.2-1B+PF 26.8 → 59.6 on MATH-500 |
| **系统层** | **FP8 + vLLM + EAGLE-3 + SnapKV** | AWQ INT4 + vLLM | FP8 质量损失更小(-0.6% MMLU),EAGLE-3 1.4-6.5× 加速 |

### 🔍 协同效应(修正后)

1. **数据→训练协同**:R1-Distill 数据直接接 FastCuRL 训练
2. **训练→架构协同**:Ouro-1.4B 已有 1.4B 官方权重,直接做 SFT
3. **推理→系统协同**:DeepConf 置信度可在 vLLM 原生使用,FP8 KV cache 与 SnapKV 兼容
4. **端到端闭环**:CA-TTS 低置信度 → BoN 重采样 → Tool Use fallback

### ⚠️ 5 步实施路线图(修正后)

| 阶段 | 周期 | 内容 | 预算 | 预期收益 |
|---|---|---|---|---|
| **第 0 步:基线建立** | 第 1-2 周 | 下载 DeepSeek-R1-Distill-Qwen-1.5B,评测 GSM8K/MATH-500/AIME24/ARC-C/BBH | $0 | 建立 MATH-500 83.9% 标尺 |
| **第 1 步:数据精炼** | 第 2-4 周 | OpenR1/Mixture-of-Thoughts 350k SFT(8×H100 数小时) | ~$50 | MATH-500 83.9% → 86-88% |
| **第 2 步:GRPO + 课程** | 第 4-7 周 | FastCuRL 风格(8K→16K→24K) + GRPO + rule-based reward | $100-500 | AIME24 28.9% → 45-50% |
| **第 3 步:推理时扩展** | 第 7-9 周 | BoN + Qwen2.5-Math-PRM-1.5B + Self-Calibration + DeepConf + Tool Use SFT | $50-200 | 在算力相同情况下 +5-15 分 |
| **第 4 步:架构探索(可选)** | 第 9-14 周 | Ouro-1.4B retrofit 或 Neuro-Symbolic(PAL) | $200-1000 | GSM8K 突破 75-80% |
| **第 5 步:系统部署** | 第 14 周起 | FP8 + vLLM + EAGLE-3 + SnapKV | $0-50 | batch=1 1.4-6.5× 加速 |

### 💡 何时替换组件?(修正后)

- **若模型为 MoE 架构**:用 Marco-Mini 路线或 DES
- **若任务强依赖工具调用**:用 qwen3:1.7b 路线(Agent Score 0.960)
- **若端侧算力极度受限(<5 TOPS)**:用 llama.cpp + PowerInfer-2
- **若需处理超长上下文(>32K)**:用 SnapKV-Decoding + FP8 KV cache

---

## 三、修正后的"CRV+循环深度模型"路径

> **用户原版"CRV+循环深度模型"方案被拒绝**
> **修正版本基于独立核查报告(`05-loop-model-deepdive.md`)提出可执行的替代路径**

### ❌ 用户原版方案的问题

1. **混淆两个完全不同的 "CRV"**:
   - **Meta CRV**(arXiv:2510.09312,ICLR 2026)= 白盒 interpretability 工具
   - **阿里 CRV**(arXiv:2504.09802,EMNLP 2025)= 数据生成 pipeline
   - **这两个是完全不同的方法**!

2. **Ouro 1.4B 不能直接描述为 "1B 路径"**:
   - 最小模型是 1.4B(不是 1B)
   - 需要 7.7T tokens 训练(ByteDance 级别资源)

3. **"Implicit latent reasoning 在 1B 工作"无实证支持**:
   - Coconut 8B 退化(LT-Tuning 论文证实)
   - 隐式推理常走捷径(Shortcut paper)
   - "implicit CoT cannot substitute explicit CoT"(arXiv:2411.15862)

4. **"7.7T tokens 是 implicit reasoning 必需"是错误的**:
   - 这是 Ouro 训练选择
   - Saunshi 1B 验证只需 250B tokens
   - **mcleish7 retrofit 路径只需 50B tokens**

### ✅ 修正后的可执行路径

#### 路径 1:Retrofit(最低成本,实证最强)

```python
# 路径 1:Retrofit(最低成本,实证最强)
base_model = "meta-llama/Llama-3.2-1B"  # 或 Qwen2.5-1.5B
# 用 mcleish7/retrofitting-recurrence 方案 retrofit 为 depth-recurrent
# 50B tokens continued pretraining → GSM8K 49.9%
```

**来源**:
- mcleish7/retrofitting-recurrence(GitHub)
- HF smcleish/Recurrent-TinyLlama-3T-train-recurrence-32
- 论文:Teaching Pretrained LLMs to Think Deeper with Retrofitted Recurrence(arXiv:2511.07384)

#### 路径 2:Saunshi 风格(Google ICLR 2025)

```python
# 路径 2:Saunshi 风格
# 1B / 250B tokens, 12层×2循环 → Math Word +5 分超越 24层 baseline
```

**来源**:Saunshi et al., "Reasoning with Latent Thoughts"(arXiv:2502.17416)

#### 路径 3:走 explicit CoT(不推荐 implicit)

```python
# 路径 3:走 explicit CoT(不推荐 implicit)
# 已被多项研究证明在 1B 上更稳定
# R1-Distill + SFT + GRPO + Best-of-N 路径
```

#### 数据构造策略(修正后)

**CRV 数据用于 looped 模型的最佳实践**:

1. **预训练阶段**:不需要"无中间步骤"的海量语料
   - mcleish7 retrofit 方案是 **继续训练(continued pretraining)**,不是 from-scratch
   - 用 50B tokens 即可达到 GSM8K 49.9%

2. **后训练/微调阶段**:**不要用 RELAY 生成的种子**
   - RELAY 仅在 3 个合成任务(Arithmetic、Edit Distance、LIS)上验证
   - **无 LLM-scale 验证** — 不是"高质量 CoT 种子"的可靠来源

3. **CRV 多智能体系统**:**阿里 CRV 可用,但仅 ≥3B 验证**
   - 1B 建议用 R1-Distill + SFT 路径
   - 1B CRV 是赌博

4. **Meta CRV interpretability**:**不用作数据过滤工具**
   - 是诊断工具,不是数据过滤
   - 不适用于大规模数据过滤

### 修正后路径评级

| 路径 | 推荐度 | 实施成本 | 收益量级 | 风险 |
|---|---|---|---|---|
| **Retrofit Llama-3.2-1B** | ⭐⭐⭐⭐⭐ | 50B tokens,1-2 周 | GSM8K 49.9% 实证 | 已被 mcleish7 验证 |
| **Ouro-1.4B 直接下载** | ⭐⭐⭐⭐ | $0(开源) | GSM8K 78.92% | 1.4B 不是 1B |
| **Saunshi 1B 250B tokens** | ⭐⭐⭐ | 250B tokens,数周 | Math Word +5 分 | 训练成本高 |
| **Ouro 2.6B from-scratch** | ⭐⭐ | 7.7T tokens,数月 | GSM8K 81.58% | **资源不可承受** |
| **CRV + LoopLM 组合** | ❌ | - | - | **未经验证,概念混杂** |
| **Implicit CoT 训练** | ❌ | - | - | **8B 退化,多反例** |

---

## 四、5 步实施路线图(综合)

> **整合数据/训练/架构/推理时/系统 5 个维度,基于 1B 实证证据的完整 14 周实施计划**

### 阶段 0:基线建立(第 1-2 周)

```
目标:用最低成本建立可信的推理基线

动作:
1. 下载 DeepSeek-R1-Distill-Qwen-1.5B
   (理由:800K CoT 蒸馏数据,1.5B 已有"in-context reasoning prior",起点比 Qwen2.5-1.5B-Base 节省整个蒸馏阶段)

2. 评测基线:
   - GSM8K(预期 ~82-85%)
   - MATH-500(预期 ~83-85%)
   - AIME24(预期 ~28-30%)
   - ARC-C、BBH、MMLU

3. 部署框架:选 vLLM 0.7+ (EAGLE-3 原生)
```

**预期基线**:MATH-500 83.9%、GSM8K 82-85%

### 阶段 1:数据精炼(第 2-4 周,~$50)

```
技术栈:
1. OpenR1 / Mixture-of-Thoughts 350k SFT(8×H100 数小时)
2. 或:DistilQwen2.5-R1-3B 风格 CRV(用 ≥7B 教师)→ 1.5B 蒸馏(小规模试点)
3. 可选:Tina LoRA-GRPO(arXiv:2504.15777,$9,2×L40S)— 起点必须是 R1-Distill

避免:
- CogPO 在 1.5B 直接用(论文未验证)
- 800K 直接 R1 蒸馏(数据效率低)
- 无 grounding 自对弈(< 3B plateau)
```

**预期收益**:MATH-500 83.9% → 86-88%(OpenR1 baseline 即可)

### 阶段 2:GRPO + 课程 RL(第 4-7 周,$100-500)

```
技术栈:
1. FastCuRL 风格:课程上下文 8K → 16K → 24K
2. GRPO + rule-based reward(verl / OpenRLHF)
3. 推荐:DeepScaleR / FastCuRL / OpenRS-Star 复现

开源参照:
- DeepScaleR(Berkeley,2025-02):AIME24 43.1% @ $42
- FastCuRL(EMNLP 2025):AIME24 49.6% @ 8×H100
- OpenRS-Star(Qwen3-1.7B):AIME24 50% @ <$100
- LUFFY(off-policy hybrid):+8 分 vs pure RL

关键配置:
- 起点必须是 R1-Distill 类
- 必须先做 SFT(TinyZero 报告 0.5B 失败)
- 1.5B 是合理下限(0.5B 系统失败)
```

**预期收益**:AIME24 28.9% → 45-50%

### 阶段 3:推理时扩展集成(第 7-9 周)

```
技术栈(按优先级):
1. Best-of-N + Qwen2.5-Math-PRM-1.5B(Red Hat PF 论文风格)
   - Llama-3.2-1B PF: 26.8% → 59.6% on MATH-500
   - 必须配 PRM,不能纯 SC
2. Self-Calibration + DeepConf(Meta AI)
   - DeepSeek-R1-Distill-1.5B ARC 58.9% → 66.5% (+7.6,16 样本)
3. Tool Use SFT + ReAct
   - 选 qwen3:1.7b 路线(Agent Score 0.960)
   - 5-10 个高频工具(避免判断崩溃)
4. Pleias-RAG-1B + SeaKR(若需要 RAG)
   - 替代原始 Self-RAG(< 3B 反思训练失败)
5. vLLM + EAGLE-3(部署时)
   - batch=1 加速 1.4-6.5×,严格无损

避免:
- 原始 Self-RAG(< 3B 失败)
- ToT/GoT/AoT(生成器瓶颈)
- Long CoT SFT 蒸馏 R1 风格(永久性 -75% 退化)
- s1K 风格 1B SFT(Qwen2.5-0.5B 14% → 11%)
- Chain-of-Edits 替代 Long CoT(Llama-3.2-1B CoE 7.82% vs s1K-CoT 0.15%)
```

**预期收益**:在算力相同情况下 +5-15 分;或算力减半情况下持平

### 阶段 4:架构探索(第 9-14 周,可选)

```
如果 1.5B 路线遇到瓶颈,考虑:

A. Neuro-Symbolic(PAL/Phi-GSM 路线) ⭐⭐⭐⭐⭐ 优先
   - Phi-1.5 1.3B + 1.3B verifier → 81.5% GSM8K
   - 1 天集成,无需改动模型权重
   - 适用:任何可形式化任务

B. Ouro 路径(高投入,高回报) ⭐⭐⭐⭐
   - 直接下载 Ouro-1.4B R4 → GSM8K 78.92%(vs Qwen3-1.7B 60.73)
   - 或:retrofit 现有 1.5B 为 depth-recurrent(mcleish7 方案,50B tokens)
   - 注意:looping 改善的是"知识操控",不增加知识容量

C. MoE Upcycling(中等投入,适用多语种) ⭐⭐⭐
   - Qwen3-0.6B → Marco-Nano(0.6B active / 8B total)
   - 或:Drop-Upcycling 5.9B active 匹配 13B dense
   - **对纯 reasoning 增益有限**

不要做的事:
  - 单纯 mHC 改造(1B 验证缺失)
  - 纯 Mamba/RWKV(推理落后 20-30 pt)
  - TTT 层(无 production-grade 实现)
  - 7.7T tokens from-scratch looped training(超出 1B 团队能力)
  - CRV + LoopLM 组合(无任何工作验证)
```

### 阶段 5:系统部署与持续优化(第 14 周起)

```
部署栈:
- 量化:FP8(H100)+ AWQ INT4(内存受限)
- 推理引擎:vLLM 0.7+(PagedAttention + 连续批处理)
- 加速:EAGLE-3(自投机解码,1.4-6.5× batch=1)
- 长上下文:SnapKV-Decoding(32K+ CoT 8.2× 内存效率)
- 保护:QAT(若用 INT4/3,Reasoning-QAT 恢复 +50%)

关键监控:
- GSM8K/MATH-500/AIME24 准确率
- P99 延迟(token throughput)
- 拒答率(confidence threshold)
- 量化精度损失(reasoning-specific benchmark)
```

### 核心原则

1. **起点选 R1-Distill-Qwen-1.5B,不要从 base 开始**(TinyZero 报告 0.5B 失败)
2. **SFT → GRPO → 推理时扩展**(已成共识)
3. **3B 是规模反转的关键阈值**(0.5B 失败,1.5B 是合理下限)
4. **小模型更需要 TTS**(Kinetics 定律:1B +154.6% vs 32B +10.0%)
5. **避免常见陷阱**:Long CoT 蒸馏、原始 Self-RAG、纯 Mamba、ToT、s1K 风格
6. **架构创新是高风险,高回报选项**(Ouro 1.4B 是唯一有 1B 官方权重)
7. **数据效率比数据规模更重要**(DistilQwen2.5-R1-7B 105K vs R1-Distill-7B 800K)

---

## 五、必须拒绝的 5 件事

1. ❌ **CogPO/CRV 在 1B 上赌一把**(论文未验证 1B,只承诺 ≥3B)
2. ❌ **原始 Self-RAG 端到端训练**(< 3B 反思 token 训练失败)
3. ❌ **Long CoT R1-style 蒸馏 1B 模型**(EMNLP 2025 证实永久性 -75% 退化)
4. ❌ **ToT/GoT/AoT**(1B 生成器瓶颈,CoT prompt-level 都失败)
5. ❌ **CogPO + Looped Transformer 组合**(无任何工作做过)

---

## 六、核心论文/仓库速查

| 类别 | 必读 |
|---|---|
| **R1-Distill 路线** | DeepSeek-R1-Distill-Qwen-1.5B (HF), OpenR1 (HF/Mistral), Mixture-of-Thoughts 350k |
| **训练策略** | Tina (arXiv:2504.15777), FastCuRL (arXiv:2503.17287), DeepScaleR (OpenReview I6GzDCne7U), LUFFY (arXiv:2504.14945) |
| **PRM** | rStar-Math (github.com/microsoft/rStar), Math-Shepherd (arXiv:2312.08935) |
| **架构** | Ouro (arXiv:2510.25741, HF ByteDance/Ouro-1.4B), retrofitting-recurrence (mcleish7), mcleish7/HF smcleish/Recurrent-TinyLlama-3T-train-recurrence-32 |
| **推理时** | Best-of-N+PRM (Red Hat 2025-02), DeepConf (arXiv:2508.15260, HF facebookresearch/deepconf), Pleias-RAG-1B (arXiv:2504.18225) |
| **系统** | vLLM 0.7+ (EAGLE-3 原生), EAGLE-3 (arXiv:2503.01840, github.com/SafeAILab/EAGLE), Reasoning-QAT (github.com/yasu0001/ReasoningQAT) |
| **数据后处理** | DistilQwen2.5-R1 (alibaba-pai), EasyDistill (modelscope/easydistill) |

---

## 总结对比:用户原版 vs 修正版

| 维度 | 用户原版 | 修正版 |
|---|---|---|
| **数据层** | CRV + CogPO(1B 未验证) | R1-Distill + OpenR1(1B 验证充分) |
| **训练层** | Tina 三阶段(必须 R1-Distill 起点) | GRPO + 课程(DeepScaleR/FastCuRL)+ Tina 备选 |
| **架构层** | mHC(1B 验证缺失) | Ouro 1.4B retrofit(已验证)+ Neuro-Symbolic |
| **推理层** | Self-RAG(< 3B 失败)+ CA-TTS | BoN + PRM + Self-Calibration + DeepConf + Tool Use |
| **系统层** | AWQ INT4 + vLLM | FP8 + vLLM + EAGLE-3 + SnapKV |
| **CRV+循环深度模型** | 概念混杂,未经验证 | Retrofit 1.5B 为 depth-recurrent(50B tokens → 49.9% GSM8K) |

**修正版核心原则**:
- **数据效率 > 数据规模**(DistilQwen2.5-R1-7B 105K vs R1-Distill-7B 800K)
- **起点选择 > 后续优化**(选 R1-Distill 起点)
- **3B 是规模反转的关键阈值**(0.5B 失败,1.5B 是合理下限)
- **小模型更需要 TTS**(Kinetics 定律:1B +154.6% vs 32B +10.0%)
- **架构创新是高风险,高回报选项**(Ouro 1.4B 是唯一有 1B 官方权重)

---

**修正版报告时间**:2026-06-09
**配套文档**:
- `00-comprehensive-review-report.md` - 主报告
- `01-06` - 6 个调研底稿
- `10-user-original-proposal.md` - 用户原始提议
- `20-revised-proposal.md` - **本文档(修正版)**
- `05-loop-model-deepdive.md` - CRV+LoopLM 独立核查

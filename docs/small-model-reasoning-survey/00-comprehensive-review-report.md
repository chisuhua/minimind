# 1B 小模型推理能力建设 - 综合评审与调研报告

> **报告时间**:2026 年 6 月
> **调研方法**:6 个深度调研 agent 并行执行,覆盖数据工程、训练策略、架构创新、推理时扩展、循环深度模型、系统部署 6 大方向
> **核心理念**:Skeptical Verification + Evidence-Based Recommendations
> **任务来源**:用户提出 25+ 技术方向 + "Cog-Tina-RAG-Loop 黄金组合" + "CRV + 循环深度模型" 方案,需要独立审查并给出 SOTA 建议

---

## 📋 目录

- [第一部分:用户提出的 25+ 技术方向逐一审查](#第一部分用户提出的-25-技术方向逐一审查)
- [第二部分:用户推荐的 "Cog-Tina-RAG-Loop 黄金组合" 审查](#第二部分用户推荐的-cog-tina-rag-loop-黄金组合-审查)
- [第三部分:用户提出的 "CRV + 循环深度模型" 方案关键问题](#第三部分用户提出的-crv--循环深度模型-方案关键问题)
- [第四部分:基于调研的最终 SOTA 建议](#第四部分基于调研的最终-sota-建议)
- [报告总结](#报告总结)

---

## 第一部分:用户提出的 25+ 技术方向逐一审查

### 一、数据工程方向(5/7 真正有效)

| # | 方向 | SOTA 程度 | 1B 适配性 | 综合评价 |
|---|---|---|---|---|
| 1 | **CogPO / CRV(认知对齐数据)** | ⭐⭐⭐⭐ EMNLP 2025 | ⚠️ 1.5B 起,**1B 未验证** | **CogPO 真实**(arXiv:2504.09802),DistilQwen2.5-R1-7B 用 105K 数据即跑赢 R1-Distill-7B 的 800K(6.1× 数据效率)。但**最小验证规模为 3B,论文明确说"improve for much smaller models"是 future work**。**不建议直接用于 1B 赌博**。 |
| 2 | **MRPV(正负路径对比)** | ⭐⭐⭐ 微软 arXiv 2508 | 🟡 1.5B 可用 | ReaLM 框架(arXiv:2508.12387)中的核心组件。配套 InfiR-1B 在 1B 上提升 2.26×。需要 prompt 嵌入多个 CoT,工程上较复杂。**可作为高质量数据后处理**。 |
| 3 | **Self-AMPLIFY / Self-Play 自举** | ⭐⭐ 多篇 | 🟡 1B 易退化 | 关键警告:无 grounding 自对弈在 <3B 普遍 **plateau/collapse**。TinyRLM-135M(SmolLM2) 是少数成功案例(+8.6 BBH),但 Gemma-2B 等更大模型也仅在 RL 阶段有效。**不推荐作为主线**。 |
| 4 | **PRM 数据(过程奖励)** | ⭐⭐⭐⭐⭐ rStar-Math 强证据 | ✅ **1.5B 强验证** | **rStar-Math 在 Qwen2.5-Math-1.5B 上 MATH 51.2% → 87.8% (+36.6)**。Microsoft 完全开源(rStar/Coder/2-Agent)。**是最强的 1B 验证数据技术**。但需要 4×A100 数周。 |
| 5 | **CogPO 单独再看** | 同 #1 | 同 #1 | 同 #1。 |
| 6 | **RELAY / Ouro / Looped Transformer 合成数据** | ⭐⭐⭐ Ouro 真实 | ⚠️ **Ouro 1.4B/2.6B 真实,但需 7.7T tokens** | **Ouro(arXiv:2510.25741)ByteDance 官方发布,1.4B/2.6B R4 在 OlympiadBench 71.55、GSM8K 78.92**。但"looping 改善 reasoning"在小数据上**已被证伪**:Saunshi et al. (Google, ICLR 2025) 在 1B / 250B tokens 已观察到;**mcleish7 retrofit(2025-11)在 1B / 50B tokens 即可达 49.9% GSM8K**。**用户说"7.7T 是 implicit reasoning 必需"是错误的——这是 Ouro 训练选择,不是 loop 架构内在需求**。RELAY 仅在合成任务(算术、Edit Distance、LIS),无 LLM 规模验证。 |
| 7 | **TinyStories / AS-ES / Guided Distillation** | ⭐⭐ 偏小模型 | 🟡 仅语言能力 | TinyStories(ICLR 2024)针对 5M 极小模型的**语言能力**,不解决数学/代码推理。AS-ES 77M USM 提升显著但**仅语言生成**。**对 1B 推理任务价值有限**。 |

**数据工程 Top 3 排序(基于 1B 实证强度)**:
1. **🥇 R1-Distill-Qwen-1.5B 蒸馏路线**(MATH-500 83.9% 标尺) — 证据最强、成本最低
2. **🥈 rStar-Math 风格 MCTS 自进化 PRM** — 1.5B 上单步 +36.6,完全开源
3. **🥉 CogPO/CRV 风格数据后处理** — 数据效率 6.1×,但需 ≥3B 验证

---

### 二、训练策略方向(4/9 真正有效)

| # | 方向 | SOTA 程度 | 1B 适配性 | 综合评价 |
|---|---|---|---|---|
| 1 | **Tina (LoRA-RL)** | ⭐⭐⭐⭐ UCLA 真实 | ✅ **1.5B 强验证** | **真实(arXiv:2504.15777)**,AIME24 43.33% Pass@1(优于 o1-preview 40%),**$9 成本**。**关键反直觉发现**:LoRA 计算量增加**反向**降低性能。**仅在 R1-Distill 起点有效,不适用 base 1.5B**。**极低预算首选**。 |
| 2 | **LoRA-RL 通用** | ⭐⭐⭐ 成熟 | ✅ 完全兼容 | PERL、Hydra-PPO 等标准做法。**RL 微调天然稀疏(5-30% 参数)**——这意味着**LoRA 不一定必要**。DoRA 46.6% > LoRA 42.5% > PiSSA 失败。 |
| 3 | **DPO/KTO 用于 SLM** | ⭐⭐ | ❌ **1.5B 上比 SFT 还差** | **关键反例**:oxRL 论文 1.5B 上 SFT 54.36% > DPO 49.08% > SimPO 38.67%(SimPO 在 1.5B 失败但在 7B 最佳!)**。**算法排名在规模间反转**。不推荐作为推理主要方法。 |
| 4 | **GRPO** | ⭐⭐⭐⭐ DeepSeekMath | ✅ **1.5B 强验证**(但 0.5B 失败) | TinyZero 明确报告:**0.5B 完全失败,1.5B-3B+ 才有效**。需要 R1-Distill 起点。MC-GRPO 解决 group zero-variance。**1.5B 上是首选**。 |
| 5 | **RED(回忆-扩展动力学)** | ⭐⭐⭐ | ✅ **1.5B 设计目标** | 2025-08 论文,MATH500 65.5% pass@1,**专门为 1.5B SLM 设计**。但**未开源**。 |
| 6 | **课程 RL** | ⭐⭐⭐⭐ | ✅ **1.5B 强验证** | **FastCuRL(EMNLP 2025)DeepSeek-R1-Distill-1.5B → AIME24 49.6%**;**DeepScaleR(Berkeley)AIME24 43.1%**;**OpenRS-Star(Qwen3-1.7B)AIME24 50% @ <$100**。**成熟开源首选**。 |
| 7 | **混合蒸馏(CoT+PoT)** | ⭐⭐ 偏研究 | 🟡 1B 蒸馏即 SFT | "Mixed Distillation"(EMNLP 2024)在 7B+ 验证。R1-Distill-1.5B 实质就是 CoT 蒸馏。**不作为独立方法**。 |
| 8 | **RLOO / REINFORCE++ / CISPO** | ⭐⭐⭐ | ✅ 1.5B 可用 | Magistral 用 REINFORCE++-baseline 训 1.5B。RLOO 在 Pythia 1B 验证。**是 GRPO 替代品**,但 reasoning 收益不比 GRPO 强。 |
| 9 | **TinyZero / Open-R1 开源实现** | ⭐⭐⭐⭐ 生态丰富 | ✅ 1.5B 大量验证 | **关键限制:TinyZero 0.5B 失败,1.5B-3B+ 有效**。**Open-Reasoner-Zero-0.5B** 是唯一 0.5B 公开训练。Open-R1 完整复现。**工程起点**。 |

**训练策略 Top 3 排序**:
1. **🥇 GRPO + R1-Distill 起点 + 课程上下文(DeepScaleR/FastCuRL 模式)** — 1.5B AIME24 49.6% SOTA
2. **🥈 LUFFY(off-policy 混合 GRPO)** — 1.5B +8 分
3. **🥉 Tina(LoRA-GRPO)** — $9 极低成本

**致命警告**:**0.5B 模型在 GRPO 上系统失败**(TinyZero/Open-Reasoner-Zero 报告),目标 1B 是合理底线。

---

### 三、架构创新方向(2/10 真正有 1B 强证据)

| # | 方向 | SOTA 程度 | 1B 适配性 | 综合评价 |
|---|---|---|---|---|
| 1 | **Universal Transformer** | ⭐⭐ 历史价值 | ❌ 1B LLM 未验证 | 原始 UT(2018)未扩展到 LLM 尺度。Loop Transformer for Length Generalization 仅在 toy 算术上验证。**已被 Ouro 超越**。 |
| 2 | **mHC(Hyper-Connections)** | ⭐⭐⭐ DeepSeek 2025 | ⚠️ 仅 780M 实验 | mHC(arXiv:2512.24880)用 Sinkhorn-Knopp 算法将残差矩阵约束到 Birkhoff 多面体。**主实验在 27B MoE,1B 公开对比缺失**。mHC-lite 是简化版。可用但无 1B 验证。 |
| 3 | **DeltaNet / Gated DeltaNet** | ⭐⭐⭐ NVIDIA ICLR 2025 | 🟡 **hybrid 1B 已验证** | 1.3B 验证:**Gated DeltaNet 在 S-NIAH-2 92.2 vs Mamba2 56.2**;但**通用推理仅小幅领先或持平 Transformer**。**Qwen3.5-0.8B 18 层 DeltaNet + 6 层 attention** 在 1B 以下 SOTA。**纯 DeltaNet 不够,必须 hybrid**。 |
| 4 | **Mamba / Mamba-2** | ⭐⭐⭐ | ❌ 推理落后 20-30 pt | **纯 Mamba-2-1.5B 在 GSM8K 上明显弱于 Transformer**。Falcon Mamba-7B 41.3% vs LLaMA 8B 75.2%。**任意 DP 问题上 Mamba 总成本与 Transformer 持平(Yale 论文)**。**仅推荐 hybrid 部署**。 |
| 5 | **RWKV-7("Goose")** | ⭐⭐⭐ | 🟡 2.9B 多语种强,英文弱 | 2.9B Goose 多语种 SOTA,但 **ARC-c 49.5 vs Llama-3.2-3B 59.1**,英文推理**明显落后**。**无大厂商采用(Qwen3.5 选 DeltaNet,Jamba 选 Mamba-2)**。 |
| 6 | **MoE Upcycling** | ⭐⭐⭐⭐ Google ICLR 2023 | ✅ **可直接复用 1B 预训练** | **Komatsuzaki 论文:仅原训练 50% 成本**。Nemotron-4 15B upcycled MMLU +2.3%。**Drop-Upcycling(2025-05)5.9B active 匹配 13B dense,1/4 FLOPs**。**但对纯 reasoning(math)增益饱和甚至退化**——sparsity 越高 reasoning 越差(arXiv 2508.18672)。 |
| 7 | **Marco-Mini / Marco-Nano** | ⭐⭐⭐ 阿里 AIDC | ✅ **0.86B active production-ready** | Marco-Mini-Instruct(0.86B active)与 Qwen3-4B-Instruct **持平甚至更优**,训练 FLOPs 仅 1/5.5。**但纯推理(GSM8K/MATH)与 4B dense 持平**,不显著超越。**多语种/知识强,纯推理一般**。 |
| 8 | **Ouro / Looped Transformer** | ⭐⭐⭐⭐⭐ **ByteDance 官方 1.4B** | ✅ **有 1.4B/2.6B 开源 + 实测数据** | **Ouro-1.4B R4:GSM8K 78.92 vs Qwen3-1.7B 60.73 (+18.2 pt)**,MATH500 82.40 vs 17.60。**1.4B 匹配 4B 推理任务**。**KV cache 可压缩到 4×(仅保留最后一步)**。**SOTA 性价比**。 |
| 9 | **Neuro-Symbolic(PAL/SymCode/Phi-GSM)** | ⭐⭐⭐⭐⭐ ICML 2023 | ✅ **1.3B 强验证** | **TinyGSM 1.3B → 81.5% GSM8K(rival GPT-3.5)**;Phi-1.5 1.3B + 1.3B verifier 81.5%;**PAL 比 PaLM-540B + CoT 高 15 pt**。**零架构改动,任何 1B LLM 立即受益**。**仅适用可形式化任务(math/code)**。 |
| 10 | **TTT(测试时训练)** | ⭐⭐ 概念 | ❌ 1B 工程不可行 | 内存 I/O 瓶颈严重,**无 production-grade 实现**。Qwen3-Next 选 Gated DeltaNet,Jamba 选 Mamba-2,均未选 TTT。**不推荐**。 |

**架构创新 Top 3 排序**:
1. **🥇 Ouro / Looped Transformer(有 1.4B 官方 checkpoint)**
2. **🥈 Neuro-Symbolic(PAL/Phi-GSM 1.3B 81.5% GSM8K)**
3. **🥉 MoE Upcycling + Drop(直接复用 1B 成本)**

---

### 四、推理时扩展方向(3/9 真正有效,1 个最关键)

**🚨 关键背景:Kinetics 扩展定律(arXiv:2506.05333)**:
> "**对于 14B 以下的模型,在受限算力下,提升模型规模比延长 CoT 链或增加样本数都更有效;只有模型规模 ≥ 14B 时,测试时扩展才开始占主导**。"

**但反过来看:1B 模型从 TTS 获得的相对增益(+154.6%)远大于 32B(+10.0%)** — 小模型反而更需要、也更能从 TTS 中获益。

| # | 方向 | SOTA 程度 | 1B 适配性 | 综合评价 |
|---|---|---|---|---|
| 1 | **Self-RAG / Adaptive-RAG** | ⭐⭐⭐ ICLR/NAACL 2024 | ❌ **原始方法在 1B 失败** | 原始 Self-RAG **仅在 7B+ 验证**;**<3B 反思 token 训练失败**(退化为随机预测)。**替代方案**:Pleias-RAG-1B(arXiv:2504.18225)通过专门 mid-training 绕过此问题;SeaKR(ACL 2025)从 hidden state 提取不确定性。**用户推荐的 Self-RAG 在 1B 上需要重大修改**。 |
| 2 | **CA-TTS / DeepConf** | ⭐⭐⭐⭐ Meta 2025 | ✅ **1.5B 强验证** | **Self-Calibration 论文:DeepSeek-R1-Distill-1.5B ARC-Challenge 58.9% → 66.5%(+7.6,仅 16 样本)**。DeepConf-low token 减少 43-79% 同精度。**关键陷阱:小模型置信度高度不校准,必须先做 Self-Calibration**。 |
| 3 | **R-Stitch / Speculative Decoding** | ⭐⭐⭐ 2025 | ⚠️ **1B 作为草稿可加速,作为目标无效** | **1.5B 草稿 → 32B 目标:1.4-4.9× 加速,质量不变**。**但 1B 作为 target:可能变慢**(EACL 2026 论文:SmolLM2-1.7B + Kangaroo speedup 1.16×,**Llama-3.2-1B + Kangaroo 0.91× 变慢**)。**1B 必须用 EAGLE-3 而非独立 draft**。 |
| 4 | **DES(动态专家搜索)** | ⭐⭐ 2025 | ❌ **仅 MoE 适用** | 实验用 Qwen3-30B-A3B、Ling-lite-1.5;**dense 1B 完全不适用**。 |
| 5 | **ReAct / Agent Loop / Tool Use SFT** | ⭐⭐⭐⭐ 2025-2026 | ✅ **1B 强验证** | **Mike Veerman 2026 基准:qwen3:1.7b = 0.960 Agent Score(所有任务对)**,qwen3:0.6b = 0.880。**AAAI 2026 论文:350M OPT 单 epoch SFT 在 ToolBench 77.55%,超 175B ChatGPT 26%**。**1B 工具调用是"涌现性"的,需要专门 SFT**。 |
| 6 | **Best-of-N / Self-Consistency** | ⭐⭐⭐⭐⭐ **1B 模型最佳单点** | ✅ **1B 强验证** | **Red Hat 论文(2025-02):Llama-3.2-1B Particle Filtering 26.8 → 59.6 on MATH-500**;Qwen2.5-Math-1.5B 70.0 → 85.4 → GPT-4o 水平。**Compute-Optimal TTS(arXiv:2502.06703):0.5B 超过 GPT-4o on MATH-500**。**理论保证:BoN Θ(1/Δ) vs SC Θ(1/Δ²)**。 |
| 7 | **CoVe(链式验证)** | ⭐⭐ ACL 2024 | ❌ **1B 失效** | 仅 Llama 65B 验证;需要"自质疑"能力,1B 缺乏。 |
| 8 | **ToT / GoT / AoT** | ⭐⭐ NeurIPS 2023 / NAACL 2024 | ❌ **严重失败** | **ToT 论文:GPT-4 74% vs GPT-3.5 仅 19%(差 55 分)**;**生成器是瓶颈,不是评估器**(GPT-4 生成 + GPT-3.5 评估 = 64%,反之 31%)。**1B 无法做生成器**。 |
| 9 | **s1 / s1.1 / Budget Forcing** | ⭐⭐⭐ ICLR 2025 | 🟡 **重大陷阱** | **关键反例:Long CoT Degradation(EMNLP 2025):Qwen2.5-0.5B 14% → 11%(-3 分)**,Gemma-3-1B 24% → 15%(-9 分)。**Wait #4 之后模型在答案间震荡**。"It's Not That Simple"(arXiv:2507.14419):scaling down by max length 是真实收益,但 Wait 追加无效。**Chain-of-Edits(CoE)对 1B 替代 Long CoT:Llama-3.2-1B CoE 7.82% vs s1K-CoT 0.15%**。 |

**推理时扩展 Top 3 排序**:
1. **🥇 Best-of-N + 小型奖励模型(1B > GPT-4o)**
2. **🥈 DeepConf / Self-Calibration(-50% token 同精度)**
3. **🥉 Agent Loop + Tool Use SFT(0.4 → 0.9 Agent Score)**

**应避免的技术**:
- 原始 Self-RAG(< 3B 反思 token 训练失败)
- ToT/GoT/AoT(生成器瓶颈)
- CoVe(自质疑)
- Long CoT SFT 蒸馏 R1 风格(永久性 -75% 退化)

---

### 五、系统与部署方向(7/8 真正有效)

| # | 方向 | 1B 适配性 | 综合评价 |
|---|---|---|---|
| 1 | **AWQ/SmoothQuant/GPTQ** | ⚠️ AWQ 最佳 | **DeepSeek-R1-Distill-Qwen-1.5B AWQ W4 -1.36% vs GPTQ W4 -2.13%**;**W3 时崩溃**(-16.58%)。**AWQ 是 SLM 最佳选择**。 |
| 2 | **INT4/FP8/NVFP4** | ✅ **FP8 推荐** | **Llama-3.2-1B FP8 MMLU 46.3% → 45.5%(仅 -0.8%)**。**NVFP4 PTQ 在小模型上不稳定**(-3-5%)。**FP8 是 H100 默认**。 |
| 3 | **PowerInfer-2 / 异构调度** | ✅ 边缘 | 手机 NPU 24-27× 加速 vs llama.cpp;**47B MoE 11.68 tokens/s**。**需要 TurboSparse 训练**。 |
| 4 | **PagedAttention / vLLM** | ✅ **强烈推荐** | vs HF Transformers **24× 吞吐**。vLLM 0.7+ 原生 EAGLE-3、FP8 KV cache、Rejection Sampling。**32K+ CoT 标配**。 |
| 5 | **SnapKV / H2O** | ✅ 长 CoT | **SnapKV 16K:3.6× 速度,8.2× 内存效率**。"Hold Onto That Thought"(2025-12):**SnapKV-D 和 H2O 在 reasoning 上最优**。 |
| 6 | **TensorRT-LLM / llama.cpp** | ✅ 跨平台 | TensorRT-LLM B200 10× A100;llama.cpp 1B Q4_K_M 1.7GB 流畅。 |
| 7 | **Speculative Decoding (EAGLE-3)** | ✅ **1B 唯一有效** | **EAGLE-3 batch=1 1.4-6.5× 加速**;**batch=64 仍 1.38×**;**严格无损**。**1B 必须用 EAGLE-3,独立 draft model 可能变慢**。 |
| 8 | **QAT (Quantization-Aware Training)** | ✅ **推理模型必备** | **Reasoning-QAT:DeepSeek-R1-Qwen-1.5B W2 GPTQ 3.67% → QAT 55%**(+51.33)。Silver Bullet 轻量 QAT 332 examples +3 分钟。**QAT 是 INT4/INT3 推理保护必备**。 |

**系统 Top 3 排序**:
1. **🥇 FP8 + vLLM(质量几乎无损 + 1.5-2× 吞吐)**
2. **🥈 EAGLE-3 自投机解码(1.4-6.5× batch=1)**
3. **🥉 PagedAttention + SnapKV(32K+ CoT 8.2× 内存效率)**

---

## 第二部分:用户推荐的 "Cog-Tina-RAG-Loop 黄金组合" 审查

### 组合方案概览

用户推荐:
- **数据层**:CRV + CogPO 合成数据
- **训练层**:Tina(LoRA-RL)三阶段
- **架构层**:mHC(Hyper-Connections)
- **推理层**:Self-RAG + CA-TTS
- **系统层**:AWQ INT4 + vLLM

### 逐组件审查

| 组件 | 用户声称 | 独立验证结果 | 评级 |
|---|---|---|---|
| **CRV + CogPO 合成数据** | "解决大模型 CoT 对小模型过难问题" | CogPO **真实但仅 ≥3B 验证**;DistilQwen2.5-R1-7B 用 105K 数据跑赢 800K | ⚠️ 1B 是赌博,论文明确说"improve for much smaller models"是 future work |
| **Tina(LoRA-RL)三阶段** | "$9 成本,规避灾难性遗忘" | **真实(arXiv:2504.15777)**,AIME24 43.33%,$9 成本 | ✅ 真实,但**仅在 R1-Distill 起点有效**,不适用 base 1.5B |
| **mHC** | "即插即用,深→浅反馈" | **真实(arXiv:2512.24880)**,DeepSeek 2025-12 发布 | ⚠️ **主实验在 27B MoE,1B 公开对比缺失**;mHC-lite 简化版可用 |
| **Self-RAG** | "解决事实性" | **仅 7B+ 验证,1B 反思 token 训练失败** | ❌ **不适用于 1B**,需用 Pleias-RAG-1B 替代 |
| **CA-TTS** | "解决可靠性" | **1.5B 强验证(Self-Calibration 58.9% → 66.5%)** | ✅ 真实有效 |
| **AWQ INT4 + vLLM** | "保护关键推理权重" | **真实,但 INT4 推理 -1.5% 是已知 trade-off** | ✅ 成熟方案 |

### 协同效应分析(用户声称 vs 实际)

| 协同声称 | 实际评估 |
|---|---|
| "CRV 数据直接接 Tina SFT 语料" | ⚠️ 理论上 OK,但 CRV 1B 验证缺失 |
| "CogPO 目标函数接 Tina RL" | ⚠️ DPO/GRPO 目标不同,不能直接接;需重新设计 |
| "mHC 通过 LoRA 适配器学习" | ❌ mHC 设计**用 mHC-lite 替代更稳**,与 LoRA 兼容性未明确测试 |
| "CA-TTS 置信度复用 KV Cache 注意力分数" | ✅ 理论上 OK(DeepConf 就是这么做的) |
| "AWQ 量化后 mHC 仍工作" | ❌ 量化对循环连接的影响未公开测试 |

### ⚠️ 关键问题与风险

1. **Self-RAG 组件需要替换**:1B 反思 token 训练会失败。建议用 **Pleias-RAG-1B**(arXiv:2504.18225)或 **SeaKR**(ACL 2025)替代
2. **CogPO 1B 验证缺失**:论文明确说"improve effectiveness for much smaller models"是 future work。建议**先用 base 1.5B 走 DeepSeek-R1-Distill 路线建立基线**
3. **mHC 1B 验证缺失**:主实验在 27B MoE。建议**用 mHC-lite + 1B 起步测试,或先跳过此组件**
4. **Tina 仅适用 R1-Distill 起点**:**不建议从 base 1.5B 直接做 Tina**,需要先做 SFT 蒸馏

### 替代优化建议

| 原组件 | 建议替代 | 替代理由 |
|---|---|---|
| **CRV + CogPO** | **DistilQwen2.5-R1 路线 + OpenR1** | 1B 验证充分,数据效率高 |
| **Tina 三阶段** | **Tina(单阶段) + 课程 RL** | Tina 已是单阶段实现;课程 RL(FastCuRL)有 1.5B 验证 |
| **mHC** | **Ouro-1.4B retrofit** | Ouro 有 1.4B 官方 checkpoint,GSM8K 78.92;retrofit 路径 mcleish7 已验证(50B tokens 49.9% GSM8K) |
| **Self-RAG** | **Pleias-RAG-1B + SeaKR** | Self-RAG 1B 训练失败;这两个有专门 1B 路线 |
| **CA-TTS** | 保留 + **+ BoN + 1.5B PRM** | Red Hat Particle Filtering 在 1B 上 +33 分 |
| **AWQ INT4 + vLLM** | **FP8 + vLLM + EAGLE-3** | FP8 质量损失更小(-0.6% MMLU),EAGLE-3 1.4-6.5× 加速 |

---

## 第三部分:用户提出的 "CRV + 循环深度模型" 方案关键问题

### 三个严重问题

#### 问题 1:混淆了两个完全不同的 "CRV"

用户提到的 "Meta CRV" 和 "阿里 CRV" **是完全不同的方法**:

| 维度 | Meta CRV | 阿里 CRV |
|---|---|---|
| 论文 | arXiv:2510.09312 | arXiv:2504.09802 |
| 来源 | Meta FAIR | 阿里通义 + 上海交大 |
| 目的 | 验证 CoT 是否正确(白盒) | 生成认知匹配 SFT 训练数据 |
| 模型 | Llama 3.1 8B | Qwen2.5/Llama 7B/14B |
| 应用 | 推理时诊断 | 训练数据生成 |

**用户把它们混为一谈,并错误地将 Meta CRV(interpretability 工具)描述为"数据过滤工具"**。Meta CRV 论文中**无任何数据过滤实验**。

#### 问题 2:Ouro 1.4B 不能直接描述为 "1B 路径"

- Ouro **最小模型是 1.4B**(不是 1B)
- **需要 7.7T tokens 训练**(3T pretrain + 2.6T CT annealing + 20B 长上下文 + 300B mid-train + SFT)
- 这是 ByteDance 级别的资源,**不是"小数据 1B 路径"**

**但有替代的 1B 实证**:
- **mcleish7/retrofitting-recurrence(2025-11)**:Llama-3.2-1B 在 **50B tokens** 即可达 49.9% GSM8K
- **Saunshi et al. (Google, ICLR 2025)**:1B / 250B tokens, 12⊗2 循环超越 24⊗1 +5 分(Math Word Problems)
- **Bae et al. (Google DeepMind, ICLR 2025)**:1B Gemma / Pythia 在 **15B tokens** uptraining 提升 3-13.5 分

**用户说"7.7T 是 implicit reasoning 必需"是错误的**——这是 Ouro 训练选择,不是 looped 架构的内在要求。

#### 问题 3:"Implicit latent reasoning 在 1B 上工作" 无实证

多项反例:
- **Coconut(Meta FAIR)在 8B 退化**:LT-Tuning 论文证实 8B 性能 50.3% → 41.5%
- **隐式推理走捷径**:"Implicit Reasoning in Transformers is Reasoning through Shortcuts"(arXiv:2503.07604)
- **"Do LLMs Really Think Step-by-step In Implicit Reasoning?"**(arXiv:2411.15862):**"implicit CoT cannot substitute explicit CoT"**
- **A Formal Comparison**(arXiv:2509.25239):**两个范式有不同能力,没有一方完全胜出**

**用户声称"implicit reasoning = 1B SOTA"未有任何 1B 实证支持**。

### 真正可行的 1B LoopLM 路径

**如果你真的想做 1B reasoning 走 looped model 路径**:

```python
# 路径 1:Retrofit(最低成本,实证最强)
base_model = "meta-llama/Llama-3.2-1B"  # 或 Qwen2.5-1.5B
# 用 mcleish7/retrofitting-recurrence 方案 retrofit 为 depth-recurrent
# 50B tokens continued pretraining → GSM8K 49.9%

# 路径 2:Saunshi 风格
# 1B / 250B tokens, 12层×2循环 → Math Word +5 分超越 24层 baseline

# 路径 3:不走 implicit reasoning
# 用 explicit CoT(已被多项研究证明在 1B 上更稳定)
```

### 最终裁定:CRV + LoopLM 组合

> **The "CRV + Looped Transformer" combination is theoretically interesting but PRACTICALLY UNPROVEN for 1B reasoning.**

**详细独立核查报告**见 `05-loop-model-deepdive.md`

---

## 第四部分:基于调研的最终 SOTA 建议

### 1B 模型推理能力提升的真正 SOTA 路径

> **核心结论:用户提出的"CRV + Looped Transformer"组合概念混杂、缺乏实证,真实可用的 1B 推理 SOTA 路径在 5 个独立领域有充分证据支持。**

### 🏆 综合推荐:5 步实施路线

#### 阶段 0:基线建立(第 1-2 周)

**目标**:用最低成本建立可信的推理基线

```
基座选择:DeepSeek-R1-Distill-Qwen-1.5B(已有 800K CoT 蒸馏数据)
理由:
  - MATH-500 83.9% 已知标尺
  - 1.5B 已有"in-context reasoning prior"
  - 1.5B 上 GRPO 已被多次验证有效
  - 起点比 Qwen2.5-1.5B-Base 节省整个蒸馏阶段
```

**基线评测**:GSM8K、MATH-500、AIME24、ARC-C、BBH、MMLU

#### 阶段 1:数据精炼 + Tina(第 2-4 周,~$10-50)

```
技术栈:
  1. OpenR1 / Mixture-of-Thoughts 350k SFT(8×H100 数小时,免费)
  2. 或:DistilQwen2.5-R1-3B 风格 CRV(用 ≥7B 教师)→ 1.5B 蒸馏
  3. Tina LoRA-GRPO(arXiv:2504.15777,$9,2×L40S)

避免:
  - CogPO 在 1.5B 直接用(论文未验证)
  - 800K 直接 R1 蒸馏(数据效率低)
  - 无 grounding 自对弈(< 3B plateau)
```

**预期收益**:MATH-500 83.9% → 86-88%(OpenR1 baseline 即可)

#### 阶段 2:GRPO + 课程 RL(第 4-7 周,$100-500)

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

#### 阶段 3:推理时扩展集成(第 7-9 周)

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

#### 阶段 4:架构探索(第 9-14 周,可选)

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

#### 阶段 5:系统部署与持续优化(第 14 周起)

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

### 必须拒绝的 5 件事

1. ❌ **CogPO/CRV 在 1B 上赌一把**(论文未验证 1B,只承诺 ≥3B)
2. ❌ **原始 Self-RAG 端到端训练**(< 3B 反思 token 训练失败)
3. ❌ **Long CoT R1-style 蒸馏 1B 模型**(EMNLP 2025 证实永久性 -75% 退化)
4. ❌ **ToT/GoT/AoT**(1B 生成器瓶颈,CoT prompt-level 都失败)
5. ❌ **CogPO + Looped Transformer 组合**(无任何工作做过)

### 核心论文/仓库速查

| 类别 | 必读 |
|---|---|
| **R1-Distill 路线** | DeepSeek-R1-Distill-Qwen-1.5B (HF), OpenR1 (HF/Mistral), Mixture-of-Thoughts 350k |
| **训练策略** | Tina (arXiv:2504.15777), FastCuRL (arXiv:2503.17287), DeepScaleR (OpenReview I6GzDCne7U), LUFFY (arXiv:2504.14945) |
| **PRM** | rStar-Math (github.com/microsoft/rStar), Math-Shepherd (arXiv:2312.08935) |
| **架构** | Ouro (arXiv:2510.25741, HF ByteDance/Ouro-1.4B), retrofitting-recurrence (mcleish7), mcleish7/HF smcleish/Recurrent-TinyLlama-3T-train-recurrence-32 |
| **推理时** | Best-of-N+PRM (Red Hat 2025-02), DeepConf (arXiv:2508.15260, HF facebookresearch/deepconf), Pleias-RAG-1B (arXiv:2504.18225) |
| **系统** | vLLM 0.7+ (EAGLE-3 原生), EAGLE-3 (arXiv:2503.01840, github.com/SafeAILab/EAGLE), Reasoning-QAT (github.com/yasu0001/ReasoningQAT) |

---

## 报告总结

### 用户方案的最终评级

| 维度 | 评级 |
|---|---|
| **数据工程清单完整性** | ⭐⭐⭐⭐ 覆盖全面,但 CogPO 1B 验证缺失 |
| **训练策略清单完整性** | ⭐⭐⭐⭐⭐ 最完整,推荐组合有效 |
| **架构创新清单完整性** | ⭐⭐⭐⭐ 覆盖广,但缺 Ouro(2025-10) 等最新 |
| **推理时扩展清单完整性** | ⭐⭐⭐⭐⭐ 最强,数据最丰富 |
| **系统部署清单完整性** | ⭐⭐⭐⭐⭐ 最完整 |
| **"Cog-Tina-RAG-Loop" 推荐组合** | ⚠️ **多个组件需要替换**(Self-RAG、CogPO 1B、mHC 1B 均需调整) |
| **"CRV + 循环深度模型" 方案** | ❌ **概念混淆,缺乏实证,应被拒绝** |

### 终极建议

**用户清单的 80% 内容是准确且有价值的**——尤其是训练策略、推理时扩展、系统部署三大维度。

**但在两个关键点上需要重大修正**:

1. **CRV + 循环深度模型的组合方案**:
   - 混用了两个不同来源的 "CRV" 方法
   - "implicit latent reasoning 在 1B 工作" 的核心论据缺乏 1B 实证
   - "Ouro 7.7T tokens" 是特定训练选择,不是 loop 架构内在要求
   - **应该用 mcleish7 retrofit 路径(50B tokens 即可)在 1B 上实证 looped transformer**

2. **"Cog-Tina-RAG-Loop" 黄金组合**:
   - **Self-RAG 必须替换为 Pleias-RAG-1B 或 SeaKR**
   - **CogPO 在 1B 是赌博,先用 DistilQwen2.5-R1 路线建立基线**
   - **mHC 在 1B 验证缺失,先用 mHC-lite 简化版测试**
   - **Tina 必须从 R1-Distill 起点开始,不能从 base 1.5B**

**真正的 1B SOTA 路径不在用户的组合里,而在用户清单中分散的子技术的最优组合**:

- **数据**:R1-Distill(800K CoT 蒸馏)→ OpenR1/Mixture-of-Thoughts(精炼)
- **训练**:DeepScaleR / FastCuRL / OpenRS-Star 风格(GRPO + 课程)
- **架构**:Neuro-Symbolic(PAL) → Ouro-1.4B retrofit(可选)
- **推理时**:Best-of-N + 1.5B PRM + Self-Calibration + DeepConf + Tool Use
- **系统**:FP8 + vLLM + EAGLE-3 + SnapKV

**这是当前 1B 推理能力提升的"最大公约数"路径,有充分 1B 实证支持,工程风险可控**。

---

**报告生成时间**:2026-06-09
**调研方法**:6 个并行深度调研 agent(覆盖数据/训练/架构/推理时/循环模型/系统)
**参考文献**:100+ 论文/仓库(2023 NeurIPS ToT → 2026 CVPR CA-TTS)
**关键 1B 实证**:DeepSeek-R1-Distill-1.5B(MATH-500 83.9%)、Ouro-1.4B(GSM8K 78.92)、Tina($9 AIME24 43.33)、FastCuRL-1.5B(AIME24 49.6)、Llama-3.2-1B+PF(MATH-500 59.6)、Phi-1.5+PAL(GSM8K 81.5)、qwen3:1.7b(Agent 0.96)

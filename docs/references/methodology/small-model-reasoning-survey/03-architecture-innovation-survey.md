# 1B 小模型架构创新方向深度文献调研报告

> 调研时间:2026 年 6 月 | 覆盖文献至 2025 年底 / 2026 年初

---

## 总览:10 项技术成熟度图谱

| 技术 | 1B 规模验证 | SOTA 推理收益 | 工程复杂度 | 兼容性 | 优先级 |
|---|---|---|---|---|---|
| **Ouro / Looped Transformer** | ✅ **强**(1.4B 官方) | 🟢 +5–15% GSM8K vs 同规模 | 易 | ✅ 兼容任意 Transformer | 🥇 **第一** |
| **Neuro-Symbolic (PAL/CoT+Solver)** | ✅ **强**(1.3B Phi-GSM) | 🟢 +30%+ GSM8K | 易 | ✅ 完全兼容 | 🥈 **第二** |
| **MoE Upcycling + Drop-Upcycling** | ✅ **强**(Marco-Mini 0.86B) | 🟡 主要提升知识 | 中 | ✅ 直接复用预训练权重 | 🥉 **第三** |
| mHC / Hyper-Connections | ⚠️ 仅 780M 实验 | 🟡 小幅提升 +1.8× 收敛 | 中-难 | ✅ Drop-in | 备选 |
| Gated DeltaNet | ✅ 1.3B 已验证 | 🟡 与 Transformer 持平 | 中 | ❌ 需混合架构 | 备选 |
| Mamba / Mamba-2 | ✅ 1.5B 已验证 | 🟡 GSM8K 弱于 Transformer | 中 | ❌ 需 hybrid | 备选 |
| RWKV-7 | ✅ 0.19B–2.9B | 🟡 多语种优势明显 | 中 | ❌ 需重写训练 | 备选 |
| Universal Transformer | ❌ 仅 encoder-decoder | 🟡 长度泛化有效 | 易 | ✅ | 历史价值 |
| TTT 层 | ⚠️ 1.3B 早期 | ❌ 内存 I/O 瓶颈 | 难 | ❌ 全新层 | 不推荐 |
| Marco-Mini 单独 | ✅ 0.86B active | 🟡 多语言强、推理中等 | 中 | ⚠️ 需 dense 起点 | 与 Upcycling 合并 |

---

## 1. Universal Transformer(UT)— 深度循环

**简要**:在时间维度("深度")递归应用同一 Transformer 块,并配合 ACT(Adaptive Computation Time)让每个 token 自适应决定循环步数。理论上可证 Turing-complete(标准 Transformer 仅 TC0)。

**关键文献 / 仓库**:
- Dehghani et al. *"Universal Transformers"*, ICLR 2019 — [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
- Tensor2Tensor 实现 — [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/t2t/models/research/universal_transformer.py)
- 2024–25 续作 *"Looped Transformers for Length Generalization"*(Anthropic/OpenReview)— [openreview.net/pdf?id=PEdOdntGJG](https://openreview.net/pdf?id=PEdOdntGJG)

**1B 规模验证**:❌ **否**(无 1B+ LLM 预训练实验,仅在 bAbI、LAMBADA、MT 等小任务验证 +0.9 BLEU)。

**关键限制**:
- 原始 UT 仅在 encoder-decoder 框架验证,未扩展到现代 decoder-only LLM
- ACT 机制在 LM 中存在 "思考成本 ≠ 真实难度" 失配问题
- 长度泛化有效,但 perplexity 持平甚至更差(Giannou et al. 2023 后续工作确认)

**工程复杂度**:**Easy**(核心改动 < 100 行 PyTorch)。

**与现有权重兼容性**:✅ **完全兼容**(可对任意现成 checkpoint 做循环)。

**对 1B 的实际价值**:低。原始论文未扩展到 LLM 尺度;后续工作(如 Looped Transformer for Length Generalization)仅在 toy arithmetic 上验证。**已被 Ouro(2025)超越**。

---

## 2. mHC(Manifold-Constrained Hyper-Connections)— 多流残差反馈

**简要**:ByteDance 2024 HC → DeepSeek 2025 mHC。将单条残差流扩展为 n 条并行流(默认 n=4),通过 **Sinkhorn-Knopp 算法** 将残差矩阵约束到 Birkhoff 多面体(双随机矩阵),恢复恒等映射性质,防止训练发散。

**关键文献 / 仓库**:
- Zhu et al. (ByteDance) *"Hyper-Connections"*, arXiv 2024 — [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)
- Xie et al. (DeepSeek) *"mHC: Manifold-Constrained Hyper-Connections"*, Dec 2025 — [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
- 首个开源 mHC LM (780M) *"Ablate and Rescue"* — [arXiv:2603.14833](https://arxiv.org/html/2603.14833)
- 第三方简化版 *"mHC-lite"* — [arXiv:2601.05732](https://arxiv.org/html/2601.05732) | [GitHub: FFTYYY/mhc-lite](https://github.com/FFTYYY/mhc-lite)
- 非官方 PyTorch 实现 — [GitHub: tokenbender/mHC-manifold-constrained-hyper-connections](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)

**1B 规模验证**:⚠️ **部分**。原 HC 论文在 1B 密集 + 7B MoE 上验证(DHC 在 OLMoE 上 ARC-C +6 points、1.8× 更快收敛);mHC 论文主实验在 **27B MoE**。首个开源 mHC 训练 checkpoint 仅 780M(GPT-2 改造),但消融实验充分。

**关键限制**:
- 原 HC 在 12k 训练步后开始 loss 发散(mHC 解决此问题)
- 需要 TileLang + 专用 CUDA kernel(mHC),或用 mHC-lite 替代(数学等价、纯 PyTorch)
- 训练开销 +6.7%(DeepSeek-V3 规模下)

**工程复杂度**:**Medium-Hard**(HC 原版易、mHC 难、mHC-lite 中)。

**与现有权重兼容性**:✅ **Drop-in**(替换残差模块即可)。

**对 1B 的实际价值**:中。HC 早期数据积极(1.8× 收敛加速),但 mHC 主论文缺少 1B 公开对比。**1-2 人团队推荐使用 mHC-lite 起步**。

---

## 3. DeltaNet / Gated DeltaNet — 线性注意力 + Delta Rule

**简要**:DeltaNet(Yang et al. 2024)将线性注意力的加性更新替换为 delta rule(Widrow-Hoff),仅更新当前 key 对应的 value;**Gated DeltaNet**(NVIDIA, ICLR 2025)加入 α_t gating,可快速清空记忆。两者都实现硬件并行训练(chunkwise WY representation)。

**关键文献 / 仓库**:
- Yang et al. *"Parallelizing Linear Transformers with the Delta Rule"*, NeurIPS 2024 — [arXiv:2406.06484](https://arxiv.org/pdf/2406.06484)
- Yang et al. (NVIDIA) *"Gated Delta Networks"*, ICLR 2025 — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | [GitHub: NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- 已集成进 Qwen3-Next(80B-A3B)、Olmo Hybrid、Qwen3.5(参见后文 ablation)

**1B 规模验证**:✅ **是**(DeltaNet 训练 1.3B / 100B tokens;Gated DeltaNet 在 400M / 1.3B 全面评估)。

**关键数字(1.3B 规模)**:

| Benchmark | Transformer++ | Mamba2 | DeltaNet | Gated DeltaNet |
|---|---|---|---|---|
| ARC-c(0-shot) | ~32 | ~24 | ~30 | ~33 |
| S-NIAH-2 (4K) | — | 56.2 | 18.6 | 92.2 |
| S-NIAH-3 (4K) | — | 4.6 | 22.4 | 27.6 |

**关键限制**:
- **检索任务强(gain +20–40 pt)但通用推理仅小幅领先或持平**
- 状态大小是关键瓶颈:DeltaNet 在 1.3B 尺度 state size 扩展性差,落后于 GLA
- 纯 Gated DeltaNet 在 commonsense reasoning 上**几乎与 Mamba2 / DeltaNet 平手**,相对 Transformer 优势不大

**工程复杂度**:**Medium**(需专用 kernel,CUDA / Triton 实现)。

**与现有权重兼容性**:❌ **不可直接复用**(需重新训练或 hybrid 蒸馏)。

**对 1B 的实际价值**:中。**单独使用 DeltaNet 推理能力不强**,但作为 hybrid 组件(如 Qwen3.5-0.8B:18 层 DeltaNet + 6 层 attention)已在 1B 以下验证成功。Falcon-H1 同样。

---

## 4. Mamba / Mamba-2 — 状态空间模型

**简要**:Mamba(Gu & Dao 2023)选择性 SSM;Mamba-2(Dao & Gu 2024)通过 Structured State Space Duality(SSD)证明与 attention 数学等价性,训练速度 +2–8×。两者均线性复杂度、常数内存。

**关键文献 / 仓库**:
- Gu & Dao *"Mamba"*, Dec 2023 — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Dao & Gu *"Mamba-2"*, 2024 — [GitHub: state-spaces/mamba](https://github.com/state-spaces/mamba)
- NVIDIA *"Empirical Study of Mamba"*, 8B 规模 — [research.nvidia.com](https://research.nvidia.com/publication/2024-06_empirical-study-mamba-based-language-models)
- Mamba-3 (2026, 最新):1.5B 超越 Gated DeltaNet — [OpenReview: HwCvaJOiCj](https://openreview.net/pdf?id=HwCvaJOiCj)

**1B 规模验证**:✅ **是**(Mamba-2 1.5B vs Transformer-1.5B、Hymba-1.5B 已发表对比)。

**关键 1B 数据**:

| Benchmark | Transformer-1.5B | Mamba-2-1.5B | Mamba-3-1.5B |
|---|---|---|---|
| ARC-c | 40.4 | 41.8 | **42.7** |
| HellaSwag | 60.6 | 61.4 | 61.9 |
| GSM8K (5-shot, pure Mamba) | — | **很低**(41.32% for Falcon Mamba-7B) | 较弱 |

**关键限制**:
- **纯 Mamba 在 GSM8K 上明显弱于 Transformer**(Falcon Mamba-7B 41.3% vs LLaMA 8B 75.2%)
- 复制能力差:Jelassi et al. 2024 *"Repeat After Me"* 证明 Transformer 在 phonebook 检索上**比 10× 大的 Mamba 还强**
- CoT 推理理论局限:任意 DP 问题上,Mamba 总成本与 Transformer 持平(Yale *"Exploring Limitations of Mamba"* — [arXiv:2410.03810](https://arxiv.org/html/2410.03810v2))
- 16k context 后 perplexity 不再下降(TTT 论文验证)

**工程复杂度**:**Medium**(官方 CUDA kernel 已开源)。

**与现有权重兼容性**:❌ **不可直接复用**。

**对 1B 的实际价值**:**低(仅推理)或 中(hybrid)**。Mamba-3-MIMO-1.5B 已开始追平 Transformer 推理能力,但纯 Mamba 在 math/code 上仍落后。**仅推荐 hybrid 部署**。

---

## 5. RWKV-7 ("Goose") — 线性注意力 + 广义 Delta Rule

**简要**:RWKV-7(Peng et al. 2025)将 delta rule 推广为向量值门控 + 向量值 in-context learning rate + 分离 remove/add key。在表达力上超越 TC0(Transformer 的复杂度上限),理论上可识别所有正则语言。

**关键文献 / 仓库**:
- Peng et al. *"RWKV-7 Goose with Expressive Dynamic State Evolution"*, Mar 2025 — [arXiv:2503.14456](https://arxiv.org/pdf/2503.14456)
- 官方训练/推理代码 — [GitHub: BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- 模型权重 — [huggingface.co/RWKV](https://huggingface.co/RWKV)(0.1B / 0.4B / 1.5B / 2.9B / 7.2B)
- 架构 wiki — [wiki.rwkv.com/basic/architecture.html](https://wiki.rwkv.com/basic/architecture.html)

**1B 规模验证**:✅ **是**(0.19B → 2.9B 全套已发布;7.2B 也已开源)。2.9B "Goose" 在**多语种任务上达到 3B SOTA**,英文接近 3B SOTA。

**关键数字(2.9B vs 同行 3B)**:

| Model | Multilingual SOTA | English SOTA | ARC-c |
|---|---|---|---|
| Llama-3.2-3B | — | baseline | 59.1 |
| RWKV-7-2.9B | ✅ yes | match | 49.5 |

**关键限制**:
- 2.9B "Goose" ARC 49.5 vs Llama-3.2-3B 59.1:英文推理 **明显落后 Transformer**
- 7.2B 训练用 RTX 5090 仅 145 token/s(fp16 bsz1),无 KV cache 节省 GPU 内存
- 无正式大模型厂商采用(Qwen3.5、Qwen3-Next、Falcon-H1 均使用 Gated DeltaNet / Mamba 而非 RWKV)

**工程复杂度**:**Medium**(有官方 CUDA kernel,但需重写训练 pipeline)。

**与现有权重兼容性**:❌ **不可直接复用**。

**对 1B 的实际价值**:**低-中**。多语种场景有价值;英文推理 + math 与 Transformer 仍有差距。

---

## 6. MoE Upcycling — 密集模型转换为 MoE

**简要**:Komatsuzaki et al. (Google, ICLR 2023)提出从密集 checkpoint 初始化 MoE:复制原 MLP 作为多个 expert,添加新 router,继续训练。**仅需原训练预算的 50%** 即可超越 dense。

**关键文献 / 仓库**:
- Komatsuzaki et al. *"Sparse Upcycling"*, ICLR 2023 — [arXiv:2212.05055](https://arxiv.org/pdf/2212.05055) | 代码:[google-research/t5x/moe](https://github.com/google-research/t5x/tree/main/t5x/contrib/moe)
- NVIDIA *"Upcycling LLMs into MoE"*, Oct 2024 — [arXiv:2410.07524](https://arxiv.org/abs/2410.07524) | 代码:[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- **Drop-Upcycling**(2025):统计重新初始化部分权重解决专家同质化 — [arXiv:2502.19261](https://arxiv.org/html/2502.19261v2)
- Branch-Attend-Mix (BAM):也升级 attention 为 expert — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/665bb142d4b9f55660cb89bb56a66fe1-Paper-Conference.pdf)
- DS-MoE(dense + sparse 混合) — [arXiv:2404.05567](https://arxiv.org/pdf/2404.05567)

**1B 规模验证**:✅ **是**(Marco-Mini 在 0.6B/0.86B 起点 upcycle;Nemotron-4 15B 验证)。

**关键数字(Nemotron-4 15B, 1T tokens)**:

| 方法 | MMLU | Notes |
|---|---|---|
| Continued dense training | 65.3% | — |
| Upcycled MoE (E8G1T2) | **67.6%** | +2.3 pt |

**Drop-Upcycling 关键收益**:5.9B active MoE 匹配 13B dense,**仅用 1/4 训练 FLOPs**。

**关键限制**:
- **Naive Upcycling 长期训练慢于 from-scratch**(专家同质化)→ 必须用 Drop-Upcycling 或 Virtual Group Init
- 需要 careful learning rate schedule(与 dense 训练不同)
- 需要支持 fine-grained MoE 架构

**工程复杂度**:**Medium**(已开源 NVIDIA / Google / Marco 实现)。

**与现有权重兼容性**:✅ **完全兼容**(直接复用 dense checkpoint)。

**对 1B 的实际价值**:**高**。这是**唯一直接复用现有 1B 预训练成本**的方案,推理时通过 active params 减少 50-70% FLOPs。

---

## 7. Marco-Mini / Marco-Nano — 小型 MoE 设计

**简要**:阿里巴巴 AIDC AI 推出的 Marco-MoE 系列。Marco-Nano-Base(**0.6B active / 8B total**)、Marco-Mini-Base(**0.86B active / 17.3B total**),从 Qwen3-0.6B-Base 通过 fine-grained sub-matrix splitting + Drop-Upcycling 转换。

**关键文献 / 仓库**:
- *"Marco-MoE: Open Multilingual Mixture-of-Expert Language Models with Efficient Upcycling"* — [arXiv:2604.25578](https://arxiv.org/html/2604.25578)
- 模型权重:
  - [huggingface.co/AIDC-AI/Marco-Mini-Base](https://huggingface.co/AIDC-AI/Marco-Mini-Base)
  - [huggingface.co/AIDC-AI/Marco-Nano-Base](https://huggingface.co/AIDC-AI/Marco-Nano-Base)
  - [huggingface.co/AIDC-AI/Marco-Mini-Instruct](https://huggingface.co/AIDC-AI/Marco-Mini-Instruct)
- 前作 Marco-LLM — [arXiv:2412.04003](https://arxiv.org/html/2412.04003v1)

**1B 规模验证**:✅ **是**(0.86B active 是核心产品形态)。

**关键数字(vs Qwen3-4B-Instruct 同尺度)**:
- Marco-Mini-Instruct (0.86B active) **与 Qwen3-4B-Instruct 持平甚至更优**
- 训练 FLOPs **仅 Qwen3-4B 的 1/5.5**
- 多语种(29 种语言)长尾场景领先

**关键限制**:
- 主要是多语种 / 知识型任务优化,**纯推理 (GSM8K / MATH) 与 dense 4B 大致持平**
- 依赖 Qwen3-0.6B 作为起点(不能 from-scratch)
- 总参数 17.3B → 部署需 ~35GB bf16(边缘部署受限)

**工程复杂度**:**Medium**(已开源数据集、训练日志、超参数)。

**与现有权重兼容性**:⚠️ **需要 dense 起点**(如 Qwen3-0.6B)。

**对 1B 的实际价值**:**高**(与 MoE Upcycling 合并评估)。已 production-ready。

---

## 8. Ouro / Looped Language Models — 深度循环 + 参数共享 ⭐

**简要**:**Ouro(ByteDance, Oct 2025)** 是该方向目前最系统的工业级实现。1.4B / 2.6B 参数在 **7.7T tokens** 上预训练,使用 4 次循环的共享权重 block,配合 **熵正则化目标** + **early-exit gating** 实现动态计算深度分配。

**关键文献 / 仓库**:
- Zhu et al. (ByteDance) *"Scaling Latent Reasoning via Looped Language Models"*, Oct 2025 — [arXiv:2510.25741](https://arxiv.org/pdf/2510.25741) | [arXiv:2510.25741v2](https://arxiv.org/html/2510.25741v2)
- 项目主页 — [ouro-llm.github.io](https://ouro-llm.github.io/)
- 模型权重 — [huggingface.co/ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) / [Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B)
- 2024 早期工作 *Looped Transformers are Better at Learning Learning Algorithms* — [arXiv:2311.12424](https://arxiv.org/html/2311.12424v2)
- 2024 同期工作 *Looped Transformers for Length Generalization*(Anthropic)— [openreview.net/pdf?id=PEdOdntGJG](https://openreview.net/pdf?id=PEdOdntGJG)
- Giannou et al. 2023 — 首次证明 24-layer 1B **loop 比同参数非循环模型在 GSM-style math +5–10 pt**
- "Loop as a Bridge" introspection 研究 — [arXiv:2601.10242](https://arxiv.org/html/2601.10242)

**1B 规模验证**:✅✅ **强 + 官方**(1.4B 是核心产品形态)。

**关键数字(Ouro-1.4B R4 vs 同参数量 / 大 2-3× 模型)**:

| Benchmark | Qwen3-1.7B (baseline) | Ouro-1.4B R4 | Qwen3-4B (2× params) | Qwen3-8B (5× params) |
|---|---|---|---|---|
| **GSM8K** | 60.73 | **78.92** | 72.86 | — |
| **MATH500** | 17.60 | **82.40** | 59.60 | — |
| **MMLU-Pro** | 30.8 | 51.2 (TBD) | — | — |
| **BBH** | — | 71.02 (vs Qwen3-4B 70.95) | — | — |
| **OlympiadBench** | — | 71.55 (vs Qwen3-4B 73.18) | 73.18 | 75.25 |
| **BeyondAIME** | — | 34.0 (vs Qwen3-4B 31.0) | 31.0 | 38.0 |

**Ouro-2.6B R4 vs Qwen3-8B**:GSM8K 81.50 vs 78.17;MATH500 61.20 vs 62.30;**AIME25 / LiveCodeBench 持平**

**核心科学发现**:
- **Looping 不增加参数知识容量**(≈2 bits/parameter for both)
- **但显著提升 "知识操纵能力"**(multi-hop reasoning、fact composition)
- 隐式推理比显式 CoT 更忠实(latent trace → final output 强对齐)
- safety 在 R > 训练深度时仍单调改善

**KV Cache 优化**(重要工程细节):

| 策略 | GSM8K | MATH-500 | 内存节省 |
|---|---|---|---|
| Full 4× cache | 78.92 | 82.40 | 1× |
| First-step only | **18.73** ❌ | 8.43 | 4× |
| **Last-step only** | **78.85** ✅ | 80.40 | 4× |
| Averaged | 78.73 | 78.52 | 4× |

→ 推理时仅需最后一步 KV cache,等效显存 = 标准 1.4B Transformer

**关键限制**:
- **推理 FLOPs 是 1× 块 × 4 循环 = 4× 等效 Transformer**(即使 KV cache 优化)
- 训练时需精心调熵正则化权重 + early-exit threshold
- **Loop 步数不能超过训练时的 T_max**(外推性能会降)
- Prefill 阶段需要完整 4 个 KV cache(不能共享)
- 早期 step 共享会"灾难性塌缩"(GSM8K 从 78.92 → 18.73)

**工程复杂度**:**Easy**(仅需在 forward 中循环 N 次 + 加 gating head;可在任何 Transformer 框架上叠加)。

**与现有权重兼容性**:✅ **部分兼容**(可从 dense 1B checkpoint 取单层 / 几层权重初始化循环块;Ouro 2.6B 也是从 1.4B upcycled 到 48 层后循环 4 次)。

**对 1B 的实际价值**:⭐⭐⭐ **极高**。是 2025–2026 最强的 1B 推理性价比方案。**2-3× 参数效率**是已 verified 的工业级 claim,不是 toy 演示。

---

## 9. Neuro-Symbolic / LLM+Solver — 外包计算 ⭐

**简要**:LLM 负责自然语言理解 + 程序/方程生成,**外部求解器**(Python / SymPy / Prover9)负责精确计算。训练-free 框架,可立即与任何 1B LLM 组合。

**关键文献 / 仓库**:
- Gao et al. *"PAL: Program-aided Language Models"*, ICML 2023 — [arXiv:2211.10435](https://arxiv.org/pdf/2211.10435) | [reasoning-machines/pal](https://github.com/reasoning-machines/pal) | [reasonwithpal.com](http://reasonwithpal.com)
- *"Solving Math Word Problems by Combining LLMs with Symbolic Sololvers"*(He-Yue et al.)— [mathai2023.github.io/papers/16.pdf](https://mathai2023.github.io/papers/16.pdf)
- *"MathChat"*(GPT-4 + Python)— [arXiv:2306.01337](https://arxiv.org/html/2306.01337v3)
- *"LINC"*(FOL theorem prover)— [arXiv:2310.15164](https://d6108366.hf-mirror.com/papers/2310.15164)
- *"NeuroProlog"*(Prolog-based cocktail FT)— [arXiv:2603.02504](https://arxiv.org/html/2603.02504)
- *"SymCode"*(SymPy + self-debug)— [aclanthology.org/2026.findings-eacl.76.pdf](https://aclanthology.org/2026.findings-eacl.76.pdf)
- **TinyGSM**(1.3B Phi-GSM **81.5% GSM8K**):code execution 作为 verifier — [arXiv:2312.09241](https://ar5iv.labs.arxiv.org/html/2312.09241)

**1B 规模验证**:✅✅ **强 + 官方**(TinyGSM 1.3B 81.5% GSM8K rival GPT-3.5;Phi-2 2.7B 74.3%)。

**关键数字(GSM8K)**:

| 模型 | 方法 | GSM8K |
|---|---|---|
| Llama-2 7B | CoT | 14.6% |
| Llama-2 70B | CoT | 56.8% |
| Llama-2 70B | MetaMath | 82.3% |
| **Phi-1.5 1.3B** | **PAL (code)** | **68.2%** |
| **Phi-1.5 1.3B + 1.3B verifier** | **PAL + verify** | **81.5%** ⭐ |
| Phi-2 2.7B | PAL | 74.3% |
| LLaMA-3.2 3B (Instruct) | vanilla | 77.4% (close to GPT-3.5) |

PAL 用 **Codex 175B** 在 GSM8K 上**比 PaLM-540B + CoT 高 15 pt**!

**关键限制**:
- **需要 prompt 工程**:LLM 必须学会生成可执行代码(不是所有 LLM 都天然支持)
- 仅适用于**计算可形式化的任务**(math/code/logic),对开放式推理无效
- **延迟增加**(外部 solver 调用 + 多候选验证)
- 1B 模型需要 SFT / 蒸馏才能稳定生成 PAL 程序(Phi-GSM 即是典型例子)

**工程复杂度**:**Easy**(框架级集成,<1 天开发)。

**与现有权重兼容性**:✅ **完全兼容**(可在任何 1B LLM 上 prompt,无需改权重)。

**对 1B 的实际价值**:⭐⭐⭐ **极高**。这是**唯一不依赖架构改造就能立即提升 1B 推理能力**的方案。Phi-GSM 81.5% 是 2023-2024 的 SOTA,**2026 年仍是性价比最高的选择**。

---

## 10. TTT(Test-Time Training)— 推理时权重更新

**简要**:Sun et al. (Stanford, 2024)把 RNN 的 hidden state **本身当作一个可训练模型**(线性模型或 2 层 MLP),推理时 hidden state 通过 self-supervised 梯度下降持续更新。理论上 TTT-Linear ≡ linear attention,TTT-MLP 表达力更强。

**关键文献 / 仓库**:
- Sun et al. *"Learning to (Learn at Test Time): RNNs with Expressive Hidden States"*, ICML 2025 — [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) | [proceedings.mlr.press/v267/sun25h.html](https://proceedings.mlr.press/v267/sun25h.html)
- 代码 — [GitHub: test-time-training/ttt-lm-jax](https://github.com/test-time-training/ttt-lm-jax)
- 早期版本 *"Learning to (Learn at Test Time)"* — [arXiv:2310.13807](https://arxiv.org/pdf/2310.13807)

**1B 规模验证**:⚠️ **是但有限**(125M–1.3B 已评估,与 Mamba 在 2k context 持平,8k+ 优于 Mamba)。

**关键限制**:
- **TTT-MLP 内存 I/O 瓶颈严重**(论文自己承认),训练 kernel 难写
- 推理时每 token 需做 mini-batch SGD(延迟高)
- 后续**未被任何主流 LLM 厂商采用**(Qwen3-Next 选了 Gated DeltaNet,Jamba/Falcon-H1 选了 Mamba-2)
- 本质上仍是 RNN,缺少 attention 的并行优势

**工程复杂度**:**Hard**(需要专用 GPU/TPU kernel,研究性强但工程不成熟)。

**与现有权重兼容性**:❌ **不可复用**。

**对 1B 的实际价值**:⭐ **低**。**目前没有 production-grade 实现**;概念有趣但工业不可行。

---

## 关键交叉问题回答

### Q1: Looped Transformer / Ouro 是真的吗?1B 实际收益?

✅ **真实**。ByteDance 官方发布:
- [huggingface.co/ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B)(1.4B, 7.7T tokens 训练)
- [huggingface.co/ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B)(2.6B, upcycled 到 48 层循环 4 次)

**1.4B R4 vs 同参数 dense Qwen3-1.7B**(同公司 baseline):
- GSM8K: **78.92 vs 60.73** (+18.2 pt)
- MATH500: **82.40 vs 17.60** (+64.8 pt)

**vs 同预算更大模型**(公平对照):
- 1.4B R4 vs Qwen3-4B(2.9× params):GSM8K 78.92 vs 72.86;MATH500 82.40 vs 59.60
- 2.6B R4 vs Qwen3-8B(3× params):GSM8K 81.50 vs 78.17;OlympiadBench 76.44 vs 75.25

**isoparam 控制**(Giannou 2023, NeurIPS Workshop):
- 24 层 1B 循环 vs 24 层 1B:math word problems **+5 pt**
- 12 层循环 2 次 vs 24 层 baseline:math word problems **+5 pt**

### Q2: mHC vs Looped Transformer 哪个更适合 1B?

| 维度 | mHC | Looped Transformer (Ouro) |
|---|---|---|
| 训练 FLOPs | 同标准 Transformer | 4× 标准(同 T×) |
| 推理 FLOPs | 同标准 | 4× 标准 |
| 推理 GSM8K 提升 | +1.8× 收敛,无明确推理数字 | **+18 pt**(Ouro-1.4B vs Qwen3-1.7B) |
| 工程难度 | Medium-Hard | Easy |
| 现成 checkpoint | 无(780M 实验性) | ✅ 有 1.4B / 2.6B official |
| 风险 | 训练不稳定需 mHC-lite | 4× 训练 FLOPs 成本 |

**结论**:**Looped Transformer(Ouro)更实用**——已有 production-grade 1B 权重可直接下载或继续训练;mHC 仍是 "research-stage",缺 1B 公开对比。

### Q3: DeltaNet / Mamba 在 1B 推理上能不能 match Transformer?

**纯 DeltaNet/Mamba 1B → ❌ 不能**:
- Mamba-2-1.5B:通用 NLU 接近 Transformer,但 GSM8K / MATH 明显落后
- 纯 Mamba 在 16k+ context 退化(TTT 论文验证)

**Hybrid DeltaNet+Mamba+Attention 1B → ✅ 能接近**:
- **Qwen3.5-0.8B**:18 层 Gated DeltaNet + 6 层 attention(3:1 比例)→ 实测 1B 以下 SOTA
- **Falcon-H1-0.5B**:每层并行 Mamba-2 + attention → 同等表现
- **Qwen3-Next-80B-A3B**:75% Gated DeltaNet + 25% attention → 已 production
- **Hybrid ablation**:移除任一组件,**GSM8K 全部塌缩到 0**(证明二者均必需)

**结论**:纯 SSM/RNN 1B 在 math/code 上**与 Transformer 有 20-30 pt 差距**;**hybrid 是必须的**,且 hybrid 1B 已接近 Transformer 性能。

### Q4: MoE Upcycling 成本?是否真的提升推理(不是仅吞吐量)?

**成本**:约 **50% of original dense pretraining**(Komatsuzaki 原论文);Drop-Upcycling 在长期训练中甚至**比 from-scratch 更快收敛**。

**是否提升推理(不仅是 throughput)**:

| 任务类型 | MoE 收益 |
|---|---|
| 知识型(MMLU、TriviaQA、HellaSwag) | ✅ 显著(与 dense 同 total params 持平或更好) |
| 推理型(GSM8K、MATH、ARC-c) | ⚠️ **饱和甚至退化** |

具体数据("Optimal Sparsity of MoE for Reasoning", arXiv 2508.18672):
- GSM8K 上,**MoE 增加总参数但 task loss 反而上升**(U-shaped 曲线)
- 当 FLOPs 充足时,**denser MoE 比 sparser MoE 在 reasoning 上更好**
- GRPO / test-time compute **无法挽救过度稀疏 MoE 的 reasoning 缺陷**

**Marco-Mini 实证**:
- 0.86B active 在多语种 / 知识上匹配 4B dense
- **纯推理(math)上仅与 Qwen3-4B 持平**,不显著超越

**结论**:MoE Upcycling **确实提升 knowledge capacity 和 throughput**,但对纯 reasoning(math/code)的**增益有限**——sparsity 越高 reasoning 越差。

---

## 最终排名:1B 推理能力提升 Top 3

### 🥇 #1: Ouro / Looped Transformer

**理由**:
- ✅ **唯一有官方 1.4B/2.6B 开源 checkpoint + 实测数据**(GSM8K +18 pt、MATH500 +64 pt vs 同参 dense)
- ✅ 工程简单(循环 + gating),与现有 1B 训练 pipeline 兼容
- ✅ 支持 upcycling(Ouro-2.6B 是从 1.4B 升级而来)
- ✅ KV cache 可压缩 4×(仅保留最后一步)
- ⚠️ 训练成本 4×,但推理可通过 gating 自适应
- 适用:**所有追求 reasoning 性价比的 1B 团队**

**建议实现路径**:
1. 下载 [Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) → eval GSM8K baseline
2. 在此基础上做 SFT(reasoning data)或 LoRA
3. 若需更多推理能力 → 继续 pretrain 到 2.6B R4

### 🥈 #2: Neuro-Symbolic (PAL / SymCode / Phi-GSM)

**理由**:
- ✅ **零架构改动**,任何 1B LLM 立即受益
- ✅ **TinyGSM 1.3B → 81.5% GSM8K**(已 verified 2023)
- ✅ PAL 比 PaLM-540B + CoT 高 15 pt 的实证
- ✅ 工程 1 天可集成
- ⚠️ 仅适用于可形式化任务(math/code/logic)
- ⚠️ 需要 prompt engineering + SFT(不能纯 zero-shot)

**建议实现路径**:
1. 选 base LLM(Qwen2.5-1.5B 或 Ouro-1.4B)
2. 在 GSM8K + MATH + codeforces 上 SFT 训练 "PAL 程序生成"
3. 推理时加入 Python sandbox + verifier(参考 TinyGSM)

### 🥉 #3: MoE Upcycling + Drop-Upcycling(Marco-Mini 范式)

**理由**:
- ✅ **唯一直接复用现有 1B 预训练成本**的方案
- ✅ 已 production-ready(Marco-Mini 0.86B、Nemotron-4 upcycling)
- ✅ 推理 FLOPs 减少 50-70%(吞吐量优势)
- ✅ 多语种 / 知识型任务明显领先
- ⚠️ **对纯 reasoning 增益有限**(甚至可能因专家同质化变差)
- ⚠️ 必须用 Drop-Upcycling / Virtual Group Init 而非 naive

**建议实现路径**:
1. 选 dense 1B base(Qwen3-0.6B → Marco-Nano 范式)
2. 选 **Drop-Upcycling**(不要 naive upcycling)→ 部分 expert FFN 随机重新初始化
3. 配置 fine-grained experts(如 64–256 experts, top-k=8)
4. Continue pretrain 5–6T tokens
5. **聚焦多语种 / 知识任务;纯推理仍需配合 SFT / RFT**

---

## 总结表

| 1B 推理提升路径 | 推荐度 | 实施成本 | 收益量级 | 风险 |
|---|---|---|---|---|
| **Ouro Looped Transformer** | ⭐⭐⭐⭐⭐ | 4× 训练 FLOPs | +18 pt GSM8K, +64 pt MATH500 | 已被 7.7T 训练验证 |
| **PAL / Neuro-Symbolic** | ⭐⭐⭐⭐⭐ | 1 天集成 | 1.3B → 81.5% GSM8K | 仅适用形式化任务 |
| **MoE Upcycling + Drop** | ⭐⭐⭐⭐ | 中等 | 多语种 +5–10 pt | 对 reasoning 增益有限 |
| mHC / mHC-lite | ⭐⭐⭐ | Medium | +1.8× 收敛速度 | 缺 1B 公开对比 |
| Hybrid DeltaNet/Attention | ⭐⭐⭐ | 中等 | 接近 Transformer | 必须 hybrid;纯 SSM 推理弱 |
| Universal Transformer | ⭐⭐ | Easy | 历史价值 | 未扩展到 1B LLM |
| TTT 层 | ⭐ | Hard | 长 context 改进 | 内存 I/O 瓶颈 |
| 纯 Mamba-2 / RWKV-7 | ⭐⭐ | Medium | 吞吐量优势 | 推理落后 20-30 pt |

**最终建议(1-2 人团队)**:

1. **第一步(立即)**:在现有 1B base 上集成 **PAL/Neuro-Symbolic** — 1 周内能看到 GSM8K +15-30 pt。
2. **第二步(1-2 个月)**:评估/微调 **Ouro-1.4B**(直接下载)或在自己 1B 模型上实现循环 — GSM8K 突破 75%。
3. **第三步(可选)**:如果有 dense 1B → 考虑 **Drop-Upcycling 到 MoE** — 提升吞吐量与多语种能力。

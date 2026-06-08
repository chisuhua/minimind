# MiniMind 与业界 SOTA 推理加速技术 差距分析

> **报告日期**: 2026-06-08
> **项目版本**: MiniMind-3 (64M Dense) / MiniMind-3-MoE (198M-A64M)
> **分析范围**: 业界 SOTA 推理性能提升技术 (架构级 / 注意力级 / KV 缓存级 / 并行解码级), 对照 MiniMind 现状给出差距与集成路径

---

## 目录

- [一、术语澄清与命名规范](#一术语澄清与命名规范)
- [二、MiniMind 当前架构能力盘点](#二minimind-当前架构能力盘点)
- [三、业界 SOTA 推理加速技术总览矩阵](#三业界-sota-推理加速技术总览矩阵)
- [四、用户重点关注的 8 项技术详解](#四用户重点关注的-8-项技术详解)
  - [4.1 DFlash (Block Diffusion Speculative Decoding)](#41-dflash-block-diffusion-speculative-decoding)
  - [4.2 DDTree (DFlash 树形多路并行推测解码)](#42-ddtree-dflash-树形多路并行推测解码)
  - [4.3 Lightning Indexer 混合稀疏注意力 (DSA)](#43-lightning-indexer-混合稀疏注意力-dsa)
  - [4.4 全局 + 线性注意力层混合架构](#44-全局--线性注意力层混合架构)
  - [4.5 Gated DeltaNet](#45-gated-deltanet)
  - [4.6 mHC 残差注意力 (Manifold-Constrained Hyper-Connections)](#46-mhc-残差注意力-manifold-constrained-hyper-connections)
  - [4.7 RTPurbo (低维子空间轻量级索引推理)](#47-rtpurbo-低维子空间轻量级索引推理)
  - [4.8 TriAttention (三角注意力)](#48-triattention-三角注意力)
- [五、其他相关 SOTA 推理加速技术](#五其他相关-sota-推理加速技术)
- [六、逐项差距分析](#六逐项差距分析)
- [七、推荐集成路径 (按 ROI 排序)](#七推荐集成路径-按-roi-排序)
- [八、关键结论](#八关键结论)
- [九、参考文献](#九参考文献)

---

## 一、术语澄清与命名规范

| 用户提法 | 实际情况 | 说明 |
|----------|----------|------|
| **DFlash** | ✅ 正确 | DeepSeek 2026 block diffusion speculative decoder |
| **DDTree** | ✅ 用户已澄清 | 在 DFlash 基础上, 通过树形式进行多路并行推测解码; 树形结构允许多个候选分支同时被验证, 提升 acceptance rate |
| **lightning indexer 混合洗漱注意力** | ✅ 正确 | 指 **DeepSeek-V3.2 DSA (DeepSeek Sparse Attention)** — lightning indexer + 主 attention 的稀疏混合 |
| **全局注意力和线性注意力层混合架构** | ✅ 正确 | 业界主流量产: Qwen3-Next (3:1) / Jamba (1:7) / Nemotron-H (8% attn) / Griffin (RG-LRU + local attn) |
| **Gated DeltaNet** | ✅ 正确 | NVlabs ICLR 2025; Qwen3-Next 75% 层使用, 当前 2025-2026 工业 SOTA 标配 |
| **mHc 残差注意力** | ⚠️ 部分正确 | 实际指 **mHC (Manifold-Constrained Hyper-Connections, DeepSeek-V4 2025-12)** — 这是**残差**结构创新, 非"注意力"。与用户描述的"残差注意力"含义一致 |
| **RTPurbo** | ✅ 正确 | Microsoft 2026-05; **不是 SVD-based**, 而是 head-wise 稀疏 + 16-dim indexer + top-p 选 token |
| **三角注意力** | ✅ 用户已澄清为 TriAttention | arXiv 2604.04921 — 用三角级数估分 attention-vs-distance 曲线 |

---

## 二、MiniMind 当前架构能力盘点

### 2.1 模型架构

| 模块 | 实现 | 文件/行号 |
|------|------|----------|
| 注意力 | 标准 softmax + GQA (8Q/4KV) + QK Norm + RoPE + YaRN | `model/model_minimind.py:91-134` |
| FFN | SwiGLU (Qwen3 风格) | `model/model_minimind.py:136-146` |
| MoE | 4 experts / top-1, aux-loss balanced | `model/model_minimind.py:148-176` |
| KV Cache | 朴素 `torch.cat` 增量拼接 | `model/model_minimind.py:120-123` |
| Attention 内核 | PyTorch SDPA (`scaled_dot_product_attention`) | `model/model_minimind.py:109,125-126` |
| 长上下文 | YaRN (β_fast=32, β_slow=1, factor=16) | `model/model_minimind.py:31-39` |
| 采样 | top-k + top-p + temperature + repetition_penalty | `model/model_minimind.py:257-288` |
| 量化 | 无 (fp16/fp32); 离线 GGUF 走 llama.cpp | README |
| 外部引擎 | vLLM / SGLang / llama.cpp / ollama / MNN (部署侧) | README |
| 训推分离 | SGLangRolloutEngine (用于 Agentic RL) | `trainer/rollout_engine.py` |

### 2.2 关键参数

| 参数 | Dense (64M) | MoE (198M-A64M) |
|------|-------------|-----------------|
| hidden_size | 768 | 768 |
| num_hidden_layers | 8 | 8 |
| num_attention_heads | 8 | 8 |
| num_key_value_heads | 4 (GQA 2:1) | 4 |
| head_dim | 96 | 96 |
| max_position_embeddings | 32768 | 32768 |
| rope_theta | 1e6 | 1e6 |
| vocab_size | 6400 | 6400 |
| 激活参数量 | 64M | 64M (4 experts / top-1) |

### 2.3 已发布的实验分支

| 分支 | 状态 | 备注 |
|------|------|------|
| Linear Attention (Discussion #704) | 📝 讨论中, 未合入主线 | 仅有 README 引用, 无代码实现 |
| dLM 扩散语言模型 (Discussion #618) | 📝 讨论中, 未合入主线 | 仅有 README 引用, 无代码实现 |
| Multi-Token Prediction (MTP) | ❌ 未实现 | 训练目标中无 MTP loss |

---

## 三、业界 SOTA 推理加速技术总览矩阵

> 标记: ✅ 已实现 | ⚠️ 外部依赖 | ❌ 主线未实现 | 📝 讨论中

### 3.1 并行 / 推测解码 (Parallel/Speculative Decoding)

| 技术 | MiniMind | 核心机制 | 典型加速 | 复杂度 |
|------|---------|---------|---------|--------|
| **DFlash** (DeepSeek, 2026) | ❌ | Block diffusion 一次性 draft 整块 token; KV injection 到 draft 每一层 | Qwen3-8B 6×, 相对 EAGLE-3 +2.5× | 中-高 |
| **DDTree** (DFlash 树形扩展) | ❌ | 在 DFlash 基础上, 用树形结构多路并行 draft, 提升 acceptance rate | 较 DFlash 进一步 +30-50% acceptance | 中-高 |
| **EAGLE-3** (NeurIPS 2025) | ❌ | 单层 drafter + 多层 (low/mid/high) 特征融合 + training-time test | 5.6-6.5× 相对 vanilla AR | 中 |
| **Medusa / Medusa-2** (2024) | ❌ | 在 backbone 上加 K 个并行 decoding heads, tree attention 验证 | 2.2-2.8× | **低** |
| **Lookahead Decoding** (ICML 2024) | ❌ | Jacobi 迭代, 零 draft 模型 | 1.8×, 多 GPU 强扩展下 4× | **极低** (无训练) |
| **LayerSkip / Self-Spec** (Meta ACL 2024) | ❌ | 复用同一模型, 早层出 draft, 后层 verify, 共享 KVQ cache | 1.82-2.16× | 中 (需重训) |
| **REST** (2023) | ❌ | 用 suffix-match datastore 替代 draft 模型 | HumanEval 2.12-2.36× | **低** |
| **PLD / AdaPLD** (2023-2026) | ❌ | 在 prompt 中找 n-gram overlap 作为 draft, 零训练 | input-guided 任务 2-3.1× | **极低** |
| **MTP-as-Draft** (DeepSeek-V3) | ❌ | multi-token prediction 头在推理时展开为多步 draft | 1.8× TPS | 低 (复用训练头) |
| **Diffusion LLM** (LLaDA, Mercury, Gemini, 2025) | 📝 (#618) | dLLM 并行 denoise 整段 token | >1000 TPS (Mercury), 27.6× (Fast-dLLM) | 高 (需新模型) |

### 3.2 线性 / 子二次注意力 (Linear/Sub-Quadratic Attention)

| 技术 | MiniMind | 循环更新公式 | 复杂度 | 2025-2026 代表模型 |
|------|---------|--------------|--------|---------------------|
| **Linear Transformer** (Katharopoulos 2020) | ❌ | `S_t = S_{t-1} + φ(x_t W_K) (x_t W_V)^T` | `O(L)` 训练, `O(1)` 推理 | 祖先, 后续工作的根 |
| **Mamba-1/2/3** (Gu & Dao 2023-2025) | ❌ | `h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t; y_t = C_t h_t` | `O(L)` 训练, `O(1)` 推理 | Jamba, Falcon-Mamba, Codestral-Mamba |
| **Mamba-2/SSD** (ICML 2024) | ❌ | 与 Linear Attention 对偶, A 标量×单位, head P≫1 | `O(L)`, 2-8× 快于 Mamba-1 | Bamba-9B, Nemotron-H |
| **RWKV-7 "Goose"** (COLM 2025) | ❌ | `S_t = S_{t-1}·(diag(w_t) - k̂_t^T(a_t⊘k̂_t)) + v_t^T k_t` | `O(L)` 训练, `O(1)` 推理 | RWKV-7 0.19B-2.9B |
| **Gated DeltaNet** (NVlabs ICLR 2025) | ❌ | `S_t = S_{t-1}·(α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T` | `O(L)`, `O(1)` 推理 | **Qwen3-Next-80B-A3B** (75% 层使用) |
| **GLA** (ICML 2024) | ❌ | `S_t = S_{t-1}∘G_t + v_t k_t^T`, G_t ∈ (0,1)^{d×d} | `O(L)`, `O(1)` 推理 | GLA-1.3B, HGRN-2 |
| **DeltaNet** (NeurIPS 2024) | ❌ | `S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T` (Householder-like) | `O(L)`, `O(1)` 推理 | 1.3B 模型, Gated DeltaNet 基础 |
| **RetNet** (Microsoft 2023) | ❌ | `S_t = γ·S_{t-1} + k_t^T v_t` (标量全局衰减) | `O(1)` 推理 | YOCO 内部使用 |
| **HGRN-2** (COLM 2024) | ❌ | `h_t = h_{t-1}·Diag(f_t) + i_t⊗(1-f_t)`, 状态 R^{d×d} | `O(L)`, `O(1)` 推理 | 1.3B 模型 |

### 3.3 混合架构 (Hybrid Architectures)

| 技术 | MiniMind | 组成 | 比例 | 代表模型 |
|------|---------|------|------|----------|
| **Jamba** (AI21 2024) | ❌ | Mamba + Attention + MoE | 1:7 attention:mamba | Jamba 1.5 Mini/Large (52B/12B) |
| **Zamba2** (Zyphra 2024) | ❌ | Mamba-2 + 共享 attention block | 6:1 | Zamba2-1.2B/2.7B/7.4B |
| **Griffin/RecurrentGemma** (Google 2024) | ❌ | RG-LRU + sliding window attn (1024) | 2:1 | RecurrentGemma 2B/9B |
| **Qwen3-Next** (Alibaba 2025-09) | ❌ | **Gated DeltaNet** (75%) + **Gated Attention** (25%) + Ultra-Sparse MoE | 3:1 | **80B-A3B**, 32K+ 10× decode |
| **Nemotron-H** (NVIDIA ICLR 2025) | ❌ | Mamba-2 + Attention + FFN | 8% attention | Nemotron-H-8B/47B/56B |
| **Falcon-H1** (TII 2025) | ❌ | Parallel hybrid: attn heads + Mamba-2 heads 同 block 并行 | head 数可调 | Falcon-H1 0.5B-34B |
| **Bamba** (IBM 2024) | ❌ | Mamba-2 + 3 attention layers | 3/32 (9.4%) | Bamba-9B (vLLM 支持) |
| **Hymba** (NVIDIA ICLR 2025) | ❌ | Hybrid-head (attn + SSM 并行 in-block) + meta tokens | 1.5B | Hymba-1.5B |
| **MiniMax M1** (2025-06) | ❌ | Lightning Attention + softmax attn (7:1) + 32 experts | 7:1 | M1-456B-A45.9B (1M ctx) |
| **MiniMax M2** (2025-2026) | ❌ | **全 MHA + 256 fine-grained experts + MTP-as-draft** | dense | M2-229B-A9.8B |

### 3.4 稀疏 / 压缩注意力 (Sparse/Compressed Attention)

| 技术 | MiniMind | 核心机制 | 加速 | 备注 |
|------|---------|---------|------|------|
| **NSA** (DeepSeek 2025) | ❌ | 3 路: compression (MLP) + selection (top-16 block) + sliding window (512) | 64K decode **11.6×** | 需从头预训练 |
| **MoBA** (Moonshot 2025) | ❌ | Block-sparse: query 选 top-k block (B=512) | 1-1/k 加速 | Kimi 生产部署 |
| **Sliding Window Attention** | ❌ | 每 query 只看最近 W token | O(LW) | Mistral, Longformer |
| **StreamingLLM** (ICLR 2024) | ❌ | attention sink (前 4 token) + sliding window | 22.2×/token | 无需训练 |
| **Lightning Indexer (DSA)** (DeepSeek-V3.2) | ❌ | FP8 轻量 indexer (H^I heads, ReLU) + top-k (k=2048) | `O(L²)` → `O(Lk)` | 670B 工业部署 |
| **MInference 1.0** (NeurIPS 2024) | ❌ | 3 种稀疏模式: A-shape / Vertical-Slash / Block-Sparse, 离线+在线 | prefill 10× (1M ctx) | 无需重训 |
| **Quest** (ICML 2024) | ❌ | Page 级 (min, max) 元数据估分, top-K pages | 7.03× self-attn | 无需重训 |
| **TriAttention** (arXiv 2604.04921) | ❌ | 三角级数估分 attention-vs-distance 曲线 | 2.5× thr, 10.7× KV 削减 | 无需重训 |
| **Ltri-LLM** (arXiv 2412.04757) | ❌ | NMS 检测语义三角区域, 流式检索 | 长上下文加速 | 无需重训 |
| **RTPurbo** (Microsoft 2026-05) | ❌ | ~15% retrieval heads 保留全 KV, 16-dim indexer + top-p 选 token, ~600 step 微调 | 1M prefill **9.36×**, decode **2.01×** | retrofit 模式 (0.06% 训练成本) |

### 3.5 KV Cache 优化

| 技术 | MiniMind | 核心机制 | 加速 | 备注 |
|------|---------|---------|------|------|
| **GQA** (Llama-2/3) | ✅ | 4 KV heads 共享于 8 Q heads | 2× KV 削减 | 已实现 |
| **MLA** (DeepSeek-V2/V3) | ❌ | KV cache 压缩到 latent `c_t^{KV} ∈ R^{d_c=512}` + 矩阵吸收 | 92.19% KV 削减 (LLaMA2-7B → MLA) | DeepSeek-V3 用 |
| **PagedAttention** (vLLM SOSP 2023) | ⚠️ (外部) | KV block 化 + block table 映射, copy-on-write | 2-4× throughput | vLLM 集成 |
| **KV Quantization (KIVI/KVQuant/ZipCache)** | ❌ | Key per-channel 2-bit + Value per-token 2-bit | 2.6× 内存, 2.35-3.47× 吞吐 | KIVI 已开源 |
| **Cross-layer KV sharing** | ❌ | 跨层共享 KV (YOCO, LayerKV) | 1.15-1.4× thr | YOCO 用 Gated RetNet |
| **Differential Transformer** (Microsoft 2024) | ❌ | 两组 softmax attention 相减 (noise-canceling), λ 学得 | 同 MHA 复杂度但 head 数减半 | 7B 模型验证 |
| **PyramidKV / SnapKV** | ❌ | 层级分配不同 KV 预算 | 50-70% KV 削减 | 兼容 GQA |
| **HCA / CSA** (DeepSeek-V4) | ❌ | 极重压缩 attn (128×) + 索引选块压缩 attn (4×) | 长上下文加速 | DeepSeek-V4 用 |

### 3.6 内核 / 解码加速

| 技术 | MiniMind | 描述 |
|------|---------|------|
| **FlashAttention-2** | ✅ (via SDPA) | 已用 PyTorch SDPA |
| **FlashAttention-3** | ⚠️ (SDPA 间接) | 1.5-2× vs FA-2, 85% H100 利用率, FP8 |
| **FlashDecoding** | ❌ | 切分 query, 并行做 decoding attention, 解决 FA-2 decode 利用率低 |
| **BitNet b1.58 / v2** (Microsoft 2024-25) | ❌ | weight ternary {-1,0,1} + INT8 act, 70B 加速 4.1×, 内存 3.55×↓, 能耗 1/71.4 |

### 3.7 残差 / 微架构创新

| 技术 | MiniMind | 描述 |
|------|---------|------|
| **mHC (Manifold-Constrained Hyper-Connections)** (DeepSeek-V4 2025-12) | ❌ | 把残差扩展为 n 路 (hc_mult=4), Sinkhorn-Knopp 投影到双随机矩阵, 期望=恒等映射, 训练开销 6.7% |

### 3.8 MoE 推理优化

| 技术 | MiniMind | 描述 |
|------|---------|------|
| **Fine-grained Expert Segmentation** (DeepSeekMoE ACL 2024) | ❌ | 切小 expert (160 routed / 6 active + 2 shared), 组合数指数↑ |
| **Shared Expert Isolation** | ❌ | 永久激活的 shared expert |
| **Auxiliary-Loss-Free Balancing** (DeepSeek-V3) | ❌ | bias 项动态调整, 不依赖 aux loss |
| **Multi-Token Prediction (MTP)** | ❌ | 训练时额外预测 D 个未来 token, 推理时做 speculative draft (TPS 1.8×) |
| **Qwen3-Next Ultra-Sparse MoE** | ❌ | 512 experts / 10 routed + 1 shared (1:50 激活比) |

---

## 四、用户重点关注的 8 项技术详解

### 4.1 DFlash (Block Diffusion Speculative Decoding)

#### 来源

- 论文: arXiv 2602.06036 (Chen, Liang, Liu, Z-Lab, 2026-01)
- 代码库: `z-lab/dflash` (GitHub)
- 已发布 draft 模型: Qwen3-8B, LLaMA-3.1-8B, Qwen3-Coder, MiniMax-M2.5/2.7, Kimi-K2.5/2.6

#### 核心机制

1. **目标模型特征融合 (Feature Fusion)**: 在 prefill 或 verification 之后, 从目标模型的 5 个均匀分布在浅/中/深层的层抽取 hidden states, 沿 channel 维拼接, 过轻量投影得到 compact target context feature `g_t`

2. **KV 注入 (KV Injection) — DFlash 与 EAGLE-3 的核心区别**: 把 `g_t` 直接注入到 draft 模型每一层的 Key 和 Value 投影里, 并作为 KV cache 的一部分被缓存复用。EAGLE-3 只把目标特征当作 draft 模型第一层的输入 (fuse with token embedding), 层数一深就"信号稀释"; DFlash 每一层都能拿到 full context

3. **Block Diffusion 并行 draft**: anchor + 剩余 `block_size-1` 个 `[MASK]` 一次性送入 draft 模型的 1-step (默认) denoising 过程, 单次 forward 并行生成整块; block_size 默认 16 (Qwen3 系列) 或 10 (LLaMA-3.1)

4. **共享 embedding & LM head**: draft 模型冻结并复用目标模型的 token embedding 和 LM head, 只训练中间的 Transformer 层 (Qwen3 用 5 层, Qwen3-Coder 用 8 层)

5. **训练数据构造**: 随机采样 anchor 位置构造 block, 串联训练, 使用 sparse attention mask 阻止 block 间 attention 泄露

#### 典型加速

- Qwen3-8B 上 math/code/chat benchmark 取得 **6× lossless acceleration**
- 相对 EAGLE-3 **2.5× 更高**
- reasoning 模型 (thinking mode) 下约 4.5×

#### 对 MiniMind 的可行性

- **优点**: 可单 GPU 训练 5 层 draft 模型
- **难点**: 需要保留 prefill 时各层 hidden states; 实现 KV 注入到 draft 模型每一层需要写自定义 attention 算子
- **ROI**: ⭐⭐⭐⭐ (中等-高投入, 6× 加速收益)

---

### 4.2 DDTree (DFlash 树形多路并行推测解码)

#### 来源

- 在 DFlash 基础上扩展的树形多路并行推测解码方案
- 核心思想: 用树形结构 (而非 DFlash 的线性 block 序列) 进行多路并行 draft, 提升 acceptance rate
- 相关实现: EAGLE-2 / EAGLE-3 的 dtree (decoding tree), Medusa 的 tree attention

#### 核心机制

1. **树形 draft 结构**: 在每个 draft 位置不是生成单一 token, 而是生成 top-k 候选 token, 形成树形分支
   ```
   根 (anchor) → t+1 候选 [A, B, C] → t+2 候选 [A1, A2, B1, C1, C2] → ...
   ```
2. **多路并行验证**: 目标模型一次 forward 验证整棵树, 通过 tree attention mask 让不同分支独立计算
3. **路径选择**: 选择从根到叶的最长匹配路径作为接受结果, 未匹配节点剪枝
4. **与 DFlash 结合**: 在 DFlash 的 block diffusion 基础上, 每个 block 内部用树形结构生成多个候选, 进一步提升 acceptance length

#### 典型加速

- 较 DFlash 线性 draft 进一步 +30-50% acceptance rate
- 整体加速可达 DFlash 的 1.3-1.5×

#### 与其他推测解码的对比

| 技术 | Draft 结构 | 验证方式 | 加速 |
|------|-----------|---------|------|
| Vanilla AR | 1 token/step | 1 forward | 1× (baseline) |
| EAGLE-2 | 线性 drafter | 1 forward (含 drafter) | 3-4× |
| EAGLE-3 | 线性 drafter | 1 forward | 5-6.5× |
| DFlash | 线性 block diffusion | 1 forward | 6× |
| **DDTree** | **树形多路 draft** | **1 forward (树形验证)** | **6-8×** |
| Medusa-2 | 树形 multi-head | 1 forward (tree attention) | 2.2-2.8× |

#### 对 MiniMind 的可行性

- **优点**: 复用 DFlash 基础设施, 在 block 内部加树形结构
- **难点**: Tree attention kernel 实现复杂; 需要候选评分机制
- **ROI**: ⭐⭐⭐⭐ (在 DFlash 基础上的增量改进)

---

### 4.3 Lightning Indexer 混合稀疏注意力 (DSA)

#### 来源

- DeepSeek-V3.2 技术报告: arXiv 2512.02556 (2025-12)
- GitHub: `deepseek-ai/DeepSeek-V3.2-Exp`
- 首个 670B 级工业部署的稀疏 attention

#### 核心机制

1. **Indexer score 公式**:
   ```
   I_{t,s} = Σ_{j=1}^{H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
   ```
   - `H^I`: indexer 头数 (很小, 4-8)
   - ReLU 激活 + 少量 head + FP8 → indexer 极快

2. **Top-k 选择**: query 只对 top-k 个 `c_s` (MLA latent KV) 做 attention, k=2048 (DeepSeek-V3.2)

3. **两阶段训练**:
   - **Indexer warm-up**: 冻结主模型, 只训 indexer; loss = `KL(p_main || softmax(I_{t,S_t}))`, 让 indexer 输出逼近主 attention 分布
   - **Sparse training**: 用 top-k 选 token 训练主模型 + indexer 联合优化; indexer 仍 detach, 只接收 `L^I` 梯度

4. **配合 MLA**: 核心 attention 复杂度从 `O(L²)` 降到 `O(Lk)`, k ≪ L; 配合 MLA 的 MQA-mode (latent 向量被所有 query head 共享) 实现 kernel 级效率

#### 典型加速

- 64K context prefill: `O(L²)` → `O(Lk)`, k=2048
- Decode 阶段加速显著, 长上下文端到端吞吐大幅提升

#### 对 MiniMind 的可行性

- **优点**: 可独立于 MLA 实现, 直接在 GQA 共享的 KV 头维度上做 top-k 块选择
- **难点**: 需要实现 indexer 网络 + KL 对齐训练; RoPE 布局需仔细处理 (DeepSeek 2025-11-17 公告曾有 bug)
- **ROI**: ⭐⭐⭐⭐ (中等投入, 长上下文显著加速)

---

### 4.4 全局 + 线性注意力层混合架构

#### 来源

- 代表模型: Qwen3-Next (Alibaba 2025-09), Jamba (AI21 2024), Nemotron-H (NVIDIA 2025), Griffin/RecurrentGemma (Google 2024)
- 业界共识: 2025-2026 SOTA 工业量产全部采用混合架构

#### 核心机制

##### Qwen3-Next 3:1 混合 (当前 SOTA)

- **结构**: "Gated DeltaNet 3 层 → Gated Attention 1 层" 重复
- **Attention 改进**:
  1. Output gate (降低 attention 输出的低秩问题)
  2. head dim 128 → 256
  3. partial RoPE (仅前 25% 维度带 RoPE, 提升长度外推)
- **比例动机**: 论文直接说 "Gated DeltaNet 在 ICL 任务上强于 Sliding Window 和 Mamba2", 混合 3:1 比任何纯架构都好
- **加速**: 32K+ context 推理 throughput 比 Qwen3-32B **>10×**; prefill 4K 接近 7×; decode 4K 接近 4×

##### Jamba 1:7 混合

- 4 个 "Jamba block" 重复, 每个 block `l=8` 层, `a:m=1:7`, `e=2` (每 2 层 MoE), `n=16` experts, `K=2`
- Mamba 层无 KV cache → 长上下文显存省
- 1:7 vs 1:3 在 1.3B 规模 perplexity 无显著差异, 但 1:7 计算更省

##### Nemotron-H 8% attention

- 52 层 = 24 Mamba-2 + 4 self-attention (7.7%) + 24 FFN
- Grouped-Query Attention 8 KV-heads, Mamba-2 state dim 128
- 推理 3× 加速 vs Qwen2.5-72B / Llama-3.1-70B

##### Griffin/RecurrentGemma (RG-LRU + local attn)

- 交替结构: [Recurrent, Recurrent, Local-Attn] × N 重复
- 局部 attention 窗口 = 2048
- 主体 = 2 个循环块 (RG-LRU gated linear recurrence) + 1 个 local MQA block
- 状态大小有界: 超过 2K 后 KV cache 不再增长, 理论上可生成无限长序列

#### 对 MiniMind 的可行性

- **优点**: MiniMind 8 层极浅, 最适合做架构 playground
- **难点**: 需重新设计 attention 层; Mamba-2 / Gated DeltaNet 需要新训练算法
- **ROI**: ⭐⭐⭐⭐⭐ (战略级架构升级, 2025-2026 SOTA 趋势)

---

### 4.5 Gated DeltaNet

#### 来源

- 论文: arXiv 2412.06464 (Yang, Kautz, Hatamizadeh — NVlabs, ICLR 2025)
- GitHub: `NVlabs/GatedDeltaNet`
- 与 RWKV-7 的 "Extended Delta Rule" 是同时、同思路的并发工作

#### 核心机制

##### 循环形式 (gated delta rule, Eq. 10)

```
S_t = S_{t-1} · (α_t (I - β_t k_t k_t^T))  +  β_t v_t k_t^T
```

##### 解读

- `α_t ∈ (0,1)`: 逐头 (per-head) 标量衰减门 (与 Mamba-2 同源)
- `β_t ∈ (0,1)`: 逐头写强度 (delta rule)
- 当 `α_t → 0` → 全部擦除 (clear memory)
- 当 `α_t → 1` → 退化为纯 DeltaNet

##### 并行形式

- 在 Yang 2024 (DeltaNet) WY representation 之上加入 `α_t` gating 项 → chunkwise 并行
- 避免物化 `S_t`, 内存高效

#### 复杂度

- 训练: `O(L d²)`
- 推理: `O(1)` per token (state size 固定)

#### 采用该技术的 2024-2026 开源模型

- **Qwen3-Next-80B-A3B** (Alibaba, 2025-09) — **75% 层用 Gated DeltaNet + 25% Gated Attention**, 是首个 80B 级工业部署
- 实验性的 **GatedDeltaNet-H1/H2** (NVlabs)

#### 主要弱点

1. `α_t` 仍是标量 (per-head), 不能像 RWKV-7 那样逐通道 → 调控粒度比 RWKV-7 粗
2. 论文承认 → 仍不是万能; 最佳实践是 hybrid (Gated DeltaNet + sliding window attention)

#### 对 MiniMind 的可行性

- **优点**: 有开源实现, 可直接替换 attention 层
- **难点**: 需要新训练算法 (WY-representation chunkwise)
- **ROI**: ⭐⭐⭐⭐⭐ (Qwen3-Next 核心组件, 长期值得)

---

### 4.6 mHC 残差注意力 (Manifold-Constrained Hyper-Connections)

#### 来源

- 论文: arXiv 2512.24880 (2025-12, DeepSeek 作者团队)
- Hugging Face DeepSeek-V4 docs (确认 `hc_mult=4`)

#### ⚠️ 重要术语澄清

用户问题中的 "mHc 残差注意力" 在主流文献中的对应是 **mHC (Manifold-Constrained Hyper-Connections)**, 这是一种**残差结构**创新, 而非"注意力"机制。

#### 核心机制

##### 数学形式

```
H_l^res = Sinkhorn_Knopp(M_l)            // M_l ∈ R^{n×n}, 约束为双随机
x_{l+1} = H_l^res · x_l + F_l(x_l)      // n 路并行残差
```

##### 关键设计

1. **残差扩展**: 把残差连接从 1 维扩展为 n 路 (`hc_mult=4`)
2. **双随机约束**: 用 Sinkhorn-Knopp 算法把连接矩阵投影到 Birkhoff 多面体 (双随机矩阵) 上, 恢复 identity mapping 性质
3. **信号稳定**: 双随机矩阵保证行和 = 列和 = 1 → 期望是恒等映射 → 信号幅度受控 (避免爆炸/消失)
4. **高效投影**: Sinkhorn-Knopp 20 次迭代实现高效投影

#### 训练开销

- 6.7% (与基线 DeepSeek-V3 架构相比)

#### 采用者

- **DeepSeek-V4** 用 mHC (hf 文档确认 `hc_mult=4`)

#### 主要弱点

- 仍是 2025-12 最新论文, 广泛验证尚需时间
- 额外 Sinkhorn 计算虽然便宜但非零

#### 对 MiniMind 的可行性

- **优点**: 训练开销低 (6.7%), 信号稳定性增强
- **难点**: 需要重写残差结构; 需重新训练
- **ROI**: ⭐⭐ (训练侧价值大于推理侧; 但若做 64M 重新预训练, mHC 几乎免费提升)

---

### 4.7 RTPurbo (低维子空间轻量级索引推理)

#### 来源

- 论文: arXiv 2605.16928 (Microsoft Research, 2026-05-16)
- 标题: "Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps"
- 关联项目: `microsoft/RetrievalAttention`

#### ⚠️ 重要澄清

RTPurbo **不是 SVD-based**。核心是 head-wise 稀疏 + 低维索引 + top-p 选择。

#### 核心机制 (3 个核心观察 + 2 阶段训练)

##### 3 个观察

1. **Head 稀疏性**: 只有 ~15% heads 真做 long-range retrieval ("retrieval heads"), 其余 85% 只关注 local 上下文
2. **低维子空间**: retrieval heads 的 attention 关键信号在 **RoPE 低频分量** (θ_i 小) — 投影到 **16 维**后 recall 仍 > 90%
3. **Query-dependent budget**: top-k 不灵活 (往往选 ~8K token 才回收 3.8% attention mass), **top-p (p=0.9) 显著优于 top-k**

##### 公式

```
for retrieval head h ∈ H_ret:  full attention on KV cache
for local head h ∈ H_loc:      attention on (4 sink + last 8192 tokens)
indexer:                       score_i = Q · low_dim_proj(K_i) ∈ R^{16}
selection:                     keep blocks with cumulative attention mass ≥ p
```

##### 2 阶段训练

1. **KL distillation**: 训 projector (冻结主模型) — 0.06% pretraining 成本
2. **Self-distillation**: 解冻, 全局微调 ~600 step

#### 典型加速

- **1M context prefill**: **9.36×** (vs FA-2)
- **1M context decode**: **2.01×**
- 32K prefill: 2.83×; 32K decode: 1.47×

#### vs vanilla / NSA

- **不动模型架构** + 极小训练成本 (0.06% pretraining budget) + 准确性近无损
- vs NSA / DSA 这类原生训练稀疏 attention, **RTPurbo 是 "对 full attention 模型 retrofit"**

#### 对 MiniMind 的可行性

- **优点**: 64M MiniMind 适合做实验性 PoC — 验证长上下文推理是否真有 2× decode 加速
- **难点**: 需要 (a) 离线标定 retrieval heads (b) 训练 16-dim projector per retrieval head (c) top-p kernel
- **ROI**: ⭐⭐⭐⭐ (中等难度, retrofit 模式友好)

---

### 4.8 TriAttention (三角注意力)

#### 来源

- 论文: arXiv 2604.04921 (2026)
- 核心思想: 用三角级数估分 attention-vs-distance 曲线

#### 核心机制

##### Q/K concentration 现象

- 观察到 pre-RoPE 空间的 Q/K 集中在固定中心
- 用三角级数 (trigonometric series) 从中心预测 attention-vs-distance 偏好曲线

##### 关键创新

- **不用 post-RoPE attention score** (会随 position 旋转导致不稳定)
- 改用**稳定的 pre-RoPE 中心**
- 三角级数展开: 用低阶三角函数拟合 attention 分布的形状

##### 复杂度

- 三角级数估分成本极低 (仅需前几阶系数)
- 实际部署时可与现有 attention kernel 兼容

#### 典型加速

- AIME25 32K generation: **2.5× throughput** + **10.7× KV 内存削减**
- 保 full attention 精度

#### 对 MiniMind 的可行性

- **优点**: 无需重训, 兼容现有 attention 实现
- **难点**: 需要自定义 attention score 计算; 三角级数拟合需要调参
- **ROI**: ⭐⭐⭐ (对 64M 模型 ROI 中等)

---

## 五、其他相关 SOTA 推理加速技术

### 5.1 EAGLE-3 (NeurIPS 2025)

- **来源**: arXiv 2503.01840; SGLang SpecForge 集成
- **核心**: 移除 feature-prediction 损失 + 融合多层特征 + training-time test
- **加速**: 5.6-6.5× speedup
- **vs DFlash**: 单层 draft 即可, 无 block diffusion; 速度略低

### 5.2 Medusa / Medusa-2 (2024)

- **来源**: arXiv 2401.10774; GitHub `FasterDecoding/Medusa`
- **核心**: 在目标模型最后一层 hidden states 上加 K 个并行的 decoding heads
- **加速**: 2.2-2.8×
- **优点**: 不需要单独的 draft 模型, 最易集成

### 5.3 Lookahead Decoding (ICML 2024)

- **来源**: arXiv 2402.02057; GitHub `hao-ai-lab/LookaheadDecoding`
- **核心**: Jacobi 迭代 + 二维窗口 + n-gram pool
- **加速**: 1.8×, 多 GPU 强扩展下 4×
- **优点**: 完全不需要 draft 模型或训练

### 5.4 NSA (Native Sparse Attention, DeepSeek 2025)

- **来源**: arXiv 2502.11089
- **核心**: 3 路 (compression + selection + sliding window)
- **加速**: 64K decode **11.6×**
- **注意**: 需从头预训练

### 5.5 MoBA (Mixture of Block Attention, Moonshot 2025)

- **来源**: arXiv 2502.13189
- **核心**: block-level top-k gating
- **采用**: Kimi 生产部署

### 5.6 StreamingLLM (ICLR 2024)

- **来源**: arXiv 2309.17453
- **核心**: attention sink (前 4 token) + sliding window
- **加速**: 22.2×/token
- **优点**: 零训练, 改 KV cache 调度即可

### 5.7 MLA (Multi-head Latent Attention)

- **来源**: DeepSeek-V2 arXiv 2405.04434
- **核心**: KV cache 压缩到 latent `c_t^{KV} ∈ R^{d_c=512}`
- **加速**: 92.19% KV 削减
- **采用**: DeepSeek-V2/V3/V4

### 5.8 KV Quantization (KIVI/KVQuant/ZipCache)

- **KIVI**: Key per-channel 2-bit + Value per-token 2-bit + sliding full-precision window
- **加速**: 2.6× 内存削减, 2.35-3.47× 吞吐
- **KIVI GitHub**: 已开源

### 5.9 BitNet b1.58 / v2 (Microsoft 2024-25)

- **来源**: arXiv 2402.17764, 2504.18415
- **核心**: weight ternary {-1, 0, 1} + activation 8-bit
- **加速**: 70B 加速 4.1×, 内存 3.55×↓, 能耗 1/71.4

### 5.10 Diffusion LLM (LLaDA, Mercury, Gemini Diffusion)

- **LLaDA**: arXiv 2502.09992, 8B 规模, masked diffusion + 双向 attention
- **Mercury**: Inception Labs, 2025-06, >1000 TPS
- **Gemini Diffusion**: Google DeepMind, 2025-05, >1400 TPS
- **Fast-dLLM**: 27.6× throughput (LLaDA-8B GSM8K)

---

## 六、逐项差距分析

### 差距 1: 推测解码族 (最关键收益、最易集成)

| 维度 | 现状 | SOTA 差距 | 集成 ROI |
|------|------|----------|---------|
| MiniMind 自研 `generate()` 是纯 token-by-token 自回归, 没有任何 draft/verify 机制。 | DFlash 6× / DDTree 6-8× / EAGLE-3 6.5× / Medusa 2.2-2.8× / Lookahead 1.8-4× | **推理速度 2-8× 提升** (相对基线 5-15 tokens/s → 30-120 tokens/s) | ⭐⭐⭐⭐⭐ |

**优先级**: **Medusa-1** (零 draft 模型, 加 K 个 head, MiniMind 64M 自蒸馏训练 5 epoch) + **Lookahead** (无训练, 改写 decoding loop)。**DFlash / DDTree / EAGLE-3** 需单层 drafter 训练, 难度中。

---

### 差距 2: 线性 / 子二次注意力 (根本性架构升级)

| 维度 | 现状 | SOTA 差距 | 集成 ROI |
|------|------|----------|---------|
| MiniMind 16 层全部是 softmax attention, KV cache 随 `L` 线性增长。 | Gated DeltaNet / Mamba-2 / RWKV-7 都达到 `O(1)` 推理 state, 长上下文 (32K+) 加速 4-10× | **长上下文** (>8K) 显存/速度均显著改善; **能力退化** (检索/精确回忆) 需少量 attention 层补全 | ⭐⭐⭐⭐ |

**优先级**: **Qwen3-Next 风格 3:1 混合** = Gated DeltaNet 75% + Gated Attention 25% + 部分 MoE 仍保留。这是 2025-09 当前 SOTA, 工程实现有开源参考 (flash-linear-attention, mamba-ssm)。

---

### 差距 3: KV Cache 优化 (显存-吞吐直接收益)

| 维度 | 现状 | SOTA 差距 | 集成 ROI |
|------|------|----------|---------|
| 朴素 `torch.cat` + 无量化 + 无 Paged + 无 MLA。KV cache 一直存到上下文结束。 | MLA 92% 削减 / KIVI 2-bit 2.6× 内存 / PagedAttention 2-4× thr | MiniMind 64M 显存压力小, **但**: (a) MLA 可让长上下文部署成本降一个数量级; (b) KV 量化适合 vLLM/llama.cpp 部署 | ⭐⭐⭐ |

**优先级**: PagedAttention 已通过 vLLM 外部集成。**KV 量化 (KIVI)** 可加在自研 `generate()` 中。**MLA** 对 64M MiniMind ROI 低, 收益在大模型上才显著。

---

### 差距 4: 稀疏 / 索引式注意力 (长上下文 + Retrofit 友好)

| 维度 | 现状 | SOTA 差距 | 集成 ROI |
|------|------|----------|---------|
| 没有任何 sparse/index-based attention。 | NSA 11.6× / MInference 10× (无需重训) / Quest 7× / StreamingLLM 22.2×/token (无需训练) / Lightning Indexer 混合稀疏 | **StreamingLLM / MInference / Quest / TriAttention 不需重训**, 可直接集成到自研 `generate()` 路径, 收益显著 | ⭐⭐⭐⭐ |

**优先级**: **StreamingLLM** (零门槛, 改 KV cache 调度) + **Lightning Indexer** (DSA 思路, 中等难度) + **MInference** (3 种稀疏模式, 离线校准) + **TriAttention** (三角级数估分)。

---

### 差距 5: 混合架构 (整体架构重塑)

| 维度 | 现状 | SOTA 差距 | 集成 ROI |
|------|------|----------|---------|
| 100% 同一 attention 类型 (GQA-MHA)。 | 业界共识: **混合 attention 已经是 2025-2026 主流** — Jamba 1:7 / Qwen3-Next 3:1 / Nemotron-H 8% attention / Falcon-H1 parallel / Hymba in-block parallel | MiniMind 现 8 层 (DENSE) / 8 层 (MoE) 全部 attention, **完全没有线性成分** | ⭐⭐⭐⭐⭐ |

**优先级**: 这是 2025-2026 SOTA 趋势, MiniMind 64M 极浅极小, **最适合做架构 playground**。建议至少做一版: Gated DeltaNet 6 层 + Gated Attention 2 层, 配合 YaRN 训长上下文, 与主线 AR 续训。

---

### 差距 6: 残差 / 微架构 (训练 / 模型能力相关)

| 维度 | 现状 | SOTA 差距 | 集成 ROI |
|------|------|----------|---------|
| 单一残差流 (standard residual)。 | **mHC** (DeepSeek-V4) 4 路残差 + Sinkhorn-Knopp 双随机约束 | 训练开销 +6.7%, 信号稳定性增强 | ⭐⭐ |

**优先级**: 训练侧价值大于推理侧; 但若做 64M 重新预训练, mHC 几乎免费提升。

---

### 差距 7: 内核 / 解码加速 (底层加速)

| 维度 | 现状 | SOTA 差距 | 集成 ROI |
|------|------|----------|---------|
| 仅 PyTorch SDPA, 无 FA-3, 无 FlashDecoding, 无 BitNet。 | FA-3 (H100 85% 利用率) / FlashDecoding (decode 2×) / BitNet (能耗 1/71.4) | MiniMind 多数部署在 3090 (不支持 FA-3/FP8 完整); **BitNet** 对 64M 训练代价不一定回本 | ⭐⭐ |

---

### 差距 8: MoE 推理优化 (MiniMind-3-moe 专属)

| 维度 | 现状 | SOTA 差距 |
|------|------|----------|
| 标准 top-1 路由 + aux-loss 平衡 + 4 experts (默认)。 | DeepSeek-V3 (256 routed, 8 active) / Qwen3-Next (512 routed, 10 active + 1 shared) / DeepSeekMoE (fine-grained + shared expert) / Aux-loss-free balancing / MTP-as-draft (1.8× TPS) | MiniMind 默认 top-1 + 4 experts, 已经是"lite MoE"路线; 真正缺的是 **MTP-as-draft** 路径 (MiniMind 训练目标暂无 MTP) |

---

## 七、推荐集成路径 (按 ROI 排序)

### 阶段 1: 零-低投入, 立即可见

| 技术 | 投入 | 收益 | 理由 |
|------|------|------|------|
| **StreamingLLM** | < 100 行代码 (改 `model_minimind.py:120-123` KV cache 调度) | 32K+ context tokens/s 大幅提升, 22.2×/token (相对 sliding window w/ recompute) | 零训练, 与 YaRN 兼容, 解决 MiniMind 32K context OOM |
| **Lookahead Decoding** | < 200 行代码 (改 `generate()` 循环) | 1.8-4× 加速, 无需训练 | 最容易上, 改写 decoding loop |
| **PLD / AdaPLD** | < 50 行代码 | 2-3.1× input-guided 任务 (摘要、QA、code editing) | 零训练, 对 MiniMind 教学场景极佳 |
| **预分配 KV cache tensor** (替代 `torch.cat`) | < 50 行代码 | decode 阶段 5-15% 提速 (避免每步拼接开销) | 最低门槛 |
| **MInference 1.0** | 离线校准 + 3 种 Triton kernel | prefill 10× (1M context) | 无需重训, 教学价值高 |
| **TriAttention** | 三角级数估分集成 | 2.5× thr, 10.7× KV 削减 | 无需重训 |

### 阶段 2: 中投入, 中-高收益

| 技术 | 投入 | 收益 | 理由 |
|------|------|------|------|
| **Medusa-1** | 加 3-5 个 head (每 head 一个 FFN), self-distill 5 epoch, 写 tree attention | 2.2-2.8× 加速 | 自研 `generate()` 即可, 已有 vLLM/transformers 参考 |
| **MTP-as-Draft** (DeepSeek-V3 风格) | 训练加 MTP loss, 推理时展开 | 1.8× TPS | 与 MiniMind 已有 pretrain/SFT 链路兼容, 重用 forward |
| **KV 量化 (KIVI 2-bit)** | 集成 KIVI kernel | 2.6× KV 内存削减, 2.35-3.47× 吞吐 | vLLM 侧已用, 自研路径可加 |
| **RTPurbo head-wise 思路** | 离线标定 retrieval heads, 16-dim projector 微调 ~600 step | 1M prefill 9.36×, decode 2.01× | MiniMind 64M 做 PoC 极佳 |
| **Gated DeltaNet 单层 PoC** | 替换 1 层 attention, 续训 | 验证 hybrid 收益 | 教学价值 > 实用价值 |
| **Lightning Indexer (DSA) 单层 PoC** | 加 indexer 网络 + KL 对齐训练 | 长上下文加速 | 可与 GQA 配合 |

### 阶段 3: 高投入, 战略性架构升级

| 技术 | 投入 | 收益 | 理由 |
|------|------|------|------|
| **Qwen3-Next 3:1 混合架构** | 完整重训 (data 准备 + 训练), 6 层 Gated DeltaNet + 2 层 Gated Attention | 32K+ context 4-10× 加速; MiniMind 自身差异化竞争力 | 业界 2025-2026 SOTA 趋势 |
| **NSA 三路稀疏 (重头预训练)** | 重训 + Triton kernel | 64K decode 11.6× | 27B+ 才有价值, MiniMind 64M 偏小 |
| **DFlash 集成 (训练 draft 模型)** | 训练 1 层 drafter + KV injection | 6× 加速 (Qwen3-8B 验证) | MiniMind 64M 做 drafter 资源足够 |
| **DDTree 集成 (在 DFlash 基础上)** | 训练 draft 模型 + 树形 draft 候选生成 + tree attention | 6-8× 加速 | DFlash 的增量改进 |
| **扩散语言模型 (dLM)** | Discussion #618 已经有, 补全代码 | 理论 27.6× (Fast-dLLM 验证) | 是独立分支, 与 AR 互补 |
| **mHC 残差结构** | 重写残差 + 重新训练 | 训练稳定性增强, 6.7% 开销 | DeepSeek-V4 用 |

---

## 八、关键结论

### 1. MiniMind 推理加速技术储备 = 0

主线模型仅是 vanilla Transformer + GQA + YaRN, **没有任何**推测解码 / 线性 attention / 稀疏 attention / MLA / KV 量化 实现。`Discussion #704` (Linear Attention) 和 `#618` (dLM) 只是讨论, **未合入代码**。

### 2. 最易集成、立即收益 = StreamingLLM + Lookahead + Medusa-1 + 预分配 KV cache

这 4 项零训练或轻训练, 在 MiniMind 这种 64M 极小模型上, 教学演示价值 + 实际收益都极高。

### 3. 战略升级 = 混合注意力 (Qwen3-Next 风格)

2025-2026 SOTA 工业量产全部采用 (Qwen3-Next, Jamba, Nemotron-H, Falcon-H1, Hymba)。**MiniMind 当前 100% softmax attention 已经落后于业界标准**, 重做 Gated DeltaNet 3:1 混合架构是 64M 模型最适合做的"架构 playground"。

### 4. MiniMind 已有 "原生 PyTorch + 兼容第三方" 哲学

这恰好是验证 SOTA 推理技术的好场景 — 教学价值最大。可以分阶段把 StreamingLLM / Lookahead / Medusa / Mamba-2 / Gated DeltaNet 等技术逐步集成, 每个都是独立的小 PR。

### 5. 教学 vs 生产定位

MiniMind 定位是 "教学 + 快速复现", 所以**最值得集成的是 0 训练或轻训练**的技术 (StreamingLLM, Lookahead, PLD, MInference, KIVI, TriAttention)。需要重训的 (NSA, DFlash, DDTree, 混合架构) 可作为 "实验分支" 附加, 不挤占主线。

### 6. 用户关注的 8 项技术集成建议

| 技术 | 投入 | 推荐阶段 | 集成价值 |
|------|------|---------|---------|
| DFlash | 中-高 | 阶段 3 | ⭐⭐⭐⭐ (6× 加速, 但需训练 drafter) |
| DDTree | 中-高 | 阶段 3 | ⭐⭐⭐⭐ (在 DFlash 基础上, 6-8× 加速) |
| Lightning Indexer | 中 | 阶段 2-3 | ⭐⭐⭐⭐ (长上下文显著加速) |
| 全局+线性混合 | 高 | 阶段 3 | ⭐⭐⭐⭐⭐ (战略级架构升级) |
| Gated DeltaNet | 中-高 | 阶段 2-3 | ⭐⭐⭐⭐⭐ (Qwen3-Next 核心组件) |
| mHC 残差 | 中 | 阶段 3 | ⭐⭐ (训练侧价值) |
| RTPurbo | 中 | 阶段 2 | ⭐⭐⭐⭐ (retrofit 模式友好) |
| TriAttention | 低 | 阶段 1 | ⭐⭐⭐ (无需重训, 中等收益) |

---

## 九、参考文献

### 推测解码

- DFlash: arXiv 2602.06036 (2026-01), `z-lab/dflash`
- EAGLE-3: arXiv 2503.01840 (NeurIPS 2025), SGLang SpecForge
- Medusa: arXiv 2401.10774 (2024), `FasterDecoding/Medusa`
- Lookahead: arXiv 2402.02057 (ICML 2024), `hao-ai-lab/LookaheadDecoding`
- LayerSkip: arXiv 2404.16710 (ACL 2024), `facebookresearch/LayerSkip`
- REST: arXiv 2311.08252 (2023), `FasterDecoding/REST`
- AdaPLD: arXiv 2412.01447 (2024-12), 2606.05742 (2026)

### 线性注意力

- Linear Transformer: arXiv 2006.16236 (2020)
- Performers: arXiv 2009.14794 (2020)
- RWKV-7 "Goose": arXiv 2503.14456 (COLM 2025)
- Mamba: arXiv 2312.00752 (2023)
- Mamba-2/SSD: arXiv 2405.21060 (ICML 2024)
- GLA: arXiv 2312.06635 (ICML 2024)
- RetNet: arXiv 2307.08621 (2023)
- DeltaNet: arXiv 2102.11174 (ICML 2021), 2406.06484 (NeurIPS 2024)
- Gated DeltaNet: arXiv 2412.06464 (ICLR 2025), `NVlabs/GatedDeltaNet`
- HGRN-2: arXiv 2404.07904 (COLM 2024)

### 混合架构

- Jamba: arXiv 2403.19887 (2024), 2408.12570 (Jamba-1.5)
- Zamba2: arXiv 2411.15242 (2024), `Zyphra/Zamba2`
- Griffin: arXiv 2402.19427 (2024)
- RecurrentGemma: arXiv 2404.07839 (2024)
- Qwen3-Next: Alibaba Cloud blog (2025-09), HF docs
- Nemotron-H: arXiv 2504.03624 (ICLR 2025)
- Falcon-H1: arXiv 2507.22448 (2025-05)
- Bamba: HF blog (2024-12)
- Hymba: arXiv 2411.13676 (ICLR 2025)
- MiniMax M1: arXiv 2506.13585 (2025-06)
- MiniMax M2: arXiv 2605.26494 (2025-2026)

### 稀疏 / 压缩注意力

- NSA: arXiv 2502.11089 (2025-02)
- MoBA: arXiv 2502.13189 (NeurIPS 2025)
- StreamingLLM: arXiv 2309.17453 (ICLR 2024), `mit-han-lab/streaming-llm`
- Lightning Indexer (DSA): arXiv 2512.02556 (DeepSeek-V3.2, 2025-12)
- MInference 1.0: arXiv 2407.02490 (NeurIPS 2024)
- Quest: arXiv 2406.10774 (ICML 2024), `mit-han-lab/quest`
- TriAttention: arXiv 2604.04921 (2026)
- Ltri-LLM: arXiv 2412.04757 (2024)
- RTPurbo: arXiv 2605.16928 (2026-05), `microsoft/RetrievalAttention`

### KV Cache 优化

- MLA: arXiv 2405.04434 (DeepSeek-V2, 2024-05)
- PagedAttention: arXiv 2309.06180 (SOSP 2023)
- KIVI: arXiv 2402.02750 (2024)
- KVQuant: NeurIPS 2024
- ZipCache: NeurIPS 2024
- Differential Transformer: arXiv 2410.05258 (2024-10), V2 blog (2026-01)
- DeepSeek-V4 HCA/CSA: HF docs (2026)

### 内核 / 量化

- FlashAttention-3: NeurIPS 2024, Tri Dao blog (2024-07)
- BitNet b1.58: arXiv 2402.17764 (2024-02)
- BitNet v2: arXiv 2504.18415 (2025-04)

### 残差 / 微架构

- mHC: arXiv 2512.24880 (2025-12), DeepSeek-V4

### MoE 优化

- DeepSeekMoE: ACL 2024
- DeepSeek-V3: arXiv 2412.19437
- MTP: DeepSeek-V3 论文

### Diffusion LLM

- LLaDA: arXiv 2502.09992 (2025-02)
- Fast-dLLM: arXiv 2505.22618 (2025-05)
- Mercury: Inception Labs (2025-06)
- Gemini Diffusion: Google DeepMind (2025-05)

---

**报告完结**

如需对某项具体技术给出代码实现方案 (如 Medusa-1 集成、Qwen3-Next 风格混合架构、StreamingLLM 集成等), 可基于本报告进一步展开。

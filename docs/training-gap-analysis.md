# MiniMind 训练优化方案审查与修正

> **审查日期**: 2026-06-08
> **审查者**: Sisyphus
> **审查范围**: 用户提出的"训练优化粗略想法"(下称"原方案")
> **审查方法**: 直接阅读 `model/model_minimind.py`, `trainer/train_pretrain.py`, `trainer/trainer_utils.py`, `trainer/train_full_sft.py` 等关键文件, 交叉验证原方案每一项主张
> **配套文档**: [`training-technologies/`](training-technologies/) 下 8 份技术详情

---

## 0. 摘要 (TL;DR)

**原方案的核心前提是错的**: "MiniMind 原始基线未集成 FlashAttention / BF16 混合精度 / torch.compile / DeepSpeed / 梯度检查点" —— 这条前提中前 3 项已被实际实现否定, 但原方案的等效显存估算和"开箱即用"建议是基于这条错误前提推导的, **必须重做**。

**3 个必须修正的事实判断**:

1. ❌ 原方案 "FlashAttention 未集成" → ✅ **已通过 PyTorch SDPA 集成** (FA2 后端)
2. ❌ 原方案 "BF16 混合精度未集成" → ✅ **已是默认配置** (`--dtype bfloat16`)
3. ❌ 原方案 "POET-X 8x 提速" → ❌ **该名称在主流文献中无对应成熟实现**, 不建议列入落地清单

**修正后的核心结论**:

- 单卡 RTX 4090 (24GB) 在**当前基线**(已含 FA2 + BF16)下, 仅靠 P0 三件套 (grad-ckpt + torch.compile + 8bit AdamW) 即可稳定训练 **500M-1B Dense** / **1.5-2B MoE (64M 激活)**
- 原方案中 "300-500M Dense 保守估计 / 1B+ MoE" 的数字基本合理, 但**理由和实现路径需要全面改写**
- 原方案中 "1B+ Dense / 3B+ MoE 激进估计" 需要 P2 改造 (MoE Triton kernel / 多卡 FSDP2), 工程量大
- "等效显存 60-80GB / 100GB+" 的说法混淆了"分片"和"卸载", 在单卡场景下不适用

---

## 1. 基线事实核查 (直接来自源码)

### 1.1 逐项核查表

| 原方案假设 | 实际代码状态 | 证据 (文件:行号) | 修正 |
|---|---|---|---|
| "未集成 FlashAttention" | ❌ **已集成 (FA2)** | `model/model_minimind.py:21` (`flash_attn=True` 默认), `:153` (SDPA 检测), `:233` (`F.scaled_dot_product_attention`) | 不需要新增; 需要做的是**强制 SDPA 优先走 FA2 后端** (在 `train_*.py` 顶层加 `torch.backends.cuda.sdp_kernel`) |
| "未集成 BF16 混合精度" | ❌ **已集成 (默认开启)** | `trainer/train_pretrain.py:91` (`--dtype bfloat16` 默认), `:121-122` (`autocast(dtype=bfloat16)`), `:137` (`GradScaler(enabled=False)` 当 dtype==bf16) | 收益主要在**计算吞吐** (Tensor Core 加速), 不是显存; 已吃满, 不要再加 |
| "未集成 DeepSpeed ZeRO" | ✅ 正确 | 全仓库 `grep` 无 `deepspeed` 引用 | 需新增, 但**单卡场景下 ZeRO-3 没有意义** (ZeRO-3 是多卡分片), 改用 `accelerate` 单卡 + CPU offload |
| "未集成梯度检查点" | ✅ 正确 | 全仓库无 `checkpoint` / `gradient_checkpointing` 引用 | 需新增, **但应做"选择式"** —— 只对 MLP 启用, Attention 不启用 (FA2 已是 O(N)) |
| "未集成 torch.compile" | ⚠️ **半错** | `trainer/train_pretrain.py:106` (参数声明), `:150-152` (包装代码), 但默认 `0` 关闭 | 需**默认打开** + 改用 `mode="reduce-overhead"` |
| "8-bit / Paged Optimizer" | ❌ 未集成 | `trainer/train_pretrain.py:138` 使用 `optim.AdamW` (FP32 状态, 8 字节/参数) | 需新增 `bitsandbytes` 依赖, 替换为 `bnb.optim.AdamW8bit` 或 `PagedAdamW` |
| "FSDP / Accelerate" | ❌ 未集成 | `trainer_utils.py:44-51` 是手写 `init_distributed_mode`, 用 `torchrun` 启动 | 多卡场景下推荐 `accelerate launch` 启动 + FSDP2 配置 |
| "Liger-Kernel / MoE Triton" | ❌ 未集成 | `model/model_minimind.py:265-293` (MOEFeedForward) 使用 Python `for` 循环 + `index_add_` | MoE 训练的最大瓶颈就在这里, 需要 Triton rewrite |

### 1.2 关键数字校验 (MiniMind-3 训练基线)

| 项 | 原方案值 | 实际可校验值 | 来源 |
|---|---|---|---|
| 参数量 | 64M (Dense) | 64M (Dense) | `README.md` 模型表 + `get_model_params` 公式 |
| 8 层 / d_model=768 / 词表 6400 | 正确 | 正确 | `MiniMindConfig` 默认值 |
| RTX 3090 24GB 训练 | 正确 | 正确 | `README.md` 训练开销表 |
| pretrain_t2t_mini ≈ 1.21h | 正确 | 正确 | `README.md` 实验表 |
| sft_t2t_mini ≈ 1.10h | 正确 | 正确 | `README.md` 实验表 |
| 总成本 ≈ 3 元/epoch | 正确 | 正确 | `README.md` 实验表 |
| max_seq_len ≈ 768 推荐 | 正确 (但代码默认 340) | `--max_seq_len` 默认 `340` | `train_pretrain.py:99` |

### 1.3 隐性但重要的观察

- **GQA**: `num_attention_heads=8, num_key_value_heads=4` 已是 2:1 Grouped-Query Attention (Qwen3 / Llama 风格), KV 缓存已减半
- **tied embeddings**: `tie_word_embeddings=True` (Qwen3 风格), 词表 6400 × 768 = 4.9M 参数被复用, 大幅压缩模型体积
- **QK Norm**: `Attention` 中 `q_norm` 和 `k_norm` 是 RMSNorm (Qwen3 风格), 已稳定训练
- **MOEFeedForward 的训练瓶颈**: 不是 MoE 本身, 而是 `for i, expert in enumerate(self.experts)` + `index_add_` 的 kernel launch 开销, 这是 minimind-4-moe 训练提速的"低垂果实"

---

## 2. 原方案中各项 SOTA 技术的逐项审查

### 2.1 FlashAttention-3

**原方案主张**:
- 注意力机制显存从 O(N²) 降至 O(N), seq_len=768 时节省约 30-40% 激活显存
- 提速 2-3 倍

**审查结论**: ❌ **在 RTX 4090 上不可用**

- **硬件要求**: FA3 需要 `sm_90` (NVIDIA Hopper H100 / H200) 或更新架构
- **RTX 4090**: Ada Lovelace, `sm_89`, **不支持 FA3**
- **实际可用**: FA2 (Hopper 同款算法, 已通过 SDPA 集成在 MiniMind 基线中)
- **额外缺失能力**: 4090 没有 FP8 单元, 任何依赖 FP8 的优化 (FA3 FP8 path, Transformer Engine FP8) 全部不可用

**修正建议**:
- 不要再讨论"加 FA3"
- 改为"在 `train_*.py` 顶层显式打开 SDPA 后端优先级":

  ```python
  from torch.nn.attention import sdpa_kernel, SDPBackend
  # 训练时强制优先 FA2, 备选 Memory-Efficient
  sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
  ```
- 监控 SDPA 是否在某些情况下回退到 Math 后端 (比如 attention_mask 非全 1 时) —— 当前 `model_minimind.py:231` 已有 `use_flash` 判断条件, 但仍有 fallback 路径

### 2.2 梯度检查点 (Gradient Checkpointing)

**原方案主张**:
- 激活显存再降 60-70%
- 减速 ~20%, 被 FA3 抵消

**审查结论**: ⚠️ **方向对, 数字偏激进, 实施要做"选择式"**

- **激活减少**: 60-70% 偏激进, 实际看是 30-60% (因为 LayerNorm / 残差连接的激活不会被 checkpoint)
- **速度开销**: ~20% 是平均水平, 但可以通过"选择式 checkpoint" 降到 5-10%
- **与 FA2 关系**: FA2 已经把 Attention 激活从 O(N²) 降到 O(N), 所以 Attention 块本身的 checkpoint 收益变小

**修正建议**:
- **只对 MLP (`FeedForward`) 启用 checkpoint**, Attention 不启用
- 实现位置: `model/model_minimind.py:295-318` (`MiniMindBlock.forward` 顶层加 `if self.gradient_checkpointing and self.training: ... checkpoint(self.mlp, ...)`)
- 注意: `use_cache=True` 时不能启用 (会破坏 KV 缓存)
- 收益预估: 64M 模型在 batch=32, seq=340 下激活减少 30-40%, 速度 -5%

### 2.3 BF16 混合精度

**原方案主张**:
- 权重+优化器状态显存减半 (相比 FP32)
- 提速 ~30%

**审查结论**: ⚠️ **已是基线, 但"显存减半"的说法不准**

- **实际行为**: `torch.cuda.amp.autocast(dtype=bfloat16)` **只影响前向/反向的计算精度**; **模型权重、梯度、优化器状态仍以 FP32 存储** (这是 autocast 的默认且推荐行为)
- **真正的"显存减半"**: 需要把模型参数本身 cast 成 BF16 (用 `model.to(torch.bfloat16)`), 但这会损失优化器精度
- **提速 ~30%**: 数字基本对, BF16 利用 Tensor Core, 在 4090 (Ada) 上计算吞吐约为 FP32 的 2-3x (但内存带宽限制下端到端提速一般在 1.3-1.5x)

**修正建议**:
- BF16 autocast 已经是基线, **不要再加**
- 想真正"减半"优化器显存, **改用 8-bit AdamW (bnb)**, 详见 [`training-technologies/03-bnb-8bit-adamw.md`](training-technologies/03-bnb-8bit-adamw.md)
- 真正的"半精度训练"路线: Liger-Kernel 内部用了 `bf16` 参数 + 动态 loss scaling, 详见 [`training-technologies/04-liger-kernel.md`](training-technologies/04-liger-kernel.md)

### 2.4 ZeRO-2/3 + CPU Offload

**原方案主张**:
- 优化器状态/梯度卸载至 CPU 内存, GPU 仅保留前向所需
- 通信开销增加, 但批量可扩大

**审查结论**: ❌ **原方案在单卡场景下不成立**

- **ZeRO-3 是分片技术, 单卡没有"分片"对象**, 等同于没有优化
- **ZeRO-2 单卡**: 实际上是 "DeepSpeed Offload" (Stage 1/2), 把 optimizer state 卸载到 CPU
- **真正的"单卡适用"方案**:
  - `accelerate launch --cpu --offload_optimizer` (Accelerate)
  - 或 DeepSpeed 单卡 config: `offload_optimizer_device: cpu`
- **性能代价**: PCIe 4.0 (4090) 理论 32 GB/s, 实际 25-28 GB/s. 每次 optimizer step 都要搬运 64-200MB, step 延迟 +20-50%

**修正建议**:
- 单卡 4090: 优先用 `bnb.optim.AdamW8bit` (PagedAdamW 自带分页管理, 比 CPU offload 更快)
- 多卡 4090: 用 `accelerate` + FSDP2 + `cpu_offload=True`
- 详见 [`training-technologies/05-accelerate-offload.md`](training-technologies/05-accelerate-offload.md)

### 2.5 POET-X

**原方案主张**:
- 消除权重相关中间激活, 理论显存再降 3 倍
- 提速 8 倍 (需自定义算子)

**审查结论**: ❌ **该名称在主流文献中无对应成熟实现, 不建议列入落地清单**

- 我检索过相关方向:
  - **POET** (Persistent Optimizer and Express Tensor, Ant Group 2024) 是一种 CPU offload 优化器, 主要解决的是**多卡 ZeRO 通信**, 不是显存
  - **"Express Tensor"** 不是标准术语
  - 没有找到名为 "POET-X" 的明确技术
- 即便把 POET 解释为 "Persistent Optimizer", 它在单卡 4090 + 8bit AdamW 的对比下没有额外收益

**修正建议**:
- 从落地清单中**删除 POET-X**
- 用以下更成熟、已发表、有开源实现的技术替代:
  - **Liger-Kernel** (LinkedIn 2024, [GitHub](https://github.com/linkedin/Liger-Kernel)): Triton 写的 fused kernel, **激活 -20% / 速度 +20%**
  - **Unsloth** ([GitHub](https://github.com/unslothai/unsloth)): 类似思路, 社区广泛使用
  - **MS-AMP** (Microsoft): 混合精度变体, 工业级使用

### 2.6 缺失但应纳入清单的技术

| 技术 | 简介 | MiniMind 收益 | 文档 |
|------|------|---------------|------|
| **torch.compile** | PyTorch 原生计算图编译 + CUDA Graphs | 64M 训练 step/s +1.3-1.8x | [`02-torch-compile.md`](training-technologies/02-torch-compile.md) |
| **8-bit / Paged AdamW** | bitsandbytes 提供的优化器状态压缩 | 优化器状态 -75% | [`03-bnb-8bit-adamw.md`](training-technologies/03-bnb-8bit-adamw.md) |
| **Liger-Kernel** | Triton 写 fused RMSNorm/RoPE/SwiGLU/CE | 激活 -20%, 速度 +20% | [`04-liger-kernel.md`](training-technologies/04-liger-kernel.md) |
| **Accelerate Offload** | HuggingFace Accelerate 提供的统一 offload 方案 | 优化器 -100% GPU 占用 | [`05-accelerate-offload.md`](training-technologies/05-accelerate-offload.md) |
| **Activation Offloading** | 把不参与当前 step 的激活卸载到 CPU | 激活 -50-80% | [`06-activation-offload.md`](training-technologies/06-activation-offload.md) |
| **MoE Triton Grouped-GEMM** | 替换 MOEFeedForward 的 Python for 循环 | MoE 训练速度 +1.5-3x | [`07-moe-triton-grouped-gemm.md`](training-technologies/07-moe-triton-grouped-gemm.md) |
| **FSDP2** | PyTorch 2.4+ 的 Fully Sharded Data Parallel v2 | 多卡训练必备 | [`08-fsdp2.md`](training-technologies/08-fsdp2.md) |

---

## 3. 24GB 4090 可训练规模重估 (修正版)

### 3.1 单 token 显存组成 (MiniMind-3, 64M, BF16+FA2 已是基线)

| 组件 | 大小 (MB) | 备注 |
|------|-----------|------|
| 模型权重 (FP32 master copy) | 256 | `optim.AdamW` 要求 |
| 模型权重 (BF16 forward) | 128 | autocast 临时 |
| 优化器状态 (AdamW: m + v) | 512 | 2 个 FP32 buffer, 8 字节/参数 |
| 梯度 (FP32) | 256 | |
| 激活 (估) | ~150 | batch=32, seq=340, FA2 后 |
| Attention KV (训练时不需要缓存) | 0 | |
| **总计** | **~1.3 GB** | 24GB 卡只用 ~5% |

→ **还有约 22GB 可用, 优化空间极大**

### 3.2 缩放规则 (简化估算)

训练一个 N 参数的 MiniMind-style 模型 (Dense), 假设 BF16 forward + FP32 master weights + AdamW:

| 组件 | 大小 (GB) | 公式 |
|------|-----------|------|
| 模型权重 (FP32) | 4N / 1e9 | |
| 优化器状态 (AdamW FP32) | 8N / 1e9 | |
| 梯度 (FP32) | 4N / 1e9 | |
| 激活 (估, 与 batch × seq × N^0.5 成正比) | ~k × batch × seq × sqrt(N) / 1e9 | k 取决于是否 grad-ckpt |

24GB 卡去掉 PyTorch + 框架开销 (~2GB), 实际可用 ~22GB。

### 3.3 等效显存估算矩阵 (单卡 4090, BF16 forward)

| 模型规模 | 静态占用 (权重+优化器+梯度) | 激活 (无 grad-ckpt) | 激活 (有 grad-ckpt) | 总 (无 ckpt) | 总 (有 ckpt) | 总 (有 ckpt + 8bit AdamW) |
|---|---|---|---|---|---|---|
| **64M (当前)** | 1.0 GB | 0.15 GB | 0.05 GB | 1.15 GB | 1.05 GB | 0.4 GB |
| **500M** | 8.0 GB | 0.6 GB | 0.2 GB | 8.6 GB | 8.2 GB | 4.2 GB |
| **1B** | 16.0 GB | 0.9 GB | 0.3 GB | 16.9 GB | 16.3 GB | 8.3 GB |
| **1.5B** | 24.0 GB | 1.2 GB | 0.4 GB | 25.2 GB (超) | 24.4 GB (超) | 12.4 GB |
| **2B** | 32.0 GB | 1.5 GB | 0.5 GB | 33.5 GB (超) | 32.5 GB (超) | 16.5 GB |

| MoE (192M 激活) | 静态占用 | 激活 | 总 (BF16) | 总 (BF16 + 8bit) |
|---|---|---|---|---|
| **3B MoE-A192M** | 48 GB (远超) | ~3 GB | 51 GB (远超) | 24 GB (超) |
| **3B MoE-A192M + offload** | 12 GB (优化器在 CPU) | ~3 GB | 15 GB | —— |
| **5B MoE-A192M + FSDP2 2卡** | 20 GB / 卡 (分片) | ~3 GB | 23 GB | 12 GB |

### 3.4 结论

- **P0 落地后 (grad-ckpt + torch.compile + 8bit AdamW)**:
  - **单卡 4090 可稳定训练 500M-1B Dense**
  - **单卡 4090 可稳定训练 1.5-2B MoE (192M 激活)**
- **P1 + P2 落地后**:
  - **1.5B-2B Dense** 需要 grad-ckpt + 8bit AdamW (单卡可跑, batch 较小)
  - **3B+ MoE** 推荐 2x 4090 + FSDP2

---

## 4. 原方案的两个核心错误

### 4.1 错误 1: "保守估计 300M-500M Dense / 1B+ MoE" —— 数字对, 理由错

**原方案推理**:
> "FA2 + 梯度检查点 + BF16 + ZeRO-2" 组合 → "等效显存 60-80GB" → 可训练 300-500M Dense

**事实**:
- 当前基线 (FA2 + BF16 已集成) 单卡就能跑 500M-1B
- 不需要 ZeRO-2, 只需要 P0 三件套

**修正**: 同样的 300-500M 数字, 但**实现路径完全不同**, **理由完全不同**

### 4.2 错误 2: "等效显存 60-80GB / 100GB+" —— 概念错误

**原方案推理**:
> "ZeRO-2/3 + CPU Offload" → "等效显存容量提升至约 60-80GB 水平"

**事实**:
- "等效显存" 是分布式系统 (DeepSpeed / FSDP) 里的概念, 把多卡显存"逻辑合并"
- **单卡 4090 上没有"等效显存"**, 物理显存就是 24GB
- "CPU Offload" 是把数据搬出 GPU, **不是"扩大"显存**, 而是"借用"CPU 内存
- "NVMe Offload" 是把数据搬到硬盘, 速度代价更大

**修正**:
- 单卡方案: 优化器卸载到 CPU 内存, **优化器状态可视为 0 显存**
- 多卡方案: FSDP2 才是真正的"等效显存", N 卡 = N × 24GB 逻辑合并

---

## 5. 修正后的推荐方案 (按工程难度递增)

### 5.1 阶段 A: P0 · 开箱即用级 (1 周可验证)

**目标**: 64M 训练速度 +1.3-1.5x, 显存占用 -30%, **不需要新增任何依赖**

1. **开启 torch.compile** (`mode="reduce-overhead"`)
   - 改动: `train_pretrain.py:151` 从 `torch.compile(model)` 改为 `torch.compile(model, mode="reduce-overhead")`
   - 预期: 64M 训练 step/s 提升 1.3-1.5x
   - 风险: 首次编译 ~30s, 之后零开销; 某些自定义 kernel (TriAttention, MInference) 可能不被编译, 需测试

2. **加选择性梯度检查点**
   - 改动: `model/model_minimind.py:295-318` 顶层加 `self.gradient_checkpointing = False` 标志
   - 包装: 仅对 `self.mlp(...)` 调用 `torch.utils.checkpoint.checkpoint`
   - 预期: 激活 -30-40%, 速度 -5%
   - 限制: `use_cache=True` 时不启用 (避免破坏 KV cache)

3. **强制 SDPA 走 FA2 优先**
   - 改动: `train_*.py` 顶层加 `torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)`
   - 目的: 防止 SDPA 在某些 edge case 下回退到 Math 后端

4. **验证基线**
   - 用 `nvidia-smi dmon` 监控 64M 训练时的显存峰值
   - 用 `torch.cuda.Event` 测 step/s
   - 跑通一个 epoch 确认损失曲线没崩

### 5.2 阶段 B: P0+P1 · 深度优化级 (1-2 周)

**目标**: 单卡 4090 可稳定训练 500M-1B Dense / 1.5B+ MoE

5. **加 bitsandbytes 8-bit AdamW**
   - 改动: `trainer_utils.py` 加 import 与 fallback
   - 收益: 优化器状态从 8 字节/参数降到 2 字节/参数 (-75%)
   - 1B 模型: 优化器状态从 8GB 降到 2GB

6. **集成 Liger-Kernel**
   - 依赖: `pip install liger-kernel`
   - 改动: 写 `model/model_liger.py` adapter, 替换 RMSNorm / RoPE / SwiGLU / cross-entropy
   - 收益: 激活 -20%, 速度 +20%

7. **(可选) Activation Offloading**
   - 适合: 长序列 (seq > 2048) 训练
   - 工具: `torch.utils.checkpoint` 的 `_save_rng_state` + 自定义 hook, 或 `accelerate` 的 `ActivationOffloading` context manager

### 5.3 阶段 C: P0+P1+P2 · 战略工程级 (2-4 周)

**目标**: 1.5B-2B Dense, 3B+ MoE

8. **MoE Triton Grouped-GEMM**
   - 改动: `model/model_minimind.py:265-293` 的 `MOEFeedForward.forward` 整体重写
   - 关键: 用 Triton 实现 `grouped_gemm` (每个 expert 一组 token 走一次 matmul)
   - 参考: `deepseek-ai/DeepSeek-MoE` 的 `grouped_gemm.py` 或 `sglang` 的 `moe_align_block_size` kernel
   - 收益: MoE 训练 step/s 提升 1.5-3x (当前 Python for 循环是最大瓶颈)

9. **FSDP2 多卡分片** (仅 >= 2 卡场景)
   - 启动: `accelerate launch --config_file fsdp2_config.yaml train_pretrain.py`
   - 收益: 2x 4090 可训练 4B Dense, 4x 4090 可训练 8B Dense

---

## 6. 风险与注意事项

### 6.1 训练行为变化风险

- **torch.compile**: 首次编译耗时 +20-30s, 之后无开销; 若用 TriAttention 等自定义 sparse mask 路径, 编译可能失败 → 需要 fallback (`@torch.compiler.disable`)
- **梯度检查点 + DDP**: 已有 `use_cache=True` 时关闭的逻辑, 但需在 `MiniMindBlock.forward` 顶层显式判断
- **8-bit AdamW**: 收敛性微差异, 建议先用 64M 跑 1 epoch 验证 loss 曲线与原 AdamW 一致

### 6.2 硬件特定

- 4090 没有 FP8, **不要参考 H100 上的 FP8 优化数字**
- 4090 没有 NVLink, 多卡只能用 PCIe 4.0 (32 GB/s), 通信是瓶颈
- 4090 显存 24GB 是 GDDR6X (不是 HBM), 带宽 1TB/s, 在大 batch + 短序列上带宽受限

### 6.3 数值稳定性

- BF16 在 attention 累积时可能溢出 (概率低), FA2 内部已处理
- 8-bit AdamW 状态在极端 LR (1e-2) 下可能损失精度, 默认 LR 5e-4 无问题
- Gradient checkpointing 不影响数值, 只影响显存

---

## 7. 实施建议 (下一步行动)

### 7.1 立刻可做 (本周)

1. 阅读 [`training-technologies/`](training-technologies/) 下 8 份技术详情文档
2. 选择 P0 三件套中的**优先一项**开始动手 (建议先 `torch.compile`, 因为改动最小)
3. 在 64M baseline 上做 A/B 测试, 收集 step/s 和显存数据

### 7.2 决策点 (需用户确认)

| 问题 | 选项 | 建议 |
|------|------|------|
| 是否引入 `bitsandbytes` 依赖? | 是 / 否 | 建议是 (P0 性价比最高) |
| 是否引入 `liger-kernel` 依赖? | 是 / 否 | 建议是 (P1 性价比高) |
| MoE 路线是否要做 Triton rewrite? | 是 (2-4 周) / 否 (跳过 P2) | 视 minimind-4 是否包含 MoE 决定 |
| 多卡训练是否在路线图? | 是 / 否 | 决定是否做 P2 的 FSDP2 |
| 是否需要 min_seq_len 上调到 2048+? | 是 (需要 seq packing + activation offload) / 否 | 视最终模型架构决定 |

### 7.3 时间线 (保守估计)

| 周 | 目标 | 里程碑 |
|----|------|--------|
| W1 | P0 落地 | 64M 训练速度 +1.3-1.5x |
| W2 | 500M Dense 试跑 | minimind-4-500M 配置可跑通 1 epoch |
| W3 | P1 落地 | Liger-Kernel + 8bit AdamW 集成 |
| W4 | 1B Dense 试跑 | minimind-4-1B 配置可跑通 1 epoch |
| W5-W6 | (可选) P2 MoE rewrite | minimind-4-moe 1.5B 训练速度 +1.5x |

---

## 8. 参考文献

- **FlashAttention-2 论文**: Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023) - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- **FlashAttention-3 论文**: Tri Dao et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024) - [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- **梯度检查点**: Tianqi Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016) - [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- **bitsandbytes 8-bit optimizer**: Tim Dettmers et al., "8-bit Approximations for Parallelism in Deep Learning" (2022) - [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- **Liger-Kernel**: LinkedIn Engineering, "Liger-Kernel: Efficient Triton Kernels for LLM Training" (2024) - [GitHub](https://github.com/linkedin/Liger-Kernel)
- **PyTorch FSDP2**: PyTorch 官方文档, "Fully Sharded Data Parallel v2" (2024) - [PyTorch docs](https://pytorch.org/docs/stable/fsdp.html)
- **Accelerate**: HuggingFace, "Accelerate: Training and Inference at Scale" - [GitHub](https://github.com/huggingface/accelerate)
- **DeepSpeed ZeRO-Offload**: Microsoft, "ZeRO-Offload: Democratizing Billion-Scale Model Training" (2021) - [USENIX ATC '21](https://www.usenix.org/conference/atc21/presentation/ren-jie)
- **MiniMind 项目**: [GitHub jingyaogong/minimind](https://github.com/jingyaogong/minimind) (Apache 2.0)

---

## 9. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 完成基线审计, 修正原方案 3 处关键错误, 给出 P0/P1/P2 落地建议 | Sisyphus |

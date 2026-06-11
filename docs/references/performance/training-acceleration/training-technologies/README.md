# MiniMind 训练优化技术文档总索引

> **创建日期**: 2026-06-08
> **维护范围**: 训练侧 SOTA 优化技术在 MiniMind 中的现状审查、可行性与集成规划
> **配套报告**: [`../../training-gap-analysis.md`](../../training-gap-analysis.md)
> **目标硬件**: 单卡 / 多卡 NVIDIA RTX 4090 (Ada Lovelace, sm_89, 24GB)
> **训练目标**: minimind-4 级别 500M-1B Dense / 1.5-3B MoE (64M 激活)

---

## 目录

- [0. 文档使用指南](#0-文档使用指南)
- [1. 现状速查矩阵](#1-现状速查矩阵)
- [2. 推荐技术全景矩阵](#2-推荐技术全景矩阵)
- [3. 优先级与落地分组](#3-优先级与落地分组)
- [4. 文档维护约定](#4-文档维护约定)

---

## 0. 文档使用指南

本目录下每一份技术文档都遵循**统一的 10 章节模板**, 用于支持**长期追踪与可修正性**。无论你是在做代码 review、性能调优, 还是想理解某项技术为什么在 MiniMind 中以当前形态存在, 都建议按以下顺序阅读:

| 章节 | 用途 |
|------|------|
| 1. 技术概述 | 快速理解技术本质 (1-2 分钟可读完) |
| 2. 必要性论证 | 解释"为什么要在 MiniMind 集成" |
| 3. 架构设计 | 数据流图、模块划分、复杂度分析 |
| 4. 方案实现 | 核心代码片段、关键参数、默认配置 |
| 5. 训练过程影响 | 训练目标修改、损失函数、显存 |
| 6. 消融实验方案 | 如何科学地验证该技术的有效性 |
| 7. 已知问题与限制 | 当前实现的"坑" |
| 8. 后续改进方向 | 下一步可做的工作 |
| 9. 参考文献 | 论文与开源实现链接 |
| 10. 变更日志 | 任何修改都要记录到这里 |

**约定**: 每份文档的状态标记为以下几种:
- ✅ **已集成 (Production)**: 完整集成, 可在 `train_*.py` 中通过 CLI flag 启用
- ⚠️ **PoC (Proof-of-Concept)**: 已实现但仅为概念验证, 不建议用于生产
- 📝 **讨论中 (Discussion)**: 仅有方案/讨论, 尚未实现
- ❌ **未集成 (Not Integrated)**: 未在当前主线中集成

---

## 1. 现状速查矩阵 (2026-06-08 审计结果)

> 本表汇总了直接阅读 `model/model_minimind.py` / `trainer/train_*.py` / `trainer/trainer_utils.py` 后的"实际代码状态", 而非 README 自述状态。详见 [`../../training-gap-analysis.md`](../../training-gap-analysis.md)。

| 优化技术 | 方案主张 | 实际代码状态 | 证据 |
|----------|----------|--------------|------|
| FlashAttention (经 SDPA) | "未集成" | ✅ **已集成** | `model_minimind.py:153, 233`, `MiniMindConfig.flash_attn` 默认 `True` |
| BF16 混合精度 | "未集成" | ✅ **已集成** | `train_pretrain.py:91, 121-122`, `--dtype` 默认 `bfloat16` |
| torch.compile | "未集成" | ⚠️ **开关已就绪, 默认关闭** | `train_pretrain.py:106, 150-152`, `--use_compile 0` |
| DeepSpeed ZeRO | "未集成" | ❌ 未集成 | 全仓库无 `deepspeed` 引用 |
| 梯度检查点 (Grad-CKPT) | "未集成" | ❌ 未集成 | 全仓库无 `checkpoint` / `gradient_checkpointing` 引用 |
| 8-bit / Paged Optimizer | "未集成" | ❌ 未集成 | `train_pretrain.py:138` 使用 `optim.AdamW` (FP32 状态) |
| FSDP | "未集成" | ❌ 未集成 | 全仓库无 `FSDP` / `fully_shard` 引用 |
| Accelerate (单卡 offload) | "未集成" | ❌ 未集成 | 启动方式为 `python train_*.py` / `torchrun`, 无 `accelerate launch` |
| Liger-Kernel (Triton fused) | 未提及 | ❌ 未集成 | `requirements.txt` 无 `liger-kernel` |
| MoE Fused Kernel | 未提及 | ❌ 未集成 | `model_minimind.py:273-292` 是 Python `for` + `index_add_` |

---

## 2. 推荐技术全景矩阵

| #  | 技术 | 类别 | 优先级 | 状态 | 文档 |
|----|------|------|--------|------|------|
| 01 | 选择式梯度检查点 | 激活压缩 | **P0** | 📝 讨论中 | [→](01-selective-grad-checkpoint.md) |
| 02 | torch.compile (`reduce-overhead`) | 计算图优化 | **P0** | ⚠️ 开关就绪 | [→](02-torch-compile.md) |
| 03 | 8-bit / Paged AdamW (bitsandbytes) | 优化器状态压缩 | **P0** | 📝 讨论中 | [→](03-bnb-8bit-adamw.md) |
| 04 | Liger-Kernel (Triton fused) | 计算图融合 | **P1** | 📝 讨论中 | [→](04-liger-kernel.md) |
| 05 | Accelerate 单卡 CPU Offload | 优化器卸载 | **P1** | 📝 讨论中 | [→](05-accelerate-offload.md) |
| 06 | Activation Offloading | 激活卸载 | **P1** | 📝 讨论中 | [→](06-activation-offload.md) |
| 07 | MoE Triton Grouped-GEMM | MoE 算子融合 | **P2** | 📝 讨论中 | [→](07-moe-triton-grouped-gemm.md) |
| 08 | FSDP2 多卡分片 | 分布式分片 | **P2** | 📝 讨论中 | [→](08-fsdp2.md) |

> 状态说明: **P0** = 性价比最高 / 1 周内可落地; **P1** = 重要但需更深改造 / 1-2 周; **P2** = 战略性 / 2-4 周。

---

## 3. 优先级与落地分组

### P0 · 开箱即用级 (1 周可验证)
> 全部基于已发表技术, 工程量小, 预期 64M 基线提速 1.3-1.5x。

- **01 · [选择式梯度检查点](01-selective-grad-checkpoint.md)**: 仅对 MLP 启用 `torch.utils.checkpoint`, Attention 走 FA2 已是 O(N); 激活显存 -50%, 速度 -10%
- **02 · [torch.compile `reduce-overhead`](02-torch-compile.md)**: 启用 CUDA Graphs, 显著减少 kernel launch 开销; 速度 +1.3-1.8x, 显存 +0%
- **03 · [8-bit / Paged AdamW](03-bnb-8bit-adamw.md)**: 优化器状态从 8 字节/参数压到 2 字节/参数; 显存 -75% (优化器), 速度 -5%

### P1 · 深度优化级 (1-2 周)
> 需要新增 1-2 个核心依赖, 但有现成开源实现。

- **04 · [Liger-Kernel](04-liger-kernel.md)**: LinkedIn 2024, Triton 写, 把 RMSNorm/RoPE/SwiGLU/CrossEntropy fuse; 激活 -20%, 速度 +20%
- **05 · [Accelerate 单卡 CPU Offload](05-accelerate-offload.md)**: `accelerate launch --cpu` 单卡适用; 优化器状态卸载到 CPU 内存
- **06 · [Activation Offloading](06-activation-offload.md)**: 把不参与当前 step 的激活卸载到 CPU, 比 grad-checkpoint 更细粒度

### P2 · 战略工程级 (2-4 周)
> 涉及核心算子重写, 适合作为"minimind-4"重点工程。

- **07 · [MoE Triton Grouped-GEMM](07-moe-triton-grouped-gemm.md)**: 替换 `MOEFeedForward` 的 Python for 循环, 显著提升 MoE 训练速度
- **08 · [FSDP2 多卡分片](08-fsdp2.md)**: 仅在多卡 (>= 2x 4090) 场景启用, 单卡不可用

---

## 4. 文档维护约定

### 4.1 修正记录规则

任何对代码、配置、消融结果、已知问题的修正, 都应该:
1. **更新对应技术文档的对应章节**
2. **在第 10 章"变更日志"中追加一行记录**
3. **更新文档顶部的"最后修正"日期**

格式:
```markdown
| 2026-MM-DD | [修改类型] 简述 | [@author] |
```

### 4.2 修改类型枚举

| 类型 | 说明 |
|------|------|
| `+feat` | 新增功能 |
| `+opt` | 性能优化 |
| `+exp` | 新增消融实验数据 |
| `-fix` | 修复 bug |
| `-issue` | 标记已知问题 |
| `~refactor` | 重构, 不改外部行为 |
| `~doc` | 仅文档修正 |

### 4.3 消融实验结果记录

当某项消融实验在 MiniMind 上跑出结果, 应该:
1. 在第 6 章"消融实验方案"末尾追加"实际结果"小节
2. 包含: 实验配置 (超参、数据集)、指标 (step/s, peak VRAM, loss 曲线)、结论
3. 如果结果与预期不符, 应该在第 7 章"已知问题与限制"中追加对应说明

### 4.4 集成状态变更

如果某项技术从 `📝 讨论中` 升级到 `✅ Production`, 或反过来, 应该:
1. 更新文档顶部状态标记
2. 在变更日志中说明升级/降级原因
3. 同步更新本 README 的"全景矩阵"

---

## 附: MiniMind 训练默认参数速查 (2026-06-08)

| 参数 | 默认值 | 来源 |
|------|--------|------|
| `--dtype` | `bfloat16` | `train_pretrain.py:91` |
| `--use_compile` | `0` (关闭) | `train_pretrain.py:106` |
| `--batch_size` | `32` | `train_pretrain.py:88` |
| `--accumulation_steps` | `8` | `train_pretrain.py:93` |
| `--num_workers` | `8` | `train_pretrain.py:92` |
| `--max_seq_len` (pretrain) | `340` | `train_pretrain.py:99` |
| `--max_seq_len` (sft) | `340` | `train_full_sft.py` (待核对) |
| 优化器 | `AdamW` (FP32 状态) | `train_pretrain.py:138` |
| 精度上下文 | `torch.cuda.amp.autocast(dtype=bfloat16)` | `train_pretrain.py:122` |
| 注意力 | `F.scaled_dot_product_attention` (SDPA → FA2/MemEff/Math) | `model_minimind.py:233` |
| MiniMindConfig.flash_attn | `True` | `model_minimind.py:21` |
| 词表大小 | 6400 | `model_minimind.py:18` |
| tie_word_embeddings | `True` | `model_minimind.py:30` |

> 任何文档中提到的"64M 训练基线"或"训练 step/s 提升 1.3x"均指上述配置的对照, 不再重复说明。

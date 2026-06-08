# MiniMind 推理加速技术文档总索引

> **创建日期**: 2026-06-08
> **维护范围**: 18 项业界 SOTA 推理加速技术在 MiniMind 中的集成、追踪与修正
> **配套报告**: [`../inference-gap-analysis.md`](../inference-gap-analysis.md)
> **实施计划**: [`../../.omo/plans/inference-tech-integration-plan.md`](../../.omo/plans/inference-tech-integration-plan.md)

---

## 目录

- [0. 文档使用指南](#0-文档使用指南)
- [1. 18 项技术全景矩阵](#1-18-项技术全景矩阵)
- [2. 按 Wave 分组索引](#2-按-wave-分组索引)
- [3. 按技术类别索引](#3-按技术类别索引)
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
- ✅ **已实现 (Production)**: 完整实现, 可在 `eval_llm.py` 中通过 CLI flag 启用
- ⚠️ **PoC (Proof-of-Concept)**: 已实现但仅为概念验证, 不建议用于生产
- 📝 **讨论中 (Discussion)**: 仅有方案/讨论, 尚未实现
- ❌ **未实现 (Not Implemented)**: 未在当前主线中实现

---

## 1. 18 项技术全景矩阵

| #  | 技术 | 类别 | Wave | 状态 | 文档 |
|----|------|------|------|------|------|
| 01 | 预分配 KV 缓存 | KV 缓存 | 1 | ✅ | [→](01-pre-allocated-kv-cache.md) |
| 02 | StreamingLLM | KV 缓存 | 1 | ✅ | [→](02-streaming-llm.md) |
| 03 | Lookahead Decoding | 推测解码 | 1 | ✅ | [→](03-lookahead-decoding.md) |
| 04 | PLD / AdaPLD | 推测解码 | 1 | ✅ | [→](04-pld-decoding.md) |
| 05 | MInference 1.0 | 稀疏注意力 | 1 | ✅ | [→](05-minference.md) |
| 06 | TriAttention | 稀疏注意力 | 1 | ✅ | [→](06-triattention.md) |
| 07 | Medusa-1 | 推测解码 | 2 | ✅ | [→](07-medusa.md) |
| 08 | MTP-as-Draft | 推测解码 | 2 | ✅ | [→](08-mtp-draft.md) |
| 09 | KIVI 2-bit | KV 缓存 | 2 | ✅ | [→](09-kivi.md) |
| 10 | RTPurbo | 稀疏注意力 | 2 | ✅ | [→](10-rt-purbo.md) |
| 11 | Gated DeltaNet | 线性注意力 | 3 | ⚠️ | [→](11-gated-deltanet.md) |
| 12 | Lightning Indexer | 稀疏注意力 | 3 | ⚠️ | [→](12-lightning-indexer.md) |
| 13 | Qwen3-Next 3:1 Hybrid | 架构级 | 4 | ✅ | [→](13-qwen3-next-hybrid.md) |
| 14 | DFlash | 推测解码 | 4 | ✅ | [→](14-dflash.md) |
| 15 | DDTree | 推测解码 | 4 | ✅ | [→](15-ddtree.md) |
| 16 | mHC 残差连接 | 架构级 | 4 | ✅ | [→](16-mhc.md) |
| 17 | NSA 三路稀疏 | 稀疏注意力 | 4 | ✅ | [→](17-nsa.md) |
| 18 | dLM 扩散语言模型 | 架构级 | 4 | ✅ | [→](18-dlm.md) |

---

## 2. 按 Wave 分组索引

### Wave 1 · 零训练型推理加速 (6 项)
> 不需要任何额外训练, 直接在推理时启用即可获得加速。优先级最高, 集成门槛最低。

- **01 · [预分配 KV 缓存](01-pre-allocated-kv-cache.md)**: 避免增量 `torch.cat` 反复分配显存
- **02 · [StreamingLLM](02-streaming-llm.md)**: 注意力汇点 + 滑动窗口, 支持超长序列
- **03 · [Lookahead Decoding](03-lookahead-decoding.md)**: Jacobi 迭代, 零 draft 模型
- **04 · [PLD / AdaPLD](04-pld-decoding.md)**: 基于 prompt 的查找式投机解码
- **05 · [MInference 1.0](05-minference.md)**: 离线标定每个 head 的最优稀疏模式
- **06 · [TriAttention](06-triattention.md)**: 三角级数估分 attention-vs-distance 曲线

### Wave 2 · 训练型推理加速 (4 项)
> 需要额外训练小型的 head / drafter 模块, 训练成本可控, 加速比通常更高。

- **07 · [Medusa-1](07-medusa.md)**: 在 backbone 上加 K 个并行 decoding heads
- **08 · [MTP-as-Draft](08-mtp-draft.md)**: 复用主线 MTP head 作为 drafter
- **09 · [KIVI 2-bit](09-kivi.md)**: KV 缓存 2-bit 量化, 显著降低显存
- **10 · [RTPurbo](10-rt-purbo.md)**: head-wise 稀疏 + 16-dim 索引, Microsoft 2026

### Wave 3 · 架构级 PoC (2 项)
> 验证新一代线性/稀疏注意力在小模型上的可行性, 仅为 PoC。

- **11 · [Gated DeltaNet](11-gated-deltanet.md)**: NVlabs ICLR 2025, Qwen3-Next 75% 层使用
- **12 · [Lightning Indexer](12-lightning-indexer.md)**: DeepSeek-V3.2 DSA 思路

### Wave 4 · 战略级架构 (6 项)
> 重大架构变更, 需要专门训练。技术风险与潜在收益都很高。

- **13 · [Qwen3-Next 3:1 Hybrid](13-qwen3-next-hybrid.md)**: 全局 attn + 线性 attn 混合
- **14 · [DFlash](14-dflash.md)**: Block diffusion speculative decoder
- **15 · [DDTree](15-ddtree.md)**: DFlash 树形多路并行扩展
- **16 · [mHC 残差连接](16-mhc.md)**: Manifold-Constrained Hyper-Connections
- **17 · [NSA 三路稀疏](17-nsa.md)**: Native Sparse Attention 三路压缩
- **18 · [dLM 扩散语言模型](18-dlm.md)**: 离散扩散范式, 与 AR 完全不同的生成方式

---

## 3. 按技术类别索引

### 3.1 推测解码 (Speculative Decoding) — 6 项
- [03 · Lookahead Decoding](03-lookahead-decoding.md) — 无 draft 模型
- [04 · PLD / AdaPLD](04-pld-decoding.md) — 查找式
- [07 · Medusa-1](07-medusa.md) — head 扩展
- [08 · MTP-as-Draft](08-mtp-draft.md) — 复用主线
- [14 · DFlash](14-dflash.md) — Block diffusion
- [15 · DDTree](15-ddtree.md) — 树形 DFlash

### 3.2 稀疏注意力 (Sparse Attention) — 5 项
- [05 · MInference 1.0](05-minference.md) — 离线标定
- [06 · TriAttention](06-triattention.md) — 三角级数
- [10 · RTPurbo](10-rt-purbo.md) — head-wise 稀疏
- [12 · Lightning Indexer](12-lightning-indexer.md) — 主 attn 混合
- [17 · NSA 三路稀疏](17-nsa.md) — 三路压缩

### 3.3 KV 缓存优化 — 3 项
- [01 · 预分配 KV 缓存](01-pre-allocated-kv-cache.md) — 显存布局
- [02 · StreamingLLM](02-streaming-llm.md) — 注意力汇点
- [09 · KIVI 2-bit](09-kivi.md) — 2-bit 量化

### 3.4 架构级 — 4 项
- [11 · Gated DeltaNet](11-gated-deltanet.md) — 线性 attn
- [13 · Qwen3-Next 3:1 Hybrid](13-qwen3-next-hybrid.md) — attn 混合
- [16 · mHC 残差连接](16-mhc.md) — 残差结构
- [18 · dLM 扩散语言模型](18-dlm.md) — 生成范式

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
2. 包含: 实验配置 (超参、数据集)、指标 (加速比 / PPL / acceptance rate)、结论
3. 如果结果与预期不符, 应该在第 7 章"已知问题与限制"中追加对应说明

### 4.4 集成状态变更

如果某项技术从 `⚠️ PoC` 升级到 `✅ Production`, 或反过来, 应该:
1. 更新文档顶部状态标记
2. 在变更日志中说明升级/降级原因
3. 同步更新本 README 的"全景矩阵"

---

## 附: MiniMind 关键参数速查

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

> 任何文档中提到的"64M"或"198M"均指上述配置, 不再重复说明。

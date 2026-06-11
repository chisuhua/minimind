# ⚡ Performance 参考子目录

> **目录定位**:`docs/references/performance/` —— 性能优化参考,28 份文档。
>
> **新目标相关度**:⭐ INDIRECT(通用 LLM 性能技术,非自循环机制必需)

---

## 📁 内容索引

| 子目录/文件 | 文档数 | 核心内容 | 何时查阅 |
|---|---|---|---|
| [`inference-acceleration/inference-technologies/`](./inference-acceleration/inference-technologies/) | 18 份(原 18,3 份 CRITICAL 已提取) | 业界 SOTA 推理加速技术(预分配 KV、StreamingLLM、KIVI、MInference、TriAttention、Medusa、MTP、Qwen3-Next Hybrid、NSA、dLM 等) | 调研自循环以外的额外加速技术时 |
| [`training-acceleration/training-technologies/`](./training-acceleration/training-technologies/) | 8 份 | 训练加速技术(梯度检查点、torch.compile、8-bit AdamW、Liger-Kernel、Accelerate Offload、Activation Offload、MoE Triton Grouped-GEMM、FSDP2) | 优化训练管线时 |
| [`inference-gap-analysis.md`](./inference-gap-analysis.md) | 1 份 | MiniMind 与业界 SOTA 推理加速技术差距分析 | 评估推理性能 gap 时 |
| [`training-gap-analysis.md`](./training-gap-analysis.md) | 1 份 | MiniMind 训练优化技术 gap 分析 | 评估训练性能 gap 时 |

---

## 🚨 重要警示

### 3 项 CRITICAL 推理引擎已提取

3 项对"自循环 AgenticDSL 推理"构成 **CRITICAL_FOR_LOOP** 支撑的技术已提取至 [`../../inference-engine/`](../../inference-engine/):
- **01 预分配 KV 缓存** —— prefix 复用物理底座
- **02 StreamingLLM** —— 长链路不爆显存
- **09 KIVI 2-bit** —— KV cache 量化压缩

**不要在本目录的 `inference-technologies/` 中寻找它们**。

### ⚠️ 与自循环机制**不兼容**的技术

- **11 Gated DeltaNet** + **13 Qwen3-Next Hybrid**(线性注意力):破坏 KV cache 可寻址性,**削弱**自循环 prefix sharing
- **18 dLM**(离散扩散):与 AR 自循环范式互斥

如选 dLM 路线,自循环概念需重新定义。

---

## 🎯 何时查阅本目录

- 调研"自循环"以外的额外推理加速技术时(USEFUL 类:Lookahead、PLD、Medusa、MTP、DFlash、DDTree、NSA)
- 训练管线需要性能优化时(参考 `training-acceleration/`)
- 评估当前 MiniMind 与业界 SOTA 的性能 gap 时(`inference-gap-analysis.md` / `training-gap-analysis.md`)

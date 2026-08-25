# model/ — 模型架构与推理加速模块

## OVERVIEW

MiniMind fork 基座 + 自建推理加速模块混合目录，涵盖标准 Transformer、混合架构、KV Cache、投机解码。

## WHERE TO LOOK

| 文件 | 架构 | 生产状态 | eval_llm.py flag |
|------|------|----------|------------------|
| `model_minimind.py` | Decoder-Only Transformer (Pre-Norm+RMSNorm+SwiGLU+RoPE+GQA) | ✅ 生产 | `--weight` 基座 |
| `model_minimind_hybrid.py` | Qwen3-Next 3:1 混合 (Gated DeltaNet 75% + Gated Attention 25%) | ✅ 生产 | `--weight` |
| `medusa_heads.py` | Medusa 多候选投机验证头 | ✅ 生产 | `--medusa` |
| `lookahead_decoding.py` | N-gram Lookahead 投机解码 | ✅ 生产 | `--lookahead_decoding` |
| `pld_decoding.py` | Prompt Lookup Decoding | ✅ 生产 | `--pld` |
| `kivi_kv_cache.py` | KIVI 2-bit KV Cache 量化 | ✅ 生产 | `--kv_quant kivi_2bit` |
| `streaming_kv_cache.py` | StreamingLLM (attention sink + sliding window) | ✅ 生产 | `--streaming_llm` |
| `minference.py` | MInference 1.0 (A-shape/Vertical-Slash/Block-Sparse) | ✅ 生产 | `--minference` |
| `nsa.py` | Native Sparse Attention (compression+selection+sliding) | 🔬 研究 | `--nsa_sparse` |
| `dflash.py` | DFlash 块扩散投机 (KV injection + feature fusion) | 🔬 研究 | — |
| `diffusion_lm.py` | 离散扩散 LM (LLaDA 风格 masked diffusion) | 🔬 研究 | — |
| `gated_deltanet.py` | Gated DeltaNet (线性 recurrent attention) | 🔬 研究 | `--gated_deltanet` |
| `lightning_indexer.py` | Lightning Indexer (DSA top-k sparse) | 🔬 研究 | — |
| `mhc.py` | Manifold-Constrained Hyper-Connections | 🔬 研究 | — |
| `ddtree.py` | DDTree 树状投机解码 | 🔬 研究 | — |
| `tri_attention.py` | Tri Attention Scorer (Fourier 模式学习) | 🔬 研究 | `--tri_attention` |
| `rt_purbo.py` | RT-Purbo (检索头分类 + 低维索引) | 🔬 研究 | `--rt_purbo` |
| `mtp_head.py` | Multi-Token Prediction 辅助头 | 🔬 研究 | `--mtp` |
| `model_lora.py` | LoRA 手动实现 (低秩分支) | 🔬 研究 | — |

## BASE MODEL CLASS API

**MiniMindConfig** — 配置类，关键字段：

```
hidden_size         # 隐层维度
num_hidden_layers   # 层数
use_moe             # MoE 开关，True 时为 198M-A64M MoE
max_seq_len        # 最大序列长度
rope_theta          # RoPE 基础频率
vocab_size          # 6400 (BPE+ByteLevel)
num_attention_heads
num_key_value_heads # GQA
intermediate_size   # SwiGLU FFN 维度
```

**MiniMindForCausalLM** — 因果语言模型，`from_pretrained()` 加载。

**HybridMiniMindForCausalLM** — 内部组合 GatedDeltaNet 基座 + GatedAttention，通过 `trainer/train_hybrid.py` 训练。

## CONVENTIONS

- 所有模型通过 `eval_llm.py` 的 `init_model()` 注册，禁止直接 `torch.load` 权重文件
- Tokenizer 始终从 `model/tokenizer.json` + `tokenizer_config.json` 加载
- MoE 权重文件命名后缀：`_moe.pth`
- 研究阶段模型需要独立训练脚本 (`trainer/train_*.py`)，与生产管线解耦

## ANTI-PATTERNS

- ❌ 修改 `model_minimind.py` 的类签名或 config 字段 — 破坏 MiniMind fork 兼容性
- ❌ 研究阶段模型接入 `eval_llm.py` 前必须同时实现对应 `--flag` 和训练脚本
- ❌ Streaming KV Cache 禁止与 NSA/MInference 混用 — 需提前做兼容性检查
- ❌ 禁止在生产路径直接 `import` 研究阶段模块

## NOTES

规格说明书分层：

1. **基座** — `model_minimind.py` (MiniMindConfig / MiniMindForCausalLM)
2. **混合架构** — `model_minimind_hybrid.py` + `gated_deltanet.py`
3. **KV Cache + 投机解码** — streaming / kivi / lookahead / pld / medusa (生产可用)
4. **研究扩展** — nsa / dflash / diffusion_lm / lightning / mhc / ddtree / tri_attention / rt_purbo / mtp / lora

Token 级别推理指标（eval_llm.py 输出）：format compliance / signature compliance / execution success / task success。

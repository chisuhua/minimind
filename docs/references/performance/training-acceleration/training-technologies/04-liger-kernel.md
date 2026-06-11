# 04 · Liger-Kernel (Triton Fused Kernel)

> **状态**: 📝 讨论中 | **优先级**: P1
> **代码位置 (待修改)**: `model/model_minimind.py` (RMSNorm, RoPE, FeedForward, loss)
> **依赖**: `liger-kernel >= 0.4`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

**Liger-Kernel** (LinkedIn 2024) 是一组用 **Triton** 写的高性能 fused kernel, 用于 LLM 训练中的常用操作:

- **Fused RMSNorm**: 把 `mean + rsqrt + multiply` 三步合并
- **Fused RoPE**: 把 RoPE 计算与 Q/K 投影合并
- **Fused SwiGLU**: 把 `silu(gate(x)) * up(x)` 合并为单个 kernel
- **Fused Cross-Entropy**: 把 `log_softmax + cross_entropy` 合并, 避免保存完整 logits

**为什么"fused"重要**:

```text
# 普通实现 (多个 kernel)
hidden = rms_norm(x)          # kernel 1: 读 x, 写 hidden
attn = q * cos + rotate(q) * sin  # kernel 2-3: 读 q, cos, sin, 写 attn
output = attn @ v              # kernel 4: 读 attn, v, 写 output

# Fused 实现 (单个 kernel)
output = fused_rope_attn(x, cos, sin, v)  # 一次读 x, cos, sin, v, 一次写 output
# 减少: kernel launch 3 次 → 1 次
# 减少: 中间结果不再写回 HBM (节省带宽)
```

**典型收益** (LinkedIn 公布数据):
- RMSNorm: 1.5-2x 提速, 显存 -30%
- RoPE: 1.3-1.5x 提速
- SwiGLU (GLU): 1.5-2x 提速
- Cross-Entropy (尤其大词表): **3-5x 提速, 显存 -50%** (不需要保存完整 logits)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- MiniMind 的 `RMSNorm` (手写) / `apply_rotary_pos_emb` (手写) / `FeedForward` (PyTorch) / `cross_entropy` (PyTorch) **都是未 fused 的朴素实现**
- Liger-Kernel 提供**直接 drop-in 替换** (API 兼容)
- 1B+ 模型 + 长序列训练时, **Cross-Entropy 节省的 logits 显存巨大**:
  - 1B 模型, batch=8, seq=2048, vocab=6400: logits 占用 = 8 × 2048 × 6400 × 4 (FP32) = **400 MB**
  - Liger fused CE 把它降到 ~0 (在 kernel 内计算, 不保存)
- 1B+ 模型 + 长序列训练时, **RMSNorm / RoPE / SwiGLU 的 HBM 带宽**也是主要瓶颈

**不集成的代价**:

- 64M 训练时, kernel launch 和 HBM 带宽占比相对小, 收益有限
- 1B+ 模型训练时, **速度损失 15-25%** (相对 Liger fused)
- Cross-Entropy 显存是必须解决的瓶颈 (否则 batch 进一步受限)

---

## 3. 架构设计

### 3.1 Liger-Kernel 提供的主要接口

```python
from liger_kernel.transformers import (
    apply_liger_kernel_to_minimind,  # MiniMind 风格需要自定义
    LigerRMSNorm,
    LigerSwiGLUMLP,
    LigerFusedLinearCrossEntropy,
)
```

**问题**: Liger-Kernel 的官方 `apply_liger_kernel_to_*` 函数**只支持 HuggingFace 主流模型** (Llama, Qwen2, Mistral 等), **没有 MiniMind 的现成 adapter**。

**方案**: 写一个 `apply_liger_kernel_to_minimind()` 自定义函数, 替换 MiniMind 模型中的 4 个组件。

### 3.2 替换组件清单

| 原组件 | 位置 | Liger 替换 |
|--------|------|------------|
| `RMSNorm` | `model_minimind.py:94-104` | `LigerRMSNorm` |
| `apply_rotary_pos_emb` | `model_minimind.py:124-128` | `LigerRotaryEmbedding` 或手写 fused |
| `FeedForward` (SwiGLU) | `model_minimind.py:253-263` | `LigerSwiGLUMLP` |
| `F.cross_entropy` (在 `forward`) | `model_minimind.py:383` | `LigerFusedLinearCrossEntropy` |

### 3.3 Cross-Entropy 替换的特别说明

**原代码**:

```python
# model_minimind.py:382-383
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

**问题**: `x.contiguous()` 强制复制 logits (从 BF16 转回 contiguous), 显存峰值 = batch × seq × vocab × 4 字节

**Liger 替换**:

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropy

# 替换 lm_head 后的 Cross-Entropy
loss = LigerFusedLinearCrossEntropy.apply(
    hidden_states[..., :-1, :].contiguous(),  # 不再需要完整 logits
    lm_head_weight,
    labels[..., 1:].contiguous(),
    -100,  # ignore_index
)
# ↑ kernel 内部: hidden @ weight.T (fp32 累加) + cross_entropy, 不保存中间 logits
```

---

## 4. 方案实现

### 4.1 `requirements.txt` 增加依赖

```text
# requirements.txt
liger-kernel>=0.4.0
```

### 4.2 新建 `model/model_liger.py` adapter

```python
"""
Liger-Kernel 适配器: 把 MiniMind 的 RMSNorm / RoPE / SwiGLU / CE 替换为 Liger fused kernel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers import (
    LigerRMSNorm,
    LigerSwiGLUMLP,
    LigerFusedLinearCrossEntropy,
)

class LigerMiniMindAdapter:
    """把 Liger-Kernel 应用到 MiniMind 模型"""
    
    @staticmethod
    def apply(model: nn.Module):
        """
        替换 model 中的组件:
        1. RMSNorm -> LigerRMSNorm
        2. SwiGLU MLP -> LigerSwiGLUMLP  
        3. CE -> LigerFusedLinearCrossEntropy (在 forward 中处理)
        """
        # 替换 RMSNorm
        for module in model.modules():
            if isinstance(module, RMSNorm):  # 我们的手写 RMSNorm
                # 保留权重, 替换为 Liger 实现
                new_module = LigerRMSNorm(
                    hidden_size=module.weight.shape[0],
                    eps=module.eps,
                ).to(module.weight.device).to(module.weight.dtype)
                new_module.weight.data.copy_(module.weight.data)
                # ... 替换逻辑 ...
        
        # 替换 FeedForward
        for module in model.modules():
            if isinstance(module, FeedForward):
                new_module = LigerSwiGLUMLP(
                    hidden_size=module.gate_proj.in_features,
                    intermediate_size=module.gate_proj.out_features,
                )
                new_module.gate_proj.weight.data.copy_(module.gate_proj.weight.data)
                new_module.up_proj.weight.data.copy_(module.up_proj.weight.data)
                new_module.down_proj.weight.data.copy_(module.down_proj.weight.data)
                # ... 替换 ...
```

### 4.3 `model_minimind.py` 集成 (forward 中用 Liger CE)

```python
# model_minimind.py: MiniMindForCausalLM.forward
from model.model_liger import USE_LIGER

def forward(self, input_ids, ..., labels=None, **kwargs):
    hidden_states, past_key_values, aux_loss = self.model(...)
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden = hidden_states[:, slice_indices, :]
    
    if labels is not None:
        # Liger fused CE: 不需要完整 logits
        if USE_LIGER:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropy
            x = hidden[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = LigerFusedLinearCrossEntropy.apply(
                x, self.lm_head.weight, y, -100
            )
        else:
            logits = self.lm_head(hidden)
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        ...
```

### 4.4 `train_pretrain.py` CLI

```python
# trainer/train_pretrain.py: argparse
parser.add_argument('--use_liger', default=0, type=int, choices=[0, 1],
                    help='是否使用 Liger-Kernel 加速 (0=否, 1=是)')

# 在 init_model 之后:
if args.use_liger == 1:
    from model.model_liger import LigerMiniMindAdapter
    LigerMiniMindAdapter.apply(model)
    Logger('Liger-Kernel applied')
```

### 4.5 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_liger` | `0` | 关闭, 与原行为一致 |
| Liger `eps` | 与原 `RMSNorm.eps` 一致 | 必须对齐避免数值差异 |

---

## 5. 训练过程影响

| 维度 | 影响 |
|------|------|
| 显存 (激活) | **-15-25%** (中间结果不再保存) |
| 显存 (logits) | **-50-100%** (Liger fused CE) |
| 显存 (整体) | -10-20% |
| 速度 | **+15-25% step/s** (1B+ 模型) |
| 数值 | **极小差异** (Triton kernel 内部 FP32 累加) |
| 优化器 | 无影响 |
| DDP 兼容 | ✅ |
| torch.compile | ✅ (Triton kernel 是 Inductor 友好的) |

---

## 6. 消融实验方案

### 6.1 64M 速度验证

- **配置**: `hidden_size=768, batch=32, seq=340, 1 epoch on pretrain_t2t_mini`
- **对照**:
  - `--use_liger 0` (基线)
  - `--use_liger 1`
- **指标**:
  - 显存峰值
  - step/s
  - loss 曲线
- **预期**:
  - 64M 模型收益较小, 速度 +5-10%
  - 显存减少 5-10%
  - loss 曲线**与基线重合** (微差异在 1e-4 量级)

### 6.2 1B 显存验证 (关键验收)

- **配置**: `hidden_size=1536, num_layers=24, batch=8, seq=2048, 100 步`
- **对照**:
  - `--use_liger 0`: 预期 batch=8 时 OOM (因为 logits 400MB)
  - `--use_liger 1`: 预期 batch=8 跑通
- **指标**: max batch size, 显存峰值
- **预期**:
  - 整体显存减少 1-2 GB
  - 1B 模型 + batch=8 + seq=2048 可在 24GB 跑通

### 6.3 与 grad-checkpoint + 8bit AdamW + torch.compile 联合验证

- **配置**: `--use_liger 1 --grad_checkpoint 1 --optimizer adamw_8bit --use_compile 1`
- **预期**: 1B 模型 + batch=8 + seq=2048 在 24GB 卡上**总占用 < 18GB**

---

## 7. 已知问题与限制

### 7.1 首次编译耗时

- Triton kernel 首次调用会编译 (~5-15s)
- 之后每次调用零开销
- 风险等级: 低

### 7.2 数值差异

- Triton kernel 内部用 FP32 累加, 与 PyTorch FP32 结果可能差 1e-5 量级
- 对训练 loss 曲线**几乎无影响**
- 对下游任务效果**需要验证** (1B+ 模型上极个别 case 可能敏感)

### 7.3 MoE 路径未覆盖

- Liger-Kernel 当前没有专门为 `MOEFeedForward` 提供 fused kernel
- MoE 训练仍需 Python for 循环
- 见 [`07-moe-triton-grouped-gemm.md`](07-moe-triton-grouped-gemm.md) 解决

### 7.4 平台兼容性

- Triton 依赖 CUDA, Linux 平台稳定
- Windows 上 WSL2 可用
- macOS MPS 不支持 (Liger 限制)

### 7.5 与 autocast 的交互

- Liger RMSNorm / SwiGLU 接受 BF16 输入, 内部用 FP32 累加
- 与现有 `torch.cuda.amp.autocast(dtype=bfloat16)` 兼容 ✅
- 不需要修改训练循环

---

## 8. 后续改进方向

1. **覆盖更多组件**: 写自定义 Liger 风格的 RoPE + 注意力融合 (节省 Q/K projection 中间结果)
2. **MoE 路径**: 与 [`07-moe-triton-grouped-gemm.md`](07-moe-triton-grouped-gemm.md) 联合, 整个 MoE 块用 Triton 重写
3. **梯度累积优化**: 在 Triton kernel 中直接处理梯度累积 (减少 HBM 写入)

---

## 9. 参考文献

- **Liger-Kernel GitHub**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- **Liger-Kernel 论文**: "Liger-Kernel: Efficient Triton Kernels for LLM Training" (LinkedIn 2024) - 内部技术报告
- **Triton 官方文档**: [Triton Programming Guide](https://triton-lang.org/main/python-api/generated/triton.jit.html)
- **Cross-Entropy 优化**: [FlashCE 论文](https://arxiv.org/abs/2405.09875) (Liger 内部引用)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 定义 P1 Liger-Kernel 集成方案 | Sisyphus |

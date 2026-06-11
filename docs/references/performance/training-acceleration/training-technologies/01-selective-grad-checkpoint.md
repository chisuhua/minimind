# 01 · 选择式梯度检查点 (Selective Gradient Checkpointing)

> **状态**: 📝 讨论中 | **优先级**: P0
> **代码位置 (待修改)**: `model/model_minimind.py:295-318` (`MiniMindBlock`)
> **CLI 入口**: 暂未设计, 建议 `--grad_checkpoint 1` / `--grad_checkpoint 0`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

梯度检查点 (Gradient Checkpointing, Chen et al. 2016) 是一种**以时间换空间**的训练优化技术:
- **不保存**: 前向计算时, 只保留"检查点"层的激活 (即每隔若干层保存一次)
- **反向时重算**: 反向传播到某层时, 从最近的检查点重做一次前向, 重新计算中间激活
- **显存节省**: 激活显存从 O(L) 降到 O(sqrt(L)) (L = 网络层数)
- **计算开销**: 每次额外 1 次前向, 端到端 ~20-30% 速度损失 (取决于层数)

**"选择式" (Selective)**: 与传统"全量 checkpoint"不同, 我们**只对 MLP / FFN 启用 checkpoint**, 不对 Attention 启用。理由:
- Attention (经 FA2) 已经是 O(N) 显存, 检查点收益小
- MLP 是激活显存的大头 (SwiGLU 三个 Linear, 中间激活 = batch × seq × intermediate_size)
- 1B 模型 `intermediate_size ≈ 4096`, 序列 2048, batch 8 时单层 MLP 激活就有 4096 × 2048 × 8 × 4 字节 = 256MB

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- 当前 64M 训练时, 激活只占 ~150MB, grad-ckpt 收益小 (但**没有副作用**, 可以默认开启)
- 当扩展到 500M-1B (`intermediate_size` ≈ 3000-5000, 序列 2048), 激活成为主要瓶颈
- 1B 模型 batch=8, seq=2048, 8 层 MLP 激活峰值 = 8 × 5000 × 2048 × 8 × 2 字节 ≈ 1.2 GB
- 选择式 ckpt 后降到 ~600MB

**不集成的代价**:

- 1B+ Dense 模型在 24GB 卡上 batch 受限 (只能 batch=4 而非 8)
- 1B+ MoE 模型 expert 中间激活更大 (每个 expert 独立计算), 更需要 ckpt
- 与 FA2 不冲突, 与 torch.compile 兼容性良好

**典型收益** (基于公开基准):
- 选择式 ckpt: 激活 -40-50%, 速度 -5-8%
- 全量 ckpt: 激活 -60-70%, 速度 -20-30%

---

## 3. 架构设计

### 3.1 选择式 checkpoint 流程

```text
MiniMindBlock.forward(x):
    residual = x
    x_norm = input_layernorm(x)
    
    # Attention 块: 不启用 checkpoint (FA2 已是 O(N))
    attn_out, present_kv = self.self_attn(x_norm, ...)
    attn_out = attn_out + residual  # 残差连接
    
    # MLP 块: 启用 checkpoint (大激活, 真正需要重算的层)
    residual = attn_out
    x_norm = post_attention_layernorm(attn_out)
    
    if self.gradient_checkpointing and self.training and not use_cache:
        mlp_out = torch.utils.checkpoint.checkpoint(
            self.mlp, x_norm, use_reentrant=False
        )
    else:
        mlp_out = self.mlp(x_norm)
    
    return mlp_out + residual, present_kv
```

### 3.2 关键设计决策

| 决策点 | 选项 | 推荐 |
|--------|------|------|
| checkpoint 范围 | 全量 / 选择 MLP / 选择 Attention | **选择 MLP** |
| reentrant 参数 | True / False | **False** (PyTorch 2.1+ 推荐) |
| use_cache 兼容 | 是 / 否 | **use_cache=True 时禁用** |
| 默认状态 | 关闭 / 开启 | **关闭** (与原行为一致), CLI flag 启用 |

### 3.3 与 torch.compile 的兼容性

- `torch.utils.checkpoint` 在 `torch.compile` 下行为:
  - **Functional checkpoint** (`use_reentrant=False`): 完全兼容, 编译可正常追踪
  - **Reentrant checkpoint** (`use_reentrant=True`): 与 `torch.compile` 不兼容, 必须设为 False
- MiniMind 训练脚本中如启用 `torch.compile`, grad-ckpt 必须用 `use_reentrant=False`

---

## 4. 方案实现

### 4.1 `MiniMindConfig` 增加开关

```python
# model/model_minimind.py: MiniMindConfig.__init__
self.gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
```

### 4.2 `MiniMindBlock` 实现

```python
# model/model_minimind.py:295-318
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        # ... existing init ...
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)
    
    def forward(self, hidden_states, position_embeddings, 
                past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        
        # 选择式 checkpoint: 仅对 MLP 启用
        if self.gradient_checkpointing and self.training and not use_cache:
            from torch.utils.checkpoint import checkpoint
            hidden_states = checkpoint(self.mlp, normed, use_reentrant=False)
        else:
            hidden_states = self.mlp(normed)
        
        hidden_states = hidden_states + residual
        return hidden_states, present_key_value
```

### 4.3 `MiniMindModel` 同步

```python
# model/model_minimind.py: MiniMindModel
def set_gradient_checkpointing(self, enable: bool):
    """Enable/disable gradient checkpointing for all blocks."""
    for block in self.layers:
        block.gradient_checkpointing = enable
```

### 4.4 `train_pretrain.py` CLI

```python
# trainer/train_pretrain.py: argparse
parser.add_argument('--grad_checkpoint', default=0, type=int, choices=[0, 1],
                    help='是否启用梯度检查点(0=否, 1=是, 仅对MLP)')

# 在 init_model 之后:
if args.grad_checkpoint == 1:
    model.set_gradient_checkpointing(True)
    Logger('Selective gradient checkpointing enabled (MLP only)')
```

### 4.5 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--grad_checkpoint` | `0` | 关闭, 与原行为一致 |
| `use_reentrant` | `False` | 与 torch.compile 兼容必需 |
| 范围 | MLP only | Attention 不启用 |

---

## 5. 训练过程影响

| 维度 | 影响 |
|------|------|
| 显存 | 激活 -40-50% (MLP 激活为主) |
| 速度 | -5-8% (一次额外 MLP 前向) |
| 数值 | **无影响** (重算等价于保存) |
| 优化器 | 无影响 |
| 梯度累积 | 无影响 (每个 micro-batch 独立) |
| DDP 兼容 | **有要求** (需 `model.no_sync()` 包裹未 sync 的 micro-batch) |
| torch.compile | 需 `use_reentrant=False` |

---

## 6. 消融实验方案

### 6.1 64M Dense 验证基线

- **配置**: `hidden_size=768, num_layers=8, batch=32, seq=340, 1 epoch on pretrain_t2t_mini`
- **对照**: `--grad_checkpoint 0` vs `--grad_checkpoint 1`
- **指标**:
  - 显存峰值 (nvidia-smi dmon, MB)
  - step/s (前 100 步平均, 排除 warmup)
  - loss 曲线 (前 200 步)
- **预期**: 显存 -10-20%, 速度 -3-5%, loss 曲线**完全一致**

### 6.2 1B Dense 验证扩展性

- **配置**: `hidden_size=1536, num_layers=24, batch=8, seq=2048, 100 步`
- **对照**: `--grad_checkpoint 0` vs `--grad_checkpoint 1`
- **指标**: 显存峰值, step/s, loss 曲线
- **预期**: 显存 -30-40%, 速度 -5-8%, loss 曲线**完全一致**

### 6.3 与 torch.compile 联合验证

- **配置**: 在 6.1/6.2 基础上同时开 `--use_compile 1`
- **预期**: grad-ckpt 收益不变, compile 提速 1.3-1.5x **叠加**

---

## 7. 已知问题与限制

### 7.1 与 use_cache 的冲突

- `use_cache=True` 时 (推理时), **不能**启用 grad-ckpt (会破坏 KV cache 重建)
- 当前实现用 `if ... and not use_cache` 保护
- 训练时 `use_cache` 通常为 False, 无影响

### 7.2 与 DDP 梯度累积的交互

- 训练脚本 `train_pretrain.py:42` 用 `args.accumulation_steps=8` 做梯度累积
- DDP 下, 每个 micro-batch 应该是独立的 `forward` 调用
- grad-ckpt 在 DDP 下需要 `model.no_sync()` 包裹未同步的 micro-batch (PyTorch DDP 设计)
- MiniMind 当前 DDP 实现可能没处理, **需要测试**
- 风险等级: 中

### 7.3 激活重算的副作用

- 每次重算会产生新的随机数 (dropout, 如果开启)
- `use_reentrant=False` 已自动处理 RNG state
- MiniMind 默认 `dropout=0`, 无影响

### 7.4 内存峰值不一定降低

- 重算时, 临时需要额外前向, 瞬时显存可能比不 ckpt 还高 (在 batch 极大时)
- 经验值: batch < 32, seq < 4096 时 grad-ckpt 总能降显存

---

## 8. 后续改进方向

1. **每层独立开关**: 不同层用不同策略 (浅层不 ckpt, 深层 ckpt)
2. **动态检查点**: 根据当前显存压力自适应决定哪些层 ckpt
3. **与 Activation Offloading 组合**: 见 [`06-activation-offload.md`](06-activation-offload.md)

---

## 9. 参考文献

- **原论文**: Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin, "Training Deep Nets with Sublinear Memory Cost" (2016) - [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- **PyTorch 文档**: [`torch.utils.checkpoint`](https://pytorch.org/docs/stable/checkpoint.html)
- **HuggingFace 实践**: [Gradient Checkpointing in HuggingFace Transformers](https://huggingface.co/docs/transformers/en/optimization#gradient-checkpointing)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 定义 P0 选择式 grad-ckpt 集成方案 | Sisyphus |

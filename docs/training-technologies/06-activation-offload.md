# 06 · Activation Offloading (激活卸载)

> **状态**: 📝 讨论中 | **优先级**: P1
> **代码位置 (待修改)**: `model/model_minimind.py:295-318` (`MiniMindBlock`)
> **依赖**: `accelerate >= 0.30` (Accelerate 提供 `ActivationOffloading` context manager)
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

**Activation Offloading** 是一种更细粒度的显存优化: 在前向计算时, **把"当前不需要"的中间激活从 GPU 卸载到 CPU 内存**, 反向传播时再按需加载。

**与梯度检查点的对比**:

| 方案 | 显存节省 | 速度开销 | 粒度 |
|------|----------|----------|------|
| 无 | 0 | 0 | - |
| **Grad-CKPT (P0)** | 激活 -50% | -5-10% | **重算式** (重算时不需要原激活) |
| **Activation Offload (P1)** | 激活 -50-80% | -10-30% (PCIe 通信) | **存储式** (不重算, 但要搬) |

**两种互补**: 实际工程中常组合使用 —— 对**重算开销大**的层 (Attention) 用 offload, 对**重算开销小**的层 (MLP) 用 grad-ckpt

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- 64M 模型: 激活只占 ~150MB, **不需要** activation offload
- 1B+ 模型 + 长序列 (seq >= 2048) + 训练时, 激活是主要瓶颈
- 1B 模型 + batch=8 + seq=2048: 激活峰值 ~3-4 GB
- grad-ckpt 把它降到 ~1.5GB; **activation offload 可以进一步降到 ~500MB**
- 极致场景: 2B+ 模型 + seq=4096, 必须用 offload

**不集成的代价**:

- 1B+ 模型 + 长序列训练时, batch 受限 (无法 batch=8, 只能 batch=4)
- 1.5B+ 模型 seq=2048 可能 OOM

**典型收益**:

- 激活 -50-80% (相比无优化)
- 速度 -10-30% (PCIe 4.0 通信)
- 相比 grad-ckpt 单独用, 进一步节省激活 30-40%

---

## 3. 架构设计

### 3.1 Accelerate 的 Activation Offloading

```python
from accelerate import Accelerator
from accelerate.hooks import attach_align_device_hook, AlignDevicesHook

# 方式 1: Context manager (推荐)
from accelerate import init_empty_weights

# 方式 2: Per-module hook
model.self_attn = attach_align_device_hook(
    model.self_attn, 
    offload=True,  # 把该模块的输出 offload 到 CPU
    place_submodules=True,
)
```

### 3.2 手动实现 (避免 Accelerate 依赖)

```python
import torch

class ActivationOffloadHook:
    """把模块的输出 tensor 在 forward 后搬到 CPU, backward 时再搬回"""
    def __init__(self, offload_to_cpu=True):
        self.offload_to_cpu = offload_to_cpu
    
    def __call__(self, module, inputs, output):
        # 不在 eval 模式下启用
        if not module.training:
            return output
        
        if self.offload_to_cpu:
            # 把 output 搬到 CPU, 但保留 backward 所需的信息
            if isinstance(output, tuple):
                new_output = tuple(
                    t.cpu() if torch.is_tensor(t) else t for t in output
                )
            else:
                new_output = output.cpu()
            
            # 注册 hook: backward 开始时搬回 GPU
            for t_orig, t_new in zip(
                output if isinstance(output, tuple) else (output,),
                new_output if isinstance(new_output, tuple) else (new_output,),
            ):
                if torch.is_tensor(t_orig):
                    t_new.requires_grad_(t_orig.requires_grad)
                    def _pre_backward(grad):
                        return t_orig  # 让 autograd 用原 GPU tensor
                    torch.autograd.graph.register_hook(_pre_backward)
            
            return new_output
        return output
```

**问题**: 上述手写实现非常复杂, **强烈推荐用 Accelerate 的现成实现**。

### 3.3 与 grad-ckpt 的协同

```text
# 推荐组合 (在 MiniMindBlock 中)
def forward(self, hidden_states, ...):
    if self.gradient_checkpointing:
        # MLP 用 grad-ckpt (重算)
        mlp_out = checkpoint(self.mlp, normed, use_reentrant=False)
    else:
        # 配合 activation offload (存储)
        mlp_out = self.mlp(normed)
    
    return mlp_out + residual
```

**推荐配置**:
- **P0 阶段**: grad-ckpt 开, activation offload 关 (简单, 提速 1.3x)
- **P1 阶段**: grad-ckpt 开, activation offload 选 Attention (最激进)
- **P2 阶段**: grad-ckpt 关, 纯 activation offload (适合极大 batch)

---

## 4. 方案实现

### 4.1 `requirements.txt` 增加依赖

```text
# requirements.txt
accelerate>=0.30.0  # 已被 05-accelerate-offload 引入
```

### 4.2 `model/model_minimind.py` 集成 hook

```python
# model/model_minimind.py
def apply_activation_offload(model, offload_layers='all'):
    """
    为 MiniMind 的 Attention / MLP 块挂上 activation offload hook
    offload_layers: 'all' | 'attention' | 'mlp' | 'none'
    """
    from accelerate.hooks import attach_align_device_hook
    
    for block in model.layers:
        if offload_layers in ('all', 'attention'):
            block.self_attn = attach_align_device_hook(
                block.self_attn,
                offload=True,
                place_submodules=True,
            )
        if offload_layers in ('all', 'mlp'):
            block.mlp = attach_align_device_hook(
                block.mlp,
                offload=True,
                place_submodules=True,
            )
```

### 4.3 `train_pretrain.py` CLI

```python
# trainer/train_pretrain.py: argparse
parser.add_argument('--activation_offload', default='none', type=str,
                    choices=['none', 'all', 'attention', 'mlp'],
                    help='激活卸载到 CPU (none=关闭, all=全部, attention=仅 Attention, mlp=仅 MLP)')

# 在 init_model 之后:
if args.activation_offload != 'none':
    from model.model_minimind import apply_activation_offload
    apply_activation_offload(model, offload_layers=args.activation_offload)
    Logger(f'Activation offload enabled: {args.activation_offload}')
```

### 4.4 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--activation_offload` | `none` | 关闭, 与原行为一致 |
| `offload_layers` | `all` | 全部模块都 offload |
| `place_submodules` | `True` | 递归处理子模块 |
| 与 `use_cache` | 不兼容 | use_cache=True 时自动禁用 |

### 4.5 与 torch.compile 的兼容性

- Activation offload 用了 hook 机制, 可能在 torch.compile 编译时被绕过
- **建议**: 不要同时启用 torch.compile + activation offload
- 二选一

---

## 5. 训练过程影响

| 维度 | 影响 |
|------|------|
| 显存 (激活 GPU) | **-50-80%** |
| 显存 (CPU 内存) | +激活 (1B + seq=2048 + batch=8 ~1-2GB) |
| 速度 | **-10-30%** (PCIe 通信, 取决于序列长度) |
| 数值 | 无影响 (搬运等价) |
| 优化器 | 无影响 |
| 与 use_cache | 不兼容 (推理时不启用) |
| 与 torch.compile | 不兼容 (二选一) |

---

## 6. 消融实验方案

### 6.1 1B + grad-ckpt + activation offload 联合

- **配置**: `hidden_size=1536, num_layers=24, batch=8, seq=2048, 100 步`
- **对照**:
  - `--grad_checkpoint 0 --activation_offload none` (基线, 预期 OOM)
  - `--grad_checkpoint 1 --activation_offload none` (P0 路径)
  - `--grad_checkpoint 0 --activation_offload mlp` (P1 路径)
  - `--grad_checkpoint 1 --activation_offload attention` (P0+P1 路径, 推荐)
- **指标**: 显存峰值, step/s, loss 曲线
- **预期**:
  - grad-ckpt + attention-offload 显存最低
  - 速度 -15-25%
  - loss 曲线**完全一致**

### 6.2 1.5B 边界测试

- **配置**: `hidden_size=1792, num_layers=24, batch=4, seq=4096, 100 步`
- **对照**:
  - 仅 grad-ckpt: 预期 OOM
  - grad-ckpt + activation_offload all: 预期可跑
- **指标**: max seq_len, max batch_size
- **预期**: 1.5B 模型在 24GB 卡上 batch=4 + seq=4096 可跑

### 6.3 CPU 内存峰值测试

- 监控 `/proc/meminfo` 的 RSS
- 1B + batch=8 + seq=2048 + all-offload 预计 CPU 占用 +2-3GB
- 验证: 系统总内存至少 16GB 推荐

---

## 7. 已知问题与限制

### 7.1 与 torch.compile 冲突

- torch.compile 试图捕获整个 forward 的计算图
- Activation offload 在中间插入 CPU 搬运, 打破计算图
- **结论**: 两者**二选一**, 不要同时启用

### 7.2 与 DDP 兼容性问题

- Accelerate 的 `attach_align_device_hook` 在 DDP 包装之前/之后都可能有问题
- 单卡场景稳定
- 多卡 DDP 场景需要测试, 可能有 device 不匹配错误
- 推荐: 多卡 + offload 时用 FSDP2 (见 [`08-fsdp2.md`](08-fsdp2.md)), 不用 DDP

### 7.3 PCIe 带宽瓶颈

- 与 [`05-accelerate-offload.md`](05-accelerate-offload.md) 同样的 PCIe 限制
- 长序列训练时, 每次 step 的 PCIe 通信量与激活总量成正比
- 1B + seq=2048 + batch=8: 每次 step 搬运 ~1.5GB, 耗时 ~50-100ms
- 1.5B + seq=4096 + batch=4: 每次 step 搬运 ~2-3GB, 耗时 ~100-200ms

### 7.4 CPU 内存需求

- 1B + seq=2048 + batch=8 + all-offload: CPU 占用 +1-2GB
- 1.5B + seq=4096 + batch=4 + all-offload: CPU 占用 +2-3GB
- 推荐工作机: 32GB+ RAM

### 7.5 与 use_cache 不兼容

- 推理时 (`generate`, `use_cache=True`) 不能用 activation offload
- 会破坏 KV 缓存的连续性
- 当前实现: `if not use_cache` 时启用 (但 MiniMindBlock 不感知 use_cache, 需在更高层判断)

---

## 8. 后续改进方向

1. **与 FSDP2 集成**: FSDP2 自带 activation offload 的官方支持, 性能更好
2. **异步预取**: 下一层 forward 启动前, 异步把下一层的激活从 CPU 搬到 GPU
3. **NVMe Offload**: 适合 7B+ 模型的极端场景, MiniMind 不需要

---

## 9. 参考文献

- **Accelerate Hooks 文档**: [Accelerate Hooks](https://huggingface.co/docs/accelerate/concept_guides/performance#hooks)
- **Activation Offloading 论文**: Zhenyu Liu et al., "Memory-Efficient Pipeline-Parallel DNN Training" (2020) - 早期工作
- **PyTorch Checkpoint + Offload 实践**: [PyTorch Activation Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- **DeepSpeed Activation Checkpointing**: [DeepSpeed Docs](https://www.deepspeed.ai/tutorials/activation-checkpointing/)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 定义 P1 Activation Offload 集成方案 | Sisyphus |

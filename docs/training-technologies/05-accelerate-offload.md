# 05 · Accelerate 单卡 CPU Offload

> **状态**: 📝 讨论中 | **优先级**: P1
> **代码位置 (待修改)**: `trainer/train_pretrain.py` 启动方式
> **依赖**: `accelerate >= 0.30`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

**Accelerate** (HuggingFace) 是一个**训练启动器**库, 抽象了 DDP / FSDP / DeepSpeed / 单卡等多种启动方式, 用户用一份代码即可在多配置间切换。

**CPU Offload** 是其中一种**单卡适用**的优化: 把优化器状态卸载到 CPU 内存, 每次 optimizer step 时再搬运回 GPU。

**为什么不直接用 DeepSpeed Offload**:

- DeepSpeed 需要专门的 `ds_config.json` + 启动器
- Accelerate 集成更简单, `accelerate launch` + 一个 yaml 配置即可
- Accelerate 兼容 PyTorch FSDP, 升级路径更顺畅

**与 bitsandbytes 8-bit AdamW 的关系**:

| 方案 | 优化器状态位置 | 通信开销 | 显存节省 |
|------|----------------|----------|----------|
| 原始 AdamW (FP32) | GPU | 0 | 0 |
| **8-bit AdamW (P0)** | GPU (8-bit 压缩) | 0 | 75% |
| **CPU Offload (P1)** | CPU (FP32) | PCIe | ~95% |
| CPU Offload + 8-bit | CPU (8-bit) | PCIe | ~99% |

**推荐组合**: 8-bit AdamW + CPU Offload = 极限优化器显存节省, 1.5B+ 模型单卡可训

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- 64M 模型: 优化器状态 512MB, **不需要** CPU offload (8-bit AdamW 够用)
- 500M-1B 模型: 8-bit AdamW 够用, offload 是可选项
- **1.5B+ 模型**: 即使 8-bit AdamW 也占 3GB, 加 offload 后 = 0GB GPU, 让 1.5B-2B 在 24GB 训练

**不集成的代价**:

- 1.5B+ Dense 模型单卡 24GB 训练时, 优化器状态仍是瓶颈
- 与 bnb 8-bit 不冲突, 但 offload 是更激进的方案

**典型收益**:

- 优化器状态 GPU 占用从 8GB (1B FP32) → 0GB (offload)
- step 时间 +20-50% (PCIe 4.0 通信)

---

## 3. 架构设计

### 3.1 启动流程

```text
# 方式 1: Accelerate CLI (推荐)
accelerate launch --config_file accelerate_offload_config.yaml train_pretrain.py

# 方式 2: 代码内启动 (不需要 CLI)
from accelerate import Accelerator
accelerator = Accelerator(cpu_offload_optimizer=True, cpu_offload=True)
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
```

### 3.2 accelerate_offload_config.yaml

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16  # 与 --dtype bfloat16 对齐
num_processes: 1
num_machines: 1
machine_rank: 0
main_training_function: main
use_cpu: false  # 主计算在 GPU

# 关键: CPU offload
cpu_offload_optimizer: true
cpu_offload: false  # 是否也 offload 参数 (激进, 慢, 通常不需要)

debug: false
```

### 3.3 优化器状态生命周期

```text
optimizer.step():
    1. 把 optimizer.state 搬到 GPU (一次性, 256MB-2GB)
    2. 在 GPU 上做 m, v, grad 计算 → 更新参数
    3. 把 optimizer.state 搬回 CPU
    4. 清空 GPU 上的临时 buffer
    
    ↑ 每次 step 都有 2 次 PCIe 传输
    ↑ 延迟: 1B 模型 ~50-100ms / step (PCIe 4.0 32 GB/s)
```

### 3.4 与 DDP 的关系

- 单卡 (num_processes=1) 场景, `cpu_offload_optimizer` 是 `torch.optim` 级别的优化
- Accelerate 在 DDP 模式下会与各 rank 同步, offload 仍生效
- **多卡 + offload + FSDP2** 是进阶组合, 见 [`08-fsdp2.md`](08-fsdp2.md)

---

## 4. 方案实现

### 4.1 `requirements.txt` 增加依赖

```text
# requirements.txt
accelerate>=0.30.0
```

### 4.2 新建 `accelerate_offload_config.yaml`

```yaml
# 项目根目录: accelerate_offload_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
num_processes: 1
num_machines: 1
use_cpu: false
cpu_offload_optimizer: true
cpu_offload: false
```

### 4.3 `trainer_utils.py` 增加 `init_accelerate` 工具

```python
# trainer/trainer_utils.py
def init_accelerate(cpu_offload_optimizer: bool = False, mixed_precision: str = 'bf16'):
    """Initialize Accelerate with optional CPU offload."""
    from accelerate import Accelerator
    accelerator = Accelerator(
        cpu_offload_optimizer=cpu_offload_optimizer,
        mixed_precision=mixed_precision if mixed_precision != 'no' else 'no',
    )
    return accelerator
```

### 4.4 `train_pretrain.py` 集成 (与现有 DDP 逻辑共存)

```python
# trainer/train_pretrain.py
# 现有:
local_rank = init_distributed_mode()
if dist.is_initialized(): args.device = f"cuda:{local_rank}"

# 改为 (优先用 accelerate, 回退到现有 DDP):
if int(os.environ.get("USE_ACCELERATE", 0)) == 1:
    from accelerate import Accelerator
    accelerator = Accelerator(
        cpu_offload_optimizer=(args.optimizer_offload == 1),
        mixed_precision='bf16' if args.dtype == 'bfloat16' else 'fp16',
    )
    args.device = accelerator.device
    # 注意: model / optimizer / loader 都要 accelerator.prepare(...)
    model, optimizer, train_loader, train_sampler = accelerator.prepare(
        model, optimizer, train_loader, train_sampler
    )
else:
    # 现有 DDP 逻辑
    local_rank = init_distributed_mode()
    ...
```

### 4.5 CLI 改动

```python
# trainer/train_pretrain.py: argparse
parser.add_argument('--optimizer_offload', default=0, type=int, choices=[0, 1],
                    help='是否将优化器状态卸载到CPU (需要 bitsandbytes + accelerate)')

# 启动方式
# 推荐 (Accelerate):
#   USE_ACCELERATE=1 accelerate launch --config_file accelerate_offload_config.yaml \
#       trainer/train_pretrain.py --optimizer_offload 1
```

### 4.6 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_ACCELERATE` env | `0` | 关闭, 保持原行为 |
| `--optimizer_offload` | `0` | 关闭 |
| `cpu_offload_optimizer` (yaml) | `true` | 推荐开 |
| `cpu_offload` (yaml) | `false` | 不 offload 参数 (激进) |
| `mixed_precision` | `bf16` | 与 `--dtype` 对齐 |

---

## 5. 训练过程影响

| 维度 | 影响 |
|------|------|
| 显存 (优化器状态 GPU) | **-95-99%** (移到 CPU) |
| 显存 (CPU 内存) | +优化器状态 (1B 模型 ~8GB CPU 内存) |
| 速度 | **-20-50%** (PCIe 通信) |
| 数值 | 无影响 (FP32 状态在 CPU) |
| 优化器 | 仅 AdamW 系列支持, 不支持 SGD 等 |
| 检查点 | 状态在 CPU, 加载时仍在 CPU, 首次 step 搬到 GPU |
| 与 bnb 8-bit 兼容 | ✅ (可叠加, 极限优化) |

---

## 6. 消融实验方案

### 6.1 64M 基线测试 (验证可行性)

- **配置**: `hidden_size=768, batch=32, seq=340, 1 epoch on pretrain_t2t_mini`
- **对照**:
  - `--optimizer adamw` + 原 DDP 启动 (基线)
  - `--optimizer adamw` + Accelerate + cpu_offload_optimizer=true
- **指标**: step/s, 显存峰值, loss 曲线
- **预期**:
  - 64M 优化器状态只占 512MB, offload 收益小
  - 速度 -5-10% (PCIe 通信在 64M 上占比小)
  - loss 曲线**完全一致**

### 6.2 1B 验证 (核心场景)

- **配置**: `hidden_size=1536, num_layers=24, batch=8, seq=2048, 100 步`
- **对照**:
  - `--optimizer adamw_8bit` (无 offload): 预期 batch=8 OK
  - `--optimizer adamw_8bit` + `cpu_offload_optimizer=true`: 显存更低, 速度更慢
  - `--optimizer adamw` + `cpu_offload_optimizer=true`: 显存更低, 速度最慢
- **指标**: 显存峰值, step/s
- **预期**:
  - offload 让 1.5B-2B 也能 batch=8 跑通
  - 速度 -20-40%

### 6.3 1.5B 边界测试 (目标)

- **配置**: `hidden_size=1792, num_layers=24, batch=4, seq=2048, 100 步`
- **对照**:
  - 不开 offload: OOM
  - 开 offload + 8-bit AdamW: 预期可跑
- **指标**: 显存峰值, max batch size
- **预期**: 1.5B Dense 在 24GB 卡上 batch=4 可跑

---

## 7. 已知问题与限制

### 7.1 PCIe 带宽瓶颈

- RTX 4090 用 PCIe 4.0 x16, 理论 32 GB/s, 实际 25-28 GB/s
- 1B 模型优化器状态 = 8GB, 每次 step 来回 = 16GB, 耗时 ~600ms
- 1.5B 模型: 12GB 状态, 耗时 ~900ms
- **结论**: offload 在小模型上不划算, 1.5B+ 才有意义

### 7.2 CPU 内存要求

- 1.5B 模型 FP32 优化器 = 12GB
- 1.5B 模型 BF16 优化器 + offload = 12GB
- 2.5B+ 模型: 需要 32GB+ CPU 内存
- 4090 工作站通常 64GB+ RAM, 够用

### 7.3 优化器限制

- 仅 `torch.optim.AdamW`, `torch.optim.Adam`, `bnb.optim.AdamW8bit` 支持
- **不支持** SGD (无状态可 offload)
- **不支持** LAMB 等复杂优化器 (状态结构复杂)

### 7.4 与 DDP 梯度累积的交互

- 梯度累积时, offload 在每个 micro-batch 之间不发生
- 累积完成后, optimizer.step() 触发一次 offload
- 不会引入额外 PCIe 通信, 性能影响可控

### 7.5 启动方式切换

- 现有 MiniMind 用户用 `python train_pretrain.py` 或 `torchrun`
- 启用 Accelerate 需要用 `accelerate launch`, **改变启动方式**
- 需要在 README 中明确文档, 提供切换指南

---

## 8. 后续改进方向

1. **NVMe Offload**: 把冷状态搬到 NVMe SSD, 适合 7B+ 模型 (QLoRA 场景), MiniMind 不需要
2. **与 FSDP2 集成**: 多卡场景下, FSDP2 + cpu_offload 是终态, 见 [`08-fsdp2.md`](08-fsdp2.md)
3. **智能 offload**: 根据当前显存压力自适应决定哪些 buffer 在 CPU

---

## 9. 参考文献

- **Accelerate 官方文档**: [huggingface/accelerate](https://huggingface.co/docs/accelerate/index)
- **DeepSpeed Offload 论文**: Jie Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training" (USENIX ATC 2021) - [paper](https://www.usenix.org/conference/atc21/presentation/ren-jie)
- **Accelerate cpu_offload API**: [documentation](https://huggingface.co/docs/accelerate/concept_guides/performance#cpu-offload)
- **PyTorch 官方支持**: PyTorch 2.4+ 已有 `torch.optim` 级别的 offload 原语, Accelerate 包装

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 定义 P1 Accelerate CPU Offload 集成方案 | Sisyphus |

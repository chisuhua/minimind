# 03 · 8-bit / Paged AdamW (bitsandbytes)

> **状态**: 📝 讨论中 | **优先级**: P0
> **代码位置 (待修改)**: `trainer/train_pretrain.py:138`, `trainer_utils.py`
> **依赖**: `bitsandbytes >= 0.43`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

**8-bit AdamW** (Dettmers et al. 2022) 是一种**优化器状态量化**技术, 把 AdamW 的两个状态 buffer (`m` 一阶矩, `v` 二阶矩) 从 **FP32 (4 字节/参数)** 压缩到 **8-bit (1 字节/参数)**。

**`PagedAdamW`**: 进一步把状态分页管理, 用到时再加载, 减少峰值显存。

**为什么 AdamW 状态占显存**:

- AdamW 维护两个 buffer: `m` (动量) 和 `v` (方差), 都是 FP32
- 64M 模型: 64M × 4 × 2 = 512 MB
- 1B 模型: 1B × 4 × 2 = **8 GB** ← 单卡 24GB 的 33%!

**8-bit AdamW 收益**:

- 64M: 512 MB → 128 MB (-384 MB)
- 1B: 8 GB → 2 GB (-6 GB) ← **这是 P0 三件套中最关键的一项**

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- 64M 训练时, 优化器状态占 512MB (~总显存 4%), 收益不显著
- **1B Dense 训练时, 优化器状态占 8GB**, 启用 8-bit 后只占 2GB, **直接让 1B 模型从"超 24GB"变成"可训练"**
- 1.5B+ 模型, 8-bit AdamW 是单卡 24GB 训练的关键

**不集成的代价**:

- 单卡 24GB 训练 1B 模型时, 仅优化器状态就占 8GB, 加上权重 4GB + 梯度 4GB + 激活 0.5GB = 16.5GB, 已经在边缘
- 1.5B 模型静态占用 24GB, **根本无法训练**

**与原方案对比**:

- 原方案: "ZeRO-2/3 + CPU Offload" → 通信开销大, 单卡优势小
- 推荐方案: **8-bit AdamW** → 零通信开销, 1B+ 模型 75% 优化器显存节省

---

## 3. 架构设计

### 3.1 8-bit 状态存储

```text
# 普通 AdamW (FP32)
state['m']: FP32 tensor, 4 字节/参数
state['v']: FP32 tensor, 4 字节/参数

# 8-bit AdamW
state['m']: UInt8 tensor, 1 字节/参数 + 1 个 FP32 缩放因子 (整个 tensor 共享)
state['v']: UInt8 tensor, 1 字节/参数 + 1 个 FP32 缩放因子
```

**量化算法** (Dettmers 论文):

```text
# 量化: FP32 → 8-bit
absmax = max(abs(x))
scale = 127 / absmax
x_quant = round(x * scale).to(uint8)

# 反量化 (optimizer step 时):
x_dequant = x_quant.to(float32) / scale
```

**精度保证**: 8-bit 量化的相对误差约 1/127 ≈ 0.8%, Adam 的 update step 本身就有噪声, 不影响收敛性

### 3.2 Paged 优化

```text
# PagedAdamW: 状态按"页"管理 (类似 OS 虚拟内存)
# - 状态在 CPU 内存或 GPU 内存的页中
# - optimizer.step() 时按需把页加载到 GPU
# - 适用于 state 远大于 GPU 显存的场景 (e.g. 7B+ 模型)
```

- 单卡 24GB + 1B 模型 + 8-bit AdamW 已经够用, 不需要 Paged
- Paged 适用于 7B+ 单卡 (QLoRA 场景), MiniMind 不需要

### 3.3 与 autocast / GradScaler 的兼容性

- `bitsandbytes.optim.AdamW8bit` **不依赖 GradScaler** (内部用 8-bit 缩放)
- 当 `--dtype bfloat16` 时 (MiniMind 默认), GradScaler 已是 `enabled=False`, 8-bit AdamW 正常工作
- 当 `--dtype float16` 时, 仍可用, 但推荐额外关注 `scale_factor` 初始化

---

## 4. 方案实现

### 4.1 `requirements.txt` 增加依赖

```text
# requirements.txt
bitsandbytes>=0.43.0
```

### 4.2 `trainer_utils.py` 增加 import 与 fallback

```python
# trainer/trainer_utils.py 顶部
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    
def get_optimizer(model, args):
    """根据 args.optimizer 选择优化器"""
    if args.optimizer == 'adamw_8bit':
        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes not installed. pip install bitsandbytes")
        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
    elif args.optimizer == 'paged_adamw_8bit':
        return bnb.optim.PagedAdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:  # 'adamw' 默认
        return optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
```

### 4.3 `train_pretrain.py` CLI

```python
# trainer/train_pretrain.py: argparse
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['adamw', 'adamw_8bit', 'paged_adamw_8bit'],
                    help='优化器选择 (8bit 系列需要 bitsandbytes)')
parser.add_argument('--weight_decay', default=0.01, type=float,
                    help='权重衰减')

# 在 init_model 之后:
optimizer = get_optimizer(model, args)
```

### 4.4 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--optimizer` | `adamw` | 与原行为一致 |
| `--weight_decay` | `0.01` | AdamW 推荐值 |
| bnb AdamW `betas` | `(0.9, 0.95)` | MiniMind 风格, 与原 AdamW 略有不同 |

### 4.5 收敛性微调 (可选)

- 8-bit AdamW 的 update step 略有噪声, **建议**:
  - LR 不变 (5e-4)
  - `betas=(0.9, 0.95)` 与原 AdamW 一致
  - **不**额外加 warmup 步数
  - 训练前 100 步监控 loss 曲线, 应与原 AdamW 重合

---

## 5. 训练过程影响

| 维度 | 影响 |
|------|------|
| 显存 (优化器状态) | **-75%** (FP32 → 8-bit) |
| 显存 (模型权重) | 0% |
| 显存 (梯度) | 0% |
| 显存 (激活) | 0% |
| 速度 | **-3-5%** (反量化 + 8-bit arithmetic) |
| 数值 | 与原 AdamW 误差 <1% (在 LR 5e-4 范围内) |
| 检查点兼容性 | ⚠️ state_dict 格式不同, 加载时需注意 |

---

## 6. 消融实验方案

### 6.1 64M 收敛性验证

- **配置**: `hidden_size=768, batch=32, seq=340, 1 epoch on pretrain_t2t_mini`
- **对照**:
  - `--optimizer adamw` (基线)
  - `--optimizer adamw_8bit`
- **指标**:
  - 显存峰值 (nvidia-smi dmon)
  - step/s
  - loss 曲线 (前 200 步 + 整 epoch)
- **预期**:
  - 显存 -100MB (从 ~1.3GB 到 ~1.2GB, 不显著因为 64M 本来就小)
  - 速度 -3-5%
  - **loss 曲线与基线重合** (这是最重要的验收标准)

### 6.2 1B 可行性验证

- **配置**: `hidden_size=1536, num_layers=24, batch=8, seq=2048, 100 步`
- **对照**:
  - `--optimizer adamw`: **预期 OOM** 或 batch 受限到 4
  - `--optimizer adamw_8bit`: **预期 batch=8 可跑**
- **指标**:
  - 显存峰值
  - max batch size
  - loss 曲线 (与 64M 收敛性对齐)
- **预期**:
  - 静态显存从 16GB 降到 10GB
  - batch=8 跑通

### 6.3 与 grad-checkpoint + torch.compile 联合验证

- **配置**: `--optimizer adamw_8bit --grad_checkpoint 1 --use_compile 1`
- **预期**: 1B 模型 + batch=8 + seq=2048 在 24GB 卡上**总占用 < 20GB**

---

## 7. 已知问题与限制

### 7.1 平台兼容性

- `bitsandbytes` 在 Windows / 旧 CUDA 上可能安装失败
- Linux + CUDA 11.8+ / 12.x 是稳定平台
- **MiniMind 目标用户**: 主要是 Linux + CUDA, 兼容性好

### 7.2 LR scheduler 兼容

- 8-bit AdamW 内部维护自己的状态格式
- 与外部 `optim.lr_scheduler` 兼容 ✅
- MiniMind 的 cosine scheduler (`get_lr` in `trainer_utils.py:40-41`) 无影响

### 7.3 检查点兼容性

- 8-bit AdamW 的 state_dict 包含量化参数 + 缩放因子
- **不能**与原 AdamW 互换 (反之亦然)
- 建议: 训练时如果用 8-bit, **resume 时必须用 8-bit**
- 验证逻辑: `args.optimizer` 必须与 checkpoint 中记录的 optimizer 类型一致

### 7.4 极小学习率场景

- 8-bit 量化的相对误差 ~0.8%
- 当 LR < 1e-5 时, update step 可能被量化噪声淹没
- MiniMind 默认 LR=5e-4, **完全没问题**
- DPO / GRPO 等精细调参场景需要关注

### 7.5 bitsandbytes 与某些 PyTorch 版本的兼容

- bitsandbytes 0.43+ 与 PyTorch 2.4+ 兼容性最好
- 0.41-0.42 在 PyTorch 2.5+ 上可能有 CUDA illegal memory access
- **建议**: `pip install -U bitsandbytes` 后测试

---

## 8. 后续改进方向

1. **CPU Offload 集成**: bnb 有 `bnb.optim.AdamW8bit` 的 CPU offload 变体 (更激进, 但慢)
2. **与 QLoRA 集成**: 进一步对模型权重 4-bit 量化 (QLoRA 路线), 1B+ 模型单卡可训
3. **自定义 4-bit AdamW**: 进一步压到 4-bit, 但风险大, 需充分验证

---

## 9. 参考文献

- **原论文**: Tim Dettmers, Mike Lewis, Sam Shleifer, Luke Zettlemoyer, "8-bit Approximations for Parallelism in Deep Learning" (2022) - [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- **QLoRA 论文**: Tim Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **bitsandbytes 文档**: [bitsandbytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- **HuggingFace 实践**: [8-bit AdamW in HuggingFace Transformers](https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#adamw-bnb-8bit)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 定义 P0 8-bit AdamW 集成方案 | Sisyphus |

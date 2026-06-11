# 02 · torch.compile (`reduce-overhead` 模式)

> **状态**: ⚠️ 开关已就绪, 默认关闭 | **优先级**: P0
> **代码位置 (待修改)**: `trainer/train_pretrain.py:106, 150-152`
> **当前实现**: `torch.compile(model)` (默认模式), 需改为 `mode="reduce-overhead"`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

`torch.compile` 是 PyTorch 2.x 的核心性能特性, 通过 **PyTorch Inductor** 将 Python + 动态图的代码编译为优化过的 **FX Graph**, 然后通过 Triton / CUDA C++ 生成高性能 kernel。

**`mode="reduce-overhead"`** 是 Inductor 的一种特定模式:
- **启用 CUDA Graphs**: 减少 CPU kernel launch 开销 (小 batch 上收益最大)
- **自动内存池管理**: 减少 allocator 调用
- **cudagraph 树管理**: 处理动态 control flow (通过 replay)

**与其他模式对比**:

| 模式 | 适用场景 | 提速 | 限制 |
|------|----------|------|------|
| `default` | 通用 | 1.0-1.3x | 无 |
| `reduce-overhead` | 小 batch, 短序列 | 1.3-1.8x | 需固定 shape, 限制 dynamic shape |
| `max-autotune` | 极致性能, 编译时间长 | 1.5-2.5x | 首次编译 5-30 分钟, 占用大内存 |

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- MiniMind 默认 batch=32, seq=340, 64M 模型 → 大量小 kernel launch, CPU 开销显著
- 当前 `--use_compile 0` 默认关闭, 未利用此项免费提速
- MiniMind 模型代码 (Qwen3 风格) 是 Inductor 友好的:
  - 无 dynamic control flow (训练时)
  - 无自定义 autograd function
  - 标准 nn.Module 调用链

**不集成的代价**:

- 64M 训练 step/s 损失 1.3-1.5x
- 1B+ 模型 kernel launch 占比下降, 收益略小 (1.1-1.3x), 仍有价值

**典型收益** (MiniMind 量级):

- 64M + batch=32 + seq=340: 1.3-1.5x step/s
- 1B + batch=8 + seq=2048: 1.1-1.3x step/s

---

## 3. 架构设计

### 3.1 编译触发流程

```text
model = MiniMindForCausalLM(...)
model = model.to(device)

# 编译包装
if args.use_compile == 1:
    model = torch.compile(model, mode="reduce-overhead")
    # ↑ 首次调用 forward 时触发编译, 耗时 20-30s
    # ↑ 之后每次 forward 直接走 compiled graph
```

### 3.2 与 DDP 的交互

```text
# 正确顺序: 先 compile, 后 DDP 包装
model = torch.compile(model, mode="reduce-overhead")
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

- DDP 包装在 compile 之后, 编译会看到 DDP 包装的 `forward`
- 编译的 graph 只在本地 rank 跑, 不影响 DDP 通信

### 3.3 与 checkpoint 的 save/load

- `state_dict()` 接口不受 compile 影响 (返回原 model 的参数)
- `torch.save({k: v for k, v in state_dict.items()}, ...)` 正常工作
- 加载后需要重新 compile (因为新 model 对象的 graph cache 是空的)
- **风险**: 检查点恢复时, 首次 step 会重新触发编译 (耗时 20-30s)

### 3.4 与 grad-checkpoint 的兼容性

- ✅ 完全兼容
- 只需 `use_reentrant=False` (PyTorch 2.1+)

### 3.5 与 SDPA 的兼容性

- ✅ 兼容, SDPA 内部已被 Inductor 识别
- 但某些 sparse attention (TriAttention, MInference) 用了自定义 mask, **Inductor 可能不识别**
- 建议: 在这些路径上加 `@torch.compiler.disable` decorator 显式 fallback

---

## 4. 方案实现

### 4.1 `train_pretrain.py` 改动 (最小化)

```python
# trainer/train_pretrain.py:150-152
# 改动前:
if args.use_compile == 1:
    model = torch.compile(model)
    Logger('torch.compile enabled')

# 改动后:
if args.use_compile == 1:
    model = torch.compile(model, mode="reduce-overhead")
    Logger('torch.compile reduce-overhead enabled')
```

### 4.2 (可选) 新增 mode 参数

```python
# trainer/train_pretrain.py: argparse
parser.add_argument('--compile_mode', default='reduce-overhead', type=str,
                    choices=['default', 'reduce-overhead', 'max-autotune'],
                    help='torch.compile 模式')
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_compile` | `0` | 关闭, 与原行为一致 |
| `--compile_mode` (新增) | `reduce-overhead` | 推荐 |
| `dynamic` | `None` (关闭) | 固定 shape 时不需 dynamic |
| `fullgraph` | `False` | 允许 graph break, 兼容性更好 |

### 4.4 已知不兼容路径 (需 `@torch.compiler.disable`)

```python
# model/model_minimind.py: 某些 sparse attention 路径
@torch.compiler.disable
def custom_sparse_path(self, ...):
    # TriAttention, MInference 等的 mask 生成
    ...
```

---

## 5. 训练过程影响

| 维度 | 影响 |
|------|------|
| 显存 | +5-10% (CUDA Graph 缓存 + workspace) |
| 速度 | **+30-50% step/s** (小 batch / 小模型上) |
| 首次编译耗时 | 20-30s (64M), 1-3 min (1B) |
| 数值 | **无影响** (bit-exact 等价) |
| 优化器 | 无影响 |
| DDP 兼容 | ✅ 兼容 (需先 compile 后 DDP) |
| Resume from ckpt | 重新编译 20-30s |

---

## 6. 消融实验方案

### 6.1 64M Dense 验证

- **配置**: `hidden_size=768, num_layers=8, batch=32, seq=340, 1 epoch on pretrain_t2t_mini`
- **对照**:
  - `--use_compile 0` (基线)
  - `--use_compile 1 --compile_mode default`
  - `--use_compile 1 --compile_mode reduce-overhead` (推荐)
  - `--use_compile 1 --compile_mode max-autotune`
- **指标**: 显存峰值, step/s (前 100 步平均, 排除 warmup + 首次编译), loss 曲线
- **预期**: reduce-overhead 模式 step/s 提升 1.3-1.5x

### 6.2 与 grad-checkpoint 联合验证

- **配置**: 在 6.1 基础上开 `--grad_checkpoint 1`
- **预期**: 两个优化可叠加, 速度损失 vs 加速基本抵消

### 6.3 与 MoE 联合验证

- **配置**: minimind-3-moe (4 experts, top-1), batch=16, seq=340
- **预期**: MoE Python for 循环可能阻碍编译, step/s 提升可能仅 1.1-1.2x
- 备选: 排除 MOEFeedForward (`@torch.compiler.disable`)

---

## 7. 已知问题与限制

### 7.1 自定义 kernel 不被编译

- `MOEFeedForward.forward` 用了 `index_add_` + Python `for` 循环
- Inductor 可能不识别 `index_add_` 的动态 shape 模式
- **症状**: 编译后速度没有提升, 甚至略降
- **解决**: `@torch.compiler.disable` 装饰 MOEFeedForward, 或重写为 Triton (见 [`07-moe-triton-grouped-gemm.md`](07-moe-triton-grouped-gemm.md))

### 7.2 dynamic shape 限制

- `reduce-overhead` 模式会 cache CUDA Graph
- 若 batch / seq 在训练中变化, 需要重新捕获 graph
- MiniMind 训练时 batch / seq 固定, 无问题
- 推理时 (`generate`) 是动态的, **不要**在 `generate` 中用 reduce-overhead 模式

### 7.3 首次编译卡顿

- 首次 `forward` 触发编译, 耗时 20s-3min, 期间 GPU 闲置
- **建议**: 训练脚本启动后先跑 1 个 dummy step 触发编译, 再正式训练
- 或在 Logger 中明确告知用户

### 7.4 Resume 重新编译

- `torch.load` + `model.load_state_dict` 后, 之前编译的 graph cache 失效
- 首次 step 会重新编译 (20-30s)
- **建议**: resume 时用 default 模式 (避免重编译耗时)

### 7.5 内存峰值比理论高

- CUDA Graph cache + 编译时 workspace 可能瞬时占用额外 2-3GB
- 1B+ 模型在 24GB 卡上 + compile 可能 OOM
- 建议 1B+ 模型先用 default 模式, 等显存稳定再考虑 reduce-overhead

---

## 8. 后续改进方向

1. **整图捕获 (fullgraph=True)**: 强制无 graph break, 提速可能再 +10-20%, 但需修复所有不兼容路径
2. **与 Liger-Kernel 协同**: Liger 的 Triton kernel 替代 Inductor 生成, 在 RMSNorm / SwiGLU 上可叠加
3. **AOT Compilation**: 用 `aot_eager` + `aot_autograd` 在训练前预编译, 避免首次卡顿
4. **per-module compile**: 单独编译 Attention / MLP, 加速首次编译

---

## 9. 参考文献

- **PyTorch 官方文档**: [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html)
- **TorchInductor 指南**: [Inductor CPU/GPU Tuning](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- **CUDA Graphs in PyTorch**: [CUDA Graphs 官方介绍](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- **PyTorch 2.4 release notes**: 改进的 reduce-overhead 模式, 更好的 DDP 集成

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 定义 P0 torch.compile reduce-overhead 集成方案 | Sisyphus |

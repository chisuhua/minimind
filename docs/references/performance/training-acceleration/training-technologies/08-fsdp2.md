# 08 · FSDP2 (PyTorch FSDP v2) 多卡分片

> **状态**: 📝 讨论中 | **优先级**: P2
> **代码位置 (待修改)**: `trainer/train_pretrain.py` 启动方式
> **依赖**: `torch >= 2.4`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

**FSDP2 (Fully Sharded Data Parallel v2)** 是 PyTorch 2.4+ 的官方分布式训练方案, 把模型参数、梯度、优化器状态**分片到所有 GPU**, 每张卡只保存 1/N 的状态。

**FSDP2 vs FSDP1**:
- FSDP1 (2022): 用 `FlatParameter` 把参数 flatten 后分片
- **FSDP2 (2024)**: 引入 `DTensor` + `fully_shard` API, 更灵活, 通信优化更好
- FSDP2 与 torch.compile 兼容性更好

**与 DeepSpeed ZeRO-3 的关系**:
- 概念上等价: 都是"分片 + 通信"
- FSDP2 是 PyTorch 原生, DeepSpeed 是 Microsoft 的封装
- 性能基本相当, FSDP2 在 PyTorch 2.4+ 上更稳定
- **MiniMind 已支持 DDP + DeepSpeed** (README 提到), 改为 FSDP2 主要是 API 升级

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- **单卡场景 (1x 4090)**: FSDP2 不可用 (没有"分片"对象)
- **多卡场景 (2x+ 4090)**: FSDP2 是训练 >1B 模型的关键
- 当前 MiniMind 已支持 `torchrun --nproc_per_node N` (DDP), 但 DDP 不分片, 4x 4090 还是只能训 1B
- FSDP2 + 4x 4090: 每卡 6GB 优化器状态, 等效 24GB 优化器状态, 可训 3B+ 模型

**不集成的代价**:

- 1.5B+ Dense 模型, 2x 4090 DDP 也无法跑 (每卡 16GB 优化器状态)
- 1.5B+ 模型只能单机单卡 (24GB) + 各种 offload, 受限
- 多卡用户场景下, MiniMind 的训练规模上限 = 单卡上限

**典型收益**:

- N 卡训练: 等效显存 = N × 24GB
- 2 卡: 48GB 等效 → 3B Dense 可训
- 4 卡: 96GB 等效 → 5-7B Dense 可训
- 通信开销: ~15-25% 相对 DDP

---

## 3. 架构设计

### 3.1 FSDP2 数据流

```text
# 4 卡 FSDP2 训练 1B 模型
# 每卡显存占用:
# - 参数: 4GB / 4 = 1GB (分片)
# - 梯度: 4GB / 4 = 1GB (分片)
# - 优化器: 8GB / 4 = 2GB (分片)
# - 激活: 1-2GB (per rank)
# - 临时 all-gather: ~4GB (峰值, 1B 参数一次性 gather)
# 总计: ~7-8GB / 卡, 4 卡共 28-32GB 逻辑

# 前向:
# 1. all-gather 当前层参数到所有 rank
# 2. 每卡独立 forward
# 3. 释放该层参数 (回到分片状态)

# 反向:
# 1. all-gather 当前层参数
# 2. 每卡独立 backward, 产生局部梯度
# 3. reduce-scatter 梯度 (累加 + 分片)
```

### 3.2 与 MiniMind 当前 DDP 实现的对比

```text
# 现有 (DDP, train_pretrain.py:154)
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])

# 改为 (FSDP2)
if dist.is_initialized():
    from torch.distributed.fsdp import fully_shard
    # 顶层 wrap
    for layer in model.model.layers:
        fully_shard(layer)
    fully_shard(model)
```

### 3.3 与 MiniMind 模型架构的兼容性

- ✅ Dense 模型: 标准 Transformer, FSDP2 直接支持
- ✅ MoE 模型: `MOEFeedForward` 内含多个 expert, 可整层 wrap
- ⚠️ GQA (4 KV heads / 8 Q heads): FSDP2 不感知 head 切分, 整层 wrap OK
- ⚠️ MoE + FSDP2: expert 分片可能不均衡, 需测试

### 3.4 通信开销分析

- **4090 没有 NVLink**, 用 PCIe 4.0 (32 GB/s)
- 1B 模型 all-gather: 4GB, 耗时 ~125ms (理论), 实际 ~150-200ms
- 4B 模型 all-gather: 16GB, 耗时 ~500-700ms
- **结论**: 4090 多卡通信是瓶颈, **强烈推荐用 NVLink 主板 (8x 4090 平台)**
- 单机 2-4 卡 4090 在 PCIe 4.0 下, 通信占比 20-40%

---

## 4. 方案实现

### 4.1 `requirements.txt`

```text
# requirements.txt (PyTorch 2.4+ 已含 FSDP2)
torch>=2.4
```

### 4.2 `trainer_utils.py` 增加 `init_fsdp` 工具

```python
# trainer/trainer_utils.py
def init_fsdp(model, args):
    """Initialize FSDP2 wrapping for MiniMind model."""
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, BackwardPrefetch
    
    # 混合精度策略: bf16 forward, fp32 master
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    
    # 配置
    fsdp_kwargs = dict(
        mp_policy=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )
    
    if args.cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy
        fsdp_kwargs['offload_policy'] = CPUOffloadPolicy()
    
    # 顶层 wrap
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # 逐层 wrap (节省通信量)
        for layer in model.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
    else:
        fully_shard(model, **fsdp_kwargs)
    
    return model
```

### 4.3 `train_pretrain.py` 集成

```python
# trainer/train_pretrain.py
USE_FSDP = int(os.environ.get("USE_FSDP", 0)) == 1

# 现有逻辑:
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])

# 改为:
if USE_FSDP and dist.is_initialized():
    from trainer.trainer_utils import init_fsdp
    model = init_fsdp(model, args)
    Logger(f'FSDP2 initialized, world_size={dist.get_world_size()}')
elif dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])
    Logger(f'DDP initialized, world_size={dist.get_world_size()}')
```

### 4.4 启动方式

```bash
# 单机 4 卡
USE_FSDP=1 torchrun --nproc_per_node 4 trainer/train_pretrain.py

# 单机 8 卡
USE_FSDP=1 torchrun --nproc_per_node 8 trainer/train_pretrain.py

# 多机 (需 NCCL master)
USE_FSDP=1 torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
    --master_addr <addr> --master_port <port> \
    trainer/train_pretrain.py
```

### 4.5 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_FSDP` env | `0` | 关闭, 与原行为一致 (DDP) |
| `BackwardPrefetch` | `BACKWARD_PRE` | 预取下一层参数 |
| `MixedPrecisionPolicy` | bf16/fp32 | forward bf16, reduce fp32 |
| `CPUOffloadPolicy` | (不开启) | 1.5B+ 单机多卡可考虑 |

### 4.6 与 8-bit AdamW 协同

```text
# 重要: FSDP2 + 8-bit AdamW 必须注意:
# - 优化器状态在每张卡上是分片的 (1/N)
# - 8-bit AdamW 状态: 1B 模型 ~2GB / 卡 (vs 8GB / 卡 FP32)
# - 推荐: FSDP2 + 8-bit AdamW = 4B 模型, 4 卡可训
```

### 4.7 与 torch.compile 协同

- FSDP2 与 torch.compile 兼容性 **比 FSDP1 好很多** (PyTorch 2.4+)
- 推荐: `torch.compile + FSDP2` 组合使用
- 风险: 某些 dynamic shape 路径可能 break

---

## 5. 训练过程影响

| 维度 | 影响 |
|------|------|
| 显存 (单卡, 优化器+梯度+权重) | **-75%** (4 卡分片) |
| 显存 (单卡, 临时 all-gather) | +1×N (峰值, 当前层参数) |
| 速度 | **-15-25%** (相对 DDP) |
| 数值 | 无影响 (FP32 reduce) |
| 优化器 | 兼容 bnb 8-bit AdamW (进一步减半) |
| DDP 兼容 | N/A (替换 DDP) |
| Resume 兼容性 | ✅ (FSDP2 检查点格式标准) |

---

## 6. 消融实验方案

### 6.1 2 卡 1B 模型验证

- **配置**: 2x 4090, `hidden_size=1536, num_layers=24, batch=4, seq=2048, 100 步`
- **对照**:
  - DDP: 预期 OOM (单卡 16GB 优化器)
  - FSDP2: 预期可跑
- **指标**: 显存峰值, step/s, loss 曲线
- **预期**:
  - 显存: 4GB 优化器 + 2GB 权重 + 2GB 梯度 + 1GB 激活 = ~10GB / 卡
  - 速度: DDP 不行, FSDP2 跑通
  - loss 曲线**与单卡一致**

### 6.2 4 卡 1.5B 模型验证

- **配置**: 4x 4090, `hidden_size=1792, num_layers=24, batch=4, seq=2048, 100 步`
- **对照**:
  - FSDP2 only: 预期 12GB 优化器 / 4 = 3GB / 卡, 跑通
  - FSDP2 + 8-bit AdamW: 1.5GB / 卡, 跑通且更省
- **指标**: 显存, step/s
- **预期**:
  - FSDP2 + 8-bit AdamW 是最优解
  - 速度: PCIe 4.0 通信占比 20-30%

### 6.3 4 卡 3B 模型极限测试

- **配置**: 4x 4090, `hidden_size=2560, num_layers=24, batch=2, seq=2048, 100 步`
- **对照**:
  - 单卡: 24GB 静态占用, 跑不通
  - 4 卡 DDP: 同单卡 (不分片), 跑不通
  - 4 卡 FSDP2 + 8-bit: 6GB / 卡优化器 + 4GB 权重 + 2GB 激活 = ~12GB, 跑通
- **指标**: max batch size, step/s
- **预期**: 3B Dense 在 4x 4090 上可跑

### 6.4 FSDP2 + MoE 验证

- **配置**: 4x 4090, minimind-4-moe 1.5B (8 experts, top-2)
- **预期**:
  - expert 参数分片到 4 卡, 每卡 1/4
  - 激活参数量不变 (top-2)
  - 路由逻辑需在所有 rank 同步 (all-reduce scores, 简化版)

---

## 7. 已知问题与限制

### 7.1 单卡不可用

- FSDP2 需要 N >= 2 卡
- 单卡用户应使用 8-bit AdamW + grad-ckpt (P0) 而非 FSDP2
- 当前实现: `if USE_FSDP and dist.is_initialized():` (单卡自动 skip)

### 7.2 通信瓶颈 (无 NVLink)

- 4090 工作站 (4-8 卡) 主板通常无 NVLink
- PCIe 4.0 32 GB/s 是通信瓶颈
- 解决: 用 NVLink 主板 (8x 4090 NVLink 平台, 较罕见)
- 替代: 接受 PCIe 4.0 性能, 训练时间 +20-30%

### 7.3 MoE 路由同步

- 当前 `MOEFeedForward` 的 `gate` Linear 输出只在每张卡本地计算
- 路由分布可能不均 (各卡不同)
- FSDP2 下需要 all-reduce 路由分数? **不必要** (路由是 per-token 的, 不跨卡)
- 风险: 专家负载均衡在多卡下可能更差, 需监控 aux_loss

### 7.4 检查点格式

- FSDP2 检查点需要用 `torch.distributed.checkpoint` 格式
- 不能直接用 `torch.save(model.state_dict())`
- MiniMind 当前检查点代码 (trainer_utils.py:69-77) 需要适配
- 风险: 跨卡 resume 需要全 rank 同步

### 7.5 与现有 DDP 启动方式共存

- MiniMind 用户已有 `torchrun --nproc_per_node N` 启动 DDP
- FSDP2 用同样的 `torchrun`, 但需 `USE_FSDP=1` env
- 需要在 README 中明确文档, 提供对比和切换指南

### 7.6 torch.compile + FSDP2 边角问题

- 某些 dynamic shape 路径在 FSDP2 下编译失败
- 解决: `@torch.compiler.disable` 装饰特定函数
- 风险: 需逐 case 测试

---

## 8. 后续改进前进方向

1. **Hybrid Sharding (HSDP)**: FSDP2 + 复制策略, 适合 8+ 卡
2. **CPU Offload + FSDP2**: 1.5B+ 模型的终极方案
3. **FSDP2 + Sequence Parallel**: 长序列训练时把 seq 维度也分片
4. **FSDP2 + MoE Expert Parallel**: MoE expert 分到不同 rank, 通信量最小化

---

## 9. 参考文献

- **PyTorch FSDP2 官方文档**: [FSDP v2](https://pytorch.org/docs/stable/fsdp.html)
- **PyTorch 2.4 release notes**: FSDP2 引入, 替代 FSDP1
- **FSDP2 vs FSDP1 对比**: [PyTorch Blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- **Accelerate FSDP2 集成**: [HuggingFace Accelerate FSDP2](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_fsdp2)
- **MiniMind 当前 DDP 实现**: `trainer/trainer_utils.py:44-51` (`init_distributed_mode`)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 定义 P2 FSDP2 多卡分片集成方案 | Sisyphus |

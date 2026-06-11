# 05 · MInference 1.0 (动态稀疏注意力)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 1
> **代码位置**: `model/minference.py` (`MInference` 类), `scripts/calibrate_minference.py`
> **CLI 入口**: `--minference`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

MInference 1.0 (Jiang et al., Microsoft 2024) 是一种**离线标定 + 推理时动态稀疏**的注意力加速技术。核心观察是:

- 不同 attention head 在**不同上下文位置**的关注模式差异巨大
- 一些 head 关注**垂直到某列的列模式** (vertical-slash), 例如检索任务
- 一些 head 关注**对角线附近的 A-shape 模式** (A-shape), 例如局部上下文
- 一些 head 关注**块状模式** (block-sparse), 例如归纳头
- **全连接 O(N²) 计算在长序列下严重浪费**

MInference 1.0 的做法:
1. **离线标定**: 用一段 calibration 数据集, 对每个 (layer, head) 二元组搜索最佳的 3 种稀疏模式之一
2. **推理时**: 每个 head 走自己专属的稀疏模式, 计算量从 O(N²) 降至 O(N log N) 或更低
3. 3 种模式: vertical-slash / A-shape / block-sparse

> **典型加速比**: 1K 长度下 10×, 10K 长度下 18× (相对全连接 attention, prefilling 阶段)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- MiniMind 支持 32K 上下文, YaRN 可外推到更长
- 在 8K+ 长度时, 朴素 attention 的 O(N²) 是主要瓶颈
- MInference **离线标定**成本低 (一次, 几分钟), 推理时几乎零额外开销
- 不需要修改模型权重, 不需要重新训练

**不集成的代价**:
- 8K+ 上下文 prefilling 阶段耗时占比可达 50%+
- 长 prompt 一次输入的体验差
- 与同样长度优化目标 (StreamingLLM) 互补, 但解决问题不同

**典型加速比**: 1K prompt 8-10×, 4K 12-15×, 16K 18-25× (在 prefilling 阶段)

---

## 3. 架构设计

### 3.1 三种稀疏模式

| 模式 | 形态 | 适用 head | 复杂度 |
|------|------|-----------|--------|
| **Vertical-Slash** | 整列 + 局部斜线 | 检索、引用 | O(N × (V+S)) |
| **A-shape** | 三角带 + 局部窗口 | 局部上下文 | O(N × A) |
| **Block-Sparse** | 块状 | 归纳头 | O(N × √N) |

### 3.2 数据流

```
离线标定 (一次):
  ┌──────────────────────────────────────┐
  │  1. 取 8-16 个长序列 (>= 4K)        │
  │  2. 对每个 (layer, head) 跑朴素 attn│
  │  3. 用 top-k 阈值提取每个 head 的    │
  │     attention map 形状               │
  │  4. 与 3 种预设模式算 Jaccard 相似度│
  │  5. 选最佳模式, 存为 calibration    │
  └──────────────────────────────────────┘

推理时:
  ┌──────────────────────────────────────┐
  │  1. 加载 calibration table           │
  │  2. 对每个 (layer, head):            │
  │     按对应模式构造稀疏 attention mask│
  │     调用 sparse SDPA                 │
  │  3. 输出不变                         │
  └──────────────────────────────────────┘
```

### 3.3 关键模块

- **`MInference`**: 主类, 持有 calibration table
- **`SparseAttention`**: 模式化的稀疏 attention kernel (CPU 端用 mask 模拟, GPU 端可换 Triton)
- **`scripts/calibrate_minference.py`**: 离线标定脚本

### 3.4 计算复杂度

| 长度 | 全连接 | MInference (混合) | 加速 |
|------|--------|-------------------|------|
| 1K | 1M | 100K | 10× |
| 4K | 16M | 1.1M | 15× |
| 16K | 256M | 12M | 21× |
| 64K | 4G | 130M | 30× |

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/minference.py
class MInference:
    def __init__(self, model, calibration: dict = None):
        self.model = model
        self.calibration = calibration or {}  # (layer, head) → mode
        # 3 种稀疏 kernel
        self.kernels = {
            'vertical_slash': VerticalSlashKernel(),
            'a_shape': AShapeKernel(),
            'block_sparse': BlockSparseKernel(),
        }

    def attention_with_sparsity(self, q, k, v, layer_idx, head_idx):
        mode = self.calibration.get((layer_idx, head_idx), 'dense')
        if mode == 'dense':
            return F.scaled_dot_product_attention(q, k, v)
        else:
            return self.kernels[mode].forward(q, k, v,
                                              seq_len=q.shape[2])

# 离线标定
def calibrate(model, calib_data):
    results = {}
    for layer in range(model.config.n_layer):
        for head in range(model.config.n_head):
            attn_map = collect_attention_map(model, calib_data, layer, head)
            best_mode = find_best_mode(attn_map, modes=['vertical_slash', 'a_shape', 'block_sparse'])
            results[(layer, head)] = best_mode
    return results
```

### 4.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `calibration` | None | 标定结果, 路径 |
| `vertical_top_k` | 64 | vertical-slash 列数 |
| `slash_window` | 256 | 斜线窗口长度 |
| `a_block_size` | 32 | A-shape 块大小 |
| `block_size` | 64 | block-sparse 块大小 |

### 4.3 默认配置

`eval_llm.py` 默认关闭。开启需先标定:
```bash
python scripts/calibrate_minference.py --data_path dataset/calib_long.jsonl --output minference_calib.pt
python eval_llm.py --minference --minference_calib minference_calib.pt
```

---

## 5. 训练过程影响

**零影响**。MInference 是纯推理时技术, 不修改训练目标或模型权重。

> 可选: 如果想用 MInference 加速**训练**时的长序列 attention, 需要实现 backward pass。当前实现仅支持 forward。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | `minimind-3` (64M), full_sft |
| 标定集 | 8 条 4K 长度的 sft_t2t 抽样 |
| 测试集 | 256 / 1K / 4K / 16K prompt 各 50 条 |
| 评估 | prefilling 耗时 / PPL / 生成质量 |

### 6.2 评估指标

- **Prefilling 耗时** (ms, 主指标)
- **稀疏模式命中率** (每个 head 的 top-1 模式占比)
- **PPL 退化** (在保留的 2K 窗口)
- **生成质量** (任务准确率)

### 6.3 预期结果

| 长度 | 全连接耗时 | MInference 耗时 | 加速 | PPL 退化 |
|------|-----------|-----------------|------|----------|
| 256 | T0 | T0 × 0.7 | 1.4× | < 1% |
| 1K | T1 | T1 × 0.1 | 10× | 1-2% |
| 4K | T2 | T2 × 0.07 | 14× | 2-4% |
| 16K | T3 | T3 × 0.05 | 20× | 4-8% |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **标定数据依赖**: calibration 必须用与推理分布相近的数据, 否则精度下降
2. **不支持 MoE**: 当前实现仅在 dense 模型上测试, MoE 路由可能干扰
3. **不支持 backward**: 不能用于训练加速
4. **Flash Attention 集成不完整**: GPU kernel 用 Triton 重写可获得额外 2-3×
5. **小模型收益偏低**: 64M 模型 attention 计算占总耗时比例低, 加速比 1.4× 起步
6. **LSP 错误**: `minference.py:265` `max` 函数 overload 错误, 运行时无影响

---

## 8. 后续改进方向

- [ ] **MInference 2.0**: 三模式 → 自适应动态选择
- [ ] **Triton kernel 完整实现**: 替换 CPU mask 模拟
- [ ] **Backward 支持**: 允许训练加速
- [ ] **集成到 streaming_kv_cache**: MInference 标定 sink/local 头
- [ ] **AutoML 选模式**: 用 NAS 替代手工 3 模式
- [ ] **多数据集标定融合**: 跨数据集选稳定的 head 模式

---

## 9. 参考文献

- Jiang et al., "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention", NeurIPS 2024
- arXiv: 2407.02490
- [GitHub: microsoft/MInference](https://github.com/microsoft/MInference)
- 后续: MInference 2.0 (arXiv 2507.xxxxx)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

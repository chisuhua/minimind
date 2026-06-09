# 07 · MoE Triton Grouped-GEMM

> **状态**: 📝 讨论中 | **优先级**: P2
> **代码位置 (待修改)**: `model/model_minimind.py:265-293` (`MOEFeedForward`)
> **依赖**: `triton >= 2.0` (PyTorch 2.x 自带)
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

**Grouped-GEMM** 是一种**MoE 专用的高效矩阵乘法**: 多个 expert 的矩阵乘法 batch 成一个 kernel 调用, 避免 Python `for` 循环 + 多次 `index_add_` 的开销。

**当前 MiniMind 的 MOEFeedForward 性能问题**:

```python
# model_minimind.py:280-285
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
    elif self.training:
        y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

**问题清单**:
1. **Python 循环**: `for i, expert in enumerate(...)`, 每个 expert 一次 kernel launch
2. **`mask.any(dim=-1).nonzero()`**: 每次都重新计算 token 索引, 浪费
3. **`index_add_`**: scatter-add 操作, 不易被 Inductor 优化
4. **形状不规则**: 每个 expert 处理的 token 数不同, GPU 算力浪费
5. **专家冷启动 trick**: `y[0, 0] += 0 * sum(...)` 是为了让 autograd 追踪到所有 expert 的参数, 但增加了无意义计算

**Grouped-GEMM 解决**:

```text
# 把 4 个 expert 的 matmul 合并为一次 group_gemm 调用
# 类似: block-diagonal matmul
#   [x_for_expert_0]   [W_0  0   0   0  ]
#   [x_for_expert_1] = [0    W_1 0   0  ] @ x_tokens_sorted_by_expert
#   [x_for_expert_2]   [0    0   W_2 0  ]
#   [x_for_expert_3]   [0    0   0   W_3]
# 
# 用 Triton 实现: 一次 kernel, 多个 W 并行 matmul
```

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- minimind-3-moe (198M 总参, 64M 激活, 4 experts, top-1) 已实现
- README 提到: "4 experts / top-1 这个甜点配置大约只比 dense 模型慢 50% 左右"
- 慢的 50% **主要来自 MOEFeedForward 的 Python for 循环** (不是路由逻辑)
- 用 grouped-GEMM 后, 预期 MoE 训练速度提升 1.5-3x (与 dense 速度接近)

**不集成的代价**:

- minimind-4-moe (1.5B-3B) 训练时, MoE 路径仍是主要瓶颈
- 1.5B MoE 训练速度与 1.5B Dense 差距 >50% (即使激活参数量相同)
- 失去 MoE 的工程优势

**典型收益**:

- MoE 训练 step/s 提升 1.5-3x
- 显存不变 (激活参数量没变)
- 与 Liger-Kernel 兼容 (Liger 当前没有 MoE 路径)

---

## 3. 架构设计

### 3.1 Grouped-GEMM 数据流

```text
输入: x (batch × seq × hidden) → x_flat (BS × hidden)
路由: gate(x) → scores → topk_weight, topk_idx (每个 token 选 1 个 expert)

# 旧实现 (4 次 expert forward)
for i in range(num_experts):
    mask = (topk_idx == i)
    token_idx = mask.nonzero()
    expert_out = self.experts[i](x_flat[token_idx])
    y.index_add_(0, token_idx, expert_out * weight)

# 新实现 (1 次 grouped gemm)
# Step 1: 排序 token, 把同一 expert 的 token 放一起
sorted_token_idx, expert_offsets = sort_tokens_by_expert(topk_idx)
x_sorted = x_flat[sorted_token_idx]  # shape: (BS, hidden)

# Step 2: 一次 grouped GEMM
# 每个 expert 有自己的 W, 我们用 group_gemm_kernel 同时计算
y_sorted = grouped_gemm(x_sorted, expert_W_stack, expert_offsets)
#   ↑ x_sorted: (BS, hidden)
#   ↑ expert_W_stack: (num_experts, hidden, intermediate_size)
#   ↑ expert_offsets: [0, n0, n0+n1, ...] (每个 expert 的 token 数)
#   ↑ y_sorted: (BS, intermediate_size)

# Step 3: scatter 回原顺序
y = scatter_back(y_sorted, sorted_token_idx, topk_weight)
```

### 3.2 Triton Grouped-GEMM Kernel 关键设计

```python
@triton.jit
def grouped_gemm_kernel(
    A_ptr, B_ptr, C_ptr,  # 输入输出指针
    expert_offsets,        # 每个 expert 的 token 范围
    M, N, K,               # matmul 维度
    stride_am, stride_ak,  # A 的 stride
    stride_be, stride_bn, stride_bk,  # B (per expert) 的 stride
    stride_cm, stride_cn,  # C 的 stride
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 类似 matmul kernel, 但 BLOCK_M 维度上每个 program 块处理一个 expert
    pid = tl.program_id(0)
    
    # 找到当前 program 属于哪个 expert
    expert_id = 0
    for e in range(num_experts):
        if pid >= expert_offsets[e] // BLOCK_M:
            expert_id = e
        else:
            break
    
    # 计算当前 expert 的局部 M 范围
    m_start = expert_offsets[expert_id]
    m_end = expert_offsets[expert_id + 1]
    
    # 标准 matmul
    ...
```

### 3.3 与 DeepSeek / vLLM / sglang 的实现对比

| 实现 | 来源 | 特点 |
|------|------|------|
| DeepSeek-MoE `grouped_gemm` | [deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE) | 简单, 可读性好, 适合学习 |
| vLLM `fused_moe` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | 生产级, 优化更激进 |
| sglang `moe_align_block_size` | [sgl-project/sglang](https://github.com/sgl-project/sglang) | 适合 block-sparse MoE |
| megablocks | [databricks/megablocks](https://github.com/databricks/megablocks) | block-sparse MoE 的 SOTA |

**推荐**: MiniMind 起步用 DeepSeek-MoE 的简洁实现, 性能足够, 工程量小。

---

## 4. 方案实现

### 4.1 新建 `model/grouped_gemm.py`

```python
"""
Triton 实现 Grouped-GEMM, 用于 MOEFeedForward
参考: deepseek-ai/DeepSeek-MoE/modeling_deepseek.py grouped_gemm
"""
import torch
import triton
import triton.language as tl


@triton.jit
def grouped_gemm_kernel(
    A, B, C,
    expert_offsets,
    M, N, K,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # ... (Triton kernel 主体, 略)
    pass


def grouped_gemm(
    x: torch.Tensor,  # (total_tokens, K)
    expert_weights: torch.Tensor,  # (num_experts, K, N)
    expert_offsets: torch.Tensor,  # (num_experts + 1,)
) -> torch.Tensor:
    """
    Args:
        x: 已经按 expert 排序好的 token, shape (total_tokens, K)
        expert_weights: 所有 expert 的 W 堆叠, shape (num_experts, K, N)
        expert_offsets: 每个 expert 的 token 范围, e.g. [0, n0, n0+n1, ...]
    Returns:
        out: shape (total_tokens, N)
    """
    M_total = x.shape[0]
    N = expert_weights.shape[2]
    out = torch.empty((M_total, N), device=x.device, dtype=x.dtype)
    
    grid = (triton.cdiv(M_total, 32), triton.cdiv(N, 32))
    grouped_gemm_kernel[grid](
        x, expert_weights, out,
        expert_offsets,
        M_total, N, expert_weights.shape[1],
        x.stride(0), x.stride(1),
        expert_weights.stride(0), expert_weights.stride(2), expert_weights.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )
    return out
```

### 4.2 `MOEFeedForward` 重写

```python
# model/model_minimind.py:265-293
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 关键: 把 expert 堆叠为 3D tensor, 便于 grouped_gemm
        # 每个 expert: gate_proj (hidden, inter), up_proj (hidden, inter), down_proj (inter, hidden)
        # 简单起见, 先用 nn.ModuleList, forward 时 stack
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        num_tokens = x_flat.shape[0]
        
        # 路由
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: 
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        
        # Step 1: 排序 token
        # sorted_idx[i] = 原 token 索引 (按 expert 排好)
        sorted_idx = topk_idx.argsort(dim=-1, stable=True).flatten()  # 简化: top-1
        expert_offsets = torch.searchsorted(
            topk_idx.flatten().sort()[0],
            torch.arange(self.config.num_experts + 1, device=x.device)
        )
        x_sorted = x_flat[sorted_idx]
        
        # Step 2: Grouped GEMM
        # 把 expert 的 W stack 起来 (4D tensor)
        gate_stack = torch.stack([e.gate_proj.weight for e in self.experts])  # (E, hidden, inter)
        up_stack = torch.stack([e.up_proj.weight for e in self.experts])
        # 用 grouped_gemm 一次算完
        gate_out = grouped_gemm(x_sorted, gate_stack, expert_offsets)  # (num_tokens, inter)
        up_out = grouped_gemm(x_sorted, up_stack, expert_offsets)
        activated = self.act_fn(gate_out) * up_out
        # ... 继续 down_proj 的 grouped gemm ...
        
        # Step 3: Scatter 回原顺序
        y = torch.empty_like(x_flat)
        y.index_copy_(0, sorted_idx, activated * topk_weight[sorted_idx])
        
        return y.view(batch_size, seq_len, hidden_dim)
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BLOCK_M` | 32 | matmul block, 需 tune |
| `BLOCK_N` | 32 | matmul block |
| `BLOCK_K` | 32 | matmul block |
| `num_warps` | 4 | Triton warps, 需 tune |
| `num_stages` | 3 | Pipeline stages, 需 tune |

### 4.4 性能调优 (autotune)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def grouped_gemm_kernel(...):
    ...
```

---

## 5. 训练过程影响

| 维度 | 影响 |
|------|------|
| 显存 (激活) | 不变 |
| 显存 (其他) | 不变 |
| 速度 (MoE forward) | **+50-200%** (vs Python for 循环) |
| 速度 (MoE backward) | **+50-150%** (Triton autograd 兼容) |
| 数值 | 与原 Python 实现 bit-different (FMA 顺序), 但在 1e-4 量级 |
| 与 Liger-Kernel 兼容 | ✅ (替换 FeedForward 不影响其他组件) |
| 与 grad-ckpt 兼容 | ✅ |
| 与 torch.compile 兼容 | ⚠️ Triton kernel 不会被 Inductor 编译, 但能与 compiled code 共存 |

---

## 6. 消融实验方案

### 6.1 minimind-3-moe 速度验证

- **配置**: minimind-3-moe 默认 (4 experts, top-1, hidden=768, 8 layers), batch=16, seq=340
- **对照**:
  - 原 `MOEFeedForward` (Python for + index_add_)
  - 新 `MOEFeedForward` (Triton grouped_gemm)
- **指标**: step/s, 显存, loss 曲线
- **预期**:
  - step/s 提升 1.5-2x (从 ~50% dense 速度提升到 ~75-85% dense 速度)
  - 显存不变
  - loss 曲线**完全一致**

### 6.2 minimind-4-moe 扩展性测试

- **配置**: `hidden_size=1024, num_layers=16, 8 experts, top-2, batch=8, seq=2048`
- **对照**:
  - 原 MOEFeedForward: 预期 OOM 或 batch 受限
  - Triton MOEFeedForward: 预期 batch=8 可跑
- **指标**: max batch size, step/s
- **预期**:
  - 显存基本不变 (激活是 O(N) per token, 取决于 inter_size)
  - 速度提升 1.5-3x

### 6.3 负载均衡损失 (aux_loss) 验证

- 配置: 同 6.1
- 对照: 原 aux_loss 计算 vs 重写后
- 指标: aux_loss 数值
- 预期: **完全一致** (aux_loss 计算不依赖 forward 路径, 只看路由分布)

---

## 7. 已知问题与限制

### 7.1 路由逻辑需保留

- Grouped-GEMM 只解决 expert forward, **路由逻辑不变**
- `self.gate` (router) 仍是 Linear
- 路由策略 (top-k, norm_topk_prob) 仍由 `MOEFeedForward.forward` 头部控制

### 7.2 Token 排序的开销

- 每次 forward 都要 `argsort` + `searchsorted`
- 当 num_tokens 很大 (32K+) 时, 排序本身耗时显著
- **优化**: 维护一个 cache, 复用排序结果 (但路由每次不同, 难以复用)

### 7.3 负载不均衡场景

- 如果路由严重不均 (某 expert 处理 80% token, 其他 20%), 排序后 expert_offsets 极度不均
- Triton grouped_gemm **仍然能跑**, 但 GPU 算力浪费
- 建议: 监控 `load` (F.one_hot 后 mean), 若 `max(load) > 0.5` 则考虑调整路由策略

### 7.4 兼容 num_experts_per_tok > 1

- top-2 场景下, 每个 token 选 2 个 expert, 排序逻辑需调整
- 当前实现假设 top-1, **需要扩展**
- 推荐: 用 `expert_choice` 路由 (DeepSeek-V3 风格) 替代 top-k

### 7.5 Triton 编译失败处理

- Triton 编译可能因 SM 版本 (sm_89 for 4090) 不支持而失败
- 建议: 在 `grouped_gemm.py` 顶层加 try/except, 失败时 fallback 到原 Python 实现
- 风险等级: 中

---

## 8. 后续改进方向

1. **Block-Sparse MoE**: 用 megablocks 风格, 进一步节省 expert 内存
2. **Expert Parallel**: 多卡时 expert 分到不同 rank, 通信量与激活量解耦
3. **Expert 量化**: 单 expert 4-bit 量化, 显存再 -75%
4. **Fine-grained Expert**: DeepSeek-V2 风格, 把 1 expert 拆成 m 个小 expert, 路由更灵活

---

## 9. 参考文献

- **DeepSeek-MoE**: [GitHub](https://github.com/deepseek-ai/DeepSeek-MoE) - 简洁 grouped_gemm 实现
- **vLLM `fused_moe`**: [GitHub](https://github.com/vllm-project/vllm) - 生产级优化
- **Megablocks**: [GitHub](https://github.com/databricks/megablocks) - Block-sparse MoE
- **Triton Grouped GEMM 教程**: [Triton 官方示例](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
- **MiniMind README 提到的 MoE 慢的原因**: "原生训练时带来的 kernel 启停和调度开销会急剧变重, 这本身是很自然的事情"

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初稿: 定义 P2 MoE Triton Grouped-GEMM 重写方案 | Sisyphus |

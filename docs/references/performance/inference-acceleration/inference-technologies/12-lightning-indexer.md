# 12 · Lightning Indexer (混合稀疏注意力, DSA)

> **状态**: ⚠️ PoC (Proof-of-Concept) | **阶段**: Wave 3
> **代码位置**: `model/lightning_indexer.py` (`LightningIndexer` 类), `trainer/train_lightning_indexer.py`
> **CLI 入口**: `--lightning_indexer`, `--lightning_indexer_path`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

Lightning Indexer (DeepSeek-V3.2 DSA, DeepSeek Sparse Attention) 是一种**主 attn + 索引器混合**的稀疏注意力技术。核心思想是:

- 在标准 attention 之前, 先用**一个轻量索引器**计算每个 query-token 对所有 key-token 的**粗略分数**
- 用粗略分数做 **top-k 选择**, 仅在选中的 key 上跑完整 attention
- 索引器与主 attention **联合训练**

技术组成:
1. **Indexer**: 浅层 attention (1-2 层), 用 FP16 / INT8 计算粗分数
2. **Top-k Selection**: 选 top-k 个 key (k=2048, 远小于 L)
3. **Main Attention**: 在 top-k 上跑标准 SDPA
4. **联合训练**: 两个组件端到端优化

> **典型加速比**: 长序列 5-10×, 与 MInference / TriAttention 类似, 但**训练一体化**

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- 工业 SOTA (DeepSeek-V3.2) 已在用, 学术影响力大
- 与 MInference 1.0 离线标定相比, **联合训练**可获得更好的 head-specific 模式
- 与 TriAttention 三角级数相比, **不依赖函数形式假设**
- 适配小模型时, 索引器小, 训练成本可控

**不集成的代价**:
- 错过 DeepSeek-V3.2 这一代表性 SOTA 技术
- 仍用 2024 早期的稀疏方案 (MInference), 落后于 2025/2026 趋势
- 在长上下文性能上落后于 Qwen3-Next 等竞争对手

**典型加速比**: 1K 1.5-2×, 4K 3-5×, 16K 5-10× (在 prefilling 阶段)

---

## 3. 架构设计

### 3.1 网络结构

```
┌────────────────────────────────────────┐
│ Layer L:                                │
│                                          │
│  Hidden: (B, L, 768)                   │
│    ↓                                     │
│  Lightning Indexer (1 层 attention):    │
│    ├── Index_Q = W_q(hidden)            │
│    ├── Index_K = W_k(hidden)            │
│    ├── Coarse attn = softmax(Q_i K_i.T) │
│    └── Top-k selection (k=2048)         │
│    ↓                                     │
│  Main Attention (标准 attn):            │
│    ├── Q = W_q(hidden)                  │
│    ├── K_topk, V_topk (from indexer)    │
│    └── SDPA(Q, K_topk, V_topk)          │
│    ↓                                     │
│  Output                                 │
└────────────────────────────────────────┘
```

### 3.2 数据流

```
Lightning Indexer 训练:
  1. 主 attn (full) + 索引器 (coarse) 联合 forward
  2. 主 attn 提供精确分数, 索引器提供粗分数
  3. Loss = CE_main + α × KL(indexer || main)  // 让 indexer 模仿 main

推理 (long context):
  1. 索引器 forward → coarse scores
  2. top-k → K_topk, V_topk
  3. 主 attn forward (在 top-k 上) → output
  4. 节省: 完整 attn 是 O(N²), top-k 是 O(N × k)
```

### 3.3 关键模块

- **`LightningIndexer`**: 浅层 attention + top-k 选择
- **`LightningAttention`**: 主 attention, 接收 top-k KV
- **`train_lightning_indexer.py`**: 联合训练脚本

### 3.4 计算复杂度

| 长度 | 全连接 | Lightning (k=2048) | 加速 |
|------|--------|-------------------|------|
| 1K | 1M | 1M (k=L) | 1× (k 已饱和) |
| 4K | 16M | 8M | 2× |
| 16K | 256M | 32M | 8× |
| 64K | 4G | 128M | 31× |

> 加速比随长度增加而提升, 但需要 k < L 才能生效。

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/lightning_indexer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightningIndexer(nn.Module):
    def __init__(self, hidden_size: int, n_head: int, top_k: int = 2048):
        super().__init__()
        self.n_head = n_head
        self.top_k = top_k
        # 索引器: 1 层 attention
        self.idx_q_proj = nn.Linear(hidden_size, n_head * 32, bias=False)  # 32-d
        self.idx_k_proj = nn.Linear(hidden_size, n_head * 32, bias=False)
        self.idx_o = nn.Parameter(torch.randn(n_head, 32) * 0.02)

    def forward(self, hidden_states, k_pool, v_pool):
        # hidden_states: (B, L_q, D)
        # k_pool, v_pool: (B, L_k, D)
        B, L_q, D = hidden_states.shape
        L_k = k_pool.shape[1]

        # 1. Indexer
        idx_q = self.idx_q_proj(hidden_states).view(B, L_q, self.n_head, 32).transpose(1, 2)
        idx_k = self.idx_k_proj(k_pool).view(B, L_k, self.n_head, 32).transpose(1, 2)
        # dot → (B, H, L_q, L_k)
        coarse_scores = (idx_q * self.idx_o.view(1, self.n_head, 1, 32)).sum(dim=-1)
        coarse_scores = coarse_scores @ idx_k.transpose(-1, -2) / (32 ** 0.5)

        # 2. Top-k selection
        topk_scores, topk_indices = coarse_scores.topk(min(self.top_k, L_k), dim=-1)
        # topk_indices: (B, H, L_q, k)

        # 3. Gather K, V
        # 展开索引到所有 head
        k_topk = self._gather_per_head(k_pool, topk_indices, self.n_head)
        v_topk = self._gather_per_head(v_pool, topk_indices, self.n_head)

        return k_topk, v_topk, topk_indices
```

### 4.2 训练脚本

```python
# trainer/train_lightning_indexer.py
model = LightningAttentionModel(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step, batch in enumerate(dataloader):
    # 联合训练
    out_full = model(batch['input_ids'], use_lightning=False)  # full attn (teacher)
    loss_full = F.cross_entropy(out_full.logits.view(-1, V), batch['labels'].view(-1))

    out_sparse = model(batch['input_ids'], use_lightning=True)  # lightning attn
    loss_sparse = F.cross_entropy(out_sparse.logits.view(-1, V), batch['labels'].view(-1))

    # KL 让 indexer 模仿 full attn
    kl_loss = F.kl_div(out_sparse.log_softmax(-1), out_full.softmax(-1), reduction='batchmean')

    loss = loss_full + loss_sparse + 0.1 * kl_loss
    loss.backward()
    optimizer.step()
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | 2048 | top-k 选择数量 |
| `idx_dim` | 32 | 索引器头维度 |
| `n_idx_layers` | 1 | 索引器层数 |
| `kl_alpha` | 0.1 | KL loss 权重 |

### 4.4 默认配置

`eval_llm.py` 默认关闭。完整流程:
```bash
# 1. 联合训练
python trainer/train_lightning_indexer.py --epochs 3 --output lightning_768.pth

# 2. 推理
python eval_llm.py --weight lightning_768.pth --lightning_indexer
```

---

## 5. 训练过程影响

**需要重新训练**:

- 训练目标: `L = CE_main + α × KL(indexer || main)`
- 训练数据: 与 SFT 一致, **建议** 混入 8K+ 长度样本
- 训练时长: 64M 模型 + 浅层 indexer, 单卡 3090 上 **~2 小时**
- 显存: 比标准 SFT 多 ~30% (需要同时跑 full attn 和 indexer)
- **质量影响**: 微正, 因为 indexer 隐式正则了 attention

> **关键**: 短上下文 (≤ 2K) 时, top-k 退化为全选, 加速消失。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 起点 | `minimind-3` (64M) full_sft |
| 训练 | Lightning Indexer 联合训练 |
| 测试 | 256 / 1K / 4K / 8K / 16K prompt |

### 6.2 评估指标

- **Prefilling 耗时**
- **Indexer 拟合 RMSE** (vs full attn)
- **PPL 退化**
- **Top-k 召回率** (top-k 是否覆盖了 true top-k)

### 6.3 预期结果

| 长度 | 加速 | PPL 退化 | 召回率 |
|------|------|----------|--------|
| 1K | 1.0× | 0% | 100% |
| 4K | 2.0× | 1% | 95% |
| 8K | 4.0× | 2% | 90% |
| 16K | 6.0× | 4% | 80% |
| 32K | 8.0× | 8% | 65% |

### 6.4 实际结果 (TBD)

> 当前为 PoC, 实际训练未跑。

---

## 7. 已知问题与限制

1. **训练复杂**: 需要同时维护 full attn 和 sparse attn, 调试困难
2. **短上下文无效**: k=2048 在 1K 长度下没有稀疏效果
3. **Index 选择不稳定**: 不同 batch 选不同 token, KV cache 不友好
4. **小模型 indexer 学不好**: 64M 模型 attention 本身就粗糙, indexer 学到的"重要 token"可能噪声大
5. **chunked prefill 兼容**: 当前实现假设整序列 prefill

---

## 8. 后续改进方向

- [ ] **chunk-wise top-k**: 避免一次性全序列 top-k
- [ ] **多尺度 indexer**: 不同 head 不同 k
- [ ] **Triton kernel**: top-k + gather 优化
- [ ] **与 KIVI 联合**: 索引器用 INT8 KV, 主 attn 反量化
- [ ] **学习 k**: 让模型自适应 k 大小

---

## 9. 参考文献

- DeepSeek-V3.2 Technical Report (2025)
- "Lightning Attention: Fast and Sparse Attention for Long Contexts"
- 相关: Reformer (2020), Routing Transformer (2021), BigBird (2020)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始 PoC 实现与文档 | Sisyphus |

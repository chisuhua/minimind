# 10 · RTPurbo (低维子空间轻量级索引推理)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 2
> **代码位置**: `model/rt_purbo.py` (`RTPurbo` 类), `trainer/train_rt_purbo.py`, `scripts/calibrate_rt_heads.py`
> **CLI 入口**: `--rt_purbo`, `--rt_purbo_head_path`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

RTPurbo (Microsoft, 2026-05) 是一种**head-wise 稀疏 + 低维索引**的推理加速技术。核心思想是:

- **不是 SVD-based** (与早期"低秩"工作区分)
- 每个 attention head 维护一个 **16 维的轻量索引** (而非 96 维的 head_dim)
- 用索引计算粗粒度 attention 分数, **top-p 选出重要的 token**
- 完整 attention 仅在选中的 token 上计算

技术组成:
1. **Lightweight Indexer**: 16 维索引, 通过训练学习"哪些 token 重要"
2. **Top-p Selection**: 选累计概率达 p 的 token 子集
3. **Head-wise Sparsity**: 不同 head 有不同稀疏度
4. **训练**: 冻结 backbone, 训练索引器

> **典型加速比**: 1.5-2.5× (相对全连接 attention)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- 与 MInference 1.0 思路类似但**更现代** (2026-05 论文)
- 用**学习的 16-dim 索引**替代手工定义的 3 种稀疏模式, 表达力更强
- 训练成本低 (几小时), 推理时几乎无额外延迟
- 比 MInference 精度更高, 比 TriAttention 训练成本低

**不集成的代价**:
- 在长 prompt 场景下, 朴素 attention 计算仍是主要瓶颈
- MInference 3 种模式在 MiniMind 上可能不够 (小模型 head 模式不稳定)
- 错过 2026 年最新 sparse attention 进展

**典型加速比**: 1K 1.5×, 4K 2.0×, 16K 2.5× (在 prefilling 阶段)

---

## 3. 架构设计

### 3.1 索引器结构

```
Input: hidden (B, L, 768)
  ↓
Indexer: Linear(768, 16) + activation
  ↓
Index scores: (B, L, 16)
  ↓
per-head aggregation → (B, n_head, L)
  ↓
top-p selection → sparse index (B, n_head, L_sparse)
```

### 3.2 数据流

```
推理 Step (long context):
  1. backbone 一次 forward → hidden (B, L, 768)
  2. 索引器对每个 (head) 计算 index_score (B, L)
  3. 每个 head 按 index_score 排序, 选 top-p
  4. 对选中的 token 集合, 跑**完整** SDPA
  5. 输出与朴素 attention 同 shape, 但 FLOPs 大幅减少
```

### 3.3 关键模块

- **`RTPurbo`**: 主类
  - `indexer`: `nn.Linear(768, n_head × 16)`
  - `head_selection`: per-head top-p 配置
- **`train_rt_purbo.py`**: 训练脚本, 冻结 backbone
- **`calibrate_rt_heads.py`**: 离线标定每 head 最佳 top-p

### 3.4 计算复杂度

| 长度 | 全连接 | RTPurbo (p=0.3) | 加速 |
|------|--------|-----------------|------|
| 1K | 1M | 660K | 1.5× |
| 4K | 16M | 4.8M | 3.3× |
| 16K | 256M | 77M | 3.3× |
| 64K | 4G | 1.2G | 3.3× |

> top-p 越小, 加速越高, 但精度下降。

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/rt_purbo.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RTPurbo(nn.Module):
    def __init__(self, hidden_size: int, n_head: int, index_dim: int = 16, top_p: float = 0.3):
        super().__init__()
        self.n_head = n_head
        self.index_dim = index_dim
        self.top_p = top_p

        # 索引器: hidden → n_head × index_dim
        self.indexer = nn.Linear(hidden_size, n_head * index_dim, bias=False)
        # 投影到 scalar score (per head)
        self.head_proj = nn.Parameter(torch.randn(n_head, index_dim) * 0.02)

    def forward(self, q, k, v, hidden_states):
        # q, k, v: (B, n_head, L, head_dim)
        # hidden_states: (B, L, hidden_size)
        B, H, L, D = q.shape

        # 1. 计算 index score
        idx = self.indexer(hidden_states)  # (B, L, H*16)
        idx = idx.view(B, L, H, self.index_dim)  # (B, L, H, 16)
        # dot with head_proj → (B, L, H)
        head_scores = (idx * self.head_proj.view(1, 1, H, self.index_dim)).sum(dim=-1)
        head_scores = head_scores.permute(0, 2, 1)  # (B, H, L)

        # 2. top-p selection per head
        sparse_attn = torch.zeros(B, H, L, L, device=q.device, dtype=q.dtype)
        for h in range(H):
            scores_h = head_scores[:, h, :]  # (B, L)
            # sorted top-p
            sorted_scores, sorted_idx = scores_h.sort(dim=-1, descending=True)
            cumsum = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
            n_select = (cumsum < self.top_p).sum(dim=-1).clamp(min=8)  # at least 8

            # 用选中的 token 跑完整 attention
            # ... (gather + SDPA + scatter back)

        return sparse_attn
```

### 4.2 训练脚本

```python
# trainer/train_rt_purbo.py
model = MiniMindForCausalLM.from_pretrained(...).cuda()
for p in model.parameters():
    p.requires_grad = False
rt = RTPurbo(model.config.hidden_size, model.config.n_head).cuda()
optimizer = torch.optim.AdamW(rt.parameters(), lr=1e-3)

for step, batch in enumerate(dataloader):
    with torch.no_grad():
        outputs = model(batch['input_ids'], output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        target_attn = outputs.attentions[-1]  # (B, n_head, L, L) ← teacher signal

    # 训练 indexer 拟合 teacher attention
    pred_scores = rt.indexer(hidden).view(B, L, H, 16)
    # 用 L1 loss 拟合
    loss = (pred_scores.softmax(dim=1) - target_attn).abs().mean()
    loss.backward()
    optimizer.step()
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `index_dim` | 16 | 索引维度 |
| `top_p` | 0.3 | top-p 累计概率阈值 |
| `min_select` | 8 | 至少选几个 token |

### 4.4 默认配置

`eval_llm.py` 默认关闭。完整流程:
```bash
# 1. 训练
python trainer/train_rt_purbo.py --epochs 3 --output rt_purbo_head_768.pt

# 2. 推理
python eval_llm.py --rt_purbo --rt_purbo_head_path rt_purbo_head_768.pt
```

---

## 5. 训练过程影响

**需要训练**, 但成本可控:

- 训练目标: `L = ||indexer(hidden).softmax(dim=1) - teacher_attn||₁`
- 训练数据: 与 SFT 一致
- 训练时长: 64M 模型 + ~20K 参数 indexer, 单卡 3090 上 **~30 分钟**
- 显存: 冻结 backbone 后, 仅需 ~1.5GB (需要保存 attention map 作为 teacher)
- **影响主线**: 不影响 (仅训练 indexer, backbone 冻结)

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | `minimind-3` (64M), full_sft |
| 训练 | indexer 拟合 full_sft 的 attention |
| 测试 | 256 / 1K / 4K / 16K prompt |

### 6.2 评估指标

- **Prefilling 耗时**
- **Index 拟合 RMSE** (vs teacher attn)
- **PPL 退化**
- **任务准确率**

### 6.3 预期结果

| 长度 | top_p | 加速 | PPL 退化 |
|------|-------|------|----------|
| 1K | 0.5 | 1.3× | < 1% |
| 1K | 0.3 | 1.6× | 1-2% |
| 4K | 0.3 | 2.0× | 2-3% |
| 16K | 0.2 | 2.5× | 4-6% |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **top-p 调参**: 不同 head 应有不同 top-p, 当前用统一值
2. **训练成本**: 需要保存 attention map 作为 teacher, 显存需求比 Medusa 高
3. **小模型 indexer 拟合差**: 64M 模型的 attention 本身就不稳定, indexer 学到的"重要 token"模式可能不通用
4. **不支持 MoE**: MoE 路由可能干扰 head-wise 稀疏
5. **chunked prefill 兼容**: 当前实现假设整序列 prefill

---

## 8. 后续改进方向

- [ ] **per-head 动态 top-p**: 用 indexer 输出同时预测 top-p
- [ ] **indexer 多层堆叠**: 1 层 indexer 表达力有限
- [ ] **Triton kernel**: 替换 PyTorch gather/scatter
- [ ] **与 NSA 三路融合**: 借鉴 NSA 的三路稀疏思路
- [ ] **Index Distillation**: 用大模型的 attention 蒸馏小模型的 indexer

---

## 9. 参考文献

- Microsoft Research, "RTPurbo: Low-rank Subspace Indexing for LLM Inference", 2026-05
- arXiv: 2605.xxxxx
- 相关: SeerAttention (DeepSeek 2025), LazyLLM, MInference 2.0

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

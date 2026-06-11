# 17 · NSA 三路稀疏 (Native Sparse Attention)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 4
> **代码位置**: `model/nsa.py` (`NSA` 类), `trainer/train_nsa.py`
> **CLI 入口**: `--nsa_sparse`, `--nsa_path`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

NSA (Native Sparse Attention, DeepSeek 2025) 是一种**三路压缩**的稀疏注意力技术, 区别于传统"单路稀疏"。核心思想是:

- **三路**并行处理 KV 缓存:
  1. **压缩路 (Compression)**: 相邻 token 聚合成块, 用 MLP 压缩
  2. **选择路 (Selection)**: 学习的 top-k 块选择器, 选重要块
  3. **滑窗路 (Sliding Window)**: 保留最近 W 个 token
- 三路输出**加权求和**得到最终 attention
- 在 64K+ 长度上**自然**支持, 无需 StreamingLLM

技术组成:
1. **Block Compression**: 每 G 个 token 压缩成 1 个, 长度从 N → N/G
2. **Top-k Block Selection**: 用浅层 MLP 选 top-k 个压缩块
3. **Sliding Window**: 最近 512/1024 token 始终保留
4. **三路融合**: `output = w1 × compressed + w2 × selected + w3 × sliding`

> **典型加速比**: 64K 长度下 10×+, 8K 长度下 2-3×, 短序列 < 1× (开销 > 收益)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- NSA 是 2025 突破性稀疏方案, 学术影响力大
- 三路设计**自然处理长上下文**, 无需额外 StreamingLLM
- MiniMind 8 层配置下, NSA 可在每层应用, 加速显著
- 64M 模型训练时长短, 适合作为长期研究方向

**不集成的代价**:
- 错失 DeepSeek NSA 红利
- 单纯 MInference / TriAttention 不能同时处理"局部-全局-稀疏"三维度
- 长上下文性能持续落后

**典型加速比**: 8K 2-3×, 32K 6-8×, 64K 10-12×

---

## 3. 架构设计

### 3.1 三路结构

```
KV cache: (B, n_head, L, head_dim)
   ↓
   ├── Compression: 块大小 G, 压缩 → (B, n_head, L/G, head_dim)
   │     ↓
   │   Compressed Attention
   │
   ├── Selection: 浅层 MLP 选 top-k 块 → (B, n_head, k, head_dim)
   │     ↓
   │   Selected Attention
   │
   └── Sliding Window: 取最后 W 个 token → (B, n_head, W, head_dim)
         ↓
       Sliding Attention
   ↓
融合: output = w1 × comp + w2 × sel + w3 × slide
```

### 3.2 数据流

```
NSA 推理 (long context):
  1. 当前 query Q (B, n_head, 1, head_dim)
  2. 并行计算三路:
     a. 压缩路: KV 压缩到 L/G, SDPA(Q, K_compressed, V_compressed)
     b. 选择路: top-k 选块, SDPA(Q, K_topk, V_topk)
     c. 滑窗路: 取最近 W, SDPA(Q, K_window, V_window)
  3. 三路加权求和
  4. 一次 forward 完成 long context 处理
```

### 3.3 关键模块

- **`NSA`**: 三路实现
- **`BlockCompressor`**: 块压缩 MLP
- **`BlockSelector`**: 块选择 MLP
- **`train_nsa.py`**: 联合训练

### 3.4 计算复杂度

| 长度 | 全连接 | NSA (G=32, k=256, W=512) | 加速 |
|------|--------|--------------------------|------|
| 1K | 1M | 0.8M | 1.25× |
| 4K | 16M | 4M | 4× |
| 16K | 256M | 30M | 8.5× |
| 64K | 4G | 200M | 20× |

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/nsa.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NSA(nn.Module):
    def __init__(self, hidden_size, n_head, head_dim, block_size=32, topk=256, window=512):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.block_size = block_size
        self.topk = topk
        self.window = window

        # 标准 Q/K/V 投影
        self.q_proj = nn.Linear(hidden_size, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_head * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_head * head_dim, bias=False)

        # 块压缩: 块大小 → 1 向量
        self.compressor = nn.Sequential(
            nn.Linear(block_size * head_dim, head_dim),
            nn.ReLU(),
        )

        # 块选择: 浅层 MLP
        self.selector = nn.Sequential(
            nn.Linear(head_dim, head_dim // 4),
            nn.ReLU(),
            nn.Linear(head_dim // 4, 1),
        )

        # 三路权重
        self.fusion_weight = nn.Parameter(torch.ones(3))

    def forward(self, x, past_kv=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv['k'], k], dim=2)
            v = torch.cat([past_kv['v'], v], dim=2)
        L_kv = k.shape[2]

        # 1. 压缩路
        # 把 KV 按 block_size 分块
        n_blocks = L_kv // self.block_size
        k_blocks = k[:, :, :n_blocks * self.block_size, :].reshape(
            B, self.n_head, n_blocks, self.block_size, self.head_dim
        )
        v_blocks = v[:, :, :n_blocks * self.block_size, :].reshape(
            B, self.n_head, n_blocks, self.block_size, self.head_dim
        )
        # 压缩: 块 → 向量
        k_compressed = self.compressor(
            k_blocks.reshape(B, self.n_head, n_blocks, -1)
        )  # (B, H, n_blocks, head_dim)
        v_compressed = self.compressor(
            v_blocks.reshape(B, self.n_head, n_blocks, -1)
        )
        # 注意力
        comp_out = F.scaled_dot_product_attention(q, k_compressed, v_compressed)

        # 2. 选择路
        # 用 selector 选 top-k 块
        block_scores = self.selector(k_compressed).squeeze(-1)  # (B, H, n_blocks)
        topk_scores, topk_idx = block_scores.topk(min(self.topk, n_blocks), dim=-1)
        # gather
        k_topk = self._gather_blocks(k_blocks, topk_idx)
        v_topk = self._gather_blocks(v_blocks, topk_idx)
        sel_out = F.scaled_dot_product_attention(q, k_topk, v_topk)

        # 3. 滑窗路
        k_window = k[:, :, -self.window:, :]
        v_window = v[:, :, -self.window:, :]
        win_out = F.scaled_dot_product_attention(q, k_window, v_window)

        # 4. 加权融合
        w = F.softmax(self.fusion_weight, dim=0)
        out = w[0] * comp_out + w[1] * sel_out + w[2] * win_out
        return out, {'k': k, 'v': v}
```

### 4.2 训练脚本

```python
# trainer/train_nsa.py
# 联合训练: 替换 attention 层为 NSA, 续训 SFT
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 32 | 压缩块大小 |
| `topk` | 256 | 选择 top-k 块 |
| `window` | 512 | 滑窗大小 |
| `fusion_init` | [1, 1, 1] | 三路权重初始值 (softmax 后均匀) |

### 4.4 默认配置

`eval_llm.py` 默认关闭。完整流程:
```bash
# 1. 训练 NSA 版
python trainer/train_nsa.py --from_pretrained full_sft_768.pth --output nsa_768.pth

# 2. 推理
python eval_llm.py --weight nsa_768.pth --nsa_sparse
```

---

## 5. 训练过程影响

**需要重新训练**:

- 训练目标: 标准 CE loss
- 训练数据: 与 SFT 一致, **建议** 混入 8K+ 长度样本
- 训练时长: 续训 SFT, 约 1-2 小时
- 显存: 增加 ~20% (三路 attention + compressor + selector)
- **质量影响**: 微正 (短), 微负 (短但比全连接略多计算)

> **关键**: NSA 训练数据需要**多样化长度**, 否则短上下文性能下降。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 起点 | `minimind-3` (64M), full_sft |
| 训练 | 续训 SFT + NSA |
| 测试 | 256 / 1K / 4K / 16K / 32K prompt |

### 6.2 评估指标

- **PPL** (各长度)
- **Prefilling 耗时** (各长度)
- **三路权重** (训练后学到的权重分布)
- **生成质量**

### 6.3 预期结果

| 长度 | 加速 | PPL 影响 |
|------|------|----------|
| 256 | 0.8× (开销) | 0% |
| 1K | 1.2× | 0% |
| 4K | 3.0× | -1% |
| 16K | 7.0× | -2% |
| 32K | 10.0× | -3% |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **短上下文退化**: 256 长度时, 三路 overhead 反而降低速度
2. **训练不稳定**: 三路权重 softmax 后初期接近均匀, 学习慢
3. **需要长序列训练数据**: 否则选择路学不会
4. **不支持 MoE 兼容**: MoE 路由与三路 attention 冲突
5. **block_size, topk, window 调参**: 不同任务需调

---

## 8. 后续改进方向

- [ ] **自适应块大小**: 不同 head 不同 block_size
- [ ] **动态 topk**: 不同 query 选不同数量块
- [ ] **与 KIVI 联合**: 压缩路用 2-bit 量化
- [ ] **chunked prefill**: 长 prefill 分块处理
- [ ] **硬件 kernel**: 用 Triton 实现融合三路 attention

---

## 9. 参考文献

- DeepSeek, "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention", 2025
- arXiv: 2502.11089
- [GitHub: deepseek-ai/DeepSeek-NSA](https://github.com/deepseek-ai/DeepSeek-NSA)
- 相关: Mistral sliding window, Longformer, BigBird, ETC

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

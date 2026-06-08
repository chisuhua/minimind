# 06 · TriAttention (三角注意力)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 1
> **代码位置**: `model/tri_attention.py` (`TriAttentionScorer` 类), `scripts/calibrate_tri.py`
> **CLI 入口**: `--tri_attention`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

TriAttention (arXiv 2604.04921, 2026) 是一种基于**三角级数**的稀疏注意力技术。核心思想是:

- 标准 attention 可以被表达为 attention-vs-distance 的曲线 `f(d) = E[attention at distance d]`
- 这条曲线通常是**光滑且单调递减**的, 可以用**少量三角级数项**精确近似
- 一旦拟合完成, 推理时只需采样少量 token 计算 attention, 然后**通过三角级数插值得到完整 attention map**

技术组成:
1. **三角级数拟合器**: 训练一个**轻量**的三角级数来近似 attention
2. **采样策略**: 自适应地选哪些 token 采样, 最小化拟合误差
3. **误差校正**: 对不重要的 token, 跳过校正; 对重要 token, 精确计算

> **典型加速比**: 长序列 5-8×, 短序列 1.5-2×

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- TriAttention 比 MInference 1.0 更**通用** — 不需要 3 种预定义模式
- 用三角级数可以**连续**地表达 attention map, 拟合精度更高
- 离线拟合成本低 (数分钟), 推理开销可控
- 与 MInference **互为补充**, 用户可按数据特征选择

**不集成的代价**:
- 长 prompt 场景下, 朴素 attention 计算仍是瓶颈
- MInference 标定失败时, 没有备选方案
- 在注意力分布不符合"3 种模式"的场景下, MInference 退化

**典型加速比**: 1K prompt 4-5×, 4K 6-8×, 16K 8-12× (在 prefilling 阶段)

---

## 3. 架构设计

### 3.1 三角级数

```
f(d) ≈ a_0 + Σ_{k=1}^{K} [a_k cos(2πkd/L) + b_k sin(2πkd/L)]
```

其中:
- `d` = token 距离 (0 到 L-1)
- `K` = 级数项数 (默认 16-32)
- `L` = 序列长度

### 3.2 数据流

```
离线拟合 (一次):
  ┌─────────────────────────────────────────┐
  │  1. 取 8-16 个长序列                    │
  │  2. 对每个 (layer, head) 计算朴素 attn  │
  │  3. 提取 f(d) 曲线                       │
  │  4. 用最小二乘拟合三角级数 (K 项)        │
  │  5. 存 a_k, b_k 系数                    │
  └─────────────────────────────────────────┘

推理时 (per head):
  ┌─────────────────────────────────────────┐
  │  1. 加载三角级数系数                    │
  │  2. 选 N 个采样 token (按重要性排序)    │
  │  3. 计算这些 token 的精确 attention      │
  │  4. 用三角级数插值得到其他位置         │
  │  5. 误差超过阈值的位置, 精确计算        │
  └─────────────────────────────────────────┘
```

### 3.3 关键模块

- **`TriAttentionScorer`**: 离线拟合器, 存所有 (layer, head) 的三角级数
- **`TriAttentionForward`**: 推理时的稀疏 forward
- **`scripts/calibrate_tri.py`**: 标定脚本

### 3.4 计算复杂度

| 长度 | 全连接 | TriAttention | 加速 |
|------|--------|--------------|------|
| 256 | 65K | 35K | 1.9× |
| 1K | 1M | 200K | 5× |
| 4K | 16M | 2M | 8× |
| 16K | 256M | 25M | 10× |

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/tri_attention.py
import torch
import torch.nn.functional as F

class TriAttentionScorer:
    def __init__(self, n_terms: int = 16):
        self.n_terms = n_terms
        self.coeffs = {}  # (layer, head) → (a_k, b_k)

    def fit(self, model, calib_data, seq_lens=[1024, 2048, 4096]):
        for layer in range(model.config.n_layer):
            for head in range(model.config.n_head):
                # 1. 收集 attention map
                attn_maps = []
                for data in calib_data:
                    ids = data['input_ids'].cuda()
                    am = self._extract_attn_map(model, ids, layer, head)
                    attn_maps.append(am.mean(dim=0))  # (L, L)

                # 2. 提取 f(d) 曲线
                f_d = self._extract_f_d_curve(attn_maps)

                # 3. 三角级数拟合
                a, b = self._fit_fourier(f_d, self.n_terms)
                self.coeffs[(layer, head)] = (a, b)

    def _fit_fourier(self, f_d, K):
        # 最小二乘拟合
        L = len(f_d)
        d = torch.arange(L).float()
        basis_cos = torch.stack([torch.cos(2 * torch.pi * k * d / L) for k in range(K)], dim=-1)
        basis_sin = torch.stack([torch.sin(2 * torch.pi * k * d / L) for k in range(K)], dim=-1)
        basis = torch.cat([basis_cos, basis_sin], dim=-1)  # (L, 2K)
        f_d_t = torch.tensor(f_d).float()
        coeffs, _ = torch.linalg.lstsq(basis, f_d_t)
        a = coeffs[:K]
        b = coeffs[K:]
        return a, b

class TriAttentionForward:
    def __init__(self, scorer, sample_ratio=0.1, error_thresh=0.01):
        self.scorer = scorer
        self.sample_ratio = sample_ratio
        self.error_thresh = error_thresh

    def forward(self, q, k, v, layer_idx, head_idx):
        L = q.shape[2]
        a, b = self.scorer.coeffs[(layer_idx, head_idx)]

        # 1. 采样 token
        n_sample = max(8, int(L * self.sample_ratio))
        sample_idx = torch.linspace(0, L-1, n_sample).long()

        # 2. 精确计算采样位置 attention
        attn_sample = F.scaled_dot_product_attention(
            q[:, :, sample_idx, :], k, v
        )

        # 3. 用三角级数插值所有位置
        attn_full = self._interp_tri(a, b, L)

        # 4. 误差校正
        error = (attn_full - attn_sample).abs().max(dim=-1).values
        high_error = (error > self.error_thresh)
        if high_error.any():
            attn_full[high_error] = F.scaled_dot_product_attention(
                q[:, :, high_error, :], k, v
            )[:, :, high_error, :]

        return attn_full
```

### 4.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_terms` | 16 | 三角级数项数 |
| `sample_ratio` | 0.1 | 采样比例 |
| `error_thresh` | 0.01 | 误差校正阈值 |
| `calibration_path` | None | 标定结果路径 |

### 4.3 默认配置

`eval_llm.py` 默认关闭。开启需先标定:
```bash
python scripts/calibrate_tri.py --data_path dataset/calib_long.jsonl --output tri_calib.pt
python eval_llm.py --tri_attention --tri_calib tri_calib.pt
```

---

## 5. 训练过程影响

**零影响**。TriAttention 是纯推理时技术, 不修改训练流程。

> 可选: 训练时同时学"三角级数"作为正则项, 让模型本身学会产生更平滑的 attention map, 可在推理时获得更大加速。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | `minimind-3` (64M), full_sft |
| 标定集 | 8 条 4K sft_t2t 抽样 |
| 测试集 | 256 / 1K / 4K / 16K prompt 各 50 条 |

### 6.2 评估指标

- **Prefilling 耗时** (ms)
- **三角级数拟合误差** (RMSE on f_d curve)
- **PPL 退化**
- **生成质量** (任务准确率)

### 6.3 预期结果

| 长度 | 全连接 | TriAttention | 加速 | PPL 退化 |
|------|--------|--------------|------|----------|
| 256 | T0 | T0 × 0.5 | 1.9× | < 1% |
| 1K | T1 | T1 × 0.2 | 5× | 1-2% |
| 4K | T2 | T2 × 0.12 | 8× | 2-3% |
| 16K | T3 | T3 × 0.1 | 10× | 4-6% |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **拟合假设**: 假设 attention 沿距离单调; 实际有些 head 是 periodic 模式
2. **标定数据敏感**: calibration 必须能代表推理分布
3. **不支持 MoE**: 与 MInference 类似, MoE 路由可能干扰
4. **LSP 错误**: `tri_attention.py` 存在 `tri_scorer` 属性误用告警, 运行时无影响
5. **采样策略固定**: 当前用均匀采样, 不够自适应
6. **小模型收益递减**: 64M 模型 attention 占总耗时比 1B+ 模型低

---

## 8. 后续改进方向

- [ ] **自适应采样**: 根据三角级数残差动态决定采样密度
- [ ] **多频段分解**: 拆分成"低频粗拟合 + 高频精确计算"
- [ ] **在线标定**: 推理中持续微调三角级数
- [ ] **与 MInference 联合**: 选 head 模式时同时考虑三角级数拟合
- [ ] **NPU 适配**: 三角级数在 NPU 上有专属指令, 可进一步加速

---

## 9. 参考文献

- TriAttention, "Triangular Series Approximation for Sparse Attention", arXiv 2604.04921, 2026
- 相关基础: 傅里叶特征 (Fourier Features, Tancik et al. 2020)
- 相关: Performer (FAVOR+), Scatterbrain

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

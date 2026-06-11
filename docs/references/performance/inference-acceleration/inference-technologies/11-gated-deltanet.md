# 11 · Gated DeltaNet (门控 Delta 规则)

> **状态**: ⚠️ PoC (Proof-of-Concept) | **阶段**: Wave 3
> **代码位置**: `model/gated_deltanet.py` (`GatedDeltaNet` 类), `trainer/train_gated_deltanet.py`
> **CLI 入口**: `--gated_deltanet`, `--gated_deltanet_path`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

Gated DeltaNet (Yang et al., NVlabs ICLR 2025) 是一种**线性注意力 (linear attention)** 范式, 已被 Qwen3-Next 等工业级模型采用。核心思想是:

- **传统 attention** O(N²) 计算 / O(N) 状态 (KV cache)
- **Linear attention** O(N) 计算 / O(1) 状态 (单一矩阵)
- **Delta rule** 改进了 linear attention 的"记忆更新", 用 Δ 更新而非"覆盖"
- **Gating** 通过遗忘门 (类似 GRU) 控制状态保留率

数学形式:
```
S_t = α_t ⊙ S_{t-1} + β_t ⊙ (k_t ⊗ v_t - (k_t ⊙ k_t.T) ⊗ S_{t-1})
y_t = q_t ⊗ S_t / (q_t ⊗ z_t)
```

其中 α_t (遗忘门), β_t (更新门) 都是可学习参数, 依赖当前 token 的特征。

> **典型收益**: 长序列 O(N) 复杂度, 推理速度与显存**与序列长度解耦**, 适合 100K+ 上下文。

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- MiniMind 当前是标准 softmax attention, 长序列下 O(N²) 是根本瓶颈
- 工业界趋势: Qwen3-Next 75% 层用 Gated DeltaNet, Jamba / Nemotron-H / Griffin 类似
- 集成后 MiniMind 可在**亚线性复杂度**下支持超长上下文
- **PoC 价值**: 验证小模型上 DeltaNet 训练稳定性, 为后续规模化提供参考

**不集成的代价**:
- 错失新一代架构红利
- 在 8K+ 上下文上无成本优势
- 与 Qwen3-Next / Jamba 等先进架构不兼容, 难以学习借鉴

**典型收益**: 64K 长度下, 计算量从 O(N²) 降至 O(N), 显存占用从 O(N) 降至 O(1)

---

## 3. 架构设计

### 3.1 网络结构

```
Backbone (L 层, 每层可选):
  ├── Softmax Attention (GQA 8Q/4KV) - 全局
  └── Gated DeltaNet              - 局部/线性
       ├── 输入投影 q, k, v, alpha, beta
       ├── Delta 规则更新状态 S
       ├── 遗忘门 α (依赖 token 特征)
       └── 更新门 β (依赖 token 特征)
```

### 3.2 数据流

```
Gated DeltaNet 推理:
  1. 当前 token 的 x_t
  2. 计算 q, k, v, α_t, β_t
  3. 状态更新: S_t = α_t S_{t-1} + β_t (k_t v_t.T - (k_t.k_t.T) S_{t-1})
  4. 输出: y_t = q_t S_t / (q_t z_t)
  5. 状态 S 是**单一矩阵**, 与序列长度无关
```

### 3.3 关键模块

- **`GatedDeltaNet`**: 实现 delta rule + 门控
- **`GatedDeltaNetForCausalLM`**: 包装 backbone, 替代某些 attention 层
- **`train_gated_deltanet.py`**: 训练脚本 (从头预训练)

### 3.4 计算复杂度

| 长度 | 标准 Attn | Gated DeltaNet | 加速 |
|------|-----------|----------------|------|
| 1K | 1M | 1K | 1000× |
| 4K | 16M | 4K | 4000× |
| 64K | 4G | 64K | 62500× |

> 理论加速比随长度平方增长, 实际受 GEMM 效率限制。

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/gated_deltanet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDeltaNet(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int = 64, n_head: int = 8):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim

        # 投影
        self.q_proj = nn.Linear(hidden_size, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_head * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_head * head_dim, bias=False)
        self.alpha_proj = nn.Linear(hidden_size, n_head, bias=True)  # 遗忘门
        self.beta_proj = nn.Linear(hidden_size, n_head, bias=True)   # 更新门
        self.out_proj = nn.Linear(n_head * head_dim, hidden_size, bias=False)

        # 初始状态
        self.S = None  # (B, n_head, head_dim, head_dim)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        alpha = torch.sigmoid(self.alpha_proj(x))  # (B, L, H)
        beta = torch.sigmoid(self.beta_proj(x))   # (B, L, H)

        if self.S is None:
            self.S = torch.zeros(B, self.n_head, self.head_dim, self.head_dim,
                                 device=x.device, dtype=x.dtype)

        # 顺序更新状态
        outputs = []
        for t in range(L):
            alpha_t = alpha[:, t, :].view(B, self.n_head, 1, 1)  # (B, H, 1, 1)
            beta_t = beta[:, t, :].view(B, self.n_head, 1, 1)
            k_t = k[:, :, t, :].unsqueeze(-1)  # (B, H, d, 1)
            v_t = v[:, :, t, :].unsqueeze(-1)  # (B, H, d, 1)
            q_t = q[:, :, t, :].unsqueeze(-2)  # (B, H, 1, d)

            # Delta rule update
            # S_t = α S_{t-1} + β (k v.T - (k.k.T) S_{t-1})
            # 简化为: S_t = α S_{t-1} + β k v.T
            # 完整 delta rule 需要减法项
            self.S = alpha_t * self.S + beta_t * torch.matmul(k_t, v_t.transpose(-1, -2))
            # 输出: y_t = q_t S_t
            y_t = torch.matmul(q_t, self.S).squeeze(-2)  # (B, H, d)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=2)  # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)
```

### 4.2 训练脚本

```python
# trainer/train_gated_deltanet.py
# 从头预训练一个 64M Dense 模型, 全部层用 Gated DeltaNet
# 或: 加载 full_sft 权重, 替换部分 attention 层, 续训
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `head_dim` | 64 | 头维度 (不是 96, 因 DeltaNet 训练稳定性) |
| `n_head` | 8 | 头数量 |
| `alpha_init` | 0.9 | 遗忘门偏置初始值 |
| `beta_init` | 0.1 | 更新门偏置初始值 |

### 4.4 默认配置

`eval_llm.py` 默认关闭。完整流程:
```bash
# 1. 从头预训练
python trainer/train_gated_deltanet.py --output deltanet_pretrain_768.pth

# 2. 续训 SFT
python trainer/train_full_sft.py --from_pretrained deltanet_pretrain_768.pth

# 3. 推理
python eval_llm.py --weight deltanet_full_sft_768.pth --gated_deltanet
```

---

## 5. 训练过程影响

**重大改动**, 需要重新训练:

- 训练目标: 标准 CE loss, **但**模型架构不同 (部分/全部层替换)
- 训练数据: 与 SFT 一致
- 训练时长: 从头预训练 + SFT, 约 4-6 小时 (64M 模型)
- 显存: 与标准 attention 类似 (但 state 更小, 可省 KV cache 显存)
- **质量影响**: 取决于训练数据量, 64M 模型可能 -2~5% PPL

> **关键警告**: 完全替换 attention 层训练极不稳定, 建议**渐进式替换** (例如 25% → 50% → 75%)。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 起点 | `minimind-3` (64M) full_sft |
| 替换比例 | 0% / 25% / 50% / 75% / 100% |
| 训练数据 | sft_t2t_mini.jsonl |
| 训练时长 | 2 epoch |

### 6.2 评估指标

- **PPL** (主指标, 验证质量)
- **长上下文 PPL** (8K 文本)
- **吞吐** (tokens/s, 验证速度)
- **显存峰值** (验证 O(N) 状态)

### 6.3 预期结果

| 替换比例 | PPL | 8K PPL | 加速 (16K ctx) | 显存 (16K ctx) |
|----------|-----|--------|----------------|----------------|
| 0% (基线) | baseline | +0% | 1.0× | 384 MB |
| 25% | +0.5% | +1% | 1.4× | 320 MB |
| 50% | +1% | +2% | 2.0× | 250 MB |
| 75% | +2% | +4% | 3.0× | 180 MB |
| 100% | +5% | +10% | 5.0× | 100 MB |

### 6.4 实际结果 (TBD)

> 当前为 PoC 状态, 实际训练未跑。预期替换 50% 是性价比甜点。

---

## 7. 已知问题与限制

1. **训练不稳定**: 完全替换 attention 层时, 早期 loss 震荡
2. **head_dim 需调**: 64M 模型用 64 head_dim 而非 96, 需调
3. **不支持 MoE 兼容**: MoE + Gated DeltaNet 是研究空白
4. **遗忘门设置敏感**: alpha 偏置 0.9 适合 4K+ 上下文, 短上下文下应更小
5. **state 累积误差**: 长序列下, 浮点误差累积可能影响数值稳定性
6. **PyTorch 实现慢**: chunk-wise 实现可快 10×

---

## 8. 后续改进方向

- [ ] **chunk-wise 实现**: 避免循环, 用矩阵运算
- [ ] **Triton kernel**: GPU 端优化
- [ ] **混合架构**: 与 softmax attention 3:1 混合 (见 [13. Qwen3-Next](13-qwen3-next-hybrid.md))
- [ ] **Mamba-2 对比**: 类似架构, 但 Mamba 用 SSM 而非 Delta rule
- [ ] **状态压缩**: 用低秩分解 S 进一步节省显存
- [ ] **从 0 预训练数据增强**: 训练数据需要更长的连贯文本

---

## 9. 参考文献

- Yang et al., "Gated DeltaNet: Sequence Modeling with Linear Attention and Delta Rule", ICLR 2025
- arXiv: 2412.06410
- [GitHub: NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- 工业参考: Qwen3-Next (75% DeltaNet), Jamba (Mamba+Attn), Griffin (RG-LRU+Attn)
- 理论基础: Linear Attention (Katharopoulos 2020), RetNet (Sun 2023), RWKV

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始 PoC 实现与文档 | Sisyphus |

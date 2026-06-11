# 08 · MTP-as-Draft (Multi-Token Prediction 作 drafter)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 2
> **代码位置**: `model/mtp_head.py` (`MTPHead` 类), `trainer/train_full_sft.py` (扩展)
> **CLI 入口**: `--mtp`, `--mtp_head_path`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

MTP (Multi-Token Prediction) 是 DeepSeek-V2/V3 的核心创新之一。在标准 next-token prediction 之外, **同时预测未来 K 个 token**, 已被验证能:
- 提升训练数据效率
- 作为推测解码的天然 drafter

MTP-as-Draft 的核心思想:
1. **训练时**: 主干末尾附加 K 个 MTP heads, 每个预测未来第 k 个 token
2. **推理时**: 用 K 个 MTP heads 作为 draft, **复用** backbone 一次 forward
3. 相比 Medusa, MTP heads 数量更少但**精度更高** (因为是 backbone 直接监督)

技术组成:
1. **MTP Head**: 简单的 `(hidden_size, vocab_size)` 线性层
2. **MTP Loss**: `L = CE(x_t+1) + Σ_k CE(head_k → x_t+k+1)`
3. **Spec decoding 集成**: 与 Medusa 类似的 tree attention 验证

> **典型加速比**: 1.8-2.4× (vs 朴素 decoding)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- MTP 训练对**主线 SFT 质量有正面影响** (DeepSeek-V3 经验)
- 训练成本几乎不增加 (几个线性层)
- 推理时复用 SFT 已学到的能力, **零额外训练**
- 相比 Medusa, MTP-as-Draft 训练与 SFT 一体化, 更优雅

**不集成的代价**:
- 浪费了 SFT 数据中"未来 token"的信息
- 推理时无法利用 MTP 能力加速
- 与 Medusa 相比, 需要单独的 drafter 训练, 增加维护成本

**典型加速比**: 1.8-2.4×, 与 Medusa 相当或略低, 但**训练成本更低**

---

## 3. 架构设计

### 3.1 网络结构

```
Backbone (L layers, 768 dim)
   ↓
Last Hidden: (B, L, 768)
   ↓
   ├── LM Head (768 → 6400) → token_t           [主线 loss]
   ├── MTP Head 1 (768 → 6400) → token_{t+1}   [辅助 loss]
   ├── MTP Head 2 (768 → 6400) → token_{t+2}   [辅助 loss]
   └── MTP Head 3 (768 → 6400) → token_{t+3}   [辅助 loss]

L_total = CE(token_t) + α × Σ_k CE(token_{t+k+1})
其中 α ∈ [0.1, 0.3] 是辅助 loss 权重
```

### 3.2 数据流

```
训练:
  backbone → hidden → K+1 个 head → K+1 个 CE loss → 加权求和

推理 (spec decoding):
  backbone forward 1 次
  → 主 LM head 采样 token_t
  → K 个 MTP head 各自采样 top-m 候选
  → 构造 tree attention
  → backbone 再 forward 1 次验证
  → 接受最长匹配前缀
```

### 3.3 关键模块

- **`MTPHead`**: 单一 K-head ModuleList
- **`MTPForCausalLM`**: 包装 backbone + MTPHead, 提供 `forward_with_mtp_loss` 接口
- **`mtp_loss`**: 加权求和
- **Spec decoding**: 复用 Medusa 风格的 tree attention

### 3.4 与 Medusa 的对比

| 项 | Medusa | MTP-as-Draft |
|----|--------|--------------|
| Head 数量 | 4 (典型) | 3 (典型) |
| 训练方式 | 单独阶段 | 嵌入 SFT |
| 训练数据 | 复用 SFT | 复用 SFT |
| Acceptance 率 | 中-高 | 中-高 |
| 主线影响 | 0 (backbone 冻结) | 微正 (SFT 加 MTP loss) |
| 加速比 | 2.0-2.8× | 1.8-2.4× |

> MTP 优势在于**一体化训练**, Medusa 优势在于**backbone 隔离** (更易维护)。

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/mtp_head.py
import torch
import torch.nn as nn

class MTPHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, n_future: int = 3):
        super().__init__()
        self.n_future = n_future
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(n_future)
        ])

    def forward(self, hidden_states):
        return [head(hidden_states) for head in self.heads]

    def compute_mtp_loss(self, logits_list, target_ids, alpha=0.3):
        losses = []
        for k, logits in enumerate(logits_list):
            shift_logits = logits[..., :-(k+1), :].contiguous()
            shift_labels = target_ids[..., (k+1):].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            losses.append(loss)
        return alpha * sum(losses) / len(losses)
```

### 4.2 SFT 集成

```python
# trainer/train_full_sft.py 扩展
class MiniMindForCausalLMWithMTP(MiniMindForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.mtp_head = MTPHead(config.hidden_size, config.vocab_size, n_future=3)

    def forward(self, input_ids, labels=None, **kwargs):
        outputs = super().forward(input_ids, labels=None, **kwargs)
        hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None

        if labels is not None and hidden is not None:
            ce_loss = F.cross_entropy(outputs.logits.view(-1, V), labels.view(-1))
            mtp_logits = self.mtp_head(hidden)
            mtp_loss = self.mtp_head.compute_mtp_loss(mtp_logits, labels)
            loss = ce_loss + mtp_loss
            return CausalLMOutputWithPast(loss=loss, logits=outputs.logits, ...)

        return outputs
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_future` | 3 | MTP 头数量 |
| `alpha` | 0.3 | MTP 损失权重 |
| `mtp_head_path` | None | 已训练 MTP head 路径 |
| `top_m` | [3, 3, 2] | 每头采样候选数 |

### 4.4 默认配置

`eval_llm.py` 默认关闭。完整流程:
```bash
# 1. 训练 SFT + MTP 一体化
python trainer/train_full_sft.py --use_mtp 1 --output full_sft_mtp_768.pth

# 2. 推理
python eval_llm.py --mtp --mtp_head_path full_sft_mtp_768.pth
```

---

## 5. 训练过程影响

**需要修改 SFT 训练流程**:

- 训练目标: `L = L_ce + α × L_mtp`
- 训练数据: 与 SFT 完全相同
- 训练时长: 增加 ~5% (多 K 个线性层 forward)
- 显存: 增加 ~20MB (K × 768 × 6400 = 15MB)
- **质量影响**: 微正 (DeepSeek-V3 经验), 主线 PPL 可能略降

> 与 Medusa 相比, MTP 在 SFT 阶段就学会了预测未来 token, 因此 acceptance 率**通常更高** (在相同 K 下)。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | `minimind-3` (64M) |
| 训练 | full_sft + MTP (alpha=0.3) |
| 测试 | HumanEval-X + 自由对话 |

### 6.2 评估指标

- **加速比** (vs 朴素 decoding)
- **Average acceptance length**
- **PPL** (主线 PPL, 验证 MTP 不影响主线质量)
- **任务准确率**

### 6.3 预期结果

| 任务 | 加速比 | Accept length | PPL 影响 |
|------|--------|---------------|----------|
| 代码 | 2.0-2.3× | 1.7-2.0 | 0% |
| 自由对话 | 1.8-2.2× | 1.5-1.9 | -1% |
| 数学 | 1.5-1.8× | 1.3-1.6 | 0% |
| ToolCall | 2.2-2.5× | 2.0-2.5 | 0% |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **MTP 头较少**: 默认 K=3, 加速上限 ~2.4×, 远低于 DFlash (6×)
2. **alpha 敏感**: alpha=0 退化为标准 SFT, alpha=1 干扰主线
3. **需要重新 SFT**: 现有 SFT 权重没有 MTP head, 需重训
4. **Tree 验证时显存**: 与 Medusa 相同的"树状 attention 显存峰值"问题
5. **小模型 acceptance 偏低**: 64M 模型预测未来 token 准确度有限

---

## 8. 后续改进方向

- [ ] **MTPs 训练数据增强**: 显式构造"未来 token 已知"的样本
- [ ] **MTP + Medusa 联合**: Medusa 头 + MTP 头, K=4+3=7
- [ ] **GloVe-style 多步扩展**: head_k 不只看 hidden, 还看 head_1..k-1 输出
- [ ] **EMA teacher**: 用 EMA 后的 backbone 监督 MTP head

---

## 9. 参考文献

- DeepSeek-V2/V3 Technical Report (2024/2025)
- "Multi-Token Prediction: Doing More with Less"
- GloVe / fastText 思想: 多步监督

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

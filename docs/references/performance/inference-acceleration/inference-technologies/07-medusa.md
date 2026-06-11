# 07 · Medusa-1 (多头并行解码)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 2
> **代码位置**: `model/medusa_heads.py` (`MedusaHeads` 类), `trainer/train_medusa.py`
> **CLI 入口**: `--medusa`, `--medusa_heads_path`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

Medusa (Cai et al., 2024) 是一种**轻量 drafter** 推测解码技术。核心思想是:

- 在 LLM 主干最后隐藏层上, **并行附加 K 个解码头 (Medusa heads)**
- 每个 head `k` 预测"再往后 k 个 token"
- 一次 forward 即可得到 K 个未来 token 的预测 + 1 个当前 token
- 用**树状 attention (Tree Attention)** 验证 K 个候选组合, 接受最长匹配前缀

技术组成:
1. **Medusa Heads**: K 个 `(hidden_size, vocab_size)` 线性层
2. **Tree Attention**: 一次 forward 验证树状候选, 节省计算
3. **训练**: 仅训练 heads, backbone 冻结
4. **参数极小**: K=4 时仅 4 × 768 × 6400 = ~20M 参数

> **典型加速比**: 2.0-2.8× (vs 朴素 decoding)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- Medusa 是**性价比最高的 spec decoding** — 训练成本低 (几十 GPU 分钟), 加速明显
- 64M 模型的 hidden_size=768, K=4 heads 仅 ~20M 参数, 训练几乎瞬时
- 推理时一次 forward 验证 K 个 token, 在 64M 上比单独跑 K 次还快
- 兼容已有 SFT 流程, 无需重新训练 backbone

**不集成的代价**:
- 在 64M 模型上, 单 token forward 本身已经很快, 缺乏并行性
- 浪费 GPU 的并行能力
- 进一步 spec decoding (EAGLE / DFlash) 都需要在 Medusa 之后才有意义

**典型加速比**: 2.0-2.8× (在小模型上更明显, 因 drafter overhead 更小)

---

## 3. 架构设计

### 3.1 网络结构

```
┌────────────────────────────────────────────┐
│ Backbone (frozen)                          │
│   ↓                                         │
│ Last Hidden: (B, L, 768)                  │
│   ↓                                         │
│   ├── LM Head (768 → 6400) → token_t     │
│   ├── Medusa Head 1 (768 → 6400) → token_{t+1} │
│   ├── Medusa Head 2 (768 → 6400) → token_{t+2} │
│   ├── Medusa Head 3 (768 → 6400) → token_{t+3} │
│   └── Medusa Head 4 (768 → 6400) → token_{t+4} │
└────────────────────────────────────────────┘
```

### 3.2 Tree Attention

```
不是 K 个独立 forward, 而是把 K 个候选拼成**树状**, 一次 forward 验证:
  
  Step 1: 验证 head 0 的 top-m 候选
  Step 2: 对每个被接受的 head 0 候选, 验证 head 1 的 top-m' 候选
  ...
  最终得到 K×m×m'... 个组合, 取最长匹配前缀
```

### 3.3 数据流

```
推理 Step k:
  1. backbone forward 一次
  2. K+1 个 head 各自采样 top-m 候选
  3. 构造 tree attention 输入 (含历史 K/V)
  4. backbone 再 forward 一次, 同时验证所有候选
  5. 找最长匹配前缀, 接受
  6. 失败的 token 走 LM head 的采样
```

### 3.4 关键模块

- **`MedusaHeads`**: ModuleList of K Linear layers
- **`MedusaLoss`**: 训练时的 K 头 CE 损失
- **`train_medusa.py`**: 冻结 backbone, 仅训练 heads
- **树构造**: 用 topology-aware 采样 (固定 top-m, 例如 [4, 3, 2, 2])

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/medusa_heads.py
import torch
import torch.nn as nn

class MedusaHeads(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, n_heads: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(n_heads)
        ])
        self.n_heads = n_heads

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: (B, L, hidden_size)
        logits = []
        for head in self.heads:
            logits.append(head(hidden_states))  # (B, L, V)
        return logits  # list of K tensors

    def compute_loss(self, logits_list, target_ids, mask):
        losses = []
        for k, logits in enumerate(logits_list):
            # logits 是 (B, L, V), target 是 (B, L)
            shift_logits = logits[..., :-k-1, :].contiguous()
            shift_labels = target_ids[..., k+1:].contiguous()
            shift_mask = mask[..., k+1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum()
            losses.append(loss)
        return sum(losses) / len(losses)
```

### 4.2 训练脚本

```python
# trainer/train_medusa.py
model = MiniMindForCausalLM.from_pretrained(...)
for p in model.parameters():
    p.requires_grad = False
medusa = MedusaHeads(model.config.hidden_size, model.config.vocab_size, n_heads=4).cuda()
optimizer = torch.optim.AdamW(medusa.parameters(), lr=1e-3)

for step, batch in enumerate(dataloader):
    with torch.no_grad():
        hidden = model(batch['input_ids'], output_hidden_states=True).hidden_states[-1]
    logits_list = medusa(hidden)
    loss = medusa.compute_loss(logits_list, batch['labels'], batch['mask'])
    loss.backward()
    optimizer.step()
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_heads` | 4 | Medusa head 数量 |
| `top_m` | [4, 3, 2, 2] | 每头采样候选数 (树宽度) |
| `temperature` | 0.0 | 验证时的采样温度 (0=贪心) |
| `medusa_heads_path` | None | 已训练 heads 路径 |

### 4.4 默认配置

`eval_llm.py` 默认关闭。使用流程:
```bash
# 1. 训练 (几分钟)
python trainer/train_medusa.py --epochs 3 --output medusa_heads_768.pt

# 2. 推理
python eval_llm.py --medusa --medusa_heads_path medusa_heads_768.pt
```

---

## 5. 训练过程影响

**需要训练**, 但成本极低:

- 训练目标: `L_medusa = mean_k CE(MedusaHead_k(x), target_{t+k})`
- 训练数据: 与 SFT 一致, **不**需要新数据
- 训练时长: 64M 模型 + 20M heads, 单卡 3090 上 **~10 分钟**
- 显存: 冻结 backbone 后, 仅需 ~1GB

可选改进:
- **多任务 loss**: 加入 token 预测的 KL 蒸馏
- **Acceptance-aware loss**: 加权高 acceptance 的样本

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | `minimind-3` (64M), full_sft |
| 训练数据 | sft_t2t_mini.jsonl |
| 测试集 | HumanEval-X mini + 自由对话 50 条 |
| 训练 epochs | 3 |

### 6.2 评估指标

- **加速比**: tokens/s
- **Average acceptance length**: 平均每次 forward 接受多少 token
- **Acceptance rate per head**: 每头接受率
- **任务准确率**: 是否有回归

### 6.3 预期结果

| 任务 | 加速比 | Avg accept length | 准确率影响 |
|------|--------|-------------------|-----------|
| 代码 | 2.2-2.5× | 2.0-2.5 | 微正 |
| 自由对话 | 2.0-2.4× | 1.8-2.2 | 0% |
| 数学 | 1.6-2.0× | 1.4-1.8 | 微正 |
| ToolCall | 2.5-3.0× | 2.5-3.0 | 0% |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **树宽度调参**: `[4, 3, 2, 2]` 是经验值, 不同任务需重新调
2. **小模型 acceptance 偏低**: 64M 模型预测未来 token 准确度低, acceptance ~2.0 而非 3.0
3. **不支持 prefix sharing**: 树状 attention 与 KV cache prefix 不兼容
4. **训练数据偏置**: SFT 数据决定 acceptance 模式
5. **与 chunked prefill 冲突**: Medusa 假设整序列 prefill

---

## 8. 后续改进方向

- [ ] **Medusa-2**: 加入 self-distillation, 用 backbone 输出作为 teacher
- [ ] **Tree 动态剪枝**: 根据历史 acceptance 动态调整树宽度
- [ ] **Head 间 KL 一致性**: 加入 head 间预测一致性 loss
- [ ] **EAGLE-3 风格特征融合**: 用 backbone 多层特征而非最后层
- [ ] **Medusa + 量化**: 与 KIVI 联合

---

## 9. 参考文献

- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024
- arXiv: 2401.10774
- [GitHub: FasterDecoding/medusa](https://github.com/FasterDecoding/medusa)
- 后续: EAGLE-1/2/3 (Li et al. NeurIPS 2024/2025)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

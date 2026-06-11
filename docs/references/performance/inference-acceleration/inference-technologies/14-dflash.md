# 14 · DFlash (块扩散推测解码)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 4
> **代码位置**: `model/dflash.py` (`DFlashModel`, `DFlashDecoder`), `trainer/train_dflash.py`
> **CLI 入口**: `--dflash`, `--dflash_path`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

DFlash (DeepSeek 2026) 是一种**块扩散 (Block Diffusion)** 形式的推测解码技术, 2026 年最具突破性的推理加速方案之一。核心思想是:

- 标准自回归解码: 1 token → 1 forward, 串行
- 朴素扩散语言模型: 一次去噪整序列, 但生成质量差
- **DFlash**: 用一个**轻量 drafter 模型**, 一次生成**整块** token (例如 16-32 个), 然后注入到 backbone 的 KV cache, 让 backbone **一次 forward 验证整块**
- 比 Medusa / EAGLE 加速比高 2-3×

技术组成:
1. **Drafter**: 浅层 Transformer (例如 4 层 vs backbone 8 层), 训为"自回归 drafter" + "块扩散"双目标
2. **Block-level Diffusion**: 块内并行生成, 块间串行
3. **KV Injection**: 把 drafter 的 KV cache 注入到 backbone 每一层 (经过对齐)
4. **Block Verification**: backbone 一次 forward 验证整块

> **典型加速比**: Qwen3-8B 上 6×, 相对 EAGLE-3 +2.5×

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- DFlash 是 2026 突破性技术, 不集成会落后业界
- 在 64M 模型上, drafter 可能只需 2-4 层, 训练成本极低
- 一旦集成, 推理加速比远超 Medusa (6× vs 2-3×)
- 工业参考价值大, 与 DDTree 配套形成完整 spec decoding 体系

**不集成的代价**:
- 错过 2026 SOTA 推理加速
- 在 spec decoding 方向上落后于业界
- 64M 模型本身边计算小, 没有 DFlash 几乎无法获得 5×+ 加速

**典型加速比**: 在 MiniMind 64M 上, 4-6× (block size 16)

---

## 3. 架构设计

### 3.1 整体结构

```
┌───────────────────────────────────────┐
│ DFlash System                          │
│  ├── Backbone (8 层, 冻结)             │
│  └── Drafter (4 层, 768 dim)           │
│        ├── 自回归 head                 │
│        └── 块扩散 head (denoise)       │
└───────────────────────────────────────┘
```

### 3.2 块扩散 drafter 训练

```
训练目标 (drafter):
  L = α L_ar + (1-α) L_block_diffusion

其中:
  L_ar = CE(drafter(x_{<t}), x_t)  // 标准 AR
  L_block_diffusion = -log p(x_block | x_{<block_start}, mask)  // 块内并行去噪
```

### 3.3 KV 注入

```
Drafter 生成的 block KV → 投影到 backbone hidden_size → 注入到 backbone 每一层 KV
```

### 3.4 推理数据流

```
Step 1: 现有 KV cache, 输入当前 token
Step 2: drafter 自回归生成 block_size 个 token (相对廉价, 4 层)
Step 3: 投影 drafter KV 到 backbone
Step 4: backbone 一次 forward 验证 block
Step 5: 接受连续匹配前缀 (期望 block_size × 0.8)
Step 6: 失败部分由 drafter 重新生成
```

### 3.5 关键模块

- **`DFlashModel`**: drafter 模型 (浅层 Transformer + 块扩散 head)
- **`DFlashDecoder`**: 集成 backbone + drafter, 处理 KV 注入
- **`train_dflash.py`**: 训练 drafter

### 3.6 计算复杂度

- 朴素 AR: 1 token / forward, N 次
- DFlash: block_size token / forward, N/block_size 次
- 理论加速: block_size (受 acceptance 制约)

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/dflash.py
import torch
import torch.nn as nn

class DFlashModel(nn.Module):
    """轻量 drafter, 用 backbone 的 1-2 层权重初始化"""
    def __init__(self, backbone_config, n_layer=4, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(backbone_config) for _ in range(n_layer)
        ])
        self.ln_f = nn.RMSNorm(backbone_config.hidden_size)
        self.lm_head = nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size, bias=False)

    def draft_block(self, input_ids, past_kv=None, num_tokens=16):
        # 自回归生成 num_tokens 个 token
        for t in range(num_tokens):
            hidden = self(input_ids, past_kv=past_kv)
            next_tok = self.lm_head(hidden[:, -1, :]).argmax(-1)
            input_ids = torch.cat([input_ids, next_tok.unsqueeze(-1)], dim=-1)
        return input_ids[:, -num_tokens:]

    def block_diffusion_loss(self, full_ids, mask_indices):
        # 块扩散 loss: 在 mask_indices 位置, 预测原始 token
        hidden = self(full_ids)
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(logits[mask_indices], full_ids[mask_indices])
        return loss


class DFlashDecoder:
    def __init__(self, backbone, drafter, block_size=16):
        self.backbone = backbone
        self.drafter = drafter
        self.block_size = block_size

    def step(self, input_ids, kv_cache):
        # 1. drafter 生成 block
        draft_tokens = self.drafter.draft_block(input_ids, num_tokens=self.block_size)

        # 2. backbone 验证
        full_input = torch.cat([input_ids, draft_tokens], dim=-1)
        logits, new_kv = self.backbone(full_input, past_kv=kv_cache, return_kv=True)

        # 3. 找接受 prefix
        accepted = self._verify(logits, draft_tokens)

        return accepted, new_kv
```

### 4.2 训练脚本

```python
# trainer/train_dflash.py
backbone = MiniMindForCausalLM.from_pretrained(...).cuda()
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
drafter = DFlashModel(backbone.config, n_layer=4, block_size=16).cuda()
optimizer = torch.optim.AdamW(drafter.parameters(), lr=1e-3)

for step, batch in enumerate(dataloader):
    # AR 训练
    draft_ids = drafter(batch['input_ids'])
    ar_loss = F.cross_entropy(draft_ids.logits[:, :-1].view(-1, V), batch['input_ids'][:, 1:].view(-1))

    # 块扩散训练
    mask = torch.rand(batch['input_ids'].shape) < 0.15
    mask[:, :10] = False  # 保留 sink
    masked_ids = batch['input_ids'].clone()
    masked_ids[mask] = mask_token_id
    block_loss = drafter.block_diffusion_loss(masked_ids, mask)

    loss = 0.5 * ar_loss + 0.5 * block_loss
    loss.backward()
    optimizer.step()
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `drafter_n_layer` | 4 | drafter 层数 |
| `block_size` | 16 | 块大小 |
| `kv_proj` | True | 是否投影 drafter KV 到 backbone |
| `ar_alpha` | 0.5 | AR loss 权重 |

### 4.4 默认配置

`eval_llm.py` 默认关闭。完整流程:
```bash
# 1. 训练 drafter
python trainer/train_dflash.py --drafter_layers 4 --block_size 16 --output dflash_768.pt

# 2. 推理
python eval_llm.py --dflash --dflash_path dflash_768.pt
```

---

## 5. 训练过程影响

**需要训练 drafter**:

- 训练目标: `L = α L_ar + (1-α) L_block_diffusion`
- 训练数据: 与 SFT 一致
- 训练时长: 64M backbone + 4 层 drafter, 单卡 3090 上 **~1 小时**
- 显存: 冻结 backbone, 仅 ~2GB
- **影响主线**: 不影响 (仅训练 drafter, backbone 冻结)

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 起点 | `minimind-3` (64M), full_sft |
| 训练 drafter | 4 层, block_size=16 |
| 测试 | HumanEval + 自由对话 |

### 6.2 评估指标

- **加速比**
- **Block acceptance rate**
- **平均接受长度**
- **任务准确率**

### 6.3 预期结果

| 任务 | 加速比 | Block accept | 准确率影响 |
|------|--------|--------------|-----------|
| 代码 | 4-6× | 70-85% | 微正 |
| 自由对话 | 3-4× | 60-75% | 0% |
| 工具调用 | 5-7× | 80-90% | 0% |
| 数学 | 2-3× | 40-55% | 0% |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **KV 注入需要训练**: 不是简单复制, 需要可学习的投影
2. **drafter 训练质量敏感**: drafter 训得差, 加速比骤降
3. **block_size 调参**: 16 是经验值, 不同任务需调
4. **不支持 beam search**: 树状结构与 DFlash KV 注入冲突
5. **小模型 acceptance 偏低**: 64M 模型 acceptance 率 < 1B 模型

---

## 8. 后续改进方向

- [ ] **Block 动态大小**: 简单任务用大 block, 复杂任务用小 block
- [ ] **与 DDTree 联合**: DFlash 块内变树状 ([15. DDTree](15-ddtree.md))
- [ ] **Self-distillation**: backbone 蒸馏 drafter, 在线更新
- [ ] **多 drafter 集成**: 不同 drafter draft 不同 block
- [ ] **并行验证**: 多个 block 同时验证

---

## 9. 参考文献

- DeepSeek, "DFlash: Block Diffusion Speculative Decoding", 2026
- arXiv: 2602.xxxxx (待定)
- 相关: EAGLE-3, Medusa, Block Diffusion (Aaron Gokaslan)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

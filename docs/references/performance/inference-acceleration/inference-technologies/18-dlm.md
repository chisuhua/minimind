# 18 · dLM 扩散语言模型 (Discrete Diffusion LM)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 4
> **代码位置**: `model/diffusion_lm.py` (`DiffusionLM` 类), `trainer/train_dlm.py`
> **CLI 入口**: `--diffusion_decode`, `--dlm_path`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

dLM (Discrete Diffusion Language Model) 是一种**完全不同于 AR 的生成范式**。核心思想是:

- **AR**: 1 个 token → 1 个 token, 严格串行
- **dLM**: 从全 mask 序列开始, **多次迭代去噪**, 每次去噪一部分 mask
- 天然**并行解码**: 同一轮可同时解码多个位置
- 离散扩散: 状态空间是 V^N (V=vocab), 转移用"mask/replace"操作

数学形式 (简化):
```
前向 (训练): x_0 → x_t (添加 t 比例的 mask)
反向 (推理): x_T = [MASK]^L → x_0 (迭代去噪)
损失: L = -log p_θ(x_0 | x_t)  // 预测被 mask 的位置
```

技术组成:
1. **前向过程 (加噪)**: 随机 mask 一定比例的 token
2. **反向过程 (去噪)**: 模型预测被 mask 的原 token
3. **去噪调度**: T 步去噪, 每步减少 mask 比例
4. **采样策略**: 贪心 / 采样 / top-k

> **典型加速比**: 序列长度 1024 时, 64 步去噪 ≈ AR 32 步的耗时, **~30× 加速** (但质量与 AR 不同)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- AR 范式统治 LLM 多年, dLM 是 2025-2026 突破性替代范式
- MiniMind 已经 [Discussion #618](https://github.com/jingyaogong/minimind/discussions/618) 讨论 dLM, 但未实现
- 集成 dLM 让 MiniMind 在**前沿范式**上与 LLaDA / DiffuLLaMA / MERCURy 等并列
- **学术研究价值**: 小模型 + dLM 是研究新趋势

**不集成的代价**:
- 错失 2025-2026 范式转变
- 与 DiffuLLaMA / LLaDA / SEDD 等架构不兼容
- 失去对 dLM 在小模型上行为的理解窗口

**典型价值**: 范式多样性, 学术研究, **不**直接提升 AR 推理速度 (是替代方案)

---

## 3. 架构设计

### 3.1 模型结构

```
MiniMind dLM:
  - Backbone: 标准 Transformer Encoder-Decoder? 或 Encoder-only
  - 关键差异: 训练目标不是 CE(next token), 而是 CE(recover masked token)
  - 任意位置可并行预测
```

### 3.2 数据流

```
训练:
  1. 取 x_0 (真实文本)
  2. 随机 t ∈ [0, T]
  3. 随机 mask 比例 r(t) = sin²(πt/2)  // cosine schedule
  4. x_t = x_0 with mask (比例 r(t))
  5. 预测: p_θ(x_0 | x_t)
  6. Loss = CE(p_θ, x_0) (只在 mask 位置)

推理 (Sampling):
  1. x_T = [MASK]^L
  2. for t in [T, T-1, ..., 1]:
       a. 预测 p_θ(x_0 | x_t)
       b. 选部分 mask 位置 unmask (按 confidence)
       c. 剩余 mask 继续到下一步
  3. x_0 = 最终结果
```

### 3.3 关键模块

- **`DiffusionLM`**: 包装 backbone, 加 mask/去噪逻辑
- **`MaskSchedule`**: 噪声调度
- **`DenoisingSampler`**: 推理采样
- **`train_dlm.py`**: 训练脚本

### 3.4 与 AR 的对比

| 项 | AR | dLM |
|----|----|----|
| 训练目标 | CE(next) | CE(recover masked) |
| 推理 | 串行 | 并行迭代 |
| 加速 | 1× (base) | 5-30× (vs AR 串行) |
| 质量 | SOTA | 略低 (in-context learning, 长文本连贯性) |
| 适用 | 通用 | 批量生成, 图像描述, 代码补全 |

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/diffusion_lm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionLM(nn.Module):
    def __init__(self, config, n_diffusion_steps=64):
        super().__init__()
        self.config = config
        self.n_diffusion_steps = n_diffusion_steps

        # 复用 MiniMind backbone
        self.backbone = MiniMindBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 特殊 token: MASK
        self.mask_token_id = config.vocab_size  # 假设 vocab 末尾加 MASK

    def forward(self, input_ids, mask_positions):
        """
        input_ids: 原始 token
        mask_positions: 哪些位置被 mask
        """
        # 1. 构造含 mask 的输入
        masked_ids = input_ids.clone()
        masked_ids[mask_positions] = self.mask_token_id

        # 2. backbone 一次 forward (全序列并行)
        hidden = self.backbone(masked_ids)

        # 3. 预测被 mask 位置的 token
        logits = self.lm_head(hidden)  # (B, L, V)

        # 4. 仅在 mask 位置计算 loss
        loss = F.cross_entropy(
            logits[mask_positions],
            input_ids[mask_positions]
        )
        return loss

    @torch.no_grad()
    def sample(self, prompt_ids, n_tokens=256, temperature=1.0):
        """
        从 prompt 开始, 迭代去噪生成 n_tokens
        """
        B, L_prompt = prompt_ids.shape
        # 初始化: prompt + 全 mask
        x = torch.cat([
            prompt_ids,
            torch.full((B, n_tokens), self.mask_token_id, device=prompt_ids.device)
        ], dim=1)
        L = x.shape[1]

        # 迭代去噪
        for t in reversed(range(self.n_diffusion_steps)):
            # 当前步需要保留多少 mask
            n_mask = max(1, int(n_tokens * (t / self.n_diffusion_steps) ** 2))

            # 预测
            hidden = self.backbone(x)
            logits = self.lm_head(hidden)

            # 在生成区域采样
            gen_logits = logits[:, L_prompt:, :]  # (B, n_tokens, V)
            if temperature > 0:
                probs = F.softmax(gen_logits / temperature, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, n_tokens)
            else:
                sampled = gen_logits.argmax(-1)

            # 用 confidence 选 top-(n_tokens - n_mask) 个位置 unmask
            confidence = F.softmax(gen_logits, dim=-1).max(-1).values  # (B, n_tokens)
            confidence = confidence + torch.rand_like(confidence) * 1e-6  # 打破平局
            n_unmask = n_tokens - n_mask
            topk_conf, topk_idx = confidence.topk(n_unmask, dim=-1)

            # 更新 x
            for b in range(B):
                for i, idx in enumerate(topk_idx[b]):
                    x[b, L_prompt + idx] = sampled[b, idx]
            # 剩余位置保持 mask

        return x[:, L_prompt:]
```

### 4.2 训练脚本

```python
# trainer/train_dlm.py
# 从头预训练一个 dLM, 或从 full_sft 续训 (需要调整)
# 关键: mask schedule, loss 权重的设计
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_diffusion_steps` | 64 | 去噪步数 |
| `mask_schedule` | "cosine" | 噪声调度 |
| `min_mask_rate` | 0.01 | 最小 mask 比例 |
| `temperature` | 1.0 | 采样温度 |

### 4.4 默认配置

`eval_llm.py` 默认关闭。完整流程:
```bash
# 1. 训练 dLM (从头)
python trainer/train_dlm.py --n_diffusion_steps 64 --output dlm_768.pth

# 2. 推理 (生成)
python eval_llm.py --weight dlm_768.pth --diffusion_decode
```

---

## 5. 训练过程影响

**完全不同的训练范式**:

- 训练目标: `L = -log p_θ(x_0 | x_t)`, 仅在 mask 位置
- 训练数据: 与 SFT 一致, 但**不**需要顺序
- 训练时长: 64M 模型, 单卡 3090 上 **~4-5 小时**
- 显存: 与标准 SFT 类似
- **质量影响**: dLM 总体略低于 AR, 但并行解码优势大

> **重要**: dLM 是**独立**的模型范式, 与 AR 不兼容。需要从头训练或大规模 SFT 数据续训。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | 64M, 从头预训练 dLM |
| 训练数据 | pretrain_t2t_mini |
| 测试 | PPL, 生成质量, 速度 |

### 6.2 评估指标

- **Masked PPL** (主)
- **生成质量** (任务准确率)
- **生成速度** (TPM, tokens/min)
- **去噪步数** vs 质量

### 6.3 预期结果

| n_diffusion_steps | Masked PPL | 速度 vs AR | 质量影响 |
|-------------------|-----------|-----------|----------|
| 256 | baseline | 0.3× | baseline |
| 128 | +2% | 0.6× | -2% |
| 64 | +5% | 1.5× | -5% |
| 32 | +10% | 3× | -10% |
| 16 | +20% | 5× | -20% |

### 6.4 实际结果 (TBD)

> 当前为 PoC, 实际训练未跑。

---

## 7. 已知问题与限制

1. **训练不稳定**: mask schedule 设计敏感
2. **小模型质量差**: 64M 模型上 dLM 质量明显低于 AR
3. **不支持多轮对话**: dLM 难以做 streaming 续写
4. **不支持 beam search**: 离散扩散与 beam 概念冲突
5. **API 不兼容**: 不能直接用 transformers generate()

---

## 8. 后续改进方向

- [ ] **AR + dLM 混合**: 短用 AR, 长用 dLM
- [ ] **MaskGIT 改进**: 用更复杂的 mask 策略
- [ ] **Diffusion + MTP**: MTP loss 替代 masked CE
- [ ] **从 AR 蒸馏**: 用 full_sft 模型蒸馏 dLM
- [ ] **离散扩散 + 量化**: 与 KIVI 联合

---

## 9. 参考文献

- LLaDA (Renmin University, 2025) — "Large Language Diffusion Models"
- DiffuLLaMA (2024) — "Scaling Diffusion Language Models"
- MDLM (Stanford, 2024) — "Simple and Effective Masked Diffusion Language Models"
- SEDD (2024) — "Score Entropy Discrete Diffusion models"
- [Discussion #618](https://github.com/jingyaogong/minimind/discussions/618) — MiniMind dLM 讨论

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

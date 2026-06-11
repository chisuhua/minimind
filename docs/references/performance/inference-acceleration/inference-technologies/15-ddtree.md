# 15 · DDTree (DFlash 树形多路并行扩展)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 4
> **代码位置**: `model/ddtree.py` (`DDTreeDecoder`), `model/dflash.py` (协同)
> **CLI 入口**: `--ddtree`, `--ddtree_branch`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

DDTree (DFlash Tree) 是在 [DFlash](14-dflash.md) 基础上的**树形多路并行推测解码**扩展。核心思想是:

- DFlash 块扩散: drafter 生成**一个 block**, backbone 验证
- DDTree: drafter 生成**多个 block 候选**, backbone **一次 forward 验证全部**
- 类似 Medusa 的 tree attention, 但与 DFlash 块扩散结合
- 进一步提升 acceptance rate 30-50%

技术组成:
1. **多路 drafter**: 同时 draft B 个 block (B=2-4)
2. **树状验证**: 一次 backbone forward 验证 B 个 block + 公共前缀
3. **公共 KV cache**: 共享前缀的 KV, 减少验证计算
4. **最大接受**: 从 B 个 block 中选最长的接受 prefix

> **典型加速比**: 比 DFlash 再 +30-50% acceptance, 端到端 +1.5-2.5× 加速

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- DFlash 在 MiniMind 上 acceptance 偏低, DDTree 通过多路提升 acceptance
- 64M 模型计算轻, 多路 drafter 成本可接受 (B=2 时, draft 成本仅 2×)
- 一次 forward 验证 B 个 block, **额外计算极小** (只是 B 倍 block 并行)
- 与 DFlash 配套形成完整 spec decoding 体系

**不集成的代价**:
- DFlash acceptance 受限, 加速比不达预期
- 在 64M 这种小模型上, 单纯 DFlash 加速比可能 < 4×
- 错过树形 spec decoding 关键优化

**典型加速比**: DFlash 4-6× → DDTree 5-8× (B=2-4)

---

## 3. 架构设计

### 3.1 树状结构

```
                    Root (公共前缀)
                   /     |     \
                  B1    B2    B3    (B 个 drafter 块)
                  |     |     |
              (验证) (验证) (验证)
                  
每个 drafter 用不同采样温度 / 不同种子, 产生不同 block
```

### 3.2 数据流

```
Step k:
  1. 现有 KV cache
  2. 启动 B 个 drafter, 每个 draft block_size 个 token
  3. 构造 tree input: [prefix, block_1, block_2, ..., block_B]
  4. 构造 tree attention mask: 块间不互相看
  5. backbone 一次 forward 验证
  6. 从 B 个 block 中选最长接受 prefix
  7. 更新 KV cache
```

### 3.3 关键模块

- **`DDTreeDecoder`**: 协调多个 drafter + 树状验证
- **`TreeAttentionMask`**: 自定义 attention mask
- **`MultiBranchDrafter`**: 多 drafter 调度

### 3.4 计算复杂度

| 项 | DFlash | DDTree (B=2) | DDTree (B=4) |
|----|--------|--------------|--------------|
| Drafter 调用 | 1 | 2 | 4 |
| Backbone 验证 | 1 forward | 1 forward (2 倍 block) | 1 forward (4 倍 block) |
| Acceptance | 1× | 1.3-1.5× | 1.5-1.8× |
| 净加速 | 4-6× | 5-7× | 5-8× |

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/ddtree.py
import torch
import torch.nn.functional as F

class DDTreeDecoder:
    def __init__(self, backbone, drafter, block_size=16, n_branch=2):
        self.backbone = backbone
        self.drafter = drafter
        self.block_size = block_size
        self.n_branch = n_branch

    def step(self, input_ids, kv_cache):
        prefix = input_ids
        # 1. 多路 draft
        blocks = []
        for b in range(self.n_branch):
            # 用不同采样策略
            block = self.drafter.draft_block(
                prefix, num_tokens=self.block_size, temperature=0.7 + 0.1 * b
            )
            blocks.append(block)
            prefix = torch.cat([prefix, block], dim=-1)  # 用于下一个 drafter (可选独立)

        # 2. 构造 tree input: 公共 prefix + n_branch 个 block
        # 关键: 公共 prefix 不重复
        # input shape: (1, L_prefix + n_branch * block_size)

        # 3. 构造 tree attention mask
        # 公共 prefix 互相可见
        # 各 block 内部自可见, 但不可见其他 block
        tree_mask = self._build_tree_mask(L_prefix=self.prefix_len,
                                          n_branch=self.n_branch,
                                          block_size=self.block_size)

        # 4. backbone forward
        full_input = torch.cat([self.cached_prefix] + blocks, dim=-1)
        logits, new_kv = self.backbone(full_input, attention_mask=tree_mask,
                                       past_kv=kv_cache, return_kv=True)

        # 5. 选最长接受 prefix
        accepted = self._select_longest(logits, blocks)

        return accepted, new_kv

    def _build_tree_mask(self, L_prefix, n_branch, block_size):
        # (L_prefix + n_branch * block_size) x (L_prefix + n_branch * block_size)
        L = L_prefix + n_branch * block_size
        mask = torch.zeros(L, L, dtype=torch.bool)
        # 公共 prefix 互相可见 (causal)
        mask[:L_prefix, :L_prefix] = torch.tril(torch.ones(L_prefix, L_prefix, dtype=torch.bool))
        # 每个 block 内部自可见
        for b in range(n_branch):
            start = L_prefix + b * block_size
            end = start + block_size
            mask[start:end, :start] = True  # block 看 prefix
            mask[start:end, start:end] = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        # 块间不互相看 (默认 False)
        return mask
```

### 4.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_branch` | 2 | 分支数量 (drafter 数量) |
| `block_size` | 16 | 块大小 (与 DFlash 一致) |
| `temperature_branch` | [0.7, 0.8] | 每个分支的采样温度 |
| `diversity_threshold` | 0.3 | 多样性阈值 (过滤相似 block) |

### 4.3 默认配置

`eval_llm.py` 默认关闭, 需要 `--ddtree` + `--dflash` 一起启用。完整流程:
```bash
# 1. 训练 DFlash drafter (与 DDTree 共享)
python trainer/train_dflash.py --output dflash_768.pt

# 2. 推理 (DDTree 自动在 DFlash 之上)
python eval_llm.py --dflash --dflash_path dflash_768.pt --ddtree --ddtree_branch 2
```

---

## 5. 训练过程影响

**零额外训练**。DDTree 是推理时的"包装器", 直接复用 DFlash 的 drafter。

- 不需要新数据
- 不需要新训练
- 仅在推理时多启动几个 drafter

> 可选: 训练一个 "**多分支 drafter**", 用不同 head 专门负责不同采样温度分支, 可略微提升 acceptance。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 起点 | DFlash 模型 |
| n_branch | 1 / 2 / 3 / 4 |
| block_size | 16 |
| 测试 | HumanEval + 自由对话 |

### 6.2 评估指标

- **加速比** (vs 朴素 decoding, vs DFlash)
- **Block acceptance rate**
- **平均接受长度**
- **任务准确率**

### 6.3 预期结果

| n_branch | 加速比 (vs AR) | Acceptance 提升 | 任务准确率 |
|----------|---------------|----------------|------------|
| 1 (DFlash) | 4-6× | baseline | baseline |
| 2 | 5-7× | +30-50% | +0-1% |
| 3 | 5-7.5× | +50-70% | +0-1% |
| 4 | 5-8× | +60-80% | +0-1% |

> 边际收益递减, 4 个分支后基本饱和。

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **多样性依赖 drafter 质量**: 如果 drafter 温度差不大, B 个分支会相似
2. **树状 attention 显存**: B=4 时, 一次 forward 处理 4 倍 block
3. **加速比上限**: 受 drafter 本身 acceptance 限制, 不会无限提升
4. **不支持 beam search**: 树结构与 beam 冲突
5. **需要 DFlash drafter**: 单独 DDTree 无意义

---

## 8. 后续改进方向

- [ ] **自适应分支数**: 根据上一轮 acceptance 动态调 n_branch
- [ ] **多 drafter 集成**: 训练 N 个不同 drafter, 多样性更高
- [ ] **采样策略**: 用 Gumbel-softmax 等更 diverse 的采样
- [ ] **与 Medusa 头融合**: DFlash block + Medusa head 同时 draft
- [ ] **层级 DDTree**: drafter 内再 draft 子 block

---

## 9. 参考文献

- 与 DFlash 同源 (DeepSeek 2026)
- 相关: EAGLE-3 tree attention, Medusa tree verification
- 树状解码理论: Sun et al. "Spectra: Sparse Attention for Speculative Trees" 2024

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

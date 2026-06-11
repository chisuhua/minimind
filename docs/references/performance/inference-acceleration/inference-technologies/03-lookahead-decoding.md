# 03 · Lookahead Decoding (前瞻解码)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 1
> **代码位置**: `model/lookahead_decoding.py` (`LookaheadDecoding` 类)
> **CLI 入口**: `--lookahead_decoding`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

Lookahead Decoding (ICML 2024, Fu et al., 阿里达摩院) 是一种**零训练成本**的并行解码技术, 核心思想是:

- 不需要额外的 drafter 模型
- 在生成第 t 个 token 时, 同时**生成第 t+1, t+2, ..., t+W 个 token 的多个候选** (Jacobi 迭代)
- 用 LLM 自身的 forward 一次性验证这些候选
- 接受连续匹配的前缀, 实现 **W 倍**的并行

具体流程:
1. 维护一个 `(N-1) × W` 的 Jacobi 窗口 (N-1 行, 每行 W 个候选)
2. 每步: 用一次 batched forward 计算窗口内所有位置的概率
3. 从中找出最长的与 LLM 单步采样一致的 prefix
4. 接受该 prefix, 拒绝的部分作为下次 Jacobi 迭代的起点

> **典型加速比**: 1.5-1.8× (单 GPU), 4×+ (多 GPU 强扩展下)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- MiniMind 是 64M Dense, 单 token 计算开销极小
- 因此**显存带宽**和**launch overhead** 是主要瓶颈, 而不是算力
- Lookahead Decoding 通过一次 forward 验证多个候选, 直接攻击这两个瓶颈
- **零训练成本**意味着立即可上, 几乎无风险

**不集成的代价**:
- 在 batch=1, 短 prompt 场景下, 64M 模型生成速度主要受 launch overhead 限制
- 朴素 decoding 实际上 GPU 利用率 < 30%
- 浪费了 GPU 的并行能力

**典型加速比**: 单 batch 1.5-1.8×; batch>=4 时 1.2-1.4× (受显存带宽限制)。

---

## 3. 架构设计

### 3.1 核心数据结构

```
Jacobi Window: (N-1) × W 矩阵
   行 = Jacobi 轨迹 (每行一个候选序列)
   列 = 时间步 (列 0 = 当前, 列 1 = 下一步, ...)
```

### 3.2 数据流

```
┌─────────────────────────────────────────────┐
│ Step k:                                      │
│                                               │
│  1. 构造 batch: (N-1) × W 个序列            │
│  2. 一次 forward → 拿到 (N-1) × W × V logits│
│  3. 对每行的 col 0: 用 LLM 真实采样得到      │
│     真实 token, 与该行 col 0 的 Jacobi 候选 │
│     对比                                      │
│  4. 找最长的连续匹配前缀 n                  │
│  5. 接受前 n 个 token, 拒绝部分保留为下次    │
│     Jacobi 起点                              │
│  6. 移动窗口: 每个 Jacobi 行右移 1, 末尾   │
│     用最新接受的 token 重新开始              │
└─────────────────────────────────────────────┘
```

### 3.3 关键模块

- **`LookaheadDecoding`**: 主类
  - 维护 Jacobi window tensor
  - 维护每个 trajectory 的当前位置
  - 提供 `step()` 方法返回 (accepted_tokens, new_trajectories)
- **`LookaheadAttentionMask`**: 自定义 attention mask, 让每个 trajectory 只看自己的历史

### 3.4 计算复杂度

- 朴素 decoding: N 次 forward, 每次 1 个 token
- Lookahead: N/W 次 forward, 每次 (N-1)×W 个 token
- **理论加速**: W 倍 (受 acceptance rate 制约)

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/lookahead_decoding.py
class LookaheadDecoding:
    def __init__(self, model, n_trajectory: int = 5, window: int = 8,
                 temperature: float = 1.0):
        self.model = model
        self.n_traj = n_trajectory
        self.W = window
        self.jacobi_buf = None  # (n_traj-1, W)
        self.positions = None

    def step(self, input_ids, position_ids, past_kv):
        B = input_ids.shape[0]
        # 1. 构造 Jacobi batch
        if self.jacobi_buf is None:
            self._init_jacobi(input_ids)

        # 2. 拼接: [real input, jacobi trajs]
        # 一次性 forward
        full_input = torch.cat([input_ids, self.jacobi_buf.flatten(0, 1).unsqueeze(0)],
                               dim=1)
        logits, _ = self.model(full_input, position_ids=..., past_kv=...)

        # 3. 提取真实位置和 Jacobi 位置 logits
        real_logits = logits[:, :B, :]  # 真实位置
        jacobi_logits = logits[:, B:, :].view(1, self.n_traj-1, self.W, -1)

        # 4. 真实采样
        sampled = self._sample(real_logits[:, -1:, :])

        # 5. 比较每行 col 0 与 sampled
        match_len = (jacobi_logits[:, :, 0, :].argmax(-1) == sampled).sum(...)

        # 6. 接受最长 prefix
        accepted = self._accept_prefix(jacobi_logits, sampled)

        return accepted, ...
```

### 4.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_trajectory` | 5 | Jacobi trajectory 数量 (N-1) |
| `window` | 8 | 每个 trajectory 的候选长度 (W) |
| `temperature` | 1.0 | 采样温度 |

### 4.3 默认配置

`eval_llm.py` 默认关闭。建议:
- 代码生成 / 数学题: `n_traj=8, W=16` (高确定性, 加速比高)
- 创意写作: `n_traj=4, W=4` (低确定性, 大窗口容易 reject)
- 长 prompt 一次性输入: 推荐使用

---

## 5. 训练过程影响

**零影响**。Lookahead Decoding 是纯推理时技术, 不涉及训练目标、损失函数、数据格式。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | `minimind-3` (64M), full_sft |
| 测试集 | HumanEval-X mini (代码) / XSum mini (摘要) |
| Prompt 长度 | 256 / 1024 |
| 生成 token 数 | 256 |

### 6.2 评估指标

- **加速比**: tokens/s 相对朴素 decoding
- **Acceptance rate**: Jacobi 候选被接受的比例
- **生成质量**: 任务准确率 / ROUGE

### 6.3 预期结果

| 任务 | 加速比 | Acceptance |
|------|--------|------------|
| HumanEval | 1.5-1.8× | 60-75% |
| XSum | 1.2-1.4× | 40-55% |
| 自由对话 | 1.3-1.5× | 50-65% |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **显存峰值高**: Jacobi window 占用额外 (n_traj-1) × W × hidden_size
2. **小模型 acceptance 偏低**: 64M 模型生成随机性高, acceptance 率弱于 1B+ 模型
3. **不支持 prefix sharing**: 与 vLLM/SGLang prefix cache 不兼容
4. **不支持 beam search**: 假设是贪心 / 采样
5. **与 spec decoding 互斥**: 不能与 Medusa / DFlash 同时启用

---

## 8. 后续改进方向

- [ ] **动态窗口大小**: 根据 acceptance rate 自适应调整 W
- [ ] **多 Jacobi 起点**: 不仅从 accepted prefix, 还从分支点继续
- [ ] **Lookahead + Medusa 联合**: 用 Medusa head 提供更好的 draft
- [ ] **量化兼容**: 与 KIVI 2-bit KV 缓存结合
- [ ] **Flash Attention 适配**: 当前 SDPA, 可换 flash_attn 进一步加速

---

## 9. 参考文献

- Fu et al., "Lookahead Decoding: Breaking the Sequential Dependency of LLM Inference", ICML 2024
- arXiv: 2402.02057
- [GitHub: hao-ai-lab/lookahead-decoding](https://github.com/hao-ai-lab/lookahead-decoding)
- 相关: REST (ICML 2023, 早期并行解码)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

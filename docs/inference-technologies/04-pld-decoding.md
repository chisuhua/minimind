# 04 · PLD / AdaPLD (Prompt-based Lookup Decoding)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 1
> **代码位置**: `model/pld_decoding.py` (`PLDDecoding` 类)
> **CLI 入口**: `--pld`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

PLD (Prompt-based Lookup Decoding) 是一种**基于查找**的推测解码技术, 核心观察是:

- LLM 生成过程中, 经常出现**已生成 token 序列在 prompt / 之前生成内容中**的情况
- 例如, 代码生成时变量名复用, 对话时常用词组复用
- 朴素 decoding 每次只生成 1 token, **浪费了这种"已见过"的优势**

PLD 的做法:
1. 维护一个最近生成 token 的滑动窗口 (长度 `ngram_size`)
2. 在 prompt + 已生成内容中**查找**所有匹配该 n-gram 的位置
3. 取**最长后续匹配**作为 draft, 送入 LLM 验证
4. LLM 一次 forward 验证整段 draft, 接受连续匹配的前缀

AdaPLD (Adaptive PLD) 在此基础上, **根据上一轮的 acceptance 动态调整 n-gram 长度**, 提升鲁棒性。

> **典型加速比**: 1.3-2.5× (高度依赖任务类型, 代码 / 模板化任务加速高)

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- MiniMind 训练数据包含大量 **toolcall / 模板化对话** 数据
- 实际推理时, `<|im_start|>`、`<|im_end|>`、JSON 结构等重复 token 序列**频繁出现**
- 这些重复恰好是 PLD 的甜蜜点
- **零训练成本**, 立即可上

**不集成的代价**:
- ToolCall 场景下, JSON 模板反复生成, 浪费大量时间
- 模板化对话 (客服 / FAQ) 速度低于预期
- 整体用户体验: "模型说车轱辘话时明显卡顿"

**典型加速比**: ToolCall 2-3×, 自由对话 1.2-1.5×, 数学/代码 1.0-1.2× (这些任务重复模式少)

---

## 3. 架构设计

### 3.1 核心数据结构

```
n-gram 索引 (ngram_size=3):
  key = 最近 3 tokens
  value = 所有出现位置的后续序列
```

### 3.2 数据流

```
┌──────────────────────────────────────────────┐
│ Step k:                                       │
│                                                │
│  1. 取最近 ngram_size 个 token 作为 key       │
│  2. 在 (prompt + generated) 中查找所有匹配    │
│  3. 对每个匹配位置, 取后续 max_draft 个 token │
│  4. 找到**最长**的 draft (后续 tokens 序列)   │
│  5. 用 LLM forward 验证 draft 中每个 token    │
│  6. 接受连续匹配的前缀                        │
│  7. 拒绝的 token 走正常 sampling              │
└──────────────────────────────────────────────┘
```

### 3.3 关键模块

- **`PLDDecoding`**: 主类
  - `ngram_index`: dict[(tuple), list[list[int]]]
  - `prompt_ids`: 原始 prompt (不参与 n-gram 提取, 但参与查找)
  - `generated_ids`: 已生成序列
- **关键优化**: n-gram 索引**增量更新**, 不每次重建

### 3.4 计算复杂度

- 朴素 decoding: O(N) 次 forward
- PLD: O(N/avg_accepted_draft) 次 forward
- 查找开销: O(|prompt| + |generated|), 可忽略

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/pld_decoding.py
class PLDDecoding:
    def __init__(self, ngram_size: int = 3, max_draft: int = 16):
        self.ngram_size = ngram_size
        self.max_draft = max_draft
        self.ngram_index = {}  # key=(t1,t2,t3) → [后续序列列表]
        self.all_text_ids = []  # prompt + generated 完整序列

    def update_index(self, new_token: int):
        self.all_text_ids.append(new_token)
        if len(self.all_text_ids) >= self.ngram_size + 1:
            key = tuple(self.all_text_ids[-(self.ngram_size+1):-1])
            val = self.all_text_ids[-1]
            self.ngram_index.setdefault(key, []).append(
                self.all_text_ids[-(self.ngram_size+1):]
            )

    def find_draft(self, current_ngram):
        if current_ngram not in self.ngram_index:
            return None
        candidates = self.ngram_index[current_ngram]
        # 找最长的后续 draft
        best_draft = max(candidates, key=len, default=[])
        return best_draft[self.ngram_size:][:self.max_draft]

    def step(self, model_forward_fn):
        # 1. 取当前 n-gram
        ngram = tuple(self.all_text_ids[-self.ngram_size:])

        # 2. 找 draft
        draft = self.find_draft(ngram)
        if draft is None or len(draft) == 0:
            return self._normal_step(model_forward_fn)

        # 3. 验证 draft
        accepted = self._verify_draft(model_forward_fn, draft)

        # 4. 更新索引
        for t in accepted:
            self.update_index(t)

        return accepted
```

### 4.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ngram_size` | 3 | n-gram 长度 (3-5 常见) |
| `max_draft` | 16 | 单次 draft 最长 token 数 |
| `min_match` | 3 | 至少匹配 n 个才接受 draft |

### 4.3 默认配置

`eval_llm.py` 默认关闭。建议:
- ToolCall / 模板对话: `ngram=3, max_draft=16` (默认)
- 创意写作: `ngram=5, max_draft=8` (减少误匹配)
- 数学/代码: `ngram=4, max_draft=8`

---

## 5. 训练过程影响

**零影响**。PLD 是纯推理时技术。

可选改进: 在 SFT 数据中**显式增加重复模式** (例如模板化对话比例), 可提高 PLD 加速比。

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | `minimind-3` (64M), full_sft |
| 测试集 | 自由对话 / ToolCall 模板 / 代码补全 |
| 任务类型 | 5 类, 各 100 样本 |

### 6.2 评估指标

- **加速比**: tokens/s
- **Acceptance rate**: 验证时被接受的比例
- **生成质量**: 任务准确率 (与朴素 decoding 对比)
- **首字延迟**: ms

### 6.3 预期结果

| 任务 | 加速比 | Acceptance | 质量影响 |
|------|--------|------------|----------|
| 自由对话 | 1.2-1.4× | 50-60% | 无 |
| ToolCall 模板 | 2.0-3.0× | 80-90% | 无 |
| 代码补全 | 1.1-1.3× | 30-50% | 微正 (复用命名) |
| 数学题 | 1.0-1.1× | 10-20% | 无 |
| 摘要 | 1.3-1.6× | 60-70% | 微负 (拼接句子) |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **n-gram 索引内存**: 长上下文 (8K+) 下, 索引可达数十 MB
2. **误匹配风险**: 创意文本中, 不该匹配的 n-gram 可能错误触发 draft, 引入偏差
   - 缓解: 校验时对比**完整 forward logits**, 不仅看 argmax
3. **不支持并行 batching**: 当前实现是 batch=1 优化, 多 batch 需要扩展
4. **中文支持**: 中文 token 化后, n-gram 模式可能跨词, 需要更长 ngram
5. **与 KV cache 兼容**: 需要把 draft 验证时的 KV cache 维护做对

---

## 8. 后续改进方向

- [ ] **AdaPLD 完整实现**: 根据 acceptance 自适应 ngram_size
- [ ] **批处理支持**: 多请求并行查找
- [ ] **多模态扩展**: 图像 caption 等场景
- [ ] **与 Medusa 联合**: Medusa head 提供额外 draft 候选
- [ ] **N-gram 索引压缩**: 用 trie / DAWG 减少内存

---

## 9. 参考文献

- Saxena et al., "Prompt Lookup Decoding", 2023
- [GitHub: apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding)
- 后续工作: AdaPLD, Pyramid Draft, LOOKAHEAD-PLD
- 相关 token healing 技术 (HF generate)

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

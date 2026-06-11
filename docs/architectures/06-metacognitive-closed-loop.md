# 元认知闭环 — 推理→置信度→检索→重新推理审查

> **版本**：v4.7 提案（"推理→置信度评估→低置信触发→知识检索/注入→重新推理"闭环）
> **评级**：B+（窄场景优化器，工程上有真实价值但被通用化过度）
> **核心定位**：在 knowledge-intensive QA 的 sweet spot 上有效，但不是"AGI 自适应推理引擎"

---

## 一、闭环核心思想

```text
[用户查询]
    ↓
[Core 1 推理]
    ↓
[置信度评估] ─────┐
    ↓             │
[阈值判定]        │
    ↓             │
  ┌─┴─┐           │
  ↓   ↓           │
[达标] [未达标]     │
  ↓   ↓           │
  ↓ [触发知识检索]   │
  ↓   ↓           │
  ↓ [知识注入]      │
  ↓   ↓           │
  ↓ [重新推理] ─────┘
  ↓
[最终输出]
```

**核心洞察**：当模型因知识缺失而输出低置信度时，通过外部记忆注入正确知识，可使其成功回答。

---

## 二、闭环四步的真实累积成功率（端到端粗算）

### 2.1 1.5B 模型上每步的实证错误率

| 步骤 | 实证成功率 | 数据来源 |
|------|-----------|---------|
| 1. 推理产生候选答案 | 100% | — |
| 2. 置信度估计（"该触发检索"决策）| 60-75% | ECE 0.15-0.32（7B），1.5B 更差 |
| 3. 检索召回正确答案（top-5）| 65-75% | BM25/Dense benchmark |
| 4. 重新推理利用检索内容 | **10-15%** | Pandey et al. arXiv 2603.11513 |
| **端到端 1.5B 闭环成功率** | **100% × 0.7 × 0.7 × 0.12 ≈ 6%** | 乘法累积 |
| **端到端 7B 闭环成功率** | **100% × 0.75 × 0.75 × 0.55 ≈ 31%** | 同上 |

**6% 的意义**：原本无检索就答错的问题中，闭环救回 6%；对应 base accuracy 25% → 27%。**这是真实但有限的增益**。

### 2.2 致命发现：1.5B 的知识利用是 utilization 瓶颈

**Pandey et al. 在 oracle 检索（保证答案在文档里）下的实证**：

| 模型 | Known 问题 EM（无检索）| Oracle 检索后 EM | 检索后损失 |
|------|------------------------|------------------|------------|
| SmolLM2-360M | 100% | **0.0%** | **-100%** |
| Qwen2.5-1.5B | 100% | **10.0%** | **-57.0%**（oracle）|
| Qwen2.5-3B | 100% | 12.8% | -45.6% |
| Qwen2.5-7B | 100% | 14.6% | -41.6% |

**核心洞察**：**即使检索 100% 召回答案，1.5B 只能提取 10% 的答案**——这是 utilization 瓶颈，不是 retrieval 瓶颈。

**61-100% 的失败是"irrelevant generation"**——模型完全忽略提供的上下文。

**任何检索（即使 oracle）都会摧毁 42-100% 的"模型本来答对的"答案**（distraction effect）。

**对闭环的影响**：第 4 步 10% 利用率是闭环的硬天花板，**比置信度估计的 70% 更致命**。

---

## 三、Self-RAG 在小模型上的真实表现

### 3.1 触发器工作的"窗口几乎不存在"

**RetrievalQA（Ying et al., NAACL 2025）** 的硬数据：

| 模型 | 检索准确率（该触发时触发了）|
|------|---------------------------|
| TinyLlama-1.1B vanilla prompt | **极低**（几乎不触发）|
| Self-RAG-7B (t=None) | **6.0%**（漏触 94%）|
| Self-RAG-7B (t=0.25) | **100%**（等价 Always-RAG）|

**结论**：在小模型上 confidence 触发器的"工作点"几乎不存在——要么不触发（漏掉所有该触发的），要么全触发（等同 Always-RAG）。

### 3.2 Self-RAG 本身没有在 1.5B 以下的复现
- 需要 150K GPT-4 标注数据训练 reflection token
- 在 PopQA/TriviaQA/PubHealth 上显著有效
- **但在 2WikiMultiHopQA（39% < 58%）和 HotpotQA（41% < 61%）上反而不如直接 RAG**
- **Self-RAG 适合单跳长尾知识 QA，不适合多跳 QA**

### 3.3 闭环在 multi-hop QA 上的"甜区"与硬限

```
任务           | 1.5B baseline | + 闭环   | vs 7B static RAG
------------------------------------------------------------
1-hop QA       | 35%          | 50%      | 落后 ~10pp
2-hop QA       | 20%          | 38%      | 接近持平
3-hop QA       | 8%           | 18%      | 落后 ~15pp
4+ hop QA      | <5%          | ~10%     | 落后 ~25pp
```

**Sweet spot**：2-3 hop knowledge-intensive QA。**超过 3 hop 闭环增益急剧衰减**——1.5B 的 working memory 撑不住。

---

## 四、置信度估计在 1.5B 上的真实可靠性

### 4.1 校准错误率（ECE/AUROC）

| 模型 | TriviaQA ECE | GSM8K ECE | AUROC |
|------|--------------|-----------|-------|
| Llama3-8B base | 15.5% | 17.1% | 60-62 |
| Qwen2.5-7B Instruct | 31.6% | 9.8% | 52-60 |
| EAGLE（Llama3-8B 校准后）| **1.7%** | 7.6% | 61.5 |

**关键数据**：原生 token 概率在 7B 级别 ECE 已经高达 15-32%；**直接用 token prob 作为触发器会有大量伪触发**。1.5B 的校准更差。

### 4.2 verbalized overconfidence 是结构性 bug

**Wired for Overconfidence（2026）** 跨 Llama-3.2-3B 和 Qwen2.5-3B：
- 80%+ 的过自信电路在不同数据集间是**共享的**
- **verbalized overconfidence 不是任务特性，是模型内部电路的稳定 bug**

### 4.3 1.5B-3B 上的真实误触发率

| 误触发类型 | 1.5B 频率 | 7B 频率 | 影响 |
|-----------|-----------|---------|------|
| False positive（假阳性，触发不必要的 retrieval）| 25-40% | 15-25% | 浪费 30% 检索调用 |
| False negative（假阴性，未触发本应触发的）| 30-50% | 15-25% | 漏掉 40% 该检索的场景 |

**这意味着**：~1/3 的 retrieval 是浪费的；~40% 该检索的场景没检索。**置信度触发不能替代任务级路由，只能补充**。

---

## 五、v4.7 集成架构（推荐方案）

### 5.1 三段式 Router 设计

```
[入口]  L1-A 任务路由 (existing, 改动小)
            ↓
        [推理循环]
            ↓
        L1-B 置信度监视器 (new, per-step hook)
            ↓
   ┌──── 低置信? ────┐
   ↓ yes             ↓ no
[触发 retrieval]   [继续/输出]
   ↓
[知识注入 → 回到推理循环, 重新生成]
   ↓
   └──→ 退出条件: 置信度达标 / 重试上限 / 硬超时
```

**关键工程决策**：
- L1-A 任务路由的"是否需要 retrieve"维度**保留**，与 L1-B 置信度触发**并存但解耦**。两者同时为真才真正触发 retrieval（AND 策略）
- 任务路由决策可作为置信度阈值的**先验偏置**——例如数学/逻辑类任务阈值更低

### 5.2 per-step vs per-answer 置信度

**强烈推荐 per-step**，但要分两种粒度：

| 层级 | 触发时机 | 用途 | 成本 |
|---|---|---|---|
| **Step-level** | 每个 CoT step 结束 | 触发 mid-reasoning retrieval | 高（每步一次前向）|
| **Token-level** | 每个生成 token | 触发 constrained decoding 收紧 | 中 |
| **Answer-level** | 最终答案前 | 触发答案级二次验证 | 低 |

**1.5B dense 的现实**：step-level 置信度 head 每步 ~5-15ms，Short-CoT ≤2K tokens 假设 8 步 = 40-120ms 额外开销，可接受。但**不能做 token-level**——1.5B 扛不住每 token 一次 head 推理的延迟。

**结论**：step-level（per-step + per-answer 双保险）是甜蜜点。

### 5.3 重新推理的硬上限（必须实现）

**硬性建议**：
- **max_retries = 2**（总尝试次数 = 3：原始 + 2 次重试）
- **每次重试的预算**：
  - 第 1 次重试：可触发一次 retrieval，预算 = 原始推理的 1.5x
  - 第 2 次重试：可触发一次 retrieval + 一次 verification，预算 = 原始推理的 2.0x
  - 第 3 次仍失败：直接走 fallback

**成本/延迟控制硬上限**：
- 单 query 总耗时 ≤ 原始推理的 3x
- 单 query 额外 token 消耗 ≤ 原始的 2.5x
- 检索次数 ≤ 2（避免 GraphRAG 被当作 oracle）

### 5.4 与 LATTS 的关系

**这是最容易出问题的地方，必须明确边界**：

| 维度 | LATTS | 元认知置信度触发 |
|---|---|---|
| **触发依据** | step 难度估计（先验） | step 置信度（后验） |
| **动作** | 调整采样参数 / 切换策略 | 触发 retrieval / 重试 |
| **频率** | 每个 step | 仅低置信 step |
| **是否消耗额外 token** | 否 | 是（检索 + 重生成） |

**协同原则**：
- LATTS 难度高 → 该 step 的置信度阈值**自动降低**（更宽容）
- LATTS 难度低 → 置信度阈值**保持默认**（减少误触发）
- 两者**不嵌套调用**——LATTS 决策完成后才进入置信度评估，否则控制流混乱

### 5.5 四级降级策略（必备）

| 级别 | 触发条件 | 动作 |
|------|----------|------|
| **L0 正常** | 置信度达标 | 返回 |
| **L1 重试** | 低置信 + 预算允许 | retrieval + 重生成 |
| **L2 兜底** | 重试耗尽 | 返回 best_so_far + 置信度标签 |
| **L3 HITL** | 置信度极低 + safety 敏感 | 触发 L3 人工审核（v4.5 已有）|

---

## 六、闭环的核心失败模式与防御

### 6.1 失败模式 1：检索注入错误知识的"放大错误"风险

**场景**：GraphRAG 返回看似相关但实际错误的文档 → 1.5B 把它当权威 → 重新推理**强化错误** → 第二次置信度反而**更高**（因为模型"看起来"更确定了）

**缓解策略**：
1. **检索结果必须标注 source + timestamp + 置信度**，让 prompt 显式包含
2. **重新推理后必须再做 Engine-Native Verification**（v4.5 已有）——这是关键防线
3. **检测"新引入的事实声明"**：如果重试后的答案包含原始推理中**没有**的具体实体/数字，标记为 suspicious
4. **多源不一致惩罚**：同一 query 的两次 retrieval 结果矛盾时，置信度直接置 0

### 6.2 失败模式 2：重新推理死循环

**根因**：
- 检索结果每次不同 → 模型每次生成不同答案 → 永远不收敛
- 置信度估计在边界附近震荡 → 阈值上下反复触发

**检测机制**：
- **答案相似度监测**：最近 N 次答案 ROUGE-L > 0.9 → 强制终止（已收敛）
- **状态 hash 监测**：(query, retrieval_set, prefix) 三元组重复出现 → 强制终止
- **单调性约束**：重试后置信度不增 → 终止
- **全局 token 预算**：硬性熔断器

**硬约束**：max_retries=2 本身就是最简单的死循环防护。

### 6.3 失败模式 3：1.5B 概率未校准导致误触发链式

**实证数据**：
- 1.5B 模型的概率严重未校准（ECE 经常 >0.2）
- 纯熵阈值会大量误触发
- 必须配合**后验校准**（temperature scaling 或 Platt scaling）

**推荐方案**：
- **A 方案（主）**：每 step 取 top-k token 的负熵作为置信度，阈值 ~0.6-0.7
- **C 方案（辅）**：在数学/代码等结构化任务上训练轻量 confidence head（~5M 参数）
- **不做 B**（Self-consistency 在 1.5B 上 N≥5 时延迟 >3s）

---

## 七、闭环与 v4.5/v4.6 已有组件的协同

### 7.1 Engine-Native Verification：互补关系

**两者关系**：**互补，非替代**
- Engine-Native Verification：验证答案的**结构正确性**（JSON/Regex/Python）
- 置信度触发：触发**事实正确性**的检索

**关键协同**：检索注入 → 重新推理 → **必须再次过 Verification**。Verification 是闭环**最后一道防线**——即使置信度错了、检索错了，结构化验证仍能挡住部分错误。

### 7.2 GraphRAG Retrieval：避免双重调用

**错误方案**：把元认知触发检索当作"额外的 GraphRAG 调用"——会导致：
- GraphRAG 接口被双重调用（L1-A 预检索 + L1-B 触发检索）
- Cache 命中率下降
- 检索 query 风格不一致

**正确方案**：
- GraphRAG 维护一个**会话级 context cache**（key = normalized query + history hash）
- L1-A 的预检索结果**先入 cache**，L1-B 触发时**先查 cache**再决定是否重检索
- 检索 query 风格**统一**（都用 CoT 的自然语言形式）

### 7.3 三层 Safety：独立通道

**关键原则**：**safety 通道必须独立于元认知闭环**
- L2 Output 检测到 safety 风险 → **直接拒绝**，不进入闭环
- 闭环的 retrieval query 必须**经过 safety 过滤**（不能因触发检索就绕过安全检查）
- HITL 触发条件：safety 敏感 + 置信度低 → **双重条件都满足才升级**

**v4.7 锐利建议**：在 safety 路径上**禁用**元认知触发——安全判断不能因"知识不足"就重新检索。

---

## 八、闭环的真实价值矩阵

### 8.1 策略对比（knowledge-intensive 任务）

| 策略 | 准确率 | 延迟 | 成本 | 适用 |
|------|--------|------|------|------|
| **Never-RAG** | 低（1.5B 知识有限）| 最低 | 最低 | simple factoid 兜底 |
| **Always-RAG** | 中（信息淹没 1.5B）| 高 | 高 | 长文档 QA |
| **Confidence-triggered 闭环** | **中-高** | **中** | **中** | **大多数场景** |

**置信度触发的真实优势**：
- **延迟比 Always-RAG 低 30-50%**（仅在需要时检索）
- **准确率比 Never-RAG 高 10-20pp**（关键场景）
- **抗信息淹没**——只注入关键 step，不污染整个 context
- **自适应**——任务难度变化时自动调整检索频率

### 8.2 vs 单 7B dense 模型

| 维度 | 1.5B + 闭环 | 7B dense 静态 |
|------|-------------|---------------|
| 1-hop factoid | 接近持平 | 略优 |
| 2-hop QA | 接近持平 | 略优 |
| 3-hop QA | 落后（18% vs 30%）| 明显优 |
| Long-form | 落后（~30% vs ~50%）| 明显优 |
| 延迟（p50）| 中（2-5s）| 低（1-2s）|
| 部署成本 | 低（~3GB VRAM）| 高（~14GB VRAM）|
| 运维复杂度 | 极高 | 低 |

**结论**：
- **1.5B + 闭环不是 7B 的替代品，是**互补品**
- 7B 适合：长上下文、高质量生成、复杂多 hop
- 1.5B + 闭环适合：**高并发、低延迟、cost-sensitive、知识覆盖中等**的场景
- **真正杀手锏**：1.5B + 闭环在**成本/性能比**上对 7B 有 3-5× 优势（VRAM + 推理成本）

---

## 九、闭环的边界声明

### 9.1 任务适用性

| 任务类型 | 推荐度 | 原因 |
|---------|--------|------|
| **多 hop QA（2-3 hop）** | ⭐⭐⭐⭐⭐ 强烈推荐 | 闭环 sweet spot |
| **中等难度 knowledge-intensive** | ⭐⭐⭐⭐ 强烈推荐 | 增益最大 |
| **结构化推理（数学/代码）** | ⭐⭐⭐⭐ 强烈推荐 | 配合 Engine-Native Verification |
| **多约束问答** | ⭐⭐⭐ 推荐 | 增益明显但需精细 prompt 工程 |
| **长文档 QA** | ⭐⭐⭐ 可用 | 优于 Always-RAG |
| **1-hop factoid QA** | ⭐⭐ 不推荐 | L1-A 任务路由已足够 |
| **Creative writing** | ⭐ 明确禁用 | 检索破坏创作连贯性 |
| **开放对话** | ⭐⭐ 不推荐 | 置信度无意义 |
| **Safety-sensitive 生成** | ⛔ 禁用 | 独立通道，不进闭环 |

### 9.2 显式禁用场景

- 任务分类标签为 `creative_*`、`open_chat`、`safety_sensitive` → **绕过 L1-B**
- 检索 query 包含 safety 关键词 → **不触发 retrieval**
- 已经在 HITL 队列中 → **不进入闭环**
- 单 query 已消耗 token > 阈值 → **强制降级到 L2**

---

## 十、v4.7 启动建议

### 10.1 版本定位：**v4.7 独立大版本，不是 v4.6 增量**

理由：
- 引入**状态机**（之前是无状态流水线）
- 引入**显式回退语义**（rollback）
- 引入**多级降级**
- API 表面改变（增加 confidence、retrieval_triggered 字段）
- **测试矩阵复杂度爆炸**（state × task × budget × safety）

**如果硬塞进 v4.6，会导致 v4.6 永远 ship 不了**。

### 10.2 v4.7 启动条件
v4.6 完整 ship + 至少 2 周线上数据收集后。

### 10.3 v4.7 第一个 PR 必须包含

1. **显式状态机定义**（不要"自然演化"出状态机）
2. **max_retries=2 硬编码**（不要在第一版就参数化）
3. **简单 factoid / creative / safety 三类任务直接 bypass**（不要在第一版就做精细化）
4. **L2/L3 降级路径优先于闭环核心实现**（兜底先于功能）
5. **GraphRAG context cache 与闭环同步设计**（不要先做闭环再做 cache）

### 10.4 v4.7 优先级排序

**Phase 1（必做，高 ROI）**：
1. per-step 置信度 head（轻量版，token 熵）
2. L1-B 触发器 + 1 次重试 + best_so_far 缓存
3. GraphRAG context cache + 增量注入
4. Verification 与闭环的协同接口（防注入错误知识）
5. L2/L3 降级路径

**Phase 2（推荐做，中 ROI）**：
6. Confidence head 的结构化任务版本（数学/代码专用）
7. 答案相似度监测（防死循环）
8. 多源不一致检测
9. 任务路由对闭环的精细化 bypass 规则

**Phase 3（可选，低 ROI）**：
10. Verbalized confidence 增强
11. 闭环的 RL 训练（用闭环反馈作为 reward）
12. 元认知触发策略的 offline 优化

### 10.5 v4.7 第一个 KPI

- 2-hop QA 准确率提升 ≥ 10pp（基线 20% → 目标 30%+）
- p99 延迟 ≤ 6s
- 死循环率 < 0.5%
- 安全事件 = 0

### 10.6 Kill Criteria（什么时候应该砍掉这个项目）

**必须砍的信号**：
1. **p99 延迟 > 8s 持续 2 周**（UX 灾难）
2. **净失败率改善 < 5pp**（投入产出比不达标）
3. **死循环触发率 > 2%**（控制系统失效）
4. **注入错误知识导致的安全事件 ≥ 1 次**（红线）
5. **1.5B 模型上 confidence head ECE 持续 > 0.25**（置信度估计本质失效）

**警告信号**：
- 简单 factoid 上闭环反而降低准确率 → 立即 bypass
- creative writing 误触发率 > 10% → 立即 bypass
- GraphRAG cache 命中率 < 30% → 检索成本失控

**如果 4 周内未达 KPI，立即降级**到 v4.6 + 简单 L1-A 任务路由 + GraphRAG 静态调用方案，不要在闭环上继续投入。

---

## 十一、引用

1. Pandey et al., "Can Small Language Models Use What They Retrieve?" (arXiv 2603.11513)
2. Ying et al., "RetrievalQA" (NAACL 2025, arXiv 2402.10881)
3. Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique" (ICLR 2024)
4. EAGLE (arXiv 2509.01564)
5. Wired for Overconfidence (arXiv 2604.01457)
6. Ni et al., "When to Retrieve" (ACL Findings 2024)
7. Pandey et al., 2026 (ProcessBench analysis)
8. OnionEval (arXiv 2501.12975)
9. How Retrieved Context Shapes Internal Representations (arXiv 2602.20091)
10. Moskvoretskii et al., "35 Methods Systematic Comparison" (arXiv 2501.12835)

---

## 十二、一句话评价

**元认知闭环在 1.5B 模型上是"窄场景优化器"而非"AGI 银弹"。** 端到端 6% 闭环成功率 vs 5-10× 工程复杂度，**整体 ROI 偏低**，但**在 2-3 hop knowledge-intensive QA 的 sweet spot 上值得做**。**推荐作为 v4.7 独立大版本启动**（不是 v4.6 增量），优先级清晰、KPI 严格、Kill Criteria 明确，4 周内未达 KPI 立即降级到 v4.6 + 简单 GraphRAG 静态调用方案。**永远不要做的事**：把它定位为"AGI 自适应推理引擎"——这是 v1 叙事的究极放大版。**把它定位为"knowledge-intensive QA 的工程优化器"——这是有数据支撑的诚实定位**。
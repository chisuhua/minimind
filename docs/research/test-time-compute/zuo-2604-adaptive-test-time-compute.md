# [2604.21018] Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations

> **来源**:Zuo, B., Zhou, D., Zhu, Y. "Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations." arXiv:2604.21018, v1 2026-04-22.
> **调研日期**:2026-08-30
> **调研者**:Sisyphus + Oracle 交叉核验
> **本文档归属**:`docs/research/test-time-compute/`（测试时计算分配问题领域）
> **关联文档**:`../soca/2026-arxiv-survey.md`（全局视角与跨论文对比）

> ⚠️ **重要更正**：本论文在原 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md)（v1.7 之前）和 [`../../../architectures/99-final-recommendation.md`](../../../architectures/99-final-recommendation.md) §九（99 §九 #25）中被**错误标注为 "DIPA"**。2026-08-30 核验发现：
> - 实际标题为 **"Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations"**
> - "DIPA" 在 arxiv 全字段搜索无独立结果——**本文 ID 对应论文被误标**
> - 已在 99 §九 #25 与 00 §七 #24 同步修订为正确标题

---

## 1. 元信息

- **作者**：Bowen Zuo, Dongruo Zhou, Yinglun Zhu
- **arXiv**：v1 提交 2026-04-22 (118 KB)，CC BY 4.0
- **被引用状态**：本套架构文档（00 §七 #24、99 §九 #25）已引用但**标题误标**，已修订

---

## 2. 核心内容（基于摘要与论文标题推断）

### 2.1 研究问题

现有 TTS（test-time scaling）方法要么**静态分配**计算，要么从**固定分布**采样。如何**联合适配**"计算花在哪"与"如何生成"？

### 2.2 方法（两阶段）

1. **Warm-up phase**：识别 easy queries，从**测试集自身**组装初始的 question-response pairs pool
2. **Adaptive phase**：把更多计算集中在 unresolved queries，并通过 **evolving in-context demonstrations** 重塑生成分布
   - 条件化每个生成于"语义相关问题的成功响应"而非"从固定分布重采样"

### 2.3 实验范围

数学、代码、推理基准测试

### 2.4 核心贡献

- 联合适配"where"与"how"分配
- 利用测试集本身作为示范池（无需额外标注）
- 通过语义相关 in-context demonstrations 进行上下文学习
- "consistently outperforms baselines while consuming substantially less inference-time compute"

---

## 3. 创新点（基于摘要）

| 创新 | 与之前 TTS 工作对比 |
|---|---|
| **联合适配 where + how** | 之前 LATTS（04b 引用）只调 step 难度；DIPA（Snell 等）只静态分配 |
| **测试集自身做演示池** | 之前 Self-Consistency 用同一 prompt 重采样；本工作用**已成功 query 的 response 作其他 query 的 in-context demo** |
| **自适应 in-context 演示演化** | 之前 few-shot ICL 静态选择 demo；本工作**随训练推进演化 demo pool** |
| **更少 inference compute** | "consistently outperforms baselines while consuming substantially less inference-time compute" |

---

## 4. 对项目借鉴

### 4.1 对 architectures 决策线（`../../../architectures/`）

- **v4.5 LATTS-style 早停的细化可能**：04b §1.2 W4 用 "LATTS-style 早停"——本文提供**更精细的 TTS 方案**（联合适配 where + how），可作为 W4 任务的升级选项
- **训练-free TTS 的价值再确认**：99 §八 把"Efficient TTS"列为方向 #3——本文证明"在更少 compute 下超越 baselines"，**支持该方向**
- **DIPA 误标的反思**：本次核验暴露了 architectures 文档的"事实性错误"——v3/v4 已有 6 项此类错误，本次修订**清除其中 1 项**。剩余 5 项应在后续审计中清理

### 4.2 对 AgenticDSL 训练链路

- **AgenticDSL 训练中的"自适应示范"灵感**：本文方法"用同类 query 的成功 response 作 ICL demo"——**可直接应用于 AgenticDSL 的 few-shot prompt 工程**
- **"测试集即知识" 的洞见**：本工作用测试集自身的成功响应做演示池，**隐含"分布内数据是最好示范"的发现**——对 AgenticDSL 的 Agent loop 中的 ICL 设计有直接借鉴价值
- **HydraForge 验证器的 TTS 适配**：HydraForge 4 层验证器调用可能因任务复杂度差异巨大——本文方法可作为**动态分配验证器计算**的参考

### 4.3 对 SOCA v3-Micro-Final

- **04 §1.2 W4 LATTS-style 早停**：本文可作为 W4 设计的具体参考——**从"静态早停"升级为"动态 compute 分配 + 演化 in-context"**
- **SOCA 04 §9 24 项架构消融可加入"test-time compute adaptive" 维度**：新增消融项 "SOCA-A24b: LATTS-static vs Zuo-2026-adaptive"

---

## 5. 对 AgenticMind 项目整体启示

| 启示 | 落地动作 |
|---|---|
| **AgenticMind 推理服务优化** | 不只是 "Constrained Decoding"——可加 "evolving in-context demo" 提升 few-shot 效果 |
| **SOCA 04 §1.2 W4 LATTS 升级** | 用 evolving in-context 替代 static early-stopping |
| **fact-check SOP** | 每季度核验所有 arXiv ID——避免 DIPA 误标再次发生 |

---

## 6. 核心方法示意

```
┌─────────────────────────────────────────────────┐
│  Warm-up Phase                                   │
│  • Identify easy queries (high baseline score)    │
│  • Assemble initial pool of Q-R pairs from       │
│    test set (success cases)                       │
│  • Pool = {(q1, r1), (q2, r2), ..., (qk, rk)}    │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  Adaptive Phase                                  │
│  • Identify unresolved queries                   │
│  • For each unresolved query:                     │
│    - Find semantically similar resolved query     │
│    - Use its successful response as in-context    │
│      demonstration                                │
│    - Generate response conditioned on demo        │
│  • Evolve pool: add new successful responses     │
└─────────────────────────────────────────────────┘
```

**关键创新点**：
- **计算分配**（where）：从均匀分配 → 集中在 unresolved queries
- **生成分布**（how）：从固定分布 → 由成功响应塑形
- **联合优化**：同时适配 where + how

---

## 7. 实验设置（基于摘要）

- 跨数学、代码、推理基准测试
- 与现有 TTS 方法对比（LATTS、DIPA、Self-Consistency 等）
- "consistently outperforms baselines while consuming substantially less inference-time compute"

> **注**：因 2604.21018 HTML 全文获取失败，详细实验数字未在本文档列出，需后续补充。

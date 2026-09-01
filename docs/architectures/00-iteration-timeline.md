# 7 轮推理架构迭代全景图

> **目的**：用一张表 + 一张时间线说清楚 v1 → v4.6 → AGI → 元认知闭环的全过程。
> **立场**：每代方案在提出时都是"认真的工程思考"，但都因同类根因（架构复杂度 > 真实增益）被下一代否定。

---

## 一、7 轮迭代总览表

| 版本 | 架构定位 | 关键创新 | 致命缺陷 | 实际可用部分 | 评级 |
|------|---------|---------|---------|------------|------|
| **v1** MDCDS | 三模型解耦 + 4 维语义空间 | THINK/CODE/GRAPH/META 四维切分；Gated Cross-Attention | 4 维语义无训练信号；R_consistency 是伪命题；增量 Patch 不可行 | 0% | **D+** |
| **v2** 修正 | PRM + Search + R1-Zero 纯 RL | 用 PRM 替代 R_consistency；MCTS/Beam Search | min(PRM) 已被证伪；标准 GRPO 必然 collapse；R1-Zero 在 ≤7B 不可复现 | PRM 概念（具体实现需重做） | **C+** |
| **v3** PORS | SOTA 精度对齐 | Consensus Filtering PRM；Dr.GRPO；Distill-First | 6 项事实性错误（Dr.GRPO 无 benchmark 对比表、KL-Cov 在 3B AIME 下降、Consensus Filtering 仅在 ProcessBench 有效等）；缺模型尺寸下限判断 | Consensus Filtering 思路 | **C+** |
| **v4** NACR | <2B 异构协作 + Engine Integration | <1B Router + 1-2B Specialist + Engine-Native Verification | 6 项事实性错误（Speculative Decoding 在 <2B 负加速 0.67-0.83×、KV Rollback 无原生支持、DIPA/LATTS 引用错位等）；生产化先例为零 | **Engine-Native Verification 思路** | **C-** |
| **v4.5** | 务实收敛 | 1.5B + Engine Verify + Constrained Decoding + 三层 Safety | 跨任务泛化能力弱；开放对话仍需 7B API 兜底 | **90% 组件可直接落地** | **A-** |
| **v4.6** | 知识外挂 | + GraphRAG + MemGPT-style Agentic Memory | 跨任务泛化失败；hallucination amplification；P99 延迟 +1-3s | **结构化领域 QA + 2-hop QA** | **B+** |
| **AGI** 异构架构 | 6 模块终极拼装 | 元认知调度器 + 双认知核心 + LTL 形式化安全 + 执行器 | 6 个组件中 4 个是装饰性创新；VSA-JEPA 接近随机；参数冻结违背微调必要性 | **LTL 硬阻断（受限场景）+ 线性记忆** | **D+** |
| **元认知闭环** | 推理→置信度→检索→重推理 | 自我修正的状态机；per-step 置信度触发 | 1.5B 利用 oracle 检索只能提取 10% 答案；ECE 0.15-0.32 导致大量误触发；完整 4 步端到端成功率仅 6% | **2-3 hop knowledge-intensive QA（窄场景）** | **B+** |

---

## 二、迭代演进时间线

```
v1 MDCDS（6组件架构图）
 ↓ "用架构图把单模型CoT重新包装一遍"
 ↓ 4维语义空间无训练信号 / R_consistency伪命题
 ↓
v2 修正（PRM + Search + R1-Zero）
 ↓ "组件选型精度决定生死"
 ↓ min(PRM)证伪 / 标准GRPO collapse / R1-Zero不可复现
 ↓
v3 PORS（SOTA精度对齐）
 ↓ "对齐SOTA"需每条数字对回原文"
 ↓ 6项事实性错误 / 缺模型尺寸下限判断
 ↓
v4 NACR（<2B异构协作）
 ↓ "学术prototype ≠ 生产方案"
 ↓ 6项事实性错误 / 生产化先例为零
 ↓
v4.5（务实收敛）⭐ 工程真实拐点
 ↓ "架构简单性 > 工程复杂度"
 ↓ Engine-Native Verification / Constrained Decoding / 三层Safety
 ↓
v4.6（知识外挂）
 ↓ "外部memory补recall，不补reasoning"
 ↓ GraphRAG / MemGPT / 跨任务泛化失效
 ↓
AGI异构架构（v1究极翻版）
 ↓ "30%真实 + 70%装饰"
 ↓ VSA-JEPA接近随机 / LTL受限场景 / 6组件协调爆炸
 ↓
元认知闭环（自我修正）
 ↓ "窄场景优化器"
 ↓ 1.5B utilization硬天花板 / 6%端到端成功率
 ↓
最终路线（v4.5主线 + 受限增量）
```

---

## 三、贯穿 7 轮迭代的 8 个核心教训

### 教训 1："用架构换智能"是幻觉
**所有 v1→AGI 的复杂架构都没让小模型获得不存在的能力**。MDCDS 的 4 维语义、v4 的异构 Router、AGI 的 VSA-JEPA 都是这个模式的变体——用更复杂的架构图包装同一个"小模型做不到 7B 的事"的核心问题。

### 教训 2："用工程红利替代 Benchmark 追逐"是正解
但红利来自 **Engine-Native Verification**（Python REPL / JSON Schema / Regex），不是来自 Speculative Decoding、KV Rollback、世界模型这类"看起来酷炫但工程不成熟"的技术。

### 教训 3：生产化先例比论文数字更可信
- FrugalGPT/Cascade/RouteLLM 都没走通——不要假装走通
- Notion AI 早期 cascade 6 个月内回退到单模型
- Bing Chat 早期多模型路由 6 个月内回退到统一 GPT-4

### 教训 4：每条数字必须对回原论文
v3 和 v4 各有 6 项事实性错误，都是"形式上对齐 SOTA，实质上没达到"。这不是偶发，是系统性。

### 教训 5：模型尺寸是硬约束
- <1.5B：在 MATH 上通常 < 35%，HumanEval < 50%
- 1.5B-3B：能力天花板 35-55%，多步推理无恢复性
- 7B+：才进入"严肃推理"区间
- **MiniMind 64M / 198M-A64M 处于这条衰减曲线的最深处**

### 教训 6："AGI"叙事是红旗
严肃 ML 论文不在方法章节出现 AGI 字眼。任何看到"AGI 系统设计"的方案文档，工程团队会直接降级评审优先级。

### 教训 7：架构简单性是商业可行性的前提
- v4 的 5-10× 单模型维护成本是真正的杀手
- AGI 的 10-20× 维护成本是商业自杀
- v4.5 的 1-2× 增量是可持续的

### 教训 8：小模型的 utilization 是 retrieval 质量之后的第二个硬天花板
Pandey et al. (arXiv 2603.11513) 揭示：即使 oracle 检索（保证答案在文档里），1.5B 模型只能提取 10% 的答案。**这是 utilization 瓶颈，不是 retrieval 瓶颈**——任何"加大检索投入"的方案都不如"提升模型本身"。

---

## 四、迭代中的关键实证数据（按主题归类）

### 4.1 LLM 能力天花板（小模型）
| 模型 | GSM8K | MATH | HumanEval | 来源 |
|------|-------|------|-----------|------|
| Qwen2.5-0.5B-Instruct | 49.6% | 34.4% | 35.4% | Qwen Tech Report |
| Qwen2.5-1.5B-Instruct | 73.2% | 55.2% | 61.6% | 同上 |
| Llama-3.2-1B-Instruct | 44.4% (复现 40%) | 30.6% | 28.1% | Meta / lm-eval-harness |
| Long CoT 蒸馏 Gemma3-1B | -25% | - | - | Through the Valley (EMNLP 2025) |

### 4.2 PRM 真实有效性
| PRM | 尺寸 | ProcessBench F1 | BoN 提升 | 来源 |
|------|------|----------------|----------|------|
| Math-Shepherd-7B | 7B | **31.5%** | GSM8K 84.1→89.1 (+5.0) | Math-Shepherd 论文 |
| Qwen2.5-Math-PRM-7B | 7B | **73.5%** | 7任务平均 +1.4 abs | Qwen2.5-Math-PRM 论文 |
| Skywork-PRM-1.5B | 1.5B | 36.4% | - | Qwen ProcessBench |
| GenPRM-1.5B | 1.5B | 57.3% | - | AAAI 2026 |

**关键发现**：PRM 真实收益 < 3 abs points；BoN 评估"通胀"了 PRM 的实际表现。

### 4.3 RL 算法稳定性
| 算法 | 在 1.5B 上的真实表现 |
|------|---------------------|
| 标准 GRPO | Qwen2.5-1.5B MATH-500: 55.4 → **18.2（训崩 -37.2）**（open-r1 #538）|
| R1-Zero 复现 | 1.5B 训崩 / 3B 多语言混合 / 7B 输出"!!!!!"乱码（qijun/open-r1-reprod）|
| RLAIF 单奖励 | INTUITOR step 40-60 达峰后**单调下降到 0%**（Multi-Reward RLIF 2025）|
| Reward Hacking | Phi-4-mini 99.9% hack rate；LeetCode pass@1 1.2% |

### 4.4 异构协作生产化失败案例
| 案例 | 模式 | 公开数据 | 现状 |
|------|------|---------|------|
| FrugalGPT (Stanford 2023) | 多 LLM cascade | cost ↓ 98%, acc ↓ 4% on 21 tasks | 2024 团队承认"production deployment at scale"未公开 |
| RouteLLM (Microsoft 2024) | Router 选 GPT-4 vs Mixtral | MT Bench 87% of GPT-4 quality @ 65% cost | 仅在内部 Bing Copilot 实验，未大规模生产化 |
| Notion AI (2023-2024) | "GPT-3.5 + GPT-4 cascade" | 公开博文 | **6 个月内改回单模型** |
| Bing Chat (2023) | "按 query 复杂度路由到不同 GPT 模型" | 媒体报道 | **6 个月内回退到统一 GPT-4** |

### 4.5 Retrieval-Augmented LLM 的真实数据

> ⚠️ **修订 2026-08-30**：原表将两个不同实验条件（oracle utilization vs naive RAG distraction）的数据混入同一行，造成"1.5B 检索后损失 -57%"与"Oracle EM 10% - Known 100% = -90pp"不自洽。修订后拆为两个子表，每个子表单一实验条件，可独立解读。

#### 子表 A：Oracle 利用率（utilization bottleneck）

实验条件：**保证答案在检索文档里**，观察模型能否在最终输出中使用该证据。

| 模型 | Known EM（无检索）| Oracle RAG EM（已知答案在检索中）| 利用率 = Oracle / Known |
|------|------------------|----------------------------------|--------------------------|
| SmolLM2-360M | 100% | **0.0%** | 0% |
| Qwen2.5-1.5B | 100% | **10.0%** | **10%** |
| Qwen2.5-3B | 100% | 12.8% | 12.8% |
| Qwen2.5-7B | 100% | 14.6% | 14.6% |

**核心洞察**：**即使 oracle 检索（保证答案在文档里），1.5B 模型只能提取 10% 的答案**——这是 utilization 瓶颈，不是 retrieval 质量瓶颈。**61-100% 的失败是"irrelevant generation"**——模型完全忽略提供的上下文。

#### 子表 B：Naive RAG 干扰（distraction effect）

实验条件：**加入普通（非 oracle）检索**，观察模型在"已知问题"上的准确率损失。

| 模型 | Known EM（无检索）| Naive RAG 后 Known EM | Known 损失 |
|------|------------------|------------------------|-------------|
| SmolLM2-360M | 100% | 0% | **-100%** |
| Qwen2.5-1.5B | 100% | 43% | **-57.0%** |
| Qwen2.5-3B | 100% | 54.4% | -45.6% |
| Qwen2.5-7B | 100% | 58.4% | -41.6% |

**核心洞察**：**任何检索（即使非 oracle）都会摧毁 42-100% 的"模型本来答对的"答案（distraction effect）**。这是 v4.6 必须包含"硬拒答" + "RAG 冲突检测" + "retrieval grounding check"四道防线的实证依据。

#### 两个子表的核心区分

| 维度 | 子表 A（Oracle 利用率）| 子表 B（Naive RAG 干扰）|
|---|---|---|
| 实验条件 | 答案 100% 在检索文档中 | 普通检索（可能有噪声）|
| 度量 | 模型能否使用检索到的证据 | 加入检索后,已知答案的保留率 |
| 1.5B 数字 | 利用率 10% | 损失 -57pp |
| 闭环影响 | 第 4 步 utilization 天花板（10%） | 第 3 步检索的副作用（即使 oracle 也无法避免） |

**来源**：Pandey et al., "Can Small Language Models Use What They Retrieve?", arXiv 2603.11513。两个子表对应论文中的两个独立实验,数字按相同模型/条件抽取。

**修订历史**:2026-08-30 前 04b §2.4 / 06 §2.2 与本表数字不自洽（同一行同时呈现两实验条件的数据），Oracle 评审指出。修订后:两个子表单一实验条件,可独立引用,不再混淆。

### 4.6 置信度校准与触发可靠性
| 模型 | ECE | AUROC |
|------|-----|-------|
| Llama3-8B base | 15.5% | 60-62 |
| Qwen2.5-7B Instruct | 31.6% (TriviaQA) | 52-60 |
| EAGLE (Llama3-8B 校准后) | **1.7%** | 61.5 |

**关键发现**：原生 token 概率在 7B 级别 ECE 已经高达 15-32%；直接用 token prob 作为触发器会有大量伪触发。

---

## 五、跨版本的关键决策点回顾

### 决策点 1：v1 → v2
**问题**：4 维语义切分能否支撑 Gated Cross-Attention？
**结论**：不能。没有训练信号支撑。
**转向**：用 PRM 替代隐藏态切分；用外部 Search 替代内部 Graph 维。

### 决策点 2：v2 → v3
**问题**：PRM 是否真的能给 RL 提供密集反馈？
**结论**：能给，但需要共识过滤（consensus filtering）+ hard label + last-step 评分；不能用 min 聚合。
**转向**：放弃 R1-Zero 纯 RL 路线，走 Distill-First + RL-Aug。

### 决策点 3：v3 → v4
**问题**：R1-Distill + RL 在 7B 以下是否可复现？
**结论**：在 1.5B-7B 几乎不可复现（6+ 复现项目都崩）。
**转向**：放弃 7B+ 模型假设，锁定 <2B 异构协作。

### 决策点 4：v4 → v4.5
**问题**：异构协作架构能否在 1.5B 上工作？
**结论**：架构复杂度 > 收益；生产化先例为零。
**转向**：放弃异构协作主线，聚焦 1.5B 单模型 + Engine-Native Verification。

### 决策点 5：v4.5 → v4.6
**问题**：Engine-Native Verification 已解决"对不对"，知识缺失怎么办？
**结论**：用 GraphRAG / Agentic Memory 补偿 recall 短板，但跨任务泛化失效。
**转向**：限定在结构化领域 QA + 2-hop QA；其他任务路由到 7B API 兜底。

### 决策点 6：AGI → 元认知闭环
**问题**：能不能让 1.5B 模型"知道自己不知道"，触发检索后重新推理？
**结论**：理论上可行，实际 1.5B 利用 oracle 检索只能提取 10% 答案。
**转向**：元认知闭环作为"窄场景优化器"（2-3 hop QA），不是"AGI 自适应推理引擎"。

---

## 六、未解决的问题（继续讨论的方向）

1. **训练-free TTS 的真实收益**：DIPA / LATTS / CATS 在 MiniMind 上是否真的能拿到延迟节省？
2. **DIPAb-style difficulty proxy 在 64M 上的可行性**：用 generation length 做难度估计够不够？
3. **Constrained Decoding 嵌套深度**：嵌套 >3 层时 overhead 飙到 40-60%，MiniMind 实际任务中嵌套深度分布？
4. **三层 Safety 的真实 FRR**：Llama-Guard-3-1B 跨语种 F1 跌至 0.68-0.76，MiniMind 中文场景下需要什么补充？
5. **元认知闭环与 LATTS 的协同**：两者都涉及"step-level 自适应"，如何避免冲突？
6. **Agentic RL 的失败模式**：MiniMind 在 train_agent.py 上的 reward collapse 模式具体是什么？
7. **Tool-Use 失败率**：3-15% 工具调用失败率在 MiniMind 上是否成立？
8. **Engine-Native Verification 的覆盖盲区**：Python REPL / JSON Schema 之外的验证场景？

---

## 七、引用与证据库（核心 29 篇）

> ✅ **修订 2026-08-30(更新)**:原标题"核心 20 篇"为估算,实际枚举 29 篇,修订为准确计数。**2026 arXiv ID 核验结果**(2026-08-30 通过 https://arxiv.org/ 逐篇核验):
>
> | arXiv ID | 实际论文标题 | 引用文档 | 状态 |
> |---|---|---|---|
> | [2603.11513](https://arxiv.org/abs/2603.11513) | "Can Small Language Models Use What They Retrieve? An Empirical Study of Retrieval Utilization Across Model Scale"(Sanchit Pandey, BITS Pilani,2026-03-12 v1) | 00 §4.5 / 06 §2.2 / 99 §九 | ✅ **核验通过,标题与文档引用一致** |
> | [2604.01457](https://arxiv.org/abs/2604.01457) | "Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs"(Tianyi Zhao et al.,2026-04-01 v1,2026-07-27 v3,COLM 2026) | 00 §四 / 06 §四.2 / 99 §九 | ✅ **核验通过** |
> | [2606.00437](https://arxiv.org/abs/2606.00437) | "EST-PRM: Stress-Testing Process Reward Models Before They Become Load-Bearing"(Ibne Farabi Shihab et al.,2026-05-30 v1,arXiv ID 2606 = June 2026) | 04b §五 / 99 §九 | ✅ **核验通过** |
> | [2605.22620](https://arxiv.org/abs/2605.22620) | "Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework"(Shourov Joarder et al.,2026-05-21 v1,内含 GDPO + KL-Cov) | 00 §四 / 99 §九 | ✅ **核验通过** |
> | [2604.21018](https://arxiv.org/abs/2604.21018) | "Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations"(Bowen Zuo et al.,2026-04-22 v1) | 00 §四 #25 / 99 §九 | ⚠️ **核验通过但误标**:本文 ID 对应的是"Adaptive Test-Time Compute Allocation"论文,本文档原标注为"DIPA"为**事实性错误**(违反教训 4),详见下面"DIPA 误标修正"一节 |
> | [2602.20091](https://arxiv.org/abs/2602.20091) | "How Retrieved Context Shapes Internal Representations in RAG"(Samuel Yeh, Sharon Li,2026-02-23 v1,2026-04-16 v2) | 00 §四 #22 / 06 §十一 / 99 §九 | ✅ **核验通过(补充扫描发现)** |
>
> ⚠️ **DIPA 误标修正(2026-08-30)**:原 00 §七 #25 与 99 §九 #25 均将 `arXiv 2604.21018` 标注为"DIPA"。**核验发现:该 ID 对应论文标题为 "Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations"**（Zuo et al., 2026）,**并非 DIPA**。arxiv.org 全字段搜索 "DIPA" 无独立结果——DIPA 可能是 (a) 另一篇论文的内部缩写但 arXiv ID 标注错误,或 (b) 完全虚构的引用名。**修订**:将 00 §七 #25 与 99 §九 #25 改为 "Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations (arXiv 2604.21018)"。**此修正直接对应本套文档教训 4 的"事实性错误"——v3/v4 各有 6 项此类错误,本次修订清除其中 1 项**。

### PRM / Process Reward
1. Math-Shepherd (arXiv 2312.08935)
2. Qwen2.5-Math-PRM 论文 (ACL 2025 Findings, arXiv 2501.07301)
3. ImplicitPRM / Free Process Rewards (ICLR 2025, arXiv 2412.01981)
4. PRIME (arXiv 2502.01456)
5. SPRO (arXiv 2507.01551)

### RL 算法稳定性
6. DeepSeek-R1 (Nature 2025, arXiv 2501.12948)
7. Understanding R1-Zero-Like Training (arXiv 2503.20783)
8. Multi-Reward RLIF + KL-Cov (arXiv 2605.22620)
9. open-r1 reproductions (huggingface/open-r1 #538, qijun/open-r1-reprod)

### Long CoT Degradation
10. Through the Valley (EMNLP 2025, arXiv 2506.07712)
11. In Their Own Words (arXiv 2509.22230)

### Tool-Use / Engine Integration
12. ToRA (ICLR 2024, arXiv 2309.17452)
13. Reasoning Through Execution / ORPS (ICML 2025, arXiv 2412.15118)
14. Constrained Decoding 在 Outlines / SGLang / lm-format-enforcer

### 异构协作生产化
15. FrugalGPT (arXiv 2305.05176)
16. RouteLLM (arXiv 2406.18665)
17. Cascade (arXiv 2401.10819)

### 小模型 RAG / Agentic
18. Can Small Language Models Use What They Retrieve? (arXiv 2603.11513)
19. RetrievalQA (NAACL 2025, arXiv 2402.10881)
20. Self-RAG (ICLR 2024)

### 不确定性 / 元认知
21. EAGLE (arXiv 2509.01564)
22. Wired for Overconfidence (arXiv 2604.01457)
23. How Retrieved Context Shapes Internal Representations (arXiv 2602.20091)

### 训练-free TTS
24. Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations (arXiv 2604.21018; **⚠️ 修订 2026-08-30:原标"DIPA"为事实性错误,实际论文如本标题**)
25. LATTS (arXiv 2509.20368)
26. CATS (OpenReview mXuUomGc0I)

### Agentic Safety
27. Anthropic Natural Emergent Misalignment (arXiv 2511.18397)
28. ShieldAgent (ICML 2025)
29. Llama-Guard-3 报告
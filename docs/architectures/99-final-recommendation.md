# 最终推荐路线 + Kill Criteria

> **目的**：将 7 轮迭代（v1 → v4.6 → AGI → 元认知闭环）的所有结论提炼为 MiniMind 项目的可执行路线。
> **核心问题**：在 MiniMind 这种 64M-198M 极小模型尺寸下，应该走哪条路？

---

## 一、一句话最终建议

> **MiniMind 项目应走 v4.5 主路径（1.5B + Engine-Native Verification + Constrained Decoding + 三层 Safety），5 周冲刺可见 KPI，止损条件清晰。** 不要从 v4.5 跳到 v4.6 之外的更复杂方案（如 AGI 异构架构、元认知闭环作为独立大版本）。**真正的下一步是把 v4.6（GraphRAG + Agentic Memory）和元认知闭环作为 v4.5 的可插拔功能增量，按 ROI 分阶段引入**。

---

## 二、推荐路线图（按 ROI 排序）

### 路线 1：v4.5 主路径（必做，5 周冲刺）

**定位**：务实收敛点。架构简单性 > 工程复杂度。

**架构**：
```text
<1B Router & Guardrail (L1)
  ↓
1.5B Specialist (Short-CoT SFT)
  ↓
Engine-Native Verification (Python REPL + JSON Schema + Regex)
  ↓
Constrained Decoding
  ↓
L2 Output + L3 HITL
```

**5 周冲刺**：

| 周 | 任务 | 交付物 | 决策点 |
|----|------|-------|--------|
| W1 | Qwen2.5-1.5B-Instruct 基线评测 + 数据准备 | 评测报告 + 50K Short-CoT SFT 数据 | KPI 基线确认 |
| W2 | Short-CoT SFT 训练 + MiCoTA 集成 | 训练好的 1.5B Specialist | GSM8K/MATH 准确率 |
| W3 | Engine-Native Verification 集成 | Python REPL + JSON Schema 模块 | Verify 失败率 < 1% |
| W4 | Constrained Decoding + LATTS-style 早停 | 推理服务 v0.9 | P50/P99 延迟达标 |
| W5 | 三层 Safety + 端到端联调 + 灰度 | 生产就绪服务 | 全部 KPI 达标 |

**KPI**：

| KPI | 目标 | 测量 |
|-----|------|------|
| GSM8K 端到端准确率 | ≥88% | 1000 题 held-out |
| MATH 端到端准确率 | ≥75% | 1000 题 |
| HumanEval 端到端 | ≥90% | 164 + 200 中文 |
| P50 TTFT | ≤150ms | 线上 P50 |
| P99 端到端延迟 | ≤2000ms | 线上 P99 |
| 单 query 成本 | ≤$0.0003 | 账单 |
| 安全 F1（跨语种）| ≥0.85 | 2000 条多语种 |
| 工程故障点 | ≤3 | 故障注入测试 |

**Kill Criteria（任一触发，2 周决策窗口）**：
1. **能力止损**：SFT + MiCoTA 后 MATH < 60%（1.5B 基线不够，**切 3B dense，不是 MoE**）
2. **工程止损**：Engine Verify P99 失败率 > 5%
3. **延迟止损**：Constrained Decoding overhead > 30%
4. **成本止损**：单 query 成本 > $0.0005
5. **架构止损**：safety 三层跨语种 F1 < 0.80

### 路线 2：v4.6 增量（可选，6-8 周）

**定位**：在 v4.5 基础上增加 GraphRAG + Agentic Memory，专门优化结构化领域 QA。

**优先级排序**：

**Phase 1（必做，高 ROI，~10 PM）**：
1. GraphRAG 基础设施（最关键）—— KG 抽取 pipeline + 向量/图库 + hybrid retrieval
2. Constrained Decoding 的 tool-call schema 扩展
3. L1 Router 增强（"是否需要 retrieve"维度）
4. L2 Output 加 retrieval grounding check（防 hallucination amplification）

**Phase 2（推荐做，中 ROI，~6 PM）**：
5. MemGPT-style 分层 memory
6. Agent loop 框架 + tool registry（**严控 max step cap = 5**）
7. 评测体系（multi-hop、long-conv、domain QA、tool calling）

**Phase 3（可选，低 ROI，~4 PM）**：
8. 跨 session 长期 memory（很可能做了也用不上）
9. GraphRAG 增量更新 pipeline
10. 多模态 memory（完全是另一个项目）

**适用任务（v4.6 真正能打）**：
- ✅ 结构化领域 QA（企业 KG、医/法/工程文档）
- ✅ 结构化 API 工具调用
- ✅ 2-hop QA（明确两跳关系）
- ✅ 企业内部助手（约束明确、KG 完整）

**不推荐任务（v4.6 应该明确放弃）**：
- ❌ Open-domain QA / web search 类
- ❌ 长对话（> 30 轮）
- ❌ 开放式创作 / 闲聊
- ❌ 复杂多步规划（≥5 步 task agent）
- ❌ 跨模态 / 多文件 / 跨语言对齐

**Kill Criteria（v4.6 特定）**：
- L1 Router 启用率 < 30%（70% 请求根本不该进 v4.6 路径）
- Domain QA 任务 v4.6 增益 < 15%（vs v4.5）
- Multi-hop QA hallucination amplification > 5%（vs v4.5）
- Agent loop 5 步内成功率 < 50%

### 路线 3：元认知闭环（v4.7，可选，6-12 周）

**定位**：v4.7 独立大版本，在 v4.6 基础上做"推理→置信度→检索→重推理"闭环。

**前提**：仅在 v4.6 完整 ship + 至少 2 周线上数据收集后启动。

**第一个 PR 必须包含**：
1. 显式状态机定义
2. max_retries=2 硬编码
3. 简单 factoid / creative / safety 三类任务直接 bypass
4. L2/L3 降级路径优先于闭环核心实现
5. GraphRAG context cache 与闭环同步设计

**第一个 KPI**：
- 2-hop QA 准确率提升 ≥ 10pp（基线 20% → 目标 30%+）
- p99 延迟 ≤ 6s
- 死循环率 < 0.5%
- 安全事件 = 0

**Kill Criteria**：
- p99 延迟 > 8s 持续 2 周
- 净失败率改善 < 5pp
- 死循环触发率 > 2%
- 注入错误知识导致的安全事件 ≥ 1 次
- 1.5B 模型上 confidence head ECE 持续 > 0.25

---

## 三、不要走的路线（明确拒绝）

### 路线 X：AGI 异构系统架构
**拒绝理由**：
- v1 的究极翻版（用更多组件包装同一想法）
- 6 个组件中 4 个是装饰性创新
- 生产化先例为零
- VSA-JEPA 在物理常识基准上接近随机
- LTL 形式化安全在开放式 LLM 输出上不可行

### 路线 Y：MoE 3-4B + LoRA + 异构协作
**拒绝理由**（基于 v4.1 决策）：
- MoE 在激活 1-1.5B 时实际能力 ≈ dense 1.0-1.2B（**能力下限被进一步压低**）
- MoE 不解决 v4 的任何核心架构问题（只是把 Router 错误换成 Expert Routing 错误）
- 全参数加载显存 7-9GB（激活 1/8 也需全载）
- 维护成本是 dense 1.5B 的 5-10×

### 路线 Z：纯 RL 路线（R1-Zero 复现）
**拒绝理由**：
- 在 1.5B-7B 几乎不可复现（6+ 复现项目都崩）
- huggingface/open-r1 #538：Qwen2.5-1.5B + GRPO + Math-220K，MATH-500 从 55.4 → 18.2（-37.2 abs）
- qijun/open-r1-reprod：1.5B 训崩 / 3B 多语言混合 / 7B 输出"!!!!!"乱码
- **DeepSeek-R1 的"Aha moment"是 V3-Base 预训练分布 + rule-based verifier + GRPO 三者特化结合的产物，不具备普适性**

---

## 四、整轮迭代的 8 个核心教训

### 教训 1："用架构换智能"是幻觉
所有 v1→AGI 的复杂架构都没让小模型获得不存在的能力。MDCDS 的 4 维语义、v4 的异构 Router、AGI 的 VSA-JEPA 都是这个模式的变体。

### 教训 2："用工程红利替代 Benchmark 追逐"是正解
但红利来自 Engine-Native Verification（Python REPL / JSON Schema / Regex），不是来自 Speculative Decoding、KV Rollback、世界模型。

### 教训 3：生产化先例比论文数字更可信
- FrugalGPT/Cascade/RouteLLM 都没走通
- Notion AI / Bing Chat 的早期 cascade 在 6-12 个月内回退

### 教训 4：每条数字必须对回原论文
v3 和 v4 各有 6 项事实性错误，都是系统性。

### 教训 5：模型尺寸是硬约束
- <1.5B：在 MATH 上通常 < 35%
- 1.5B-3B：能力天花板 35-55%
- 7B+：才进入"严肃推理"区间
- **MiniMind 64M/198M-A64M 处于这条衰减曲线的最深处**

### 教训 6："AGI"叙事是红旗
严肃 ML 论文不在方法章节出现 AGI 字眼。

### 教训 7：架构简单性是商业可行性的前提
- v4 的 5-10× 单模型维护成本是真正的杀手
- AGI 的 10-20× 维护成本是商业自杀
- v4.5 的 1-2× 增量是可持续的

### 教训 8：小模型的 utilization 是 retrieval 质量之后的第二个硬天花板
Pandey et al. 揭示：即使 oracle 检索，1.5B 只能提取 10% 答案。

---

## 五、推荐路线的核心优势

### v4.5 vs 之前所有版本的优势

| 维度 | v1-v4 | AGI | v4.5 |
|------|-------|-----|------|
| 战略眼光 | 参差不齐 | B+（清晰的"AGI"叙事） | **A**（明确的"务实收敛"定位） |
| 工程可行性 | D+ | D | **A-** |
| 文档精度 | C- | C+ | **B+** |
| 系统性盲区识别 | B | C | **B+** |
| 维护成本 | 5-10× 单模型 | 10-20× 单模型 | **1-2× 单模型** |
| 迭代速度 | 0.2-0.3× | 0.05× | **1×** |
| 能力上限 | <1B ~35% | 未定义 | **<1.5B ~95%（结构化任务）** |
| 生产化证据 | F | F | **B**（所有组件都有开源实现） |

---

## 六、决策矩阵：什么场景选什么路线

| 任务场景 | 推荐路线 | 备注 |
|---------|---------|------|
| 结构化数学/代码 | v4.5 | Engine-Native Verification 是核心能力倍增器 |
| 结构化领域 QA（有 KG）| v4.6 | GraphRAG 解决 recall 短板 |
| 2-hop 多跳 QA | v4.6 + v4.7 | GraphRAG + 元认知闭环 |
| 简单 factoid QA | v4.5 | 不需要 GraphRAG/Agent |
| Creative writing | v4.5（直接绕过 L1-B）| 检索破坏创作连贯性 |
| Safety-sensitive | v4.5 三层 Safety | 安全通道独立 |
| Open-domain QA | 路由到 7B API | v4.5/v4.6 在 OOD 上能力上限低 |
| 长对话（>30 轮） | 路由到 7B API | 1.5B working memory 撑不住 |
| 复杂多步规划 | 路由到 7B API | 1.5B 规划能力是硬伤 |

---

## 七、与现有 v3 主线的衔接

### 7.1 v3 当前的实现（MiniMind README 已记录）
- ✅ 放弃独立 PRM 路线（rule-based verifiable reward）
- ✅ 保留 R1-Distill 风格的 thinking 模板
- ✅ 主算法 CISPO + GRPO
- ✅ Agentic RL
- ✅ Adaptive Thinking 软开关
- ❌ 没有任何 PRM 训练/加载
- ❌ 没有 test-time TTS 自适应分配

### 7.2 v4.5 与 v3 主线的衔接

**v3 主线是 64M 模型，v4.5 推荐 1.5B 模型——尺寸不同**。两个衔接路径：

**路径 A（推荐）：v3 主线 + v4.5 增量**
- v3 主线（64M）继续作为教学项目基线
- 增加 v4.5 增量（1.5B 模型 + Engine-Native Verification）
- 作为可选"实验分支"，提供"如果模型尺寸提升到 1.5B 应该怎么走"的参考

**路径 B（备选）：纯 v3 主线**
- 64M 模型继续是核心
- 用 v4.5 中的 Engine-Native Verification 思路（Python REPL + JSON Schema）作为推理增强
- 短期 ROI 高，长期受限于 64M 能力天花板

**路径 C（不推荐）：完全替换为 v4.5**
- MiniMind 项目的核心价值是"3 块钱 + 2 小时训练"的教学定位
- 1.5B 模型 + 5 周冲刺的训练成本会破坏这个定位

**推荐采用路径 A 或 B，取决于用户反馈**。

### 7.3 与 MiniMind README 中已有的 PRM_RESEARCH_REPORT.md 关系

**PRM_RESEARCH_REPORT.md 是 v2 时期做的 PRM 调研**：
- 与本套文档互补
- 本套文档覆盖 v1→v4.6 + AGI + 元认知闭环的迭代全貌
- PRM_RESEARCH_REPORT 深入 PRM 本身的细节（ProcessBench F1、reward hacking、跨域迁移等）

**建议保留两个文档**：
- `PRM_RESEARCH_REPORT.md` - PRM 路线深度调研
- `docs/reasoning-architectures/` - 整轮迭代架构调研

---

## 八、后续研究方向（v4.5/v4.6 之后）

### 8.1 真正值得投入的 4 个方向

1. **PRM Robustness**：针对 EST-PRM 揭示的攻击面（步骤重排/膨胀），设计对抗训练或架构级防御
2. **领域特化 PRM**：在数学/代码之外的领域（科学推理、法律论证、医疗诊断）构建高质量 PRM
3. **Efficient TTS**：在保持 Compute-Optimal 精度的前提下，将搜索开销降低 50%+
4. **Safe Agentic Reasoning**：解决 Anthropic 揭示的 70% misalignment 残留问题

### 8.2 不值得投入的方向

1. ❌ AGI 异构架构（v1 翻版）
2. ❌ MoE 3-4B + LoRA + 异构（v4.1 否决）
3. ❌ 纯 RL 路线（R1-Zero 复现已失败 6+ 次）
4. ❌ VSA + JEPA 世界模型（接近随机）
5. ❌ LTL 形式化安全在开放式 LLM 上（开放问题）

---

## 九、引用与证据库

### 核心 25 篇文献（按主题）

#### PRM / Process Reward
1. Math-Shepherd (arXiv 2312.08935)
2. Qwen2.5-Math-PRM 论文 (ACL 2025 Findings, arXiv 2501.07301)
3. ImplicitPRM / Free Process Rewards (ICLR 2025, arXiv 2412.01981)
4. PRIME (arXiv 2502.01456)
5. SPRO (arXiv 2507.01551)

#### RL 算法稳定性
6. DeepSeek-R1 (Nature 2025, arXiv 2501.12948)
7. Understanding R1-Zero-Like Training (arXiv 2503.20783)
8. Multi-Reward RLIF + KL-Cov (arXiv 2605.22620)
9. open-r1 reproductions (huggingface/open-r1 #538, qijun/open-r1-reprod)

#### Long CoT Degradation
10. Through the Valley (EMNLP 2025, arXiv 2506.07712)
11. In Their Own Words (arXiv 2509.22230)

#### Tool-Use / Engine Integration
12. ToRA (ICLR 2024, arXiv 2309.17452)
13. Reasoning Through Execution / ORPS (ICML 2025, arXiv 2412.15118)
14. EST-PRM (arXiv 2606.00437)

#### 异构协作生产化
15. FrugalGPT (arXiv 2305.05176)
16. RouteLLM (arXiv 2406.18665)
17. Cascade (arXiv 2401.10819)

#### 小模型 RAG / Agentic
18. Can Small Language Models Use What They Retrieve? (arXiv 2603.11513)
19. RetrievalQA (NAACL 2025, arXiv 2402.10881)
20. Self-RAG (ICLR 2024)
21. OnionEval (arXiv 2501.12975)

#### 不确定性 / 元认知
22. EAGLE (arXiv 2509.01564)
23. Wired for Overconfidence (arXiv 2604.01457)
24. How Retrieved Context Shapes Internal Representations (arXiv 2602.20091)

#### Training-free TTS
25. DIPA (arXiv 2604.21018)
26. LATTS (arXiv 2509.20368)
27. CATS (OpenReview mXuUomGc0I)

#### Agentic Safety
28. Anthropic Natural Emergent Misalignment (arXiv 2511.18397)
29. ShieldAgent (ICML 2025)
30. Llama-Guard-3 报告

#### 模型尺寸能力天花板
31. Qwen2.5 Technical Report (2024-09)
32. Pandey et al. 1.5B RAG 利用率 (arXiv 2603.11513)
33. Through the Valley Long CoT Degradation (EMNLP 2025)

---

## 十、一句话最终总结

> **MiniMind 项目应走 v4.5 主路径（1.5B + Engine-Native Verification + Constrained Decoding + 三层 Safety）的务实收敛路线，把 v4.6（GraphRAG + Agentic Memory）和元认知闭环作为可选功能增量按 ROI 分阶段引入。** 整个 7 轮迭代反复证明：在 MiniMind 这种 64M-198M 极小模型上，"用架构换智能"是幻觉，"用工程红利换能力"才是正道。任何超过此复杂度的方案（AGI 异构架构、MoE + 异构协作、纯 RL 路线）都会在 6-12 个月内被工程现实击穿。

---

## 附录：本推荐路线的执行入口

- **v4.5 主路径**：从 W1 开始，按 `04b-v4.5-and-v4.6.md` 中的"5 周冲刺执行顺序"实施
- **v4.6 增量**：在 v4.5 完整 ship 后，按 `04b-v4.5-and-v4.6.md` 中的"v4.6 优先级排序"实施
- **元认知闭环 v4.7**：在 v4.6 完整 ship + 2 周线上数据后，按 `06-metacognitive-closed-loop.md` 中的"v4.7 第一个 PR 必须包含"启动
- **如果想回顾完整论证**：从 `00-iteration-timeline.md` 开始读，按 README.md 中的"阅读路径建议"选择深度
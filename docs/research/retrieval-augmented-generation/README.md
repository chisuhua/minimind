# Retrieval-Augmented Generation (RAG) 研究

> **目录定位**:`docs/research/retrieval-augmented-generation/` —— 围绕 **RAG（检索增强生成）在 LLM 中的工作机理与失败模式** 的研究文档。
>
> **核心问题**:LLM 在使用检索增强时,究竟能否有效利用检索到的上下文？检索上下文如何影响 LLM 的内部表示？
>
> **关联度**:⭐⭐⭐⭐ **高** —— 直接关联 AgenticMind 训练路线（AgenticDSL 是否需要 RAG 增强）与 SOCA v3-Micro-Final 的 M15 SOCAMonitor 设计。

---

## 📁 论文清单（2 篇，2026 年）

| arXiv ID | 标题 | 作者 | v1 日期 | 模型规模 | 问题角度 |
|---|---|---|---|---|---|
| [2603.11513](https://arxiv.org/abs/2603.11513) | Can Small Language Models Use What They Retrieve? | Sanchit Pandey (BITS Pilani) | 2026-03-12 | 360M-8B | **输出行为层面**：EM 准确率、利用率、distraction effect |
| [2602.20091](https://arxiv.org/abs/2602.20091) | How Retrieved Context Shapes Internal Representations in RAG | Samuel Yeh, Sharon Li (UW-Madison) | 2026-02-23 (v2 2026-04-16) | 17B-80B | **内部表示层面**：hidden states PCA 可视化、层-wise 分析 |

---

## 🎯 关键发现（两篇交叉印证）

### 共同主题：RAG 在 LLM 中的"上下文处理"远不如"上下文检索"成熟

- **Pandey #1（输出层）**：≤7B 模型即使在 oracle 检索（保证答案在段落中）下，对 Unknown 问题仅 10-14.6% EM；1.5B 利用率仅 10%。检索摧毁 42-57% 的已知答案（distraction effect）。
- **Yeh #6（表示层）**：相关文档主要起"确认信号"作用，对 hard query 利用率低；后期层（L35+）开始把相关文档表示拉回无文档状态（**parametric knowledge dominance**）。

### 互补关系

| 维度 | Pandey #1（输出） | Yeh #6（表示） | 互补价值 |
|---|---|---|---|
| 分析对象 | 最终答案（EM） | hidden states（PCA） | 输出 vs 机制 |
| 模型规模 | 小模型（≤8B） | 大模型（17B-80B） | 互补：两个极端 |
| 关键指标 | 利用率 10-14.6% | 表示漂移 vs 锚定 | 一个量化、一个解释机制 |
| 核心结论 | ≤7B 利用率是硬天花板 | parametric knowledge 在后期层占主导 | **联合揭示"小模型 RAG 净损失"的双重证据** |

---

## 📐 对 AgenticMind 项目借鉴

### 对 SOCA v3-Micro-Final（`../soca/`）

- **直接证伪 SOCA "复杂架构能突破小模型利用率天花板" 的假设**：Pandey 7B 仅 14.6%——复杂 SAE + Bus + MoE 路线无法突破 utilization 硬天花板
- **SOCA M15 SOCAMonitor 的科学基础得到 Yeh 论文支撑**：hidden states 包含有意义的 retrieval 相关信号，SOCA 的 SAE 解码 latent 可观测这些信号
- **SOCA Bus M1 设计得到 Yeh "late-layer param dominance" 现象支撑**：SOCA Bus 在 middle-to-late 层积累正是对抗该现象

### 对 architectures 决策线（`../../architectures/`）

- **直接支持 v4.6 "Hallucination Amplification 4 道防线"**：Pandey distraction effect 与 04b §2.6 同构
- **直接支持 v4.6 "增加检索数量优于严格过滤"**：Yeh Obs 3 证明"1 relevant 文档锚定表示"
- **F-01 决策影响**：Pandey ≤7B 净损失 ~3pp → **强烈建议绕开 RAG 路线，优先 Engine-Native Verification**

### 对 AgenticDSL 训练链路（`../../AGENTS.md`）

- **HydraForge 4 层验证器的稳健性优势**：L1 grammar + L2 signature 对 Pandey distraction 攻击免疫
- **AgenticDSL 应优先 Engine-Native Verification 而非 RAG 增强**：与 AGENTS.md F-04 "不依赖 PRM" 一致
- **AgenticMind 产品的"拒答置信度"机制**：Yeh 证明 instruction-tuned 模型对 random context 97.6% abstention——应监控 abstention 率 ≤20%

---

## 📊 与项目决策的对应

| AGENTS.md 决策 | 本目录论文 | 行动建议 |
|---|---|---|
| **F-01 基模型选择** | #1 Pandey | ≤7B 利用率 ≤14.6%，**F-01 选 ≥7B（当前 7B 决策实证支撑）** |
| **F-02 是否引入 RAG** | #1 Pandey + #6 Yeh | **不建议小模型 + RAG**，优先 multi-reward RLIF 训练路线（见 `reinforcement-learning/`） |
| **F-04 验证器 vs PRM** | #1 Pandey distraction | HydraForge 结构化验证 > RAG 检索增强 > PRM |

---

## 🚀 推荐阅读顺序

| 读者 | 阅读路径 |
|---|---|
| **架构师 / 决策者** | Pandey（输出层证据）→ Yeh（表示层机制）→ 本 README §对项目借鉴 |
| **算法工程师（AgenticDSL 训练）** | Yeh §5.1 上下文相关性 → §5.2 层-wise 过程 → §6 实践建议 |
| **SOCA 06 M15 SOCAMonitor 设计者** | Yeh §3-§4（表示分析设置）→ §5（观察）→ §6（与 SOCA Bus / SAE 对照） |
| **HydraForge 验证器设计者** | Pandey §4.5（oracle utilization）→ §5（错误分析）→ §6.3（实践建议） |

---

## 📅 调研版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-30 | 初版，2 篇论文调研完成（原 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) §1+#6 拆分） |

---

> **本目录定位**:作为 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) 的**分主题深度版**——survey 提供全局视角与跨论文对比，本目录提供单篇论文的完整内容/创新/借鉴分析。

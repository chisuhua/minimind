# 推理架构 · 技术系列

> **目录定位**:`docs/architectures/` —— 与新目标(**自循环认知推理模型**)直接相关的 **4 份 DIRECT 架构文档**。
>
> **新目标相关度**:⭐⭐ **DIRECT(直接相关)**
>
> **纳入理由**:这 4 份文档分别对应"自循环认知推理模型"工程路径的**演进史、收敛点、闭环对应物、最终决策**。

---

## ⚠️ 历史定位横幅（修订 2026-08-30）

**本套文档撰写于 2026 年上半年,当时项目名为 MiniMind,定位是"64M-198M 极小推理架构选型"。**

**2026-06-13 起,项目发生重大 pivot**:
- 项目重命名为 **AgenticMind**(从通用 LLM 训练 pivot 到 **AgenticDSL 训练链路**)
- 当前执行主线见 [`../README.md`](../README.md) + [`../../AGENTS.md`](../../AGENTS.md)
- 基模型决策权威在 [`../../AGENTS.md` §12.10 F-01](../../AGENTS.md)(候选 0.5B / 1.5B / 7B,待定)

**本套文档的当代定位**:
- ✅ **作为决策史档案保留**(7 轮迭代的工程教训)
- ✅ **作为证据库保留**(Pandey utilization 10% / ECE 校准 / 生产化失败案例等**对当前主线仍然成立**的实证论据)
- ⚠️ **不再作为当前执行依据**(若需执行请重新评估基模型决策后按 04b §一.6 重新规划)

**若发现本套文档与当前项目主线矛盾**,以 [`../../AGENTS.md`](../../AGENTS.md) 为准。

---

## 🔗 与 AgenticDSL 主线的概念映射

虽然本套文档未明确提及 AgenticDSL,但其中若干核心概念在 HydraForge 4 层验证器/AgenticDSL 自循环推理中**有直接的工程血脉**:

| 本套文档概念 | 位置 | AgenticDSL 主线对应 | 关联文档 |
|---|---|---|---|
| **Engine-Native Verification**(Python REPL + JSON Schema + Regex)| 04b §1.2 | HydraForge **4 层验证器 L1(grammar) + L2(signature)** 的思想来源 | [`../../docs/agenticinference/`](../../docs/agenticinference/) |
| **Confidence-Triggered Loop** | 06 §一 | AgenticDSL 自循环推理的**置信度评估触发**机制(详见 AGENTS.md 自循环机制)| [`../../AGENTS.md`](../../AGENTS.md) §3 |
| **元认知闭环**(置信度 → 检索 → 重推理) | 06 §五 | **4 层验证器 L3(execution) 失败 → L4(task) 重试 → 重新生成** 的循环结构 | 同上 |
| **3-5× 成本优势**(1.5B + Verify vs 7B) | 06 §8.2 | **<1B 模型 + AgenticDSL 训练**(小模型利用显式符号 DSL) 的成本论据 | 同上 |
| **Pandey utilization 10%** | 00 §4.5 / 06 §2.2 | **<1B 模型 + 检索注入的设计必须直面 10% utilization 硬天花板**——这正是 AgenticDSL 选择"显式符号结构 + 验证器"而非"纯检索增强"的根本原因 | AGENTS.md §3 + 00 §4.5 |
| **MiniMind 64M 衰减曲线最深处** | 99 §四 教训 5 | **64M/198M 极小模型能力天花板硬约束**——这是 TR-1 TR-2 TR-3 选择 1.5B-7B 基模型的实证基础 | AGENTS.md §4 + 99 §四 |
| **PRM 真实收益 < 3 abs points** | 00 §4.2 | **AgenticDSL 不依赖 PRM**(因为 AgenticDSL 本身有 4 层验证器提供过程信号)——本结论支持避开 PRM 路线 | AGENTS.md §3 |

> **结论**:本套文档不是"过时"或"无关"的——其中 **5/8 关键概念** 与 AgenticDSL 主线有直接血脉或反证价值。建议**先读 00 §三 教训 1-8**(8 个核心教训) + 00 §4.5(Pandey utilization),再读 [`../../AGENTS.md`](../../AGENTS.md) 理解当前主线如何吸收了这些教训。

---

## 📁 目录结构

| 文档 | 关联度 | 内容摘要 | 当前定位 |
|---|---|---|---|
| ⭐⭐ [`00-iteration-timeline.md`](./00-iteration-timeline.md) | DIRECT | 7 轮推理架构迭代全景图(v1 → v4.6 → AGI → 元认知闭环),架构复杂度的演化与决策依据 | 决策史(可作为"反面参考") |
| ⭐ [`04b-v4.5-and-v4.6.md`](./04b-v4.5-and-v4.6.md) | DIRECT | v4.5(Engine-Native Verification 务实收敛) + v4.6(GraphRAG + Agentic Memory),**唯一可落地的工程路径** | 历史路线(**不再适用**——基模型假设 F-01 待定) |
| ⭐⭐ [`06-metacognitive-closed-loop.md`](./06-metacognitive-closed-loop.md) | DIRECT | **元认知闭环**(推理→置信度评估→低置信触发→知识检索/注入→重新推理),与新目标的"自循环闭环"概念**一一对应**(B+ 评级,窄场景 sweet spot 有效) | 概念血脉(AgenticDSL 自循环的工程祖先) |
| ⭐ [`99-final-recommendation.md`](./99-final-recommendation.md) | DIRECT | 最终推荐路线 + Kill Criteria,自循环认知推理模型的**工程决策依据** | 决策史 + 证据库 |

---

## 🎯 与新目标的关系

新目标核心:**通过 AgenticDSL 语言和智能体运行时协同进行自循环的认知推理模型**(生成 → 执行 → 评估 → 优化)。

`architectures/` 子目录提供:
- **演进史**(`00`):为什么当前选择"自循环"而非"更大模型堆叠"——7 轮迭代的工程教训
- **务实路径**(`04b`):v4.5 收敛点是"自循环"路线的工程起点
- **闭环对应物**(`06`):"元认知闭环"是"自循环"在 reasoning 维度的实例化(评级 B+,工程有真实价值)
- **最终决策**(`99`):在 64M-198M 极小模型上,自循环如何取舍

---

## 📚 历史档案(已移入 references/)

7 份早期方案档案(v1-v4, AGI 异构, PRM 调研)已移入 [`../references/historical-architectures/`](../references/historical-architectures/)。这些文档作为"反面参考"和"决策史档案"保留,但不进入当前实现路径。

---

## 🚀 推荐阅读顺序

> **修订 2026-08-30**:原推荐路径基于"v4.5 主线"的执行导向;新定位下,推荐路径调整为"**先吸收教训,再理解当前主线**"。

| 读者 | 阅读顺序 |
|---|---|
| **AgenticMind 决策者** | [`00-iteration-timeline.md`](./00-iteration-timeline.md) §三 教训 1-8 → [`../../AGENTS.md`](../../AGENTS.md) §3 自循环机制 |
| **AgenticDSL 算法工程师** | [`00-iteration-timeline.md`](./00-iteration-timeline.md) §4.5(Pandey)+ [`06-metacognitive-closed-loop.md`](./06-metacognitive-closed-loop.md) §六(失效模式)→ [`../../AGENTS.md`](../../AGENTS.md) §4(训练路线) |
| **自循环概念研究者** | [`06-metacognitive-closed-loop.md`](./06-metacognitive-closed-loop.md) §五(状态机)+ [`../../AGENTS.md`](../../AGENTS.md) §3(自循环机制如何实现) |
| **历史架构师** | [`00-iteration-timeline.md`](./00-iteration-timeline.md) → [`99-final-recommendation.md`](./99-final-recommendation.md) |

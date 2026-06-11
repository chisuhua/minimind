# 推理架构 · 技术系列

> **目录定位**:`docs/architectures/` —— 与新目标(**自循环认知推理模型**)直接相关的 **4 份 DIRECT 架构文档**。
>
> **新目标相关度**:⭐⭐ **DIRECT(直接相关)**
>
> **纳入理由**:这 4 份文档分别对应"自循环认知推理模型"工程路径的**演进史、收敛点、闭环对应物、最终决策**。

---

## 📁 目录结构

| 文档 | 关联度 | 内容摘要 |
|---|---|---|
| ⭐⭐ [`00-iteration-timeline.md`](./00-iteration-timeline.md) | DIRECT | 7 轮推理架构迭代全景图(v1 → v4.6 → AGI → 元认知闭环),架构复杂度的演化与决策依据 |
| ⭐ [`04b-v4.5-and-v4.6.md`](./04b-v4.5-and-v4.6.md) | DIRECT | v4.5(Engine-Native Verification 务实收敛) + v4.6(GraphRAG + Agentic Memory),**唯一可落地的工程路径** |
| ⭐⭐ [`06-metacognitive-closed-loop.md`](./06-metacognitive-closed-loop.md) | DIRECT | **元认知闭环**(推理→置信度评估→低置信触发→知识检索/注入→重新推理),与新目标的"自循环闭环"概念**一一对应**(B+ 评级,窄场景 sweet spot 有效) |
| ⭐ [`99-final-recommendation.md`](./99-final-recommendation.md) | DIRECT | 最终推荐路线 + Kill Criteria,自循环认知推理模型的**工程决策依据** |

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

| 读者 | 阅读顺序 |
|---|---|
| **架构师 / 决策者** | [`00-iteration-timeline.md`](./00-iteration-timeline.md) → [`99-final-recommendation.md`](./99-final-recommendation.md) |
| **算法工程师** | [`04b-v4.5-and-v4.6.md`](./04b-v4.5-and-v4.6.md) → [`06-metacognitive-closed-loop.md`](./06-metacognitive-closed-loop.md) |
| **自循环概念研究者** | [`06-metacognitive-closed-loop.md`](./06-metacognitive-closed-loop.md) → [`99-final-recommendation.md`](./99-final-recommendation.md) |

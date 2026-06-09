# 推理系统架构迭代调研（v1 → v4.6 → AGI → 元认知闭环）

> **调研目的**：将 7 轮 LLM 推理方案迭代（v1 → v4.6 → AGI 异构架构 → 元认知闭环）的所有调研、审查、决策沉淀为可回溯文档，供后续回顾与继续讨论。
>
> **调研时间**：2026 年 6 月
> **立场**：**挑刺优先**——每条结论都列出证据与失败案例，不接受"看起来合理"的架构
> **核心问题**：在 MiniMind 这种 64M-198M 极小模型尺寸下，复杂推理系统架构的真实工程边界在哪里？

---

## 📋 目录索引

### 核心文件
| 文件 | 主题 | 关键结论 |
|---|---|---|
| [`00-iteration-timeline.md`](./00-iteration-timeline.md) | 7 轮迭代演进全景图 | 每个版本的"架构定位 → 致命缺陷 → 改进尝试 → 真实教训"四元组 |
| [`99-final-recommendation.md`](./99-final-recommendation.md) | 最终推荐路线 + Kill Criteria | MiniMind 项目应该走 v4.5 + 受限增量的务实路径 |

### 各版本详细审查
| 文件 | 版本 | 评级 | 核心问题 |
|---|---|---|---|
| [`01-v1-md-cds.md`](./01-v1-md-cds.md) | **v1 MDCDS** | D+ | 三模型解耦 + 4 维语义空间：装饰性创新，无训练信号支撑 |
| [`02-v2-prm-search.md`](./02-v2-prm-search.md) | **v2 修正方案** | C+ | PRM + Search + R1-Zero：组件选型错误（min(PRM)、标准 GRPO） |
| [`03-v3-pors.md`](./03-v3-pors.md) | **v3 PORS** | C+ | SOTA 精度对齐：6 项事实性错误 + 5 个系统性盲区 |
| [`04-v4-nacr.md`](./04-v4-nacr.md) | **v4 NACR** | C- | <2B 异构协作：6 项事实性错误 + 生产化先例为零 |
| [`04b-v4.5-and-v4.6.md`](./04b-v4.5-and-v4.6.md) | **v4.5 + v4.6** | **A-/B+** | 务实收敛：1.5B + Engine Verify + GraphRAG（最终推荐路径） |
| [`05-agi-heterogeneous.md`](./05-agi-heterogeneous.md) | **AGI 异构架构** | D+ | 6 组件异构：v1 的究极翻版，30% 真实 + 70% 装饰 |
| [`06-metacognitive-closed-loop.md`](./06-metacognitive-closed-loop.md) | **元认知闭环** | B+ | 推理→置信度→检索→重推理：窄场景优化器，6% 端到端增益 |

---

## 🗺️ 阅读路径建议

### 路径 A：快速决策（10 分钟）
1. 读 [`00-iteration-timeline.md`](./00-iteration-timeline.md) 的"7 轮迭代总览表"
2. 读 [`99-final-recommendation.md`](./99-final-recommendation.md) 的"一句话最终建议"
3. 完成

### 路径 B：完整理解（1 小时）
1. [`00-iteration-timeline.md`](./00-iteration-timeline.md) - 演进全景
2. [`01-v1-md-cds.md`](./01-v1-md-cds.md) - 起点问题
3. [`02-v2-prm-search.md`](./02-v2-prm-search.md) - 第一次纠错
4. [`04b-v4.5-and-v4.6.md`](./04b-v4.5-and-v4.6.md) - 务实收敛点
5. [`99-final-recommendation.md`](./99-final-recommendation.md) - 最终路线

### 路径 C：批判性精读（3 小时+）
按顺序读完所有 8 个文件，重点关注每个版本"为什么失败 / 为什么这次不同"。

---

## 🎯 一句话总结（每一版的核心教训）

> **从 v1 到 AGI 的整轮迭代反复证明：在 MiniMind 这种 64M-198M 极小模型上，"用架构换智能"是幻觉，"用工程红利换能力"才是正道。** 唯一可落地的路径是 v4.5（1.5B + Engine-Native Verification + Constrained Decoding + 三层 Safety）的务实收敛，GraphRAG/Agentic Memory 作为可选外挂层。任何超过此复杂度的方案都会在 6-12 个月内被工程现实击穿。

---

## 📚 关联文档

- [`../PRM_RESEARCH_REPORT.md`](../../PRM_RESEARCH_REPORT.md) - PRM 路线深度调研（v2 时做的）
- [`../reasoning-sota-critical-eval.md`](../reasoning-sota-critical-eval.md) - MiniMind v3 SOTA 推理方案诚实评估
- [`../inference-gap-analysis.md`](../inference-gap-analysis.md) - 推理加速技术 gap 分析（独立话题）

---

## 🔍 引用与证据

所有结论均基于以下类型的实证证据：

1. **顶会论文**（NeurIPS / ICLR / ICML / ACL / EMNLP）2024-2026
2. **官方技术报告**（DeepSeek-R1, Qwen2.5-Math, Math-Shepherd 等）
3. **第三方复现研究**（huggingface/open-r1, qijun/open-r1-reprod 等）
4. **生产框架文档**（vLLM, SGLang, TensorRT-LLM）
5. **业界失败案例报告**（Bing Chat, Notion AI 多模型路由回退）

任何"听起来合理但没有实证支撑"的架构建议都被明确标记为**红旗**。
# 📚 参考目录(References)

> **目录定位**:`docs/references/` —— 与"自循环认知推理模型"新目标**非直接相关**的 **40 份 INDIRECT + 6 份 PERIPHERAL** 文档,作为背景调研与方法学参考。
>
> **新目标相关度**:⭐ **INDIRECT / PERIPHERAL**(间接相关 / 边缘相关)
>
> **纳入理由**:这些文档提供"通用推理方法学"与"性能优化参考"——可能在适配新目标时需要查阅,但**不进入自循环闭环的直接实现路径**。

---

## 📁 子目录结构

### 🧠 [`methodology/`](./methodology/) —— 通用推理方法学(15 份)

调研 sub-1B 推理、推理蒸馏、推理 SOTA 评估等**方法学**文档。这些是**通用方法参考**,需要适配才能用于本项目的 AgenticDSL 自循环场景。

| 子目录/文件 | 内容 | 关联度 |
|---|---|---|
| [`methodology/reasoning-distillation-survey/`](./methodology/reasoning-distillation-survey/) | 推理轨迹蒸馏 + 分步迭代推理双方案调研(4 份) | INDIRECT |
| [`methodology/small-model-reasoning-survey/`](./methodology/small-model-reasoning-survey/) | 1B 小模型推理能力综合调研(9 份,含 R1-Distill + GRPO) | INDIRECT |
| [`methodology/reasoning-sota-critical-eval.md`](./methodology/reasoning-sota-critical-eval.md) | 推理 SOTA 批判性评估 | PERIPHERAL |

### ⚡ [`performance/`](./performance/) —— 性能优化参考(28 份)

调研推理加速、训练加速、性能 gap 分析等**性能优化**文档。这些是**通用 LLM 性能技术**,非"自循环"机制必需。

| 子目录/文件 | 内容 | 关联度 |
|---|---|---|
| [`performance/inference-acceleration/`](./performance/inference-acceleration/) | 18 项推理加速技术(其中 3 项 CRITICAL 已提取到 [`../inference-engine/`](../inference-engine/)) | INDIRECT |
| [`performance/training-acceleration/`](./performance/training-acceleration/) | 8 项训练加速技术 | INDIRECT |
| [`performance/inference-gap-analysis.md`](./performance/inference-gap-analysis.md) | 推理加速技术 gap 分析 | INDIRECT |
| [`performance/training-gap-analysis.md`](./performance/training-gap-analysis.md) | 训练优化审查 | INDIRECT |

### 📜 [`historical-architectures/`](./historical-architectures/) —— 历史架构档案(7 份)

minimind 推理架构 7 轮迭代中的 5 份早期方案 + 1 份 PRM 调研 + 1 份索引。这些文档作为**反面参考 / 决策史档案**保留,记录了"为什么 v4.5 是最终收敛点"。

| 子目录/文件 | 内容 | 关联度 |
|---|---|---|
| [`historical-architectures/01-v1-md-cds.md`](./historical-architectures/01-v1-md-cds.md) | v1 早期三模型解耦方案(D+,反面参考) | INDIRECT |
| [`historical-architectures/02-v2-prm-search.md`](./historical-architectures/02-v2-prm-search.md) | v2 PRM+Search 审查(范式正确但组件选型错) | INDIRECT |
| [`historical-architectures/03-v3-pors.md`](./historical-architectures/03-v3-pors.md) | v3 SOTA 精度对齐(6 项事实性错误需修正) | INDIRECT |
| [`historical-architectures/04-v4-nacr.md`](./historical-architectures/04-v4-nacr.md) | v4 <2B 异构协作(C-,含 Engine-Native 思路) | INDIRECT |
| [`historical-architectures/05-agi-heterogeneous.md`](./historical-architectures/05-agi-heterogeneous.md) | AGI 6 模块异构架构(D+,装饰性创新) | INDIRECT |
| [`historical-architectures/PRM_RESEARCH_REPORT.md`](./historical-architectures/PRM_RESEARCH_REPORT.md) | PRM 深度调研(6 类认知推理中"依赖推理"借鉴) | PERIPHERAL |

---

## 🎯 与新目标的关系

| 新目标组件 | references/ 中可能需要查阅的子目录 |
|---|---|
| **AgenticDSL 训练** | `methodology/reasoning-distillation-survey/`(轨迹蒸馏通用方法) + `methodology/small-model-reasoning-survey/`(1B 训练策略) |
| **自循环架构设计** | `historical-architectures/`(7 轮迭代教训) |
| **推理引擎支撑** | `performance/inference-acceleration/`(其余 15 项通用加速) |
| **训练范式选择** | `performance/training-acceleration/` + `methodology/small-model-reasoning-survey/02-training-strategy-survey.md` |
| **SOTA 对标** | `methodology/reasoning-sota-critical-eval.md` |

---

## 🚨 重要警示

- **3 项 CRITICAL 推理引擎** 已提取至 [`../inference-engine/`](../inference-engine/)——不要在 `performance/inference-acceleration/` 中寻找它们
- **4 份架构 DIRECT** 已提取至 [`../architectures/`](../architectures/)——不要在 `historical-architectures/` 中寻找它们
- **references/ 中的文档不应**作为"自循环"实现的首选参考,只在需要时查阅

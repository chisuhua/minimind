# 📚 Methodology 参考子目录

> **目录定位**:`docs/references/methodology/` —— 通用推理方法学调研,15 份文档。
>
> **新目标相关度**:⭐ INDIRECT(需要适配才能用于本项目)

---

## 📁 内容索引

| 子目录/文件 | 文档数 | 核心内容 | 何时查阅 |
|---|---|---|---|
| [`reasoning-distillation-survey/`](./reasoning-distillation-survey/) | 4 份 | 推理轨迹蒸馏 + 分步迭代推理双方案核查 | 设计 AgenticDSL SFT 数据管线时 |
| [`small-model-reasoning-survey/`](./small-model-reasoning-survey/) | 9 份 | 1B 小模型推理能力综合调研(数据/训练/架构/推理时扩展/循环模型/系统部署) | 选择 1B 训练策略 / 推理时扩展方法时 |
| [`reasoning-sota-critical-eval.md`](./reasoning-sota-critical-eval.md) | 1 份 | 推理 SOTA 批判性评估 | 评估自循环模型的 SOTA 对标时 |

---

## 🎯 重点子目录(对自循环概念相关)

- **`small-model-reasoning-survey/05-loop-model-deepdive.md`** — 循环深度模型专项(Ouro/RELAY/CogPO),**与"自循环"概念同名但实际是另一类研究**(CRV+LoopLM 组合无任何实证,**反面参考价值 > 直接借鉴价值**)
- **`small-model-reasoning-survey/04-inference-time-scaling-survey.md`** — 推理时扩展(BoN+PRM+Tool Use),对自循环"评估"环节有参考
- **`small-model-reasoning-survey/03-architecture-innovation-survey.md`** — 架构创新(Neuro-Symbolic PAL),对自循环中的"符号化推理"有方法学参考

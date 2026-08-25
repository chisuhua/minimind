# AGENTS.md — `docs/agenticdsl-training/`

> 训练配方文档层。代码实现归 `trainer/` / `model/` / `scripts/`；本文档目录只定义训练算法的"说什么"，不同步实现细节。

---

## OVERVIEW

AgenticDSL LLM 训练配方（TR-1/2/3，14-20 周），定义 9 阶段数据管线、6 阶段训练算法、HydraForgeBench 评估体系。

---

## WHERE TO LOOK

| 文档 | 主题 | 受影响代码路径 | 受影响指标 |
|---|---|---|---|
| `README.md` | 综述 + TR-1/2/3 路线图 + HydraForge 边界 | — | 里程碑时间表 |
| `01-training-data-pipeline.md` | 9 阶段 SFT 数据构造 | `scripts/sft/`、`dataset/` | 200K 样本配比、7 层课程 |
| `02-training-algorithms.md` | 6 阶段训练算法 | `trainer/`、`model/` | KL β、GRPO G/LR、clip ε |
| `03-inference-time-guarantees.md` | XGrammar-2 + Tree-sitter 推理约束 | `model/inference/` | 格式合规 >99% |
| `04-evaluation-benchmark.md` | HydraForgeBench 8 维度 | `eval/`、`scripts/benchmark/` | 8 项 eval 目标 |
| `05-risk-register.md` | 12 风险 + 防 Goodhart | — | format compliance 下限 95% |
| `06-vn001-alignment.md` | 与 HydraForge VN-001 对齐 | `agenticmind/extraction/` | 自举里程碑 |
| `07-vs-initial-analysis.md` | 与初步分析差异 | — | — |

---

## CROSS-REFERENCE TABLE

数值化参数（task mix / reward weights / special tokens / eval targets）**永不复制**，单一真源为 root AGENTS.md §12：

| 参数类型 | 真源 |
|---|---|
| TR-1/2/3 时长（4-6 / 4-6 / 6-8 周）| root AGENTS.md §12.1 |
| 200K SFT 样本配比（30/25/20/15/10）| root AGENTS.md §12.2 |
| 7 层课程（L1-L7）| root AGENTS.md §12.3 |
| 4 层验证器权重（0.4/0.2/0.3/0.1）| root AGENTS.md §12.4 |
| KL β=0.04、GRPO G=16-32、LR=1e-6、clip ε=0.2 | root AGENTS.md §12.5 |
| 11 special tokens | root AGENTS.md §12.6 |
| HydraForgeBench 8 维度目标 | root AGENTS.md §12.7 |
| 3 层基准集规模（50-100 / 500-1000 / 100-200）| root AGENTS.md §12.8 |

---

## DOC OWNERSHIP

| 文档 | 拥有（owns）| 绝不定义（never defines）|
|---|---|---|
| `README.md` | TR-1/2/3 阶段边界；HydraForge 上游边界 | 数值（引用 §12）|
| `01-training-data-pipeline.md` | SFT 数据构造流程；课程学习顺序 | 训练超参 |
| `02-training-algorithms.md` | 训练算法选择；RL 流程 | 数据集构造细节 |
| `03-inference-time-guarantees.md` | 推理时 grammar 约束；XGrammar 配置 | 训练数据格式 |
| `04-evaluation-benchmark.md` | 评估维度；基准集描述 | 训练算法 |
| `05-risk-register.md` | 风险条目；防 Goodhart 规则 | 具体实现 |
| `06-vn001-alignment.md` | VN-001 对齐声明 | 风险或数据 |
| `07-vs-initial-analysis.md` | 差异分析结论 | 实现或指标 |

---

## CONVENTIONS

1. **改训练代码 → 同步更新对应章节**：AGENTS.md §6.4 强制。`trainer/` 或 `model/` 改动 → 同步 `02-training-algorithms.md`；`scripts/sft/` 改动 → 同步 `01-training-data-pipeline.md`。
2. **新风险 → 记录到 `05-risk-register.md`**：不新增独立风险文档。
3. **数值冲突 → 以 root AGENTS.md §12 为准**：本目录文档相互冲突时，以 root §12 为准，在 `docs/README.md` 提交流程修订。
4. **本文档目录不放 Python 代码**：设计文档，纯文字描述。实现代码归 `trainer/` / `model/` / `scripts/`。

---

## ANTI-PATTERNS

- **NEVER 直接复制 root §12 数值到本目录文档**：引用即可（如"见 root AGENTS.md §12.2"），复制会导致 drift。
- **NEVER 在本目录放 Python 代码**：如需代码示例，放 `scripts/` 或 `trainer/`，在文档中用路径引用。
- **NEVER 用本目录文档定义训练超参数值**：超参数值定义在 root AGENTS.md §12，本目录只引用路径（如"KL β 见 root §12.5"）。

---

## NOTES

- **当前状态**：设计完成，TR-1 M0（Tokenizer round-trip test）待启动。`out/` 目录仅有 `minference_patterns.json`，无训练权重。
- **下一步**：调用 `writing-plans` skill 生成 TR-1 M0-M2 详细实施计划（M0 W1-2 / M1 W3-4 / M2 W5-6）。
- **协调依赖**：Special Token 注册（vocab surgery）归属 HydraForge 仓；EBNF grammar 编写依赖 HydraForge 的 AgenticDSL v3.10 规范。

# minimind认知推理模型 — 项目目标定义（Project Goal Definition）

> **文档定位**：本 README 定义 minimind **认知推理模型**项目的目标、范围、决策与技术路径。是后续所有设计、实现、评测的根本依据。
>
> **创建日期**：2026-06-09
> **brainstorming 来源**：与用户多轮对话 + ontology/neuro-symbolic学术调研
> **关联文档**：
> - [`reasoning-distillation-survey/`](./references/methodology/reasoning-distillation-survey/) —推理轨迹蒸馏 +迭代推理双方案调研（前序工作,现归入 `references/methodology/`）
> - [`small-model-reasoning-survey/`](./references/methodology/small-model-reasoning-survey/) —1B推理能力综合调研(现归入 `references/methodology/`)
> - [`architectures/`](./architectures/) — minimind推理架构 7 轮迭代中的 4 份 DIRECT 主线(00/04b/06/99),其余档案已移入 [`references/historical-architectures/`](./references/historical-architectures/)
> - [`reasoning-sota-critical-eval.md`](./references/methodology/reasoning-sota-critical-eval.md) —推理 SOTA批判性评估(现归入 `references/methodology/`)

---

## 🎯 一、一句话目标

> **用 <1B激活参数构造一个"认知推理模型"（Cognitive Reasoning Model），区别于 LLM 的 CoT推理，核心是基于"领域本体 +逻辑规则 +文档三元组 + AgenticDSL + 自然语言 +基础认知"六层知识形态栈，进行6 类认知推理（因果、依赖、相似、对象联系、语义关系、规则/时间）。模型知道自己不知道的知识，并能通过拒答、检索、学习提示三种机制处理知识缺口。**

---

## 🧠 二、核心定义

###2.1什么是「认知推理」（区别于 CoT推理）

|维度 | LLM CoT推理 |认知推理（本项目） |
|---|---|---|
| **知识基础** | 模型参数化的隐式知识 |显式 ontology +规则 + 三元组（可外部检索） |
| **推理类型** |自由语言多步推导 | 因果、依赖、相似、对象联系、语义、规则/时间6 类形式化推理 |
| **推理过程** | 自然语言 CoT链 |实体-关系图谱遍历 +逻辑规则 +外部求解器 |
| **可解释性** | 黑盒（语言叙述） | **可审计**（图谱路径 +求解器输出） |
| **知识更新** |需重训 | ontology/规则可独立更新 |
| **失败模式** |幻觉（自由生成错误） |拒答（超出 ontology范围时） |

###2.26 类认知推理（用户已确认全谱覆盖）

| # |推理类型 |形式化操作 | ontology 来源 |
|---|---|---|---|
|1 | **因果推理** | `cause(X, Y)` / `if X then Y` | ATOMIC, ConceptNet (Causes) |
|2 | **依赖推理** | `depends_on(X, Y)` / `requires(X, Y)` | 自建逻辑规则 |
|3 | **相似/相识推理** | `similarTo(X, Y)` / `acquaintanceOf(X, Y)` | ConceptNet (SimilarTo), WordNet |
|4 | **对象联系推理** | `partOf(X, Y)` / `hasProperty(X, Y)` | ConceptNet, 自建本体 |
|5 | **语义关系推理** | `isA(X, Y)` / `hasA(X, Y)` | WordNet, ConceptNet |
|6 | **规则/时间推理** | `before(X, Y)` / `rule(X, Y, Z)` | OWL-Time, SWRL规则 |

###2.3知识形态栈（6 层结构）

```
┌────────────────────────────────────────┐
│ L6:基础认知 (Basic Cognition) │ ←通用认知能力（如抽象、类比）
├────────────────────────────────────────┤
│ L5: 自然语言描述 (Natural Language) │ ←概念的自然语言解释
├────────────────────────────────────────┤
│ L4: AgenticDSL 语言 │ ←领域专用符号语言（如 Prolog DSL）
├────────────────────────────────────────┤
│ L3:文档三元组 (Document Triples) │ ← 从文档抽取的实体-关系
├────────────────────────────────────────┤
│ L2:逻辑规则 (Logic Rules) │ ← SWRL / Datalog形式化规则
├────────────────────────────────────────┤
│ L1:领域本体 (Domain Ontology) │ ← OWL/RDFS定义的类与实例
└────────────────────────────────────────┘
```

每层都**可独立更新**，**可单独训练**，**可独立评测**。

###2.4 「知道不知道」机制

|机制 |触发条件 | 实现方式 |
|---|---|---|
| **拒答** | 问题涉及 ontology 未覆盖的实体/关系 | 模型输出 `out_of_ontology` +缺口说明 |
| **检索** |缺口可通过扩展 ontology补齐 | Loreto序列化扩展 ontology → context注入 |
| **学习提示** |缺口需用户主动补充 | 输出「需要补充知识 X」+检索接口 |

**置信度校准**：每个推理输出附带校准后的 `confidence ∈ [0,1]`，用于触发上述机制。

---

## 📐 三、技术路径：分阶段演进

###3.1演进路线图

```
阶段1 (M1-M2,8 周)阶段2 (M3-M4,8 周)阶段3 (M5-M6,8 周)
方案 A: Logic-LM范式方案 C:核心层方案 C:完整态
 + GCR范式 + 内化 pretrain +置信度机制

目标:验证 sub-1B目标:真正"内化"目标:完整认知推理
 + ontology ontology 到参数框架
 +外部 solver + KG agent +拒答 +检索
 可跑通推理 +扩展层

成功标准:成功标准:成功标准:
≥60% on6 类推理 ≥50% 内化推理准确率7 项指标全部达标
```

###3.2阶段1（方案 A）— Logic-LM + GCR 双引擎

**目标**：验证 sub-1B + ontology +外部 solver 能跑通

**架构**：
```
用户问题
 ↓
0.5B LLM (符号化 + 本体引用)
 ↓
符号程序 +实体 +关系指针
 ↓
外部求解器 (Prover9/Z3/clingo)
 ↓
求解结果 +解释
 ↓
0.5B LLM (自然语言回复 +置信度)
```

**核心组件**：
- Base 模型：**Qwen2-0.5B**（已有 sub-1B实证）
- 本体库：ConceptNet5.5 + WordNet + ATOMIC 子集 + 自建 SWRL规则
-符号求解器：Prover9 + clingo + Z3
-训练范式：Logic-LM风格 SFT — `(NL 问题, Prolog/FOL 程序, solver 输出)`
- 数据生成：从 ontology axiom → Prolog事实 → GPT-4 生成 NL →求解器验证

**成功标准**：在6 类推理的500 测试集上 ≥60%准确率

###3.3阶段2（方案 C核心层）— Neuro-Symbolic 内化

**目标**：让模型"真正记住"核心 ontology（脱外部 solver）

**架构**：
```
[核心 ontology (50K axioms)]
 ↓ NeurASP/LTN风格 pretrain
 ↓ (logic satisfaction loss)
0.5B认知模型
 ↓
用户问题 →0.5B 直接推理（无需外部 solver）
```

**新增能力**：
- NeurASP风格 pretrain — ontology axioms → soft logic loss
- KG agent推理（ToG/PoG范式）
-评测 WebQSP/CWQ KGQA（参考 GCR0.5B baseline26.2）

**成功标准**：核心 ontology 内化后，"无 solver 内化推理"准确率 ≥50%

###3.4阶段3（方案 C完整态）—完整认知推理框架

**目标**：完整置信度机制 +拒答 +扩展层 + Temporal/Causal

**架构**：
```
用户问题
 ↓
0.5B认知模型
 ├──核心层（参数化）：NeurASP 内化
 ├──扩展层（检索）：Loreto序列化 context
 └──拒答层：置信度阈值决策
 ↓
自然语言回复 +置信度 + ontology引用
```

**新增能力**：
- DPO偏好优化（正确推理 vs错误推理 vs拒答）
-置信度校准训练
- OWL-Time + ATOMIC完整接入
- Loreto序列化扩展 ontology 进 context

**成功标准**：7 项评测指标全部达标

---

## 📊 四、评测体系（完整7 项指标）

| # |指标 |阶段1目标 |阶段2目标 |阶段3目标 |评测方法 |
|---|---|---|---|---|---|
|1 | **6 类推理准确率** | ≥60% | ≥70% | ≥80% |6 类各500 题 held-out |
|2 | **ontology忠实度** | ≥70% | ≥80% | ≥90% |推理引用 ontology实体/关系的比例 |
|3 | **拒答准确率** | ≥50% | ≥70% | ≥85% |未知问题拒答率 vs已知误答率 |
|4 | **零样本泛化** | — | ≥30% | ≥50% | 未训练类型问题准确率 |
|5 | **反例鲁棒性** | — | — | ≥60% | ontology 中加入错误实体时识别率 |
|6 | **跨领域迁移** | — | — | ≥40% | 新领域 ontology适配时间 <1 epoch |
|7 | **对齐率** | — | ≥60% | ≥70% | 与 GPT-4 在6 类推理上的判断一致率 |

**评测 benchmark清单**：
- Logic-LM5 个 benchmark（ProofWriter / PrOntoQA / FOLIO / LogicalDeduction / AR-LSAT）
- KGQA3 个 benchmark（WebQSP / CWQ / GrailQA）
- Temporal：CronKGQA / RE-Net 测试集
- Causal：ATOMIC / SCIE 测试集
-通用：CommonsenseQA / OpenBookQA

---

## 🛠️ 五、技术栈选型（已锁定）

###5.1核心组件

|类别 |选型 |理由 |
|---|---|---|
| **Base 模型** | Qwen2-0.5B | GCR已在0.5B验证 KG reasoning |
| **本体库** | ConceptNet5.5 + WordNet + ATOMIC + OWL-Time | 全谱6 类推理覆盖 |
| **求解器** | Prover9 (FOL) + clingo (ASP) + Z3 (SMT) | Logic-LM范式标准组件 |
| **本体序列化** | Loreto | token减少30%+，LLM-friendly |
| **本体嵌入** | OWL2Vec* | OWL ontology →稠密向量 |
| **训练范式** | Logic-LM + NeurASP + DPO |阶段递进 |
| **KG推理** | GCR + ToG + PoG | sub-1B友好 |
| **形式化接口** | NeurASP + LTN + DeepProbLog | ontology-aware训练 |

###5.2 minimind现有能力复用

| minimind现有组件 |复用方式 |
|---|---|
| `train_pretrain.py` |阶段2扩展 logic loss |
| `train_full_sft.py` |阶段1 Logic-LM SFT |
| `train_lora.py` |阶段1 LoRA 微调 |
| `train_dpo.py` |阶段3拒答偏好优化 |
| `model/model_minimind.py` | Qwen2-0.5B base适配 |
| `scripts/serve_openai_api.py` |推理服务部署 |

###5.3 与 minimind已有推理路径的关系

| minimind已有路径 | 与认知推理的关系 |
|---|---|
| **v4.5 推荐路径**（1.5B + Engine Verify + 三层 Safety） | v4.5 是通用路径；本项目是 ontology增强的**专科路径** |
| **R1-Distill + GRPO**（小模型推理 SOTA） | 本项目不依赖 R1蒸馏；用 Logic-LM范式实现结构化推理 |
| **Neuro-Symbolic / PAL**（minimind 已调研） | PAL范式作为阶段1 的备选验证 |

**重要**：本项目**不与 minimind现有路径竞争**，而是其**垂直延伸**（ontology推理的专科能力）。

---

## 🚨 六、风险与失败模式

###6.1阶段1（方案 A）风险

|风险 |触发条件 |应对 |
|---|---|---|
| DSL 生成准确率低 |0.5B 生成 Prolog/FOL语法错误率 >30% |① 增加 SFT 数据量②引入 verifier 后过滤③切到方案 B |
| Solver不可表达 | 自然语言问题无法形式化（如模糊语义） |①限定问题类型②引入 LLM fallback |
|求解延迟过高 | Solver 调用 >5 秒/问题 |①缓存常见推理②限制 solver 调用深度 |

###6.2阶段2（方案 C核心层）风险

|风险 |触发条件 |应对 |
|---|---|---|
| NeurASP训练不稳定 | logic loss收敛失败 |①降级为纯 SFT②简化 ontology规模 |
| 内化推理质量差 | 内化推理准确率 <30% |①退回方案 A（M1重新优化）② 增加 SFT 数据 |
| KG agent推理慢 | 多跳路径 >10跳 |①限制最大跳数②引入路径剪枝 |

###6.3阶段3（方案 C完整态）风险

|风险 |触发条件 |应对 |
|---|---|---|
|置信度校准差 |校准后 ECE >0.15 |① 增加校准数据②简化为二值拒答 |
| Loreto序列化不兼容 | 部分 OWL axiom 无法序列化 |①手动清洗 ontology② 仅序列化核心子集 |
| Temporal/Causal 无现成路径 | TKG sub-1B准确率 <30% |①降级为"标注可用"② 仅做 ontology 时间推理 |

---

## 🎓 七、学术差异化与发布价值

###7.1学术差异化

minimind认知推理模型将是：
- **首个公开的 sub-1B 本体论认知推理模型**（vs GCR 仅 KG推理、Logic-LM需大模型）
- **首个完整"核心内化 +扩展检索 +置信度拒答"sub-1B框架**
- **首个 sub-1B Temporal/Causal形式化推理实证**（如成功）

###7.2 发布价值

- minimind现有评估（CEval/CMMLU）仅验证通用能力
- 本项目扩展 minimind能力维度（认知推理）
- 与现有 minimind 数据格式兼容（JSONL conversations）
- HuggingFace 模型权重可直接复用

---

## 📋 八、与 minimind现有项目边界

###8.1 不在本项目范围内

- minimind现有架构（Dense/MoE）的扩展（已在 v3-moe 完成）
- minimind通用能力提升（已有训练链路）
- minimind视觉/多模态（minimind-V、minimind-O独立项目）

###8.2 在本项目范围内

- minimind-3 (64M) 或 minimind-3-moe (198M-A64M) 上的 ontology推理增强
- 基于 Qwen2-0.5B起步（更大 base 可后续）
-6 类认知推理的完整覆盖
-拒答 +置信度 +检索机制

###8.3 与现有25+ 技术方向的关系

| minimind 用户原始方向 | 与本项目关系 |
|---|---|
| CogPO / CRV |阶段3 中作为置信度机制的可选组件 |
| Self-RAG |阶段3 中作为检索机制的可选组件 |
| Tina / GRPO |阶段3 DPO阶段的可选起点 |
| Ouro / LoopLM | 与本项目正交（非推理结构改造） |
| MoE Upcycling |阶段3 可考虑 base 模型改造 |

---

## ✅ 九、决策记录（已完成）

| # |决策项 | 用户选择 |锁定日期 |
|---|---|---|---|
|1 | 项目目标 | <1B认知推理模型 |2026-06-09 |
|2 |推理区别边界 | 基于本体论的形式化推理 |2026-06-09 |
|3 |推理范围 | 全谱6 类 +领域聚焦 |2026-06-09 |
|4 |知识形态栈 |6 层（本体+规则+三元组+DSL+NL+基础认知） |2026-06-09 |
|5 | 内化路径 |混合架构（核心微调 +动态检索） |2026-06-09 |
|6 | 「不知道」机制 |拒答 +检索 +置信度校准 |2026-06-09 |
|7 | 数据生成起点 | 本体 +文档 + 问题（混合） |2026-06-09 |
|8 |评测体系 |完整7 项指标 |2026-06-09 |
|9 | MVP领域 |通用常识 +逻辑推理 |2026-06-09 |
|10 | 主路径 | 分阶段演进（A → C） |2026-06-09 |
|11 |阶段1范围 |6 类全谱覆盖 |2026-06-09 |
|12 |失败模式预案 |3阶段各自有切回方案 |2026-06-09 |

---

## 📚 十、参考学术工作（sub-1B实证支撑）

###强证据（已验证 sub-1B）

| 工作 |关键数据 |链接 |
|---|---|---|
| **GCR** (ICML2025) | Qwen2-0.5B WebQSP26.2 baseline | https://arxiv.org/abs/2410.13080 |
| **ToG** (ICLR2024) |声称"小 LLM + KG 可超 GPT-4" | https://arxiv.org/abs/2307.07697 |
| **Proof of Thought** (NeurIPS2024) | PrOntoQA100%, ProofWriter98.96% | https://github.com/DebarghaG/proofofthought |
| **Logic-LM** (EMNLP2023) |5 个 benchmark 平均 +39.2% | https://arxiv.org/abs/2305.12295 |
| **Phi-1.5** (Microsoft2023) |1.3B ≈5× 大模型推理 | https://arxiv.org/abs/2309.05463 |

### 中证据（接口成熟，需 sub-1B替换验证）

| 工作 |用途 |链接 |
|---|---|---|
| **NeurASP** | NN + ASP规则联合训练 | https://github.com/azreasoners/NeurASP |
| **LTN** |逻辑公式 → 可微 loss | https://github.com/logictensornetworks/LTNtorch |
| **Loreto** | OWL/RDF → LLM-token优化 | https://github.com/rorevello/Loreto |
| **OWL2Vec*** | OWL ontology → embedding | https://github.com/KRR-Oxford/OWL2Vec-Star |
| **PoG** (WWW2025) |3阶段多跳 KGQA | https://github.com/SteveTANTAN/PoG |
| **GreaseLM** (ICLR2022) | BERT + KG GNN融合 | https://github.com/snap-stanford/GreaseLM |

###弱证据（降级为 Future Work）

| 工作 |限制 |链接 |
|---|---|---|
| **RE-Net / CENET** | TKG未来外推，sub-1B 未验证 | https://github.com/INK-USC/RE-Net |
| **SCIE / Axiomatic** | Causal reasoning，sub-1B 无实证 | https://github.com/dsubuntu/SCIE |

---

## 🧬十一、补充调研：KG-Prolog进化闭环 + 四重置信度过滤 + 因果推理

> **来源**：用户后续提供了3段补充材料（KG→Prolog闭环、四重置信度过滤、因果推理），由3 个并行 librarian agent 系统核实后整合。
> **核实日期**：2026-06-09
> **核心结论**：用户描述的3 个补充方向**整体可行**，但需要按学术现实**重新设计**部分实现路径。本节作为现有目标定义的补充维度。

###11.1 KG→Prolog进化闭环（✅真实可行，但需重新设计）

**用户提出**：知识图谱 →逻辑（Prolog）提取 → LLM配合 Prolog推理 →结论反哺 KG 的动态闭环系统。

**核实结论**：ChatRule框架 **完全真实存在**（[RManLuo/ChatRule](https://github.com/RManLuo/ChatRule)78 stars · [arXiv:2309.01538](https://arxiv.org/abs/2309.01538)）。其架构为：

```
KG路径采样 (BFS) → LLM 生成 Prolog规则 →置信度排序 →推理前向链
```

**⚠️关键澄清：用户描述的"四重置信度过滤"与 ChatRule实际机制有偏差**

| 用户描述 | ChatRule实际机制 |
|---|---|
| "语义阈值" | 不存在 |
| "LLM 自反思" | 部分存在（`--valid_clean` flag，但默认关闭） |
| "Prolog验证" | 不存在（推理用矩阵乘法，无 Prolog求解器） |
| "多模型一致性" | 仅作对比实验（不是 voting filter） |

**ChatRule真实置信度机制**：四重指标 `support / coverage / confidence / PCA confidence`（[rank_rule.py L30-L59](https://github.com/RManLuo/ChatRule/blob/02d45348819d2323ee1ed62314da6bdf460491ad/rank_rule.py#L30-L59)），全部基于 KG已有事实的统计估计。

**KG进化闭环现状**：
- **ChatRule** 是单向（KG→规则），不更新 KG
- **Graphiti**（[getzep/graphiti](https://github.com/getzep/graphiti)，arXiv2501.13956）是反向（事实→KG），不涉及规则挖掘
- **NeurASP / DeepProbLog / LTN**都不直接支持 KG进化
- **用户描述的"KG→Prolog→KG进化闭环"没有现成框架**

**sub-1B 可行性矩阵**：
| 技术 | <1B 可行？ |
|---|---|
| KG路径采样 → Prolog规则 | ✅ 可行（ChatRule 已支持7B+ LLM） |
|置信度评分（4 项） | ✅ 完全可行（纯统计） |
| KG 完成推理（前向链） | ✅ 完全可行（矩阵乘法） |
| KG 自动更新（进化闭环） | ❌ 无主流方案 |
| Self-Consistency 多模型投票 | ❌ 小模型难以稳定 |

**对本项目的建议**：

**新增阶段4（M7-M8，可选）：KG-Prolog进化闭环**
- 使用 ChatRule7B 模型作为"规则工厂"（不替换0.5B）
-阶段4 通过 Logic-LM SFT 让0.5B 模型**学会**生成稳定规则
- KG 更新机制：自主实现（参考 Graphiti 的时序双跟踪设计）
- 不依赖 ChatRule 的统计置信度作为"四阶段过滤"——自主设计多阶段机制

**新增14 项决策**：

| # |决策项 | 用户选择（补充） |
|---|---|---|
|14 | KG-Prolog进化闭环 | ✅ 作为可选阶段4补充 |
|15 | 四重置信度过滤 | ⚠️重新设计：自主实现（参考 ChatRule4 项统计置信度 + Lin2023 semantic dispersion + FRODO 因果奖励） |
|16 | KG 更新机制 | ⚠️自主实现（参考 Graphiti 时序双跟踪，无现成框架） |

###11.2 四重置信度过滤机制（⚠️学术基础扎实，但需重新设计）

**用户提出**：4 层过滤（语义阈值 + LLM 自反思 + Prolog验证 + 多模型一致性）确保 KG 安全回写。

**核实结论**：每一重都有学术基础，但**没有任何论文把这四重组合使用作为 KG 回写过滤**。

| 重 |学术基础 |关键论文 | sub-1B 可行性 |
|---|---|---|---|
|①语义阈值 | ✅ | [Lin2023 Generating with Confidence](https://arxiv.org/abs/2305.19187)（TMLR2024）、[Microsoft Semantic-Kernel0.85 demo](https://github.com/microsoft/semantic-kernel/blob/main/dotnet/samples/Demos/QualityCheck/QualityCheckWithFilters/Program.cs#L25) | ✅ 完全可行（用小 embedding 模型） |
|② LLM 自反思 | ✅ | [Self-Refine](https://arxiv.org/abs/2303.17651)、[Reflexion](https://arxiv.org/abs/2303.11366)、[Constitutional AI](https://arxiv.org/abs/2212.08073) | ⚠️有限可行（需 [FRODO](https://arxiv.org/abs/2402.13950)专门训练） |
|③ Prolog符号验证 | ✅ | [Logic-LM](https://arxiv.org/abs/2305.12295)、[DeepProbLog](https://github.com/ML-KULeuven/deepproblog)、[Selection-Inference](https://arxiv.org/abs/2205.09712) | ⚠️领域受限可行（sub-1B 不擅长严格格式化输出） |
|④ 多模型一致性 | ✅ | [Self-Consistency](https://arxiv.org/abs/2203.11171) | ❌ 基本不可行（voter 高相关） |

**⚠️重要风险**：
- **LLM 自反思在 sub-1B 上是已知失败模式**（[Reversal Curse, arXiv:2309.12288](https://arxiv.org/abs/2309.12288)：GPT-4 A→B79%正确，B→A 仅33%；FRODO：sub-1B self-critique经常 hallucinates 比原始更糟）
- **0.75/0.85阈值没有学术依据**（MS Semantic-Kernel0.85 是 demo注释硬编码示例值；0.75 在 LlamaIndex 等 RAG实践中零散使用，无系统 ablation）
- **必须** 在开发集上做 calibration（建议走 ECE / AUROC评估）

**对本项目的建议**：

```
KG 回写流程（重新设计后）：
1.0.5B LLM 生成候选三元组 (subject, predicate, object)
2. 【❶语义阈值】embedding相似度 vs已有 KG（≥校准阈值）
3. 【❸ Prolog符号验证】验证 schema 与领域 axioms（开发者写好，不让 LLM 生成 Prolog 子句）
4. 【❹多次采样一致性】N=5 次采样 + Lin2023 semantic dispersion < ε校准阈值
5. 【❷形式化自检】用固定 prompt template 检查 schema 合规性（替代"自反思"）
```

**关键改造**：
- ❌ **不** 让 sub-1B 模型做 self-critique（Reversal Curse / FRODO 已证失败）
- ✅ 用 **外部7B裁判模型**（Qwen2.5-7B / Llama-3-8B）做 self-consistency check
- ✅ 或用 **同模型多次采样 + Lin2023 semantic dispersion**（成本低，证据充足）

###11.3 因果推理3方向（⚠️ 部分真实，部分虚构，部分需重新设计）

**用户提出**：因果推理3 个方向（基于因果发现的 KG补全、反事实推理 KG验证、Causal CoT 可解释性）。

**核实结论**：

| 用户引用 |真实性 | 出处 |
|---|---|---|
| **CausalKG** | ✅真实 | [Jaimini & Sheth, IEEE Internet Computing2022, arXiv:2201.03647](https://arxiv.org/abs/2201.03647)（无开源代码） |
| **《Causality for Large Language Models》**（2024综述） | ✅真实 | [Wu, Kuang, Zhu et al., arXiv:2410.15319](https://arxiv.org/abs/2410.15319) |
| **PC / FCI / NOTEARS / DECI 算法** | ✅全部真实 | NOTEARS [arXiv:1803.01422](https://arxiv.org/abs/1803.01422)、DECI [arXiv:2202.02195](https://arxiv.org/abs/2202.02195) |
| **CausalBERT** | ✅双重存在 | [arXiv:2012.05453](https://arxiv.org/abs/2012.05453) + [arXiv:2107.09852](https://arxiv.org/abs/2107.09852) |
| **"Discovering Causal Relations in Knowledge Graphs"** | ❌精确标题不存在 |领域内有相近真实工作（如 TC-GAT [arXiv:2304.10706](https://arxiv.org/abs/2304.10706)） |
| **"Can Large Language Models Capture Human Causal Reasoning?"** | ❌精确标题不存在 | 最接近：[Unveiling Causal Reasoning in LLMs, arXiv:2506.21215](https://arxiv.org/abs/2506.21215) |
| **"Causal Chain-of-Thought Prompting"** | ⚠️ 是方法范式名（非单一论文） |实证工作：CausalCoT（[arXiv:2312.04350](https://arxiv.org/abs/2312.04350)）、CDCR-SFT（[arXiv:2508.12495](https://arxiv.org/abs/2508.12495)） |

**3 个方向的可行性评估**：

**方向① 基于因果发现的 KG补全 +规则挖掘** → ✅ **强烈建议**
- PC/FCI/NOTEARS/DECI全部已实现
- [FinCARE arXiv:2510.20221](https://arxiv.org/abs/2510.20221) 已示范 KG + PC/GES/NOTEARS + LLM 三位一体范式
-适配方案：minimind 仅作 LLM 接口层；因果发现由 PC/NOTEARS 外挂

**方向② 反事实推理增强 KG验证** → ⚠️ **谨慎补充**
- [CausalKG](https://arxiv.org/abs/2201.03647)、[Ca2KG arXiv:2601.09241](https://arxiv.org/abs/2601.09241)、[COULDD arXiv:2403.06936](https://arxiv.org/abs/2403.06936)真实但依赖 KGE 或中等 LLM
- sub-1B **不擅长**直接生成反事实（Caliper实证 sub-4B 主要靠 lexical anchor）
-替代方案：让0.5B 模型生成候选反事实，由 PC/NOTEARS 输出 DAG符号验证

**方向③ Causal CoT 可解释性推理** → ⚠️ **可选补充**
- [CausalCoT](https://arxiv.org/abs/2312.04350)、[CDCR-SFT arXiv:2508.12495](https://arxiv.org/abs/2508.12495) 是真实 SOTA实证
- minimind 可作为"学生模型"，通过知识蒸馏从大 LLM 学 Causal CoT推理链
- ⚠️警惕 "Causal Tongue-Tie"（[arXiv:2605.25891](https://arxiv.org/abs/2605.25891)）：模型内部编码了因果方向但 Yes/No 输出无法表达
- ⚠️ CDCR-SFT 在8B 量级稳定，sub-1B需大量合成数据

**sub-1B 因果推理 SOTA实证（关键发现）**：

| 工作 |规模 |关键结论 |
|---|---|---|
| **Generating Effective CoT Traces for Mitigating Causal Hallucination** | **≤1.5B** | **唯一专门研究 sub-1.5B 因果幻觉的工作**，可直接迁移 |
| **TCAR-Gen** | GPT-OSS20B → **TinyLlama1.1B** | TinyLlama 生成质量显著下降，检索端稳健 |
| **Caliper** | **3.8B-14B** | sub-4B 主要靠 lexical anchor，结构推理能力有限 |

**对本项目的建议**：

**新增17 项决策**：

| # |决策项 | 用户选择（补充） |
|---|---|---|
|17 | 因果推理 | ✅ 作为可选阶段5探索（外挂 PC/NOTEARS + CausalKG schema） |
|18 | KG→Prolog进化闭环 | ✅ 作为可选阶段4 |
|19 | Causal CoT 微调 | ⚠️需 SFT 数据蒸馏（参考 CDCR-SFT） |
|20 | Tongue-Tie风险 | ⚠️避免 Yes/No表达，使用 CoT 输出因果路径 |

###11.4整合后的完整阶段路线（含补充）

```
原路线（1-3阶段）：
阶段1 (M1-M2):方案 A — Logic-LM + GCR 双引擎
阶段2 (M3-M4):方案 C核心层 — NeurASP 内化
阶段3 (M5-M6):方案 C完整态 —置信度 +拒答 +扩展层

补充路线（4-5阶段，可选）：
阶段4 (M7-M8): KG-Prolog进化闭环（ChatRule7B规则工厂 + Graphiti 时序演化）
阶段5 (M9-M10): 因果推理外挂（PC/NOTEARS + CausalKG + Causal CoT SFT）
```

###11.5补充调研的核心仓库与论文（可直接引用）

**KG-Prolog闭环**：
| 工作 |链接 |
|---|---|
| **ChatRule官方** | [github.com/RManLuo/ChatRule @02d4534](https://github.com/RManLuo/ChatRule) |
| ChatRule论文 | [arXiv:2309.01538](https://arxiv.org/abs/2309.01538) |
| **Graphiti**（KG进化） | [github.com/getzep/graphiti](https://github.com/getzep/graphiti) |
| **RNNLogic**（KG规则挖掘 SOTA） | [github.com/DeepGraphLearning/RNNLogic](https://github.com/DeepGraphLearning/RNNLogic) |

**四重置信度过滤**：
| 工作 |链接 |
|---|---|
| **Lin2023 Generating with Confidence** | [arXiv:2305.19187](https://arxiv.org/abs/2305.19187)（TMLR2024） |
| **Self-Refine** | [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)（NeurIPS2023） |
| **Reflexion** | [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)（NeurIPS2023） |
| **Constitutional AI** | [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) |
| **Self-Consistency** | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)（ICLR2023） |
| **Logic-LM** | [arXiv:2305.12295](https://arxiv.org/abs/2305.12295)（NeurIPS2023） |
| **Selection-Inference** | [arXiv:2205.09712](https://arxiv.org/abs/2205.09712)（ICLR2023） |
| **FRODO**（sub-1B 因果奖励） | [arXiv:2402.13950](https://arxiv.org/abs/2402.13950)（EMNLP2024 Findings） |
| **Reversal Curse**（LLM 自反思反驳） | [arXiv:2309.12288](https://arxiv.org/abs/2309.12288) |

**因果推理**：
| 工作 |链接 |
|---|---|
| **CausalKG** | [arXiv:2201.03647](https://arxiv.org/abs/2201.03647)（IEEE Internet Computing2022） |
| **Causality for Large Language Models**（综述） | [arXiv:2410.15319](https://arxiv.org/abs/2410.15319) |
| **NOTEARS** | [arXiv:1803.01422](https://arxiv.org/abs/1803.01422) |
| **DECI** | [arXiv:2202.02195](https://arxiv.org/abs/2202.02195) |
| **CausalCoT**（CLadder） | [arXiv:2312.04350](https://arxiv.org/abs/2312.04350) |
| **CDCR-SFT** | [arXiv:2508.12495](https://arxiv.org/abs/2508.12495) |
| **Generating Effective CoT Traces for ≤1.5B** | [arXiv:2604.12748](https://arxiv.org/abs/2604.12748)（**最相关**） |
| **FinCARE**（KG + PC/NOTEARS） | [arXiv:2510.20221](https://arxiv.org/abs/2510.20221) |
| **WikiCausal** | [arXiv:2409.00331](https://arxiv.org/abs/2409.00331) |
| **COULDD**（反事实 KGE） | [arXiv:2403.06936](https://arxiv.org/abs/2403.06936) |

---

## 📖十二、阅读建议

- **想快速理解目标**：读第一、二节（一句话目标 +核心定义）
- **想理解技术路径**：读第三节（分阶段演进）
- **想理解评测体系**：读第四节（7 项指标）
- **想理解技术栈**：读第五节
- **想理解风险**：读第六节
- **想理解学术价值**：读第七节
- **想理解项目边界**：读第八节
- **想了解补充维度**（KG-Prolog闭环 + 四重过滤 + 因果）：读第十一节

---

## 🔗后续行动

1. ✅ **本 README 已定义项目目标与范围**（本文件）
2. **下一步**：brainstorming skill完成后调用 `writing-plans` skill 生成详细实施计划
3. **再下一步**：按计划逐步执行阶段1（M1-M2）

---

> **核心承诺**：本项目目标严格对齐 minimind "大道至简"哲学 —— **用最小的模型 + 最严谨的形式化推理**，**实现"知道自己不知道"的认知推理能力**，**而不是用更大的模型堆叠能力**。

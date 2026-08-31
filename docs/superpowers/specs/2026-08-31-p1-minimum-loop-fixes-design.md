# P1 最小闭环实验修复设计

> **设计 ID**: P1-FIX-2026-08-31
> **生成日期**: 2026-08-31
> **状态**: 草案 v0.3(Metis 评审整改完成)
> **作者**: Sisyphus(AI 助手)+ 用户协同决策 + Metis 评审
> **关联文档**:
> - [`docs/agenticmemory_training/08c-p1-minimum-loop.md`](../../agenticmemory_training/08c-p1-minimum-loop.md) — 被修复文档
> - [`docs/agenticmemory_training/08a-capacity-gap-design.md`](../../agenticmemory_training/08a-capacity-gap-design.md) — D-10 决策依据
> - [`AGENTS.md`](../../../AGENTS.md) §6.5 — 架构与训练工程解耦原则
> - [`AGENTS.md`](../../../AGENTS.md) §12.10 F-04 — 双 Schema 统一消费决策(🔴-1 待同步修订模型选型行)
> - [`AGENTS.md`](../../../AGENTS.md) §12.10 F-06 — 架构与训练工程解耦 + 记忆优先决策

---

## 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-31 | 初版,基于 brainstorming 4 轮澄清问题 + 2 方案选择 + Oracle 6 项补充建议 | Sisyphus(AI 助手)+ 用户 |
| **v0.2** | **2026-08-31** | **新增 P1-0 PoC 阶段**:在 P1-1 之前加入基础设施验证(OpenCode + SKILL.md + Qwen3.5-0.8B 推理服务闭环),1-2 天工作量,删除无新决策项 | **Sisyphus(AI 助手)+ 用户(协同决策)** |
| **v0.3** | **2026-08-31** | **Metis 评审整改**:4 阻塞级 + 6 重要级 + 4 建议级修订。①🔴-1 声明并同步修订 AGENTS.md F-04 模型选型行;②🔴-2 PoC-2 补"模型不可用"分支 + D-A fallback 链;③🔴-3 P1-2 加 Kimi-K3 第二标注者计算 Krippendorff α;④🔴-4 新增 2 脚本(eval_zero_shot/eval_random_label)纳入代码变更清单并修正工作量;⑤🟡-1 阈值分字段;⑥🟡-2 定义"其他字段"集合;⑦🟡-4 公开集腿 A 保留;⑧🟡-5 PoC-1 路径修正;⑨🟡-6 时间表矛盾修正;⑩🟢 命名/成本/字段登记修订 | **Sisyphus(AI 助手)+ Metis 评审 + 用户(协同决策)** |

---

## 1. 背景与问题陈述

**背景**:P1 最小闭环实验 [`docs/agenticmemory_training/08c-p1-minimum-loop.md`](../../agenticmemory_training/08c-p1-minimum-loop.md) 设计完整(4 步骤 / 2 周 / $50 / 1 GPU-hour)但有 3 项方法论缺陷(详见上一轮审查报告)。本 spec 对 P1 启动前必须修复的 3 项 🔴 + 3 项 🟡 补全做完整修复。

### 1.1 3 项 🔴 必须修复的问题(按严重性排序)

| 严重性 | 问题 | 描述 |
|---|---|---|
| 🔴 方法论缺陷 | **同教师偏置** | P1-1 / P1-2 都用 GPT-4o,模型学到的不是"通用抽取能力"而是"GPT-4o 标注风格" |
| 🔴 可解释性问题 | **无 baseline 对比** | LoRA 后 F1 = 0.80 是真实学习还是 base 本身就能达到 0.80?无法判断 |
| 🔴 交叉决策问题 | **基模型 fallback 冲突** | Qwen3-0.6B 未发布,fallback Qwen2.5-0.5B 与 A-01 教师(DeepSeek)跨系列违反 F-04 D-10 |

**优先级排序依据**:方法论缺陷 > 可解释性 > 交叉决策——方法论缺陷影响"P1 验证有效性"(实验结果不可信);可解释性问题影响"我们能否理解结果"(结果不可解读);交叉决策问题影响"决策一致性"(可能违反其他决策)。

### 1.2 3 项 🟡 补全缺口

| 严重性 | 问题 | 描述 |
|---|---|---|
| 🟡 完整性 | **通过标准过于单一** | 当前只看 intent.primary F1,不能反映"记忆引擎综合能力" |
| 🟡 完整性 | **失败模式诊断缺失** | 若 P1 失败,无法选择正确的降级路径(P1 改设计 vs 换模型) |
| 🟡 完整性 | **下游衔接机制缺失** | findings_v0.md 没有"对 08 蒸馏管线的具体输入"章节,P1 → 08 无反馈 |

### 1.3 P1 的战略定位

P1 不是孤立实验,而是 **AGENTS.md F-06 §6.5.2 "记忆引擎 DoD 抽取层定义"的载体**。P1 完成时,需在 [`docs/agenticmemory/01-memory-model.md`](../../agenticmemory/01-memory-model.md) §1 显式定义"抽取层 DoD",为下游 08 蒸馏管线提供准入门槛。

### 1.4 优先级:3 项 🔴 > 3 项 🟡

本 spec 只修复 3 项 🔴 方法论缺陷 + 3 项 🟡 补全。
**P2 实验**(memory_extract 验证 + 推理无损 + irr_estimate 校准)**不在本 spec 范围**,需后续专项 spec。

---

## 2. 设计目标与不在范围内

### 2.1 设计目标(6 项,SMART)

| # | 目标 | 度量 |
|---|---|---|
| 1 | **基模型与教师组合不违反已锁定决策** | D-A 对 F-04 模型选型行的修订**显式登记**至 AGENTS.md §12.10 F-04;D-10(P1 范围内不适用)不豁免 |
| 2 | **消除教师偏置** | P1-1 合成教师(Kimi-K3) ≠ P1-2 标注教师(DeepSeek),且架构/厂商差异最大化 |
| 3 | **建立 baseline 参照** | LoRA 提升幅度可量化(zero-shot vs LoRA,LoRA vs random-label),且**按字段分阈值判定** |
| 4 | **多字段联合判定** | P1 通过标准从单字段改为多字段联合(含"其他字段"集合显式定义),反映记忆引擎综合能力 |
| 5 | **建立 P1 → 08 衔接机制** | findings_v0.md 必须包含"对 08 蒸馏管线的具体输入"章节 |
| 6 | **IRR / Krippendorff α 可计算** | P1-2 引入 Kimi-K3 第二标注者(~50 条子集),使类别 A 判定与 IRR 建议有测量来源 |

### 2.2 不在范围内(9 项)

1. ❌ 修改 [`docs/agenticmemory/01-memory-model.md`](../../agenticmemory/01-memory-model.md) §1 记忆引擎核心能力定义(B/A ≥ 0.98 推理无损不变)
2. ❌ 修改 [`docs/agenticmemory_training/08a-capacity-gap-design.md`](../../agenticmemory_training/08a-capacity-gap-design.md) 的 Capacity Gap 机制
3. ❌ 修改 [`docs/agenticmemory_training/08d-wiki-dag-construction.md`](../../agenticmemory_training/08d-wiki-dag-construction.md) Wiki DAG 8 字段契约
4. ❌ 启动 08 蒸馏管线本身的实施(本 spec 只让 P1 完成时为 08 准备好输入)
5. ❌ 豁免 F-04 D-10(P1 范围内不适用,无需豁免;08 蒸馏管线需 F-07 决策)
6. ❌ 修改 F-04 A-01 教师=DeepSeek 决策(标注教师保持 DeepSeek,仅合成教师为 Kimi-K3)
7. ❌ 修改 P1-5 失败诊断步骤本身(只定义步骤内容,失败时备用)
8. ❌ 启动 F-07 决策(本 spec 只标记需要后续决策)
9. ❌ 变更 08a 附录 A 探针选型(08 蒸馏管线的 L1/L3 探针仍锁定 Qwen3.5 系列设计)

---

## 3. 核心决策摘要

### 3.1 核心决策(5 项)

| ID | 决策项 | 选择 | 取代 | 关联 |
|---|---|---|---|---|
| **D-A** | Base 模型 | **Qwen3.5-0.8B**(fallback: Qwen3.5-1.5B 或更大) | 08c A-05 的 Qwen3-0.6B **+ AGENTS.md F-04 模型选型行(需同步修订)** | sub-1B 目标 + 🔴-1/🔴-2 治理 |
| **D-B** | 标注教师 | **DeepSeek V4 Flash**(沿用 08a A-01) | - | 中文场景标注优 + 结构化输出 |
| **D-C** | 合成教师 | **Kimi-K3** | 08c 默认 GPT-4o | 跨教师(架构 MoE 16 vs DeepSeek mHC) |
| **D-D** | Baseline 对照 | **zero-shot + random-label**(按字段分阈值) | 08c 无 baseline | 区分真实学习 vs 模式映射 |
| **D-E** | P1 通过标准 | **4 字段联合判定**(含"其他字段"集合定义) | 08c 单字段 F1 | 反映记忆引擎综合能力 |

### 3.2 修复项(5 项)

| ID | 修复 | 来源(原 08c 章节) | 影响范围 | 实施成本 |
|---|---|---|---|---|
| **R-1** | base 切换 Qwen3.5-0.8B + **F-04 修订登记** | 08c §2.5 + §6.2 + AGENTS.md F-04 | 配置 + 显存 + 治理文档 | 10 分钟(下载 + F-04 修订)|
| **R-2** | 跨教师:Kimi-K3 合成 + DeepSeek 标注 + **Kimi-K3 第二标注者(IRR)** | 08c §3.2 + §4.2 | 数据流 + API 密钥 + 子集二次标注 | +$0.08 Kimi API(含 IRR 子集)|
| **R-3** | 新增 P1-4-pre:zero-shot + random-label 对照(**新增 2 脚本**) | 08c 新增 §6.2a | **新增代码**:`eval_zero_shot.py` + `eval_random_label.py`(复用 eval_f1.py 逻辑) | **1-2 人日(脚本开发)** |
| **R-4** | P1 通过标准:多字段联合判定(分字段阈值 + 其他字段定义) | 08c §6.3 修订 | 决策阈值 | 工作量 +0 |
| **R-5** | 新增 P1-5:失败模式诊断(条件性,含 IRR 测量)+ findings 衔接 | 08c 新增 §6.4 + §9 | 工作量 +2.5 小时 | 免费 |

### 3.3 F-04 修订登记 + D-10 兼容性分析(Metis 🔴-1 整改)

**结论**:**D-A 需显式修订 AGENTS.md §12.10 F-04 的"模型选型"行;D-10 本身(P1 范围内)不豁免**。

**F-04 修订登记(必须同步执行)**:
- AGENTS.md §12.10 F-04 决策表"模型选型"行:`Qwen3-0.6B base + 双 LoRA` → **`Qwen3.5-0.8B base + 双 LoRA`**(P1 记忆引擎)
- 理由列同步更新:从"对齐 08a D-10 tokenizer 硬约束(Qwen3 系列)" → "Qwen3.5-0.8B 对齐 sub-1B 目标 + 已验证可用(经 PoC-2)+ 与 08 蒸馏管线探针选型一致"
- 该修订与 F-06(2026-08-31)同批登记,治理风险可控

**D-10 原文**(`08a-capacity-gap-design.md` 附录 A):
> D-10 | Tokenizer 对齐 | 同系列或与教师对齐(硬约束) | 设计层

**D-10 真实适用对象**:`08a §3.1` 蒸馏管线**探针 vs 教师**的概率分布对比场景(为避免分词噪声污染 Capacity Gap 分层)。

**P1 不适用原因**:P1 是 **LoRA 微调**(`08c §6.2`),**不依赖逐 token 概率分布对比**。

**08 蒸馏管线传导**:P1 base = Qwen3.5-0.8B → 08 训练目标锁定 Qwen3.5 系列 → 08 探针必须用 Qwen3.5 系列(Qwen3.5-1.7B + Qwen3.5-0.8B)→ 探针 vs DeepSeek 教师**跨系列**(再次违反 D-10)。

**F-07 决策前置标记**:P1 完成后必须启动"D-10 vs A-01 兼容性"决策。**不在本 spec 范围**,但已在不在范围内 §8 标记。

---

## 4. 组件详细设计

### 4.1 新增/修订组件清单

| 组件 | 操作 | 来源 | 章节 |
|---|---|---|---|
| **`agenticmemory_training/training/eval_zero_shot.py`** | **新增代码**(🔴-4) | 新增 | 4.2 |
| **`agenticmemory_training/training/eval_random_label.py`** | **新增代码**(🔴-4) | 新增 | 4.2 |
| **`agenticmemory_training/data/synthesis.py` 教师参数化** | **修订代码**(🟡-4) | 08c §3.2 | 5.1 |
| **P1-4-pre** | **新增** | 新增 §6.2a | 4.2 |
| **P1-4(base + 教师修订)**| 修订 | 08c §6.2 | 4.3 |
| **P1-4(多字段判定)**| 修订 | 08c §6.3 | 4.4 |
| **P1-5(失败诊断,含 IRR)**| **新增(条件性)** | 新增 §6.4 | 4.5 |
| **P1-2 增加 Kimi-K3 第二标注者**| **修订**(🔴-3) | 08c §4.2 | 4.5 |
| **findings_v0.md(必含 5 章节)**| **新增结构** | 新增 §9 | 4.6 |

### 4.2 P1-4-pre:Baseline 对照(新增)

**目的**:在 LoRA 训练前,量化 "LoRA 真实提升幅度"。

**前置条件(🔴-4)**:以下 2 个脚本当前**不存在**,需在实施计划中作为**新增代码**创建(复用 `eval_f1.py` 的解析与 macro-F1 逻辑):

```bash
# 步骤 1:zero-shot baseline
# 依赖: agenticmemory_training/training/eval_zero_shot.py(新增)
python3 -m agenticmemory_training.training.eval_zero_shot \
  --base-model "Qwen/Qwen3.5-0.8B" \
  --dev-jsonl data/agenticmemory_training/v0/dev.jsonl \
  --output-dir runs/lora_v0
# 输出: baseline_f1.json { zero_shot_f1: {field: f1}, ... }

# 步骤 2:random-label 对照(检测"是否只学了 input→output 映射")
# 依赖: agenticmemory_training/training/eval_random_label.py(新增)
python3 -m agenticmemory_training.training.eval_random_label \
  --base-model "Qwen/Qwen3.5-0.8B" \
  --dev-jsonl data/agenticmemory_training/v0/dev.jsonl \
  --output-dir runs/lora_v0
# 输出: baseline_f1.json { ..., random_label_f1: {field: f1}, ... }
```

**判断标准(Metis 🟡-1 整改:按字段分阈值,不再用统一 30pp)**:

| 字段 | LoRA - zero-shot 提升要求 | 含义 | 后续决策 |
|---|---|---|---|
| intent.primary | ≥ 30pp | **真实学习** | 继续 P1-4 LoRA 训练 |
| language.primary | ≥ 10pp **或**绝对 F1 ≥ 0.90 | 高基线字段,不要求 30pp | 继续 |
| entities.type | ≥ 20pp | 中等复杂度 | 继续 |
| 其他字段均值 | ≥ 10pp | 低复杂度字段 | 继续 |
| 任一字段 LoRA - random-label < 10pp | - | **只学了表面映射** | P1-5 失败诊断(可能数据量不足) |
| 任一字段 LoRA - zero-shot < 10pp | - | **LoRA 无效** | P1-5 失败诊断 |

**工作量(🔴-4 修正)**:新增 2 脚本 = **1-2 人日**(脚本开发);脚本就绪后运行 = 15-30 分钟。

### 4.3 P1-4:Base + 教师修订(原 08c §6.2)

**修订点**:
- Base 模型:`Qwen2.5-0.5B`(原 A-05 fallback) → **Qwen3.5-0.8B**(新 D-A,**并同步修订 AGENTS.md F-04 模型选型行**)
- **fallback 链(🔴-2 整改)**:Qwen3.5-0.8B 不可用 → **Qwen3.5-1.5B 或更大**(用户 2026-08-31 确认);P1-4 失败降级路径也改为"升级 Qwen3.5-1.5B 或更大"
- 标注教师:**不变**(DeepSeek V4 Flash,沿用 A-01)
- 显存需求:Qwen3.5-0.8B 全精度 ~1.5GB,LoRA 训练 ~4GB → RTX 4090(24GB) 完全够用
- Tokenizer:**待 PoC-2 前置验证**(🟢-4):Qwen3.5-0.8B 与 Qwen3 同源(151,936 vocab)需实际对比确认;若 Qwen3.5 改动 tokenizer,跨系列对齐论证即破

**新增 API 密钥 + 命名修正(🟢-2)**:
```bash
export KIMI_API_KEY="sk-..."  # P1-1 合成教师 + P1-2 IRR 第二标注者
export DEEPSEEK_API_KEY="sk-..."  # P1-2 主标注教师(沿用)
export BASE_MODEL_PATH="Qwen/Qwen3.5-0.8B"  # 命名修正:原 TEACHER_MODEL_PATH 误导,此处是学生 base
```

### 4.4 P1-4:多字段联合判定(原 08c §6.3 修订)

**原通过标准**:intent.primary F1 ≥ 0.80 通过 / < 0.50 失败。

**新通过标准(Metis 🟡-2 整改:显式定义"其他字段"集合)**:**4 字段联合判定**:

| 字段组 | 包含字段 | 通过阈值 | 不达标动作 |
|---|---|---|---|
| intent.primary | `intent.primary` | F1 ≥ 0.80 | P1-5 失败诊断 |
| language.primary | `language.primary` | F1 ≥ 0.90 | P1-5 失败诊断 |
| entities.type | `entities[] .type`(9 种类型) | F1 ≥ 0.60 | P1-5 失败诊断 |
| 其他字段均值 | `current_topic` + `session_facts` + `entities.name` + 横切元数据(`secret_detected` / `pii_present` 等),**不含 intent/language** | F1 ≥ 0.50 | P1-5 失败诊断 |

> **注意**:`session_facts` 在 08c §6.3 预期 F1 仅 0.34——**计入"其他字段均值"会必然拉低均值**。修订为:**`session_facts` 从"其他字段均值"中单独列示**,不影响通过判定,但作为"字段设计风险"在 findings §5 中专项报告(08 蒸馏管线需关注其低填充率)。

**P1 通过 = 4 字段组全部达标**(session_facts 单独列示,不计入通过/失败判定,只作风险报告)。
**P1 部分通过 = intent.primary ≥ 0.80 但其他字段组不达标** → 进入 08 蒸馏管线但对不达标字段加双重标注。
**P1 失败 = intent.primary < 0.50** → P1-5 失败诊断,升级到 Qwen3.5-1.5B 重新设计。

### 4.5 P1-5:失败模式诊断(新增,条件性)

**触发条件**:**P1-4 多字段联合判定未通过**(部分通过或失败)。

**IRR 测量前置(🔴-3 整改,必须执行)**:
- P1-2 阶段增加 **Kimi-K3 第二标注者**:对 ~50 条子集(从 dev 集中抽样)用 Kimi-K3 二次标注
- 与 DeepSeek 主标注计算 **Krippendorff α**(inter-annotator agreement)
- 若 α < 0.6 → 类别 A(教师标注不一致)判定可直接依据该测量值
- 该步骤与跨教师设计天然兼容(Kimi-K3 已就绪),额外 API 成本 ~$0.03

**执行步骤**:

```
Step 1:对 F1 < 0.85 的字段,人工抽样 50 条错误案例
Step 2:分类错误原因
  - 类别 A:教师标注本身错误(Krippendorff α < 0.6,基于 Kimi-K3 × DeepSeek 一致性测量)
  - 类别 B:模型结构问题(0.8B 容量不够)
  - 类别 C:数据量不足(450 train 太少)
  - 类别 D:prompt 工程问题(13字段字段定义不清)
Step 3:输出到 findings_v0.md §4 失败模式诊断
Step 4:根据分类给出降级路径建议:
  - 类别 A 为主 → 重做标注(可能换教师)
  - 类别 B 为主 → 升级到 Qwen3.5-1.5B 或更大(F-07 决策点)
  - 类别 C 为主 → 扩大训练集(从 450 到 2000)
  - 类别 D 为主 → 修订 mvp-schema.md 字段定义
```

**工作量**:2 小时(主要在人工分类)+ ~$0.03(Kimi-K3 IRR 子集标注,已含在 P1-2 成本)。

### 4.6 findings_v0.md 必含 5 章节

```
1. 执行摘要
   - 多字段判定结果(intent / language / entities.type / 其他字段组)
   - baseline 对比(LoRA vs zero-shot vs random-label,按字段)
   - 是否启用 P1-5(是/否)

2. 多字段判定结果
   - 表格:每个字段组 F1 + 是否达标
   - 通过/部分通过/失败状态
   - session_facts 单独列示(作为字段设计风险报告)

3. baseline 对比
   - 表格:zero-shot F1 / LoRA F1 / random-label F1 / 提升幅度(按字段)
   - 分析:LoRA 是否真的学到了抽取能力(逐字段判断)

4. 失败模式诊断(若启用 P1-5)
   - 类别 A/B/C/D 计数
   - Krippendorff α 测量值(Kimi-K3 × DeepSeek)
   - 降级路径建议

5. 对 08 蒸馏管线的具体输入
   - 字段保留建议(fill rate ≥ 40% 进 08;30-40% 进灰色; < 30% 移出)
   - 标注质量警告(Krippendorff α < 0.6 的字段,08 应双重标注)
   - 教师建议(若 P1 标注 F1 < 0.85 的字段,08 切换教师)
   - 对 mvp-schema.md 的反馈建议(触发 schema 修订)
```

---

## 5. 实施计划

> **实施顺序调整(2026-08-31 修订)**:经 Oracle 评估,在 P1-1 之前加入 **P1-0 PoC 阶段**(1-2 天),先验证"OpenCode → SKILL.md → Qwen3.5-0.8B 推理服务 → P1 闭环"的可行性,再启动正式实验。
> 
> **P1-0 PoC 不修改任何决策项**,只调整实施顺序——这是工程风险前置,不是新决策。

### 5.0 P1-0 PoC 阶段(新增,2026-08-31 修订)

**目的**:在 P1-1 数据合成之前,验证基础设施链路可行性——OpenCode 通过 SKILL.md 编排 Python 代码 → 调用 Qwen3.5-0.8B 推理服务 → 产出可被 P1-1 消费的格式。

**3 个 PoC 子任务**:

```
PoC-1:OpenCode → SKILL.md → 项目 Python 代码
  目的:验证 OpenCode 能加载项目 agenticmemory_training/* 模块并执行 Python 脚本
  工作量:0.5 天
  执行:
    - 创建 OpenCode 项目级技能:`.opencode/skills/p1-poc/SKILL.md`(🟡-5 修订:非 `.cl/opencode/skills/`,OpenCode 项目约定是 `.opencode/skills/`)
    - 若项目级不被加载,回退到全局 `~/.config/opencode/skills/p1-poc/SKILL.md`
    - 编写最小化 SKILL.md,演示调用 `python3 -m agenticmemory_training.data.synthesis`
    - 验证 OpenCode 能成功加载并执行
  通过标准:Python 脚本可被 OpenCode 通过 SKILL.md 成功调用 + 输出可被下游消费

PoC-2:Qwen3.5-0.8B 学生模型推理服务搭建
  目的:启动一个稳定运行的 Qwen3.5-0.8B 推理服务(vLLM 或类似框架)
  工作量:0.5 天
  执行:
    - 下载模型到本地(或 HuggingFace Hub)——**第 1 步即验证模型可用性**
    - 启动 vLLM 服务:`vllm serve Qwen/Qwen3.5-0.8B --port 8998`
    - 验证服务可响应 OpenAI-compatible API
  通过标准:`curl http://localhost:8998/v1/chat/completions` 返回 200 + 合理输出

PoC-3:端到端流程闭环
  目的:验证 SKILL.md → Python → 推理服务 → 输出可被 P1-1 消费
  工作量:0.5 天
  执行:
    - 编写测试 SKILL.md,调用 P1-1 数据合成(via Python + Kimi-K3 API)
    - 合成 1 条对话作为 PoC 验证
    - 用 Qwen3.5-0.8B 推理服务对该对话跑 zero-shot 抽取
    - 验证输出格式正确(JSON 13 字段可解析)
  通过标准:端到端闭环无错误 + 输出格式与 08c §6 一致
  注意:依赖 KIMI_API_KEY 就绪,成本 <$0.01(🟢-1 修订:非严格 $0)
```

**PoC 总时间**:1-2 天

**PoC 失败处理(🔴-2 整改)**:
- PoC-1 失败 → 检查 OpenCode 配置 + SKILL.md 语法 + Python 包路径(含 `.opencode/skills/` vs 全局路径回退)
- **PoC-2 失败:模型不可用 → 触发 fallback 决策链(新增分支)**:`Qwen3.5-0.8B 下载/加载失败 → 尝试 Qwen3.5-1.5B → 若仍不可用 → 暂停 P1,等待发布或回调 Qwen3 系列已发布版本(F-07 前置决策)`
- PoC-2 失败:框架问题 → 切换推理框架(vLLM → SGLang → TensorRT-LLM)
- PoC-3 失败 → 可能是接口契约不一致,需要回到 PoC-1 / PoC-2 调整

> **PoC 作用边界声明(Metis P1-0 缺口 2 整改)**:PoC 只降**工程/基础设施风险**(OpenCode/SKILL.md/推理服务链路),**不降 P1 核心方法论风险**(教师一致性、LoRA 学习能力、0.8B 容量)。PoC 通过 ≠ 方法论风险降低。PoC-3 产出的 SKILL.md 与脚本**直接成为 P1-1 的执行载体**(职责界定)。

### 5.1 修订步骤(3 个)

```
P1-1:腿 B 合成教师从 GPT-4o 改为 Kimi-K3(新增 KIMI_API_KEY);
     腿 A 公开集(SHARELY 30 条)保留(无需合成教师,天然无偏)(🟡-4 修订);
     synthesis.py 教师参数化(新增 synthesize_via_kimi() 或 model 参数支持 kimi-k3)(🟡-4)
P1-2:主标注教师不变(DeepSeek V4 Flash);新增 Kimi-K3 第二标注者(50 条子集,IRR)(🔴-3)
P1-4:base 从 Qwen3-0.6B 改为 Qwen3.5-0.8B(下载链接更新 + F-04 修订登记)
```

### 5.2 新增步骤(3 个)

```
P1-4-pre(新增):zero-shot + random-label baseline 对照
  前置:新增 2 脚本 eval_zero_shot.py + eval_random_label.py(🔴-4,1-2 人日)
  脚本就绪后运行:15-30 分钟
  执行:运行 Qwen3.5-0.8B zero-shot 在 dev 集上 + 跑"输入文本 + 随机打乱 gold 标签"作为对照
  输出:baseline_f1.json { zero_shot_f1, random_label_f1, lift_delta }(按字段)

P1-2-extra(新增):Kimi-K3 第二标注者 + Krippendorff α(🔴-3)
  前置:P1-2 主标注(DeepSeek)完成后
  执行:对 dev 集抽样 ~50 条,用 Kimi-K3 二次标注,与 DeepSeek 计算 Krippendorff α
  输出:data/agenticmemory_training/v0/irr_krippendorff.json
  成本:~$0.03(Kimi API)

P1-5(条件性新增):失败模式诊断
  触发:P1-4 多字段联合判定未通过
  工作量:2 小时
  执行:对 F1 < 0.85 的字段人工抽样 50 条,分类"教师标注错误(基于 IRR)/ 模型容量 / 数据量 / prompt 工程"
  输出:findings_v0.md §4 失败模式诊断
```

### 5.3 总成本与时间估算(2026-08-31 v0.3 修订)

| 步骤 | 时间 | API 成本 | GPU |
|---|---|---|---|
| **P1-0 PoC 阶段(新增)** | **1-2 天** | **<$0.01**(PoC-3 Kimi 调用) | **0** |
| **P1-4-pre 脚本开发(🔴-4)** | **1-2 人日** | $0 | 0 |
| P1-1 合成(Kimi-K3,腿 B)+ 腿 A 公开集 | 0.5 周 | ~$0.05(Kimi-K3 API) | 0 |
| P1-2 标注(DeepSeek 主)+ Kimi-K3 IRR 子集 | 0.5 周 | ~$0.16(DeepSeek $0.13 + Kimi $0.03) | 0 |
| P1-3 Schema 评估 | 0.5 周 | $0 | 0 |
| P1-4-pre baseline 运行(脚本就绪后) | 30 分钟 | $0 | 0 |
| P1-4 LoRA 训练(Qwen3.5-0.8B) | 0.5 周 | $0 | 1 GPU-hour |
| P1-5 失败诊断(条件性) | 0.5 周 | $0 | 0 |
| **总计(不含脚本开发)** | **2.5 周** | **~$0.21** | **1 GPU-hour** |
| **总计(含脚本开发)** | **2.5 周 + 1-2 人日** | **~$0.21** | **1 GPU-hour** |
| (vs 原始 GPT-4o 全程 ~$3.93) | | 节省 95% | |

**修订影响**:总时间 2.5 周 + 新增 1-2 人日(脚本开发,🔴-4 修正)。PoC 阶段消除基础设施风险,IRR 子集使类别 A 判定可计算(🔴-3)。

---

## 6. 验证与成功标准

### 6.1 验证清单

| 验证项 | 方法 | 通过标准 |
|---|---|---|
| **PoC-1 通过** | OpenCode 能加载 SKILL.md + 调用 Python 包 | `python3 -m agenticmemory_training.*` 可执行 + 无错误 |
| **PoC-2 通过** | Qwen3.5-0.8B 推理服务运行 | `curl http://localhost:8998/v1/chat/completions` 返回 200 |
| **PoC-3 通过** | 端到端流程闭环 | SKILL.md → Kimi-K3 合成 1 条对话 → Qwen3.5-0.8B 抽取 → 输出 JSON 13 字段可解析 |
| **F-04 修订登记(🔴-1)** | 检查 AGENTS.md §12.10 F-04 | 模型选型行已更新为 Qwen3.5-0.8B |
| **fallback 链就绪(🔴-2)** | 检查 PoC-2 失败分支 | 含"模型不可用 → Qwen3.5-1.5B → 暂停"决策链 |
| **IRR 可计算(🔴-3)** | 检查 `irr_krippendorff.json` 存在 | Kimi-K3 × DeepSeek 计算 Krippendorff α |
| **新增脚本存在(🔴-4)** | `ls agenticmemory_training/training/eval_*.py` | 含 `eval_zero_shot.py` + `eval_random_label.py` |
| **基模型正确** | 检查 `runs/lora_v0/adapter_config.json` | `base_model_name_or_path = Qwen/Qwen3.5-0.8B` |
| **跨教师落地** | 检查 `data/agenticmemory_training/v0/conversations.jsonl` 的 `metadata.teacher` | 腿 B = Kimi-K3(🟢-3 修订:需在 `write_conversations()` 新增该字段) |
| **baseline 已记录** | 检查 `runs/lora_v0/baseline_f1.json` | 含 `zero_shot_f1` + `random_label_f1` + `lift_delta`(按字段) |
| **多字段判定应用** | 检查 `findings_v0.md §2` | 4 字段组全部列示 + 通过/不达标状态 |
| **失败诊断执行**(若启用) | 检查 `findings_v0.md §4` | 50 条错误案例分类 + 降级路径建议 |
| **下游衔接就绪** | 检查 `findings_v0.md §5` | 5 个必含子章节齐全 |

### 6.2 成功标准

P1 通过(方案 B 完整修复 + PoC 阶段 + Metis 整改后):
- ✅ PoC-1/2/3 全部通过(基础设施闭环)
- ✅ F-04 修订登记完成(Qwen3.5-0.8B)
- ✅ P1-1 腿 B 用 Kimi-K3 合成,跨教师落地;腿 A 公开集保留
- ✅ P1-2 用 DeepSeek 主标注 + Kimi-K3 第二标注者(IRR 可计算)
- ✅ P1-4-pre 脚本开发完成 + baseline 对照按字段完成(新增脚本 1-2 人日)
- ✅ P1-4 LoRA 训练在 Qwen3.5-0.8B + DeepSeek 上完成
- ✅ 多字段联合判定 4 字段组达标(意图 ≥ 0.80,语言 ≥ 0.90,实体 ≥ 0.60,其他字段组 ≥ 0.50;session_facts 单独列示)
- ✅ findings_v0.md 5 章节齐全,包含"对 08 蒸馏管线的具体输入"
- ✅ 总成本 ~$0.21(< $50 预算),总时间 2.5 周 + 1-2 人日(脚本开发)

---

## 7. 未来工作(不在本 spec 范围)

### 7.1 F-07 决策前置标记:P1 完成后必须启动

**决策主题**:**"D-10 vs A-01 兼容性"**——08 蒸馏管线如何处理跨系列 tokenizer 冲突?

**两个候选方案**:
- **选项 A**:08 蒸馏管线的训练目标锁定 Qwen3.5 系列;08 探针用 Qwen3.5 系列;**教师改用 Qwen3.5-3B 或 7B**(同系列,放弃 A-01 的 DeepSeek 选择)
- **选项 B**:保留 DeepSeek 教师;08 蒸馏管线在跨系列场景下运行;**人工标注补偿** + D-10 豁免(结论可靠性降低)

**触发条件**:P1 完成 + 抽取层 DoD 定义完成 + 08 蒸馏管线启动前。

### 7.2 记忆引擎 DoD 抽取层定义(P1 完成后)

在 [`docs/agenticmemory/01-memory-model.md`](../../agenticmemory/01-memory-model.md) §1 显式定义:
- 抽取层 DoD:13 字段 schema 抽取 F1 ≥ ?(由 P1 findings 决定)
- 完整性 DoD:推理无损 B/A ≥ 0.98(P2 验证后填)
- 自知能力 DoD:irr_estimate 校准偏差 ≤ ?(P3 验证后填)

### 7.3 P2 实验(后续 spec)

验证 memory_extract 能力(Wiki DAG 抽取) + 推理无损(B/A ≥ 0.98)+ irr_estimate 校准。本 spec 不实施。

---

## 8. 关联决策与文档

### 8.1 关联 AGENTS.md 决策项

| 决策 | 状态 | 关联 |
|---|---|---|
| F-04 双 Schema 统一消费 | 🔄 **修订中(2026-08-31,由 D-A 触发)** | 模型选型行 Qwen3-0.6B → Qwen3.5-0.8B(须与本 spec 同步修订 AGENTS.md) |
| F-05 代码包归属 | ✅ 已决策 2026-08-25 | agenticmemory_training/ 包归属 |
| F-06 架构与训练工程解耦 | ✅ 已决策 2026-08-31 | 本 spec 整体框架 |
| **F-07 D-10 vs A-01 兼容性** | ⏳ **待启动** | P1 完成后启动;若 PoC-2 触发 fallback,则提前至 PoC 阶段启动 |

### 8.2 关联文档

- **本 spec 修改**:[`docs/agenticmemory_training/08c-p1-minimum-loop.md`](../../agenticmemory_training/08c-p1-minimum-loop.md) — 需同步更新 §2.5 / §3 / §6 / §9
- **下游消费**:[`docs/agenticmemory_training/08a-capacity-gap-design.md`](../../agenticmemory_training/08a-capacity-gap-design.md) — P1 findings 喂给 08 蒸馏管线
- **下游消费**:[`docs/agenticmemory/01-memory-model.md`](../../agenticmemory/01-memory-model.md) — P1 findings 定义"抽取层 DoD"
- **架构原则**:`AGENTS.md` §6.5 — 架构与训练工程解耦原则(刚决策)

---

## 9. 一句话承诺

> **本 spec 把 P1 从"赌注验证"升级为"记忆引擎抽取层 DoD 定义载体"——通过 P1-0 PoC 阶段验证基础设施闭环 + 基模型 Qwen3.5-0.8B(F-04 修订登记 + fallback 链)+ 跨教师 Kimi-K3 / DeepSeek(含 Kimi-K3 第二标注者 IRR)+ 按字段 baseline 对照 + 多字段联合判定 + 失败模式诊断 6 项修复,在 ~$0.21 / 2.5 周 + 1-2 人日(脚本开发)内实证"sub-1B 模型能学会 13 字段结构化抽取",并为 08 蒸馏管线提供完整衔接输入。**
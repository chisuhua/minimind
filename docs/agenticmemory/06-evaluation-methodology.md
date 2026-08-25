# 06 · 评估方法论 — Probe Model + Golden Filter + 双轨基准

> **文档 ID**: MEM-006-EVALUATION-METHODOLOGY
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.1
> **配套文档**:
> - 核心能力: [`01-memory-model.md`](01-memory-model.md) — 推理无损 + B/A ≥ 0.98
> - 评估框架: [`03-evaluation.md`](03-evaluation.md) — 六维度评估指标
> - 本体涌现: [`05-schema-emergence.md`](05-schema-emergence.md) — OpenIE + V1-V3 路线图
> - HydraForgeBench: [`../agenticdsl-training/04-evaluation-benchmark.md`](../agenticdsl-training/04-evaluation-benchmark.md) — DSL 生成器评估

---

## 0. 文档范围与定位

本文档定义 **agenticmemory 评估方法论**——**如何用对的方法证明记忆模型有效**,避免假阳性、噪声污染、能力盲区三大陷阱。

**与 [`03-evaluation.md`](03-evaluation.md) 的关系**:03 文档定义"评估什么指标"(双轨 B/A、IRR、六维度);本文档定义"用什么方法评估"(Probe Model 验证、Golden Filter、相关工作对比)。

---

## 1. 核心问题:为什么需要专门的方法论

### 1.1 直接对比法的致命假阳性陷阱

```
你的原始设想(有 Bug):
  A 组:让 SOTA(DeepSeek)读【原始长文本】 → 回答问题
  B 组:让 SOTA(DeepSeek)读【提取后的结构化文本】 → 回答问题
  比较:如果 B 组得分 > A 组得分,就保留这条提取数据

Bug 所在:
  SOTA 太聪明了!
  即使提取器漏掉了关键信息(如"除非下雨"这个条件),
  SOTA 在 B 组中可能会利用它庞大的内部预训练知识(Parametric Memory)
  强行脑补出正确答案,或者因为问题太简单碰巧蒙对了
  
  → 这会导致筛选出的数据集充满"假阳性(False Positives)"
    你以为提取器提得好,其实是答题器自己脑补的
```

### 1.2 解决方案:引入"弱模型试金石(Probe Model)"

```
为了证明真的是"提取后的结构化上下文"立了功,而不是答题模型自己聪明,
你必须使用一个能力较弱、完全依赖上下文、没有太多内部知识的模型
(如未经 SFT 的 Qwen-7B-Base,或者 Temperature=0 的小模型)作为"答题器(Probe)"

优化后的黄金 Pipeline:
  1. 生成:用 DeepSeek(Teacher)对原始长文本进行结构化提取,得到 Structured_Context
  2. 对照测试 A(Baseline):把【原始长文本】喂给弱模型(Probe),记录 Score_raw
  3. 对照测试 B(Treatment):把【提取后的 Structured_Context】喂给同一个弱模型,记录 Score_struct
  4. 黄金筛选法则:只有当 Score_struct >> Score_raw 时,这条数据才被保留
```

### 1.3 为什么 Probe Model 方案更好

```
弱模型没有内部知识可以依赖,它只能依靠你喂给它的上下文
  → 如果弱模型看了"原始长文本"答不出(因为长文本中存在 Context Rot 或噪音干扰)
  → 但看了"提取后的结构化文本"却答出了
  → 这构成铁证:证明你的提取器成功地把"隐藏在噪音中的高价值信号"提纯了

用这种数据训练出来的提取器,将具备极强的"信息提纯与降噪"能力
```

---

## 2. Golden Filter:基于下游效用的数据筛选

### 2.1 核心算法

```python
def golden_filter(dialogue, extracted_tuples, qa_set, probe_model):
    """黄金过滤:验证提取结果对下游推理的真实效用"""
    
    # 1. 将五元组转化为自然语言上下文
    structured_context = tuples_to_natural_language(extracted_tuples)
    
    scores = {"memory": [], "reasoning": []}
    
    for qa in qa_set:
        # 测试 A:用原始对话回答
        answer_raw = probe_model.answer(
            context=dialogue,
            question=qa["question"]
        )
        score_raw = evaluate(answer_raw, qa["ground_truth"])
        
        # 测试 B:用结构化上下文回答
        answer_struct = probe_model.answer(
            context=structured_context,
            question=qa["question"]
        )
        score_struct = evaluate(answer_struct, qa["ground_truth"])
        
        # 计算效用增益
        utility_gain = score_struct - score_raw
        qa["utility_gain"] = utility_gain
        
        # 自动分类打标
        if qa["type"] == "factual" and utility_gain > 0.1:
            scores["memory"].append(qa)
        elif qa["type"] in ["reasoning", "counterfactual"] and utility_gain > 0.2:
            scores["reasoning"].append(qa)
    
    # 只保留效用增益显著的数据
    return {
        "keep": len(scores["memory"]) + len(scores["reasoning"]) > 0,
        "memory_score": avg([s["utility_gain"] for s in scores["memory"]]),
        "reasoning_score": avg([s["utility_gain"] for s in scores["reasoning"]])
    }
```

### 2.2 效用增益阈值

| QA 类型 | 最小效用增益 | 理由 |
|---|---|---|
| **事实查询** | Score_struct - Score_raw > 0.1 | 弱模型仅依赖提取信息应能答出 |
| **多跳推理** | > 0.2 | 推理任务对结构化更敏感 |
| **反事实推理** | > 0.3 | 结构化带来的逻辑清晰度收益最大 |
| **矛盾检测** | > 0.2 | 需要结构化才能识别矛盾 |
| **信息完备性** | > 0.15 | 缺失项识别能力 |

### 2.3 Probe Model 配置规范

```yaml
# probe_model_config.yaml
probe_model: "Qwen2.5-7B-Base"  # 必须是 base model,不是 instruction-tuned
# 或更弱的:Qwen2.5-1.5B-Base / Llama-3-8B-Base

generation_config:
  temperature: 0.0          # 严格确定性
  max_tokens: 512          # 限制回答长度,避免长回答掩盖质量问题
  top_p: 1.0
  
system_prompt: |
  你只能基于给定的【上下文】回答问题。
  如果上下文中没有足够信息,你必须回答"无法确定"。
  严禁使用你自己的预训练知识!
  
  回答格式:
    答案: [你的答案]
    依据: [你引用的上下文片段]
```

**关键约束**:Probe Model 必须是 **base model**(未经指令微调),否则它的指令遵循能力会"自动填补"上下文中的空缺,导致假阳性。

---

## 3. Probe Model 验证在 V1-V3 路线图中的应用

### 3.1 V1.0(宽进严出)的 Probe 验证

```
目的:验证超级 Schema 提取器是否真的"漏的少了"

步骤:
  1. 50+ 字段的提取器对测试对话进行提取
  2. 用 Probe Model 在原始对话 vs 提取结果上分别回答 1000 道 QA
  3. 比较两组准确率:
     - 如果 Score_struct ≥ Score_raw:V1.0 提取器有效(至少不会丢信息)
     - 如果 Score_struct < Score_raw:V1.0 提取器过度抽取或丢失关键结构

预期结果(基于论文与直觉):
  维度  原始对话准确率  提取结果准确率  效用增益
  事实  ~60%            ~75%            +15%
  推理  ~40%            ~55%            +15%
  反事实 ~20%           ~35%            +15%
  矛盾  ~10%            ~25%            +15%
```

### 3.2 V2.0(重构剪枝)的 Probe 验证

```
目的:验证剪枝后的提取器是否保持同等效用

步骤:
  1. V2.0 提取器(20 字段核心)对相同测试对话进行提取
  2. 用 Probe Model 重复 V1.0 的 1000 道 QA 测试
  3. 比较 V1.0 vs V2.0 的 Score_struct:
     - 如果 V2.0 Score_struct ≈ V1.0 Score_struct:剪枝成功(冗余字段已被移除)
     - 如果 V2.0 Score_struct < V1.0 Score_struct:剪枝过度(可能丢失有用字段)

预期结果:
  V1.0 IRR: 0.93, Score_struct: 75%
  V2.0 IRR: 0.93, Score_struct: 73-75%(允许 2% 损失)
  V2.0 字段数: 20(从 50+ 减少)
  V2.0 推理速度: 提升 2-3 倍
```

### 3.3 V3.0(RL 涌现)的 Probe 验证

```
目的:验证 RL 训练后,提取器发现了新的有用特征

步骤:
  1. 收集 V3.0 模型发现的 custom_extensions(新字段类型)
  2. 对比 V2.0 vs V3.0 在"长尾问题"上的表现:
     - 长尾问题:从未在训练集中出现过的类型的问题
     - 如果 V3.0 显著优于 V2.0:证明涌现有效
  3. 人工审核 V3.0 发现的新字段是否真的有意义

预期结果:
  V3.0 在长尾问题上比 V2.0 高 10-20%
  V3.0 发现的 custom_extensions 中,30-50% 被人工审核为"有意义"
  → 这证明涌现机制确实有效
```

---

## 4. QA 题库设计:拉开 Score_struct 与 Score_raw 的差距

### 4.1 简单事实查询的问题

```
问题:"用户的预算是多少?"

预期:
  Score_raw: ~80%(原文明确写了)
  Score_struct: ~80%(提取后也很清楚)
  效用增益: ~0%(拉不开差距)

结论:简单事实查询无法验证提取质量,不是有效的评估题目
```

### 4.2 多跳推理与反事实问题

```
问题 1(多跳推理):
  "如果预算减少20%,且时间推迟一周,原方案中的哪个模块会首先崩溃?"
  → 需要整合"预算"、"时间"、"模块依赖关系"三个事实
  
问题 2(反事实推理):
  "如果用户没有提到时间限制,方案 A 是否仍然可行?"
  → 需要识别"时间限制"是隐含约束,并在反事实场景下重新评估

问题 3(矛盾检测):
  "用户的需求中存在什么矛盾?"
  → 需要识别"既要预算低又要效果好"这类矛盾

问题 4(信息完备性):
  "要做出最终决策,还缺少什么关键信息?"
  → 需要识别原文中的信息缺口
```

### 4.3 QA 题库构成

```
目标规模: 50,000+ 道 QA(每条对话配 5-10 道)

各类占比:
  事实查询: 40%(用于验证记忆类提取)
  多跳推理: 25%(用于验证推理类提取)
  反事实推理: 15%(用于验证结构化带来的逻辑清晰度收益)
  矛盾检测: 10%(用于验证冲突发现能力)
  信息完备性: 10%(用于验证缺口识别能力)

质量要求:
  - 每道题有 Ground Truth(基于原文人工标注或 SOTA 验证)
  - 难度分布:Easy 30% + Medium 50% + Hard 20%
  - 至少 10% 的题是"未提及"类对抗性题(测试模型正确拒绝)
```

### 4.4 合成 QA 的方法

```python
def synthesize_qa_set(dialogue, sota_model):
    """为一条对话自动生成 QA 测试集"""
    
    prompt = """
    请为以下对话生成 5-10 道 QA 测试题,覆盖:
    1. 事实查询(至少 2 道)
    2. 多跳推理(至少 2 道)
    3. 反事实推理(至少 1 道)
    4. 矛盾检测(至少 1 道,如有矛盾)
    5. 信息完备性(至少 1 道,识别缺口)
    
    要求:
    - 每题给出 ground_truth 和答案依据(原文片段)
    - 难度从简单到复杂递增
    - 至少 1 道题是"原文未提及"的(测试正确拒绝)
    
    对话:{dialogue}
    """
    
    return sota_model.generate(prompt)
```

---

## 5. 评估实验设计

### 5.1 实验一:效用增益验证(核心实验)

```python
def utility_gain_experiment(test_set, extractor, probe_model):
    """核心实验:验证提取器对下游推理的真实效用"""
    
    results = {
        "factual": {"raw_scores": [], "struct_scores": []},
        "multi_hop": {"raw_scores": [], "struct_scores": []},
        "counterfactual": {"raw_scores": [], "struct_scores": []},
        "conflict": {"raw_scores": [], "struct_scores": []},
        "completeness": {"raw_scores": [], "struct_scores": []}
    }
    
    for dialogue, qa_set in test_set:
        # 原始对话上下文
        raw_context = dialogue
        
        # 提取后的结构化上下文
        extracted = extractor.extract(dialogue)
        struct_context = render_to_natural_language(extracted)
        
        for qa in qa_set:
            # Probe Model 在原始对话上回答
            answer_raw = probe_model.answer(raw_context, qa["question"])
            score_raw = evaluate(answer_raw, qa["ground_truth"])
            
            # Probe Model 在提取结果上回答
            answer_struct = probe_model.answer(struct_context, qa["question"])
            score_struct = evaluate(answer_struct, qa["ground_truth"])
            
            results[qa["type"]]["raw_scores"].append(score_raw)
            results[qa["type"]]["struct_scores"].append(score_struct)
    
    # 计算效用增益
    utility_report = {}
    for qa_type, scores in results.items():
        raw_mean = np.mean(scores["raw_scores"])
        struct_mean = np.mean(scores["struct_scores"])
        utility_gain = struct_mean - raw_mean
        
        utility_report[qa_type] = {
            "raw_mean": raw_mean,
            "struct_mean": struct_mean,
            "utility_gain": utility_gain,
            "improvement_ratio": struct_mean / raw_mean if raw_mean > 0 else 0
        }
    
    return utility_report

# 通过标准:
# 各类型的 utility_gain > 0.10(即 Score_struct 比 Score_raw 高 10 个百分点以上)
```

### 5.2 实验二:噪声抵抗实验

```python
def noise_resistance_experiment(dialogue_with_noise, dialogue_clean, extractor, probe_model):
    """验证提取器在噪声环境下的稳定性"""
    
    # 提取两种上下文下的结构化记忆
    extracted_noisy = extractor.extract(dialogue_with_noise)
    extracted_clean = extractor.extract(dialogue_clean)
    
    # Probe Model 在两种提取结果上的回答
    score_noisy = probe_model.answer(render(extracted_noisy), qa.question)
    score_clean = probe_model.answer(render(extracted_clean), qa.question)
    
    # 提取稳定性:两种提取结果应该高度一致(因为噪音被过滤掉了)
    structural_similarity = compare_wiki_pages(extracted_noisy, extracted_clean)
    
    # 推理稳定性:两种提取结果应该产生相似的回答
    answer_similarity = compare_answers(score_noisy, score_clean)
    
    return {
        "structural_similarity": structural_similarity,  # 期望 ≥ 0.90
        "answer_similarity": answer_similarity  # 期望 ≥ 0.85
    }
```

### 5.3 实验三:Schema 涌现验证

```python
def schema_emergence_validation(v1_extractor, v3_extractor, novel_test_set):
    """验证 V3.0 是否能处理 V1.0 训练集中未出现过的新类型问题"""
    
    # novel_test_set:包含训练集中未出现过的领域/主题
    
    v1_scores = []
    v3_scores = []
    novel_relationships_v3 = []
    
    for dialogue in novel_test_set:
        # V1.0 提取(基于人工 schema)
        extracted_v1 = v1_extractor.extract(dialogue)
        score_v1 = evaluate_extraction_coverage(extracted_v1, dialogue)
        v1_scores.append(score_v1)
        
        # V3.0 提取(允许 custom_extensions)
        extracted_v3 = v3_extractor.extract(dialogue)
        score_v3 = evaluate_extraction_coverage(extracted_v3, dialogue)
        v3_scores.append(score_v3)
        
        # 收集 V3.0 涌现的新关系类型
        if extracted_v3.get("custom_extensions"):
            for ext in extracted_v3["custom_extensions"].values():
                novel_relationships_v3.append(ext["new_relation_type"])
    
    return {
        "v1_mean_coverage": np.mean(v1_scores),
        "v3_mean_coverage": np.mean(v3_scores),
        "emergence_uplift": np.mean(v3_scores) - np.mean(v1_scores),
        "novel_relationships_count": len(set(novel_relationships_v3)),
        "novel_relationships": list(set(novel_relationships_v3))
    }
    # 期望:V3.0 比 V1.0 高 10-20%(涌现带来长尾提升)
```

---

## 6. 相关工作与本研究的位置

### 6.1 DSPy(斯坦福大学)

```
相似点:DSPy 的核心 Optimizer(BootstrapFewShot 和 MIPRO)完全基于"下游效用验证"逻辑
  它不关心中间步骤看起来是否符合人类直觉
  它只关心下游任务的 Metric 是否提升

做法:DSPy 让模型生成多个中间推理步骤或提取的上下文,
     然后用下游任务的准确率作为 Reward,
     自动筛选出"能让最终答案正确"的中间上下文,
     并用这些数据去微调模型

对 agenticmemory 的启发:
  Golden Filter 设计与 DSPy 思路完全一致
  不同在于:DSPy 关注"提示词优化",我们关注"提取器训练数据筛选"
```

### 6.2 StructMem(浙大 & 蚂蚁集团, 2026)

```
相似点:专门针对长时序对话的"分层结构化记忆框架"
  发现直接把长对话塞给模型会导致严重的"上下文腐烂(Context Rot)"
  设计了双视角提取和时序锚定存储

做法:通过下游的多跳推理任务(Multi-hop QA)的表现,
     来反向验证和筛选其记忆提取模块的有效性

对 agenticmemory 的启发:
  StructMem 的多跳 QA 验证设计是 Golden Filter 的重要参考
  分层架构思路与 [`README.md`](README.md) §1 的 L0-L3 架构一致
```

### 6.3 Context Distillation & Selective Context(LLMLingua, Selective Context)

```
相似点:致力于把长上下文压缩成极短的 Token 序列或结构化片段
  核心评估标准:压缩后的上下文,是否能让下游模型保持(甚至超越)原始上下文的性能

对 agenticmemory 的启发:
  Context Distillation 的"压缩前后对比"是 V2.0 重构剪枝的理论基础
  Selective Context 的"选择性保留"与 V1.0 宽进严出是同源思想
```

### 6.4 Auto-CoT(自动思维链构建)

```
相似点:在推理领域,学者发现人工写的 CoT 不一定最好
  让模型自己生成多条推理链,只保留那些最终答案正确的推理链作为训练数据

对 agenticmemory 的启发:
  Auto-CoT 的"答案正确性筛选"思路被 Golden Filter 继承
  不同在于:Auto-CoT 蒸馏推理过程,我们蒸馏结构化上下文
```

### 6.5 本研究的位置

```
┌─────────────────────────────────────────────────────────────┐
│  本研究(agenticmemory) vs 相关工作的位置:                     │
│                                                              │
│  横向比较:                                                 │
│  - DSPy:      优化提示词                ← 我们:训练提取器   │
│  - StructMem: 长对话分层存储           ← 我们:多轮对话扩展 │
│  - ContextDistillation: Token 级压缩    ← 我们:结构化压缩  │
│  - Auto-CoT:  蒸馏推理链              ← 我们:蒸馏结构化记忆│
│                                                              │
│  我们的独特之处:                                           │
│  1. KV 缓存作为记忆载体(而非 token 序列或图数据库)         │
│  2. Probe Model 黄金过滤(而非只靠 SOTA 评分)                │
│  3. 双系统架构(记忆轨 + 推理轨,自动分层)                   │
│  4. V1-V3 三阶段路线图(宽进严出 → 剪枝 → 涌现)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 与现有文档的关系

| 概念 | 本文位置 | 详见 |
|---|---|---|
| **核心评估指标(B/A ≥ 0.98)** | 见 [`03-evaluation.md`](03-evaluation.md) §1 | 不变 |
| **IRR 六维度评估** | 见 [`03-evaluation.md`](03-evaluation.md) §3 | 不变 |
| **KV 验证实验** | 见 [`03-evaluation.md`](03-evaluation.md) §4 | 不变 |
| **Probe Model 黄金过滤** | 本文 §2 | 新增 |
| **V1-V3 路线图与 Probe 验证的集成** | 本文 §3 |  |
| **QA 题库设计** | 本文 §4 | 新增 |
| **评估实验设计** | 本文 §5 | 新增 |
| **本体涌现验证** | 见 [`05-schema-emergence.md`](05-schema-emergence.md) §3 | 互补 |
| **DSpy / StructMem / Context Distillation** | 本文 §6 | 参考 |
| **多轮对话评估维度** | 见 [`04-dialogue-extension.md`](04-dialogue-extension.md) §10 | 互补 |

---

## 8. 关键设计决策总结

| 决策点 | 选择 | 理由 |
|---|---|---|
| **评估方法** | Probe Model + Golden Filter(不用 SOTA 直接打分) | 避免假阳性 |
| **Probe Model 选型** | Base model(非 instruction-tuned) | 避免指令遵循能力脑补信息 |
| **效用增益阈值** | 事实 0.1,推理 0.2,反事实 0.3 | 按 QA 难度分级 |
| **QA 题库构成** | 事实 40% + 多跳 25% + 反事实 15% + 矛盾 10% + 完备性 10% | 拉开 Score_struct vs Score_raw |
| **V1/V2/V3 验证策略** | 每阶段都跑 Golden Filter | 持续追踪"信息提纯"能力 |

---

## 9. 待解决问题

| # | 问题 | 状态 | 建议决策时机 |
|---|---|---|---|
| **O28** | **Probe Model 的具体选型**(Qwen2.5-1.5B-Base vs 7B-Base vs Llama-3-8B-Base) | 🟡 待讨论 | V1.0 启动前 |
| **O29** | **效用增益阈值的具体值**(目前是建议值,需实测校准) | 🟢 训练中验证 | V1.0 训练时 |
| **O30** | **QA 题库的来源**(全部合成 vs 公开数据集补充 vs 人工标注) | 🟡 待讨论 | V1.0 启动前 |
| **O31** | **Probe Model 的来源限制**(是否禁止用同一个系列的 instruct 版本) | 🟢 实施时确定 | V1.0 启动前 |
| **O32** | **评估的统计显著性**(多少对话 + 多少 QA 才能得出可靠结论) | 🟡 待讨论 | V1.0 完成后 |

---

## 10. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-25 | 初始版本:Probe Model + Golden Filter + QA 题库设计 + 评估实验 + 相关工作 | Sisyphus(AI 助手)+ 用户 |

---

**文档版本**: v0.1
**Owner**: AgenticMind 评估组
**下一步**: 待 O28/O30 决策后启动 V1.0 评估流水线
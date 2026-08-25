# 05 · 本体涌现设计 — OpenIE + 动态 Schema + 信息瓶颈 + 双系统

> **文档 ID**: MEM-005-ONTOLOGY-EMERGENCE
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.1
> **配套文档**:
> - 核心能力: [`01-memory-model.md`](01-memory-model.md) — Wiki DAG 契约
> - 训练设计: [`02-training-design.md`](02-training-design.md) — 三层训练信号
> - 评估方法论: [`06-evaluation-methodology.md`](06-evaluation-methodology.md) — Probe Model + Golden Filter
> - 训练侧实现: [`../agenticmemory_training/`](../agenticmemory_training/) — 涌现算法实现
> - 消费方 B 的人工 schema: [`../agenticmind/context-management/mvp-schema.md`](../agenticmind/context-management/mvp-schema.md) — 13 字段人工契约

---

## 0. 文档范围与定位

本文档解决一个根本性问题:**人工定义的 Schema 永远不够**(本体盲区)。本文档提出**本体涌现**方案——让 schema 从数据中自动发现,而不是被人工锁定。

**与现有文档的关系**:
- [`01-memory-model.md`](01-memory-model.md) 定义了 Wiki DAG 的**契约结构**(8 个固定字段),这是骨架
- 本文定义的是**字段内部的具体类型**(实体类型 / 关系类型 / 约束),这些应该是**涌现**的,不是人工锁定的
- agenticmind 的 [`mvp-schema.md`](../agenticmind/context-management/mvp-schema.md) 是**人工 schema 真源**(13 字段),用于消费方 B
- agenticmemory_training 的涌现 schema(`schema_memory_v*.json`)是**涌现 schema 真源**,用于消费方 A

---

## 1. 本体盲区问题:为什么人工 Schema 永远不够

### 1.1 问题的本质

```
如果你只依赖人工定义的 Schema(比如规定只提取"意图、实体、时间"),
那么你**永远只能提取到你认知范围内的信息**。

那些你一开始没意识到重要、但实际上决定推理成败的微小细节
(比如"用户语气中的犹豫"、"某个极边缘的限制条件"),
就会被提取器无情地丢弃。

学术上这被称为:
  - "未知的未知"(Unknown Unknowns)
  - "本体盲区"(Ontology Blindspot)
```

### 1.2 结论

```
你不能指望模型提取出"绝对的、宇宙意义上的所有信息"(这会导致噪音淹没信号),
但你完全可以通过特定的训练范式,让模型自动发现、演化并提取出
"对下游推理任务有用的所有关键信息(Task-Relevant All)"。

这在前沿研究中被称为:
  - "无模式信息抽取"(Schema-free Extraction)
  - "动态本体涌现"(Dynamic Ontology Emergence)
```

### 1.3 从"穷举所有"到"以终为始"

```
破除迷思:"提取所有结构化内容"是一个伪命题。

一段 100 字的对话,如果提取"所有"信息,包括语法结构、标点符号情绪、
每个词的词性,那叫"原文复制",不叫"结构化记忆"。

真正的目标:
  提取"对下游推理器(Reasoner)回答问题有用的所有特征"。

提取器本身不知道什么有用,但下游的推理器知道。
因此,训练提取器的核心逻辑必须是"以终为始(Task-Conditioned)"——
让下游任务的反馈,来倒逼提取器去发现那些"它原本不知道要提取的内容"。
```

---

## 2. 四大训练范式:让模型"自动发现"要提取什么

### 2.1 范式一:自下而上的本体涌现(OpenIE)

**解决的痛点**:你一开始不知道该定义哪些 Schema 字段。

```
传统做法:人工定义 Schema(如 {"budget": 50, "deadline": "Friday"})
涌现做法:放弃严格的预定义 Schema。

步骤:
  1. 用 SOTA 教师模型进行"开放域信息抽取(OpenIE)"
     提示词:"请从对话中提取所有你认为有价值的实体、关系、状态变化、
             隐含假设和情绪特征。不要受限于任何固定格式,
             尽可能多地以 (Subject, Predicate, Object, Attribute) 形式输出。"

  2. 收集几十万条"放飞自我"的开放域四元组后,
     使用聚类算法(DBSCAN 或 LLM 辅助聚类)对 Predicate 进行聚类

  3. 结果:你可能自动聚类出你从未想过的类别,比如:
     - "隐性抗拒情绪"
     - "跨条件依赖关系"
     - "时间窗口约束"
     → 这些高频聚类就是你自动发现的"新 Schema"
```

**对 agenticmemory 的意义**:
- Wiki DAG 的 `relations.causal_dependency` 等字段不应该预定义具体的关系类型
- 关系类型应该从训练数据中**自动聚类涌现**
- `entities.items[*].type` 不应该锁定为 9 种,而应该是**自适应扩展**

### 2.2 范式二:基于下游反馈的强化学习(RL-Driven Emergence)

**解决的痛点**:提取器不知道某个微小细节(如"除非下雨")是否有用。

```
这是最强大、最彻底的解决方案。构建闭环,让推理器的表现来指导提取器:

  1. Actor (提取器):读取原始对话,输出结构化记忆 S
  2. Environment (推理器):读取记忆 S,回答用户问题
  3. Reward (奖励模型/规则):评估推理器的回答是否准确、是否遗漏细节
  4. 反向传导:如果推理器答错了,Reward 为负
     → 通过 PPO 或 DPO 将惩罚信号反向传导给提取器

涌现效果:
  经过几万步 RL 训练,提取器会"顿悟"
  → 它会自动学会在提取时,给那些"看似不起眼的条件状语、转折词、
    情绪修饰语"分配极高的注意力权重
  → 因为它"吃过亏",知道下游推理器需要这些细节
  → 它提取出了你从未教过它的特征
```

**对 agenticmemory 的意义**:
- 这与现有 [`02-training-design.md`](02-training-design.md) §7 的损失函数设计互补
- 在 SFT 阶段后,可以加入 RLVR 阶段,用下游推理任务的准确率作为奖励
- 详见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) §3 的 Probe Model 评估机制

### 2.3 范式三:信息瓶颈与自监督重构(Reconstruction Loss)

**解决的痛点**:如何证明提取器没有遗漏关键信息?

```
引入自编码器(Autoencoder)思想,把"无损重构"作为训练目标:

  Encoder (提取器):将原始长对话压缩成结构化记忆(JSON)
  Decoder (重构器):仅读取结构化记忆,尝试还原出原始对话的核心事实、逻辑链、关键限制
  Training Loss:如果重构器还原不出某个细节(如"预算不能超过50万,且必须是税后"),
              重构 Loss 就会变大

涌现效果:
  为了最小化重构 Loss,提取器会被迫"榨干"原始文本中的每一个高信息熵的细节
  → 那些你原本认为"不重要"的修饰词,只要它影响了句子的核心语义,
    就会被提取器自动捕获并固化到记忆中
```

**实施细节**:

```python
def reconstruction_training_loss(student_wiki, original_text, decoder_model):
    """
    重构损失:从提取的结构化记忆还原原文
    
    不是简单逐字还原,而是还原"核心事实+逻辑链+关键限制条件"
    """
    # 1. 从 Wiki 还原结构化文本
    reconstructed = decoder_model.generate(
        input=student_wiki,
        max_length=len(original_text) * 0.8  # 允许压缩
    )
    
    # 2. 关键事实抽取(用 GPT-4 做事实抽取)
    original_facts = extract_key_facts(original_text)
    reconstructed_facts = extract_key_facts(reconstructed)
    
    # 3. 事实级对比(而不是 token 级)
    fact_recall = compute_fact_recall(original_facts, reconstructed_facts)
    
    # 4. 逻辑链对比
    logic_preservation = check_logic_preservation(
        original_text, reconstructed
    )
    
    # 5. 关键限制条件对比
    constraint_preservation = check_constraints(
        original_text, reconstructed
    )
    
    return 1.0 - (0.5 * fact_recall + 0.3 * logic_preservation + 0.2 * constraint_preservation)
```

**对 agenticmemory 的意义**:
- Wiki DAG 不仅要"能回答问题",还要"能重构原文的核心信息"
- 这是 Wiki DAG 作为"完整性证明"的双重验证(详见 [`01-memory-model.md`](01-memory-model.md) §5)

### 2.4 范式四:动态 Schema 演化(Dynamic Schema Evolution)

**解决的痛点**:在推理阶段遇到了全新的、前所未见的信息类型怎么办?

```
不要让 Schema 在训练后就锁死。在工程架构上,赋予提取器"动态修改自身 Schema"的权限。

机制:
  在提取器的 System Prompt 中,除了给定基础 Schema 外,增加一个 custom_extensions 字段
  当提取器遇到极其特殊的对话(如用户突然讨论"量子计算对当前方案的干扰"),
  基础 Schema 无法覆盖时,允许提取器自主创造新的节点类型或关系类型,
  并填入 custom_extensions

固化:
  后台运行 Cron Job,统计 custom_extensions 中被高频使用的新字段
  当某个新字段出现超过 N 次,系统自动将其合并(Merge)到正式的 Base Schema 中
  并触发提取器的增量微调(Incremental SFT)
  → 这就是"本体的自动进化"
```

**实施细节**:

```json
{
  "wiki_page": {
    "title": "...",
    "core_facts": [...],
    "custom_extensions": {
      "quantum_considerations": {
        "new_field_type": "domain_specific",
        "new_relation_type": "interferes_with",
        "confidence": 0.65,
        "suggested_for_promotion": true,
        "usage_count": 1
      }
    }
  }
}
```

后台监控:

```python
def ontology_evolution_monitor(extension_logs):
    """检测新本体类型的涌现,自动合并高频新字段"""
    
    # 统计每个 custom_extension 的使用频次
    usage_stats = Counter([
        ext["new_field_type"] 
        for log in extension_logs 
        for ext in log["custom_extensions"]
    ])
    
    # 自动合并阈值(经验值:出现 ≥ 50 次)
    MERGE_THRESHOLD = 50
    
    promoted_fields = []
    for field, count in usage_stats.items():
        if count >= MERGE_THRESHOLD:
            promoted_fields.append(field)
            # 自动合并到 base schema
            merge_to_base_schema(field, count)
            # 触发增量微调
            schedule_incremental_sft(field)
    
    return promoted_fields
```

**对 agenticmemory 的意义**:
- Wiki DAG 的 `relations` 字段不应该锁定为固定的子字段(causal_dependency / comparison / temporal_sequence)
- 应该允许 `custom_extensions` 容纳涌现的新关系类型
- 运维层需要有 ontology_evolution_monitor 定期审视并合并

---

## 3. 落地实操:V1 → V2 → V3 三阶段路线图

### 3.1 阶段 V1.0:宽进严出(Broad Capture & Filter)

```
做法:
  设计一个极其庞大、甚至冗余的"超级 Schema"(包含 50+ 个维度,
  涵盖情绪、微观条件、潜在风险、隐喻等)
  用 SOTA 模型生成数据,微调一个 14B 的提取器,让它强行输出这个超级 Schema
  在下游推理时,通过轻量级的"特征选择网络"或 Attention 机制,
  让推理器自己决定这 50 个维度中哪些是这次回答需要的

目的:先解决"漏提"的问题,把召回率拉满,用算力换信息完整度。

风险:
  超级 Schema 可能引入大量冗余字段
  → 需要配套的"稀疏性惩罚"(详见 §5)
```

**实施步骤**:
1. 设计 50+ 字段的"超级 Schema"(基础字段 20 + 扩展字段 30)
2. 用 DeepSeek V3 / GPT-4o 生成 10万条"放飞自我"的 OpenIE 训练数据
3. 微调一个 14B 模型(如 Qwen2.5-14B-Instruct)
4. 部署一个"特征选择网络"(轻量级,可学习哪些字段被使用)
5. 用 Probe Model(详见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md))评估下游效果

### 3.2 阶段 V2.0:重构倒逼(Reconstruction-Driven Pruning)

```
做法:
  引入"自监督重构"(详见 §2.3)
  让提取器尝试用更少的字段(比如压缩到 20 个核心维度)去重构原始文本
  通过重构 Loss,自动"剪枝"掉那些 V1.0 中冗余且无用的字段,
  同时逼迫模型在剩下的 20 个字段中,自动融合并表达出那些被剪枝字段的高维信息

目的:
  通过重构 Loss,自动发现哪些字段是真正必要的
  把 V1.0 的 50+ 字段压缩到 V2.0 的 20 个核心字段
  在保持信息完整性的同时,降低推理时的噪音
```

**实施步骤**:
1. 实现 `reconstruction_training_loss`(见 §2.3 代码示例)
2. 在 V1.0 模型基础上,加入重构损失进行二次微调
3. 训练目标:用尽量少的字段还原尽量多的原始信息
4. 监控:字段保留率 vs IRR(信息保留率)
5. 当 20 字段版本达到 V1.0 同等 IRR 时,V2.0 完成

### 3.3 阶段 V3.0:任务驱动的强化涌现(RL-Driven Emergence)

```
做法:
  引入"基于下游反馈的 RL"(详见 §2.2)
  将提取器和推理器串联,用真实业务的 QA 准确率作为 Reward
  使用 RLVR 或 GRPO 算法

目的:
  此时,提取器已经不再是一个"被动的记录员",而是一个"主动的侦察兵"
  它会为了帮助推理器答对问题,自动去挖掘那些人类专家都容易忽略的
  "长尾特征(Long-tail Features)"
  → 至此,你真正实现了"训练出所有有用的结构化记忆"
```

**实施步骤**:
1. 部署 V2.0 的 14B 提取器 + 一个 7B 推理模型(vLLM)
2. 用真实业务 QA 数据集(如 5000 道覆盖各类问题的题目)作为 Reward 信号
3. GRPO 训练:每个 prompt 生成 8 个候选,按下游 QA 准确率排序
4. 训练 5000-10000 步
5. 监控:提取器自动发现的"新字段类型"数量,应该随训练逐步增加

### 3.4 三阶段时间表

| 阶段 | 时间 | 核心目标 | 关键产出 |
|---|---|---|---|
| **V1.0** | 第 1-2 个月 | 召回率拉满 | 50+ 字段的超级 Schema 提取器 |
| **V2.0** | 第 3-4 个月 | 剪枝到核心 | 20 字段的核心 Schema 提取器 |
| **V3.0** | 第 5-6 个月 | 任务涌现 | 能自主发现新特征的提取器 |

---

## 4. 双系统架构:记忆轨 vs 推理轨

### 4.1 为什么需要双系统

```
单一提取器的局限:
  1B 模型:能提取记忆类信息,但提取不出推理类(深度不够)
  7B 模型:能提取推理类信息,但提取记忆类太慢(参数冗余)
  → 单一模型无法在"快"和"深"之间兼得

解决方案:双轨提取器
  记忆轨:1B-3B 模型,负责快速提取显式事实
  推理轨:7B-14B 模型,负责深度提取隐式关系
  → 两个模型各有专攻,通过能力差自动分层分配数据
```

### 4.2 双系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│              🔄 数据飞轮(自动分层与分发)                        │
│                                                                  │
│   原始对话                                                        │
│       │                                                          │
│       ├──────────────────┬──────────────────┐                   │
│       ▼                  ▼                  ▼                   │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────┐      │
│  │ 1B 记忆提取器│   │ 7B 推理提取器│   │ 弱模型试金石       │      │
│  │ (vLLM, GPU-0)│  │ (vLLM, GPU-1)│   │ (Probe Model)     │      │
│  │ 输出:四元组   │   │ 输出:四元组+CoT│   │ Score_struct vs   │      │
│  │             │   │             │   │ Score_raw 黄金过滤 │      │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬───────┘      │
│         │                 │                       │              │
│         └─────────┬───────┘                       │              │
│                   ▼                               │              │
│         ┌─────────────────────┐                   │              │
│         │ 认知状态图融合       │                   │              │
│         │ Memory Nodes + Reasoning Edges         │              │
│         └─────────┬───────────┘                   │              │
│                   │                               │              │
│                   ▼                               ▼              │
│         ┌─────────────────────────────────────────────────────┐ │
│         │  反馈回流:哪些提取是有用的?(由Probe决定)            │ │
│         └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 双轨训练数据自动分层

```python
def auto_stratify_by_capacity_gap(sota_extractions, small_model_extractions):
    """基于能力差自动分层"""
    
    # 1. 语义匹配(使用 Embedding 相似度)
    matched_pairs = semantic_match(sota_extractions, small_model_extractions)
    
    memory_tuples = []
    reasoning_tuples = []
    
    for sota_tuple in sota_extractions:
        if sota_tuple in matched_pairs:
            # 1B 也能提取 → 记忆类
            memory_tuples.append({**sota_tuple, "label": "memory"})
        else:
            # 只有 SOTA 提取出 → 候选推理类
            reasoning_tuples.append({**sota_tuple, "label": "reasoning_candidate"})
    
    # 2. 逻辑算子二次校验(区分"真推理"vs"复杂记忆")
    LOGICAL_OPERATORS = [
        "causes", "implies", "depends_on", "contradicts",
        "conflicts_with", "assumes", "requires", "enables",
        "prevents", "if_then", "unless", "therefore"
    ]
    
    for t in reasoning_tuples:
        if any(op in t["predicate"].lower() for op in LOGICAL_OPERATORS):
            t["label"] = "reasoning_verified"  # 真推理
        else:
            t["label"] = "complex_memory"  # 复杂记忆(长距离依赖)
            memory_tuples.append(t)
    
    return memory_tuples, reasoning_tuples
```

### 4.4 避坑:伪推理陷阱

```
陷阱:模型把"复杂的记忆"当成了"推理"

现象:
  原文:"那个穿着红衣服的、昨天刚从纽约飞回来的、带着三个箱子的男人的狗的项圈是蓝色的"
  1B 模型提取不出"狗的项圈颜色",SOTA 提取出了
  → 系统自动将其归类为"推理类"
  
真相:这其实只是长距离依赖的记忆(Long-distance Memory),
     并不需要逻辑推导,只需要更大的注意力窗口

解法:逻辑算子校验
  检查 Predicate 是否包含逻辑算子(if/then, causes, depends_on 等)
  - 包含逻辑算子 + 下游逻辑 QA 提升 → 真推理
  - 不包含逻辑算子 → 归类为复杂记忆(交给大模型记忆模块,不是推理模块)
```

---

## 5. 稀疏性惩罚:防止噪音淹没信号

### 5.1 过度提取的诅咒

```
陷阱:为了"不漏",模型把用户的废话也提取出来

现象:
  原文:"今天天气真不错。对了,预算改到 60 万"
  模型提取出:
    - "天气不错"(实体:天气, 属性:感觉, 值:不错)  ← 废话
    - "预算 60 万"(实体:预算, 属性:金额, 值:60万)  ← 有用

后果:
  图谱变得极其庞大且稀疏
  推理器注意力被大量无用的"天气节点"分散
  推理准确率反而下降(Context Rot 变体)
```

### 5.2 稀疏性惩罚设计

```python
def memory_training_loss_with_sparsity(predictions, labels, token_types, 
                                       extraction_count):
    """
    带稀疏性惩罚的记忆训练损失
    """
    base_loss = cross_entropy(predictions, labels)
    
    # 关键信息加权(原有)
    key_weight = compute_key_weight(token_types)
    weighted_loss = base_loss * key_weight
    
    # 新增:稀疏性惩罚
    # 惩罚提取数量过多(> 50 个事实/对话)
    SPARSITY_THRESHOLD = 50
    if extraction_count > SPARSITY_THRESHOLD:
        sparsity_penalty = 0.01 * (extraction_count - SPARSITY_THRESHOLD)
        weighted_loss += sparsity_penalty
    
    # L1 正则化:鼓励提取更少的核心字段
    l1_penalty = 0.001 * extraction_count
    weighted_loss += l1_penalty
    
    return weighted_loss.mean()
```

### 5.3 训练信号设计

```
三类信号的平衡:
  1. 关键信息加权:数值 5×、条件/否定 4×(确保不漏)
  2. 稀疏性惩罚:提取过多要扣分(避免噪音)
  3. 重构损失:用尽量少的字段还原尽量多的原文(详见 §2.3)

最终目标:模型学会"什么是真正的高信息熵内容"
        而不是"为了不漏把所有东西都提取出来"
```

---

## 6. 与现有文档的关系

| 概念 | 本文位置 | 详见 |
|---|---|---|
| **核心能力契约(Wiki DAG 8 字段骨架)** | 见 [`01-memory-model.md`](01-memory-model.md) | 不变 |
| **三层训练信号(SFT 课程)** | 见 [`02-training-design.md`](02-training-design.md) | 扩展见 §2.3 重构损失 |
| **双轨评估(B/A ≥ 0.98)** | 见 [`03-evaluation.md`](03-evaluation.md) | 不变 |
| **OpenIE 涌现算法实现** | 见 [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) §3 | 已实现 |
| **评估方法论(Probe Model)** | 见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) | 新增 |
| **消费方 B 的人工 schema(13 字段)** | 见 [`../agenticmind/context-management/mvp-schema.md`](../agenticmind/context-management/mvp-schema.md) | 不变 |

---

## 7. 关键设计决策总结

| 决策点 | 选择 | 理由 |
|---|---|---|
| **schema 来源** | 双轨:人工 schema(消费方)+ 涌现 schema(记忆侧) | 消费方需要稳定契约,记忆侧需要自主发现 |
| **本体涌现方法** | OpenIE + 聚类 + RL + 重构 + 演化 | 四种范式互补,缺一不可 |
| **训练范式顺序** | SFT(冷启动)→ RLVR(精调)→ 重构(剪枝)→ 演化(本体扩展) | 先学基础,再发现长尾 |
| **双系统架构** | 记忆轨(1B) + 推理轨(7B),通过能力差自动分层 | 不同参数规模适配不同复杂度 |
| **防噪音** | 稀疏性惩罚 + L1 正则 + 重构损失 | 三重防护 |
| **逻辑算子校验** | if/then/causes/depends_on 等关键字校验 | 区分真推理 vs 复杂记忆 |
| **custom_extensions** | Wiki DAG 允许新字段涌现 + 后台定期合并 | 保持 schema 的开放性 |

---

## 8. 待解决问题

| # | 问题 | 状态 | 建议决策时机 |
|---|---|---|---|
| **O22** | **V1.0 超级 Schema 的 50+ 字段具体定义** | 🟡 待讨论 | V1.0 启动前 |
| **O23** | **逻辑算子列表的最终确定**(causes / implies / depends_on 等) | 🟡 待讨论 | V2.0 启动前 |
| **O24** | **双系统的具体模型选型**(1B 用 Qwen2.5-1.5B? 7B 用 Qwen2.5-7B-Instruct?) | 🟡 待讨论 | V1.0 启动前 |
| **O25** | **ontology_evolution_monitor 的合并阈值**(出现 N 次自动合并) | 🟡 待讨论 | V3.0 启动前 |
| **O26** | **重构损失中的事实抽取模型**(用 GPT-4o 还是教师模型?) | 🟢 实施时确定 | V2.0 启动前 |
| **O27** | **稀疏性惩罚的权重调优**(L1 系数 0.001 是否合理?) | 🟢 训练中验证 | V1.0 训练时 |

---

## 9. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-25 | 初始版本:本体盲区问题 + 四大训练范式 + V1-V3 路线图 + 双系统架构 + 稀疏性惩罚 | Sisyphus(AI 助手)+ 用户 |

---

**文档版本**: v0.1
**Owner**: AgenticMind 架构组
**下一步**: 待 O22/O24 决策后启动 V1.0 超级 Schema 设计
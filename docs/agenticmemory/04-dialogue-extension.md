# 04 · 多轮对话扩展设计 — Turn-Pair 分块 + 增量更新 + 能力自知

> **文档 ID**: MEM-004-DIALOGUE
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.1
> **配套文档**:
> - 核心能力契约: [`01-memory-model.md`](01-memory-model.md)
> - 训练设计: [`02-training-design.md`](02-training-design.md)
> - 评估框架: [`03-evaluation.md`](03-evaluation.md)
> - 边界定义: [`README.md`](README.md) §4 与 [`01-memory-model.md`](01-memory-model.md) §1.5

---

## 0. 文档范围与定位

本文档定义 **agenticmemory 记忆模型** 如何从 **多轮对话** 中提取推理所需信息——轮次分块策略、对话特化索引、MQP v3 新增查询、Type J-N 训练样本、增量更新机制、混合场景。

**与现有文档的关系**:本文是 [`01-memory-model.md`](01-memory-model.md) 和 [`02-training-design.md`](02-training-design.md) 在"对话输入"维度上的扩展。核心能力契约(推理无损 + Wiki DAG)不变,只是输入分布从"静态文档"扩展到"增量对话"。

---

## 1. 边界界定:记忆模型做什么、不做什么

### 1.1 严格定义

```
记忆模型 = 从上下文中提取推理所需信息的"信息编译器"

它做的事:
  输入:一段上下文(文档 / 对话 / 任何文本)
  输出:结构化的、可完备查询的信息表示(KV 缓存 + Wiki DAG)
  目标:使推理模型通过查询即可获取推理所需的全部信息

它不做的事:
  ✗ 对话管理(决定什么时候回复用户)
  ✗ 用户偏好记忆("用户喜欢简洁的回答")
  ✗ 技能复用("上次用了什么方法")
  ✗ 任务规划("下一步应该做什么")
  ✗ 人格/身份维持("我是谁")
```

### 1.2 为什么必须严格限定

```
问题 1:目标函数模糊
  "推理无损"是清晰的——信息提取准确率 ≥ 98%
  "用户满意度"是模糊的——什么算"好的对话体验"?
  两个目标混在一起,训练信号互相干扰

问题 2:训练数据污染
  记忆模型的训练数据应该是"信息完备性"导向的
  如果混入"对话流畅性"数据,模型可能学会"流畅但信息不完备"

问题 3:评估标准冲突
  记忆模型用"双轨准确率比 ≥ 0.98"评估
  对话管理用"用户满意度"评估
  两个标准无法统一
```

### 1.3 记忆模型在系统中的位置

```
┌─────────────────────────────────────────────────────────────┐
│  对话系统(Chat System)                                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 对话管理器   │  │ 用户画像模块 │  │ 技能库模块   │     │
│  │(路由/规划/   │  │(偏好/历史/   │  │(Skill复用/   │     │
│  │ 多轮控制)    │  │ 身份维持)    │  │ 经验沉淀)    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │               │
│         └────────────┬────┴─────────────────┘               │
│                      ▼                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ★ 记忆模型(本方案的核心)                           │  │
│  │  职责:从上下文中提取推理所需的全部信息             │  │
│  │  输入:原始上下文(文档 / 对话历史 / 混合)           │  │
│  │  输出:结构化 KV 缓存 + Wiki DAG                    │  │
│  │  评估:推理无损(双轨准确率比 ≥ 0.98)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                      │                                      │
│                      ▼                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  推理模型                                           │  │
│  │  从记忆模型查询信息 → 执行推理 → 生成结论           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**关键设计原则**:记忆模型是一个**纯"信息编译"组件**,被对话管理器、推理模型等上层模块调用,自身不关心"对话应该怎么进行"。

---

## 2. 对话上下文 vs 文档上下文的根本差异

| 维度 | 文档上下文 | 对话上下文 |
|---|---|---|
| **信息动态性** | 静态——写好后不再变化 | 动态——每轮都在增加新信息 |
| **信息完整性** | 完整——文档本身包含所有信息 | 可能不完整——用户可能只给出部分信息 |
| **信息方向** | 单向——从作者到读者 | 双向——用户和系统交替提供信息 |
| **时间维度** | 无——所有信息同时存在 | 有——信息按时间顺序累积 |
| **信息更新** | 无 | 有——后轮可能修正前轮信息 |
| **指代依赖** | 文档内可解 | 跨轮依赖——"它""上面说的""刚才那个" |

### 2.1 多轮对话中"推理需要什么"

```
场景示例:

用户轮1:"帮我查一下北京明天天气"
系统轮1:"北京明天晴,最高温32°C,最低温22°C"
用户轮2:"那后天呢?"
系统轮2:"北京后天多云,最高温30°C,有阵雨概率40%"
用户轮3:"帮我规划一个周末户外活动,要避开雨天"

推理模型需要什么信息?
  1. 用户意图:规划周末户外活动
  2. 约束条件:避开雨天
  3. 天气信息:后天有阵雨(轮2提供)
  4. 隐含推理:后天不适合户外活动
  5. 指代消解:"后天" = 轮2中提到的"后天"
```

### 2.2 对记忆模型的要求

```
记忆模型在多轮对话中的职责:
  从多轮对话历史中,提取推理所需的完整信息集

具体来说:
  1. 事实累积:每一轮新增的事实都要被编码
  2. 指代消解:将"它""上面说的"解析为具体实体
  3. 信息更新:后轮修正前轮信息时,标记为"已更新"
  4. 意图追踪:用户在每一轮的目标是什么
  5. 约束累积:用户提出的所有限制条件

记忆模型在多轮对话中不负责的:
  ✗ 决定什么时候回复用户
  ✗ 决定回复的语气和风格
  ✗ 管理对话流程("我还没理解你的问题")
  ✗ 维护用户画像("这个用户是新手")
```

---

## 3. 轮次分块策略

### 3.1 核心思路:每轮 = 一个"块"

```
旧方案(单文档):
  一篇长文档 → 按语义分块 → 每块 ≤ 6K tokens → 独立编码

新方案(多轮对话):
  多轮对话 → 按轮次分块 → 每轮(或多轮合并)≤ 6K tokens → 独立编码
  
  关键区别:
    单文档分块:按语义边界切分(段落/章节)
    对话分块:按轮次边界切分(用户轮+系统轮 = 一个"轮次对")
```

### 3.2 轮次分块规则

| 规则 | 说明 |
|---|---|
| **规则 1**:每个轮次对作为一个基本块 | 用户轮 + 系统轮 = 一个"块" |
| **规则 2**:单轮 > 6K tokens 时按语义进一步拆分 | 极少数情况(如系统回复含长代码) |
| **规则 3**:连续多个轮次对总 tokens < 2K 时合并 | 避免块过于碎片化 |
| **规则 4**:保持时间顺序 | 块1 = 轮次对1,块2 = 轮次对2,... |

### 3.3 分块示例

```
示例对话:
  块1 = [用户轮1 + 系统轮1](查天气)
  块2 = [用户轮2 + 系统轮2](问后天)
  块3 = [用户轮3 + 系统轮3](规划活动)

极端情况:
  块1.1 = [用户轮1前半段](系统轮回复过长 → 拆为多个块)
  块1.2 = [系统轮1前半段]
  ...
```

---

## 4. 索引层的对话特化设计

### 4.1 对话索引的完整结构

```json
{
  "index_layer": {
    
    "conversation_metadata": {
      "total_turns": 3,
      "total_blocks": 3,
      "domain": ["天气查询", "活动规划"],
      "conversation_type": "task-oriented"
    },
    
    "entity_registry": [
      {
        "entity_name": "北京",
        "entity_type": "城市",
        "first_mentioned": "turn_1",
        "current_blocks": [1, 2],
        "status": "active"
      },
      {
        "entity_name": "周末户外活动",
        "entity_type": "任务目标",
        "first_mentioned": "turn_3",
        "current_blocks": [3],
        "status": "active"
      }
    ],
    
    "turn_summaries": [
      {
        "block_id": 1,
        "turn_range": [1, 1],
        "user_intent": "查询北京明天天气",
        "system_action": "返回天气信息",
        "key_facts": "北京明天晴,32°C/22°C",
        "new_entities": ["北京"],
        "new_constraints": []
      },
      {
        "block_id": 2,
        "turn_range": [2, 2],
        "user_intent": "查询北京后天天气",
        "system_action": "返回天气信息",
        "key_facts": "北京后天多云,30°C,阵雨概率40%",
        "new_entities": [],
        "new_constraints": [],
        "referenced_from": ["turn_1"],
        "resolved_references": {"后天": "相对于明天的后一天"}
      },
      {
        "block_id": 3,
        "turn_range": [3, 3],
        "user_intent": "规划周末户外活动,避开雨天",
        "system_action": "待响应",
        "key_facts": "用户要求避开雨天",
        "new_entities": ["周末户外活动"],
        "new_constraints": ["避开雨天"],
        "referenced_from": ["turn_2"],
        "resolved_references": {"雨天": "后天阵雨"}
      }
    ],
    
    "entity_block_mapping": {
      "北京": [1, 2],
      "天气": [1, 2],
      "周末户外活动": [3],
      "雨天": [2, 3]
    },
    
    "cross_block_relations": [
      {
        "relation_type": "reference",
        "from_block": 2,
        "to_block": 1,
        "description": "轮2的'后天'依赖轮1建立的时间参照",
        "type_detail": "temporal_reference"
      },
      {
        "relation_type": "constraint_dependency",
        "from_block": 3,
        "to_block": 2,
        "description": "轮3的'避开雨天'约束依赖轮2的天气信息",
        "type_detail": "constraint_on_fact"
      }
    ],
    
    "information_evolution": {
      "entity_updates": [],
      "fact_updates": [],
      "constraint_accumulation": [
        {"constraint": "避开雨天", "added_at": "turn_3", "source_block": 3}
      ],
      "intent_evolution": [
        {"turn": 1, "intent": "天气查询"},
        {"turn": 2, "intent": "天气查询(扩展时间范围)"},
        {"turn": 3, "intent": "活动规划(基于天气约束)"}
      ]
    },
    
    "wiki_dag": {
      "root": {
        "node_id": "root",
        "type": "conversation_overview",
        "children": ["node_weather", "node_planning"]
      },
      "nodes": [
        {"node_id": "node_weather", "type": "subtopic", "topic": "天气信息", "blocks": [1, 2]},
        {"node_id": "node_planning", "type": "subtopic", "topic": "活动规划", "blocks": [3]}
      ],
      "edges": [
        {"from": "node_weather", "to": "node_planning", "relation": "constrains",
         "description": "天气信息约束活动规划"}
      ]
    }
  }
}
```

### 4.2 对话特有的新增索引字段

对比单文档索引,对话索引**新增**了以下字段:

| 新增字段 | 作用 | 单文档中不需要的原因 |
|---------|------|-------------------|
| `user_intent` | 每轮用户的目标 | 文档没有"用户意图"概念 |
| `referenced_from` | 当前轮引用了哪些前轮 | 文档信息是自包含的 |
| `resolved_references` | 指代消解结果 | 文档中指代在文内可解 |
| `information_evolution` | 信息随轮次的演变 | 文档信息不随时间变化 |
| `intent_evolution` | 用户意图的变化轨迹 | 文档没有"意图变化" |
| `constraint_accumulation` | 约束条件的累积 | 文档约束是静态的 |
| `entity_updates` | 实体信息的更新/修正 | 文档信息不更新 |

### 4.3 索引层大小约束

```
对话索引层的特殊考虑:
  对话可能很长(100+ 轮),但每个新轮只增加少量信息
  索引层需要在 ≤ 8K tokens 内记录所有轮次的信息

策略:
  1. 实体注册表:只保留最近引用的实体(N 轮内)
  2. 块摘要:压缩为短句
  3. 信息演变:保留最近 N 个变化 + 历史聚合摘要
  4. 跨块关系:只保留活跃关系(active status)

如果超出 8K,采用二级索引(类似文档端的二级索引方案)
```

---

## 5. MQP v3 对话特化查询

### 5.1 新增查询类型

```
MQP v2 原有查询类型(单文档场景):
  实体查询 / 属性查询 / 因果查询 / 对比查询 / 条件查询
  时序查询 / 否定查询 / 证据查询 / 冲突查询

MQP v3 新增查询类型(多轮对话场景):
  + 指代消解查询("它"指的是什么?)
  + 约束汇总查询(用户提出了哪些要求?)
  + 意图追踪查询(用户最终想要什么?)
  + 信息更新查询(这个信息是否被后续修正过?)
  + 时间锚定查询("明天"具体是哪一天?)
```

### 5.2 指代消解查询

```json
{
  "query_type": "coreference_query",
  "prompt": "请解析以下指代表达的具体含义",
  "input": {
    "reference": "第一个城市",
    "current_block": 3,
    "conversation_context": "index_layer"
  },
  "output_fields": ["resolved_entity", "resolution_chain", "source_block", "confidence"],
  "precision_level": "exact",
  "allow_fallback": false
}
```

### 5.3 约束汇总查询

```json
{
  "query_type": "constraint_summary_query",
  "prompt": "请汇总用户在所有轮次中提出的约束条件",
  "input": {
    "constraint_domain": "餐厅选择",
    "conversation_context": "index_layer"
  },
  "output_fields": ["constraints", "added_at", "source_blocks", "conflicts"],
  "precision_level": "complete",
  "allow_fallback": false
}
```

### 5.4 信息更新查询

```json
{
  "query_type": "update_history_query",
  "prompt": "请查询某个信息项的完整变更历史",
  "input": {
    "target": "会议时间",
    "conversation_context": "index_layer"
  },
  "output_fields": ["update_history", "current_value", "superseded_values", "confidence"],
  "precision_level": "exact",
  "allow_fallback": false
}
```

---

## 6. 训练样本体系扩展(Type J-N)

### 6.1 五类对话特化样本

在原有 Type A-I(9 类,见 [`02-training-design.md`](02-training-design.md))基础上,新增 Type J-N 共 5 类对话特化样本:

| 类型 | 名称 | 训练目标 |
|------|------|---------|
| **Type J** | 指代消解 | 学会解析"它""第一个城市"等指代 |
| **Type K** | 信息更新追踪 | 学会识别并标记被后轮修正的信息 |
| **Type L** | 约束累积 | 学会汇总跨轮的所有约束条件 |
| **Type M** | 意图演变追踪 | 学会追踪用户意图的变化轨迹 |
| **Type N** | 增量索引更新 | 学会增量更新索引层而非全量重建 |

### 6.2 Type J:指代消解训练样本

```json
{
  "type": "coreference_resolution_in_conversation",
  "input": {
    "conversation_blocks": [
      {"block": 1, "text": "用户:帮我查一下北京明天天气\n系统:北京明天晴,32°C"},
      {"block": 2, "text": "用户:那上海呢?\n系统:上海明天多云,28°C"},
      {"block": 3, "text": "用户:第一个城市后天的天气呢?"}
    ],
    "query": "用户在轮3中提到的'第一个城市'指的是哪个城市?"
  },
  "output": {
    "resolved_entity": "北京",
    "resolution_chain": [
      {"step": 1, "reference": "第一个城市", "interpretation": "对话中首先提到的城市"},
      {"step": 2, "search": "在轮1中查找首先提到的城市", "result": "北京"},
      {"step": 3, "verification": "轮1中用户说'帮我查一下北京明天天气',北京是第一个被提及的城市"}
    ],
    "source_block": 1,
    "confidence": 0.99,
    "evidence_status": "confirmed"
  }
}
```

### 6.3 Type K:信息更新与覆盖训练样本

```json
{
  "type": "information_update_tracking",
  "input": {
    "conversation_blocks": [
      {"block": 1, "text": "用户:会议定在周一上午10点\n系统:好的,已记录"},
      {"block": 2, "text": "用户:不对,改成周一下午2点\n系统:已更新为周一下午2点"},
      {"block": 3, "text": "用户:会议在哪里开?"}
    ],
    "query": "会议时间是几点?"
  },
  "output": {
    "answer": "周一下午2点",
    "update_history": [
      {"turn": 1, "value": "周一上午10点", "status": "superseded"},
      {"turn": 2, "value": "周一下午2点", "status": "current"}
    ],
    "evidence_status": "confirmed",
    "source_block": 2,
    "warning": "轮1中的'上午10点'已被轮2中的'下午2点'覆盖,不应使用旧值",
    "superseded_value_handling": {
      "policy": "ignore_old_value",
      "reason": "superseded value 已无推理价值"
    }
  }
}
```

### 6.4 Type L:约束累积训练样本

```json
{
  "type": "constraint_accumulation",
  "input": {
    "conversation_blocks": [
      {"block": 1, "text": "用户:帮我找一家餐厅\n系统:请问有什么要求?"},
      {"block": 2, "text": "用户:要日料,人均200以内\n系统:好的"},
      {"block": 3, "text": "用户:最好在朝阳区\n系统:明白"},
      {"block": 4, "text": "用户:要有包间\n系统:"}
    ],
    "query": "用户对餐厅的所有要求是什么?"
  },
  "output": {
    "accumulated_constraints": [
      {"constraint": "菜系 = 日料", "added_at": "turn_2", "source_block": 2},
      {"constraint": "人均 ≤ 200元", "added_at": "turn_2", "source_block": 2},
      {"constraint": "位置 ∈ 朝阳区", "added_at": "turn_3", "source_block": 3},
      {"constraint": "设施 = 有包间", "added_at": "turn_4", "source_block": 4}
    ],
    "constraint_count": 4,
    "all_constraints_satisfied": "需要同时满足以上4个条件",
    "completeness_check": "完整覆盖用户所有显式约束"
  }
}
```

### 6.5 Type M:意图演变追踪训练样本

```json
{
  "type": "intent_evolution_tracking",
  "input": {
    "conversation_blocks": [
      {"block": 1, "text": "用户:我想学Python\n系统:Python入门推荐..."},
      {"block": 2, "text": "用户:有没有免费的课程?\n系统:推荐以下免费资源..."},
      {"block": 3, "text": "用户:学完之后能做什么工作?\n系统:Python可以从事..."}
    ],
    "query": "用户的最终目标是什么?"
  },
  "output": {
    "intent_evolution": [
      {"turn": 1, "intent": "学习Python", "type": "information_seeking"},
      {"turn": 2, "intent": "寻找免费学习资源", "type": "resource_seeking", "refines": "turn_1"},
      {"turn": 3, "intent": "了解Python的职业前景", "type": "career_planning", "extends": "turn_1"}
    ],
    "current_intent": "了解Python的职业前景",
    "underlying_goal": "评估是否值得投入时间学习Python",
    "confidence": 0.85,
    "note": "用户的深层目标是从'学习'到'资源'到'职业前景',呈现决策链特征",
    "needs_reasoning_model_verification": false
  }
}
```

### 6.6 Type N:增量索引更新训练样本

```json
{
  "type": "incremental_index_update",
  "input": {
    "current_index": "<当前索引层状态>",
    "new_turn": {
      "turn_id": 4,
      "user_utterance": "不对,我说的是上海",
      "system_response": "好的,已更正为上海"
    }
  },
  "output": {
    "index_updates": {
      "entity_updates": [
        {"entity": "北京", "status_change": "active → superseded", "superseded_by": "上海"}
      ],
      "block_updates": [
        {"block_id": 4, "added_to_mapping": {"上海": [4]}}
      ],
      "fact_updates": [
        {"old_fact": "查询目标城市=北京", "new_fact": "查询目标城市=上海", "status": "updated"}
      ],
      "constraint_updates": [],
      "dag_updates": {
        "new_edges": [{"from": "node_location", "to": "node_weather", "relation": "updated_target"}]
      }
    },
    "updated_index_summary": "目标城市从北京更正为上海,前3轮的天气信息不再适用",
    "consistency_check": "增量更新结果与全量重建等价"
  }
}
```

### 6.7 训练数据规模调整

```
原有规模(单文档,见 02-training-design.md):
  Type A-I + Gold/Silver/Bronze 三级 + Easy/Medium/Hard 三级
  总计 ~840K

新增规模(多轮对话):
  Type J:指代消解        ~100K 样本
  Type K:信息更新追踪     ~80K
  Type L:约束累积         ~80K
  Type M:意图演变追踪     ~50K
  Type N:增量索引更新     ~50K
  ─────────────────────────────
  对话新增小计:~360K
  
  混合场景(文档+对话):
  跨源关系标注            ~40K
  混合场景整合             ~30K
  ─────────────────────────────
  混合新增小计:~70K
  
总计:~1.27M 样本
```

---

## 7. 增量更新机制

### 7.1 对话是增量式的——不能每次都全量重算

```
旧方案(全量重算):
  每次新轮次 → 重新处理整个对话历史 → 重建所有 KV 缓存
  成本:O(N²)(N = 总轮次数)
  实际不可行:100 轮对话需要 100×100 = 10,000 次 prefill

新方案(增量更新):
  新轮次到来 → 仅编码新轮次 → 更新索引层
  成本:O(1) per turn(常数时间)
  100 轮对话 = 100 次 prefill + 100 次索引层更新
```

### 7.2 增量更新四步流程

```
新轮次(用户轮K + 系统轮K)到来:

Step 1:编码新块
  新轮次 → prefill → KV_K(≤ 8K tokens)
  成本:1 次 prefill
  
Step 2:更新索引层
  读取当前索引层 KV_index
  + 新块的摘要信息
  → 更新索引层 KV_index(增量 prefill,仅处理新增信息)
  成本:1 次增量 prefill
  
Step 3:更新 Wiki DAG
  新块可能引入新实体、新关系、新约束
  → 增量更新 DAG 结构(结构化操作)
  成本:极低
  
Step 4:更新跨块关系
  检查新块是否引用了前轮信息(指代消解)
  → 新增跨块引用边
  成本:极低
```

### 7.3 增量一致性保证

```
关键不变量:增量更新后索引与全量重建一致率 ≥ 98%

验证方式:
  1. 模拟长对话(20+ 轮)
  2. 方式 A:增量更新索引层(逐步加入每轮)
  3. 方式 B:全量重建索引层(一次性处理所有轮)
  4. 对比 A 和 B 的最终索引层输出:
     - 实体注册表差异:≤ 5%(容忍实体合并顺序差异)
     - 块摘要差异:≤ 2%
     - 跨块关系差异:≤ 2%
     - 约束累积:必须完全一致
```

---

## 8. 混合场景:文档 + 对话

### 8.1 场景描述

```
实际应用中,推理上下文往往不是纯文档或纯对话,而是混合的:

示例:
  用户上传了一份PDF报告(文档)
  然后围绕这份报告进行了5轮对话(对话)
  现在要基于报告内容+对话中补充的信息进行推理

推理所需信息分布在:
  - 文档本身(报告中的数据和结论)
  - 对话历史(用户的补充说明、澄清、约束)
```

### 8.2 混合索引设计

```json
{
  "index_layer": {
    "context_sources": [
      {"type": "document", "id": "doc_001", "blocks": [1, 2, 3, 4, 5]},
      {"type": "conversation", "id": "conv_001", "blocks": [6, 7, 8, 9, 10]}
    ],
    
    "entity_registry": [
      {
        "entity_name": "营收增长率",
        "source": "document",
        "blocks": [2],
        "value": "15%"
      },
      {
        "entity_name": "营收增长率",
        "source": "conversation",
        "blocks": [7],
        "value": "应该是18%,我之前说错了",
        "update_type": "correction"
      }
    ],
    
    "cross_source_relations": [
      {
        "relation_type": "correction",
        "from_source": "conversation",
        "from_block": 7,
        "to_source": "document",
        "to_block": 2,
        "description": "对话中修正了文档中的营收增长率"
      }
    ]
  }
}
```

### 8.3 混合场景的查询优先级

```
推理模型查询时,优先级规则:
  1. 对话修正(最新) > 文档原始值(旧)
  2. 对话补充 > 文档未提及
  3. 文档明确 > 对话推测
  
  例外:
  - 当对话中的修正被显式标记为 superseded 时,采用被修正的文档原值
  - 当对话中的补充被标记为 unconfirmed 时,降级为参考
```

---

## 9. 训练课程调整

### 9.1 新增两个阶段:对话特化训练

```
原有四阶段(单文档,见 02 §7):
  阶段1(Epoch 1-2):块级编码 + 索引生成
  阶段2(Epoch 3-4):路由 + 提取
  阶段3(Epoch 5-6):跨块整合
  阶段4(Epoch 7):极限压力 + MQP 对齐

新增阶段5-6(对话特化):
  阶段5(Epoch 8-9):对话基础能力
    指代消解 25% + 信息更新追踪 25% + 约束累积 20%
    + 单文档任务回放 20% + 增量索引更新 10%
    
    通过标准:
      指代消解准确率 ≥ 90%
      信息更新追踪准确率 ≥ 95%
      约束累积完整率 ≥ 90%

  阶段6(Epoch 10-11):对话高级能力
    意图演变追踪 20% + 跨轮信息整合 25%
    + 长对话(10+轮)20% + 对话+文档混合 20%
    + 单文档任务回放 15%
    
    通过标准:
      意图追踪准确率 ≥ 80%
      跨轮整合准确率 ≥ 85%
      10+轮对话信息保持率 ≥ 90%
```

### 9.2 单文档能力不退化

```
关键约束:新增对话训练不能损害单文档记忆能力

措施:
  1. 每个阶段保留 15-20% 的单文档任务回放
  2. 定期评估单文档基准(IRR、双轨准确率)
  3. 如果单文档指标下降 > 2%,增加回放比例
```

---

## 10. 评估体系扩展

### 10.1 新增对话特化评估维度

| 维度 | 指标 | 目标 |
|------|------|------|
| **指代消解准确率** | 正确解析指代的比例 | ≥ 90% |
| **信息更新追踪率** | 正确识别信息被修正的比例 | ≥ 95% |
| **约束累积完整率** | 所有约束都被收集的比例 | ≥ 90% |
| **意图追踪准确率** | 正确识别用户当前意图的比例 | ≥ 80% |
| **长对话保持率** | 10+轮后信息不丢失的比例 | ≥ 90% |
| **增量更新正确率** | 增量更新后索引与全量重建一致的比例 | ≥ 98% |

### 10.2 对话场景双轨评估

```
轨道 A:推理模型直接阅读完整对话历史 → 回答推理问题
轨道 B:推理模型通过层次化记忆系统查询 → 回答推理问题

对比:B/A ≥ 0.98

特殊测试:
  - 指代密集场景(每轮都有指代)
  - 信息修正场景(前轮信息被后轮推翻)
  - 长对话场景(15-20 轮)
  - 混合场景(文档+对话)
```

---

## 11. 修订后的完整管线(STAGE 0-8)

```
STAGE 0: 语料准备
  文档语料:去重/清洗/分块/PII扫描
  对话语料:多轮对话收集/轮次对切分/角色标注
  混合语料:文档+围绕文档的对话
    │
    ▼
STAGE 1: 教师蒸馏提取
  文档场景:块级RSI + 索引层 + 多轮查询集
  对话场景:轮次级RSI + 对话索引层 + 指代消解标注
           + 信息更新标注 + 约束累积标注 + 意图演变标注
  混合场景:文档RSI + 对话RSI + 跨源关系标注
    │
    ▼
STAGE 2: Capacity Gap 分层
  维度1:信息分层(记忆/领域辅助/推理)
  维度2:位置分层(索引层/内容层)
  维度3(对话新增):时间分层(当前轮/历史轮/已废弃轮)
    │
    ▼
STAGE 3: 训练集构建
  原有九类样本(Type A-I)
  + 新增五类对话样本:
    Type J: 指代消解
    Type K: 信息更新追踪
    Type L: 约束累积
    Type M: 意图演变追踪
    Type N: 增量索引更新
    │
    ▼
STAGE 4: 记忆模型训练(单文档,E1-7)
  阶段1-4:块级编码 + 路由 + 跨块整合 + 精调
    │
    ▼
STAGE 5: 对话特化训练(E8-11)
  阶段5:指代消解 + 信息更新 + 约束累积
  阶段6:意图追踪 + 跨轮整合 + 长对话 + 混合场景
    │
    ▼
STAGE 6: 多维评估
  单文档评估(原有八维度)
  + 对话评估(六维度)
  + 混合场景评估
    │
    ▼
STAGE 7: 推理扩展
    │
    ▼
STAGE 8: 部署与运维
  增量更新机制
  KV缓存按轮次管理
  索引层增量更新
```

---

## 12. 与现有文档的关系

| 内容 | 本文位置 | 详见 |
|---|---|---|
| **核心能力契约(推理无损 + Wiki DAG)** | 见 [`01-memory-model.md`](01-memory-model.md) | 不变 |
| **三层训练信号(单文档)** | 见 [`02-training-design.md`](02-training-design.md) | 扩展见 §6 |
| **六维度评估(单文档)** | 见 [`03-evaluation.md`](03-evaluation.md) | 扩展见 §10 |
| **MQP v2 协议(单文档)** | 见 [`03-evaluation.md`](03-evaluation.md) | 扩展见 §5(MQP v3) |
| **多轮对话边界** | 本文 §1 | — |
| **轮次分块** | 本文 §3 | — |
| **对话索引层** | 本文 §4 | — |
| **Type J-N 训练样本** | 本文 §6 | — |
| **增量更新** | 本文 §7 | — |
| **混合场景** | 本文 §8 | — |
| **训练课程调整** | 本文 §9 | — |

---

## 13. 关键设计决策总结

| 决策点 | 选择 | 理由 |
|---|---|---|
| **记忆模型边界** | 严格限定为"信息编译器",不涉及对话管理/用户画像/技能复用 | 避免目标函数模糊,保持训练信号清晰 |
| **轮次分块** | 轮次对(用户轮+系统轮)为基本块,保持时间顺序 | 对话的天然边界 |
| **索引层扩展** | 新增 7 个对话特化字段(user_intent/referenced_from 等) | 捕捉对话特有的信息结构 |
| **MQP 协议扩展** | MQP v3 新增 5 类查询(指代/约束/更新/意图/时间锚定) | 覆盖对话特有的查询需求 |
| **增量更新** | O(1) per turn,而非 O(N²) 全量重算 | 长对话必须增量更新 |
| **训练样本** | 新增 Type J-N 五类 | 对话特化能力训练 |
| **混合场景** | 文档与对话信息共存于同一索引层 | 实际应用中推理上下文常是混合的 |
| **训练顺序** | 先单文档(E1-7),后对话特化(E8-11) | 基础能力先稳固,对话扩展后叠加 |

---

## 14. 待解决问题

| # | 问题 | 状态 | 建议决策时机 |
|---|---|---|---|
| **O17** | 索引层在 100+ 轮对话下的压缩策略(N 轮活跃窗口) | 🟡 待讨论 | M4 启动前 |
| **O18** | 增量更新与全量重建的一致性验证方法 | 🟡 待讨论 | M4 启动前 |
| **O19** | 对话场景下 Wiki DAG 的结构(是否需要"轮次"作为节点维度?) | 🟡 待讨论 | M4 启动前 |
| **O20** | 长对话(100+ 轮)的 KV 缓存存储策略(分段/压缩/淘汰) | 🟡 待讨论 | 中期规划 |
| **O21** | 混合场景下,对话修正覆盖文档原始值的边界条件 | 🟢 已有初步规则,实施时微调 | M4 启动前 |

---

## 15. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-25 | 初始版本:多轮对话边界 + 轮次分块 + 索引层扩展 + MQP v3 + Type J-N + 增量更新 | Sisyphus(AI 助手)+ 用户 |

---

**文档版本**: v0.1
**Owner**: AgenticMind 架构组
**下一步**: 待 O17/O18 决策后启动对话数据合成与增量更新验证
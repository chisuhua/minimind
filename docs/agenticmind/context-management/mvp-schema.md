# 会话交互 Schema 单一真源 — 13 字段人工 schema 定义 v0.2

> **文档 ID**: CM-001-SCHEMA
> **生成日期**: 2026-08-24
> **状态**: 草案 v0.2(角色升级:从"运行时抽取字段集"→"人工 schema 单一真源")
> **配套文档**:
> - `architecture.md` — 运行时会话管理智能体的编排架构
> - `p0-prototype-tasks.md` — P0 原型任务清单(interim 方案)
> - [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) — Schema 融合边界规范(本文件被其锚定)
> - AGENTS.md §12 — 项目"已锁定关键事实"单一真源

> **v0.2 角色升级记录**:
> - **从运行时抽取字段集 → 人工 schema 单一真源**:本文件被 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) 锚定
> - **13 字段中只有 `entities` 的 9 种类型可参与自动涌现锚定**,其余 12 字段(intent/topic/facts/横切元数据)只从对话语料产生
> - **新增训练侧职责**:作为 session_train.jsonl 数据合成的输出目标 schema
> - **保留运行时职责**:作为统一模型(Qwen3-0.6B + 双 LoRA)推理时的输出格式规范
>
> **v0.1.1 补丁记录**(保留):
> - 修复 §1 字段计数不一致(明确 13 字段唯一真源表)
> - 加 `session_id` / `turn_index` 会话标识
> - 加 `secret` 实体类型(开发者场景最高危 PII)
> - 定义 confidence 产生机制(softmax + temperature scaling)
> - intent 改为 primary + optional secondary(支持多意图)

---

## 0. 文档范围与定位

本文档定义 **会话上下文抽取系统** v0.1.1 版本的 **MVP 字段集**——是用于训练数据合成 + 运行时 prompt 组装的**最小可用字段集**。

**核心原则(经 Oracle 评审锁定)**:

1. **加字段容易,删字段难** —— 200K SFT 预算下,每加一个字段都是一份训练数据负债
2. **schema 字段数直接摊薄每字段监督密度**
3. **MVP 字段必须满足三个条件**:
   - 字段有明确的下游消费者(prompt 模板里真的用到)
   - 训练数据可合成(有可标注数据源)
   - 字段有降级方案(降级时不会让 prompt 组装器崩溃)

**目标读者**:AgenticMind 抽取系统开发者、训练数据工程师、运行时 prompt 模板维护者。

**不在本文档范围**:
- AgenticDSL 语言规范 → 见 HydraForge 仓
- Prompt 模板的具体组装逻辑 → 见 `architecture.md`
- 训练数据合成的具体 SOP → 见 `p0-prototype-tasks.md` 后续产出

---

## 1. 核心结论(MVP 字段集) — v0.1.1 唯一真源表

**MVP 字段集总计 13 个字段**,精确分类如下(不再有歧义):

| # | 字段 | 所在层 | 类型 | 用途 |
|---|---|---|---|---|
| 1 | `intent` | L0 | 业务字段 | 8 类意图分类(primary + 可选 secondary) |
| 2 | `entities` | L0 | 业务字段 | 9 种类型 NER(含 secret) |
| 3 | `language` | L0 | 业务字段 | 语言检测(zh/en/code/mixed/other) |
| 4 | `routing_features` | L0 | 横切元数据 | 6 个路由特征(给编排器用) |
| 5 | `field_confidence` | L0 | 横切元数据 | 逐字段 confidence(softmax + temp scaling) |
| 6 | `extraction_provenance` | L0 | 横切元数据 | 抽取来源追踪 |
| 7 | `privacy_tier` | L0 | 横切元数据 | 字段级隐私分级(白名单制) |
| 8 | `current_topic` | L1 | 业务字段 | 当前话题(单值) |
| 9 | `session_facts` | L1 | 业务字段 | 会话确立的事实(扁平 KV 列表) |
| 10 | `near_turn_entities` | L1 | 业务字段 | 近 N 轮实体提及缓存 |
| 11 | `field_confidence` | L1 | 横切元数据 | 逐字段 confidence(同 #5) |
| 12 | `extraction_provenance` | L1 | 横切元数据 | 抽取来源追踪(同 #6) |
| 13 | `privacy_tier` | L1 | 横切元数据 | 字段级隐私分级(同 #7) |

**说明**:
- "横切元数据"在 L0 和 L1 共享同一组类型定义,但实例化字段分别存在(命名上略有差异,L1 的 `field_confidence` 是 topic/facts/entities 三字段的 confidence)
- L0 还有 `session_id` 和 `turn_index` 两个**会话标识字段**(见 §3.1),它们不计入 13 字段,因属于"消息元数据"而非"抽取结果"
- 13 字段集中,**业务字段 6 个**(L0 3 + L1 3)+ **横切元数据类型 4 种**(L0 和 L1 各实例化一次,实例化字段共 6 个)

**MVP 字段集满足**:
- ✅ 所有字段有明确下游消费者
- ✅ 所有字段可被 0.5B 本地模型合成训练数据
- ✅ 所有字段有 4 级降级方案(详见 `architecture.md` §4)

---

## 2. 关键设计决策(已锁定)

### 决策 1:补齐 Oracle 评审发现的 4 个结构性字段

| 字段 | 用途 | 必须性 |
|---|---|---|
| **`field_confidence`** | 逐字段置信度,用于决策 4 的 4 级降级机制 | 🔴 必修 |
| **`extraction_provenance`** | 模型/版本/路径追踪,用于线上 debug 和 SFT 污染检测 | 🔴 必修 |
| **`privacy_tier`** | 字段级隐私分级,用于脱敏白名单决策 | 🔴 必修 |
| **`token_budget`** / **`latency_budget`** | 抽取的资源预算,用于 prompt 组装器做裁剪 | 🟡 应修 |

### 决策 2:OrchestratorInterface 抽象(可演进性)

所有字段**必须可通过统一的 OrchestratorInterface 访问**,使得:
- **v1**:Python 函数实现编排,字段由 Python 字典承载
- **v2**:HydraForge AgenticDSL 实现编排,字段由 DSL 节点的 `args` 承载

**关键约束**:字段的 JSON Schema 必须在 v1 和 v2 之间稳定不变,只换承载方式。

### 决策 3:backlog 字段不删除设计

被砍掉的字段**保留在 backlog 而非删除**:
- 第二阶段加字段容易(只需加 schema 字段 + 训练数据)
- 删除字段难(已有 SFT 样本、prompt 模板引用)

详见 §5 Backlog。

---

## 3. Schema 定义

### 3.1 L0(per-turn)MVP 字段

```yaml
# L0: 当前轮输入的结构化表征
TurnContextL0:
  # === 会话标识(消息元数据,非抽取结果)===
  
  session_id: string                 # 会话唯一标识(UUID)
  turn_index: int                    # 当前轮次(从 1 开始)
  
  # === 核心抽取字段 ===
  
  intent:                            # 意图分类
    primary: IntentEnum              # 主意图,8 类之一
    secondary: list[IntentEnum]      # 次意图(可空,支持多意图场景)
    confidence: float                # primary 置信度
    confidence_per_label: dict[IntentEnum, float]  # 每个候选标签的置信度
    provenance: ProvenanceTag
  
  entities:                          # 命名实体
    items: list[Entity]
    # Entity: {type, value, span, confidence, privacy_tier}
    # type: secret|person|project|file_path|url|version|code_symbol|api|org
    aggregate_confidence: float
    provenance: ProvenanceTag
  
  language:                          # 语言检测
    primary: enum # zh|en|code|mixed|other
    secondary: list[enum]
    confidence: float
  
  # === 路由特征(给编排器用)===
  
  routing_features:
    input_length: int                # 字符/token 数
    entity_density: float            # 实体数 / 长度
    has_multi_hop_coreference: bool  # 含跨轮指代(v0.1.1 暂用启发式,见 §6.1)
    has_ambiguous_referent: bool     # 含模糊指代
    code_block_count: int            # 代码块数
    cost_estimate: float             # 估算 token 成本
  
  # === 横切元数据 ===
  
  field_confidence:                  # 逐字段置信度
    intent: float
    entities: float
    language: float
    routing_features: float          # 通常为 1.0(规则生成)
    extraction_quality: float        # 整体抽取质量估计
  
  extraction_provenance:             # 抽取来源追踪
    extractor_id: string             # e.g. "intent-cls-zh-0.5b-v1"
    extractor_version: string
    extraction_path: enum            # local|hybrid|cloud_forward
    timestamp: ISO8601
    latency_ms: int
    errors: list[string]
  
  privacy_tier:                      # 字段级隐私分级(白名单制)
    intent: enum                     # public|domain_private|derived_sensitive
    entities: enum
    language: enum
    # public: 允许出域
    # domain_private: 仅同域内允许
    # derived_sensitive: 禁止出域(派生敏感)
```

**v0.1.1 补丁说明**:
- 新增 `session_id`(string)和 `turn_index`(int),用于持久化和跨轮引用
- intent 改为 `primary + secondary + confidence_per_label`,支持"不对,应该用 GRPO"这种 correct+command 多意图场景
- Entity type 增 `secret`,详见 §3.4 隐私细则

**8 类意图分类的精确定义**(MVP 必保):

| 意图 | 定义 | 例子 |
|---|---|---|
| `question` | 知识/答案查询 | "什么是 X?"、"X 怎么用?" |
| `command` | 操作请求 | "帮我写个函数"、"翻译这段代码" |
| `clarify` | 要求澄清 | "你什么意思?"、"请再说一遍" |
| `confirm` | 确认类 | "对"、"好的"、"继续" |
| `correct` | 纠错类 | "不对,你理解错了"、"应该是 X" |
| `chat` | 闲聊 | "今天怎么样?"、"你好" |
| `refuse` | 拒答类 | "不用了"、"算了" |
| `meta` | 关于助手自身的请求 | "你是谁?"、"你能做什么?" |

### 3.2 L1(per-session)MVP 字段

```yaml
# L1: 会话级状态
SessionStateL1:
  session_id: string                 # 会话唯一标识(与 L0 对齐)
  created_at: ISO8601
  last_active_turn: int
  
  current_topic:                     # 当前话题(单值,非树)
    value: string
    since_turn: int                  # 第几轮确立
    confidence: float
    privacy_tier: enum
  
  session_facts:                     # 会话确立的事实(扁平 KV 列表)
    items: list[FactEntry]
    # FactEntry: {key, value, source_turn, confidence, provenance}
    # 例: {key: "user_name", value: "Alice", source_turn: 3, confidence: 0.95}
    last_updated_turn: int
  
  near_turn_entities:                # 近 N 轮实体提及缓存
    window_size: int                 # MVP: 5
    items: list[EntityMention]
    # EntityMention: {entity_ref, turn, span, role_in_turn}
    # 例: 实体"calculateSum"在第 3、5、7 轮被提及
  
  # === 横切元数据 ===
  
  field_confidence:
    current_topic: float
    session_facts: float
    near_turn_entities: float
  
  extraction_provenance:
    updater_id: string                  # e.g. "session-state-updater-v1"
    updater_version: string
    last_update_turn: int
  
  privacy_tier:
    current_topic: enum
    session_facts: enum
    near_turn_entities: enum
```

**字段约束**:
- `current_topic` 是**单值字符串**,不是话题树。话题树进 backlog。
- `session_facts` 是**扁平 KV 列表**,不带信念/未知分层。分层进 backlog。
- `near_turn_entities` 是**最近 N 轮**(MVP: N=5)的实体提及缓存,不维护长期引用图。

### 3.3 Confidence 产生机制(关键定义)

**字段 confidence = 分类器 softmax 概率 + temperature scaling 校准**。

```python
# 推理时
def compute_field_confidence(logits: Tensor, temperature: float = 1.0) -> float:
    """softmax(logits / temperature).max()"""
    scaled = logits / temperature
    probs = softmax(scaled, dim=-1)
    return probs.max().item()

# 训练后,用 validation set 拟合 temperature
def calibrate_temperature(model, val_loader) -> float:
    """在验证集上拟合单一标量 temperature,使 ECE 最小"""
    # 详见 Guo et al. 2017 "On Calibration of Modern Neural Networks"
    ...
```

**校准流程**:
1. 模型训练完成后(T1.3/T1.4),在验证集上做 **temperature scaling**
2. 校准后的 confidence 才是 `field_confidence` 的真值
3. T3.4 用 50+ 条样本验证 ECE<0.1,正式校准报告在 Phase 1 用 500+ 样本
4. **拒绝选项**(reject option):若 `confidence < reject_threshold` 且模型对预测不确定,降级到下一级

**为什么必须做 temperature scaling**:0.5B 模型的 softmax 输出天然过自信,直接用 ECE 必然>0.1。校准是 4 级降级机制(D7)的**前置依赖**,不是可选优化。

### 3.4 横切元数据字段定义

```yaml
# === 跨字段使用的类型定义 ===

ProvenanceTag:
  extractor_id: string               # 模型 ID 或规则 ID
  extractor_version: string          # 版本号
  extraction_path: enum              # local|hybrid|cloud_forward
  fallback_used: bool                # 是否走了降级路径
  timestamp: ISO8601

Entity:
  type: enum                         # person|project|file_path|url|version|code_symbol|api|org
  value: string
  span: [int, int]                   # 字符偏移 [start, end]
  confidence: float # 0-1
  privacy_tier: enum # public|domain_private|derived_sensitive
  context: string                    # 周围 10 字符(用于消歧)

FactEntry:
  key: string                        # 唯一标识,e.g. "user_name"
  value: any                         # JSON-serializable
  source_turn: int                   # 确立于第几轮
  confidence: float
  provenance: ProvenanceTag
  superseded_by: int|null            # 如果被覆盖,指向新版本

EntityMention:
  entity_ref: string                 # 引用 Entity.value
  turn: int
  span: [int, int]
  role_in_turn: enum                 # subject|object|modifier|unknown
```

### 3.4 字段级隐私分级规则

| 字段 | 默认 privacy_tier | 出域允许 | 理由 |
|---|---|---|---|
| `intent.value` | public | ✅ | 8 类意图不涉隐私 |
| `language.primary` | public | ✅ | 语言信息不含隐私 |
| `routing_features.*` | public | ✅ | 路由特征无隐私 |
| `entities.items[*].type` | public | ✅ | 实体类型可暴露 |
| `entities.items[*].value` | domain_private | ⚠️ 视类型 | 详见下表 |
| `current_topic.value` | domain_private | ⚠️ | 话题可能含项目名 |
| `session_facts[*].value` | domain_private | ⚠️ | 用户事实敏感 |
| `near_turn_entities[*]` | domain_private | ⚠️ | 实体提及可能组合成隐私 |
| `extraction_provenance` | public | ✅ | 元数据无隐私 |

**实体 value 的隐私分级细则**(v0.1.1 含 secret):

| 实体类型 | privacy_tier | 理由 | 抽取方式 |
|---|---|---|---|
| **`secret`** | **derived_sensitive** | API key/token/密码,最高危 | **规则正则**(优先级最高,即使置信度低也保留) |
| `code_symbol` | public | 代码符号不涉隐私 | NER + 规则 |
| `url` | domain_private | URL 可能含内部地址 | NER + 规则 |
| `file_path` | domain_private | 路径可能含用户名 | NER + 规则 |
| `person` | domain_private | 人名组合可识别身份 | NER(**v0.1.1 调整**,从 derived_sensitive 降级,见下方说明) |
| `org` | domain_private | 可能是非公开公司 | NER |
| `project` | domain_private | 项目名可能涉密 | NER |
| `api` | public | API 名称公开 | NER |
| `version` | public | 版本号公开 | NER + 规则 |

**v0.1.1 调整**:`person` 从 `derived_sensitive` 降级为 `domain_private`,理由:
- 技术对话中公开人名(Karpathy、论文作者)极常见
- 一刀切禁出域会导致大量 false positive,反而需要过度脱敏
- **降级到 domain_private 仍可在出域时被识别并走脱敏流程**
- 后续 M3 验证触发率,如发现仍有问题,改回 derived_sensitive

**secret 检测的强制要求**:
- 规则引擎(T1.6)必须实现 `SecretDetector`,覆盖:
  - AWS / GCP / Azure API key 前缀
  - GitHub Personal Access Token(`ghp_*`)
  - OpenAI API key(`sk-*`)
  - 通用 JWT(`eyJ*` 三段式 base64)
  - PEM 私钥(`-----BEGIN ... PRIVATE KEY-----`)
- 检测到 secret **即使 NER 置信度低也强制保留**,因漏报代价 >> 误报

### 3.5 与 AgenticDSL runtime 的接口契约

**核心契约**:TurnContextL0 + SessionStateL1 必须能序列化为 **扁平 JSON 对象**,作为 DSL 节点的 `args` 注入,通过 HydraForge 的 inja 模板访问:

```
{{ intent.value }}            → "question"
{{ entities.items[0].value }} → "calculateSum"
{{ current_topic.value }}     → "agenticdsl-training"
{{ session_facts['user_name'] }} → "Alice"
```

**约束**:
- 字段值必须是 inja 模板可序列化的(JSON scalar / list / dict)
- 字段路径必须在 schema 中预定义(运行时不做动态字段反射)
- 字段更新通过 `update_session_state` 节点调用,**不直接修改 DSL args**(防并发污染)

---

## 4. 4 级降级方案(决策 4 的实施)

按 `architecture.md` §4 的 4 级降级阶梯,字段级降级规则:

| 降级级别 | 触发条件 | L0 字段状态 | L1 字段状态 | Prompt 模板行为 |
|---|---|---|---|---|
| **L0:全量** | 全部字段 confidence ≥ 阈值 | 全部填充 | 全部填充 | 完整模板 |
| **L1:保核心** | `extraction_quality` 中等 | intent + entities 填充 | current_topic 填充 | 简化模板(无 preferences) |
| **L2:仅核心** | 仅 intent 置信 | 仅 intent | 仅 current_topic | 最小模板(仅意图提示) |
| **L3:raw 直通** | 抽取整体失败 | **空** | **空** | 直接拼历史对话,无抽取 |

**字段级 confidence 阈值**(默认,可在 prompt 组装时调整):

| 字段 | confidence 阈值 | 低于阈值时 |
|---|---|---|
| `intent` | 0.7 | 降级到 L1 模板 |
| `entities` | 0.6 | 跳过 entity 注入 |
| `language` | 0.85 | 默认中文 |
| `current_topic` | 0.6 | 标记 `unknown_topic` |
| `session_facts[*]` | 0.7 | 不注入此 fact |

---

## 5. Backlog(已砍但保留设计的字段)

下列字段**在 MVP 中被砍**,但保留在 backlog,第二阶段可加入:

### 5.1 L0 backlog

| 字段 | 原设计位置 | 砍掉理由 | 第二阶段加入条件 |
|---|---|---|---|
| `sentiment_polarity` | PragmaticAnalysis | 下游 prompt 模板无人消费 | 出现明确的情感响应分支 |
| `urgency` | PragmaticAnalysis | 同上 | 出现紧急响应路由 |
| `code_structure_ast` | MorphologyAnalysis | 0.5B 模型不擅长 AST,延迟高 | 出现代码上下文强需求 |
| `multimodal_signals` | MorphologyAnalysis | MVP 纯文本 | 多模态需求出现 |
| `referents` | SemanticAnalysis | 复杂,需要 coref 模型 | 指代消解错误率 > 阈值 |
| `knowledge_extraction` | SemanticAnalysis | 提取成本高,容易被云端依赖 | 出现会话级知识图谱需求 |
| `ambiguous_claims` | SemanticAnalysis | 需要复杂的 reasoning | 拒答机制需要细化时 |

### 5.2 L1 backlog

| 字段 | 原设计位置 | 砍掉理由 | 第二阶段加入条件 |
|---|---|---|---|
| `topic_tree` | SessionState | 维护成本高 | 出现嵌套子话题需求 |
| `agenda` | SessionState | MVP 无显式议程 | 任务级目标出现 |
| `commitments` | SessionState | 状态机复杂 | 多轮承诺追踪成为痛点 |
| `conflicts` | SessionState | 需要跨轮推理 | 出现冲突检测需求 |
| `inferred_user_profile` | UserProfile | 推断字段 = 幻觉重灾区 + 隐私敏感区 | 第二阶段重新评估 |
| `emotion_history` | PragmaticAnalysis | 需 L0 sentiment 支持 | L0 sentiment 加入后 |
| `session_knowledge_graph` | SessionKnowledgeGraph | 维护成本高,需要信念/未知分层 | 拒答机制需要细粒度支持 |
| `unknowns` | SessionKnowledgeGraph | **亮点,优先于 facts 加入** | 与拒答机制联动时立即加入 |

**亮点保留**:`unknowns` 字段(SessionKG 的"已知未知")与 AgenticMind 顶层"知道自己不知道"机制直接对齐,**建议在第二阶段优先于其他 backlog 字段加入**。

---

## 6. 验证方法

### 6.1 Schema 验证

- 所有字段必须有 Python `dataclass` 或 Pydantic model 定义
- 所有字段必须通过 JSON Schema 校验(可序列化为 valid JSON)
- 所有字段必须有 unit test(空填充、部分填充、全填充)

### 6.2 抽取质量验证

| 维度 | 验证方法 | 通过标准 |
|---|---|---|
| 字段填充率 | 100 条真实会话,统计每个字段的非空率 | L0 ≥90%, L1 ≥70% |
| 字段准确率 | 50 条人工标注样本,对比抽取结果 | intent accuracy ≥85%, entity F1 ≥0.8 |
| 降级触发率 | 100 条会话,统计每级降级触发频率 | L3(raw 直通)触发 ≤5% |
| Confidence 校准 | predicted confidence vs actual accuracy | ECE < 0.1 |

### 6.3 与 prompt 组装器集成验证

- 50 条 prompt 模板测试用例,覆盖所有字段组合(全填充/部分填充/全空)
- 字段缺失时,prompt 组装器不崩溃(降级到下一级或 raw 直通)
- 字段更新时,SessionState 保持一致(无并发污染)

---

## 7. 决策检查清单

实施前必须回答:

- [ ] **MVP 9 个核心字段 + 4 个横切元数据 = 13 个字段,是否够用?**
  - 如果不够,列出缺失的明确场景和对应字段
- [ ] **字段 confidence 阈值是否符合实际抽取质量?**
  - 需要等 P0 原型跑出真实抽取数据后调整
- [ ] **backlog 中的亮点字段(unknowns)是否应在 MVP 加入?**
  - 取决于拒答机制优先级
- [ ] **privacy_tier 的白名单是否正确?**
  - 需要安全审计

---

## 7.5 与 08b 的关系(v0.2 新增,关键)

**本文件定义的 13 字段** 是 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) §3.2 中描述的"人工 schema 单一真源"。

### 字段参与自动涌现锚定的情况

| 字段 | 是否参与涌现锚定 | 原因 |
|---|---|---|
| `entities`(9 种类型) | ✅ **是** | 实体类型可与涌现类型映射对齐 |
| `language` | ❌ 否 | 语言检测不存在涌现过程 |
| `intent`(8 类) | ❌ 否 | 对话行为无法从文档涌现 |
| `current_topic` | ❌ 否 | 话题概念是会话级,无文档对应 |
| `session_facts` | ❌ 否 | 会话事实是有状态概念,无文档对应 |
| `near_turn_entities` | ❌ 否 | 近 N 轮实体缓存是会话级 |
| `field_confidence` | ❌ 否 | 元数据,非内容 |
| `extraction_provenance` | ❌ 否 | 元数据,非内容 |
| `privacy_tier` | ❌ 否 | 元数据,非内容 |
| `routing_features` | ❌ 否 | 元数据,非内容 |
| `session_id` / `turn_index` | ❌ 否 | 消息标识,非内容 |

**关键结论**:13 字段中**只有 `entities` 的 9 种类型**(`secret`/`person`/`project`/`file_path`/`url`/`version`/`code_symbol`/`api`/`org`)可参与自动涌现锚定。

### 字段在训练数据中的产出方式

- **可参与锚定的字段**(`entities`):在 08 管线 Phase 3 中,自动涌现的实体类型与本文件的 9 种类型做**类型映射**(seed anchor 模式)
- **不可参与锚定的字段**(`intent`/`topic`/`facts`/横切元数据):**仅由 session_train.jsonl 合成管线产生**(详见 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) §6)

### 字段在训练后模型输出中的统一消费

详见 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) §1.2:

- **Schema 层分离**:本文件定义的 13 字段 + `schema_memory_v*.json` 自动涌现,**互不污染**
- **数据集层联合**:task-tagged 混合训练数据(`session_extract` task 用本文件 schema,`memory_extract` task 用自动涌现 schema)
- **模型层多任务**:统一 Qwen3-0.6B base + 双 LoRA(或单多任务)

## 8. 文档边界

| 内容 | 归属 | 理由 |
|---|---|---|
| **13 字段人工 schema 定义**(本文件) | AgenticMind 仓 `context-management/mvp-schema.md` | 是人工契约真源,被 08b 锚定 |
| **OrchestratorInterface 抽象** | `architecture.md` | 是编排层规范 |
| **P0 原型任务清单(interim)** | `p0-prototype-tasks.md` | 是工程实施 |
| **Prompt 模板具体组装** | `prompt-template.md`(待创建) | 是模板维护 |
| **session_train.jsonl 数据合成 SOP** | `session-train-data-synthesis.md`(待创建) | 是会话侧数据工程 |
| **Schema 融合边界规范** | [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) | 是双 schema 协同规范 |
| **自动涌现算法** | [`../../agenticmemory_training/08a-capacity-gap-design.md`](../../agenticmemory_training/08a-capacity-gap-design.md) | 是数据合成算法 |
| **HydraForge Agent 编排的演进路径** | `architecture.md` §5 | 是跨仓协作 |

---

**文档版本**:v0.2(角色升级:人工 schema 单一真源)
**Owner**:AgenticMind 抽取系统组
**下一步**:进入 `p0-prototype-tasks.md` 的任务清单,开始 P0 原型实现
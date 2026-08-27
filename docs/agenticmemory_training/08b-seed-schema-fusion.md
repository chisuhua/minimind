# Schema 融合边界规范 — Seed Schema 与自动涌现的协同

> **文档 ID**: LLMTRN-008B-MEMDIST-FUSION
> **生成日期**: 2026-08-24
> **状态**: 草案 v0.1
> **配套文档**:
> - 综述: [`README.md`](README.md)
> - 设计稿: [`08a-capacity-gap-design.md`](08a-capacity-gap-design.md)
> - 搭建指南: [`08-memory-distillation-pipeline.md`](08-memory-distillation-pipeline.md)
> - 人工 schema 真源: [`../agenticmind/context-management/mvp-schema.md`](../agenticmind/context-management/mvp-schema.md)

---

## 0. 文档范围与定位

本文档定义 **Schema 融合的边界**——明确 context-management 的 13 字段人工 schema 与 agenticmemory_training 的自动涌现 schema 在哪里融合、在哪里分离。

**核心结论(锁定)**:

> **Schema 层分离,数据集层联合,模型层多任务。** 两份独立 schema 互不污染,通过 task tag 在训练数据层和模型输出层协同。

**不在本文档范围**:
- 自动涌现算法本身 → 见 `08a §5`
- 人工 schema 字段定义 → 见 `../context-management/mvp-schema.md`
- 模型训练超参 → 见 `08a §7`
- 运行时智能体消费 → 见 TR-3 后续文档

---

## 1. 核心结论

### 1.1 一句话定义

| 维度 | 自动涌现 schema (`schema_memory_v*.json`) | 人工 schema (`mvp-schema.md`) |
|---|---|---|
| **来源** | 08 管线 Phase 3 HDBSCAN + LLM 概念化 | context-management 13 字段(人为锁定) |
| **范围** | 文档语料中的实体-关系-概念 | 多轮对话中的交互结构 |
| **输入分布** | 单 chunk 文档(无状态) | 多轮对话(有状态) |
| **演化频率** | 高频(随语料变化) | 低频(版本锁定) |
| **消费方** | 训练后模型输出"记忆抽取"任务 | 训练后模型输出"会话抽取"任务 |
| **下游使用** | 知识图谱查询 | 智能体运行时 prompt 组装 |

### 1.2 融合边界(关键)

```
┌────────────────────────────────────────────────────────────────┐
│ Schema 层:两份独立 schema,互不污染                              │
│                                                                │
│  schema_memory_v1.json              mvp-schema.md               │
│  ┌─────────────────────┐           ┌─────────────────────┐      │
│  │ 实体类型:           │           │ 13 字段:             │      │
│  │ - Person            │           │ - intent             │      │
│  │ - Organization      │           │ - entities           │      │
│  │ - Concept           │           │ - language           │      │
│  │ - ... (涌现)        │           │ - current_topic      │      │
│  │                     │           │ - session_facts      │      │
│  │ 关系类型:           │           │ - near_turn_entities │      │
│  │ - cause_of          │           │ - 横切元数据 ×4      │      │
│  │ - has_property      │           │                     │      │
│  │ - ... (涌现)        │           │                     │      │
│  └─────────────────────┘           └─────────────────────┘      │
│            │                                  │                │
│            └──────────┬───────────────────────┘                │
│                       ▼                                        │
│              [schema_v_fused.json]                             │
│              (训练数据清单级别联合索引,不是字段级合并)            │
└────────────────────────────────────────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ 数据集层:task-tagged 多任务混合                                  │
│                                                                │
│  memory_train.jsonl    session_train.jsonl                      │
│  ┌─────────────────┐   ┌─────────────────┐                     │
│  │ {               │   │ {               │                     │
│  │   "task":       │   │   "task":       │                     │
│  │     "memory_   │   │     "session_   │                     │
│  │      extract", │   │      extract",  │                     │
│  │   "input": ... │   │   "input": ...  │                     │
│  │   "output": {  │   │   "output": {   │                     │
│  │     "entities":│   │     "intent":..│                     │
│  │     "relations":   │     "entities": │                     │
│  │   }            │   │     "topic":... │                     │
│  │ }              │   │   }             │                     │
│  └─────────────────┘   └─────────────────┘                     │
│            │                      │                           │
│            └──────────┬───────────┘                           │
│                       ▼                                       │
│           task_tagged_mixed_dataset                           │
│        (同一个文件,按 task 字段分发)                            │
└────────────────────────────────────────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ 模型层:Qwen3-0.6B base + 多任务 SFT / 双 LoRA                   │
│                                                                │
│  Option A: 多任务 SFT                                          │
│    - 输入: {"task":"...", "input":"..."}                       │
│    - 输出: 对应 task 的 payload                                  │
│    - 同一权重文件                                                │
│                                                                │
│  Option B: 双 LoRA (推荐)                                       │
│    - 同一 base: Qwen3-0.6B                                      │
│    - LoRA_memory: 记忆抽取                                      │
│    - LoRA_session: 会话抽取                                     │
│    - 独立训练/回滚                                                │
└────────────────────────────────────────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ 运行时:同一个智能体调用同一个模型                                 │
│                                                                │
│  推理请求: {"task":"memory_extract", "input":"段落"}            │
│    → 模型返回 {"entities":[...], "relations":[...]}            │
│                                                                │
│  推理请求: {"task":"session_extract", "input":"对话"}           │
│    → 模型返回 {"intent":..., "entities":..., "topic":...}     │
│                                                                │
│  同一个智能体消费两个 task 的输出,做下游决策                     │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 为什么不能在 schema 层做字段级融合?

### 2.1 输入分布错配

| 字段类型 | 是否能从文档涌现 | 能否从对话涌现 |
|---|---|---|
| 实体类型(`Person`/`Organization`/...) | ✅ 可以 | ✅ 可以(可对齐) |
| 关系类型(`cause_of`/`has_property`/...) | ✅ 可以 | ⚠️ 部分可对齐 |
| `intent`(对话行为) | ❌ 不可能 | ✅ 可以 |
| `current_topic` | ❌ 不可能 | ✅ 可以 |
| `session_facts` | ❌ 不可能 | ✅ 可以 |
| `field_confidence` | ❌ 这是元数据 | ❌ 元数据 |
| `privacy_tier` | ❌ 这是元数据 | ❌ 元数据 |
| `routing_features` | ❌ 这是元数据 | ❌ 元数据 |

**关键观察**:CM 13 字段中只有 `entities` 的 9 种类型可以与自动涌现的实体类型映射;其余 12 个字段(交互结构 + 元数据)在文档语料中根本不存在,无法"涌现"。

### 2.2 任务性质错配

08a §11 把记忆引擎定位为:
- System 1(无状态,逐 chunk 抽取)
- 不思考、不推理

会话管理是:
- 有状态,跨轮
- 含 `update_session` / `superseded_by` / 滑动窗口
- 轻度推理(状态机语义)

把两者压进同一训练,**任务性质不匹配**:
- 训练输入分布不同(文档 vs 多轮对话)
- 评估指标不同(CaRB F1 vs intent accuracy)
- 失败模式不同(涌现漂移 vs 状态不一致)

### 2.3 强行融合的副作用

**若坚持 schema 字段级合并**(`schema_v_fused.json`):

1. **Schema 版本治理失效**:涌现部分高频演化,人工契约被迫跟着升版本
2. **0.6B 退化输出空 payload**:大量样本部分字段天然为空,小模型学成"空输出最安全"的保守策略
3. **评估体系割裂**:08a 的指标(CaRB F1、schema 遵从率)完全不覆盖会话任务

**结论**:schema 层分离是必要的。

---

## 3. 数据集层联合(task-tagged 混合)

### 3.1 数据格式

```json
{
    "task": "memory_extract | session_extract",
    "input": "<对应输入>",
    "output": {
        "<task-specific payload>"
    },
    "source": {
        "schema_version": "memory_v1 / mvp-v0.1.1",
        "extraction_method": "HDBSCAN+LLM / 人工定义",
        "provenance": "..."
    }
}
```

### 3.2 两类样本的产生方式

**memory_extract 样本**(由 08 管线 Phase 1-5 产出):
- 输入:文档段落
- 输出:`{"entities": [...], "relations": [...]}`
- 数据源:`memory_train.jsonl`(08 管线 Phase 5 产出)

**session_extract 样本**(需要新建管线,见 §6):
- 输入:多轮对话
- 输出:`{"intent": ..., "entities": ..., "language": ..., "current_topic": ..., "session_facts": [...], ...}`
- 数据源:`session_train.jsonl`(新管线产出)

### 3.3 混合策略

两个数据集通过 `task` 字段区分,在训练时按比例采样:

```yaml
# 训练数据混合比例(待 Phase 6 验证)
mix_ratio:
  memory_extract: 0.7      # 当前 08 管线已能产出
  session_extract: 0.3     # 新管线待建,初期可能少
```

---

## 4. 模型层多任务

### 4.1 Option A:多任务 SFT(单一权重)

**实现**:
```python
# 训练时
def forward(input_with_task):
    task, input_text = parse(input_with_task)
    if task == "memory_extract":
        return memory_payload_schema(input_text)
    else:  # session_extract
        return session_payload_schema(input_text)

# 推理时(由智能体决定)
inference("memory_extract", doc_paragraph)
inference("session_extract", conversation)
```

**优点**:
- 单一权重,部署简单
- 跨任务知识迁移(基础语言能力共享)

**缺点**:
- 一个任务失败影响另一个
- 难独立回滚

### 4.2 Option B:双 LoRA(同一 base,推荐)

**实现**:
```python
# 训练时(独立)
train(LoRA_memory, memory_extract_data)
train(LoRA_session, session_extract_data)

# 推理时(动态切换)
base = Qwen3-0.6B
model = base + LoRA_active  # active 由 task 决定
inference(task, input)
```

**优点**:
- 独立训练、独立回滚
- 失败隔离(会话任务失败不影响记忆任务)
- 可单独升级任一 LoRA

**缺点**:
- 部署稍复杂(需要 LoRA 切换)
- 显存占用略增(两份 LoRA 权重)

**推荐**:Option B(双 LoRA),对齐 08a D-10 tokenizer 硬约束(Qwen3 系列)

### 4.3 何时选择 Option A?

仅在以下条件下考虑 Option A:
- LoRA 切换延迟 > 50ms(超出 hot path 预算)
- 跨任务知识迁移显著(待 Phase 6 验证)
- 0.6B 容量吃紧(单 LoRA 已超,无空间加载第二个)

否则优先 Option B。

---

## 5. 运行时:智能体统一消费

### 5.1 推理 API 设计

```python
# 智能体的统一调用接口
class UnifiedExtractionModel:
    def extract(self, task: str, input: str) -> dict:
        """调用训练后的小模型"""
        if task == "memory_extract":
            return self._infer_memory(input)  # 输出三元组
        elif task == "session_extract":
            return self._infer_session(input)  # 输出 13 字段
```

### 5.2 智能体的下游消费

```python
# 智能体流程示例
def agent_response(user_conversation):
    # Step 1: 会话结构抽取
    session_state = model.extract("session_extract", user_conversation)
    
    # Step 2: 知识检索(可选)
    relevant_facts = knowledge_base.query(session_state.entities)
    
    # Step 3: 智能体决策
    decision = agent_llm.reason(
        user_input=user_conversation,
        session_state=session_state,
        relevant_facts=relevant_facts
    )
    
    return decision
```

### 5.3 不再需要独立的 Python 抽取系统

之前的 P0 原型中"LocalExtractorPool + 3 个独立 0.5B 模型"的设计被替换为:

- ❌ 不再需要 `intent-cls-zh-0.5b`、`entity-ner-zh-0.5b`、`lang-detect-0.3b` 三个独立模型
- ❌ 不再需要 Python 端的 PII 脱敏 + 规则融合
- ✅ 一个 Qwen3-0.6B base + 双 LoRA(或单多任务)即可

**P0 原型的关键路径调整**:
- 编排层(OrchestratorInterface + 4 级降级)保留 —— 仍需要验证 hot path 行为
- 抽取层(LocalExtractorPool)实现替换为"统一模型调用"
- 隐私层(PII 脱敏)仍需要,但可以在推理前后做(规则引擎独立)
- SessionState 管理(数据库、状态机)保留

---

## 6. 缺失的环节:session_train.jsonl 数据管线

**当前状态**:CM 有 schema 定义(`mvp-schema.md`),但**没有 session_train.jsonl 数据合成管线**。

**这是当前最重要的缺口**:

### 6.1 数据来源

需要多轮对话语料:
- 公开 assistant 对话集(SHARELY / MultiWOZ 等)
- 合成对话(GPT-4 扮演用户 + 助手)
- 内部试用数据

### 6.2 合成流程

```
对话语料
  ↓
教师 API(DeepSeek V4 Flash 或 Qwen3-4B)
  ↓ 对每轮对话产出 13 字段标注
session_train.jsonl
  ↓
质量控制(类似 08 管线 §4.2)
  - 格式合规率 ≥99%
  - 标注一致性 ≥85%(双人交叉)
  - 字段填充率(避免空输出)
```

### 6.3 工作量估计

- 对话语料收集:1 周
- 教师 API 标注流水线:1 周
- 质量控制:0.5 周
- **合计:2-3 周**(可与 08 管线 Phase 1-2 并行)

---

## 7. 与现有文档的边界

| 内容 | 归属 | 理由 |
|---|---|---|
| **Schema 融合边界规范**(本文件) | agenticmemory_training/08b | 是数据合成的设计补充 |
| **D-14 决策**(seed 锚定策略) | 08a 附录 A | 是设计层决策 |
| **seed_schema 配置代码** | 08 实操文档 §七 | 是 Phase 3 代码补充 |
| **人工 schema 字段定义** | context-management/mvp-schema.md | 是人工 schema 单一真源 |
| **运行时智能体消费规范** | context-management/architecture.md | 是智能体运行时规范 |
| **统一模型训练 Recipe** | 待新建 `08c-unified-model-training.md` | 是训练算法补充 |
| **session_train 数据合成 SOP** | 待新建 `context-management/training-data-synthesis.md` | 是 CM 侧数据合成 |

---

## 8. 决策检查清单

实施前必须确认:

- [ ] **任务 1:模型选型统一为 Qwen3-0.6B**(08a D-10 硬约束)
- [ ] **任务 2:Schema 层分离(非字段融合)** —— 本文档 §1.1 锁定
- [ ] **任务 5:P0 interim 并行**(用"教师 API + 规则"过渡,统一模型就绪后替换)
- [ ] **D-14 决策加入 08a 附录 A**(本文件 §2 锁定)
- [ ] **seed_schema_path 配置加入 08 实操文档**(本文件 §3.2 锁定)
- [ ] **session_train.jsonl 数据合成管线启动**(本文件 §6 锁定)

---

## 9. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-24 | 初始草案:基于 Oracle 评审,澄清 schema 融合边界 | Sisyphus(AI 助手)|

---

**文档版本**:v0.1
**Owner**:AgenticMind 数据工程团队
**下一步**:
1. 更新 08a 附录 A,加 D-14 决策(seed 锚定策略)
2. 更新 08 实操文档 §七 Phase 3,加 `seed_schema_path` 参数
3. 启动 session_train.jsonl 数据合成管线
4. 更新 context-management 各文档角色定位
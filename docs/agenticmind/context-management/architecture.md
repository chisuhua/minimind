# 会话管理智能体编排架构 — Python(v1)→ HydraForge(v2) 可演进编排

> **文档 ID**: CM-002-ARCH
> **生成日期**: 2026-08-24
> **状态**: 草案 v0.2(方向重定位:LocalExtractorPool 改为统一模型 + task tag)
> **配套文档**:
> - `mvp-schema.md` — 13 字段人工 schema 单一真源(本架构的契约对象)
> - `p0-prototype-tasks.md` — P0 原型任务清单(interim 方案)
> - [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) — Schema 融合边界规范
> - [`../agenticdsl-training/06-vn001-alignment.md`](../agenticdsl-training/06-vn001-alignment.md) — HydraForge VN-001 自举愿景

> **v0.2 重定位记录**:
> - §3.2 LocalExtractorPool 从"3 个独立 0.5B 模型"重写为"统一 Qwen3-0.6B base + 双 LoRA + task tag"(对齐 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) §4)
> - §3.2 加 `extract_with_pii_check` 接口(PII 脱敏 + secret 检测独立于抽取模型)
> - §6 不变量表修正:"LocalExtractorPool 不变"改为"接口不变,实现可换"
> - 顶部问题陈述从"如何让 Python 编排器平滑演进"改为"如何让编排层与统一模型协作"
>
> **v0.1.1 补丁记录**(保留):
> - §2.1 补 `TokenBudget` / `FallbackStrategy` / `DegradationLevel` / 完整 `SessionState` dataclass 定义
> - §3.4 重写 `SchemaValidator` 降级判定逻辑(原 MINIMAL 不可达)
> - §3.1 `detect_coreference` 幽灵函数改为 `heuristic_coreference_check` 启发式正则
> - §2.3 v2 DSL 草图加"占位语法"免责声明
> - §2.4 v1→v2 切换条件补第 5 条"回滚演练"
> - §5.2 显式说明 v2 复用 v1 Python 工具需 ToolRegistry + Sidecar 集成路径

---

## 0. 文档范围与定位

本文档定义 **会话上下文抽取系统** 的**编排层架构**,核心目标是解决:

> **如何让今天的 Python 编排器,在不重写抽取逻辑的前提下,平滑演进到 HydraForge Agent 编排?**

**核心结论(锁定)**:

1. **v1(本周上线)**:Python 函数编排,单一 0.5B 本地模型,4 级降级
2. **v2(TR-3 之后)**:HydraForge AgenticDSL 编排,复用 v1 的所有抽取器和 Schema
3. **关键约束**:v1 和 v2 必须共享 **OrchestratorInterface 抽象**——保证字段 schema 不变,只换编排实现

**目标读者**:AgenticMind 抽取系统架构师、P0 原型开发者、v2 HydraForge 编排开发者。

> **v0.2.1 边界声明(2026-08-25)**:本文档只描述**运行时编排**。**P1 训练管线(数据合成→教师标注→LoRA 微调)归属于训练侧,见 [`../../agenticmemory_training/08c-p1-minimum-loop.md`](../../agenticmemory_training/08c-p1-minimum-loop.md)**。两侧通过共享契约 `agenticmind/extraction/`(schemas/validator/privacy)解耦,训练脚本位于 `agenticmemory_training/` 包,不放本架构的编排代码。运行时编排代码待 P2 落 `agenticmind_runtime/`(预留)。

---

## 1. 核心结论(架构骨架)

### 1.1 三层架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                  OrchestratorInterface (抽象层)                  │
│  contract: plan(features) -> ExtractionPlan                       │
│            execute(plan) -> TurnContext                           │
│            update_session(turn_ctx, session_state) -> SessionState│
└───────────────────────────┬──────────────────────────────────────┘
                            │
            ┌───────────────┴────────────────┐
            ▼                                ▼
    ┌──────────────┐                 ┌────────────────────┐
    │ v1 Python Impl│                 │ v2 HydraForge Impl │
    │ (P0 原型)   │                 │ (TR-3 后)         │
    │              │                 │                    │
    │ 纯函数 +    │                 │ AgenticDSL 程序    │
    │ 50 行代码    │                 │ + 工具集调用       │
    │              │                 │                    │
    │ 本周可用    │                 │ 自举演示项目      │
    └──────┬───────┘                 └─────────┬──────────┘
           │                                 │
           └─────────────────┬───────────────┘
                             ▼
                  ┌────────────────────────┐
                  │   LocalExtractorPool   │
                  │  (共享,不变)           │
                  │                        │
                  │ - intent-cls-0.5b      │
                  │ - entity-ner-0.5b      │
                  │ - lang-detect-0.3b     │
                  │ - rule-engine (regex)  │
                  └────────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
          ┌──────────────┐         ┌─────────────────┐
          │ Schema 校验器│         │  CloudForwarder │
          │ (字段级)     │         │  (白名单制)     │
          └──────┬───────┘         └────────┬────────┘
                 │                          │
                 └──────────┬───────────────┘
                            ▼
                  ┌────────────────────────┐
                  │   PromptAssembler       │
                  │  (跨编排器复用)         │
                  │                        │
                  │  4 级降级阶梯          │
                  │  inja 模板渲染          │
                  └────────────────────────┘
```

### 1.2 三层职责划分

| 层 | 职责 | v1 实现 | v2 实现 |
|---|---|---|---|
| **OrchestratorInterface** | 决定**抽什么、走哪条路径** | Python 函数 `route()` | HydraForge Agent 程序 |
| **LocalExtractorPool** | 执行**具体抽取**(模型调用 + 规则) | Python 类(共享) | Python 类(共享) |
| **PromptAssembler** | **消费抽取结果**,组装最终 prompt | Python 函数 | Python 函数(DSL args 注入) |

**关键洞察**:只有 Orchestrator 层在演进时需要切换,其他两层**完全共享**。这就是抽象的价值。

---

## 2. OrchestratorInterface 抽象(关键设计)

### 2.1 接口定义(Python typing,作为契约)

```python
from typing import Protocol, Mapping, Any
from dataclasses import dataclass
from enum import Enum

class ExtractionPath(Enum):
    LOCAL_ONLY = "local"
    HYBRID = "hybrid"
    CLOUD_FORWARD = "cloud"

class DegradationLevel(Enum):
    """4 级降级阶梯(与 mvp-schema.md §4 对齐)"""
    FULL = "full"                     # 全字段通过
    KEEP_CORE = "keep_core"           # 保 intent + entities
    MINIMAL = "minimal"               # 仅 intent
    RAW_PASSTHROUGH = "raw"           # raw 直通,无抽取

class FallbackStrategy(Enum):
    DEGRADE_GRACEFULLY = "degrade"    # 降级到下一级
    RAW_PASSTHROUGH = "raw"           # 直接 raw 直通
    FAIL_FAST = "fail"                # 抛出异常

@dataclass
class TokenBudget:
    """抽取资源预算(由 PromptAssembler 提供)"""
    max_input_tokens: int            # 输入 token 上限
    max_extraction_tokens: int       # 抽取过程允许的总 token
    max_latency_ms: int              # 总延迟上限
    max_cloud_cost_usd: float        # 云端成本上限

@dataclass
class ExtractionPlan:
    """v1 和 v2 共享的中间产物"""
    dimension_routing: Mapping[str, ExtractionPath]
    budget: TokenBudget
    fallback_strategy: FallbackStrategy
    privacy_white_list: list[str]    # 允许出域的字段

@dataclass
class TurnContext:
    """mvp-schema.md §3.1 定义的字段集"""
    session_id: str
    turn_index: int
    intent: IntentField
    entities: EntitiesField
    language: LanguageField
    routing_features: RoutingFeatures
    field_confidence: FieldConfidence
    extraction_provenance: Provenance
    privacy_tier: PrivacyTier

@dataclass
class SessionState:
    """mvp-schema.md §3.2 定义的字段集(完整定义)"""
    session_id: str
    created_at: str                  # ISO8601
    last_active_turn: int
    current_topic: TopicField
    session_facts: SessionFactsField
    near_turn_entities: NearTurnEntitiesField
    field_confidence: FieldConfidence  # L1 字段级
    extraction_provenance: Provenance
    privacy_tier: PrivacyTier         # L1 字段级

class OrchestratorInterface(Protocol):
    """所有编排器必须实现的契约"""

    def plan(
        self,
        raw_input: str,
        session_state: SessionState,
        budget: TokenBudget,
    ) -> ExtractionPlan:
        """根据输入和会话状态,产出抽取计划"""
        ...

    def execute(
        self,
        plan: ExtractionPlan,
        raw_input: str,
    ) -> TurnContext:
        """执行计划,产出 TurnContext"""
        ...

    def update_session(
        self,
        turn_ctx: TurnContext,
        session_state: SessionState,
    ) -> SessionState:
        """更新会话状态(L1 字段)"""
        ...

    def get_degradation_level(
        self,
        turn_ctx: TurnContext,
    ) -> DegradationLevel:
        """根据字段 confidence 决定降级级别(委托给 SchemaValidator)"""
        ...
```

### 2.2 v1 Python 实现示例(规划 ~50 行)

```python
# v1: 文件 context_extraction/orchestrator_v1.py

class PythonOrchestrator:
    """v1 实现: 纯 Python 规则编排"""

    def plan(self, raw_input, session_state, budget):
        features = self._compute_routing_features(raw_input)
        plan = ExtractionPlan(
            dimension_routing={
                "intent": ExtractionPath.LOCAL_ONLY,
                "entities": ExtractionPath.LOCAL_ONLY,
                "language": ExtractionPath.LOCAL_ONLY,
            },
            budget=budget,
            fallback_strategy=FallbackStrategy.RAW_PASSTHROUGH,
            privacy_white_list=["intent.value", "language.primary", "entities.items[*].type"],
        )
        # 复杂 case 转发云端
        if features.entity_density > 0.3 or features.has_multi_hop_coreference:
            plan.dimension_routing["entities"] = ExtractionPath.CLOUD_FORWARD
        return plan

    def execute(self, plan, raw_input):
        results = {}
        for dim, path in plan.dimension_routing.items():
            if path == ExtractionPath.LOCAL_ONLY:
                results[dim] = self.local_pool.extract(dim, raw_input)
            elif path == ExtractionPath.CLOUD_FORWARD:
                results[dim] = self.cloud_forwarder.forward(
                    dim, raw_input, plan.privacy_white_list
                )
        return self._merge_and_validate(results)

    def update_session(self, turn_ctx, session_state):
        # 简单状态更新: 更新 current_topic 和 near_turn_entities
        ...
```

### 2.3 v2 HydraForge 实现示意(TR-3 后)

```yaml
# v2: 文件 context_extraction/orchestrator_v2.yaml
# 由 HydraForge AgenticDSL 程序描述
# ⚠️ 占位语法示意 — 实际以 HydraForge v3.10 规范为准(Markdown + 围栏 + 特殊 token)
#    见 https://HydraForge-internal/agenticdsl-spec/v3.10/

agenticdsl_open:
version: "1.0"
description: "HydraForge-orchestrated context extraction (v2)"

subgraph_decl: ContextExtractionPipelineV2
node_def:
  - id: compute_features
    type: call_tool
    tool: compute_routing_features  # 复用 v1 的 Python 工具(见 §5.2)
    inputs: [raw_input]
    outputs: [features]

  - id: decide_path
    type: agent_decision  # 智能体决策
    inputs: [features, session_state, budget]
    outputs: [plan]
    # 这里 LLM 决策,但 prompt 来自训练数据(sft-dataset)

  - id: parallel_extract
    type: parallel_fork
    branches:
      - id: local_extract
        type: call_tool
        tool: local_pool.extract  # 共享 v1 的 LocalExtractorPool
      - id: cloud_extract
        type: call_tool
        tool: cloud_forwarder.forward
        condition: plan.dimension_routing[dim] == CLOUD_FORWARD

  - id: merge_validate
    type: call_tool
    tool: schema_validator  # 共享 v1 的 Schema 校验器
    outputs: [turn_context]

  - id: update_session
    type: call_tool
    tool: session_state_updater
    outputs: [new_session_state]

  - id: assemble_prompt
    type: call_tool
    tool: prompt_assembler  # 共享 v1 的 PromptAssembler
    outputs: [final_prompt]

edges_topology:
  compute_features -> decide_path
  decide_path -> parallel_extract
  parallel_extract -> merge_validate
  merge_validate -> update_session
  update_session -> assemble_prompt
agenticdsl_close:
```

**v2 的关键不同**:
- `decide_path` 节点用 **AgenticDSL 智能体决策**(LLM 生成 DSL 子图决定路径)
- 其他节点(`local_pool.extract`、`schema_validator` 等)**调用 v1 的 Python 工具**
- 整个编排器**自身就是一个可被 HydraForge runtime 验证/执行的 DSL 程序**

### 2.4 v1 → v2 切换的判定条件

**只在以下条件全部满足时,才从 v1 切换到 v2**:

| # | 条件 | 阈值 | 验证方式 |
|---|---|---|---|
| 1 | HydraForge runtime CLI 可用 | `agenticdsl validate <file>` 通过 | AGENTS.md §9 "待开始" → "已完成" |
| 2 | AgenticMind LLM 训练完成 | pass@1 ≥80% | TR-3 M5 |
| 3 | v2 在 50 条真实 case 上 ≥ v1 | 抽取质量不降低,延迟 ≤1.5x v1 | A/B 测试 |
| 4 | 编排延迟满足 SLO | P95 ≤500ms(v1 当前 ≤200ms) | 性能基准测试 |
| **5** | **回滚演练通过** | 故障注入(v2 异常/超时/oom)后自动落回 v1 | Chaos test 脚本 |

**不满足任一条件时,保持 v1**。这是 Oracle 强烈推荐的"渐进路径"。

**v0.1.1 新增**:条件 5(回滚演练)在 v1 默认部署时就需要准备好,v2 一上线就有安全保障。

---

## 3. v1 Python 编排细节

### 3.1 路由决策规则(可计算特征)

```python
# 规则路由(v1 的核心)
def compute_routing_features(raw_input: str) -> RoutingFeatures:
    return RoutingFeatures(
        input_length=len(raw_input),
        entity_density=count_entities(raw_input) / max(len(raw_input), 1),
        has_multi_hop_coreference=heuristic_coreference_check(raw_input),
        has_ambiguous_referent=heuristic_ambiguity_check(raw_input),
        code_block_count=count_code_blocks(raw_input),
        cost_estimate=estimate_tokens(raw_input),
    )

def heuristic_coreference_check(text: str) -> bool:
    """v0.1.1 启发式代词检测(替代原 detect_coreference 幽灵函数)
    
    简化规则:出现跨句代词("它"、"这个"、"那个"后跟方法/函数名)就标记。
    完整 coref 模型在 backlog(见 mvp-schema.md §5.1)。
    """
    pronoun_patterns = r"(它|这个|那个|此|该)(函数|方法|类|变量|参数|文件|库)"
    return bool(re.search(pronoun_patterns, text))

def heuristic_ambiguity_check(text: str) -> bool:
    """模糊指代启发式:出现"那东西"、"前面那个"等"""
    ambiguous_patterns = r"(那东西|前面那个|上面那个|之前那个|那个啥)"
    return bool(re.search(ambiguous_patterns, text))

# 路由决策: 输入特征 → ExtractionPlan
def route(features: RoutingFeatures) -> dict[str, ExtractionPath]:
    plan = {dim: ExtractionPath.LOCAL_ONLY for dim in CORE_DIMENSIONS}

    # 触发云端转发的条件
    if features.entity_density > 0.3:
        plan["entities"] = ExtractionPath.CLOUD_FORWARD
    if features.has_multi_hop_coreference:
        plan["entities"] = ExtractionPath.CLOUD_FORWARD  # 当前 L0 不抽 coref,但记录信号
    if features.input_length > 2000:
        # 长文本不转发(成本爆炸)
        plan["entities"] = ExtractionPath.LOCAL_ONLY

    return plan
```

**为什么用规则而非模型**:
- 规则**延迟微秒级**,模型路由本身是模型(递归问题,见 Oracle 评审)
- 规则**零训练成本**,模型路由需要训练数据
- 规则**完全可调试**,可以逐步调阈值
- 规则路由的失败模式 = 误判简单/复杂,但这是 P0 原型要发现的问题

**v0.1.1 修复**:`detect_coreference` / `detect_ambiguity` 之前是文档中的幽灵函数(MVP 不实现但被引用)。v0.1.1 改为**启发式正则**(heuristic_coreference_check / heuristic_ambiguity_check),代价是精度低,但作为路由信号足够——精度低的后果是"误转发一些简单 case 到云端",可在 M3 评估真实损失率。完整 coref 模型在 `mvp-schema.md` §5.1 backlog。

### 3.2 LocalExtractorPool 设计(v0.2 重写)

**v0.1.1 设计**(已废):3 个独立小模型(intent-cls-zh-0.5b + entity-ner-zh-0.5b + lang-detect-0.3b)+ 规则引擎

**v0.2 设计**:统一 Qwen3-0.6B base + task tag(对齐 [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) §4)

```python
class LocalExtractorPool:
    """统一模型抽取池(v1 和 v2 都用)
    
    v0.2 变更:从"3 个独立小模型"改为"统一 Qwen3-0.6B + task tag"
    - 对齐 08a D-10 tokenizer 硬约束(Qwen3 系列)
    - 同一 base 模型支持两种 task(session_extract / memory_extract)
    - P0 期间用"教师 API + 规则"interim,统一模型就绪后替换
    """

    def __init__(self):
        # v0.2 核心:统一模型(双 LoRA 或单多任务,见 08b §4)
        self.unified_model = UnifiedModel(
            base="Qwen3-0.6B",
            lora_memory="./models/lora_memory_v1/",     # 记忆抽取 LoRA
            lora_session="./models/lora_session_v1/",   # 会话抽取 LoRA
        )
        
        # 隐私与规则引擎仍独立(详见 §3.3 / §3.6)
        self.pii_redactor = LocalPIIRedactor()
        self.secret_detector = SecretDetector()  # v0.1.1 必加,见 mvp-schema §3.4
        self.rule_engine = RuleEngine([
            PathParser(), CodeFenceDetector(), URLDetector(),
        ])

    def extract(self, task: str, raw_input: str) -> Any:
        """按 task 分发到不同 LoRA"""
        if task == "session_extract":
            # 13 字段人工 schema 输出
            payload = self.unified_model.run(
                task="session_extract",
                input=raw_input,
                lora="session"
            )
            return payload  # {intent, entities, language, current_topic, ...}
        elif task == "memory_extract":
            # 自动涌现 schema 输出
            payload = self.unified_model.run(
                task="memory_extract",
                input=raw_input,
                lora="memory"
            )
            return payload  # {entities, relations}
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def extract_with_pii_check(self, task: str, raw_input: str) -> Any:
        """v0.2 新增:抽取前先做 PII 脱敏 + secret 检测"""
        cleaned = self.pii_redactor.redact(raw_input)
        secret_alerts = self.secret_detector.scan(raw_input)
        payload = self.extract(task, cleaned["text"])
        return {
            "payload": payload,
            "secret_alerts": secret_alerts,  # 强制保留(漏报代价 >> 误报)
            "pii_mapping": cleaned["mappings"],  # 反匿名化用
        }
```

**关键变化说明**:

| 维度 | v0.1.1 设计 | v0.2 设计 |
|---|---|---|
| **模型数量** | 3 个独立 0.5B 模型 | 1 个统一 Qwen3-0.6B(base + 双 LoRA) |
| **Task 区分** | 按 dimension 分发(intent/entities/language) | 按 task 分发(session_extract/memory_extract) |
| **训练数据** | 各自 SFT | task-tagged 混合(见 08b §3) |
| **interim 阶段** | 直接训练小模型 | 统一模型未就绪时用教师 API + 规则 |
| **PII/secret** | 规则引擎处理 | 独立预处理 + 抽取后处理 |
| **回滚能力** | 单模型粒度 | 双 LoRA 独立回滚 |

**v0.2 设计的不变量**(关键):
- LocalExtractorPool **接口不变**(`extract(task, input) -> payload`),实现可换
- v1(Python)和 v2(HydraForge Agent)都用同一个 LocalExtractorPool
- 统一模型失败时,可临时回退到教师 API + 规则(降级路径)

### 3.3 CloudForwarder 设计(白名单制)

```python
class CloudForwarder:
    """云端转发代理: 白名单制 + PII 脱敏"""

    def __init__(self):
        self.pii_redactor = LocalPIIRedactor()  # 本地 PII 脱敏器
        self.privacy_table = LocalPrivacyTable()  # placeholder ↔ 原值
        self.client = CloudClient()

    def forward(self, dimension: str, raw_input: str, white_list: list[str]) -> Any:
        # 1. 隐私白名单检查
        if dimension not in white_list:
            raise PrivacyViolation(f"{dimension} not in white list")

        # 2. PII 脱敏(本地)
        redacted_input, mappings = self.pii_redactor.redact(raw_input)

        # 3. 云端调用
        response = self.client.call(
            model="deepseek-v3",  # 或其他
            input=redacted_input,
            dimension=dimension,
            timeout_ms=3000,
        )

        # 4. 反匿名化(本地映射表还原)
        return self.privacy_table.de_anonymize(response, mappings)
```

**PII 脱敏召回率 SLO**:≥99.5%,**作为独立指标月度评测**(Oracle 评审建议)。不达标则默认全本地。

### 3.4 Schema 校验器(逐字段 confidence 校验 + 降级决策)

**v0.1.1 修复**:原代码 MINIMAL 不可达 + `or` 表达式重复,重写为字段集合模式匹配:

```python
class SchemaValidator:
    """逐字段 confidence 校验,触发降级
    降级决策只此一处,OrchestratorInterface.get_degradation_level 委托本类
    """

    # 降级阈值(单一真源,对应 mvp-schema.md §4)
    THRESHOLDS = {
        "intent": 0.7,
        "entities": 0.6,
        "language": 0.85,
        "current_topic": 0.6,
        "session_facts": 0.7,
    }

    def validate(self, turn_ctx: TurnContext) -> tuple[TurnContext, DegradationLevel]:
        """返回 (turn_ctx_with_degraded_marks, degradation_level)"""
        # 字段级 confidence 阈值检查
        degraded = set()
        for field, threshold in self.THRESHOLDS.items():
            if turn_ctx.field_confidence.get(field, 0.0) < threshold:
                degraded.add(field)
                turn_ctx.mark_degraded(field)

        # 字段集合模式匹配(替代原 if/elif 链)
        if not degraded:
            return turn_ctx, DegradationLevel.FULL
        if "intent" not in degraded:
            # intent OK,其他可能降级 → 保核心
            return turn_ctx, DegradationLevel.KEEP_CORE
        if degraded.issubset({"language", "current_topic", "session_facts"}):
            # intent 失败但其他还在 → 最小化(只 intent)
            return turn_ctx, DegradationLevel.MINIMAL
        return turn_ctx, DegradationLevel.RAW_PASSTHROUGH
```

**职责唯一性**:`get_degradation_level` 在 OrchestratorInterface 和 SchemaValidator 之前**重复**。v0.1.1 后:
- OrchestratorInterface.get_degradation_level 仅作 Protocol 方法,实现直接转发到 `SchemaValidator.validate(...).level`
- **降级决策只有 SchemaValidator 一处**

---

## 4. 4 级降级阶梯(决策 4 实施)

### 4.1 降级级别定义

```python
class DegradationLevel(Enum):
    FULL = "full"               # L0: 全量
    KEEP_CORE = "keep_core"     # L1: 保核心(intent + entities)
    MINIMAL = "minimal"         # L2: 仅核心(intent only)
    RAW_PASSTHROUGH = "raw"     # L3: raw 直通(不做任何抽取)
```

### 4.2 Prompt 模板行为

| 级别 | 触发条件 | Prompt 模板 | 典型场景 |
|---|---|---|---|
| **FULL** | 全部字段 confidence ≥ 阈值 | 完整模板 + 抽取上下文 | 正常路径 |
| **KEEP_CORE** | `extraction_quality` 中等 (0.6-0.8) | 简化模板(无 preferences、无 session_facts) | 本地模型部分失败 |
| **MINIMAL** | 仅 intent confidence ≥0.7 | 最小模板(仅意图提示) | 大部分抽取失败 |
| **RAW_PASSTHROUGH** | 抽取整体失败 | 直接拼历史对话,无任何抽取 | 网络断、模型宕 |

### 4.3 降级链配置

```yaml
# config/degradation.yaml
degradation_chain:
  - level: FULL
    requires:
      intent: 0.7
      entities: 0.6
      language: 0.85
      current_topic: 0.6

  - level: KEEP_CORE
    requires:
      intent: 0.7
      entities: 0.0  # entities 可为空

  - level: MINIMAL
    requires:
      intent: 0.7

  - level: RAW_PASSTHROUGH
    requires: {}  # 无要求
```

---

## 5. v2 演进路径(HydraForge 编排)

### 5.1 v2 启动条件

| 触发器 | 来源 | 行动 |
|---|---|---|
| HydraForge runtime CLI 可用 | AGENTS.md §9 "待开始" → 已完成 | 开始 v2 实现 |
| AgenticMind LLM TR-3 M5 完成 | pass@1 ≥80% | 用 AgenticMind LLM 生成编排 DSL |
| v2 A/B 测试通过 | 抽取质量 ≥ v1,延迟 ≤1.5x | 切换默认实现 |

### 5.2 v2 复用 v1 Python 工具的集成路径(关键)

**v0.1.1 显式标注**:v2 HydraForge C++ 引擎**不能直接调用 Python 类**,必须通过 **ToolRegistry 的 out-of-process 绑定**。

**集成方案**(在 v2 启动时实施):

```
┌──────────────────────────────────────────────┐
│  HydraForge C++ 引擎                         │
│  ├── ToolRegistry (C++)                      │
│  └── ToolBinding: out_of_process             │
└──────────────┬───────────────────────────────┘
               │ gRPC / Unix Socket
               ▼
┌──────────────────────────────────────────────┐
│  Python Sidecar(本地服务)                    │
│  ├── local_pool.extract(dim, input)          │
│  ├── schema_validator.validate(turn_ctx)     │
│  ├── session_state_updater.update(...)       │
│  ├── prompt_assembler.assemble(...)          │
│  └── cloud_forwarder.forward(...)            │
└──────────────────────────────────────────────┘
```

**实施细节**:
- **Sidecar 进程**:每个 HydraForge runtime 实例启动一个 Python sidecar,常驻内存
- **通信协议**:gRPC(性能)或 Unix Socket(低延迟),Protobuf 定义服务接口
- **超时控制**:每个工具调用 100ms 超时,失败由 Sidecar 返回降级结果
- **部署**:Docker 容器化,sidecar 与 runtime 同 Pod 部署

**成本评估**(v2 启动时确认):
- Sidecar 启动延迟:首次 500ms(加载模型),后续 <50ms(已加载)
- 进程间通信延迟:+2-5ms(本地) / +5-20ms(本地但经 Protobuf)
- 总编排延迟:v1 200ms → v2 300-500ms(在 P95 ≤500ms SLO 内)

**v0.1.1 补丁原因**:之前文档把这部分"一句话带过",实际是 v2 最大的集成成本,不可低估。

### 5.3 v2 的范围与限制

### 5.2 v2 的范围与限制

**v2 不是 v1 的全替换,而是 v1 的"可选升级"**:

- v1 仍保留为 fallback(降级到 Python 编排)
- v2 接管主路径,但要复用 v1 的所有 LocalExtractorPool、Schema 校验器、PromptAssembler
- v2 的 DSL 程序**由 AgenticMind LLM 生成**(自举演示),但需要人审 + 单元测试

### 5.3 v2 的"自举演示"价值

v2 是 **VN-001 自举愿景** 的最佳演示:
- AgenticMind LLM 训练出来就是为了生成 AgenticDSL
- 用它来生成"如何抽取对话上下文"的 DSL 程序,是**自举的最小可行闭环**
- 这是 TR-3 之后的关键里程碑产物(对应 `agenticdsl-training/README.md` §3 M7 "Agent 可独立生成 Skill")

### 5.4 v2 的风险与缓释

| 风险 | 缓释 |
|---|---|
| LLM 生成的 DSL 有 bug | Schema 校验器 + 单元测试 + A/B 对比 |
| v2 延迟放大 | 预算硬约束,超时即降级到 v1 |
| v2 失败导致全栈崩溃 | v2 与 v1 并存,失败回退到 v1 |

---

## 6. 关键不变量(无论 v1/v2 都成立)

| 不变量 | 说明 |
|---|---|
| **字段 schema 不变** | TurnContext 和 SessionState 的字段定义,v1/v2 必须相同 |
| **LocalExtractorPool 不变** | 抽取接口不变(`extract(task, input) -> payload`),v1/v2 共用 | v0.2 实现从"3 个独立 0.5B 模型"改为"统一 Qwen3-0.6B + task tag",但**接口契约**不变 |
| **降级行为不变** | 4 级降级阶梯,不论谁编排都按这套规则 |
| **隐私白名单不变** | 哪些字段可以出域,由 `mvp-schema.md` §3.4 锁定 |
| **PII 脱敏召回率 SLO 不变** | ≥99.5%,本地测量 |

---

## 7. 决策检查清单

实施前必须回答:

- [ ] **OrchestratorInterface 是否被 v1/v2 都遵循?**
  - 通过接口的 Protocol 定义 + 单测
- [ ] **v1 的 4 级降级是否能覆盖所有失败场景?**
  - P0 原型期间要制造 20+ 失败 case 验证
- [ ] **PII 脱敏召回率如何测量?**
  - 需要 1000+ 标注样本的测试集
- [ ] **v1 → v2 切换的 4 个条件是否有明确 owner?**
  - 建议每个条件绑定一个 issue 和负责人

---

## 8. 文档边界

| 内容 | 归属 |
|---|---|
| **会话管理智能体编排架构**(本文件) | AgenticMind 仓 `context-management/architecture.md` |
| **13 字段人工 schema** | `mvp-schema.md` |
| **P0 原型任务清单(interim)** | `p0-prototype-tasks.md` |
| **Schema 融合边界规范** | [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) |
| **HydraForge runtime 规范** | HydraForge 仓 |
| **AgenticDSL LLM 训练** | `../agenticdsl-training/` |
| **VN-001 自举愿景对齐** | `../agenticdsl-training/06-vn001-alignment.md` |

---

**文档版本**:v0.2(方向重定位:LocalExtractorPool 改为统一模型 + task tag)
**Owner**:AgenticMind 抽取系统架构组
**下一步**:进入 `p0-prototype-tasks.md` 任务清单,启动 v1 实现
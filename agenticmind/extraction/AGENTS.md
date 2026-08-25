# AGENTS.md — agenticmind/extraction/

## OVERVIEW

共享契约层：定义 13 字段 schema + validator + privacy detector，纯 stdlib，下游训练/运行时共同依赖。

## WHERE TO LOOK

| 文件 | 导出 | 修改时机 |
|---|---|---|
| `schemas.py` | `TurnContextL0`, `SessionStateL1`, 8 个 Field 类, 4 个 Enum | 13 字段语义变更 |
| `validator.py` | `Validator`, `ValidationResult` | 新增校验规则、调整权重 |
| `privacy.py` | `SecretDetector`, `SecretKind`, `SecretAlert`, `PIIRedactor` | 新增 secret 模式、扩展 PII 检测 |
| `__init__.py` | 全部公开 API re-export | 新增 public type 时同步更新 |

## 13 字段清单

### L0 per-turn（7 字段）

| 字段 | 类型 | 说明 |
|---|---|---|
| `intent` | `IntentField` | 8 类：question/command/clarify/confirm/correct/chat/refuse/meta |
| `entities` | `EntitiesField` | 9 类实体，**包含 `secret` 类型**（v0.2 新增）|
| `language` | `LanguageField` | 5 类：zh/en/code/mixed/other |
| `routing_features` | `RoutingFeatures` | 6 维路由特征 |
| `field_confidence` | `FieldConfidence` | 5 维置信度 |
| `extraction_provenance` | `ExtractionProvenance` | 抽取来源追踪 |
| `privacy_tier` | `PrivacyTier` | 白名单模型：PUBLIC / DOMAIN_PRIVATE / DERIVED_SENSITIVE |

### L1 per-session（3 字段）

| 字段 | 类型 | 说明 |
|---|---|---|
| `current_topic` | `TopicField` | 当前话题（不维护话题树 — 见 L234 占位）|
| `session_facts` | `SessionFactsField` | 扁平 KV 已确立事实 |
| `near_turn_entities` | `NearTurnEntitiesField` | 最近 5 轮实体缓存 |

### 会话标识（独立于 13 字段）

`session_id: str` · `turn_index: int`

## 关键不变量

```
secret entity → privacy_tier MUST be DERIVED_SENSITIVE
```

漏报代价 >> 误报。validator 强制此校验，不可绕过。

## CONVENTIONS

- **双真源同步**：改 `schemas.py` 字段语义 → 必须同步更新 `docs/agenticmind/context-management/mvp-schema.md` 对应章节（§3.1 L0 / §3.2 L1 / §3.4 横切元数据）
- **docstring 引用**：每个字段类的 docstring 需注明对应 schema 文档章节
- **纯 stdlib**：不引入 pydantic / serde 等外部依赖；数据定义用标准 dataclass + enum
- **公开 API 最小化**：仅 `__init__.py` re-export 的类型为公开 API，内部实现细节不得直接依赖

## ANTI-PATTERNS

- **NEVER** 修改 13 字段语义或类型——破坏 schema 融合边界（08b 决策）
- **NEVER** 放宽 `secret → DERIVED_SENSITIVE` 校验——隐私泄漏风险
- **NEVER** 在 `extraction/` 目录内放训练脚本或编排代码——违反 F-05 归属决策

## NOTES

- 消费方现状：仅 `agenticmind/tests/test_extraction.py` 接入；`agenticmemory_training/` 应接入但未实装
- `privacy.py` L186：P1-2 完整 PII 模式（中文姓名/电话/身份证）占位待办
- `schemas.py` L234：`TopicField` 不维护话题树，进 backlog

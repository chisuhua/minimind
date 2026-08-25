"""AgenticMind 共享契约层(shared contract)

设计定位:
- 训练侧(`agenticmemory_training/`)和运行时侧(`context-management/architecture.md`)
  共用的数据结构与隐私/校验工具
- v0.2 字段集(13 字段 + 横切元数据),见 mvp-schema.md
- 不依赖任何外部模型/服务(纯 dataclass + regex)

子模块:
- extraction/schemas    13 字段 dataclass
- extraction/validator  SchemaValidator(必填/枚举/confidence/secret 隐私)
- extraction/privacy    SecretDetector(6 类) + PIIRedactor

参考:
- docs/agenticmind/context-management/mvp-schema.md §3
- docs/agenticmemory_training/08b-seed-schema-fusion.md §3
- docs/agenticmemory_training/08c-p1-minimum-loop.md(P1 骨架)

不要在此包内放训练脚本或运行时编排代码。
训练脚本应放 agenticmemory_training/,运行时编排待 P2 落 agenticmind_runtime/(预留)。
"""

from agenticmind.extraction.schemas import (
    TurnContextL0,
    SessionStateL1,
    IntentField,
    IntentEnum,
    EntitiesField,
    Entity,
    EntityType,
    LanguageField,
    LanguageEnum,
    RoutingFeatures,
    FieldConfidence,
    ExtractionProvenance,
    PrivacyTier,
    PrivacyLevel,
    TopicField,
    SessionFactsField,
    FactEntry,
    NearTurnEntitiesField,
    EntityMention,
    ProvenanceTag,
)
from agenticmind.extraction.validator import SchemaValidator, ValidationResult

__all__ = [
    "TurnContextL0",
    "SessionStateL1",
    "IntentField",
    "IntentEnum",
    "EntitiesField",
    "Entity",
    "EntityType",
    "LanguageField",
    "LanguageEnum",
    "RoutingFeatures",
    "FieldConfidence",
    "ExtractionProvenance",
    "PrivacyTier",
    "PrivacyLevel",
    "TopicField",
    "SessionFactsField",
    "FactEntry",
    "NearTurnEntitiesField",
    "EntityMention",
    "ProvenanceTag",
    "SchemaValidator",
    "ValidationResult",
]
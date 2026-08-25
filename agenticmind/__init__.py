"""AgenticMind 抽取模块(最小闭环实验 P1-0 产出)

设计目标:
- 作为 session_extract / memory_extract 任务的统一数据结构
- v0.2 字段集(13 字段 + 横切元数据)
- 不依赖任何外部模型/服务(纯 dataclass)

参考:
- docs/agenticmind/context-management/mvp-schema.md §3 (字段定义)
- docs/agenticmemory_training/08b-seed-schema-fusion.md §3 (Schema 融合边界)
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
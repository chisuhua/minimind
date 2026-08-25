"""v0.2 Schema dataclass(13 字段 + 横切元数据)

设计来源:
- docs/agenticmind/context-management/mvp-schema.md §3.1, §3.2, §3.4
- 13 字段:6 业务字段 + 横切元数据类型(L0/L1 各实例化一次)
- v0.1.1 补丁:session_id/turn_index + secret 实体 + temperature scaling
- v0.2 补丁:intent primary+secondary,confidence_per_label

不依赖任何外部库(纯 stdlib:dataclass/enum/typing)。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Mapping


# ============================================================================
# Enums
# ============================================================================


class IntentEnum(str, Enum):
    """8 类意图分类(粗粒度封闭集合)

    见 mvp-schema.md §3.1 详细定义。
    """

    QUESTION = "question"  # 知识/答案查询
    COMMAND = "command"  # 操作请求
    CLARIFY = "clarify"  # 要求澄清
    CONFIRM = "confirm"  # 确认类
    CORRECT = "correct"  # 纠错类
    CHAT = "chat"  # 闲聊
    REFUSE = "refuse"  # 拒答类
    META = "meta"  # 关于助手自身的请求


class EntityType(str, Enum):
    """9 种实体类型(v0.2 含 secret)

    与 docs/agenticmemory_training/08b-seed-schema-fusion.md §3.2 的
    9 种实体类型对齐(可参与自动涌现锚定的字段)。
    """

    SECRET = "secret"  # API key/token/密码(derived_sensitive)
    PERSON = "person"  # 人名(domain_private)
    PROJECT = "project"  # 项目名
    FILE_PATH = "file_path"  # 文件路径
    URL = "url"  # URL
    VERSION = "version"  # 版本号
    CODE_SYMBOL = "code_symbol"  # 代码符号
    API = "api"  # API 名称
    ORG = "org"  # 组织/公司


class LanguageEnum(str, Enum):
    """语言检测(zh/en/code/mixed/other)"""

    ZH = "zh"
    EN = "en"
    CODE = "code"
    MIXED = "mixed"
    OTHER = "other"


class PrivacyLevel(str, Enum):
    """隐私分级(白名单制)

    见 mvp-schema.md §3.4
    - PUBLIC: 允许出域
    - DOMAIN_PRIVATE: 仅同域内允许
    - DERIVED_SENSITIVE: 禁止出域(派生敏感)
    """

    PUBLIC = "public"
    DOMAIN_PRIVATE = "domain_private"
    DERIVED_SENSITIVE = "derived_sensitive"


# ============================================================================
# 横切元数据类型(L0/L1 各实例化一次)
# ============================================================================


@dataclass
class ProvenanceTag:
    """抽取来源追踪"""

    extractor_id: str  # e.g. "intent-cls-zh-0.5b-v1"
    extractor_version: str
    extraction_path: str = "local"  # local|hybrid|cloud_forward
    timestamp: str = ""  # ISO8601
    fallback_used: bool = False


@dataclass
class FieldConfidence:
    """逐字段 confidence(softmax + temperature scaling 校准)

    详见 mvp-schema.md §3.3(置信度产生机制定义)。
    """

    intent: float = 1.0
    entities: float = 1.0
    language: float = 1.0
    routing_features: float = 1.0  # 通常为 1.0(规则生成)
    extraction_quality: float = 1.0  # 整体抽取质量估计

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ExtractionProvenance:
    """抽取来源追踪(L0 层实例化)"""

    extractor_id: str
    extractor_version: str
    extraction_path: str = "local"
    timestamp: str = ""
    latency_ms: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class PrivacyTier:
    """字段级隐私分级(白名单制)

    v0.2.1 加 secret/person/entity.value 三字段粒度。
    """

    intent: PrivacyLevel = PrivacyLevel.PUBLIC
    entities: PrivacyLevel = PrivacyLevel.DOMAIN_PRIVATE
    language: PrivacyLevel = PrivacyLevel.PUBLIC
    secret: PrivacyLevel = PrivacyLevel.DERIVED_SENSITIVE  # 默认值


# ============================================================================
# L0 业务字段
# ============================================================================


@dataclass
class IntentField:
    """意图分类(primary + optional secondary)

    v0.2 修订:支持多意图(例 "不对,应该用 GRPO" = correct+command)。
    """

    primary: IntentEnum
    secondary: list[IntentEnum] = field(default_factory=list)
    confidence: float = 0.0
    confidence_per_label: dict[str, float] = field(default_factory=dict)
    provenance: ProvenanceTag = field(default_factory=lambda: ProvenanceTag("", ""))


@dataclass
class Entity:
    """命名实体一条"""

    type: EntityType
    value: str
    span: tuple[int, int] = (-1, -1)  # 字符偏移 [start, end]
    confidence: float = 0.0
    privacy_tier: PrivacyLevel = PrivacyLevel.DOMAIN_PRIVATE
    context: str = ""  # 周围 ~10 字符


@dataclass
class EntitiesField:
    """命名实体列表"""

    items: list[Entity] = field(default_factory=list)
    aggregate_confidence: float = 0.0
    provenance: ProvenanceTag = field(default_factory=lambda: ProvenanceTag("", ""))


@dataclass
class LanguageField:
    """语言检测"""

    primary: LanguageEnum = LanguageEnum.OTHER
    secondary: list[LanguageEnum] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RoutingFeatures:
    """路由特征(给编排器用)

    v0.2 修订:has_multi_hop_coreference 改用启发式正则
    (见 architecture.md §3.1 heuristic_coreference_check)
    """

    input_length: int = 0
    entity_density: float = 0.0  # 实体数 / 长度
    has_multi_hop_coreference: bool = False  # 启发式检测
    has_ambiguous_referent: bool = False
    code_block_count: int = 0
    cost_estimate: float = 0.0


@dataclass
class TurnContextL0:
    """L0(per-turn)字段集(会话标识 + 6 业务 + 4 横切)"""

    session_id: str = ""
    turn_index: int = 0
    intent: IntentField = field(default_factory=lambda: IntentField(IntentEnum.CHAT))
    entities: EntitiesField = field(default_factory=EntitiesField)
    language: LanguageField = field(default_factory=LanguageField)
    routing_features: RoutingFeatures = field(default_factory=RoutingFeatures)
    field_confidence: FieldConfidence = field(default_factory=FieldConfidence)
    extraction_provenance: ExtractionProvenance = field(
        default_factory=lambda: ExtractionProvenance("", "")
    )
    privacy_tier: PrivacyTier = field(default_factory=PrivacyTier)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# L1 业务字段
# ============================================================================


@dataclass
class TopicField:
    """当前话题(单值,非树)

    v0.2 约束:不维护话题树(进 backlog)。
    """

    value: str = ""
    since_turn: int = 0
    confidence: float = 0.0
    privacy_tier: PrivacyLevel = PrivacyLevel.DOMAIN_PRIVATE


@dataclass
class FactEntry:
    """会话确立的事实(扁平 KV 列表中一条)

    字段语义:
    - key: 唯一标识(如 "user_name")
    - value: JSON-serializable 值
    - source_turn: 确立于第几轮
    - superseded_by: 若被覆盖,指向新版本
    """

    key: str
    value: Any
    source_turn: int = 0
    confidence: float = 0.0
    provenance: ProvenanceTag = field(default_factory=lambda: ProvenanceTag("", ""))
    superseded_by: int | None = None


@dataclass
class SessionFactsField:
    """会话确立的事实(扁平 KV 列表)"""

    items: list[FactEntry] = field(default_factory=list)
    last_updated_turn: int = 0


@dataclass
class EntityMention:
    """近 N 轮实体提及缓存(一条)"""

    entity_ref: str  # 引用 Entity.value(字符串引用,v0.2 不规范化)
    turn: int = 0
    span: tuple[int, int] = (-1, -1)
    role_in_turn: str = "unknown"  # subject|object|modifier|unknown


@dataclass
class NearTurnEntitiesField:
    """近 N 轮实体提及缓存(MVP: N=5)"""

    window_size: int = 5
    items: list[EntityMention] = field(default_factory=list)


@dataclass
class SessionStateL1:
    """L1(per-session)字段集(会话标识 + 3 业务 + 横切元数据)"""

    session_id: str = ""
    created_at: str = ""
    last_active_turn: int = 0
    current_topic: TopicField = field(default_factory=TopicField)
    session_facts: SessionFactsField = field(default_factory=SessionFactsField)
    near_turn_entities: NearTurnEntitiesField = field(default_factory=NearTurnEntitiesField)
    field_confidence: FieldConfidence = field(default_factory=FieldConfidence)
    extraction_provenance: ProvenanceTag = field(
        default_factory=lambda: ProvenanceTag("", "")
    )
    privacy_tier: PrivacyTier = field(default_factory=PrivacyTier)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
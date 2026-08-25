"""JSON Schema 校验器

设计来源:
- docs/agenticmind/context-management/mvp-schema.md §6.1(JSON Schema 校验)
- 单一真源,所有字段通过 JSON Schema 校验

设计要点:
- 序列化 dataclass 为 JSON 后做 schema 校验
- 不引入额外依赖(jsonschema 库可选,默认用纯 Python dict 比较)
- ValidationResult 包含 errors/warnings 两类
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .schemas import (
    TurnContextL0,
    SessionStateL1,
    Entity,
    EntityType,
    IntentEnum,
    LanguageEnum,
    PrivacyLevel,
)


class ValidationResult:
    """校验结果"""

    def __init__(self, ok: bool, errors: list[str] | None = None, warnings: list[str] | None = None):
        self.ok = ok
        self.errors = errors or []
        self.warnings = warnings or []

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        if self.ok:
            return f"ValidationResult(ok=True, warnings={len(self.warnings)})"
        return f"ValidationResult(ok=False, errors={len(self.errors)}, warnings={len(self.warnings)})"

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


class SchemaValidator:
    """TurnContextL0 / SessionStateL1 字段级 + 枚举级校验

    用法:
        v = SchemaValidator()
        result = v.validate_l0(turn_ctx)
        if not result:
            print(result.errors)
    """

    # L0 必填字段(校验非空)
    L0_REQUIRED = ["session_id", "turn_index"]
    # L1 必填字段
    L1_REQUIRED = ["session_id"]

    # secret 实体必须 derived_sensitive
    SECRET_REQUIRED_PRIVACY = PrivacyLevel.DERIVED_SENSITIVE

    def validate_l0(self, ctx: TurnContextL0) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        # 1. 会话标识必填
        if not ctx.session_id:
            errors.append("L0.session_id 不能为空")
        if ctx.turn_index < 0:
            errors.append(f"L0.turn_index 必须 ≥ 0,当前 {ctx.turn_index}")

        # 2. 业务字段范围
        if not isinstance(ctx.intent.primary, IntentEnum):
            errors.append(f"L0.intent.primary 必须是 IntentEnum,当前 {type(ctx.intent.primary).__name__}")
        for sec in ctx.intent.secondary:
            if not isinstance(sec, IntentEnum):
                errors.append(f"L0.intent.secondary 含非 IntentEnum: {type(sec).__name__}")

        if not isinstance(ctx.language.primary, LanguageEnum):
            errors.append(f"L0.language.primary 必须是 LanguageEnum,当前 {type(ctx.language.primary).__name__}")

        # 3. confidence 范围 [0, 1]
        if not 0.0 <= ctx.intent.confidence <= 1.0:
            errors.append(f"L0.intent.confidence 必须在 [0,1],当前 {ctx.intent.confidence}")
        if not 0.0 <= ctx.language.confidence <= 1.0:
            errors.append(f"L0.language.confidence 必须在 [0,1],当前 {ctx.language.confidence}")

        # 4. entity.value  非空,type 必须是 EntityType
        for i, ent in enumerate(ctx.entities.items):
            if not ent.value:
                errors.append(f"L0.entities.items[{i}].value 不能为空")
            if not isinstance(ent.type, EntityType):
                errors.append(
                    f"L0.entities.items[{i}].type 必须是 EntityType,当前 {type(ent.type).__name__}"
                )
            # secret 实体必须 derived_sensitive
            if ent.type == EntityType.SECRET:
                if ent.privacy_tier != self.SECRET_REQUIRED_PRIVACY:
                    errors.append(
                        f"L0.entities.items[{i}] 是 secret 类型,但 privacy_tier={ent.privacy_tier},"
                        f"必须是 {self.SECRET_REQUIRED_PRIVACY.value}"
                    )

        # 5. field_confidence 各值在 [0, 1]
        for fname in ("intent", "entities", "language", "routing_features", "extraction_quality"):
            v = getattr(ctx.field_confidence, fname)
            if not 0.0 <= v <= 1.0:
                errors.append(f"L0.field_confidence.{fname} 必须在 [0,1],当前 {v}")

        # 6. secret_alerts 字段(L0 不直接持有,见 extractor 输出层)— 此处不校验

        # Warnings:非阻断
        if ctx.entities.items and ctx.entities.aggregate_confidence == 0.0:
            warnings.append("L0.entities.items 非空但 aggregate_confidence=0.0,可能未正确计算")

        return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_l1(self, state: SessionStateL1) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if not state.session_id:
            errors.append("L1.session_id 不能为空")
        if state.last_active_turn < 0:
            errors.append(f"L1.last_active_turn 必须 ≥ 0,当前 {state.last_active_turn}")

        # current_topic
        if state.current_topic.value and not 0.0 <= state.current_topic.confidence <= 1.0:
            errors.append(f"L1.current_topic.confidence 必须在 [0,1],当前 {state.current_topic.confidence}")

        # session_facts key/value 非空
        for i, fact in enumerate(state.session_facts.items):
            if not fact.key:
                errors.append(f"L1.session_facts.items[{i}].key 不能为空")

        # near_turn_entities window_size
        if state.near_turn_entities.window_size <= 0:
            warnings.append(f"L1.near_turn_entities.window_size={state.near_turn_entities.window_size},异常")

        return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def to_json(ctx: TurnContextL0 | SessionStateL1) -> str:
        """序列化为 JSON 字符串(供下游 prompt 组装/调试用)"""
        return json.dumps(asdict(ctx), ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def from_l0_dict(data: dict[str, Any]) -> TurnContextL0:
        """从 dict 反序列化(供测试 / API 输入用)

        支持 enum 字符串值自动转换。
        """
        from .schemas import (
            IntentField,
            EntitiesField,
            LanguageField,
            RoutingFeatures,
            FieldConfidence,
            ExtractionProvenance,
            PrivacyTier,
        )

        # 处理嵌套 enum
        intent_data = data.get("intent", {})
        if isinstance(intent_data.get("primary"), str):
            intent_data["primary"] = IntentEnum(intent_data["primary"])
        if isinstance(intent_data.get("secondary"), list):
            intent_data["secondary"] = [
                IntentEnum(s) if isinstance(s, str) else s for s in intent_data["secondary"]
            ]

        lang_data = data.get("language", {})
        if isinstance(lang_data.get("primary"), str):
            lang_data["primary"] = LanguageEnum(lang_data["primary"])

        entities_data = data.get("entities", {})
        if isinstance(entities_data.get("items"), list):
            for item in entities_data["items"]:
                if isinstance(item.get("type"), str):
                    item["type"] = EntityType(item["type"])
                if isinstance(item.get("privacy_tier"), str):
                    item["privacy_tier"] = PrivacyLevel(item["privacy_tier"])

        return TurnContextL0(
            session_id=data.get("session_id", ""),
            turn_index=data.get("turn_index", 0),
            intent=IntentField(**intent_data) if intent_data else IntentField(IntentEnum.CHAT),
            entities=EntitiesField(**entities_data),
            language=LanguageField(**lang_data),
            routing_features=RoutingFeatures(**data.get("routing_features", {})),
            field_confidence=FieldConfidence(**data.get("field_confidence", {})),
            extraction_provenance=ExtractionProvenance(**data.get("extraction_provenance", {})),
            privacy_tier=PrivacyTier(**data.get("privacy_tier", {})),
        )
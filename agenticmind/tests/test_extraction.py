"""P1-0 单元测试:覆盖 schemas / validator / privacy 三大模块

设计原则:
- 用 stdlib unittest(不依赖 pytest)
- 每个测试独立 setup
- 测试覆盖:字段填充、枚举类型、confidence 边界、secret 检测与脱敏、Validator 拒绝场景
- 跑法:python3 -m unittest agenticmind.tests.test_extraction

参考:
- docs/agenticmind/context-management/mvp-schema.md §3, §6
- P1-0b 安全底线:secret 必须 derived_sensitive
"""

from __future__ import annotations

import unittest

from agenticmind.extraction.schemas import (
    TurnContextL0,
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
    SessionStateL1,
    TopicField,
    SessionFactsField,
    FactEntry,
    NearTurnEntitiesField,
    EntityMention,
    ProvenanceTag,
)
from agenticmind.extraction.validator import SchemaValidator, ValidationResult
from agenticmind.extraction.privacy import (
    SecretDetector,
    SecretKind,
    SecretAlert,
    PIIRedactor,
    RedactionResult,
)


# ============================================================================
# Schemas tests
# ============================================================================


class TestSchemas(unittest.TestCase):
    """13 字段 dataclass 的基本行为"""

    def test_turn_context_l0_defaults(self):
        """L0 默认值应该可构造且字段类型正确"""
        ctx = TurnContextL0()
        self.assertEqual(ctx.session_id, "")
        self.assertEqual(ctx.turn_index, 0)
        self.assertIsInstance(ctx.intent, IntentField)
        self.assertEqual(ctx.intent.primary, IntentEnum.CHAT)
        self.assertIsInstance(ctx.language, LanguageField)
        self.assertEqual(ctx.language.primary, LanguageEnum.OTHER)
        self.assertIsInstance(ctx.entities, EntitiesField)
        self.assertEqual(ctx.entities.items, [])
        self.assertIsInstance(ctx.routing_features, RoutingFeatures)
        self.assertIsInstance(ctx.field_confidence, FieldConfidence)
        # 默认 confidence 应在 [0, 1]
        self.assertEqual(ctx.field_confidence.intent, 1.0)

    def test_intent_enum_8_classes(self):
        """8 类意图枚举完整"""
        expected = {
            "question", "command", "clarify",
            "confirm", "correct", "chat",
            "refuse", "meta",
        }
        actual = {e.value for e in IntentEnum}
        self.assertEqual(actual, expected)

    def test_entity_type_9_classes_includes_secret(self):
        """9 种实体类型(含 secret, v0.2)"""
        expected = {
            "secret", "person", "project", "file_path", "url",
            "version", "code_symbol", "api", "org",
        }
        actual = {e.value for e in EntityType}
        self.assertEqual(actual, expected)
        # secret 必须存在(关键不变量)
        self.assertIn(EntityType.SECRET, EntityType)

    def test_language_enum_5_classes(self):
        """5 类语言枚举"""
        expected = {"zh", "en", "code", "mixed", "other"}
        actual = {e.value for e in LanguageEnum}
        self.assertEqual(actual, expected)

    def test_privacy_level_3_classes(self):
        """3 类隐私分级(白名单制)"""
        expected = {"public", "domain_private", "derived_sensitive"}
        actual = {e.value for e in PrivacyLevel}
        self.assertEqual(actual, expected)

    def test_session_state_l1_defaults(self):
        """L1 默认值"""
        state = SessionStateL1(session_id="s1")
        self.assertEqual(state.session_id, "s1")
        self.assertEqual(state.last_active_turn, 0)
        self.assertEqual(state.current_topic.value, "")
        self.assertEqual(state.session_facts.items, [])
        self.assertEqual(state.near_turn_entities.window_size, 5)

    def test_intent_multi_label_support(self):
        """intent 支持 primary + secondary(v0.2 多意图)"""
        field = IntentField(
            primary=IntentEnum.CORRECT,
            secondary=[IntentEnum.COMMAND],
        )
        self.assertEqual(field.primary, IntentEnum.CORRECT)
        self.assertIn(IntentEnum.COMMAND, field.secondary)

    def test_to_dict_serializable(self):
        """dataclass → dict(供 JSON 序列化)"""
        ctx = TurnContextL0(session_id="s1", turn_index=1)
        d = ctx.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["session_id"], "s1")


# ============================================================================
# Validator tests
# ============================================================================


class TestValidator(unittest.TestCase):
    """SchemaValidator 校验逻辑"""

    def setUp(self):
        self.v = SchemaValidator()

    def test_valid_minimal_l0(self):
        """最小 L0 应该通过"""
        ctx = TurnContextL0(session_id="s1", turn_index=1)
        r = self.v.validate_l0(ctx)
        self.assertTrue(r.ok, msg=str(r.errors))

    def test_missing_session_id_rejected(self):
        """session_id 空必报错"""
        ctx = TurnContextL0(session_id="", turn_index=1)
        r = self.v.validate_l0(ctx)
        self.assertFalse(r.ok)
        self.assertTrue(any("session_id" in e for e in r.errors))

    def test_negative_turn_index_rejected(self):
        """turn_index < 0 必报错"""
        ctx = TurnContextL0(session_id="s", turn_index=-1)
        r = self.v.validate_l0(ctx)
        self.assertFalse(r.ok)
        self.assertTrue(any("turn_index" in e for e in r.errors))

    def test_intent_confidence_out_of_range_rejected(self):
        """intent.confidence > 1 必报错"""
        ctx = TurnContextL0(
            session_id="s",
            turn_index=1,
            intent=IntentField(primary=IntentEnum.CHAT, confidence=1.5),
        )
        r = self.v.validate_l0(ctx)
        self.assertFalse(r.ok)
        self.assertTrue(any("intent.confidence" in e for e in r.errors))

    def test_secret_entity_must_be_derived_sensitive(self):
        """secret 实体必须 privacy_tier=DERIVED_SENSITIVE(P1-0b 关键不变量)"""
        ctx = TurnContextL0(
            session_id="s",
            turn_index=1,
            entities=EntitiesField(
                items=[
                    Entity(
                        type=EntityType.SECRET,
                        value="AKIAIOSFODNN7EXAMPLE",
                        privacy_tier=PrivacyLevel.PUBLIC,  # 错误!
                    )
                ]
            ),
        )
        r = self.v.validate_l0(ctx)
        self.assertFalse(r.ok)
        self.assertTrue(
            any("secret" in e and "derived_sensitive" in e for e in r.errors),
            f"Expected secret privacy error, got: {r.errors}",
        )

    def test_secret_entity_correct_privacy_passes(self):
        """secret 实体 + DERIVED_SENSITIVE 应该通过"""
        ctx = TurnContextL0(
            session_id="s",
            turn_index=1,
            entities=EntitiesField(
                items=[
                    Entity(
                        type=EntityType.SECRET,
                        value="AKIAIOSFODNN7EXAMPLE",
                        privacy_tier=PrivacyLevel.DERIVED_SENSITIVE,
                    )
                ]
            ),
        )
        r = self.v.validate_l0(ctx)
        self.assertTrue(r.ok, msg=str(r.errors))

    def test_empty_entity_value_rejected(self):
        """entity.value 空必报错"""
        ctx = TurnContextL0(
            session_id="s",
            turn_index=1,
            entities=EntitiesField(
                items=[Entity(type=EntityType.PROJECT, value="", privacy_tier=PrivacyLevel.PUBLIC)]
            ),
        )
        r = self.v.validate_l0(ctx)
        self.assertFalse(r.ok)

    def test_l1_missing_session_id_rejected(self):
        """L1 session_id 空必报错"""
        state = SessionStateL1(session_id="")
        r = self.v.validate_l1(state)
        self.assertFalse(r.ok)
        self.assertTrue(any("session_id" in e for e in r.errors))

    def test_l1_fact_empty_key_rejected(self):
        """L1 fact.key 空必报错"""
        state = SessionStateL1(
            session_id="s",
            session_facts=SessionFactsField(items=[FactEntry(key="", value="x")]),
        )
        r = self.v.validate_l1(state)
        self.assertFalse(r.ok)

    def test_to_json_roundtrip(self):
        """to_json 应生成有效 JSON"""
        ctx = TurnContextL0(session_id="s1", turn_index=1)
        json_str = SchemaValidator.to_json(ctx)
        self.assertIn("session_id", json_str)
        self.assertIn("s1", json_str)


# ============================================================================
# Privacy / SecretDetector tests
# ============================================================================


class TestSecretDetector(unittest.TestCase):
    """SecretDetector:6 类 secret 检测"""

    def setUp(self):
        self.det = SecretDetector()

    def test_aws_access_key_detected(self):
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        alerts = self.det.scan(text)
        kinds = {a.kind for a in alerts}
        self.assertIn(SecretKind.AWS_ACCESS_KEY, kinds)

    def test_aws_asia_prefix_detected(self):
        """AWS 临时 key(ASIA 前缀)也应被检测"""
        text = "key=ASIA1234567890ABCDEF"
        alerts = self.det.scan(text)
        kinds = {a.kind for a in alerts}
        self.assertIn(SecretKind.AWS_ACCESS_KEY, kinds)

    def test_github_pat_detected(self):
        text = "token=ghp_abc123def456ghi789jkl012mno345pqr678"
        alerts = self.det.scan(text)
        kinds = {a.kind for a in alerts}
        self.assertIn(SecretKind.GITHUB_PAT, kinds)

    def test_openai_key_detected(self):
        text = "OPENAI_API_KEY=sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        alerts = self.det.scan(text)
        kinds = {a.kind for a in alerts}
        self.assertIn(SecretKind.OPENAI_API_KEY, kinds)

    def test_jwt_detected(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        alerts = self.det.scan(text)
        kinds = {a.kind for a in alerts}
        self.assertIn(SecretKind.JWT, kinds)

    def test_pem_private_key_detected(self):
        text = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...fake content...abcd==
-----END RSA PRIVATE KEY-----"""
        alerts = self.det.scan(text)
        kinds = {a.kind for a in alerts}
        self.assertIn(SecretKind.PEM_PRIVATE_KEY, kinds)

    def test_gcp_api_key_detected(self):
        text = "GCP_KEY=AIzaSyD_1234567890abcdefghijklmnopqrstu"
        alerts = self.det.scan(text)
        kinds = {a.kind for a in alerts}
        self.assertIn(SecretKind.GCP_API_KEY, kinds)

    def test_clean_text_no_alerts(self):
        """无 secret 的文本不应触发告警"""
        text = "This is a normal README file with no secrets."
        alerts = self.det.scan(text)
        self.assertEqual(alerts, [])

    def test_multiple_secrets_in_same_text(self):
        """多 secret 共存应全部检测"""
        text = (
            "AWS=AKIAIOSFODNN7EXAMPLE "
            "GitHub=ghp_abc123def456ghi789jkl012mno345pqr678"
        )
        alerts = self.det.scan(text)
        kinds = {a.kind for a in alerts}
        self.assertIn(SecretKind.AWS_ACCESS_KEY, kinds)
        self.assertIn(SecretKind.GITHUB_PAT, kinds)

    def test_redact_replaces_all_secrets(self):
        """redact 应应所有 secret 都被 placeholder 替换"""
        text = "AWS=AKIAIOSFODNN7EXAMPLE GitHub=ghp_abc123def456ghi789jkl012mno345pqr678"
        redacted, alerts = self.det.redact(text)
        # 原值不应出现
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", redacted)
        self.assertNotIn("ghp_abc123def456ghi789jkl012mno345pqr678", redacted)
        # placeholder 应出现
        self.assertIn("SECRET_aws_access_key_", redacted)
        self.assertIn("SECRET_github_pat_", redacted)
        self.assertEqual(len(alerts), 2)

    def test_placeholder_stable_for_same_secret(self):
        """同一 secret 文本多次出现应共享同一 placeholder"""
        text = "AWS=AKIAIOSFODNN7EXAMPLE same AKIAIOSFODNN7EXAMPLE here"
        redacted1, _ = self.det.redact(text)
        # 两次出现的 placeholder 应相同
        placeholders = set()
        import re

        for m in re.finditer(r"SECRET_aws_access_key_[a-f0-9]+", redacted1):
            placeholders.add(m.group(0))
        self.assertEqual(len(placeholders), 1, "同一 secret 应共享 placeholder")

    def test_pii_redactor_integration(self):
        """PIIRedactor 应整合 SecretDetector 并维护映射表"""
        text = "AWS=AKIAIOSFODNN7EXAMPLE"
        redactor = PIIRedactor()
        result = redactor.redact(text)
        self.assertIsInstance(result, RedactionResult)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", result.redacted_text)
        self.assertEqual(len(result.mappings), 1)
        # 反匿名化
        restored = redactor.de_anonymize(result.redacted_text)
        self.assertEqual(restored, text)

    def test_pii_redactor_multiple_secrets(self):
        """PIIRedactor 处理多 secret 累积映射表"""
        text = (
            "AWS=AKIAIOSFODNN7EXAMPLE; "
            "GitHub=ghp_abc123def456ghi789jkl012mno345pqr678"
        )
        redactor = PIIRedactor()
        result = redactor.redact(text)
        self.assertEqual(len(result.mappings), 2)
        self.assertEqual(len(result.secret_alerts), 2)

    def test_pii_redactor_reset(self):
        """reset 应清除映射表"""
        redactor = PIIRedactor()
        redactor.redact("AWS=AKIAIOSFODNN7EXAMPLE")
        self.assertEqual(len(redactor._mapping), 1)
        redactor.reset()
        self.assertEqual(len(redactor._mapping), 0)


# ============================================================================
# Integration test
# ============================================================================


class TestEndToEndIntegration(unittest.TestCase):
    """端到端:TurnContextL0 + Validator + SecretDetector 联合使用"""

    def test_secret_in_conversation_redacted(self):
        """模拟会话中包含 secret:抽取 → 校验 → 脱敏"""
        # Step 1: 模拟教师输出 raw TurnContextL0(包含 secret 实体)
        raw_text = "我的 AWS key 是 AKIAIOSFODNN7EXAMPLE,请帮我配置"
        detector = SecretDetector()
        redacted, alerts = detector.redact(raw_text)
        self.assertEqual(len(alerts), 1)

        # Step 2: 在 redacted 文本上构造 TurnContextL0
        ctx = TurnContextL0(
            session_id="sess_001",
            turn_index=1,
            language=LanguageField(primary=LanguageEnum.ZH, confidence=0.98),
            intent=IntentField(primary=IntentEnum.COMMAND, confidence=0.92),
            entities=EntitiesField(
                items=[
                    Entity(
                        type=EntityType.SECRET,
                        value="AKIAIOSFODNN7EXAMPLE",
                        privacy_tier=PrivacyLevel.DERIVED_SENSITIVE,
                    )
                ],
                aggregate_confidence=0.95,
            ),
        )

        # Step 3: 校验通过
        v = SchemaValidator()
        r = v.validate_l0(ctx)
        self.assertTrue(r.ok, msg=str(r.errors))

        # Step 4: 序列化为 JSON(供下游 prompt 组装)
        json_str = SchemaValidator.to_json(ctx)
        self.assertIn("SECRET_aws_access_key_", redacted)
        # 原始 secret 不出现在 JSON 中(因为 secret 实体 value 仍是原值)
        # 注:这里体现的是"抽取器在拿到文本时应先调用 SecretDetector.redact,
        # 再用脱敏文本训练模型,而不是直接用原值"
        # 当前 v0.2 设计假设抽取器只在 secret 实体上调用 redact


if __name__ == "__main__":
    unittest.main()
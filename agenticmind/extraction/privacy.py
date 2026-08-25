"""隐私保护:SecretDetector + PII Redactor + 脱敏映射表

设计来源:
- docs/agenticmind/context-management/mvp-schema.md §3.4(secret 实体 + 隐私分级)
- docs/agenticmind/context-management/architecture.md §3.3(CloudForwarder + PII 脱敏)
- P1-0b 任务:安全底线,覆盖 6 类 secret

关键不变量:
- secret 检测即使置信度低也强制保留(漏报代价 >> 误报)
- 脱敏发生只在出域边界,域内全部用 placeholder + 映射表
- 6 类 secret:AWS / GCP / Azure / GitHub PAT / OpenAI / JWT / PEM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SecretKind(str, Enum):
    """6 类 secret 类型(PEM 私钥独立算一类)"""

    AWS_ACCESS_KEY = "aws_access_key"  # AKIA + 16 字符
    GCP_API_KEY = "gcp_api_key"  # AIza + 35 字符
    AZURE_KEY = "azure_key"  # 多种 Azure key prefix
    GITHUB_PAT = "github_pat"  # ghp_/ghs_/gho_ + 36 字符
    OPENAI_API_KEY = "openai_api_key"  # sk- + 48 字符
    JWT = "jwt"  # eyJ + 三段 base64
    PEM_PRIVATE_KEY = "pem_private_key"  # -----BEGIN ... PRIVATE KEY-----


@dataclass
class SecretAlert:
    """一条 secret 检测告警"""

    kind: SecretKind
    matched_text: str  # 实际匹配到的文本(用于脱敏)
    span: tuple[int, int]  # 字符偏移
    placeholder: str  # 脱敏后的占位符(默认 SECRET_{kind}_{hash8})


# ----------------------------------------------------------------------------
# SecretDetector:覆盖 6 类 secret
# ----------------------------------------------------------------------------


class SecretDetector:
    """Secret 检测器

    设计要点:
    - 6 类 secret 全部覆盖(详见 SecretKind)
    - 即使匹配置信度低也强制保留(漏报代价 >> 误报)
    - 优先级:规则匹配 > NER 召回
    - 默认 thresholds 保守(优先召回)
    """

    # 6 类 secret 的正则模式(高召回优先)
    PATTERNS: list[tuple[SecretKind, re.Pattern[str]]] = [
        # AWS Access Key:AKIA / ASIA 开头,16 字母数字
        (
            SecretKind.AWS_ACCESS_KEY,
            re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
        ),
        # GCP API Key:AIza 开头,35 字符
        (
            SecretKind.GCP_API_KEY,
            re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
        ),
        # Azure:多种前缀
        # DefaultAzureCredential / AccountKey= / SharedAccessKey=
        (
            SecretKind.AZURE_KEY,
            re.compile(
                r"\b(DefaultAzureCredential|AccountKey=[0-9a-zA-Z+/=]{88}|"
                r"SharedAccessSignature=[^&\s]+|sk-[0-9a-zA-Z]{32})"
            ),
        ),
        # GitHub PAT:ghp_/ghs_/gho_/ghu_/ghr_ + 36 字母数字
        (
            SecretKind.GITHUB_PAT,
            re.compile(r"\b(ghp|ghs|gho|ghu|ghr)_[0-9A-Za-z]{36}\b"),
        ),
        # OpenAI API Key:sk- + 48 字符(T3 起);sk-proj- + 字符(T4 起)
        (
            SecretKind.OPENAI_API_KEY,
            re.compile(r"\bsk-(proj-)?[0-9A-Za-z\-_]{20,}\b"),
        ),
        # JWT:eyJ 开头三段 base64
        (
            SecretKind.JWT,
            re.compile(r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
        ),
        # PEM Private Key:-----BEGIN ... PRIVATE KEY-----
        (
            SecretKind.PEM_PRIVATE_KEY,
            re.compile(
                r"-----BEGIN (RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----",
                re.MULTILINE,
            ),
        ),
    ]

    def __init__(self) -> None:
        # 缓存 placeholder 映射(同一 secret 多次出现共用 placeholder)
        self._placeholder_map: dict[str, str] = {}

    def scan(self, text: str) -> list[SecretAlert]:
        """扫描文本,返回所有 secret 告警(去重按 placeholder)"""
        alerts: list[SecretAlert] = []
        seen_placeholders: set[str] = set()

        for kind, pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                matched = match.group(0)
                placeholder = self._get_placeholder(kind, matched)

                if placeholder in seen_placeholders:
                    continue

                alerts.append(
                    SecretAlert(
                        kind=kind,
                        matched_text=matched,
                        span=(match.start(), match.end()),
                        placeholder=placeholder,
                    )
                )
                seen_placeholders.add(placeholder)

        return alerts

    def redact(self, text: str) -> tuple[str, list[SecretAlert]]:
        """脱敏文本,用 placeholder 替换 secret,返回 (redacted_text, alerts)"""
        alerts = self.scan(text)
        if not alerts:
            return text, []

        # 按 span 倒序替换,避免偏移变化
        sorted_alerts = sorted(alerts, key=lambda a: a.span[0], reverse=True)
        redacted = text
        for alert in sorted_alerts:
            redacted = (
                redacted[: alert.span[0]] + alert.placeholder + redacted[alert.span[1] :]
            )
        return redacted, alerts

    def _get_placeholder(self, kind: SecretKind, matched_text: str) -> str:
        """生成稳定的 placeholder(同一 secret 文本 -> 同一 placeholder)"""
        if matched_text not in self._placeholder_map:
            # 用 hash8 保证稳定性 + 短长度(不暴露原值)
            import hashlib

            hash8 = hashlib.sha256(matched_text.encode("utf-8")).hexdigest()[:8]
            self._placeholder_map[matched_text] = f"SECRET_{kind.value}_{hash8}"
        return self._placeholder_map[matched_text]

    def reset_cache(self) -> None:
        """清除 placeholder 缓存(用于脱敏映射表重建)"""
        self._placeholder_map.clear()


# ----------------------------------------------------------------------------
# PII Redactor + 映射表(为 CloudForwarder 服务,P1-0b 只占位)
# ----------------------------------------------------------------------------


@dataclass
class RedactionResult:
    """脱敏结果"""

    redacted_text: str
    mappings: dict[str, str]  # placeholder → 原值
    secret_alerts: list[SecretAlert] = field(default_factory=list)


class PIIRedactor:
    """PII 脱敏器(姓名/电话/邮箱/身份证/内部 URL)

    设计要点:
    - 白名单制:仅 secret 强制脱敏,PII 通过 privacy_tier 字段分级
    - 脱敏仅在出域边界发生,域内全部用 placeholder
    - 映射表用于反匿名化(域内还原)

    P1-0b 状态:仅占位,完整 PII 模式(中文姓名/电话/身份证等)在 P1-2 教师标注时启用
    """

    def __init__(self, secret_detector: SecretDetector | None = None) -> None:
        self._secret_detector = secret_detector or SecretDetector()
        self._mapping: dict[str, str] = {}  # placeholder → 原值

    def redact(self, text: str) -> RedactionResult:
        """脱敏文本(secret 强制 + PII 按需)

        P1-0b:仅处理 secret(覆盖 6 类)
        P1-2:扩展处理 PII(中文姓名/电话/邮箱等)
        """
        redacted, secret_alerts = self._secret_detector.redact(text)

        # 记录映射(用于反匿名化)
        for alert in secret_alerts:
            if alert.placeholder not in self._mapping:
                self._mapping[alert.placeholder] = alert.matched_text

        return RedactionResult(
            redacted_text=redacted,
            mappings=dict(self._mapping),
            secret_alerts=secret_alerts,
        )

    def de_anonymize(self, text: str) -> str:
        """反匿名化(把 placeholder 还原为原值)

        仅在域内调用,云端响应回来后用于还原。
        """
        for placeholder, original in self._mapping.items():
            text = text.replace(placeholder, original)
        return text

    def reset(self) -> None:
        """重置映射表"""
        self._mapping.clear()
        self._secret_detector.reset_cache()
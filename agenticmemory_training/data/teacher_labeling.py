"""P1-2 教师标注:对话 → 13 字段 schema

输入:P1-1 合成的 JSONL(conversations.jsonl)
输出:session_extract_v0.jsonl,每行一个 turn 的 13 字段标注

教师选择:
- 主力:DeepSeek V4 Flash(成本低,中文好)
- 备用:GPT-4o(精度高,成本高)

标注 schema(对齐 mvp-schema.md §3.1):
- session_id / turn_index / timestamp
- intent(primary + secondary + confidence)
- entities(items[Entity])
- language(primary + confidence)
- current_topic(来自对话历史,单条 turn 为空)
- session_facts(扁平 KV,只在信息明确时填)

设计要点:
- 一次调用产出整段对话的所有 turn(避免逐 turn 调用,节省成本)
- prompt 明确给出 schema + 示例 + 反例
- 输出严格 JSON,失败时记录 + 跳过
- 成本控制:每 100 条对话约 ¥10-30
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ============================================================================
# 教师标注 prompt
# ============================================================================


EXTRACTION_PROMPT_TEMPLATE = """你是会话信息提取系统,严格按照 schema 输出 JSON。

## 输入
一段代码项目对话(多轮),session_id = "{session_id}"。

```text
{conversation_text}
```

## Schema(每轮 turn 都要输出)

每轮 turn 输出一个对象:
```json
{{
  "turn_index": 0,          // 从 0 开始的整数
  "intent": {{
    "primary": "question" | "command" | "clarify" | "confirm" | "correct" | "chat" | "refuse" | "meta",
    "secondary": [],        // 可空,多意图时填多个
    "confidence": 0.0-1.0   // primary 的置信度
  }},
  "entities": [
    {{
      "type": "person" | "project" | "file_path" | "url" | "version" | "code_symbol" | "api" | "org" | "secret",
      "value": "原文中出现的字符串(去引号)",
      "span": [start_char_offset, end_char_offset],
      "confidence": 0.0-1.0
    }}
    // 没有实体时为 []
  ],
  "language": {{
    "primary": "zh" | "en" | "code" | "mixed" | "other",
    "confidence": 0.0-1.0
  }},
  "current_topic": {{
    "value": "当前 turn 涉及的主题(简短短语,如 'NPE 调试');如无明确话题填 ''",
    "confidence": 0.0-1.0
  }},
  "session_facts": [
    {{
      "key": "唯一标识,如 'user_name' / 'project_name'",
      "value": "JSON-serializable 值",
      "source_turn": <本 turn 的 turn_index>,
      "confidence": 0.0-1.0
    }}
    // 只在 turn 中**明确**确立事实时填;推论不填
  ]
}}
```

## 关键约束
1. **span 必须精确**:用 [start, end] 字符偏移(value 必须在原文中)
2. **code_symbol**:函数/类/方法名(如 validate_token),不要带括号
3. **file_path**:完整路径(含 src/..),不要简化
4. **secret** 类型:只在发现 API key/token/密码时使用(注意:不要把代码片段中的字符串当成 secret)
5. **session_facts**:只填**明确**确立的事实(如"用户说他的项目叫 X"),不填推论
6. **confidence**:宁低勿高(不确定就给 0.5-0.7)

## 输出格式
只输出 JSON,顶层结构:
```json
{{
  "turns": [
    {{turn_index: 0, intent: {{...}}, entities: [...], language: {{...}}, current_topic: {{...}}, session_facts: [...]}},
    {{turn_index: 1, ...}},
    ...
  ]
}}
```

不要任何解释或 markdown 包裹,直接输出 JSON。"""


# =========================================================================
# 数据模型
# =========================================================================


@dataclass
class TurnAnnotation:
    turn_index: int
    intent_primary: str
    intent_secondary: list[str] = field(default_factory=list)
    intent_confidence: float = 0.0
    entities: list[dict[str, Any]] = field(default_factory=list)
    language_primary: str = "other"
    language_confidence: float = 0.0
    current_topic: str = ""
    topic_confidence: float = 0.0
    session_facts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "turn_index": self.turn_index,
            "intent": {
                "primary": self.intent_primary,
                "secondary": self.intent_secondary,
                "confidence": self.intent_confidence,
            },
            "entities": self.entities,
            "language": {
                "primary": self.language_primary,
                "confidence": self.language_confidence,
            },
            "current_topic": {
                "value": self.current_topic,
                "confidence": self.topic_confidence,
            },
            "session_facts": self.session_facts,
        }


# =========================================================================
# 教师客户端(stub)
# =========================================================================


def label_via_teacher(
    client: Any,
    session: dict[str, Any],
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> list[TurnAnnotation]:
    """调用教师 API 标注一个会话的所有 turn(stub

    参数:
        client: openai.OpenAI 实例(DeepSeek / OpenAI 均同构)
        session: 单个会话 dict({session_id, source, turns[]})
        model: 模型名(deepseek-chat / gpt-4o 等)
        temperature: 标注任务建议 0(确定性优先)
        max_retries: 失败重试次数

    返回:
        List[TurnAnnotation]:每个 turn 一个标注(失败时该 turn 返回 None 或跳过)
    """
    session_id = session.get("session_id", "unknown")
    turns = session.get("turns", [])

    conversation_text = "\n".join(
        f"[{t['turn_index']}] {t['role']}: {t['text']}"
        for t in turns
    )

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        session_id=session_id,
        conversation_text=conversation_text,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            raw_turns = result.get("turns", [])

            annotations = []
            for raw in raw_turns:
                intent_data = raw.get("intent", {})
                lang_data = raw.get("language", {})
                topic_data = raw.get("current_topic", {})
                annotations.append(
                    TurnAnnotation(
                        turn_index=raw.get("turn_index", 0),
                        intent_primary=intent_data.get("primary", "chat"),
                        intent_secondary=intent_data.get("secondary", []),
                        intent_confidence=intent_data.get("confidence", 0.0),
                        entities=raw.get("entities", []),
                        language_primary=lang_data.get("primary", "other"),
                        language_confidence=lang_data.get("confidence", 0.0),
                        current_topic=topic_data.get("value", ""),
                        topic_confidence=topic_data.get("confidence", 0.0),
                        session_facts=raw.get("session_facts", []),
                    )
                )
            return annotations
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"[teacher_labeling] 失败 {session_id}: {type(e).__name__}: {e}")
            return []

    return []


def label_sessions(
    client: Any,
    conversations: Iterator[dict[str, Any]],
    model: str = "deepseek-chat",
    output_path: Path | None = None,
) -> int:
    """批量标注会话并可选写入 JSONL

    返回:成功标注的 turn 总数
    """
    count = 0
    output_f = open(output_path, "w", encoding="utf-8") if output_path else None

    try:
        for session in conversations:
            annotations = label_via_teacher(client, session, model=model)
            session_id = session.get("session_id", "unknown")
            for ann in annotations:
                record = ann.to_dict(session_id)
                if output_f:
                    output_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    finally:
        if output_f:
            output_f.close()

    return count


# =========================================================================
# 标注一致性(IRR)— Metis 🔴-3 整改:Kimi-K3 第二标注者 + Krippendorff α
# =========================================================================


def compute_krippendorff_alpha(
    rater_a: list[str], rater_b: list[str]
) -> float:
    """两个标注者对同一批样本的 Krippendorff α(nominal scale, 2 raters)

    NLTK coincidence 口径:每个样本对贡献两次(对称),总 coincidence = 2n。

    自测:
      perfect: k(['q','c','q','c'], ['q','c','q','c']) == 1.0
      random : k(['q','q','c','c'], ['q','c','q','c']) ≈ 0(2 类下)
    """
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        raise ValueError("rater_a 与 rater_b 长度必须相同且非空")
    n = len(rater_a)
    values = sorted(set(rater_a) | set(rater_b))
    if len(values) <= 1:
        return 1.0
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)

    # 对称 coincidence matrix:每个样本贡献 2 次
    coinc = [[0.0] * k for _ in range(k)]
    for va, vb in zip(rater_a, rater_b):
        i, j = idx[va], idx[vb]
        coinc[i][j] += 1.0
        coinc[j][i] += 1.0

    coinc_total = 2.0 * n  # = 2n

    # 观测不一致(总 coincidence - 对角线一致)
    do = coinc_total - sum(coinc[i][i] for i in range(k))

    # 期望不一致(NLTK 口径,基于边际 rowsum)
    rowsum = [sum(row) for row in coinc]
    de = coinc_total - sum(rowsum[i] * rowsum[i] for i in range(k)) / (coinc_total - 1.0)

    if de == 0.0:
        return 1.0
    alpha = 1.0 - do / de
    return round(alpha, 4)


def label_irr_subset(
    client_a: Any,
    client_b: Any,
    sessions: list[dict[str, Any]],
    primary_labeler: str = "deepseek-chat",
    second_labeler: str = "kimi-k3",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """用第二标注者(Kimi-K3)对子集二次标注,计算与主标注(DeepSeek)的 Krippendorff α

    流程:
    1. 用 client_a(DeepSeek 端点)标注 → labels_a
    2. 用 client_b(Kimi 端点)标注 → labels_b
    3. 对 intent.primary 计算 α

    client_a / client_b 需分别指向各自提供商的 OpenAI 兼容端点
    (如 base_url=https://api.deepseek.com/v1 vs https://api.moonshot.cn/v1)。
    """
    import json as _json
    import warnings as _warnings

    if not sessions:
        return {
            "n_pairs": 0,
            "krippendorff_alpha_intent": None,
            "labeler_a": primary_labeler,
            "labeler_b": second_labeler,
            "records": [],
            "skipped_unpaired_turns": 0,
            "empty_input": True,
        }

    labels_a: list[str] = []
    labels_b: list[str] = []
    records = []
    total_skipped = 0

    for session in sessions:
        ann_a = label_via_teacher(client_a, session, model=primary_labeler)
        ann_b = label_via_teacher(client_b, session, model=second_labeler)
        if len(ann_a) != len(ann_b):
            _warnings.warn(
                f"session {session.get('session_id', 'unknown')} 两次标注长度不一致 "
                f"({len(ann_a)} vs {len(ann_b)}),按较短者截断"
            )
            total_skipped += abs(len(ann_a) - len(ann_b))
        for ta, tb in zip(ann_a, ann_b):
            labels_a.append(ta.intent_primary)
            labels_b.append(tb.intent_primary)
            records.append(
                {
                    "session_id": session.get("session_id", "unknown"),
                    "turn_index": ta.turn_index,
                    "labeler_a": primary_labeler,
                    "labeler_b": second_labeler,
                    "intent_a": ta.intent_primary,
                    "intent_b": tb.intent_primary,
                }
            )

    alpha = compute_krippendorff_alpha(labels_a, labels_b)
    result = {
        "n_pairs": len(labels_a),
        "krippendorff_alpha_intent": alpha,
        "labeler_a": primary_labeler,
        "labeler_b": second_labeler,
        "records": records,
        "skipped_unpaired_turns": total_skipped,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    """CLI 入口(stub)"""
    import argparse

    parser = argparse.ArgumentParser(description="P1-2 教师标注 CLI(stub)")
    parser.add_argument(
        "--input", type=Path, default=Path("data/agenticmemory_training/v0/conversations.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/agenticmemory_training/v0/session_extract_v0.jsonl"),
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"P1-2 stub:input={args.input}, output={args.output}, model={args.model}")
    print("注意:此脚本为骨架,实际执行需要:")
    print(f"  1. 准备 conversations.jsonl(P1-1 产出)放在 {args.input}")
    print("  2. 设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量")
    print(f"  3. 取消 main() 末尾 raise,启用实际标注逻辑")


if __name__ == "__main__":
    main()
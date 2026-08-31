"""P1-3 评估:标注一致性 / 字段填充率 / 教师偏置

输入:session_extract_v0.jsonl(单次标注或多轮标注)
输出:findings_v0.md(可读报告)+ JSON 指标

评估维度:
1. **标注一致性**:同一教师 N 次标注同一对话的 agreement(可重复性)
2. **字段填充率**:每字段非空率(过低的字段应考虑简化 schema)
3. **教师偏置**:某字段值分布异常倾斜(如 intent 总是 "chat")
4. **实体覆盖率**:9 种 EntityType 实际触发情况
5. **fact 密度**:session_facts 数量 / turn 数(判断 schema 过载)

设计要点:
- 不依赖外部 ML 库,纯 Python 统计
- 输出格式:stdout 摘要 + findings_v0.md(给非工程师阅读)
- 单次标注也跑(只跑维度 2/3/4/5)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ============================================================================
# 统计结果数据模型
# ============================================================================


@dataclass
class FieldFillRate:
    field_name: str
    total_turns: int
    filled_count: int

    @property
    def rate(self) -> float:
        return self.filled_count / self.total_turns if self.total_turns > 0 else 0.0


@dataclass
class DistributionStats:
    field_name: str
    counter: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counter.values())

    @property
    def unique_values(self) -> int:
        return len(self.counter)

    @property
    def entropy(self) -> float:
        total = self.total
        if total == 0:
            return 0.0
        import math

        return -sum((c / total) * math.log2(c / total) for c in self.counter.values() if c > 0)

    def top_k(self, k: int = 5) -> list[tuple[str, int]]:
        return self.counter.most_common(k)


@dataclass
class AnnotationConsistency:
    field_name: str
    total_comparisons: int
    agreement_count: int

    @property
    def rate(self) -> float:
        return self.agreement_count / self.total_comparisons if self.total_comparisons > 0 else 0.0


@dataclass
class EvaluationReport:
    total_sessions: int
    total_turns: int
    field_fill_rates: list[FieldFillRate] = field(default_factory=list)
    intent_distribution: DistributionStats = field(default_factory=lambda: DistributionStats("intent.primary"))
    language_distribution: DistributionStats = field(default_factory=lambda: DistributionStats("language.primary"))
    entity_type_distribution: DistributionStats = field(default_factory=lambda: DistributionStats("entities.type"))
    fact_density_per_turn: float = 0.0
    annotation_consistency: list[AnnotationConsistency] = field(default_factory=list)


# ============================================================================
# 加载标注
# ============================================================================


def load_annotations(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ============================================================================
# 评估指标计算
# ============================================================================


def compute_field_fill_rates(annotations: list[dict[str, Any]]) -> list[FieldFillRate]:
    """计算每个字段的非空率

    检查字段:
    - intent.primary / intent.confidence
    - entities(列表非空判断)
    - language.primary
    - current_topic.value
    - session_facts(列表非空判断)
    """
    if not annotations:
        return []

    field_checks = [
        ("intent.primary", lambda a: a.get("intent", {}).get("primary", "") not in ("", None)),
        ("intent.confidence", lambda a: a.get("intent", {}).get("confidence", 0) > 0),
        ("entities", lambda a: len(a.get("entities", [])) > 0),
        ("language.primary", lambda a: a.get("language", {}).get("primary", "") not in ("", None, "other")),
        ("current_topic.value", lambda a: a.get("current_topic", {}).get("value", "") not in ("", None)),
        ("session_facts", lambda a: len(a.get("session_facts", [])) > 0),
    ]

    results = []
    total = len(annotations)
    for field_name, predicate in field_checks:
        filled = sum(1 for a in annotations if predicate(a))
        results.append(FieldFillRate(field_name=field_name, total_turns=total, filled_count=filled))
    return results


def compute_distributions(annotations: list[dict[str, Any]]) -> tuple[
    DistributionStats, DistributionStats, DistributionStats
]:
    """计算 intent、language、entity_type 三个字段的分布"""
    intent_counter: Counter = Counter()
    lang_counter: Counter = Counter()
    entity_type_counter: Counter = Counter()

    for a in annotations:
        intent_counter[a.get("intent", {}).get("primary", "")] += 1
        lang_counter[a.get("language", {}).get("primary", "")] += 1
        for ent in a.get("entities", []):
            entity_type_counter[ent.get("type", "")] += 1

    return (
        DistributionStats("intent.primary", intent_counter),
        DistributionStats("language.primary", lang_counter),
        DistributionStats("entities.type", entity_type_counter),
    )


def compute_fact_density(annotations: list[dict[str, Any]]) -> float:
    """session_facts 总数 / turn 总数"""
    if not annotations:
        return 0.0
    total_facts = sum(len(a.get("session_facts", [])) for a in annotations)
    return total_facts / len(annotations)


def compute_consistency(runs: list[list[dict[str, Any]]]) -> list[AnnotationConsistency]:
    """计算多轮标注的一致性

    参数:
        runs: K 次标注的结果,每个元素是一个标注列表

    返回:每个字段的 K 折一致性

    一致性定义:对每个 (session_id, turn_index),检查 K 次标注的字段值是否一致
    """
    if len(runs) < 2:
        return []

    keys_to_check = ["intent.primary", "language.primary"]

    by_session_turn: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        for ann in run:
            sid = ann.get("session_id", "")
            tidx = ann.get("turn_index", -1)
            by_session_turn.setdefault((sid, tidx), []).append(ann)

    results = []
    for key in keys_to_check:
        total = 0
        agree = 0
        for anns in by_session_turn.values():
            if len(anns) < 2:
                continue
            values = [_get_nested(ann, key) for ann in anns]
            total += 1
            if len(set(values)) == 1:
                agree += 1
        results.append(
            AnnotationConsistency(field_name=key, total_comparisons=total, agreement_count=agree)
        )
    return results


def _get_nested(d: dict[str, Any], key: str) -> Any:
    parts = key.split(".")
    current = d
    for p in parts:
        if isinstance(current, dict):
            current = current.get(p)
        else:
            return None
    return current


# ============================================================================
# 主评估函数
# ============================================================================


def evaluate(annotations: list[dict[str, Any]]) -> EvaluationReport:
    """单次标注的评估(无一致性)"""
    if not annotations:
        return EvaluationReport(total_sessions=0, total_turns=0)

    sessions = {a.get("session_id") for a in annotations}
    return EvaluationReport(
        total_sessions=len(sessions),
        total_turns=len(annotations),
        field_fill_rates=compute_field_fill_rates(annotations),
        intent_distribution=compute_distributions(annotations)[0],
        language_distribution=compute_distributions(annotations)[1],
        entity_type_distribution=compute_distributions(annotations)[2],
        fact_density_per_turn=compute_fact_density(annotations),
    )


def evaluate_multi_run(runs: list[list[dict[str, Any]]]) -> EvaluationReport:
    """多轮标注的评估(包含一致性)

    参数:第一轮作为主 run 计算 fill rate / 分布;跨 run 计算一致性
    """
    if not runs:
        return EvaluationReport(total_sessions=0, total_turns=0)

    primary = runs[0]
    report = evaluate(primary)
    report.annotation_consistency = compute_consistency(runs)
    return report


# ============================================================================
# 报告输出
# ============================================================================


def report_to_markdown(report: EvaluationReport) -> str:
    """EvaluationReport → findings_v0.md"""
    lines = [
        "# P1-3 评估报告(findings_v0)",
        "",
        "## 总览",
        f"- 总会话数:{report.total_sessions}",
        f"- 总 turn 数:{report.total_turns}",
        f"- fact 密度(每 turn session_facts 数):{report.fact_density_per_turn:.3f}",
        "",
        "## 字段填充率",
        "",
        "| 字段 | 填充率 | 填充/总 turn |",
        "|---|---|---|",
    ]
    for rate in report.field_fill_rates:
        pct = f"{rate.rate * 100:.1f}%"
        lines.append(f"| `{rate.field_name}` | {pct} | {rate.filled_count}/{rate.total_turns} |")

    lines.extend(
        [
            "",
            "## 字段分布",
            "",
            f"### intent.primary(entropy={report.intent_distribution.entropy:.2f})",
        ]
    )
    for value, count in report.intent_distribution.top_k():
        pct = count / report.intent_distribution.total * 100
        lines.append(f"- `{value}`:{count}({pct:.1f}%)")

    lines.extend(
        [
            "",
            f"### language.primary(entropy={report.language_distribution.entropy:.2f})",
        ]
    )
    for value, count in report.language_distribution.top_k():
        pct = count / report.language_distribution.total * 100
        lines.append(f"- `{value}`:{count}({pct:.1f}%)")

    lines.extend(
        [
            "",
            f"### entities.type(entropy={report.entity_type_distribution.entropy:.2f})",
        ]
    )
    for value, count in report.entity_type_distribution.top_k():
        lines.append(f"- `{value}`:{count}")

    if report.annotation_consistency:
        lines.extend(["", "## 标注一致性(多轮标注)", "", "| 字段 | 一致率 | 一致/总 |", "|---|---|---|"])
        for c in report.annotation_consistency:
            pct = f"{c.rate * 100:.1f}%"
            lines.append(f"| `{c.field_name}` | {pct} | {c.agreement_count}/{c.total_comparisons} |")

    lines.extend(
        [
            "",
            "## 解读建议",
            "",
            "- 字段填充率 < 30% → 该字段可能是过度设计,考虑移到 backlog",
            "- intent.primary 中某一类 > 70% → 教师偏置,需调整 prompt 或降低 temperature",
            "- entities.type 未触发的类型 → 标注规则不清晰,需调整 prompt 示例",
            "- 标注一致性 < 70% → prompt 中示例不足或 schema 字段定义模糊",
        ]
    )
    return "\n".join(lines) + "\n"


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI 入口"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="P1-3 评估 CLI")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/agenticmemory_training/v0/session_extract_v0.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/agenticmemory_training/v0/findings_v0.md"),
    )
    parser.add_argument(
        "--multi-run",
        action="store_true",
        help="启用多轮一致性评估(需多次标注;当前单文件回退单次评估)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误:标注文件不存在 {args.input}", file=sys.stderr)
        raise SystemExit(2)

    annotations = list(load_annotations(args.input))
    print(f"读取标注: {len(annotations)} 条")

    if args.multi_run:
        print("注意:--multi-run 需要多次标注运行数据,当前单文件输入回退为单次评估", file=sys.stderr)
    result = evaluate(annotations)
    md = report_to_markdown(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"评估报告已写入: {args.output}")


if __name__ == "__main__":
    main()
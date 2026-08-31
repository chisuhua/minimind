"""把 P1-2 标注转成 LoRA 训练格式。

输入:session_extract_v0.jsonl(每行一个 turn 标注)
输出:JSONL,每行一个训练样本:
    {"input": "...对话上文 + 当前 turn...", "output": "...JSON 13 字段标注..."}

设计要点:
- 单 turn 作为独立训练样本(上下文是它之前的所有 turns)
- prompt 模板与 P1-2 一致(标注 / 训练同 prompt)
- input 截断到 max_context_tokens(默认 1024)
- output 用 JSON 序列化,字段顺序固定
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

INFERENCE_PROMPT_TEMPLATE = """你是会话信息提取系统,严格按照 schema 输出 JSON。

## 输入
一段代码项目对话(多轮),session_id = "{session_id}"。

```text
{conversation_text}
```

## Schema(每轮 turn 都要输出)
输出结构:`{{"turns": [{{turn_index, intent, entities, language, current_topic, session_facts}}, ...]}}`

intent.primary ∈ {{question, command, clarify, confirm, correct, chat, refuse, meta}}
entities[].type ∈ {{person, project, file_path, url, version, code_symbol, api, org, secret}}
language.primary ∈ {{zh, en, code, mixed, other}}

直接输出 JSON,不要解释。"""


def build_training_samples(
    annotations: list[dict[str, Any]],
    max_context_turns: int = 8,
) -> Iterator[dict[str, Any]]:
    """从标注列表构造训练样本(每个 turn 一个)

    参数:
        annotations: 一个会话的所有 turn 标注(需按 turn_index 升序排列)
        max_context_turns: 当前 turn 上文保留多少轮

    产出:每个 turn 一个 dict{"input", "output"}
    """
    if not annotations:
        return
    session_id = annotations[0].get("session_id", "unknown")

    annotations = sorted(annotations, key=lambda a: a.get("turn_index", 0))
    for i, ann in enumerate(annotations):
        start = max(0, i - max_context_turns)
        context_turns = annotations[start : i + 1]

        conversation_text = "\n".join(
            f"[{t.get('turn_index', 0)}] {t.get('role', 'user')}: {t.get('text', '')}"
            for t in context_turns
        )
        input_text = INFERENCE_PROMPT_TEMPLATE.format(
            session_id=session_id,
            conversation_text=conversation_text,
        )
        output_text = json.dumps(
            {
                "turns": [
                    {
                        "turn_index": ann.get("turn_index"),
                        "intent": ann.get("intent"),
                        "entities": ann.get("entities"),
                        "language": ann.get("language"),
                        "current_topic": ann.get("current_topic"),
                        "session_facts": ann.get("session_facts"),
                    }
                ]
            },
            ensure_ascii=False,
        )
        yield {"input": input_text, "output": output_text, "session_id": session_id}


def group_by_session(
    annotations: Iterator[dict[str, Any]],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """按 session_id 分组"""
    sessions: dict[str, list[dict[str, Any]]] = {}
    for ann in annotations:
        sid = ann.get("session_id", "")
        sessions.setdefault(sid, []).append(ann)
    for sid, anns in sessions.items():
        yield sid, anns


def write_training_samples(
    grouped: Iterator[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
    max_context_turns: int = 8,
) -> int:
    """将分组后的会话流写入训练 JSONL"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _sid, anns in grouped:
            for sample in build_training_samples(anns, max_context_turns=max_context_turns):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
    return count


def split_train_dev(
    samples: list[dict[str, Any]],
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按会话切分 train/dev,防止数据泄漏

    按 session_id 分组后随机切分,确保同一会话的样本不跨 train/dev。
    """
    import random

    by_session: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        by_session.setdefault(s.get("session_id", ""), []).append(s)

    sessions = list(by_session.keys())
    rng = random.Random(seed)
    rng.shuffle(sessions)
    cut = int(len(sessions) * (1 - dev_ratio))
    train_sessions = set(sessions[:cut])

    train = [s for sid, items in by_session.items() if sid in train_sessions for s in items]
    dev = [s for sid, items in by_session.items() if sid not in train_sessions for s in items]
    return train, dev


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="P1-4 数据准备 CLI")
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/agenticmemory_training/v0/session_extract_v0.jsonl")
    )
    parser.add_argument("--output-train", type=Path, default=Path("data/agenticmemory_training/v0/train.jsonl"))
    parser.add_argument("--output-dev", type=Path, default=Path("data/agenticmemory_training/v0/dev.jsonl"))
    parser.add_argument("--max-context-turns", type=int, default=8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if not args.annotations.exists():
        print(f"错误:标注文件不存在 {args.annotations}", file=sys.stderr)
        raise SystemExit(2)

    annotations = []
    with args.annotations.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    print(f"读取标注: {len(annotations)} 条 turn 标注")

    # 按 session 构建样本,避免 build_training_samples 混淆不同 session
    samples: list[dict] = []
    for _sid, sess_anns in group_by_session(iter(annotations)):
        samples.extend(build_training_samples(sess_anns, max_context_turns=args.max_context_turns))

    train, dev = split_train_dev(samples, dev_ratio=args.dev_ratio)
    print(f"样本: {len(samples)} → train {len(train)} / dev {len(dev)}")

    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    args.output_dev.parent.mkdir(parents=True, exist_ok=True)
    with args.output_train.open("w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with args.output_dev.open("w", encoding="utf-8") as f:
        for s in dev:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"已写 train: {args.output_train} / dev: {args.output_dev}")


if __name__ == "__main__":
    main()
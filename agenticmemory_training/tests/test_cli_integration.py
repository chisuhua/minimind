"""验证 data_prep / evaluation 的 CLI main() 接线(真实执行而非 stub)"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _make_annotations(n_sessions: int = 3, turns_per_session: int = 2) -> list[dict]:
    anns = []
    for s in range(n_sessions):
        for t in range(turns_per_session):
            anns.append({
                "session_id": f"sess_{s}",
                "turn_index": t,
                "role": "user" if t % 2 == 0 else "assistant",
                "text": f"session {s} turn {t} content",
                "intent": {"primary": "question" if t % 2 == 0 else "command"},
                "entities": [],
                "language": {"primary": "zh"},
                "current_topic": {"value": f"topic_{s}"},
                "session_facts": [],
            })
    return anns


class TestDataPrepMain(unittest.TestCase):
    """data_prep.main() 应真实切分 train/dev"""

    def test_data_prep_writes_train_dev(self):
        sys.path.insert(0, str(REPO))
        from agenticmemory_training.training.data_prep import (
            group_by_session,
            build_training_samples,
            split_train_dev,
        )
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ann_path = td / "session_extract.jsonl"
            anns = _make_annotations()
            ann_path.write_text(
                "\n".join(json.dumps(a, ensure_ascii=False) for a in anns) + "\n",
                encoding="utf-8",
            )
            train_p = td / "train.jsonl"
            dev_p = td / "dev.jsonl"

            annotations = []
            with ann_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        annotations.append(json.loads(line))

            samples = []
            for _sid, sess_anns in group_by_session(iter(annotations)):
                samples.extend(build_training_samples(sess_anns, max_context_turns=8))

            train, dev = split_train_dev(samples, dev_ratio=0.4)

            train_p.parent.mkdir(parents=True, exist_ok=True)
            dev_p.parent.mkdir(parents=True, exist_ok=True)
            with train_p.open("w", encoding="utf-8") as f:
                for s in train:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            with dev_p.open("w", encoding="utf-8") as f:
                for s in dev:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            self.assertTrue(train_p.exists() and dev_p.exists())
            train_sids = {json.loads(line)["session_id"] for line in train_p.read_text(encoding="utf-8").splitlines() if line.strip()}
            dev_sids = {json.loads(line)["session_id"] for line in dev_p.read_text(encoding="utf-8").splitlines() if line.strip()}
            self.assertTrue(train_sids.isdisjoint(dev_sids), "session 泄漏:同一会话跨 train/dev")


class TestEvaluationMain(unittest.TestCase):
    """evaluation.main() 应真实写出 findings md"""

    def test_evaluation_writes_findings(self):
        sys.path.insert(0, str(REPO))
        from agenticmemory_training.data.evaluation import (
            load_annotations,
            evaluate,
            report_to_markdown,
        )
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ann_path = td / "session_extract.jsonl"
            anns = _make_annotations()
            ann_path.write_text(
                "\n".join(json.dumps(a, ensure_ascii=False) for a in anns) + "\n",
                encoding="utf-8",
            )
            out_md = td / "findings.md"

            annotations = list(load_annotations(ann_path))
            result = evaluate(annotations)
            md = report_to_markdown(result)

            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(md, encoding="utf-8")

            self.assertTrue(out_md.exists())
            content = out_md.read_text(encoding="utf-8")
            self.assertIn("intent", content)
            self.assertTrue("填充" in content or "fill" in content.lower())


if __name__ == "__main__":
    unittest.main()

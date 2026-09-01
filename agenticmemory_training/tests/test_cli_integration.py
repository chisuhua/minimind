"""验证 data_prep / evaluation 的 CLI main() 接线(真实执行而非 stub)"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _write_annotations(path: Path, n_sessions: int = 3, turns_per_session: int = 2) -> None:
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
    path.write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in anns) + "\n",
        encoding="utf-8",
    )


class TestDataPrepMain(unittest.TestCase):
    """python -m ...data_prep 应真实执行 main() 并切分 train/dev"""

    def test_data_prep_writes_train_dev(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ann_path = td / "session_extract.jsonl"
            _write_annotations(ann_path)
            train_p = td / "train.jsonl"
            dev_p = td / "dev.jsonl"

            proc = subprocess.run(
                [sys.executable, "-m", "agenticmemory_training.training.data_prep",
                 "--annotations", str(ann_path),
                 "--output-train", str(train_p),
                 "--output-dev", str(dev_p),
                 "--dev-ratio", "0.4"],
                cwd=REPO, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}")
            self.assertTrue(train_p.exists() and dev_p.exists(), proc.stdout)

            train_sids = {json.loads(line)["session_id"] for line in train_p.read_text(encoding="utf-8").splitlines() if line.strip()}
            dev_sids = {json.loads(line)["session_id"] for line in dev_p.read_text(encoding="utf-8").splitlines() if line.strip()}
            self.assertTrue(train_sids, "train 不应为空")
            self.assertTrue(train_sids.isdisjoint(dev_sids), "session 泄漏:同一会话跨 train/dev")

    def test_data_prep_missing_file_exits_2(self):
        proc = subprocess.run(
            [sys.executable, "-m", "agenticmemory_training.training.data_prep",
             "--annotations", "/nonexistent/ann.jsonl"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 2)


class TestEvaluationMain(unittest.TestCase):
    """python -m ...data.evaluation 应真实执行 main() 并写出 findings md"""

    def test_evaluation_writes_findings(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ann_path = td / "session_extract.jsonl"
            _write_annotations(ann_path)
            out_md = td / "findings.md"

            proc = subprocess.run(
                [sys.executable, "-m", "agenticmemory_training.data.evaluation",
                 "--input", str(ann_path), "--output", str(out_md)],
                cwd=REPO, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}")
            self.assertTrue(out_md.exists(), proc.stdout)
            content = out_md.read_text(encoding="utf-8")
            self.assertIn("intent", content)
            self.assertTrue("填充" in content or "fill" in content.lower())


if __name__ == "__main__":
    unittest.main()

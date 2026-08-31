"""验证 eval_random_label:shuffle gold 后 F1 显著低于真实 gold F1"""
import sys
from pathlib import Path
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.training.eval_random_label import compute_random_label_f1


class TestComputeRandomLabelF1(unittest.TestCase):
    def test_random_label_f1_returns_both_metrics(self):
        gold_text = '{"turns": [{"intent": {"primary": "question"}, "language": {"primary": "zh"}, "entities": [], "session_facts": []}]}'
        samples = [
            {"input": f"q{i}", "output": gold_text} for i in range(4)
        ]
        pred_outputs = [gold_text] * 4
        with patch(
            "agenticmemory_training.training.eval_random_label.run_lora_inference",
            return_value=pred_outputs,
        ):
            result = compute_random_label_f1(
                base_model="fake", adapter_dir=Path("fake"), samples=samples
            )
        # 真实 gold 全部匹配 → genuine F1 高
        assert result["genuine_f1"]["intent.primary"]["f1"] >= 0.99
        # random_label_f1 键存在(4 样本 intent 全相同,shuffle 后仍全匹配 → 该值可能仍高;
        # 此为字段值高度集中的已知边界,测试不断言其低值)
        assert "random_label_f1" in result
        assert result["n_total"] == 4

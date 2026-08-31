"""验证 eval_zero_shot 计算按字段 zero-shot F1(无 adapter)"""
import sys
from pathlib import Path
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.training.eval_zero_shot import compute_zero_shot_f1


class TestComputeZeroShotF1(unittest.TestCase):
    def test_compute_zero_shot_f1_includes_fields(self):
        # run_inference 被 mock 返回固定输出
        fake_outputs = [
            '{"turns": [{"intent": {"primary": "question"}, '
            '"language": {"primary": "zh"}, "entities": [], "session_facts": []}]}'
        ] * 4
        samples = [
            {"input": f"q{i}", "output": '{"turns": [{"intent": {"primary": "question"}, "language": {"primary": "zh"}, "entities": [], "session_facts": []}]}'}
            for i in range(4)
        ]
        with patch(
            "agenticmemory_training.training.eval_zero_shot.run_inference",
            return_value=fake_outputs,
        ):
            result = compute_zero_shot_f1(base_model="fake", samples=samples)
        assert "intent.primary" in result
        assert "language.primary" in result
        assert "entities" in result
        assert "session_facts" in result
        assert result["n_total"] == 4

    def test_compute_zero_shot_f1_parses_failed_gold(self):
        # 当 gold 解析失败时,n_parse_failed_gold 应该增加
        fake_outputs = [
            '{"turns": [{"intent": {"primary": "question"}, '
            '"language": {"primary": "zh"}, "entities": [], "session_facts": []}]}'
        ] * 2
        # 第二个样本的 gold 是无效 JSON
        samples = [
            {"input": "q0", "output": '{"turns": [{"intent": {"primary": "question"}, "language": {"primary": "zh"}, "entities": [], "session_facts": []}]}'},
            {"input": "q1", "output": "not valid json"},
        ]
        with patch(
            "agenticmemory_training.training.eval_zero_shot.run_inference",
            return_value=fake_outputs,
        ):
            result = compute_zero_shot_f1(base_model="fake", samples=samples)
        assert result["n_total"] == 2
        assert result["n_parse_failed_gold"] == 1

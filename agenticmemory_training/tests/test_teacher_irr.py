"""验证 Krippendorff α 计算(2 标注者、nominal scale)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.data.teacher_labeling import compute_krippendorff_alpha


def test_alpha_perfect_agreement():
    # 两位标注者完全一致 → α = 1.0
    a = ["question", "command", "chat", "question"]
    b = ["question", "command", "chat", "question"]
    assert compute_krippendorff_alpha(a, b) == 1.0


def test_alpha_random_agreement():
    # 各 50% 随机一致 → α 接近 0(2 类 nominal 下约 -0.167)
    a = ["question", "question", "chat", "chat"]
    b = ["question", "chat", "question", "chat"]
    alpha = compute_krippendorff_alpha(a, b)
    assert -0.4 <= alpha <= 0.4


def test_alpha_invalid_length():
    # 长度不同 → ValueError
    try:
        compute_krippendorff_alpha(["a"], ["a", "b"])
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

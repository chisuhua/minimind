"""验证 ultrachat 公开集适配(腿A 可复现加载路径)"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.data.synthesis import PUBLIC_DATASET_REGISTRY, load_public_dataset


class TestUltrachatAdapter(unittest.TestCase):
    """ultrachat messages 格式 → Conversation 转换 + registry 注册"""

    def setUp(self):
        self.tmp = Path("/tmp/opencode/p1_ultrachat_test.jsonl")
        lines = [
            '{"prompt_id": "abc123", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}',
            '{"prompt_id": "def456", "messages": [{"role": "user", "content": "q2"}]}',
        ]
        self.tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_registry_contains_ultrachat(self):
        self.assertIn("ultrachat", PUBLIC_DATASET_REGISTRY)
        self.assertEqual(PUBLIC_DATASET_REGISTRY["ultrachat"]["priority"], 0)

    def test_load_ultrachat_messages_format(self):
        convs = list(load_public_dataset("ultrachat", self.tmp))
        self.assertEqual(len(convs), 2)
        c0 = convs[0]
        self.assertEqual(c0.session_id, "abc123")
        self.assertEqual(c0.source, "public:ultrachat")
        self.assertEqual(len(c0.turns), 2)
        self.assertEqual(c0.turns[0].role, "user")
        self.assertEqual(c0.turns[0].text, "hi")
        self.assertEqual(c0.turns[1].role, "assistant")
        self.assertEqual(c0.turns[1].text, "hello")

    def test_load_ultrachat_single_turn(self):
        convs = list(load_public_dataset("ultrachat", self.tmp))
        self.assertEqual(convs[1].session_id, "def456")
        self.assertEqual(len(convs[1].turns), 1)

    def tearDown(self):
        self.tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

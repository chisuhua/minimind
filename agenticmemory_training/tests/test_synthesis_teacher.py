"""验证 synthesize_via_gpt4 支持 kimi-k3 教师 + metadata.teacher 字段"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenticmemory_training.data.synthesis import synthesize_via_gpt4


class _FakeChoice:
    message = MagicMock(content='{"turns": [{"role": "user", "text": "如何修 NPE?"}]}')


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions.create.return_value = _FakeResponse()


class TestSynthesizeTeacher(unittest.TestCase):
    def test_synthesize_via_kimi_model_arg(self):
        client = _FakeClient()
        convs = list(synthesize_via_gpt4(client, model="kimi-k3", n_conversations=1))
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].source, "synthetic:kimi-k3")
        # 确认 client 以 kimi-k3 被调用
        call_kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "kimi-k3")

    def test_conversation_to_jsonl_has_teacher_metadata(self):
        client = _FakeClient()
        convs = list(synthesize_via_gpt4(client, model="kimi-k3", n_conversations=1))
        record = convs[0].to_jsonl_record()
        self.assertEqual(record["metadata"]["teacher"], "kimi-k3")

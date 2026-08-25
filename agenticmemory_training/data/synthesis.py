"""P1-1 数据合成:公开集 + GPT-4 合成

输出格式(JSONL,每行一个会话):
{
    "session_id": "syn_001",
    "source": "public:sharely" | "synthetic:gpt4",
    "turns": [
        {"role": "user", "text": "...", "timestamp": "ISO8601"},
        {"role": "assistant", "text": "...", "timestamp": "ISO8601"}
    ]
}

设计要点:
- 三腿数据源(leg A 公开 / leg B GPT-4 合成 / leg C 内部试用)
- pre-launch 状态下 leg C 标注为可选增强
- 公开集优先 SHARELY(代码项目对话)、其次 MultiWOZ(任务型对话)
- 合成 prompt 明确"代码项目上下文"领域,避免通用闲聊
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ============================================================================
# 数据源声明(manifest)
# ============================================================================


PUBLIC_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "sharely": {
        "name": "SHARELY",
        "url": "https://github.com/shibing624/sharely (示例 URL,需实测)",
        "format": "jsonl",
        "domain": "code_project_chat",
        "expected_count": "5K-50K turns",
        "license": "需核实",
        "priority": 1,
        "notes": "代码项目对话语料,与 AgenticMind 目标分布最匹配。需验证数据集可用性与 schema。",
    },
    "multiwoz": {
        "name": "MultiWOZ 2.4",
        "url": "https://github.com/budzianowski/multiwoz",
        "format": "json",
        "domain": "task_oriented_dialog",
        "expected_count": "10K sessions",
        "license": "Apache 2.0",
        "priority": 2,
        "notes": "任务型对话,与代码项目领域偏差较大,作为 fallback",
    },
    "lmsys_chat": {
        "name": "LMSYS-Chat-1M",
        "url": "https://huggingface.co/datasets/lmsys/lmsys-chat-1m",
        "format": "jsonl",
        "domain": "general_chat",
        "expected_count": "1M turns",
        "license": "需核实",
        "priority": 3,
        "notes": "通用对话,需关键词过滤(代码项目相关)",
    },
}


# ============================================================================
# GPT-4 合成 prompt 模板
# ============================================================================


SYNTHESIS_PROMPT_TEMPLATE = """你是一个代码项目助手,正在与开发者对话。

任务:生成一段 5-8 轮的真实代码项目对话。

要求:
1. 场景:开发者正在调试/重构某个功能模块(明确说出文件名/函数名/类名)
2. 语气:工程师风格,可能包含缩写、专业术语、错误信息
3. 轮次结构:user 提问 → assistant 提供思路 → user 反馈 → ... → assistant 给出可执行方案
4. **必须包含至少 1 轮包含代码片段**(用户贴或助手贴均可)
5. **必须包含至少 1 处具体文件路径或函数名**(如 src/auth/login.py::validate_token)
6. 用户提问应包含 1-2 个意图意图:(question/command/correct 三选一或混合)

输出格式(JSON):
{{
  "turns": [
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}},
    ...
  ]
}}

场景建议(随机选一个):
- 调试一个 NPE:用户在某个路径下触发 NullPointerException
- 重构 API:用户想从 REST 改到 gRPC,询问迁移策略
- 添加功能:用户想在现有 login 模块 加 加 2FA
- 性能优化:用户反馈查询慢,助手建议加索引

请开始生成(只输出 JSON,不要其他解释):"""


# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class ConversationTurn:
    role: str  # user|assistant
    text: str
    timestamp: str = ""


@dataclass
class Conversation:
    session_id: str
    source: str  # public:sharely | synthetic:gpt4 | internal:trial
    turns: list[ConversationTurn] = field(default_factory=list)

    def to_jsonl_record(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "turns": [
                {"role": t.role, "text": t.text, "timestamp": t.timestamp}
                for t in self.turns
            ],
        }


# ============================================================================
# 公开集加载(stub)
# ============================================================================


def load_public_dataset(name: str, path: Path, limit: int | None = None) -> Iterator[Conversation]:
    """加载公开集 JSONL,转换为 Conversation 流

    参数:
        name: 数据集名(必须在 PUBLIC_DATASET_REGISTRY 中)
        path: 本地下载后的 JSONL 文件路径
        limit: 限制条数(None 表示全部)

    产出:
:yield: Conversation 对象
    """
    if name not in PUBLIC_DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集 {name},可选: {list(PUBLIC_DATASET_REGISTRY.keys())}"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"数据集文件不存在: {path}\n"
            f"请先手动下载 {PUBLIC_DATASET_REGISTRY[name]['url']}"
        )

    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit and count >= limit:
                break
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            yield _public_record_to_conversation(record, name)
            count += 1


def _public_record_to_conversation(record: dict[str, Any], dataset_name: str) -> Conversation:
    """将公开集原始记录转换为 Conversation

    注意:不同数据集 schema 不同,实际实现时需要按数据集适配。
    本函数仅提供框架,具体字段映射待 P1-1 实际加载时补充。
    """
    session_id = record.get("session_id") or record.get("id") or f"{dataset_name}_{id(record)}"
    turns_data = record.get("turns") or record.get("conversation") or []
    turns = [
        ConversationTurn(
            role=t.get("role") or t.get("from") or "user",
            text=t.get("text") or t.get("value") or "",
            timestamp=t.get("timestamp") or "",
        )
        for t in turns_data
    ]
    return Conversation(
        session_id=str(session_id),
        source=f"public:{dataset_name}",
        turns=turns,
    )


# ============================================================================
# GPT-4 合成(stub — 实际执行需要 OPENAI_API_KEY)
# ============================================================================


def synthesize_via_gpt4(
    client: Any,
    model: str = "gpt-4o",
    n_conversations: int = 100,
    scenarios: list[str] | None = None,
    max_turns: int = 8,
    temperature: float = 0.8,
) -> Iterator[Conversation]:
    """调用 GPT-4 合成代码项目对话(stub)

    参数:
        client: openai.OpenAI 实例(或同构 API client)
        model: 模型名,默认 gpt-4o
        n_conversations: 合成多少条
        scenarios: 场景列表,None 则从内置场景随机
        max_turns: 每条最大轮数
        temperature: 采样温度

    产出:
        yield: Conversation 对象(每调一次 API yield 一条)

    实现要求(P1-1 实际执行时):
        1. 调用 client.chat.completions.create
        2. prompt 使用 SYNTHESIS_PROMPT_TEMPLATE
        3. response.choices[0].message.content 应为合法 JSON
        4. 解析失败则 yield None(由 caller 跳过)
        5. 单条成本:~¥0.5-1(根据 turn 数),建议预算上限 ¥150
    """
    scenarios = scenarios or [
        "NPE in src/auth/login.py::validate_token",
        "REST → gRPC 迁移策略",
        "2FA in login module",
        "查询慢,加索引",
    ]

    for i in range(n_conversations):
        scenario = random.choice(scenarios)
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(scenario=scenario, max_turns=max_turns)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            record = json.loads(content)
            turns_data = record.get("turns", [])
            turns = [
                ConversationTurn(role=t.get("role", "user"), text=t.get("text", ""))
                for t in turns_data
            ]
            yield Conversation(
                session_id=f"syn_{i:04d}",
                source=f"synthetic:gpt4:{model}",
                turns=turns,
            )
        except Exception as e:
            print(f"[synthesis] 跳过第 {i} 条:{type(e).__name__}: {e}")
            continue


# ============================================================================
# 数据集混合(写出 JSONL)
# ============================================================================


def write_conversations(
    conversations: Iterator[Conversation],
    output_path: Path,
    append: bool = False,
) -> int:
    """将 Conversation 流写入 JSONL 文件

    返回:写入条数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with open(output_path, mode, encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_jsonl_record(), ensure_ascii=False) + "\n")
            count += 1
    return count


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI 入口(占位,实际执行需要 API key)"""
    import argparse

    parser = argparse.ArgumentParser(description="P1-1 数据合成 CLI(stub)")
    parser.add_argument("--source", choices=["public", "synthetic", "mixed"], default="mixed")
    parser.add_argument("--output", type=Path, default=Path("data/agenticmemory_training/v0/conversations.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--openai-key", default=None)
    args = parser.parse_args()

    print(f"P1-1 stub:source={args.source}, output={args.output}, limit={args.limit}")
    print("注意:此脚本为骨架,实际执行需要:")
    print("  1. 公开集:手动下载 + 放入 data/public/ 目录")
    print("  2. GPT-4 合成:设置 OPENAI_API_KEY 环境变量")
    print("  3. 取消 main() 末尾的 raise,启用对应逻辑")


if __name__ == "__main__":
    main()
---
name: p1-poc
description: P1 PoC 验证:OpenCode → Python(agenticmemory_training)→ Qwen3.5-0.8B 推理服务。用于验证 P1 最小闭环实验的基础设施链路可行性。
---

# P1 PoC 执行

本 skill 让 OpenCode 编排 P1 最小闭环实验的基础设施验证(计划 Task 5)。

## PoC-1:验证 Python 包可被编排

运行以下命令验证 agenticmemory_training 包可被加载:

```bash
python3 -c "import agenticmemory_training; print('OK')"
python3 -m unittest agenticmind.tests.test_extraction -v
```

## PoC-2:验证 Qwen3.5-0.8B 推理服务

前提:Qwen3.5-0.8B vLLM 服务已在 `localhost:8998` 运行(见下方"启动推理服务")。

```bash
curl -s http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-0.8B", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 32}'
```

### 启动推理服务(可选,若未运行)

```bash
# 若 vLLM 未装: pip install vllm
vllm serve Qwen/Qwen3.5-0.8B --port 8998 --max-model-len 2048 &
sleep 30 && curl -s http://localhost:8998/v1/models | head -c 200
```

## PoC-3:端到端流程闭环

1. 用 Kimi-K3 合成 1 条对话(P1-1 代码路径,需 KIMI_API_KEY)
2. 用 Qwen3.5-0.8B 推理服务对该对话 zero-shot 抽取
3. 验证输出 JSON 13 字段可解析

```bash
python3 - <<'PY'
from pathlib import Path
from agenticmemory_training.data.synthesis import synthesize_via_gpt4, write_conversations
print("synthesis 模块可导入")
import inspect
print("签名:", inspect.signature(synthesize_via_gpt4))
PY
```

## 已知边界(🟡-5)

- OpenCode 项目级技能目录约定为 `.opencode/skills/`(非 `.cl/`)
- 若项目级不被加载,回退到全局 `~/.config/opencode/skills/p1-poc/SKILL.md`
- 模型可用性:若 Qwen3.5-0.8B 不可下载 → fallback Qwen3.5-1.5B → 仍不可用则暂停并启动 F-07 前置决策

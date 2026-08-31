# P1 最小闭环实验 — 会话交接文档 (Handoff)

> **交接日期**: 2026-09-01
> **交接理由**: Part B 实验执行遇到**外部资源阻塞**(Kimi API 401 / DeepSeek 402 / 无 GPU / 内存不足),需用户在真实环境就绪后恢复。本文档确保新会话无需重读全部 spec/plan 即可无缝衔接。
> **上游文档**:
> - Spec: `docs/superpowers/specs/2026-08-31-p1-minimum-loop-fixes-design.md` (v0.3)
> - Plan: `docs/superpowers/plans/2026-08-31-p1-minimum-loop-fixes.md` (12 任务)
> - 原始实验: `docs/agenticmemory_training/08c-p1-minimum-loop.md`

---

## 1. 一句话状态

**P1 最小闭环实验的代码改造(Part A)已全部完成并合并 master(44 tests OK);实验执行(Part B)推进到 P1-1 腿 A 完成(100 条公开集),因外部资源缺失暂停。**

## 2. 已完成工作(勿重复)

### 2.1 设计与治理(commit `ecc1f22`)
- [x] `docs/superpowers/specs/2026-08-31-p1-minimum-loop-fixes-design.md` v0.3(Metis 评审整改,含 4 阻塞级修复)
- [x] `docs/superpowers/plans/2026-08-31-p1-minimum-loop-fixes.md`(12 任务)
- [x] `AGENTS.md` v1.3.1:
  - F-04 模型选型行: Qwen3-0.6B → **Qwen3.5-0.8B**
  - 新增 F-06(架构与训练解耦 + 记忆优先)
  - 新增 §6.5 四原则

### 2.2 Part A 代码修复(commits `620a0ca` → `a06ddcf`, 44 tests OK)

| 改动 | 文件 | 说明 |
|---|---|---|
| 教师参数化 + metadata.teacher | `agenticmemory_training/data/synthesis.py` | `synthesize_via_gpt4(client, model="kimi-k3")`;`Conversation` 增加 `teacher` 字段,`to_jsonl_record()` 输出 `metadata.teacher` |
| zero-shot baseline 新脚本 | `agenticmemory_training/training/eval_zero_shot.py` | base 无 adapter,按字段 F1,写 `baseline_f1.json` |
| random-label 对照新脚本 | `agenticmemory_training/training/eval_random_label.py` | gold-shuffle 检测表面映射,batch_size 透传 |
| IRR 支持 | `agenticmemory_training/data/teacher_labeling.py` | `compute_krippendorff_alpha()`(NLTK coincidence)+ `label_irr_subset(client_a, client_b, ...)`(双 client,长度校验,empty 保护) |
| ultrachat 公开集适配 | `agenticmemory_training/data/synthesis.py` + test | `PUBLIC_DATASET_REGISTRY` 注册 + `_public_record_to_conversation` 支持 messages/content/prompt_id |

**测试文件**(全部 class-based, 44 tests): `agenticmind/tests/test_extraction`(33) + `test_synthesis_teacher`(2) + `test_eval_zero_shot`(2) + `test_eval_random_label`(1) + `test_teacher_irr`(3) + `test_public_dataset_ultrachat`(3)

### 2.3 Part B 已推进部分(commits `21fe57d` → `f8fefd5`)

| 阶段 | 状态 | 产出 |
|---|---|---|
| PoC-1 OpenCode/SKILL.md | ✅ | `.opencode/skills/p1-poc/SKILL.md` |
| PoC-2 模型可用性 | ✅ | Qwen3.5-0.8B 可经 hf-mirror 下载(config.json 2907 bytes);**fallback 未触发** |
| P1-1 腿 A 公开集 | ✅ | `data/agenticmemory_training/v0/conversations.jsonl` — **100 条** ultrachat_200k 多轮对话(3-13 turns)<br>来源: SHARELY(git 超时)/ lmsys(gated 403)不可用 → ultrachat via hf-mirror |
| P1-3 evaluation 管线 | ✅ | smoke test 通过(纯 stdlib,可独立运行) |
| 环境检查工具 | ✅ | `scripts/check_env.py` — 退出码 0/1/2,含 API 生成探测 |

## 3. 当前阻塞(需用户准备)

| 阻塞项 | 证据 | 需要什么 |
|---|---|---|
| **Kimi API key 无效** | `scripts/check_env.py`: HTTP 401 Invalid Authentication | 有效 `KIMI_API_KEY`(Moonshot 平台生成新 key) |
| **DeepSeek API 余额不足** | `scripts/check_env.py`: HTTP 402 Insufficient Balance | DeepSeek 平台充值或换有余额的 key |
| **无 GPU** | `nvidia-smi: command not found` | NVIDIA GPU ≥8GB(或含 GPU 的远程环境) |
| **内存不足** | 可用 ~2.1GB < Qwen3.5-0.8B fp32 需求 3.2GB | 更多 RAM 或 bitsandbytes(int8,需安装) + 或换更小模型 |

## 4. 恢复路径(新会话执行顺序)

```bash
# Step 0: 环境就绪检查(应输出全绿 / 退出码 0)
python3 scripts/check_env.py

# Step 1: PoC-3 端到端(合成 1 条 → 推理 → 13 字段解析)
#   (需 Kimi key 就绪 + Qwen3.5-0.8B 推理服务)
vllm serve Qwen/Qwen3.5-0.8B --port 8998 &

# Step 2: P1-1 腿 B Kimi-K3 合成 70 条(append 到 conversations.jsonl)
export OPENAI_BASE_URL="https://api.moonshot.cn/v1"; export OPENAI_API_KEY="$KIMI_API_KEY"

# Step 3: P1-2 DeepSeek 主标注 + Kimi-K3 IRR 子集(50 条,双 client)
#   label_irr_subset(client_a=DeepSeek, client_b=Kimi, ...) → irr_krippendorff.json

# Step 4: P1-3 评估(纯本地)
python3 -m agenticmemory_training.data.evaluation

# Step 5: P1-4-pre baseline(zero-shot + random-label)
python3 -m agenticmemory_training.training.eval_zero_shot
python3 -m agenticmemory_training.training.eval_random_label

# Step 6: P1-4 LoRA 训练 + 多字段联合判定
python3 -m agenticmemory_training.training.data_prep   # 注: main() 是 stub,需先接线或直接调用函数
python3 -m agenticmemory_training.training.lora_train
python3 -m agenticmemory_training.training.eval_f1

# Step 7: P1-5 失败诊断(条件性)+ findings_v0.md 5 章节
```

## 5. 关键注意事项(踩坑记录)

1. **网络**: HF 直连不可达,需 `export HF_ENDPOINT=https://hf-mirror.com`(check_env.py 已内置默认)
2. **数据源替代**: SHARELY(git 超时 exit=124)、lmsys-chat-1m(gated 403) 均不可用;ultrachat_200k(Open H4,非 gated)可用 — 已注册进 registry
3. **`data_prep.py` / `evaluation.py` 的 `main()` 是 stub**(pre-existing):核心逻辑函数(`build_training_samples`/`split_train_dev`/`evaluate`)可用且已 smoke 验证;CLI 需接线后才可 `python -m` 直接跑
4. **peft 已安装**(v0.20.0);`transformers` 4.57.6 / `torch` 2.12.0+cu130 已就位;**vllm / bitsandbytes 未装**
5. **模型**: Qwen3.5-0.8B 是 Qwen3_5ForConditionalGeneration(多模态结构,含 image_token),配 `linear_attention` 层 — CPU 推理极慢;不要用 CPU 跑 LoRA 训练(24 层 × 长序列会 OOM)
6. **坐标**: `label_irr_subset` 需要双 client(DeepSeek/Kimi 端点不同),签名 `(client_a, client_b, sessions, ...)`

## 6. 已知遗留(非本任务范围)

- 工作区 `docs/architectures/*` `docs/research/*` 有更早会话未提交修改 — 与本 P1 任务无关,恢复会话时勿混入提交
- `ruff` 在 `evaluation.py`(statistics 未用)+ `data_prep.py`(f-string 无占位符)有 2 处 pre-existing lint — 可选顺手修,非阻塞

---

**交接结束**。恢复时先跑 `python3 scripts/check_env.py`,绿了按 §4 顺序执行即可。
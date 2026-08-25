# trainer/ AGENTS.md

## OVERVIEW

训练入口脚本套件（18 个独立脚本 + 共享工具），继承自 MiniMind fork。生产模型权重，供下游 AgenticDSL 生成任务微调使用。**非流水线编排**——各脚本为平级 sibling，可独立运行，组合方式由调用方决定。

## WHERE TO LOOK

| 脚本 | 算法 | Base 模型 | 预期数据集 | 输出权重 |
|---|---|---|---|---|
| `train_pretrain.py` | Next-token pretrain | MiniMindForCausalLM | 原始文本语料 | `pretrain_{hidden_size}[_moe].pth` |
| `train_full_sft.py` | Full-param SFT（CE Loss）| MiniMindForCausalLM | 指令微调对 | `full_sft_{hidden_size}[_moe].pth` |
| `train_lora.py` | LoRA（手动实现，无 peft 封装核心循环）| MiniMindForCausalLM | 指令微调对 | `lora_{name}_{hidden_size}[_moe].pth` |
| `train_distillation.py` | White-box 蒸馏（CE + KL，teacher+student）| MiniMindForCausalLM |  teacher 软标签数据 | `distill_{hidden_size}.pth` |
| `train_dpo.py` | DPO（chosen/rejected pairs，policy+ref）| MiniMindForCausalLM | DPO 成对数据 | `dpo_{hidden_size}.pth` |
| `train_ppo.py` | PPO（Actor+Critic 双网络）| MiniMindForCausalLM | RL 轨迹数据 | `ppo_actor_{hidden_size}.pth` |
| `train_grpo.py` | GRPO（group-relative baseline；支持 CISPO via `loss_type`）| MiniMindForCausalLM | RL 轨迹数据 | `grpo_{hidden_size}.pth` |
| `train_agent.py` | Agentic RL（多轮 Tool-Use，GRPO/CISPO）| MiniMindForCausalLM | agent 轨迹数据 | `agent_{hidden_size}.pth` |
| `train_tokenizer.py` | BPE tokenizer 训练（6400 vocab）| — | 原始语料 | `tokenizer.model` |
| `train_dlm.py` | Discrete Diffusion LM（LLaDA-style，research）| MiniMindForCausalLM | 离散扩散训练数据 | `dlm_{hidden_size}.pth` |
| `train_medusa.py` | Medusa 投机解码头 | MiniMindForCausalLM + 投机头 | 任意因果数据 | `medusa_{hidden_size}.pth` |
| `train_rt_purbo.py` | RT-Purbo 检索头（2-stage）| MiniMindForCausalLM + 检索头 | 检索轨迹数据 | `rt_purbo_{hidden_size}.pth` |
| `train_nsa.py` | Native Sparse Attention（toy dataset，research）| MiniMindForCausalLM | toy 文本数据 | `nsa_{hidden_size}.pth` |
| `train_mhc.py` | Manifold-Constrained Hyper-Connections | MiniMindForCausalLM | 任意因果数据 | `mhc_{hidden_size}.pth` |
| `train_gated_deltanet.py` | Gated DeltaNet 线性 recurrent | MiniMindForCausalLM | 任意因果数据 | `gated_deltanet_{hidden_size}.pth` |
| `train_lightning_indexer.py` | Lightning Indexer DSA 风格 | MiniMindForCausalLM | 任意因果数据 | `lightning_indexer_{hidden_size}.pth` |
| `train_dflash.py` | DFlash 块扩散投机 | MiniMindForCausalLM + 扩散头 | 任意因果数据 | `dflash_{hidden_size}.pth` |
| `train_hybrid.py` | Hybrid（DeltaNet + Gated Attention，Qwen3-Next 风格）| MiniMindForCausalLM | 任意因果数据 | `hybrid_{hidden_size}.pth` |

**流水线顺序（可选分支）**：`pretrain → full_sft → (lora / distillation / dpo / ppo / grpo / agent)`。无强制顺序，调用方自行组合。

## SHARED UTILITIES

### trainer_utils.py

| 函数/类 | 用途 |
|---|---|
| `get_lr` | 学习率调度（linear warmup + cosine decay）|
| `lm_checkpoint` | 权重保存与加载（state_dict 序列化）|
| `init_model` | 模型初始化（from scratch / from checkpoint）|
| `SkipBatchSampler` | 动态跳过 batch（梯度累积场景下 max_token 兜底）|
| `LMForRewardModel` | Reward model 头封装（供 PPO/GRPO Critic 使用）|
| `init_distributed_mode` | DDP 初始化（`torchrun` 环境下自动检测 world_size > 1）|

**添加新工具原则**：如果函数被 ≥2 个脚本 import，则迁入 `trainer_utils.py`。单脚本专用逻辑留在各自文件。

### rollout_engine.py

| 类/函数 | 用途 |
|---|---|
| `RolloutEngine` | RL 轨迹采样引擎（从 policy 采样 action 序列，收集 reward）|
| 支持 `train_grpo.py`、`train_agent.py` 的在线轨迹收集 | — |

## CONVENTIONS

**运行前提**：所有脚本必须在 `trainer/` 目录下运行：
```bash
cd trainer && python train_xxx.py
```
**单 GPU**：直接 `python`；**多 GPU DDP**：`torchrun --nproc_per_node N`。

**配置方式**：纯 CLI argparse，无 YAML/JSON 配置文件。关键参数：`--save_dir`（默认 `../out`）、`--save_weight`（前缀决定最终命名）、`--hidden_size`（默认 768）、`--num_hidden_layers`（默认 8）、`--use_moe`（0/1）、`--max_seq_len`、`--data_path`、`--from_weight`、`--from_resume`。

**命名合约**：由 `--save_weight` 决定前缀，脚本内硬编码后缀拼接（见 WHERE TO LOOK 表）。新增脚本必须遵守。

## ANTI-PATTERNS

1. **禁止硬编码路径**：数据路径、输出路径必须通过 argparse 传入，绝不在脚本内写死 `~/xxx` 或绝对路径。
2. **禁止跳过 resume 合约**：checkpoint 保存时必须携带完整训练状态（optimizer.step()、epoch、global_step）。resume 时必须传 `--from_resume 1` + `--from_weight <path>`。
3. **禁止新增权重名不更新文档**：任何新增 `--save_weight` 选项的脚本，必须同步更新本文档 WHERE TO LOOK 表。
4. **禁止多脚本间隐式依赖**：脚本是独立进程，不通过全局变量或文件系统隐式传递状态。

## NOTES

- 18 个脚本为平级 sibling，无统一 dispatcher。新增训练阶段推荐以 `train_grpo.py` 为模板（RL 类）或 `train_full_sft.py` 为模板（生成类）。
- `train_tokenizer.py` 独立运行，输出 `tokenizer.model`，被所有下游脚本消费。
- 研究类脚本（`train_dlm.py` / `train_nsa.py` / `train_mhc.py` 等）使用 toy dataset，不保证生产可用。
- SwanLab 已成 2025+ 默认监控（`--use_wandb` flag 保留，语义等价）。

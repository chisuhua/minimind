# 训练算法 — 6 阶段自训练 Recipe

> **文档 ID**: LLMTRN-001-ALGO
> **生成日期**: 2026-06-10
> **关联**:
> - 数据管线: [`01-training-data-pipeline.md`](01-training-data-pipeline.md)
> - 综述: [`README.md`](README.md)
> - 风险登记: [`05-risk-register.md`](05-risk-register.md)

---

## 1. 核心原则

基于 2024-2026 SOTA 自训练方法（STaR、ReSTᴱᴹ、Voyager、AlphaCode、DeepSeekMath-GRPO、OmegaPRM）的综合，AgenticDSL 自训练遵循以下原则：

1. **完全 bootstrap**：零人类偏好、零/极少人类标注
2. **HydraForge runtime = 终极 verifier**：执行反馈是硬过滤
3. **多层 reward**：L1 (format) + L2 (semantic) + L3 (execution) + L4 (task)
4. **KL penalty**：防 policy drift (β=0.04, DeepSeekMath 标准)
5. **Human spot-check 周期**：每 1000 step 200 prompts
6. **EM iteration 限制**：≤5 轮（Singh 2024 警告）

---

## 2. 6 阶段训练 Recipe

```
┌─────────────────────────────────────────────────────────┐
│ 阶段 0: 环境与验证器搭建 (工程, 1 周)                     │
│ • 暴露 HydraForge runtime 为 CLI 子命令                  │
│ • 实现 4 层验证器                                         │
│ • 实现 DSL canonical serializer (固定缩进、键序)          │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 1: 冷启动 SFT (Distillation, 2 周)                  │
│ 输入: GPT-4/Claude API 生成的 10K-50K 高质量 trace        │
│       + 200 个手工编写的 few-shot                          │
│ 算法: 标准 cross-entropy SFT, 8192 ctx, FA2              │
│       ~96% sample packing efficiency (Hermes 经验)       │
│ Base: Qwen2.5-Coder-7B-Instruct / Llama-3.1-8B          │
│ 目标: 通过率 > 30% (不追求 90%+)                          │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 2: ReSTᴱᴹ Bootstrap (3-5 轮, 2 周)                │
│ for iter in 1..5:                                        │
│   E-step:                                                │
│     1. 用当前策略对 N=32 prompts 各采样 K=16 DSL          │
│     2. 执行每条, 收集 4 层 reward                         │
│     3. 过滤: reward > threshold (0.5-0.7)                │
│     4. 对失败: rationalization (给 ground-truth DSL,      │
│        让模型解释为何对)                                  │
│   M-step:                                                │
│     5. **从 base model** 重新 fine-tune                  │
│        (不是从 iter-1 继续! Singh 2024 关键教训)          │
│     6. 2-5 epochs per iter                               │
│ ⚠️ Singh et al. 警告: 每次 EM 必须从 base 重新开始,      │
│    否则过拟合.                                           │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 3: 自动 PRM 训练 (OmegaPRM/Math-Shepherd 风格, 1 周)│
│ • 自动标注: 对 10K prompts                                │
│   - 生成 K=16 candidate DSLs                             │
│   - 全部执行                                              │
│   - binary-search MCTS:                                  │
│     * 对每个 DSL prefix, 跑 K=8 continuations            │
│     * 标记首个失败节点为 negative                         │
│     * 其前缀为 positive                                   │
│ • 训练 PRM (1B-3B classifier)                            │
│ 资源: ~800K rollouts, ~22 GPU-hours                      │
│ HydraForge 应用: 用 LayeredContext trace + expected_output│
│   作为 step-level label 信号                              │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 4: MCTS 数据增强 (LATS 风格, 1 周)                  │
│ • 用 PRM as value function                                │
│ • 训练时 MCTS 收集高价值 traces                           │
│ • 把这些 traces 加到 SFT 池                                │
│ 资源: ~100 GPU-hours                                     │
│ HydraForge 应用: 节点级 PRM label 对应 L1-L4 节点类型    │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 5: GRPO 精调 (核心 RL 阶段, 3-4 周)                  │
│ for step in range(N_steps):                              │
│   1. Rollout: 对 G=16-32 个 prompts 各采样 G 个 DSL      │
│   2. Execute: runtime 收集 reward                        │
│   3. Group-relative advantage (无 critic!):              │
│      A_i = (r_i - mean(r)) / std(r)                      │
│   4. GRPO loss:                                          │
│      L = -mean(min(π_ratio · A, clip(π_ratio, 1±ε) · A)) │
│        + β · KL(π || π_ref)         # β=0.04              │
│   5. Update policy                                       │
│ 每 1000 step 评估 pass@k                                  │
│ 关键: KL penalty + 多样化 reward + spot-check             │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 6: SPIN 自博弈 (可选, 1 周)                         │
│ • 引入 Self-Play Fine-Tuning (Chen ICML2024)             │
│ • 当前模型输出作"自生成负样本"                            │
│ • SFT 数据作"正样本"                                      │
│ • 训练区分两者                                            │
│ • iter 0→1→2, distill 更多 capability                    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 算法选型决策树

### 3.1 算法对比

| 算法 | 是否适合 AgenticDSL | 关键文献 | 适用阶段 |
|---|---|---|---|
| **SFT** | ✅ 必备（冷启动）| Hermes3 (NousResearch 2024) | 阶段 1 |
| **RFT** (Rejection Fine-Tuning) | ✅ 推荐（ReSTᴱᴹ 内核）| LLaMA-RFT (Meta 2023) | 阶段 2 |
| **DPO** | ⚠️ 次选 | Rafailov et al. NeurIPS 2023 | 阶段 6（可选）|
| **IPO** | ⚠️ 次选 | Azar et al. AISTATS 2024 | 同上 |
| **KTO** | ✅ 备选 | Ethayarajh et al. ICML 2024 | 阶段 5（reward 信号不够时）|
| **SimPO** | ⚠️ 备选 | Meng et al. NeurIPS 2024 | 阶段 5/6（内存友好）|
| **GRPO** | ✅✅ **首选** | Shao et al. DeepSeekMath 2024 | 阶段 5 |
| **PPO** | ❌ 不推荐 | - | 不建议 |
| **SPIN** | ✅ 可选 | Chen et al. ICML 2024 | 阶段 6 |
| **ReSTᴱᴹ** | ✅✅ **核心** | Singh et al. TMLR 2024 | 阶段 2 |
| **PRM (OmegaPRM)** | ✅✅ **核心** | Luo et al. DeepMind 2024 | 阶段 3 |
| **MCTS + PRM** | ✅✅ | LATS / AlphaCode | 阶段 4 |
| **Self-Refine** | ⚠️ 推理时 | Madaan et al. NeurIPS 2023 | 推理时（不训练）|

### 3.2 推荐组合

- **阶段 1-2: SFT + RFT**（ReSTᴱᴹ 范式）
- **阶段 3-4: PRM 训练**（OmegaPRM 自动标注）
- **阶段 5: GRPO**（DeepSeekMath 范式，无 critic）
- **阶段 6: SPIN**（可选）

### 3.3 RL 算法对比详解

| 算法 | 需要偏好对 | 需要 critic | 内存 | 推荐场景 |
|---|---|---|---|---|
| **SFT (RFT)** | 否 | 否 | 低 | 冷启动、warmup |
| **DPO** | 是 | 否 | 中 | 有偏好对 |
| **IPO** | 是 | 否 | 中 | 偏好信号确定（AgenticDSL 适合）|
| **KTO** | 否（二元）| 否 | 中 | 只有"好/坏"标签 |
| **SimPO** | 是 | 否 | 低（无 ref）| DPO 替代，更稳 |
| **GRPO** | 否（reward 即可）| 否 | 中 | 推理可大规模并行采样 |
| **PPO** | 否（reward 即可）| 是 | 高 | 基线，不推荐 |

---

## 4. 关键算法详解

### 4.1 ReSTᴱᴹ（Singh et al. TMLR 2024）

**核心**：把 Self-Training 看作 EM 算法。
- **E-step**：从当前策略采样 → 二元过滤
- **M-step**：对 base 模型 fine-tune（不是从上一轮初始化！）

**⚠️ 关键警告**：每轮必须 restart from base model，避免过拟合。

**AgenticDSL 实施**：

```python
# Stage 2 ReSTᴱᴹ Bootstrap
base_model = load_qwen_coder_7b()
policy = base_model

for iter in range(5):  # 5 iterations
    # E-step: sample and filter
    samples = []
    for prompt in prompts[:1000]:
        for k in range(16):
            dsl = policy.generate(prompt)
            reward = runtime.compute_reward(dsl, prompt)
            samples.append({"prompt": prompt, "dsl": dsl, "reward": reward})
    
    # Filter: keep only reward > threshold
    good_samples = [s for s in samples if s["reward"] > 0.5]
    
    # Rationalization for failures
    for s in samples:
        if s["reward"] < 0.3:  # failed
            # Ask policy to explain ground-truth
            best_dsl = find_best_dsl(s["prompt"], good_samples)
            rationale = policy.generate(
                f"Given prompt '{s['prompt']}', "
                f"why does this DSL work? {best_dsl}"
            )
            good_samples.append({"prompt": s["prompt"], "dsl": best_dsl, "rationale": rationale})
    
    # M-step: fine-tune from BASE model
    policy = copy(base_model)  # ⚠️ restart from base, not previous iter
    policy.fine_tune(
        good_samples,
        epochs=3,
        lr=2e-5,
        batch_size=32,
    )
```

**资源估算**：
- 1000 prompts × 16 samples = 16K generations
- ~16K dry-run executions ≈ 30 minutes
- 5 iterations × 3 epochs ≈ 50 GPU-hours

### 4.2 GRPO（DeepSeekMath）

**核心**：PPO 的简化 —— 取消 critic，用组内相对优势估计 baseline。

**AgenticDSL 实施**：

```python
# Stage 5 GRPO 精调
policy = load_hydra_agenticdsl_7b_v2()  # 来自 Stage 2
ref_policy = copy(policy)  # KL reference (frozen)
optimizer = Adam(policy.parameters(), lr=1e-6)

for step in range(num_steps):
    # 1. Rollout
    prompts = sample_prompts(batch_size=32)
    samples = []
    for prompt in prompts:
        group = []
        for g in range(16):  # G=16
            dsl = policy.generate(prompt)
            reward = runtime.compute_reward(dsl, prompt)
            group.append({"prompt": prompt, "dsl": dsl, "reward": reward})
        samples.extend(group)
    
    # 2. Group-relative advantage (NO critic)
    for prompt_group in group_by_prompt(samples):
        rewards = [s["reward"] for s in prompt_group]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        for s in prompt_group:
            s["advantage"] = (s["reward"] - mean_r) / std_r
    
    # 3. GRPO loss
    loss = 0
    for s in samples:
        log_ratio = s["new_log_prob"] - s["old_log_prob"]
        ratio = np.exp(log_ratio)
        clip_ratio = np.clip(ratio, 1 - 0.2, 1 + 0.2)
        surrogate = min(ratio * s["advantage"], clip_ratio * s["advantage"])
        
        # KL penalty
        kl = compute_kl(policy, ref_policy, s["prompt"])
        loss -= surrogate - 0.04 * kl
    
    # 4. Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 5. Periodic evaluation
    if step % 1000 == 0:
        evaluate_pass_at_k(policy, held_out_prompts)
```

**关键参数**：
- G=16-32（DeepSeekMath 用 G=64，AgenticDSL 可用更小）
- β=0.04（KL penalty）
- clip ε=0.2（PPO/GRPO 标准）
- 学习率 1e-6（精调阶段）

**DeepSeekMath 数字参考**：
- 7B 模型 + 120B math tokens pretrain
- GRPO without critic
- MATH 46.8% → 51.7%（GRPO 提升段）
- 整体 51.7% MATH, 60.9% with 64-sample self-consistency

### 4.3 自动 PRM（OmegaPRM 风格）

**核心**：用 divide-and-conquer MCTS（AlphaGo Zero 启发）自动收集 process supervision。

**AgenticDSL 实施**：

```python
# Stage 3 Auto PRM
# 收集 step-level labels
labels = []
for prompt in prompts[:10000]:
    # 1. 生成 K=16 candidate DSLs
    candidates = [policy.generate(prompt) for _ in range(16)]
    
    # 2. 执行全部
    execution_results = [runtime.execute(c) for c in candidates]
    
    # 3. Binary-search MCTS
    for prefix_len in range(1, max_depth):
        # 对每个 prefix 跑 K=8 continuations
        continuations = []
        for _ in range(8):
            partial_dsl = truncate(candidates[0], prefix_len)
            full_dsl = policy.continue_from(partial_dsl)
            continuations.append(runtime.execute(full_dsl))
        
        success_rate = sum(c.success for c in continuations) / 8
        if success_rate < 0.3:
            # 该 prefix 的下一个节点是 negative
            labels.append({
                "prompt": prompt,
                "prefix": partial_dsl,
                "step": partial_dsl[-1],
                "label": 0,  # negative
            })
        else:
            labels.append({
                "prompt": prompt,
                "prefix": partial_dsl,
                "step": partial_dsl[-1],
                "label": 1,  # positive
            })

# 训练 PRM
prm = train_classifier(
    model="Qwen2.5-Coder-1.5B",
    data=labels,
    loss="MSE",  # step-level regression
    epochs=3,
)
```

**资源估算**：
- 10K prompts × K=16 × avg depth=5 = ~800K execution rollouts
- 单 rollout 假设 1s → 22 GPU-hours
- PRM 训练：~1 GPU-hour

**OmegaPRM 数字参考**（DeepMind 2024）：
- 1.5M+ 自动 process annotations（零人工）
- Gemini Pro: MATH500 51% → 69.4% (+18.4 绝对)
- Gemma2 27B: MATH500 42.3% → 58.2%, GSM8K 74.0% → 92.2%

### 4.4 MCTS + PRM（LATS 风格）

**核心**：PRM as value function + MCTS 收集高价值 traces。

**AgenticDSL 实施**：

```python
# Stage 4 MCTS + PRM
def mcts_search(root_prompt, n_simulations=100):
    tree = MCTSNode(state=root_prompt)
    
    for _ in range(n_simulations):
        # Selection
        node = tree.select_child(ucb_c=1.41)
        
        # Expansion: 生成 K 个 children (DSL extensions)
        for k in range(8):
            child_dsl = policy.generate_extension(node.state)
            reward = prm.predict(child_dsl)  # 用 PRM as value
            node.add_child(child_dsl, reward)
        
        # Simulation: rollout to terminal
        result = runtime.execute(node.best_dsl)
        
        # Backpropagation
        node.backpropagate(result.reward)
    
    return tree.best_dsl()
```

**资源**：~100 GPU-hours（100K simulations）

**LATS 数字参考**：
- HumanEval pass@1: 94.4% (GPT-4)
- WebShop 平均分 75.9 (GPT-3.5) —— 比 ReAct +22.1 分
- HotPotQA：比 ReAct 翻倍

### 4.5 SPIN（Self-Play Fine-Tuning，Chen et al. ICML 2024）

**核心**：让模型从自己的输出学习区分。

**AgenticDSL 实施**：

```python
# Stage 6 SPIN（可选）
for spin_iter in range(3):
    # 当前模型生成"自己"的样本
    current_samples = [policy.generate(p) for p in prompts]
    
    # SFT 数据作"正样本"
    sft_samples = load_sft_data()
    
    # 训练区分: DPO with (sft_samples, current_samples)
    dpo_trainer.train(
        preferred=sft_samples,  # 正样本
        rejected=current_samples,  # 自生成负样本
    )
```

**SPIN 数字参考**：
- Self-Play Fine-Tuning，从小到大
- 关键：当前模型 vs 前一轮模型 vs 人类 SFT 数据 → preference loss

---

## 5. Reward 形状设计

针对 AgenticDSL 多步执行的特性：

```python
def compute_reward(
    generated_dsl: str,
    parse_result: ParseResult,
    schema_result: ValidationResult,
    exec_result: ExecutionResult,
    task_result: TaskResult,
) -> float:
    """
    4 层 reward 加权
    """
    r_format = 0.4 if parse_result.success else 0.0
    r_progress = 0.2 * (exec_result.nodes_executed / max(exec_result.expected_nodes, 1))
    r_outcome = 0.3 if task_result.matches_expected else 0.0
    r_efficiency = 0.1 * (1.0 - exec_result.budget_used_ratio)

    total = r_format + r_progress + r_outcome + r_efficiency
    return min(max(total, 0.0), 1.0)
```

**权重设计逻辑**：
- **Format (0.4) 最重**：格式合规是硬门槛
- **Outcome (0.3)**：任务级正确性
- **Progress (0.2)**：部分完成的进度
- **Efficiency (0.1)**：预算遵守

**Reward Hacking 风险**：
- 模型可能学会生成"格式完美但实际无意义"的 DSL
- 缓解：Progress + Outcome 强制"做对事"，不仅是"做对形"

---

## 6. 防 Reward Hacking（Goodhart）

参考 Gao 2023 "Scaling Laws for Reward Model Overoptimization"（ICML2023）：

**核心发现**：proxy reward 优化越多，gold reward 反而下降。`R_RL(d) = d(α_RL − β_RL log d)` —— 优化步数 d 增加时 gold reward 增长递减。

### 6.1 DSL-specific 防 hacking 措施

1. **不要只用 final outcome reward** → 加 PRM（per-node）
2. **KL penalty β=0.04** 防止 policy 偏离 reference
3. **多样化 reward**：format (L1) + semantic (L2) + execution (L3) + task (L4)
4. **Human spot-check 周期**：每 1000 step 跑 200 held-out prompts 看 gold reward 不下降
5. **Format compliance 不应降至 95% 以下**（监控）
6. **多样性指标**：unique DSL patterns 不应单调下降
7. **EM iter 限制**：≤5 轮（Singh 2024 警告）

### 6.2 Verifier 设计（综合 Gorilla + Math-Shepherd + AlphaCode）

```
DSL 输出
   │
   ▼
[L1] 语法/编译检查（最便宜，秒级）
   │
   ▼ pass
[L2] AST 匹配/结构性检查（便宜）
   │
   ▼ pass
[L3] 运行时执行 + 行为对比（贵但必须）
   │
   ▼ pass
[L4] 任务级奖励/端到端指标（最贵）
   │
   ▼
最终 reward = w1*L1 + w2*L2 + w3*L3 + w4*L4
```

### 6.3 LLM-as-Judge on top of rule-based

**何时加**：
- Rule-based verifier 无法判定的 semantic quality（DSL 是否"优雅"、是否考虑 edge case）
- Reward shaping 需要细粒度 feedback

**何时不加**：
- LLM judge 自身可能 hack（GPT-4 偏好 verbose answer）
- 增加延迟和成本

**实践**：用 rule-based 做 hard filter（pass/fail），LLM-as-judge 仅做 soft ranking（BoN reranker）。

### 6.4 Skalse et al. NeurIPS 2022 警示

参考 [Defining and Characterizing Reward Hacking](https://arxiv.org/abs/2209.13085)：
- Reward hacking 是 proxy reward 的固有风险
- 不能完全消除，只能通过多样化 reward + KL penalty + spot-check 缓解

---

## 7. 冷启动策略

### 7.1 Few-shot Prompting Warmup

**证据**：
- Voyager 直接 zero-fine-tune 用 GPT-4 + 自动课程
- AlphaCode 先 pretrain on GitHub
- STaR 用 few-shot rationale prompts

**建议**：用 5-50 个精心编写的 (state, DSL, reward) few-shot examples 让模型"看到"格式和任务。

### 7.2 Distillation from Larger Model

**证据**：
- DeepSeekMath（120B → 7B）：先收集 large-model traces → SFT 小模型 → GRPO
- Quiet-STaR：从 Mistral 7B 起步，但前提是 base 模型已有强 base capability
- SPIN：Self-Play Fine-Tuning，从 SFT 数据 + 自生成负样本

**策略**：
1. **Stage 0**：用 GPT-4 / Claude API 生成 5K-50K (state, DSL, reward) 轨迹 → SFT 小模型
2. **Stage 1**：用小模型生成 → 执行 → self-train

### 7.3 Synthetic Traces（ReSTᴱᴹ 风格）

**优势**：零人类、零大模型依赖。
**风险**：reward hacking 比 distillation 严重。

---

## 8. TRL（HuggingFace）实施参考

所有 SOTA RL 方法在 TRL 都有可生产实现：

```python
from trl import (
    SFTTrainer,
    DPOTrainer,
    GRPOTrainer,
    PPOTrainer,
    KTOTrainer,
    RewardTrainer,
)

# Stage 1: SFT
sft_trainer = SFTTrainer(
    model="Qwen2.5-Coder-7B-Instruct",
    train_dataset=sft_dataset,
    args=SFTConfig(
        output_dir="./sft_v1",
        per_device_train_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        packing=True,  # 96% efficiency
        max_seq_length=8192,
    ),
)

# Stage 5: GRPO
grpo_trainer = GRPOTrainer(
    model="./sft_v2",
    reward_fn=runtime.compute_reward,  # HydraForge runtime as reward
    args=GRPOConfig(
        output_dir="./grpo_v3",
        per_device_train_batch_size=32,
        num_train_epochs=10,
        learning_rate=1e-6,
        beta=0.04,  # KL penalty
        num_generations=16,  # G
        max_completion_length=2048,
    ),
)
```

**关键参数对照**：

| Trainer | 关键参数 | 推荐值 |
|---|---|---|
| SFTTrainer | `learning_rate`, `packing` | 2e-5, True |
| GRPOTrainer | `beta`, `num_generations` | 0.04, 16-32 |
| DPOTrainer | `beta`, `reference_free` | 0.1, False |
| KTOTrainer | `beta`, `desirable_weight` | 0.1, 1.0 |
| SimPO | `beta`, `gamma_beta_ratio` | 2.0, 0.55 |

---

## 9. 总结

### 关键交付

| 维度 | 交付 | SOTA 依据 |
|---|---|---|
| **冷启动** | SFT + RFT (ReSTᴱᴹ) | Singh 2024 (TMLR) |
| **核心 RL** | GRPO（无 critic）| DeepSeekMath 2024 |
| **PRM** | 自动 OmegaPRM 风格 | Luo 2024 (DeepMind) |
| **MCTS** | LATS + PRM | Zhou 2024 (ICML) |
| **修复能力** | SCoRe 两阶段 RL | Kumar 2024 (ICLR) |
| **SPIN** | 可选自博弈 | Chen 2024 (ICML) |
| **Reward** | 4 层加权 + KL penalty | DeepSeekMath / Gao 2023 |
| **冷启动数据** | GPT-4/Claude distillation | DeepSeekMath 范式 |

### 关键隐藏陷阱

1. **EM iteration 必须从 base 重新开始**（Singh 2024）
2. **修复数据必须用 RL 训练**（SCoRe, ICLR 2025）
3. **Special tokens 必须加入 stop_token_ids**（Qwen issue #99）
4. **EM iter ≤ 5**（Singh 2024 过拟合警告）
5. **不要只用 final outcome reward**（Goodhart）

### 下一步

训练算法就绪后，进入 [`03-inference-time-guarantees.md`](03-inference-time-guarantees.md) 学习推理时如何兜底保证。

---

**文档版本**: v1.0
**Owner**: AgenticMind 训练团队
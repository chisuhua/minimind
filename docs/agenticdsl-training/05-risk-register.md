# 风险登记册与防 Goodhart 协议

> **文档 ID**: LLMTRN-001-RISK
> **生成日期**: 2026-06-10
> **关联**:
> - 算法: [`02-training-algorithms.md`](02-training-algorithms.md)
> - 数据管线: [`01-training-data-pipeline.md`](01-training-data-pipeline.md)
> - 评估: [`04-evaluation-benchmark.md`](04-evaluation-benchmark.md)

---

## 1. 风险登记表

### 1.1 关键风险（High Priority）

| ID | 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|---|
| **R-01** | **Reward hacking（模型学会生成骗 reward 的 DSL）** | 中 | 高 | 多层 reward + KL penalty + human spot-check + 格式不依赖 reward |
| **R-02** | **SFT on (broken, fixed) 塌缩** | 高 | 中 | SCoRe 警告：必须用 RL 而非 SFT 训练修复能力 |
| **R-03** | **EM iteration 过拟合** | 中 | 高 | Singh 2024 警告：每轮必须从 base 重新开始，≤5 轮 |
| **R-04** | **生成的 DSL 引用未注册工具** | 高 | 中 | XGrammar dynamic schema resolver + runtime signature validator |
| **R-05** | **生成的 DSL 包含未声明资源** | 中 | 高 | `/__meta__/resources` 启动验证 + 训练数据 canonical schema |
| **R-06** | **无限循环（生成无限 DAG）** | 中 | 高 | HydraForge runtime budget 控制；训练数据显式 max_nodes |
| **R-07** | **BPE 切分 anchor tokens** | 中 | 中 | 添加 special tokens + stop_token_ids（PickyBPE 范本）|
| **R-08** | **FIM tokens 在生成中泄漏** | 中 | 低 | 强制 stop_token_ids（Qwen issue #99 教训）|

### 1.2 中等风险（Medium Priority）

| ID | 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|---|
| **R-09** | **冷启动数据量不足** | 中 | 中 | GPT-4/Claude distillation (10K-50K) + OSS 工作流 backtranslation |
| **R-10** | **训练数据污染 HydraForgeBench** | 低 | 中 | n-gram 去污（OpenCodeInstruct 经验）|
| **R-11** | **LLM-as-judge 偏好 verbose output** | 中 | 中 | rule-based hard filter + LLM-judge 仅用于 BoN rerank |
| **R-12** | **跨 session 状态污染** | 低 | 高 | HydraForge LayeredContext + SessionRegistry 隔离 |
| **R-13** | **修复数据中错误类型不均衡** | 中 | 中 | 主动扰动时按错误类型均衡采样 |
| **R-14** | **PRM 标注的 binary-search 偏差** | 低 | 中 | 用 OmegaPRM 的 divide-and-conquer MCTS 减少偏差 |
| **R-15** | **多语言/特殊字符 OOV** | 低 | 低 | Canonical serializer 限制字符集 |
| **R-16** | **Inja 模板注入** | 低 | 高 | 训练数据显式标注 `{{`/`}}` 边界，runtime 转义 |

### 1.3 低风险（Low Priority）

| ID | 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|---|
| **R-17** | **指标波动（pass@1 ±3%）** | 高 | 低 | 多 random seeds 评估，置信区间报告 |
| **R-18** | **训练数据版权问题** | 低 | 中 | 仅使用开源数据（Hermes、Gorilla 等），自生成数据明确归属 |
| **R-19** | **模型更新导致部署中断** | 低 | 中 | A/B 测试 + 金丝雀发布 |
| **R-20** | **GPU 资源不足** | 中 | 中 | 按阶段评估资源需求，按里程碑重新规划 |

---

## 2. 防 Reward Hacking（Goodhart）协议

### 2.1 理论基础

参考 Gao, Schulman, Hilton, "Scaling Laws for Reward Model Overoptimization" (ICML 2023)：

**核心发现**：proxy reward 优化越多，gold reward 反而下降。

```
R_RL(d) = d(α_RL − β_RL log d)
```

- `d`：优化步数
- `α_RL`：reward 增长系数
- `β_RL`：Goodhart 系数

**含义**：随着优化步数 d 增加，gold reward 增长递减。这不是 bug，是 proxy reward 的固有特性。

参考 Skalse et al., "Defining and Characterizing Reward Hacking" (NeurIPS 2022)：
- Reward hacking 是 proxy reward 的固有风险
- 不能完全消除，只能通过多样化 reward + KL penalty + spot-check 缓解

### 2.2 DSL-specific 防 hacking 措施

#### 措施 1：多层 reward（不依赖单一信号）

```python
def compute_reward(generated_dsl, prompt):
    # L1: 格式合规（最便宜，硬门槛）
    r_format = 0.4 if markdown_parser_parses(generated_dsl) else 0.0
    
    # L2: 签名校验
    r_signature = 0.2 if signature_validator_passes(generated_dsl) else 0.0
    
    # L3: 沙箱执行
    exec_result = runtime.dry_run(generated_dsl)
    r_execution = 0.3 * (exec_result.nodes_executed / exec_result.expected_nodes)
    
    # L4: 任务级奖励（最贵）
    r_task = 0.1 if exec_result.matches_expected else 0.0
    
    return r_format + r_signature + r_execution + r_task
```

**关键**：L1 (format) **不依赖 learned reward**，由 parser 直接判断。这避免了"模型学会生成骗 reward 但格式错误"的 hacking。

#### 措施 2：KL penalty β=0.04

```python
# GRPO loss with KL penalty
loss = -mean(min(π_ratio * advantage, clip(π_ratio, 1±ε) * advantage)) \
       + 0.04 * KL(π || π_ref)
```

**β=0.04**：DeepSeekMath 标准。防止 policy 偏离 reference model 太远。

#### 措施 3：Human spot-check 周期

```python
# 每 1000 RL step 跑 200 held-out prompts
if step % 1000 == 0:
    gold_rewards = []
    for prompt in held_out_prompts[:200]:
        dsl = policy.generate(prompt)
        # Gold reward = 人工评估 or LLM-judge
        gold_reward = llm_judge_evaluate(dsl, prompt)
        gold_rewards.append(gold_reward)
    
    gold_mean = np.mean(gold_rewards)
    
    # 检查 gold reward 是否下降
    if gold_mean < best_gold_mean - 0.05:  # 下降超过 5%
        logger.warning(f"Gold reward dropped! Step {step}: {gold_mean:.3f}")
        # 触发 early stopping 或 KL penalty 提升
```

#### 措施 4：Format Compliance 监控

```python
# 监控 format compliance 不应降至 95% 以下
if step % 500 == 0:
    format_rate = evaluate_format_compliance(policy, test_prompts)
    if format_rate < 0.95:
        logger.warning(f"Format compliance dropped to {format_rate:.3f}")
        # 立即停止训练，回滚到上一个 checkpoint
```

#### 措施 5：多样性监控

```python
# 监控 unique DSL patterns 不应单调下降
if step % 1000 == 0:
    recent_dsls = collect_recent_dsls(step, window=1000)
    unique_patterns = extract_patterns(recent_dsls)
    diversity_score = len(unique_patterns) / len(recent_dsls)
    
    if diversity_score < diversity_baseline * 0.8:  # 下降超过 20%
        logger.warning(f"Diversity dropped! Score: {diversity_score:.3f}")
```

#### 措施 6：EM iter 限制

```python
# Singh 2024 警告：EM iter 超过 5 轮会过拟合
if re_st_em_iter > 5:
    logger.warning("Stopping ReSTᴱᴹ after 5 iterations to prevent overfitting")
    break
```

---

## 3. 失败时的退路

| 失败场景 | 退路 |
|---|---|
| **GRPO 不收敛** | 回退到 SFT + RFT（Stage 1-2）|
| **模型生成始终无法通过 XGrammar** | 切换到 Guidance（合规率最高 0.87-1.00）|
| **HydraForge runtime dry-run 失败率高** | 增加训练数据中 expected_output 比例（>30%）|
| **修复数据塌缩** | 用 SCoRe 两阶段 RL（Stage I + Stage II）|
| **PRM 标注偏差大** | 增加 K 数量（从 16 → 32），用更大的 PRM 模型 |
| **多样性崩溃** | 增加 entropy bonus，提升 temperature 探索 |
| **Format compliance 崩溃** | 回滚 checkpoint，提升 KL penalty β |

---

## 4. 具体反 hacking 检查清单

### 4.1 训练前

- [ ] 验证 4 层 reward 独立（每层单独可计算）
- [ ] 设置 reference model（用于 KL penalty）
- [ ] 准备 held-out 200 prompts（用于 spot-check）
- [ ] 设置 format compliance 监控（> 95% 阈值）
- [ ] 设置多样性监控基线

### 4.2 训练中（每 1000 step）

- [ ] 跑 held-out 200 prompts，检查 gold reward 不下降
- [ ] 监控 format compliance（不应 < 95%）
- [ ] 监控 unique DSL patterns（不应单调下降）
- [ ] 记录 token entropy（不应 < 阈值）
- [ ] 检查异常样本（reward 异常高的样本）

### 4.3 训练后

- [ ] 完整 HydraForgeBench 评估
- [ ] 与 base model 对比（确认训练有效）
- [ ] 与 baseline 模型对比（GPT-4、Qwen2.5-Coder）
- [ ] 手动 spot-check 50 个生成的 DSL
- [ ] 检查特殊 token 泄漏（FIM、structural tokens）

### 4.4 上线后

- [ ] 实时监控生产环境 format compliance
- [ ] 实时监控任务成功率
- [ ] 定期（每周）重新评估 HydraForgeBench
- [ ] 跟踪用户反馈中的失败案例

---

## 5. 已知失败模式与缓解

### 5.1 "格式完美但语义空"

**症状**：模型生成格式合规但 `arguments: {}` 空字段的 DSL，骗过 L1 验证。

**缓解**：
- L2/L3 必须有内容检查（不能空 arguments）
- Reward 加权：r_outcome 必须 > 0.1 才算成功
- 训练数据：删除所有空字段样本

### 5.2 "循环套循环"

**症状**：模型生成合法的 DSL，但包含 `loop` 节点无终止条件。

**缓解**：
- HydraForge runtime budget 控制
- 训练数据：所有 `loop` 节点必须有 max_iter 或 condition
- Reward：超时 = r_outcome = 0

### 5.3 "短而优雅"

**症状**：模型倾向于生成最短的 DSL，避免复杂任务。

**缓解**：
- Reward 加权：r_progress 必须达到预期节点数
- LLM-judge 检测"过于简单"
- 训练数据：均衡长度分布

### 5.4 "过度依赖工具"

**症状**：模型把每个任务都用 `tool_call` 处理，不直接 `assign`。

**缓解**：
- 训练数据：包含直接 `assign` 的样本
- Reward：不区分工具调用 vs 直接赋值（避免 bias）
- LLM-judge 检测"工具滥用"

---

## 6. 安全与合规

### 6.1 数据安全

- **训练数据来源**：仅使用 HydraForge 仓库 + 开源数据
- **隐私数据**：不含用户敏感信息
- **版权问题**：所有数据明确归属

### 6.2 模型安全

- **生成内容审查**：runtime dry-run 包含权限检查
- **资源限制**：budget controller 防止滥用
- **权限隔离**：`/lib/**` 不可写，`/dynamic/**` 可选签名

### 6.3 部署安全

- **Canary 部署**：新模型先 10% 流量，验证后再 100%
- **回滚机制**：指标异常立即回滚
- **审计日志**：所有生成 DSL 记录到 Trace

---

## 7. 总结

### 关键防 hacking 协议

1. **4 层 reward**：L1 (format) + L2 (signature) + L3 (execution) + L4 (task)
2. **KL penalty β=0.04**：防 policy drift
3. **格式约束**：XGrammar 强制 schema 合规（不依赖 learned reward）
4. **Human spot-check 周期**：每 1000 step 200 prompts
5. **多样性监控**：unique DSL patterns 不应单调下降
6. **EM iter 限制**：≤5 轮（Singh 2024 警告）

### 关键陷阱（来自 SOTA 经验）

1. **EM iteration 必须从 base 重新开始**（Singh 2024）
2. **修复数据必须用 RL 训练**（SCoRe, ICLR 2025）
3. **Special tokens 必须加入 stop_token_ids**（Qwen issue #99）
4. **格式完美 ≠ 语义正确**（Goodhart 经典陷阱）
5. **多样性崩溃 ≠ 训练收敛**（隐性的 reward hacking）

### 下一步

风险登记就绪后，进入 [`06-vn001-alignment.md`](06-vn001-alignment.md) 学习如何与 VN-001 自举愿景对齐。

---

**文档版本**: v1.0
**Owner**: AgenticMind 训练团队
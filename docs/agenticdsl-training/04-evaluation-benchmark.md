# 评估与基准 — HydraForgeBench 设计

> **文档 ID**: LLMTRN-001-EVAL
> **生成日期**: 2026-06-10
> **关联**:
> - 数据管线: [`01-training-data-pipeline.md`](01-training-data-pipeline.md)
> - 算法: [`02-training-algorithms.md`](02-training-algorithms.md)
> - 推理栈: [`03-inference-time-guarantees.md`](03-inference-time-guarantees.md)
> - 风险: [`05-risk-register.md`](05-risk-register.md)

---

## 1. 评估目标

**核心问题**：我们训练的 LLM 是否能可靠生成、修复、续写、验证 AgenticDSL？

**8 个评估维度**：

| 维度 | 指标 | 目标 | 评估工具 |
|---|---|---|---|
| **格式合规** | parse rate (markdown_parser) | > 99% | Tree-sitter + HydraForge markdown_parser |
| **签名合规** | signature validation pass rate | > 95% | signature_validator |
| **执行成功率** | dry-run success rate (L3) | > 90% | HydraForge runtime |
| **任务成功率** | expected_output match rate | > 85% | expected_output 比对 |
| **多轮稳定性** | 5-step 轨迹成功率（端到端）| > 80% | runtime 全轨迹执行 |
| **预算遵守** | max_nodes / max_llm_calls 不超 | > 98% | budget_controller |
| **修复能力** | (broken, fixed) success rate | > 90% | runtime 验证 |
| **Token 效率** | avg tokens per DSL generation | < 2000 | tokenizer metrics |

---

## 2. HydraForgeBench 基准集设计

### 2.1 三层基准集

| 集合 | 规模 | 用途 | 构建方式 |
|---|---|---|---|
| **Set A — 内部基准** | 50-100 handcrafted | Unit-test 式评估 | 手工编写，覆盖 L1-L7 |
| **Set B — 执行轨迹基准** | 500-1000 | 端到端评估 | 从 HydraForge 自举阶段 0-1 真实运行中收集 |
| **Set C — 对抗性基准** | 100-200 | Stress test | 含 namespace 违规、未声明资源、循环依赖、签名冲突 |

### 2.2 Set A — 内部基准设计

**目的**：标准化、可重复的 unit-test 评估。

**结构**：

```json
{
  "benchmark_id": "A-001",
  "task_type": "nl_to_dsl",
  "complexity_level": "L1",
  "prompt": "创建一个简单的 Agent，将用户输入保存到 state",
  "expected_dsl": "### AgenticDSL '/main/start'...",
  "expected_output": {
    "saved_value": "<user_input>"
  },
  "tags": ["basic", "state_write"],
  "difficulty": "easy"
}
```

**覆盖 L1-L7 全部难度梯度**：

| 难度 | 数量 | 任务类型 |
|---|---|---|
| L1 单节点 | 15 | nl_to_dsl, dsl_validation |
| L2 参数化 | 15 | nl_to_dsl, repair |
| L3 串行组合 | 15 | nl_to_dsl, state_aware |
| L4 并行/分支 | 15 | nl_to_dsl, repair |
| L5 子图嵌套 | 15 | nl_to_dsl, state_aware |
| L6 错误处理 | 10 | repair |
| L7 长时序多轮 | 15 | state_aware |

**Set A 公开**：作为 HydraForge 仓库的一部分，附带 golden answer。

### 2.3 Set B — 执行轨迹基准设计

**目的**：从真实运行中收集，模拟生产场景。

**结构**：

```json
{
  "trajectory_id": "B-001",
  "task_type": "multi_turn_state_aware",
  "initial_prompt": "分析这个数据集",
  "trace": [
    {"turn": 1, "dsl": "...", "result": {...}, "state_snapshot": {...}},
    {"turn": 2, "dsl": "...", "result": {...}, "state_snapshot": {...}},
    ...
  ],
  "final_outcome": "success",
  "source": "production_run_2026_05",
  "complexity_score": 0.7
}
```

**收集流程**：
1. 从 HydraForge 自举阶段 0-1 的真实运行日志中筛选成功轨迹
2. 标注 expected_output 与 oracle DSL
3. 按任务类型、复杂度分层采样
4. 包含"难度无偏"的完整分布（success/failure 混合）

**参考**：AgentBank (Song et al. EMNLP Findings 2024) 51,287 轨迹 / 16 任务 / 5 维度。

### 2.4 Set C — 对抗性基准设计

**目的**：Stress test 模型的边界处理能力。

**对抗类型**：

| 对抗类型 | 期望模型行为 |
|---|---|
| **Namespace 违规**（写入 `/lib/**`） | 拒绝生成，输出 Valid=False |
| **未声明资源**（调用未注册的 tool） | 拒绝或正确处理 |
| **循环依赖**（A → B → A） | 拒绝并指出循环 |
| **签名冲突**（类型不匹配） | 修复或拒绝 |
| **预算超限**（节点数 > max_nodes） | 自动调整 |
| **多入口冲突**（同一 id 多个定义） | 拒绝并指出 |
| **Inja 注入**（恶意 `{{` 边界） | 正确转义 |
| **FIM 边界泄漏**（生成 `<\|fim_*\|>`） | 不泄漏（stop tokens 生效）|

**示例 Set C 题目**：

```json
{
  "benchmark_id": "C-001",
  "task_type": "adversarial",
  "input": "修改 lib/math/add 让它返回负数",
  "expected_behavior": "reject",
  "reason": "Cannot write to /lib/** (read-only namespace)",
  "expected_output": {
    "decision": "reject",
    "error_code": "ERR_NAMESPACE_VIOLATION",
    "suggestion": "Create a new subgraph under /dynamic/** or /main/**"
  }
}
```

---

## 3. 评估流程

### 3.1 评估 Pipeline

```
[测试集: Set A + B + C]
   │
   ▼
[Step 1] 模型生成 DSL
  - vLLM + XGrammar-2 约束
  - max_tokens=2048
  - stop tokens 启用
   │
   ▼
[Step 2] 4 层验证
  - L1: Tree-sitter + markdown_parser
  - L2: signature_validator
  - L3: HydraForge runtime dry-run
  - L4: expected_output 比对
   │
   ▼
[Step 3] 指标计算
  - 各维度单独计算 pass rate
  - 加权综合得分
  - 与 baseline 对比
   │
   ▼
[Step 4] 生成报告
  - pass@1 / pass@5 / pass@10
  - 按 L1-L7 分维度
  - 错误分析（哪些 prompt 失败）
```

### 3.2 评估脚本框架

```python
# evaluate_hydraforge_bench.py
import json
from agenticdsl import RuntimeClient
from inference import VLLMClient

class HydraForgeBenchEvaluator:
    def __init__(self, model_path: str, bench_path: str):
        self.model = VLLMClient(model_path)
        self.runtime = RuntimeClient()
        self.bench = self.load_bench(bench_path)
    
    def load_bench(self, path):
        benchmarks = []
        for line in open(path):
            benchmarks.append(json.loads(line))
        return benchmarks
    
    def evaluate_one(self, item, n_samples=1):
        """对单条 benchmark 评估"""
        results = []
        for _ in range(n_samples):
            # 1. 生成 DSL
            dsl = self.model.generate(
                item["prompt"],
                grammar="agenticdsl_v3_10",
                max_tokens=2048,
            )
            
            # 2. 4 层验证
            l1_pass = self.check_l1(dsl)
            l2_pass = self.check_l2(dsl)
            l3_pass = self.check_l3(dsl) if l1_pass and l2_pass else False
            l4_pass = self.check_l4(dsl, item) if l3_pass else False
            
            results.append({
                "l1": l1_pass,
                "l2": l2_pass,
                "l3": l3_pass,
                "l4": l4_pass,
                "overall": l4_pass,  # 全部通过才算成功
            })
        
        return {
            "pass_at_1": sum(r["overall"] for r in results) >= 1,
            "pass_at_k": sum(r["overall"] for r in results) == n_samples,
            "l1_rate": sum(r["l1"] for r in results) / n_samples,
            "l2_rate": sum(r["l2"] for r in results) / n_samples,
            "l3_rate": sum(r["l3"] for r in results) / n_samples,
            "l4_rate": sum(r["l4"] for r in results) / n_samples,
        }
    
    def check_l1(self, dsl):
        """L1: Tree-sitter + markdown_parser"""
        tree, errors = tree_sitter_parse(dsl)
        return len(errors) == 0
    
    def check_l2(self, dsl):
        """L2: signature_validator"""
        return self.runtime.validate(dsl).signature_pass
    
    def check_l3(self, dsl):
        """L3: dry-run"""
        return self.runtime.dry_run(dsl).success
    
    def check_l4(self, dsl, item):
        """L4: expected_output 比对"""
        result = self.runtime.execute(dsl)
        return self.compare_outputs(result, item.get("expected_output", {}))
    
    def run_full_eval(self):
        """运行完整评估"""
        all_results = []
        for item in self.bench:
            result = self.evaluate_one(item, n_samples=10)
            result["benchmark_id"] = item["benchmark_id"]
            result["task_type"] = item.get("task_type", "unknown")
            result["complexity"] = item.get("complexity_level", "unknown")
            all_results.append(result)
        
        # 汇总统计
        summary = {
            "overall_pass_at_1": sum(r["pass_at_1"] for r in all_results) / len(all_results),
            "overall_pass_at_10": sum(r["pass_at_k"] for r in all_results) / len(all_results),
            "l1_rate": sum(r["l1_rate"] for r in all_results) / len(all_results),
            "l2_rate": sum(r["l2_rate"] for r in all_results) / len(all_results),
            "l3_rate": sum(r["l3_rate"] for r in all_results) / len(all_results),
            "l4_rate": sum(r["l4_rate"] for r in all_results) / len(all_results),
            "by_complexity": self.aggregate_by_complexity(all_results),
            "by_task_type": self.aggregate_by_task_type(all_results),
        }
        
        return summary, all_results
```

### 3.3 报告输出格式

```json
{
  "model": "hydraforge-agenticdsl-7b-v3",
  "benchmark_set": "Set A + B + C (1200 items)",
  "timestamp": "2026-06-10T12:00:00Z",
  "summary": {
    "pass_at_1": 0.85,
    "pass_at_10": 0.92,
    "l1_format_rate": 0.99,
    "l2_signature_rate": 0.97,
    "l3_execution_rate": 0.91,
    "l4_task_rate": 0.85
  },
  "by_complexity": {
    "L1": {"pass_at_1": 0.99, "count": 50},
    "L2": {"pass_at_1": 0.95, "count": 50},
    "L3": {"pass_at_1": 0.90, "count": 50},
    "L4": {"pass_at_1": 0.82, "count": 50},
    "L5": {"pass_at_1": 0.75, "count": 50},
    "L6": {"pass_at_1": 0.68, "count": 50},
    "L7": {"pass_at_1": 0.55, "count": 50}
  },
  "by_task_type": {
    "nl_to_dsl": {"pass_at_1": 0.88, "count": 500},
    "state_aware": {"pass_at_1": 0.72, "count": 300},
    "repair": {"pass_at_1": 0.80, "count": 250},
    "validation": {"pass_at_1": 0.92, "count": 150}
  },
  "failure_analysis": {
    "most_common_errors": [
      {"type": "ERR_NAMESPACE_VIOLATION", "count": 45},
      {"type": "ERR_SIGNATURE_MISMATCH", "count": 30},
      {"type": "ERR_BUDGET_EXCEEDED", "count": 25}
    ],
    "hardest_prompts": ["A-018", "B-103", "C-005"]
  }
}
```

---

## 4. 与业界基准对齐

### 4.1 JSONSchemaBench 对齐

参考 [Geng et al. 2025 (arXiv:2501.10868)](https://arxiv.org/abs/2501.10868)：

**JSONSchemaBench** 是 2025 年最权威的结构化生成基准：
- 10K real-world JSON schemas
- 6 个框架对比
- 2 个核心指标：empirical coverage + compliance rate

**AgenticDSL 评测应参考该方法论**：
- 每个 schema 对应一个 AgenticDSL 生成任务
- Coverage = 模型尝试生成但失败的 schema 比例
- Compliance = 成功生成且通过 schema 校验的比例

### 4.2 BFCL（Berkeley Function Calling Leaderboard）

参考 [Gorilla 论文 (NeurIPS 2024)](https://arxiv.org/abs/2305.15334)：

**BFCL** 是工具调用 SOTA 基准：
- v1 + v2 Live (2.2K 真实场景)
- Multi-step + Multi-turn + Augmented Multi-Turn (800)

**AgenticDSL 与 BFCL 的关系**：
- BFCL 测的是"工具调用 JSON"生成
- AgenticDSL 测的是"完整 workflow DSL"生成（包含工具调用作为节点）
- AgenticDSL 更复杂，但更接近真实 agent 任务

**HydraForgeBench 可作为 BFCL 的超集**，但应包含 BFCL 等价任务以直接对比。

### 4.3 与 ToolACE / xLAM 对比

**ToolACE-8B 在 BFCL-v3 上达到了与 GPT-4 相当的 SOTA**。

**HydraForgeBench 应包含 ToolACE 兼容任务**，以便对比。

---

## 5. 持续评估（Continuous Evaluation）

### 5.1 训练中评估

训练过程中每个 checkpoint 都应评估：

```python
# 在训练循环中插入评估
for step in range(num_steps):
    # ... 训练步骤 ...
    
    if step % 1000 == 0:
        # 在小型 held-out 集上评估
        mini_bench = load_bench("Set A")[:50]  # 快速评估
        summary, _ = evaluator.run_full_eval()
        
        wandb.log({
            "eval/pass_at_1": summary["overall_pass_at_1"],
            "eval/l1_rate": summary["l1_rate"],
            "eval/l2_rate": summary["l2_rate"],
            "eval/l3_rate": summary["l3_execution_rate"],
            "eval/l4_rate": summary["l4_task_rate"],
        })
        
        # 早停：如果 L1 下降（format compliance 崩溃）
        if summary["l1_rate"] < 0.95:
            logger.warning("Format compliance dropped below 95%!")
```

### 5.2 上线后监控

模型部署到 ILLMProvider 后持续监控：

| 指标 | 监控频率 | 告警阈值 |
|---|---|---|
| 格式合规率（生产） | 实时 | < 95% |
| 签名合规率 | 实时 | < 90% |
| 干运行成功率 | 实时 | < 85% |
| 任务成功率 | 每小时 | < 80% |
| Token 效率 | 每小时 | > 2500 avg |
| 预算超限率 | 实时 | > 5% |

---

## 6. A/B 测试基础设施

### 6.1 模型版本对比

```python
# A/B test: v2 vs v3
v2_results = evaluate_model("hydraforge-agenticdsl-7b-v2", bench_path)
v3_results = evaluate_model("hydraforge-agenticdsl-7b-v3", bench_path)

# 显著性检验
from scipy import stats
t_stat, p_value = stats.ttest_rel(
    [r["pass_at_1"] for r in v2_results],
    [r["pass_at_1"] for r in v3_results],
)

print(f"v2 pass@1: {np.mean([r['pass_at_1'] for r in v2_results]):.3f}")
print(f"v3 pass@1: {np.mean([r['pass_at_1'] for r in v3_results]):.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05 and np.mean([r["pass_at_1"] for r in v3_results]) > np.mean([r["pass_at_1"] for r in v2_results]):
    print("✅ v3 显著优于 v2，可上线")
```

### 6.2 推理策略对比

```python
# A/B test: constrained vs free decoding
strategies = {
    "constrained": {"guided_decoding_backend": "xgrammar"},
    "free": {},
    "guidance": {"use_guidance": True},
}

results = {}
for name, kwargs in strategies.items():
    results[name] = evaluate_with_strategy(bench_path, **kwargs)

# 输出对比
for name, res in results.items():
    print(f"{name}: pass@1={res['pass_at_1']:.3f}, "
          f"avg_tokens={res['avg_tokens']:.0f}, "
          f"avg_latency={res['avg_latency_ms']:.0f}ms")
```

---

## 7. 错误分析

### 7.1 失败模式分类

| 失败模式 | 占比（参考值）| 改进方向 |
|---|---|---|
| 格式错误（括号、缩进）| 30% | 训练数据 canonical 化 |
| Namespace 违规 | 20% | 训练数据加 namespace 规则示例 |
| 签名不匹配 | 15% | 训练数据加 schema 注入 |
| 拓扑错误（循环、不可达）| 10% | 训练数据加拓扑验证 |
| 字段缺失 | 10% | 训练数据加必填字段标注 |
| 预算超限 | 5% | 训练数据加预算示例 |
| 其他 | 10% | 个案分析 |

### 7.2 错误分析方法

```python
def error_analysis(eval_results):
    """错误分析报告"""
    error_types = defaultdict(int)
    error_examples = defaultdict(list)
    
    for result in eval_results:
        if not result["overall"]:
            # 收集具体错误
            errors = runtime.collect_errors(result["generated_dsl"])
            for err in errors:
                key = (err["error_type"], err.get("location", "unknown"))
                error_types[key] += 1
                if len(error_examples[key]) < 3:
                    error_examples[key].append({
                        "prompt": result["prompt"],
                        "generated": result["generated_dsl"],
                        "error": err,
                    })
    
    # 输出报告
    return {
        "error_distribution": dict(error_types),
        "error_examples": dict(error_examples),
        "improvement_priorities": sorted(
            error_types.items(), key=lambda x: -x[1]
        )[:10],
    }
```

### 7.3 改进循环

```
[评估失败案例]
    │
    ▼
[聚类错误类型]
    │
    ▼
[针对性扩充训练数据]
    │ - 增加该类型的样本
    │ - 错误示例 + 正确示例对
    │
    ▼
[增量训练]
    │ - 在原模型上继续 SFT
    │ - 不从 base 重新开始（避免灾难性遗忘）
    │
    ▼
[重新评估]
    │
    ▼ pass → 部署
    │ fail → 增加数据 + 循环
```

---

## 8. 与 SOTA 模型对比

### 8.1 对比基线

| 模型 | HydraForgeBench pass@1 | 备注 |
|---|---|---|
| GPT-4 | ~85% (无约束) | 估计值，需实测 |
| GPT-4 + constrained | ~90% | 估计值 |
| Claude-3.5-Sonnet | ~83% | 估计值 |
| Qwen2.5-Coder-7B-Instruct (base) | ~50% | 无 AgenticDSL 训练 |
| **HydraForge-AgenticDSL-7B-v1** | ~70% | SFT 冷启动后 |
| **HydraForge-AgenticDSL-7B-v2** | ~80% | ReSTᴱᴹ 后 |
| **HydraForge-AgenticDSL-7B-v3** | ~90% | GRPO 精调后（目标）|

### 8.2 对比维度

1. **Pass@1**：单次生成成功率
2. **Pass@10**：10 次生成中至少 1 次成功
3. **Token efficiency**：每个任务的平均 token 数
4. **Latency**：P50/P95 延迟
5. **Format compliance**：格式合规率

---

## 9. 总结

### 关键交付

| 维度 | 交付 |
|---|---|
| **8 个评估维度** | format / signature / execution / task / multi-turn / budget / repair / token |
| **3 层基准集** | Set A (unit test) + Set B (production traces) + Set C (adversarial) |
| **评估 Pipeline** | 4 层验证 + 报告生成 |
| **与业界对齐** | JSONSchemaBench + BFCL + ToolACE |
| **持续评估** | 训练中 + 上线后监控 |
| **A/B 测试框架** | 模型版本对比 + 推理策略对比 |

### 与 SOTA 对比的目标

- **Format compliance**: > 99% (超 JSONSchemaBench 89% baseline)
- **Pass@1**: > 85% (与 GPT-4 相当)
- **Token efficiency**: < 2000 tokens per DSL

### 下一步

评估体系就绪后，进入 [`05-risk-register.md`](05-risk-register.md) 学习风险与防 Goodhart 协议。

---

**文档版本**: v1.0
**Owner**: minimind 评估团队
# 训练数据管线 — 9 阶段 SFT 数据构造

> **文档 ID**: LLMTRN-001-DATA
> **生成日期**: 2026-06-10
> **关联**:
> - 综述: [`README.md`](README.md)
> - 算法: [`02-training-algorithms.md`](02-training-algorithms.md)
> - 语言演进: HydraForge 仓 `/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md`
> - HydraForge Runtime: 见 `docs/specs/layer0.md`

---

## 1. 数据任务矩阵与配比

基于 Hermes/ToolACE/AgentInstruct 等 2024-2025 SOTA 项目的配比实践，AgenticDSL SFT 数据按 **30/25/20/15/10** 配比：

| 任务类型 | 占比 | 数据规模 | 输入 | 输出 | 学习目标 |
|---|---|---|---|---|---|
| **NL → DSL 生成** | 30% | 60K | 自然语言需求 + 当前 schema（可用子图/工具清单）| 合法 AgenticDSL | 基础生成、意图对齐 |
| **State-Aware 续写** | 25% | 50K | 已执行 DSL trace + LayeredContext 快照 + 上一步结果 | 下一段 DSL | 多轮循环、状态 grounding |
| **DSL Repair** | 20% | 40K | 含语法/语义错误的 DSL + 运行时错误信息 | 修正后的 DSL + 修复说明 | 鲁棒性、错误恢复 |
| **DSL → NL 解释** | 15% | 30K | 合法 DSL 子图 | 结构化自然语言描述 | 逆向理解、可解释性 |
| **DSL Validation** | 10% | 20K | DSL + schema 约束 | Valid/Invalid + 原因 | 验证能力 |

**总计 ~200K SFT 样本**（对齐 ToolACE 的 11K + 多任务扩展、AgentInstruct 的 25M 缩量版）。

---

## 2. 数据生成管线（9 阶段）

```
[Seed: 200 handcrafted (NL, DSL) + 工具/子图 schema registry]
    │
    ▼
┌─ Stage 1: 多样性引导 (Self-Instruct + Backtranslation) ─┐
│ • Self-Instruct: 200 种子 → 50K (NL, DSL) 对             │
│ • Backtranslation: 从 .agent.md 执行 trace 反推 NL      │
│ • OSS-Instruct-style: 从 GitHub DSL 工作流提取            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Stage 2: 复杂度进化 (Evol-Instruct / Skill-It) ────────┐
│ L1 单工具 → L2 参数化 → L3 串行 → L4 并行/分支         │
│ → L5 子图嵌套 → L6 错误处理 → L7 长时序状态            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Stage 3: 执行驱动过滤 (ToolBench/ToolACE 风格) ──────┐
│ • L1 解析检查: DSL parser AST 严格匹配                   │
│ • L2 静态校验: 签名、namespace、类型                     │
│ • L3 沙箱执行: HydraForge runtime dry-run                │
│ • L4 任务级奖励: expected_output 比对                   │
│ 只保留 pass 全部 4 层的样本                               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Stage 4: 多轮/状态增强 (Hermes/AgentBank 风格) ──────┐
│ • Hermes 模式: state_snapshot 注入中间轮                 │
│ • AgentBank 模式: 完整轨迹（含失败/中途错误）            │
│ • DFSDT 模式: 复杂路径的程序化生成                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Stage 5: 修复数据生成 (Constitutional AI 风格) ──────┐
│ • 主动扰动: 删除/换名/改类型 → 收集错误信息              │
│ • 三元组 (broken_dsl, error_msg, fixed_dsl)             │
│ • 错误类型均衡采样                                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Stage 6: Critic/PRM 数据 (ThinkPRM/CodePRM 风格) ────┐
│ • 教模型对每个节点写 "verification CoT"                  │
│ • 输入加 execution feedback, MSE/step label 训练         │
│ • 用 consensus filtering 保留 51% 样本                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Stage 7: 质量控制 ─────────────────────────────────────┐
│ • SemDeDup: 按 DSL embedding 去重（k=11K clusters）      │
│ • DSIR: proxy-model importance 重采样                    │
│ • n-gram decontamination vs HydraForge 测试集            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Stage 8: 课程学习排序 (Skill-It 风格) ───────────────┐
│ • L1→L7 顺序训练                                        │
│ • online reweighting: 已掌握技能少采                   │
│ • 与课程难度匹配的 sample packing                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
[最终 SFT 数据集: ~200K 高质量样本, 课程就绪]
```

---

## 3. 课程学习的 7 层难度梯度

参考 Skill-It (NeurIPS 2023 Spotlight) 在 LEGO synthetic continual pretraining 上 **+36.5 accuracy**（比 random sampling）的结果，为 AgenticDSL 设计 7 层课程：

| 层级 | 复杂度 | 节点类型 | 示例 |
|---|---|---|---|
| **L1** | 1 个简单节点 | `start`/`end`/`assign` | 单变量赋值 |
| **L2** | 参数化节点 | `tool_call`（1 个参数）| 调用 `fs.read` |
| **L3** | 串行组合 | `start → tool_call → tool_call → end` | 读取文件 → 解析 → 输出 |
| **L4** | 并行/分支 | `fork` / `join` / `assert` | 并行搜索多个数据源 |
| **L5** | 子图嵌套 | `dsl_call` 引用 `/lib/**` | 调用推理标准库子图 |
| **L6** | 错误处理 | `on_failure` / `try_catch`（实验性） | 自动回滚与重试 |
| **L7** | 长时序多轮 | `generate_subgraph` + 多轮 trace | 跨 10+ 步的 stateful 推理 |

**训练顺序**：L1 (50%) + L2 (20%) + L3 (15%) → L4 (8%) + L5 (4%) → L6 (2%) + L7 (1%)，逐步过渡。

**Skill-It 实施细节**：
- **Prerequisite graph**：先修关系（如 L5 需先掌握 L3、L4）
- **Online data sampling**：mirror-descent 风格动态调整每层采样权重
- **已掌握技能少采，影响或前修的多采**

---

## 4. HydraForge Runtime 作为 4 层验证器

HydraForge 的 C++ 引擎天然提供**多层验证器**，作为 SFT 数据的硬过滤：

```
DSL 输出
   │
   ▼
[L1: 语法解析] markdown_parser → ParsedGraph
   │ < 10ms, 无外部依赖
   │ - 检查 ### AgenticDSL 头
   │ - 解析 YAML block
   │ - 提取 nodes / signature / permissions
   │
   ▼ pass
[L2: Schema 校验] signature_validator
   │ - namespace 规则（/lib/** 不可写）
   │ - 节点引用合法性
   │ - 必填字段完整性
   │ - permission 交集
   │
   ▼ pass
[L3: 沙箱执行] dry_run executor
   │ - max_nodes / max_llm_calls 预算检查
   │ - 工具签名匹配（实际调用 mock tool）
   │ - 状态合并策略验证
   │ - 终止条件可达性
   │
   ▼ pass
[L4: 任务级奖励] expected_output 比对
   │ - 与 expected_output 字段比对
   │ - LLM-as-judge（仅 BoN rerank 时使用，不进 RL reward）
   │
   ▼
final reward = 0.4 * L1 + 0.2 * L2 + 0.3 * L3 + 0.1 * L4
```

**4 层权重说明**：
- **L1 (0.4) 最重**：格式合规是硬门槛
- **L3 (0.3) 次重**：沙箱执行是"可运行性"证明
- **L2 (0.2)**：静态校验是 L3 的前置
- **L4 (0.1)**：任务级奖励最贵，仅作软信号

---

## 5. 各阶段详细实施

### Stage 1: 多样性引导（Self-Instruct + Backtranslation）

**Self-Instruct（Wang et al. 2022）**：
- 200 个手工编写的 "DSL 模板 + 工具清单" 作为种子
- LLM 自举生成指令（NL）
- ROUGE-L 过滤低多样性（cosine > 0.7 视为重复）
- 迭代直到 50K (NL, DSL) 对
- 代码参考: [github.com/yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)

**Backtranslation（Li et al. 2023）**：
- 收集 DSL 程序语料（执行 trace、GitHub workflow、用户日志）
- 反向模型 p(x|y) 生成 NL 指令 x̂
- 前向模型打分（GPT-4 作为 judge）
- 只保留 5 分样本（高置信度）
- 代码参考: [aclanthology.org/2024.findings-emnlp.777](https://aclanthology.org/2024.findings-emnlp.777.pdf)

**OSS-Instruct-style**：
- 从 GitHub 公开的 AgenticDSL `.agent.md` 文件提取
- 用 OSS-Instruct 的启发式生成（Magicoder OSS-Instruct 经验）

### Stage 2: 复杂度进化（Evol-Instruct）

**WizardLM Evol-Instruct (Xu et al. 2023)**：
- In-depth evolving：加约束、深化、抽象
- In-breadth evolving：领域扩展
- 多轮迭代产出 easy→hard 样本链

**升级版（EMNLP2024 / ACL2025）**：
- Auto Evol-Instruct：用 LLM 自动发现并优化进化策略
- Tag-Evol：用 InsTag 细粒度标签作为进化策略

**AgenticDSL 深度进化路径**：
- L1（单工具）→ L2（多参数）→ L3（串行组合）→ L4（并行/分支）→ L5（嵌套）→ L6（错误处理）→ L7（长时序）

**广度进化**：
- 覆盖不同领域 schema（数据处理、推理、记忆、对话等）
- 不同任务类型（代码生成、文本摘要、决策制定等）

### Stage 3: 执行驱动过滤（ToolBench/ToolACE 风格）

**ToolLLM (Qin et al. 2023) 三阶段管线**：
1. API 收集：RapidAPI 抓取 + 健康检查过滤
2. 指令生成：按 I1/I2/I3 模板
3. 解答标注：**DFSDT**（Depth-First Search Decision Tree）

**ToolACE (Liu et al. 2024) 双层校验**：
- Rule-based：schema、依赖图、并行冲突
- Model-based：LLM-as-judge 评估

**AgenticDSL 应用**：
- API pool 替换为"tool/subgraph 签名池"
- Complexity evaluator 改为"DSL 节点数 / 嵌套深度"
- 双层校验改为"DSL parser + dry-run executor"

**关键参数**（参考 ToolACE）:
- API pool 大小：~500 个工具签名 + 100 个 `/lib/**` 子图
- 数据规模：11K 高质量样本（ToolACE 实测这个量级即可 BFCL SOTA）

### Stage 4: 多轮/状态增强（Hermes 风格）

**Hermes 2/3 (NousResearch)**：
- 5 个子集：func-calling-singleturn、func-calling、glaive-function-calling-5k、json-mode-agentic、json-mode-singleturn
- 格式：`<tool_call>{...}</tool_call>` + `<tool_response>...</tool_response>`
- Hermes3 总量: ~390M tokens (120M in + 270M out)
- 代码: [HF: hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)

**AgenticDSL 应用**：
- `<tool_call><agenticdsl_program></tool_call>`
- `<state_snapshot>{json}</state_snapshot>`（LayeredContext 摘要）

**AgentBank (Song et al. EMNLP Findings 2024)**：
- 51,287 轨迹 / 16 任务 / 5 维度
- 平均交互 3.9 turn，范围 1-30 turn
- 含失败/中途错误轨迹（不像前作只用成功）
- 代码: [arXiv:2410.07706](https://arxiv.org/abs/2410.07706)

**AgenticDSL 应用**：
- 构造 (state_t, action, state_t+1, reward) 轨迹
- 包含中途失败的轨迹（让模型学会恢复）

### Stage 5: 修复数据生成（Constitutional AI 风格）

**Constitutional AI (Bai et al. 2022)**：
- 两阶段：(1) SL 模型按"宪法"原则自批评 + 自修正 → 训练集含 (prompt, initial, critique, revised)
- d-RLAIF 创新：跳过 RM 训练，直接用 LLM 在线打分

**AgenticDSL "宪法"**：
- DSL 形式化约束
- 工具签名 schema
- Namespace 规则
- 预算遵守

**主动扰动类型**：
| 扰动类型 | 错误类型 | 示例 |
|---|---|---|
| 删除必填字段 | ERR_MISSING_FIELD | 删除 `type` |
| 篡改节点 ID 引用 | ERR_UNDEFINED_REFERENCE | `next: /foo` 但不存在 |
| 破坏拓扑排序 | ERR_CYCLE | A → B → A |
| 插入非法边 | ERR_NAMESPACE_VIOLATION | 写入 `/lib/**` |
| 括号/标签不匹配 | ERR_SYNTAX | `--- BEGIN AgenticDSL ---` 缺 `--- END ---` |
| 类型不匹配 | ERR_TYPE_MISMATCH | `arguments: { temp: "abc" }` 但工具期望 int |
| 预算超限 | ERR_BUDGET_EXCEEDED | 节点数 > max_nodes |

**错误类型均衡采样**：
- 每种错误类型至少 5K 样本
- 防止模型只学会修某一种 bug

### Stage 6: Critic/PRM 数据（ThinkPRM/CodePRM 风格）

**ThinkPRM (Khalifa et al. 2025)**：
- 1K synthetic verification CoTs (QwQ-32B-Preview 生成)
- 8K PRM800K step labels 过滤
- 训练 4.5 小时单 A100 80GB
- 用 1% PRM800K 标签即超过 full PRM800K-trained
- 代码: [github.com/mukhal/thinkprm](https://github.com/mukhal/thinkprm)

**AgenticDSL 应用**：
- 教模型对每个 DSL 节点写"verification CoT"：
 ```text
 Step 1: 调用 fs.read
 Expected: 返回文件内容
 Actual: 错误 - 文件不存在
 Verdict: 错误（exit code 1）
 ```
- 输入加 execution feedback
- MSE/step label 训练

**GenPRM (Zhao et al. AAAI 2026)**：
- 三步数据合成：(1) QwQ-32B 生成 CoT + Python 验证脚本 → (2) 执行反馈 → (3) consensus filtering（~51% 保留率）
- 23K 训练样本 from MATH

**AgenticDSL 应用**：
- 用 DSL interpreter 替换 Python verifier
- CoT 生成："这一步调用 fs.read 后应返回文件内容，若返回 'Permission denied' 则失败"

---

## 6. 课程学习排序（Skill-It 实施）

**核心算法**（参考 Skill-It）：

```python
# Skill-It 风格 online data sampling
def sample_batch(curriculum_state, batch_size):
    weights = []
    for layer in L1..L7:
        # 已掌握技能少采
        mastery = curriculum_state.mastery[layer]
        # 前修技能多采
        prerequisite_demand = sum(
            curriculum_state.demand[downstream]
            for downstream in get_downstream(layer)
        )
        weight = (1 - mastery) * prerequisite_demand
        weights.append(weight)
    weights = normalize(weights)
    return weighted_sample(layers, weights, batch_size)
```

**Prerequisite Graph**（AgenticDSL）：
```
L1 → L2 → L3 → L4 → L5 → L6 → L7
              ↓     ↓     ↓     ↓
              └─────┴─────┴─────┘ (互相依赖)
```

**Sample Packing**：
- 8192 ctx, FA2, ~96% packing efficiency（Hermes 经验）
- 按课程难度匹配的 packing（不同 L 层用不同 ctx 长度）

---

## 7. 数据质量保障

### 7.1 三大反直觉结论

**反直觉结论 1**（SCoRe, ICLR2025）：**离线 SFT on (broken, fixed) 对会塌缩**。
- 修复数据必须用 RL（GRPO/IPP/KTO）而非纯 SFT。
- 但 Stage 5 生成的修复数据仍可作为 SFT 冷启动；RL 精调在 Stage 2 算法层处理。
- 参考: [openreview.net/pdf?id=CjwERcAU7w](https://openreview.net/pdf?id=CjwERcAU7w)

**反直觉结论 2**（ToolACE）：**质量 > 数量**。
- 26K API + 双层校验 > 100K 浅层数据。
- AgenticDSL 应严格通过 4 层验证，不允许噪声数据通过。

**反直觉结论 3**（Tree-of-Evolution ACL2025）：**树状多路径进化 > 单链单向**。
- 75K 树状进化数据匹敌百万级 Evol-Instruct。
- Stage 2 应采用树状多路径进化而非线性难度递增。

**反直觉结论 4**（MAGPIE）：**指令模型自举生成** 比人工 Evol-Instruct 更高效。
- 3M MAGPIE-Pro > 143K Evol-Instruct（AlpacaEval2 +30%）。
- AgenticDSL 可用 GPT-4/Claude 自举生成高质量 NL。

**反直觉结论 5**（Self-Refine in fair setting, TACL2024）：**当 initial response 已是 best-possible 时，自纠错提升很小**。
- SFT 时不要把"模型的第一反应"训得过于自信。

### 7.2 必备防线

| 防线 | 方法 | 拦截目标 |
|---|---|---|
| L1 | Tree-sitter + markdown_parser AST 严格匹配 | 括号不匹配、非法 Token、缩进错误 |
| L2 | signature_validator + 静态分析 | 未定义变量引用、类型不匹配、环检测 |
| L3 | 沙箱执行（HydraForge runtime headless mode） | 运行时崩溃、死循环、输出不符合预期 |
| L4 | expected_output 比对 + LLM judge | 任务级语义合规 |

**反污染**：
- n-gram (8-gram) 去污 vs HydraForge 内部测试集与现有 `lib/`、`examples/` 工作流
- SemDeDup 按 DSL embedding 聚类去重（k=11K clusters，-50% 数据无精度损失）

### 7.3 SemDeDup 实施细节

参考 [facebookresearch/SemDeDup](https://github.com/facebookresearch/SemDeDup)：
- Embed DSL（用 sentence-transformers）
- K-means (k=11K)
- Cluster 内 pairwise cosine
- 阈值以上视为语义重复（threshold=0.85）
- 仅保留 cluster center 附近样本

### 7.4 DSIR 重要性重采样

参考 [DSIR paper](https://arxiv.org/abs/2302.03169)：
- 用小 proxy model 在 target domain（AgenticDSL）数据上训练
- 估出 importance weights
- 重采样源域数据

**AgenticDSL 应用**：
- Proxy model: 训练一个小的 DSL execution success predictor
- Target domain: HydraForge runtime 验证通过的 DSL
- 重采样 SFT 数据，使其分布对齐 HydraForge 真实使用场景

---

## 8. 工具与代码

### 8.1 HydraForge Runtime 暴露为 CLI

为支持数据生成管线，需要把 HydraForge C++ 引擎暴露为 CLI 子命令：

```bash
# 语法/签名校验
agenticdsl validate <file.agent.md>
# 输出: { "pass": true, "layer1": {...}, "layer2": {...} }

# 沙箱执行（dry-run）
agenticdsl dry-run <file.agent.md> --max-nodes=10 --max-llm-calls=1
# 输出: { "pass": true, "trace": [...], "final_state": {...} }

# 任务级评估
agenticdsl eval <file.agent.md> --task=<task_id> --expected=<expected.json>
# 输出: { "pass": true, "reward": 0.85, "details": {...} }

# 完整 trace 输出（用于 State-Aware 训练数据）
agenticdsl trace <file.agent.md> --format=jsonl
# 输出: 每行一个 node execution trace
```

**实施位置**：
- `tools/cli/validate.cpp`
- `tools/cli/dry_run.cpp`
- `tools/cli/eval.cpp`
- `tools/cli/trace.cpp`

### 8.2 Python 数据管线框架

```python
# agenticdsl_data_pipeline.py
from agenticdsl import RuntimeClient, SchemaRegistry

runtime = RuntimeClient(endpoint="http://localhost:8080")
registry = SchemaRegistry.from_standard_library(
    "/workspace/project/HydraForge/lib/"
)

# Stage 3: 执行驱动过滤
def filter_pass_4_layers(generated_dsl: str) -> bool:
    result = runtime.validate(generated_dsl)
    if not result.layer1_pass:
        return False
    if not result.layer2_pass:
        return False
    if not runtime.dry_run(generated_dsl).success:
        return False
    if not runtime.eval(generated_dsl).passes:
        return False
    return True

# Stage 4: 多轮轨迹生成
def generate_state_aware_trajectory(prompt: str, max_turns: int):
    trajectory = []
    state = initial_state
    for turn in range(max_turns):
        dsl = model.generate(prompt + state_snapshot(state))
        result = runtime.execute(dsl, state)
        trajectory.append({
            "state_before": state,
            "dsl": dsl,
            "result": result,
            "state_after": result.state,
        })
        state = result.state
        if result.is_terminal:
            break
    return trajectory

# Stage 7: 质量控制
from semdedup import SemDeDup

dedup = SemDeDup(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
deduplicated_data = dedup.fit_transform(sft_data, threshold=0.85)
```

### 8.3 开源工具依赖

| 工具 | 用途 | URL |
|---|---|---|
| **Self-Instruct** | 种子扩展 | [github.com/yizhongw/self-instruct](https://github.com/yizhongw/self-instruct) |
| **ToolBench pipeline** | 多轮轨迹 | [github.com/openbmb/toolbench](https://github.com/openbmb/toolbench) |
| **Gorilla AST eval** | API schema 校验 | [github.com/gorilla-llm/gorilla](https://github.com/gorilla-llm/gorilla) |
| **Hermes function-calling** | 多轮格式 | [HF: hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) |
| **ToolACE** | 双层校验 | [HF: Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) |
| **AgentBank** | 多轮轨迹 | [HF: Solaris99/AgentBank](https://huggingface.co/datasets/Solaris99/AgentBank) |
| **AgentInstruct1M** | 大规模合成 | [HF: microsoft/orca-agentinstruct-1M-v1](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) |
| **Skill-It** | 课程采样 | [github.com/HazyResearch/skill-it](https://github.com/HazyResearch/skill-it) |
| **SemDeDup** | 语义去重 | [github.com/facebookresearch/SemDeDup](https://github.com/facebookresearch/SemDeDup) |
| **ThinkPRM** | PRM 数据合成 | [github.com/mukhal/thinkprm](https://github.com/mukhal/thinkprm) |
| **GenPRM** | PRM + 执行验证 | [github.com/RyanLiu112/GenPRM](https://github.com/RyanLiu112/GenPRM) |

---

## 9. 数据规模与资源估算

### 9.1 SFT 数据规模

| 任务 | 样本数 | Token 数 | GPU-hour |
|---|---|---|---|
| NL → DSL | 60K | ~30M | ~100 |
| State-Aware | 50K | ~50M | ~150 |
| Repair | 40K | ~20M | ~80 |
| DSL → NL | 30K | ~20M | ~80 |
| Validation | 20K | ~10M | ~40 |
| **总计** | **200K** | **~130M** | **~450 GPU-hours** |

### 9.2 多轮轨迹数据

| 数据 | 规模 | Token 数 | GPU-hour |
|---|---|---|---|
| State-aware 轨迹 (50K 条 × 平均 5 turn) | 50K | ~100M | ~300 |
| Failure 轨迹 (5K 条) | 5K | ~10M | ~30 |
| **总计** | **55K** | **~110M** | **~330 GPU-hours** |

### 9.3 验证与过滤资源

| 阶段 | 工具调用次数 | 时间 |
|---|---|---|
| Stage 3 dry-run (200K × 1) | 200K | ~50 GPU-hours (沙箱执行) |
| Stage 5 主动扰动 (40K × 1) | 40K | ~10 GPU-hours |
| Stage 7 SemDeDup (200K embed) | 200K | ~2 GPU-hours |
| **总计** | | **~62 GPU-hours** |

---

## 10. 总结

### 关键交付

| 维度 | 交付 | SOTA 依据 |
|---|---|---|
| **数据配比** | 30/25/20/15/10（200K SFT）| Hermes3 / ToolACE |
| **9 阶段管线** | Self-Instruct → Evol → 执行过滤 → 多轮 → 修复 → PRM → 质控 → 课程 → SFT | Hermes / ToolBench / Constitutional AI |
| **7 层课程** | L1-L7 难度梯度（Skill-It 风格）| Skill-It (NeurIPS 2023) |
| **4 层验证器** | HydraForge runtime 作为硬过滤 | HydraForge 现有能力 |
| **修复数据** | 7 类扰动类型，均衡采样 | Constitutional AI / SCoRe |
| **PRM 数据** | ThinkPRM 范式，1K 数据高效 | ThinkPRM / GenPRM |

### 关键反直觉结论

1. **修复数据 SFT 会塌缩**（SCoRe）：必须用 RL
2. **质量 > 数量**（ToolACE）：26K > 100K
3. **树状进化 > 线性进化**（Tree-of-Evolution）：75K 匹敌百万级
4. **指令模型自举 > 人工 Evol**（MAGPIE）：3M > 143K
5. **Self-Refine 在 fair setting 下效果有限**（TACL2024）：不要训得过于自信

### 下一步

数据管线就绪后，进入 [`02-training-algorithms.md`](02-training-algorithms.md) 学习如何用这些数据训练模型。

---

**文档版本**: v1.0
**Owner**: AgenticMind 训练团队
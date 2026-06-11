# 与 VN-001 自举愿景的对齐路径

> **文档 ID**: LLMTRN-001-ALIGN
> **生成日期**: 2026-06-10
> **关联**:
> - 综述: [`README.md`](README.md)
> - 训练算法: [`02-training-algorithms.md`](02-training-algorithms.md)
> - 推理栈: [`03-inference-time-guarantees.md`](03-inference-time-guarantees.md)
> - HydraForge VN-001: [`docs/agenticdsl/vision/01-self-bootstrapping-vision.md`](https://github.com/chisuhua/HydraForge/blob/main/docs/agenticdsl/vision/01-self-bootstrapping-vision.md)
> - HydraForge BOOT-001: [`docs/agenticdsl/implementation/self-bootstrapping-path.md`](https://github.com/chisuhua/HydraForge/blob/main/docs/agenticdsl/implementation/self-bootstrapping-path.md)

---

## 1. VN-001 自举愿景回顾

参考 HydraForge 仓 `docs/agenticdsl/vision/01-self-bootstrapping-vision.md`：

### 1.1 自举定义

**自举**：通过 AgenticDSL 驱动推理计算图，驱动推理输出，而推理的输出内容的质量可以持续驱动 AgenticDSL 运行时工作。

**类比 Forth Bootstrap**：
```
Forth (1960s):
  机器码 → Forth解释器(42字节) → Forth编译器 → 完整OS

AgenticDSL (2020s):
  硬编码参数 → AgenticDSL运行时 → 可编程推理策略 → 持续自进化
                                  ↓
                            Oracle监控 → 生成新Skill → 更优系统
                                  ↓
                             持续进化 → 超越传统语言
```

### 1.2 四阶段自举链路

**阶段 0：硬编码参数（当前状态）**
- 推理通过 ILLMProvider 接口调用
- 推理参数在 C++ 代码或配置文件中硬编码
- Agent 工作流可以调用推理，但不能动态调整推理策略
- **云端 LLM 作为老师模型**：提供高质量推理输出

**阶段 1：可编程推理策略 + 云端集成**
- 推理标准库暴露策略控制面：`engine.md`、`model.md`、`session.md`、`kv_cache.md` 等
- Agent 根据 workload 特征选择最优策略组合
- 调用路径：云端 LLM 为主，本地 llama.cpp 为辅助/降级

**阶段 2：Agent 编排推理 + 质量评估闭环**
- Agent 工作流决定"如何推理"
- 质量评估节点（assert + on_failure）评估输出质量
- 服务分层：高质量任务 → 云端，低质量 → 本地

**阶段 3：持续自进化 + 服务化**
- Oracle 监控循环持续运行
- 发现性能瓶颈时自动生成优化 Skill
- 系统提供推理 API 服务（MCP + OpenAI 兼容）
- 系统持续自优化，无需人为干预

---

## 2. 训练路线对自举阶段的支撑

### 2.1 阶段映射

| 训练阶段 | 时间 | 对应 VN-001 阶段 | 关键交付 |
|---|---|---|---|
| **TR-1** 基础生成能力 | 4-6 周 | **阶段 0**（云端集成）| 训练好的 LLM 可作 ILLMProvider 默认后端 |
| **TR-2** 多轮与修复 | 4-6 周 | **阶段 1**（可编程推理策略）| LLM 可基于 trace 续写，配合生成新 Skill |
| **TR-3** 质量闭环 | 6-8 周 | **阶段 2**（Agent 编排推理）| LLM 在 quality_eval + 反馈循环中达到 SOTA |

**关键定位**：训练后的 LLM **不是替代** GPT-4/Claude，而是：
- **本地降级方案**：网络不可用时仍可工作
- **专用模型**：在 AgenticDSL 任务上优于通用大模型
- **自举基础**：阶段 3（持续自进化）需要专用模型快速迭代

### 2.2 与 BOOT-001 阶段 0 的对齐

参考 BOOT-001 任务 0.1（云端 LLM 适配器）：

**当前状态（C1 后 2026-06-08）**：
```
AgenticDSL → ILLMProvider
                │
                ├──→ MockLLMProvider           (测试桩)
                ├──→ LlamaAdapterProvider → LlamaAdapter → llama.cpp
                └──→ CloudLLMAdapter → HTTP → OpenAI / Anthropic
```

**加入训练后的 LLM**：
```
AgenticDSL → ILLMProvider
                │
                ├──→ MockLLMProvider
                ├──→ LlamaAdapterProvider → LlamaAdapter → llama.cpp
                │                              │
                │                              └──→ [TR-1 后] HydraForge-AgenticDSL-7B GGUF
                └──→ CloudLLMAdapter → OpenAI / Anthropic
```

**集成步骤**：
1. 训练后的 GGUF 模型放入 `models/` 目录
2. 修改 `llm_config.json`，添加 `hydraforge_agenticdsl_7b` provider
3. 在 `src/common/llm/` 实现 `HydraForgeAgenticDSLProvider`，wrap LlamaAdapter + XGrammar constraints
4. DSL 中 `dsl_call` 节点可通过 `llm_tool_name: "hydraforge_agenticdsl_7b"` 调用

### 2.3 与 BOOT-001 阶段 1 的对齐

参考 BOOT-001 任务 1.1-1.2（质量评估 + 服务分层）：

**TR-2 训练完成后**：
- LLM 可基于执行 trace 续写，配合 `/lib/reasoning/` 子图
- `quality_eval.md` 子图调用训练后的 LLM 进行深度质量评估
- 服务分层路由 `router.md` 使用训练后 LLM 作为本地降级选项

**新增子图**：
- `/lib/reasoning/agenticdsl_generate@v1`：调用训练后的 LLM 生成 DSL
- `/lib/reasoning/agenticdsl_repair@v1`：调用训练后的 LLM 修复 DSL
- `/lib/reasoning/agenticdsl_validate@v1`：调用训练后的 LLM 验证 DSL

### 2.4 与 BOOT-001 阶段 2 的对齐

参考 BOOT-001 任务 2.1-2.2（质量反馈 + 自适应优化）：

**TR-3 训练完成后**：
- LLM 在 quality_eval + 反馈循环中达到 SOTA
- `QualityFeedbackController` 收集 LLM 生成结果的质量数据
- `adaptive_optimize.agent.md` 工作流使用训练后的 LLM 选择最优策略

**自举闭环**：
```
LLM 生成 DSL → HydraForge runtime 执行 → Trace 收集
                                        ↓
                              训练数据生成（自动）
                                        ↓
                          增量训练 LLM（每周）
                                        ↓
                          更新 HydraForgeAgenticDSLProvider
                                        ↓
                            下一轮循环开始
```

---

## 3. 与 ILLMProvider 的集成架构

### 3.1 当前架构（C1 后 2026-06-08）

```cpp
// src/common/llm/llm_types.h
class ILLMProvider {
public:
    virtual ~ILLMProvider() = default;
    virtual Result<GenerationResult, LLMError>
        generate(const GenerationRequest& req, std::stop_token token) = 0;
    virtual std::unique_ptr<IGenerationStream>
        generate_stream(const GenerationRequest& req, std::stop_token token) = 0;
};
```

### 3.2 新增 HydraForgeAgenticDSLProvider

```cpp
// src/common/llm/hydraforge_agenticdsl_provider.h
class HydraForgeAgenticDSLProvider : public ILLMProvider {
public:
    HydraForgeAgenticDSLProvider(
        std::unique_ptr<LlamaAdapter> adapter,
        std::shared_ptr<XGrammarConstraint> grammar_constraint
    );
    
    Result<GenerationResult, LLMError> generate(
        const GenerationRequest& req,
        std::stop_token token = {}
    ) override;
    
    std::unique_ptr<IGenerationStream> generate_stream(
        const GenerationRequest& req,
        std::stop_token token = {}
    ) override;
    
private:
    std::unique_ptr<LlamaAdapter> adapter_;
    std::shared_ptr<XGrammarConstraint> grammar_;  // AgenticDSL EBNF
    std::vector<int> stop_token_ids_;  // FIM + structural tokens
};
```

### 3.3 DSL 中使用训练后的 LLM

```yaml
### AgenticDSL '/main/generate_workflow'
```yaml
# --- BEGIN AgenticDSL ---
graph_type: subgraph
signature: "(requirement: string, context: object) -> (dsl: string, confidence: float)"
permissions:
  - tool: hydraforge_agenticdsl.generate

nodes:
  - id: prepare_prompt
    type: assign
    assign:
      prompt: "{{ inputs.requirement }}\n\nContext: {{ inputs.context }}"
    next: ["/main/generate_workflow/call_llm"]

  - id: call_llm
    type: dsl_call
    llm_tool_name: "hydraforge_agenticdsl_7b"
    prompt_template: "{{ prompt }}"
    llm_params:
      temperature: 0.7
      max_tokens: 2048
    output_keys: ["dsl_text"]
    next: ["/main/generate_workflow/parse"]

  - id: parse
    type: tool_call
    tool: markdown_parser
    arguments:
      source: "{{ dsl_text }}"
    output_keys: ["parsed_graph"]
    next: ["/main/generate_workflow/validate"]

  - id: validate
    type: assert
    condition: "{{ parsed_graph.is_valid }}"
    on_failure: "/main/generate_workflow/handle_error"
    next: ["/main/generate_workflow/success"]

  - id: handle_error
    type: assign
    assign:
      dsl: ""
      confidence: 0.0
    output_keys: ["dsl", "confidence"]
    next: ["/end_soft"]

  - id: success
    type: assign
    assign:
      dsl: "{{ dsl_text }}"
      confidence: 0.95
    output_keys: ["dsl", "confidence"]
    next: ["/end_soft"]
# --- END AgenticDSL ---
```
```

---

## 4. 服务化路径（与 BOOT-001 阶段 3 对齐）

### 4.1 InferenceServer 集成

参考 BOOT-001 任务 3.1（推理 API 服务）：

```cpp
// src/api/inference_server.h
class InferenceServer {
public:
    struct Config {
        std::string host = "0.0.0.0";
        int port = 8080;
        bool enable_mcp = true;
        bool enable_openai_compatible = true;
        
        // 新增：训练后的模型 provider
        bool enable_agenticdsl_local = true;
        std::string local_model_path = "models/HydraForge-AgenticDSL-7B.gguf";
    };
    
    bool start(const Config& config);
    void stop();
    
    // MCP 接口
    void handle_mcp_request(const json& request, json& response);
    
    // OpenAI 兼容接口
    void handle_chat_completions(const json& request, json& response);
    
private:
    std::unique_ptr<LLMRouter> router_;
    std::unique_ptr<HydraForgeAgenticDSLProvider> local_agenticdsl_provider_;
    std::unique_ptr<QualityFeedbackController> feedback_controller_;
};
```

### 4.2 路由策略

```
[请求: chat completion 或 tool call]
   │
   ▼
[LLMRouter.route(task_profile)]
   │
   ├── 高质量 + 隐私不敏感 → CloudLLMAdapter (OpenAI/Anthropic)
   ├── 高质量 + 隐私敏感 → HydraForgeAgenticDSLProvider (本地)
   └── 低质量 + 低延迟 → MockLLMProvider (CI) 或其他
```

**Quality Tier**：

| Tier | quality_requirement | 路由 |
|---|---|---|
| `high` | > 0.95 | CloudLLMAdapter（首选）或 HydraForgeAgenticDSLProvider（降级）|
| `medium` | 0.8-0.95 | HydraForgeAgenticDSLProvider（首选）或 CloudLLMAdapter（兜底）|
| `low` | < 0.8 | MockLLMProvider / 模板填充 |

### 4.3 MCP 接口示例

```python
# MCP request: 客户端调用
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "agenticdsl_generate",
        "arguments": {
            "requirement": "创建一个能搜索并总结的 Agent",
            "context": {"available_tools": ["web_search", "fs.read"]}
        }
    }
}

# MCP response: 服务端返回
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "### AgenticDSL '/main/start'\n```yaml\n..."
            }
        ],
        "metadata": {
            "backend": "hydraforge_agenticdsl_7b",
            "quality_score": 0.92,
            "tokens_used": 845
        }
    }
}
```

---

## 5. 自举闭环设计

### 5.1 闭环结构

```
┌─────────────────────────────────────────────────────────────┐
│                   HydraForge 自举闭环                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1] 用户请求                                               │
│       ↓                                                     │
│  [2] LLM 生成 AgenticDSL（使用训练后的 GGUF）                │
│       ↓                                                     │
│  [3] HydraForge runtime 执行 DSL                            │
│       ↓                                                     │
│  [4] Trace 收集（OpenTelemetry 兼容）                        │
│       ↓                                                     │
│  [5] 质量评估（4 层验证器）                                  │
│       ↓                                                     │
│  [6a] 成功 → 更新 QualityFeedbackController                 │
│       ↓                                                     │
│  [6b] 失败 → 收集 (prompt, failed_dsl, error, trace)        │
│       ↓                                                     │
│  [7] 训练数据自动生成（每周）                                │
│       ↓                                                     │
│  [8] 增量训练 LLM                                           │
│       ↓                                                     │
│  [9] 更新 HydraForgeAgenticDSLProvider GGUF                 │
│       ↓                                                     │
│  [10] Canary 部署 → 全量上线                                 │
│       ↓                                                     │
│  [返回 1]                                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 数据生成自动化

```python
# auto_training_data_generator.py
class AutoTrainingDataGenerator:
    def __init__(self, runtime_endpoint, feedback_store):
        self.runtime = runtime_endpoint
        self.feedback = feedback_store
    
    def generate_daily(self):
        """每日生成训练数据"""
        # 1. 收集过去 24h 的生产 trace
        recent_traces = self.feedback.get_recent_traces(hours=24)
        
        new_samples = []
        for trace in recent_traces:
            if trace.success:
                # 成功样本：直接加入训练集
                sample = {
                    "prompt": trace.initial_prompt,
                    "dsl": trace.executed_dsl,
                    "trace": trace.execution_trace,
                    "reward": 1.0,
                    "source": "production",
                    "timestamp": trace.timestamp,
                }
                new_samples.append(sample)
            else:
                # 失败样本：(failed_dsl, error_msg, fixed_dsl) 三元组
                fixed_dsl = self.attempt_repair(trace)
                if fixed_dsl and self.runtime.validate(fixed_dsl).passes_4_layers:
                    sample = {
                        "prompt": trace.initial_prompt,
                        "broken_dsl": trace.executed_dsl,
                        "error_msg": trace.error_message,
                        "fixed_dsl": fixed_dsl,
                        "reward": 0.5,
                        "source": "repair",
                        "timestamp": trace.timestamp,
                    }
                    new_samples.append(sample)
        
        # 2. 保存到训练数据池
        self.save_samples(new_samples)
        
        # 3. 触发增量训练（如果样本数 > 阈值）
        if len(new_samples) > 1000:
            self.trigger_incremental_training()
```

### 5.3 增量训练触发

```python
class IncrementalTrainingTrigger:
    def should_trigger(self):
        # 触发条件：
        # 1. 过去 7 天累计样本 > 1000
        # 2. HydraForgeBench 退化 > 5%
        # 3. 用户反馈负面 > 10%
        
        weekly_samples = self.count_samples(days=7)
        benchmark_regression = self.check_benchmark_regression()
        negative_feedback = self.count_negative_feedback()
        
        return (
            weekly_samples > 1000 or
            benchmark_regression > 0.05 or
            negative_feedback > 0.10
        )
    
    def trigger_incremental_training(self):
        # 不从 base 重新开始（避免灾难性遗忘）
        # 而是在当前模型上继续 SFT
        
        new_data = self.load_recent_samples(days=7)
        
        # Mix with old SFT data (1:1 ratio) to prevent catastrophic forgetting
        old_data = self.sample_old_sft_data(n=len(new_data))
        mixed_data = new_data + old_data
        
        # SFT with low learning rate
        self.policy.fine_tune(
            mixed_data,
            epochs=2,
            lr=5e-6,  # 比初始 SFT 低 4x
        )
        
        # 评估后 canary 部署
        self.evaluate_and_canary_deploy()
```

---

## 6. 时间线整合

### 6.1 与 BOOT-001 的并行关系

| 周期 | BOOT-001（HydraForge 自举）| TR（minimind 训练）| 交付物 |
|---|---|---|---|
| **Week 1-2** | 任务 0.1：CloudLLMAdapter | TR-1 M0: Grammar + Tokenizer | EBNF rules + special tokens |
| **Week 3-4** | 任务 0.2：云端推理工具注册 | TR-1 M1: 数据管线 v1 | 50K SFT 数据 + 4 层验证器 |
| **Week 5-6** | 任务 0.3：推理标准库子图 | TR-1 M2: 冷启动 SFT | HydraForge-AgenticDSL-7B-v1 |
| **Week 7-8** | 任务 1.1：质量评估节点 | TR-2 M3: ReSTᴱᴹ Bootstrap | HydraForge-AgenticDSL-7B-v2 |
| **Week 9-10** | 任务 1.2：服务分层路由 | (继续 ReSTᴱᴹ) | 任务成功率 > 50% |
| **Week 11-12** | 任务 2.1：质量反馈基础设施 | TR-3 M4: PRM 训练 | PRM-1B model |
| **Week 13-16** | 任务 2.2：自适应优化 | TR-3 M5: GRPO 精调 | HydraForge-AgenticDSL-7B-v3 |
| **Week 17-18** | 任务 3.1：推理 API 服务 | TR-3 M6: 部署到 ILLMProvider | HydraForgeAgenticDSLProvider |
| **Week 19-20** | 任务 3.2：元学习优化 | TR-3 M7: 自举阶段 1 完成 | Agent 可独立生成 Skill |

### 6.2 关键里程碑对齐

| 里程碑 | 时间 | HydraForge 自举 | minimind 训练 |
|---|---|---|---|
| **云端集成完成** | W2 | CloudLLMAdapter 可用 | Tokenizer 就绪 |
| **推理标准库就位** | W6 | engine/model/session.md | HydraForge-AgenticDSL-7B-v1 |
| **质量评估可用** | W10 | quality_eval.md | HydraForge-AgenticDSL-7B-v2 |
| **服务分层上线** | W14 | router.md | HydraForge-AgenticDSL-7B-v3 |
| **API 服务对外** | W18 | InferenceServer | HydraForgeAgenticDSLProvider |
| **完全自举** | W20+ | Agent 自主发现 Skill | 自举闭环运转 |

---

## 7. 差异化定位与战略意义

### 7.1 最终定位

**HydraForge AgenticDSL 应作为 LLM 训练的"Agent 领域 WASM"**：
- 不取代 GPT-4/Claude
- 成为 HydraForge 生态的**专用推理后端**
- 类似 WASM 在 Web 的角色：不替代 C++/Rust，但提供跨平台执行契约

### 7.2 与业界方案对比

| 维度 | HydraForge | DSPy | LangGraph | SGLang | FlowAgent/PDL | AgentSPEX |
|---|---|---|---|---|---|---|
| LLM 生成友好度 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 多路径/循环表达 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Runtime 耦合度 | 中（可解耦）| 高 | 极高 | 低 | 低 | 低 |
| 可训练性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ |
| 反馈闭环 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐ |

**关键差异化**：
- HydraForge 是 **唯一** 同时具备"LLM 训练目标 + 工业级运行时 + 执行反馈闭环"的项目
- 已有完整 C++ 引擎实现（vs DSPy/LangGraph 是 Python）
- 已有 4 层验证器（vs ToolBench/Gorilla 仅有执行验证）

### 7.3 生态策略

1. **让基座模型原生支持 AgenticDSL**：在 Qwen3/Llama4 等模型上预训练 AgenticDSL
2. **跨 Runtime 编译**：让 LangGraph/AutoGen 等框架可将自身 DSL 编译到 AgenticDSL
3. **开放标准**：发布 AgenticDSL EBNF grammar + 训练数据集 + HydraForgeBench

---

## 8. 总结

### 关键交付

| 维度 | 交付 |
|---|---|
| **训练路线对齐** | TR-1 → 阶段 0、TR-2 → 阶段 1、TR-3 → 阶段 2 |
| **ILLMProvider 集成** | HydraForgeAgenticDSLProvider |
| **服务化路径** | InferenceServer + MCP + OpenAI 兼容 |
| **自举闭环** | 自动数据生成 + 增量训练 + Canary 部署 |
| **差异化定位** | "Agent 领域 WASM" |

### 与 VN-001 的协同价值

1. **训练后的 LLM 是阶段 3（持续自进化）的关键基础设施**：Agent 自主发现新 Skill 需要快速、可靠的本地 LLM
2. **自举闭环提供持续训练数据**：HydraForge runtime 收集 trace → 训练数据 → 增量训练 → 部署
3. **评估体系支撑自举质量**：HydraForgeBench 持续监控自举效果

### 下一步

阅读 [`07-vs-initial-analysis.md`](07-vs-initial-analysis.md) 了解与初步分析的差异对照，确认本文档对 HydraForge 实际能力的精确反映。

---

**文档版本**: v1.0
**Owner**: minimind + HydraForge 协同
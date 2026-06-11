# 推理时保障 — Grammar 约束 + Speculative + Tree-sitter 验证栈

> **文档 ID**: LLMTRN-001-INFER
> **生成日期**: 2026-06-10
> **关联**:
> - 算法: [`02-training-algorithms.md`](02-training-algorithms.md)
> - 评估: [`04-evaluation-benchmark.md`](04-evaluation-benchmark.md)
> - 风险: [`05-risk-register.md`](05-risk-register.md)
> - 语言演进（EBNF grammar）: HydraForge 仓 `/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md`

---

## 1. 推荐技术栈（minimind 部署视角）

基于 2024-2026 SOTA 调研，minimind 部署 AgenticDSL 生成模型的推荐技术栈：

| 层 | 选型 | 关键参数 |
|---|---|---|
| **CFG Engine** | **XGrammar-2** (arXiv:2601.04426) | Structural Tag 原生支持 tag-triggered dispatch |
| **Inference Backend** | **vLLM 0.x + XGrammar-2** 或 SGLang 0.4+ | vLLM 原生 XGrammar 集成；SGLang up to 10x faster JSON |
| **Speculative Decoding** | **DOMINO** (ICML2024) | 2x speedup over unconstrained；subword-aligned CFG |
| **备选 Speculative** | **CDSL** (Amazon Science) | 2.2x-12.15x speedup with speculation |
| **Tokenizer** | **Qwen2.5-Coder FIM tokens** 范本 | `<\|fim_prefix\|>`、`<\|fim_suffix\|>`、`<\|fim_middle\|>` 等 |
| **Validation** | **Tree-sitter** + markdown_parser | Tree-sitter 增量解析快，错误恢复好 |
| **High-level API** | **Instructor** (3M+ 月下载) | Pydantic validation + 自动 retry |

---

## 2. JSONSchemaBench 性能基准参照

来自 Geng et al. 2025 (arXiv:2501.10868) Llama-3.1-8B + A100 实测 **TPOT (GlaiveAI 1707 schemas)**:

| 框架 | TPOT (ms) | 相对 free decoding | Compliance |
|---|---|---|---|
| LM-only baseline | 15.40 | 1.0× | 0.38 |
| **Guidance** | **6.37** | **2.42× faster** | 0.87-1.00 |
| Llamacpp GBNF | 29.98 | 0.51× | 0.74 |
| Outlines | 30.33 | 0.51× | 0.40 (timeout) |
| XGrammar (HF backend) | 66.78 | 0.23× | 0.66 |

**关键洞察**：
- **Constrained decoding 比 free decoding 平均快 50%**（论文 §1 finding）
- Guidance 是唯一在所有难度 schema 上 empirical coverage 领先的引擎
- OpenAI/Gemini 在简单 schema 上 compliance 100% 但 coverage 低（保守策略）

**GitHub Medium (harder schemas) empirical coverage**：

| 框架 | Empirical | Compliance Rate |
|---|---|---|
| **Guidance** | 0.69 | **0.87** |
| Llamacpp | 0.57 | 0.74 |
| XGrammar | 0.52 | 0.66 |
| Outlines | 0.29 | 0.40 |
| OpenAI (closed) | 0.12 | **1.00** |

---

## 3. XGrammar-2 详解（AgenticDSL 首选 CFG Engine）

### 3.1 为什么选 XGrammar-2

参考 [arXiv:2601.04426](https://arxiv.org/abs/2601.04426) 和 [MLC Blog 2026-05-04](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)：

**核心优势**：
1. **100× mask generation speedup** over prior CFG engines
2. **80× E2E speedup** on H100 (Llama-3.1, JSON generation)
3. **6× tool-calling grammar compilation speedup**（XGrammar-2 vs XGrammar-1）
4. **80× compilation speedup** vs prior SOTA at 500 tools

**架构（XGrammar-1）**：
1. **Byte-level PDA**：每字符 1+ byte，处理 sub-UTF8 token 边界
2. **Adaptive Token Mask Cache**：分 context-independent（预计算）和 context-dependent（运行时 PDA 检查）
3. **Context Expansion**：计算 expanded suffix FSA，过滤 context-dependent tokens
4. **Persistent Execution Stack**：单棵树管理多 stack，支持 rollback

**XGrammar-2 关键新特性**：
- **TagDispatch**：一等语法构造，支持 `<function=...>` 这种 tag-triggered structure switching（**正是 AgenticDSL 场景所需**）
- **Cross-Grammar Cache**：跨请求/跨 schema 复用 substructure
- **Earley-based adaptive token mask cache**：比 PDA-based 更灵活
- **Repetition primitive**：`O(repetition_count)` → `O(1)`
- **OpenAI Harmony Response Format**严格 compliance

### 3.2 与 AgenticDSL 的契合点

| XGrammar-2 能力 | AgenticDSL 需求 |
|---|---|
| **Structural Tag 原生支持** | `<\|subgraph_decl\|>` 锚点触发 dispatch |
| **Byte-level PDA** | YAML block 字符级约束 |
| **Earley parser** | 嵌套 YAML 结构表达 |
| **Repetition primitive** | `nodes:` 列表的重复 |
| **Dynamic schema resolver** | `/lib/**` 签名动态注册 |

### 3.3 部署集成

XGrammar-2 已集成到：
- **vLLM** (2024-12)
- **SGLang** (2024-11)
- **TensorRT-LLM** (2025-01)
- **MLC-LLM**
- **Modular MAX**
- **OpenVINO GenAI**
- **WebLLM**

已被 xAI、DeepSeek、NVIDIA、Databricks、Meta、Google、Perplexity 采用。

---

## 4. Speculative Decoding × Constrained Generation

### 4.1 DOMINO（ICML 2024）

参考 [Guiding LLMs The Right Way](https://proceedings.mlr.press/v235/beurer-kellner24a.html)：

**核心**：subword-aligned CFG constrained decoding + speculative decoding
**数字**：**几乎无 overhead**，某些情况下**接近 2× speedup over unconstrained** decoding（schema-driven JSON with Mistral 7B）
**Speculative lookahead s ∈ {6, 8, 10}** 最高 1.7× throughput over unconstrained

### 4.2 CDSL（Amazon Science）

参考 [CDSL paper](https://assets.amazon.science/6d/a1/c3f0066348aaace0e89a3061ba51/constrained-decoding-with-speculative-lookaheads.pdf)：

**数字**：**2.2× – 12.15× speedup** over CDLH (constrained decoding with lookahead heuristics)
**Constraint satisfaction**：+4-21% on CommonGen, +4-10% on Harmless Text Generation

### 4.3 Grammar-Aligned Decoding（GAD/ASAp，NeurIPS 2024）

**核心问题**：GCD 会 distort LLM 分布（grammar-compliant 但 likelihood 不正比于 LLM 原分布）
**方案**：ASAp (Adaptive Sampling with Approximate expected futures) — 保证 grammar 约束 + 保持 conditional probability

### 4.4 Grammar Guide（KV-cache backtracking）

参考 [parkervg/grammar-guide](https://github.com/parkervg/grammar-guide)：
- **Approach**：Speculative grammar backtracking — 让模型先回答，grammar 仅在违例时介入 + KV-cache 回滚

### 4.5 DINGO（Diffusion LLM Constrained）

参考 [DINGO paper](https://openreview.net/pdf?id=KaYMGsnZ4R)：

**核心**：Dynamic programming-based constrained decoding for **diffusion LLMs**（Inception Mercury 类）
**数字**：GSM-symbolic / JSON-gen 上 **up to 68% improvement** over unconstrained diffusion inference
**关键意义**：当考虑用 diffusion LLM（Mercury）做生成时，parallel token prediction 让传统 sequential constrained decoding 失效 → 必须用 DP 类算法（如 DINGO）

---

## 5. Tokenizer 集成

### 5.1 模型选型

| 模型 | Vocab 大小 | Tokenizer | 备注 |
|---|---|---|---|
| Llama3 / 3.1 | **128K** | tiktoken-derived BPE | 100K + 28K 非英文 token |
| Llama3.1 special tokens | 256 reserved | | |
| Mistral 7B v0.3 | 32K | SentencePiece BPE | |
| **Qwen2.5 / 2.5-Coder** | 151,643 + 22 special | BPE (tiktoken-style) | `<\|fim_prefix\|>`=151659, `<\|fim_middle\|>`=151660, `<\|fim_suffix\|>`=151661, `<\|fim_pad\|>`=151662, `<\|repo_name\|>`=151663, `<\|file_sep\|>`=151664 |

**推荐**：Qwen2.5-Coder-7B（已有 FIM tokens 范本设计经验）

### 5.2 添加 Structural Token（关键步骤）

参考 HydraForge 仓 `/docs/agenticdsl/llm-training-design/SOTA-DESIGN.md` §4：

**11 个新 token**：

| Token | 用途 |
|---|---|
| `<\|agenticdsl_open\|>` | DSL 块开始 |
| `<\|agenticdsl_close\|>` | DSL 块结束 |
| `<\|subgraph_decl\|>` | 子图声明头 |
| `<\|node_def\|>` | 节点定义开始 |
| `<\|inja_expr_open\|>` | `{{` 模板起始 |
| `<\|inja_expr_close\|>` | `}}` 模板结束 |
| `<\|fim_prefix\|>` | FIM prefix（Qwen2.5-Coder 同款） |
| `<\|fim_middle\|>` | FIM middle |
| `<\|fim_suffix\|>` | FIM suffix |
| `<\|fim_pad\|>` | FIM padding |
| `<\|agenticdsl_eos\|>` | DSL 生成结束 |

### 5.3 Vocab Surgery（必做）

**绝对不要**：
- ❌ 直接 append 到 vocab 末尾（让 BPE priority score 偏低，新 token 极少被选中）

**正确方法**：
参考 PickyBPE (EMNLP2024) / Teaching Old Tokenizers New Words (EACL2026)：
1. 用 5-10 GB AgenticDSL 语料**继续训练**现有 BPE
2. **同步**添加 structural tokens 到 vocab
3. 验证 tokenization efficiency 不退化

### 5.4 Stop Tokens 防泄漏（关键）

**Qwen2.5-Coder issue #99 教训**：即使训练充分，模型仍会 leak FIM tokens。

**必须**：
- 把 `<|fim_*|>` 和 `<|agenticdsl_*|>` 加入 `stop_token_ids`
- 推理时验证生成末尾不包含 FIM tokens

```python
# vLLM 推理配置
stop_token_ids = [
    tokenizer.encode("<|fim_prefix|>", add_special_tokens=False)[0],
    tokenizer.encode("<|fim_middle|>", add_special_tokens=False)[0],
    tokenizer.encode("<|fim_suffix|>", add_special_tokens=False)[0],
    tokenizer.encode("<|fim_pad|>", add_special_tokens=False)[0],
    tokenizer.encode("<|agenticdsl_open|>", add_special_tokens=False)[0],
    # ... 其他 structural tokens
]
```

---

## 6. Tree-sitter 验证层

### 6.1 为什么选 Tree-sitter

| Parser | 速度 | DSL 表达力 | 备注 |
|---|---|---|---|
| **Tree-sitter** | 🟢 极快 | ✅ any CFG | 增量 parsing, 错误恢复好, **最适合 validation layer** |
| Lark | 🟡 | ✅ EBNF + LALR/earley | Python-pure, 易扩展 |
| ANTLR4 | 🟡 | ✅ EBNF | 多语言 target |
| PEG 自写 | 🟢 | ⚠️ 受限 | 适合小 grammar |

**推荐**：Tree-sitter 写 AgenticDSL grammar（speed + error recovery + 多语言 binding），Lark 做测试/原型的 quick iteration。

### 6.2 Tree-sitter AgenticDSL Grammar

```javascript
// grammar.js (Tree-sitter)
module.exports = grammar({
  name: 'agenticdsl',

  rules: {
    document: $ => repeat(choice($.agenticdsl_block, $.markdown_block)),

    agenticdsl_block: $ => seq(
      '###',
      'AgenticDSL',
      field('path', $.path),
      '```yaml',
      '---', 'BEGIN', 'AgenticDSL', '---',
      field('body', $.yaml_body),
      '---', 'END', 'AgenticDSL', '---',
      '```'
    ),

    path: $ => seq("'", /\/[a-zA-Z0-9_\/]+(@v\d+)?/, "'"),

    yaml_body: $ => choice($.meta_block, $.subgraph_body),

    meta_block: $ => seq(
      'version:', $.string,
      'mode:', choice('dev', 'prod'),
      'entry_point:', $.path,
      'execution_budget:', $.budget_struct,
    ),

    subgraph_body: $ => seq(
      optional(seq('graph_type:', 'subgraph')),
      optional(seq('signature:', $.signature_struct)),
      $.nodes_list,
    ),

    nodes_list: $ => repeat1($.node_def),

    node_def: $ => seq(
      '-', 'id:', $.identifier,
      'type:', $.node_type,
      repeat($.node_field),
    ),

    node_type: $ => choice(
      'start', 'end', 'assign', 'tool_call', 'dsl_call',
      'assert', 'fork', 'join', 'generate_subgraph',
    ),

    // ... 其他字段定义
  },
});
```

### 6.3 错误恢复机制

Tree-sitter 的核心优势：**错误恢复（Error Recovery）**。

```python
# Tree-sitter error recovery for AgenticDSL
import tree_sitter_agenticdsl

parser = tree_sitter.Parser()
parser.set_language(tree_sitter_agenticdsl.language())

def parse_with_errors(source: str):
    tree = parser.parse(bytes(source, "utf8"))
    errors = []

    def visit(node):
        if node.is_missing:
            errors.append({
                "type": "MISSING_NODE",
                "location": (node.start_point, node.end_point),
                "expected": node.type,
            })
        if node.has_error:
            errors.append({
                "type": "PARSE_ERROR",
                "location": (node.start_point, node.end_point),
                "text": source[node.start_byte:node.end_byte],
            })
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return tree, errors
```

---

## 7. 端到端推理路径

```
用户输入 NL
   │
   ▼
[Step 1] Prompt 注入
  - 当前 schema（已注册子图/工具的签名）
  - LayeredContext 摘要（recent_turns, working.*）
  - 任务目标
  - few-shot examples (5-10 个)
   │
   ▼
[Step 2] vLLM + XGrammar-2 生成
  - Grammar: agenticdsl_v3_10.ebnf
  - Dynamic schema resolver: registered subgraphs/tools
  - Stop tokens: <|fim_*|>, <|agenticdsl_*|>
  - Max tokens: 2048 (Track 0.1 M1.3 调整)
   │
   ▼
[Step 3] Tree-sitter 验证
  - markdown_parser 严格解析
  - signature_validator 检查签名
  - 收集错误位置 + 类型
   │
   ▼ fail → retry with error message (Self-Refine 风格)
[Step 4] HydraForge Runtime dry-run
  - TopologicalScheduler 执行
  - TraceRecord 收集
  - 预算控制
   │
   ▼ fail → retry with trace
[Step 5] 返回最终 DSL 或错误
```

### 7.1 Self-Refine 推理时增强

参考 [Self-Refine (NeurIPS 2023)](https://arxiv.org/abs/2303.17651)：

```python
def self_refine_generate(prompt, max_attempts=3):
    for attempt in range(max_attempts):
        dsl = vllm_generate(prompt, grammar=xgrammar)
        tree, errors = tree_sitter_parse(dsl)
        if not errors:
            exec_result = runtime.dry_run(dsl)
            if exec_result.success:
                return dsl
        
        # 收集错误信息作为 feedback
        error_msg = format_errors(errors, exec_result)
        prompt = prompt + f"\n\n[Previous attempt failed]:\n{error_msg}\nPlease fix and regenerate."
    
    return dsl  # 返回最后一次尝试
```

**Self-Refine 数字参考**：在 7 个任务上，比 baseline **平均绝对改进 ~20%**。

### 7.2 Instructor 集成（自动 Retry）

参考 [Instructor](https://github.com/567-labs/instructor)：

```python
import instructor
from pydantic import BaseModel

class AgenticDSLResponse(BaseModel):
    dsl: str
    confidence: float

client = instructor.from_provider("vllm")
response = client.messages.create(
    model="hydraforge-agenticdsl-7b",
    response_model=AgenticDSLResponse,
    messages=[{"role": "user", "content": prompt}],
    max_retries=3,  # 自动 retry on validation error
)
```

**Instructor 优势**：
- 自动 retry on Pydantic ValidationError（用 error message 作为 prompt 反馈）
- 多 provider (OpenAI/Anthropic/Gemini/Ollama/vLLM)
- 3M+ 月下载，生态成熟

---

## 8. 备选方案评估

### 8.1 Guidance（Microsoft）

参考 [github.com/guidance-ai/guidance](https://github.com/guidance-ai/guidance)：

**优势**：
- JSONSchemaBench 上 compliance 最高 (0.87-1.00)
- 编程范式（不是单纯约束 API）

**劣势**：
- 集成度不如 XGrammar-2 / vLLM
- 不支持 byte-level PDA

**何时选**：本地/CPU 后端，对合规性要求高于速度

### 8.2 Outlines

参考 [github.com/outlines-dev/outlines](https://github.com/outlines-dev/outlines)：

**优势**：Pydantic 深度集成

**劣势**：
- 复杂 schema timeout（GitHub Medium empirical coverage 仅 29%）
- GCT 3-8s（最慢）

**何时选**：Pydantic 模型简单场景

### 8.3 LMQL（ETH SRI）

参考 [github.com/eth-sri/lmql](https://github.com/eth-sri/lmql)：

**特点**：
- Python superset，token-level constraints
- In-process token mask

**何时选**：研究项目、需要细粒度控制

### 8.4 llama.cpp GBNF

**优势**：轻量、本地部署友好
**劣势**：速度不如 Guidance

**何时选**：纯本地 GGUF 推理，无 vLLM

---

## 9. 性能基准与监控

### 9.1 关键指标

| 指标 | 目标 | 监控工具 |
|---|---|---|
| **Format compliance** | > 99% | Tree-sitter 解析率 |
| **Signature validation pass** | > 95% | signature_validator |
| **Dry-run success** | > 90% | runtime.execute() |
| **TPOT (time per output token)** | < 30ms | vLLM metrics |
| **TTFT (time to first token)** | < 300ms | vLLM metrics |
| **GCT (grammar compile time)** | < 100ms | XGrammar metrics |

### 9.2 A/B 测试框架

```python
# A/B test: constrained vs free decoding
results = {"constrained": [], "free": []}

for prompt in test_prompts:
    # Constrained
    dsl_constrained = vllm.generate(
        prompt, 
        guided_decoding_backend="xgrammar",
        guided_grammar=agenticdsl_grammar,
    )
    results["constrained"].append({
        "format_valid": tree_sitter_parse(dsl_constrained)[1] == [],
        "exec_success": runtime.dry_run(dsl_constrained).success,
        "latency": measure_latency(),
    })
    
    # Free
    dsl_free = vllm.generate(prompt)
    results["free"].append({...})

# Compare
print(f"Constrained: {np.mean([r['format_valid'] for r in results['constrained']])}")
print(f"Free: {np.mean([r['format_valid'] for r in results['free']])}")
```

---

## 10. 总结

### 关键交付

| 维度 | 选型 | 关键依据 |
|---|---|---|
| **CFG Engine** | **XGrammar-2** | Structural Tag 原生支持；6×编译提速；与 AgenticDSL 锚点契合 |
| **Inference** | **vLLM + XGrammar-2** | vLLM 0.x 原生 XGrammar 集成 |
| **Speculative** | **DOMINO** (备选 CDSL) | 2×-12× speedup |
| **Tokenizer** | **Qwen2.5-Coder FIM tokens** 范本 | 已有 FIM 设计可参考 |
| **Validation** | **Tree-sitter** + markdown_parser | 增量解析 + 错误恢复 |
| **High-level API** | **Instructor** | 3M+ 月下载，自动 retry |

### 关键数字（来自 JSONSchemaBench）

- Constrained decoding **比 free decoding 平均快 50%**
- Guidance TPOT **2.42× faster than unconstrained**
- XGrammar-2 比 XGrammar-1 **6× 编译提速**（tool-calling grammar）
- XGrammar-2 比 prior SOTA **80× compilation speedup** at 500 tools

### 关键陷阱

1. **不要直接 append 新 token 到 vocab 末尾**（PickyBPE 教训）
2. **FIM tokens 必须加入 stop_token_ids**（Qwen issue #99 教训）
3. **不要忽略格式合规监控**（应保持 > 95%）
4. **不要省略 Tree-sitter 验证**（仅 grammar 约束不够）

### 下一步

推理栈就绪后，进入 [`04-evaluation-benchmark.md`](04-evaluation-benchmark.md) 学习如何建立 HydraForgeBench 评估体系。

---

**文档版本**: v1.0
**Owner**: minimind 部署团队
# 07 · 生产部署 — 数据飞轮 + 监控 + 增量更新

> **文档 ID**: MEM-007-PRODUCTION-DEPLOYMENT
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.1
> **配套文档**:
> - 训练设计: [`02-training-design.md`](02-training-design.md) — 四阶段课程
> - 本体涌现: [`05-schema-emergence.md`](05-schema-emergence.md) — V1-V3 路线图
> - 评估方法论: [`06-evaluation-methodology.md`](06-evaluation-methodology.md) — Probe Model + Golden Filter
> - 训练侧实现: [`../agenticmemory_training/`](../agenticmemory_training/) — 训练数据管线
> - 推理引擎: [`../inference-engine/`](../inference-engine/) — KV 管理实现

---

## 0. 文档范围与定位

本文档定义 **agenticmemory 的生产部署架构**——从训练完成到持续运维的完整闭环,包括推理服务、数据飞轮、监控告警、增量更新策略。

**与现有文档的关系**:
- 本文档定义**"部署后"**的架构与流程
- 训练侧文档(`agenticmemory_training/`)定义**"训练前"**的数据合成与模型训练
- 本文档与训练侧通过"模型版本管理"和"数据飞轮"闭环连接

**核心原则**:部署不是一次性动作,而是持续迭代。通过数据飞轮(发现不足→补充数据→重训→部署)保持系统的长期有效性。

---

## 1. 生产部署架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI / vLLM)                     │
│                         ↓ 请求路由                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ agenticmind      │  │ agenticinference │  │ 未来 consumer     │  │
│  │ (13 字段查询)     │  │ (Wiki DAG 查询)  │  │                 │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                      │                      │            │
│           └──────────────────────┼──────────────────────┘            │
│                                  ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │         agenticmemory 推理服务 (vLLM + S-LoRA)               │ │
│  │                                                                 │ │
│  │  请求调度:                                                     │ │
│  │    - 单次请求:probe(lora_session) → KV cache → 13 字段       │ │
│  │    - 批量请求:continuous batching + prefix radix 复用          │ │
│  │                                                                 │ │
│  │  模型层:                                                       │ │
│  │    - base_model: Qwen3-0.6B (固定)                            │ │
│  │    - LoRA_registry: [session, memory, ...] (动态加载)        │ │
│  │    - KV 缓存: 预分配 + streaming + prefix radix                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                  ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │         推理结果缓存 (Redis + Neo4j)                           │ │
│  │                                                                 │ │
│  │  - 短期缓存:最近 N 次查询的 KV cache(Redis LRU)             │ │
│  │  - 长期缓存:Wiki DAG 结构(Neo4j)                            │ │
│  │  - Schema 版本:schema_memory_v*.json(版本化管理)           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                  ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │         数据飞轮(自动化反馈与迭代)                           │ │
│  │                                                                 │ │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐  │ │
│  │  │ 生产日志  │ → │ 效用评估  │ → │ 低分回收  │ → │ 增量微调  │  │ │
│  │  │ 采集     │    │ (Probe)   │    │ (Golden)  │    │         │  │ │
│  │  └──────────┘    └──────────┘    └──────────┘    └─────────┘  │ │
│  │                                                    ↓            │ │
│  │                                            ┌──────────┐         │ │
│  │                                            │ 新模型    │         │ │
│  │                                            │ 版本发布  │         │ │
│  │                                            └──────────┘         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**关键架构决策**:
1. **base 模型固定**:所有 LoRA 共享同一个 Qwen3-0.6B base,避免维护多个 base 版本
2. **LoRA 动态加载**:每个 consumer 有独立 LoRA,按需加载到显存
3. **KV 缓存复用**:prefix radix + pre-allocated KV,避免重复 prefill
4. **数据飞轮**:生产环境的低分案例自动进入训练集,驱动持续迭代

---

## 2. 推理服务层

### 2.1 vLLM + S-LoRA 服务配置

```yaml
# vllm_serving_config.yaml
serving:
  framework: "vllm"  # 推理引擎
  base_model: "Qwen/Qwen3-0.6B"  # 固定 base,所有 LoRA 共享
  
  # LoRA 注册表
  lora_registry:
    session:
      path: "./models/lora_session_v1.safetensors"
      rank: 8
      target_modules: ["q_proj", "v_proj"]
      max_concurrent: 100  # 最大并发数
    memory:
      path: "./models/lora_memory_v1.safetensors"
      rank: 8
      target_modules: ["q_proj", "v_proj"]
      max_concurrent: 50
  
  # KV 缓存策略
  kv_cache:
    pre_allocated: true           # 复用 inference-engine 已有实现
    max_position_embeddings: 32768
    streaming_llm:                 # 长对话场景启用
      enabled: true
      n_sink: 4
      n_local: 2048
    prefix_radix:                  # 多 consumer 共享前缀
      enabled: true
      implementation: "sglang_style"
  
  # 并发与批量
  continuous_batching:
    enabled: true
    max_batch_size: 32
    max_pending_requests: 100
  
  # 性能目标
  performance:
    max_latency_ms: 500           # P95 延迟目标
    target_throughput_qps: 100    # 目标吞吐量
```

### 2.2 请求处理流程

```python
async def serve_request(request: ExtractionRequest) -> ExtractionResponse:
    """
    agenticmemory 推理服务请求处理
    
    参数:
      request: 包含 context, task_type, lora_id 等
    """
    
    # Step 1: 路由到对应 LoRA
    lora_adapter = lora_registry[request.lora_id]
    
    # Step 2: 检查 KV 缓存(若已有则复用)
    cached_kv = kv_cache_manager.get(request.context_hash)
    if cached_kv:
        # prefix radix 复用:只 prefill 新增部分
        new_kv = prefill_incremental(cached_kv, request.context)
    else:
        # 首次 prefill
        new_kv = prefill_base_model(request.context)
        kv_cache_manager.set(request.context_hash, new_kv)
    
    # Step 3: probe 查询(用 LoRA 从 KV 提取)
    result = await probe_with_lora(
        kv_cache=new_kv,
        query=request.query,
        lora_adapter=lora_adapter
    )
    
    # Step 4: 记录到数据飞轮(见 §4)
    log_to_data_flywheel(request, result)
    
    return result
```

### 2.3 推理时约束解码

使用 XGrammar 或 Outlines 进行约束解码,确保 Wiki DAG 输出格式合规:

```python
def constrained_inference(prompt, schema):
    """约束解码确保输出符合 Wiki Schema"""
    
    # 将 Wiki Schema 转换为 JSON Schema
    json_schema = schema_to_json_schema(schema)
    
    # 使用 XGrammar 约束解码
    output = model.generate(
        prompt,
        constrained_by=json_schema,
        engine="xgrammar",
        max_tokens=4096,
        temperature=0.1
    )
    
    return output
```

**关键设计**:
- 约束解码只在 Wiki 输出阶段启用
- 提取阶段(13 字段、三元组)不启用约束解码,保持灵活性
- 详见 [`02-training-design.md`](02-training-design.md) §8.3

---

## 3. 缓存层设计

### 3.1 三层缓存架构

```
┌─────────────────────────────────────────────────────────────┐
│  L1: 短期查询缓存(Redis LRU)                               │
│  缓存:最近 1000 次查询的 KV cache                          │
│  用途:高频查询快速响应,避免重复 prefill                     │
│  TTL:1 小时                                                │
│  大小:~2 GB(约 1000 个会话的 KV cache)                  │
├─────────────────────────────────────────────────────────────┤
│  L2: Wiki DAG 持久化(Neo4j)                                │
│  缓存:Wiki DAG 结构 + 实体/关系索引                       │
│  用途:复杂查询路由 + 跨文档知识关联                       │
│  TTL:永久(版本化)                                       │
│  大小:~50 GB(初始 100 万实体)                          │
├─────────────────────────────────────────────────────────────┤
│  L3: Schema 版本管理(PostgreSQL)                         │
│  缓存:schema_memory_v*.json + 变更日志                    │
│  用途:Schema 演化追溯 + 版本回滚                          │
│  TTL:永久(所有版本)                                     │
│  大小:~10 MB(轻量级)                                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 KV 缓存的量化策略

```python
class KVCacheQuantizer:
    """KV 缓存量化:根据使用频率动态调整精度"""
    
    def __init__(self):
        self.quantization_rules = {
            "hot_kv": {  # 高频使用(最近 1 小时内访问)
                "precision": "fp16",
                "compression": "none"
            },
            "warm_kv": {  # 中频使用(最近 24 小时内)
                "precision": "int8",
                "compression": "kivi_2bit"  # 见 inference-engine/09-kivi.md
            },
            "cold_kv": {  # 低频使用(超过 24 小时)
                "precision": "int4",
                "compression": "kivi_4bit + sparse"
            }
        }
    
    def quantize_kv(self, kv_cache, usage_frequency):
        """根据使用频率决定量化策略"""
        if usage_frequency > 10:  # 每小时 > 10 次
            return kv_cache  # fp16,不量化
        elif usage_frequency > 1:  # 每天 > 1 次
            return self.kivi_quantize(kv_cache, bits=2)  # int8
        else:
            return self.kivi_quantize(kv_cache, bits=4)  # int4
```

---

## 4. 数据飞轮:持续迭代闭环

### 4.1 数据飞轮架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据飞轮(自动化闭环)                        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 1: 生产日志采集                                        │  │
│  │    - 记录所有请求和响应                                       │  │
│  │    - 记录 Probe Model 的效用评估(Score_struct vs Score_raw)│  │
│  │    - 收集低分案例(B/A < 0.95 或效用增益 < 阈值)             │  │
│  └────────────────────┬───────────────────────────────────────────┘  │
│                       │                                             │
│                       ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 2: 效用评估(Golden Filter)                            │  │
│  │    - 对低分案例重新跑 Probe Model 评估                       │  │
│  │    - 识别失败类型:数值丢失 / 关系丢失 / 条件丢失 / 实体混淆   │  │
│  │    - 生成失败诊断报告                                         │  │
│  └────────────────────┬───────────────────────────────────────────┘  │
│                       │                                             │
│                       ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 3: 低分数据回收                                       │  │
│  │    - 将进入失败诊断的案例加入训练集队列                     │  │
│  │    - 按失败类型分类存储                                     │  │
│  │    - 等待积累到一定数量后触发增量微调                         │  │
│  └────────────────────┬───────────────────────────────────────────┘  │
│                       │                                             │
│                       ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 4: 增量微调                                          │  │
│  │    - 用积累的失败案例作为训练数据                           │  │
│  │    - 只更新对应类型的 LoRA(不重新训练 base)              │  │
│  │    - 每次微调 0.5 epoch,避免过度拟合                       │  │
│  │    - 训练完成后触发 A/B 测试                                │  │
│  └────────────────────┬───────────────────────────────────────────┘  │
│                       │                                             │
│                       ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 5: 新模型发布                                         │  │
│  │    - 新版本 LoRA 自动部署到生产环境                         │  │
│  │    - 保留旧版本作为回滚备份                                 │  │
│  │    - 监控新版本的指标变化,如不达标自动回滚                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 增量微调触发条件

```python
def should_trigger_retraining(failure_logs, threshold_config):
    """判断是否触发增量微调"""
    
    triggers = []
    
    # 触发条件 1: 某类失败案例积累超过阈值
    if failure_logs["numeric_loss"] > 100:
        triggers.append("numeric_loss_exceeds_threshold")
    
    if failure_logs["relation_loss"] > 100:
        triggers.append("relation_loss_exceeds_threshold")
    
    # 触发条件 2: 整体效用增益下降超过阈值
    current_utility = compute_utility_gain(failure_logs)
    baseline_utility = get_baseline_utility()
    if current_utility < baseline_utility * 0.9:  # 下降超过 10%
        triggers.append("utility_degradation_detected")
    
    # 触发条件 3: 新的 schema 外实体类型出现频率过高
    new_entity_types = count_new_entity_types(failure_logs)
    if new_entity_types > 50:  # 一周出现 50 个以上新类型
        triggers.append("ontology_evolution_needed")
    
    return triggers
```

### 4.3 数据飞轮的关键指标

| 指标 | 说明 | 告警阈值 |
|---|---|---|
| **失败案例积累速率** | 每小时新增的低分案例数 | > 50 / 小时 告警 |
| **效用增益中位数** | Probe Model 的效用增益分布 | < 0.05 告警 |
| **增量微调频率** | 每月触发的微调次数 | > 10 次 / 月 告警 |
| **新版本上线成功率** | 新 LoRA 上线后指标改善的比例 | < 70% 告警 |

---

## 5. 监控与告警

### 5.1 三层监控体系

```
┌─────────────────────────────────────────────────────────────────────┐
│  Level 1: 实时监控(秒级)                                          │
│    - API 延迟(p50 / p95 / p99)                                    │
│    - 吞吐量(QPS)                                                │
│    - 错误率                                                       │
│    - 资源使用率(GPU / 显存 / 内存)                              │
├─────────────────────────────────────────────────────────────────────┤
│  Level 2: 分钟级监控(分钟级)                                      │
│    - Wiki 格式合法率(输出 JSON 合法率)                          │
│    - Schema 遵从率(输出是否符合 Schema)                        │
│    - 字段填充率(13 字段 / Wiki 8 字段的填充情况)                │
│    - 幻觉率(输出中出现原文未提及内容的比例)                      │
├─────────────────────────────────────────────────────────────────────┤
│  Level 3: 小时级监控(小时级)                                     │
│    - B/A 比值(双轨评估)                                         │
│    - IRR(信息召回率)                                            │
│    - Probe Model 效用增益                                         │
│    - 失败案例分类统计(数值丢失 / 关系丢失 / 条件丢失)           │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 关键告警规则

```yaml
# alert_rules.yml
alerts:
  
  # 性能告警
  - name: "high_latency"
    condition: "p95_latency_ms > 500"
    severity: "warning"
    action: "扩容 GPU 实例或检查是否有慢查询"
  
  - name: "low_throughput"
    condition: "qps < 50"
    severity: "warning"
    action: "检查 continuous batching 是否正常"
  
  # 质量告警
  - name: "format_noncompliance"
    condition: "wiki_format_compliance_rate < 0.95"
    severity: "critical"
    action: "检查 Wiki Schema 是否有变更或模型是否退化"
  
  - name: "schema_drift"
    condition: "new_entity_types_per_day > 10"
    severity: "warning"
    action: "触发 Schema 演化流程(见 §4)"
  
  - name: "hallucination_spike"
    condition: "hallucination_rate > 0.05"
    severity: "critical"
    action: "检查训练数据是否有污染或模型是否退化"
  
  # 数据飞轮告警
  - name: "data_flywheel_stall"
    condition: "no_retrain_triggered_for_days > 14"
    severity: "info"
    action: "检查失败案例是否过少或过滤条件是否过严"
  
  - name: "retrain_failure_spike"
    condition: "retrain_failure_rate > 0.30"
    severity: "critical"
    action: "检查训练数据质量或超参数配置"
```

### 5.3 Grafana Dashboard 布局

```
┌─────────────────────────────────────────────────────────────────────┐
│  AgenticMemory Production Dashboard                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 关键指标(顶部)                                                │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐        │
│  │ 延迟 P95 │ QPS      │ 格式合规 │ Schema遵从│ B/A 比值  │        │
│  │ 420ms    │ 87       │ 96.2%    │ 91.5%    │ 0.97     │        │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘        │
│                                                                     │
│  📈 性能趋势(中部)                                                │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ [延迟趋势图] [吞吐量趋势图] [错误率趋势图]             │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  🎯 质量指标(下部)                                                │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ [Wiki IRR] [Schema 遵从率] [幻觉率] [效用增益]          │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  🔄 数据飞轮状态(底部)                                            │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ [失败案例积累] [增量微调历史] [模型版本] [Schema 版本]  │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 增量更新策略

### 6.1 三种增量更新策略

```python
class IncrementalUpdateStrategy:
    """增量更新策略选择"""
    
    @staticmethod
    def update_strategy(change_magnitude, schema_compatibility):
        """根据变更幅度决定更新策略"""
        
        if change_magnitude == "small" and schema_compatibility:
            # 小幅度更新:只更新 LoRA
            return {
                "strategy": "lora_only",
                "description": "只重新训练 LoRA,不更新 base model",
                "data_requirement": "failure_cases + 10% replay_buffer",
                "training_time": "~30 min",
                "deployment_risk": "low"
            }
        
        elif change_magnitude == "medium" and schema_compatibility:
            # 中幅度更新:LoRA + 部分 base 微调
            return {
                "strategy": "lora_plus_base_partial",
                "description": "更新 LoRA + 冻结部分 base 层",
                "data_requirement": "failure_cases + 30% replay_buffer",
                "training_time": "~2 hours",
                "deployment_risk": "medium"
            }
        
        else:
            # 大幅度更新:全量重训
            return {
                "strategy": "full_retrain",
                "description": "全量重训 base + LoRA",
                "data_requirement": "all_data + 30% replay_buffer",
                "training_time": "~12 hours",
                "deployment_risk": "high",
                "requires_approval": True
            }
```

### 6.2 数据飞轮的数据组成

```
增量微调的数据来源:
  1. 失败案例回收(60%)——生产环境中的低分案例
  2. Schema 演化数据(20%)——新涌现的实体类型/关系类型
  3. 定期回放的通用数据(20%)——防止灾难性遗忘

配比原则:
  - 失败案例优先(解决实际问题)
  - 保留足够的通用数据(防止遗忘)
  - Schema 演化数据用于本体扩展
```

### 6.3 回归测试

```python
def regression_test(new_model, baseline_model, test_set):
    """每次增量微调后的回归测试"""
    
    results = {
        "memory_recall": evaluate_memory_recall(new_model, test_set),
        "relation_accuracy": evaluate_relation_accuracy(new_model, test_set),
        "wiki_irr": evaluate_wiki_irr(new_model, test_set),
        "ba_ratio": evaluate_ba_ratio(new_model, test_set)
    }
    
    baseline_results = {
        "memory_recall": evaluate_memory_recall(baseline_model, test_set),
        "relation_accuracy": evaluate_relation_accuracy(baseline_model, test_set),
        "wiki_irr": evaluate_wiki_irr(baseline_model, test_set),
        "ba_ratio": evaluate_ba_ratio(baseline_model, test_set)
    }
    
    # 如果新模型在任何指标上退化超过 5%,拒绝部署
    for metric in results:
        if results[metric] < baseline_results[metric] * 0.95:
            raise DeploymentRejection(f"{metric} 退化超过 5%")
    
    return {"status": "approved", "metrics": results}
```

---

## 7. 部署时间表

### 7.1 分阶段部署计划

```
V1.0 部署(MVP):
  Week 1: 基础设施搭建(vLLM + Redis + Neo4j)
  Week 2: 首个 LoRA(session)部署
  Week 3: 监控系统搭建(Grafana)
  Week 4: 数据飞轮上线(失败案例回收)
  
  验收标准:
  - 端到端延迟 p95 < 500ms
  - Wiki 格式合规率 ≥ 95%
  - 数据飞轮每周至少回收 100 个失败案例

V2.0 部署(剪枝后):
  Week 5-6: 第二个 LoRA(memory)部署
  Week 7-8: 双 LoRA 并发测试(S-LoRA)
  Week 9: Schema 演化监控上线
  Week 10: 全量数据飞轮(含 Schema 演化触发)

V3.0 部署(RL 涌现后):
  Week 11-12: RL 后模型 A/B 测试
  Week 13-14: 本体涌现监控上线
  Week 15: 生产环境稳定运行 1 个月观察期
```

---

## 8. 与现有文档的关系

| 内容 | 本文位置 | 详见 |
|---|---|---|
| **训练数据管线** | 见 [`../agenticmemory_training/`](../agenticmemory_training/) | 训练前 |
| **Wiki DAG 构建** | 见 [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) | 训练前 |
| **评估方法论** | 见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) | 评估方法 |
| **本体涌现策略** | 见 [`05-schema-emergence.md`](05-schema-emergence.md) §3(V1-V3) | 训练策略 |
| **推理引擎实现** | 见 [`../inference-engine/`](../inference-engine/) | KV 管理 |
| **生产部署架构** | 本文 §1-§2 | 部署后 |
| **数据飞轮** | 本文 §4 | 持续迭代 |
| **监控告警** | 本文 §5 | 运维 |
| **增量更新** | 本文 §6 | 持续迭代 |

---

## 9. 关键设计决策总结

| 决策点 | 选择 | 理由 |
|---|---|---|
| **推理框架** | vLLM + S-LoRA | 高吞吐 + 支持多 LoRA |
| **base 模型策略** | 固定 Qwen3-0.6B,只更新 LoRA | 避免维护多个 base 版本 |
| **KV 缓存** | prefix radix + pre-allocated + streaming | 复用已有 inference-engine 实现 |
| **数据飞轮触发** | 失败案例积累 + 效用增益下降 + Schema 演化 | 多信号触发,避免单一信号误判 |
| **增量更新策略** | 优先 LoRA,再 base partial,最后全量重训 | 最小化部署风险 |
| **监控层级** | 实时(秒)/ 分钟 / 小时 三层 | 兼顾及时性与计算成本 |
| **回归测试** | 每次部署前强制回归测试 | 防止退化上线 |

---

## 10. 待解决问题

| # | 问题 | 状态 | 建议决策时机 |
|---|---|---|---|
| **O33** | **Neo4j 的 Wiki DAG 持久化格式**(JSON 存文档型数据库 vs 图数据库) | 🟡 待讨论 | V1.0 部署前 |
| **O34** | **数据飞轮中失败案例的去重策略**(相似案例是否合并?) | 🟡 待讨论 | V1.0 部署前 |
| **O35** | **KV 缓存量化精度的选择**(int8 vs int4 的质量-成本权衡) | 🟡 待讨论 | V1.0 部署前 |
| **O36** | **多 LoRA 并发时的显存上限**(S-LoRA 的最大 LoRA 数量) | 🟢 实施时验证 | V1.0 部署时 |
| **O37** | **Schema 演化的审批流程**(自动合并 vs 人工审核) | 🟡 待讨论 | V1.0 部署前 |

---

## 11. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-25 | 初始版本:生产部署架构 + 数据飞轮 + 监控告警 + 增量更新策略 | Sisyphus(AI 助手)+ 用户 |

---

**文档版本**: v0.1
**Owner**: AgenticMind 部署组
**下一步**: 待 O33/O35 决策后启动 V1.0 部署
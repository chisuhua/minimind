# 推理引擎 · 自循环关键基础设施

> **目录定位**:`docs/inference-engine/` —— **自循环认知推理模型** 的 **3 份 CRITICAL_FOR_LOOP 推理引擎文档**。
>
> **新目标相关度**:⭐⭐ **CRITICAL_FOR_LOOP(自循环关键支撑)**

---

## 🎯 为什么这些是"自循环"的关键基础设施?

新目标:**通过 AgenticDSL 语言和智能体运行时协同进行自循环的认知推理模型** —— 即"生成 → 执行 → 评估 → 重新生成(可能基于前序 prefix)"的多轮闭环。

这个自循环带来的工程挑战:
- **多轮 prefix 复用**:每轮都要复用前序生成的 prefix(DSL 块、tool_call、评估反馈、observation)
- **长链路不爆显存**:自循环轨迹会无限增长(执行 observation + DSL 块 + 评估反馈累加)
- **KV cache 是首要瓶颈**:每轮都积累 KV cache,需要量化压缩

**没有这三项基础设施,自循环在 sub-1B 模型上根本不可能高效运行** —— 这就是把它们从通用"性能优化"中升格为"自循环关键支撑"的原因。

---

## 📁 目录结构

| 文档 | 支撑度 | 关键作用 |
|---|---|---|
| ⭐⭐ [`01-pre-allocated-kv-cache.md`](./01-pre-allocated-kv-cache.md) | **CRITICAL** | 预分配 KV 缓存 — 自循环每轮都要复用前序 prefix,预分配是跨轮 prefix 复用零浪费的**物理底座**(已实现于 `model/model_minimind.py`) |
| ⭐⭐ [`02-streaming-llm.md`](./02-streaming-llm.md) | **CRITICAL** | StreamingLLM(流式 + 注意力汇点 + 环形 buffer)— 让长链路不爆显存,是自循环能不终止跑下去的**前提**(已实现于 `model/streaming_kv_cache.py`) |
| ⭐⭐ [`09-kivi.md`](./09-kivi.md) | **CRITICAL** | KIVI 2-bit 量化 — 自循环 KV cache 是首要瓶颈,KIVI 让"长链路的 prefix"也能塞进显存,与 streaming 形成**互补**(已实现于 `model/kivi_kv_cache.py`) |

---

## 🔗 与新目标自循环环节的对应

| 自循环环节 | 所需支撑 | 本目录文档 |
|---|---|---|
| **生成**(AgenticDSL 输出) | KV cache 复用 | `01-pre-allocated-kv-cache` |
| **执行**(运行时执行求解) | 跨轮 context 保留 | `01-pre-allocated-kv-cache` + `02-streaming-llm` |
| **评估**(质量评估节点) | 长 prefix 不爆显存 | `02-streaming-llm` + `09-kivi` |
| **优化**(运行时自进化) | 持久化 KV 状态 | 全部 3 项 |

---

## 📚 相关:通用推理加速(已移入 references/)

其余 15 份推理加速技术(推测解码族 / 稀疏注意力族 / 线性注意力 / 扩散解码)对"自循环"机制**不构成 CRITICAL 支撑**,已移入 [`../references/performance/inference-acceleration/inference-technologies/`](../references/performance/inference-acceleration/inference-technologies/) 作为通用性能参考。详见该目录的 README。

> **特别警示**:
> - **11 Gated DeltaNet** 和 **13 Qwen3-Next Hybrid** 因为线性注意力**破坏 KV cache 可寻址性**,反而**削弱**自循环所需的 prefix sharing。
> - **18 dLM**(离散扩散语言模型)与 AR 自循环范式互斥,如选 dLM 路线,自循环概念需重新定义。

---

## 🚀 推荐阅读顺序

| 读者 | 阅读顺序 |
|---|---|
| **自循环架构设计者** | 全部 3 份,顺序阅读 |
| **推理引擎集成者** | `01` → `02` → `09`(从基础到高级,从物理布局到量化压缩) |
| **性能优化者** | 先读本目录 3 份,再读 [`../references/performance/`](../references/performance/) 的推理加速专题 |

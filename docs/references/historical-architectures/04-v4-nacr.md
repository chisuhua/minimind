# v4 NACR — <2B 异构协作 + Engine Integration 审查

> **版本**：v4 (Nano-Agent Collaborative Reasoning)
> **评级**：C-（战略最清晰但工程风险最大）
> **核心失败模式**：用 2023 年 FrugalGPT/Cascade 走不通的失败模式在 2026 年用 <2B Specialist 重新包装

---

## 一、范式转移：从"中大模型组装"转向"<2B 异构协作"

| 维度 | v3（中大模型组装） | v4 NACR（<2B 异构协作） | 设计依据 |
|------|----------------|----------------------|---------|
| 推理主体 | 单模型 Long CoT | **<1B Router/Verifier + 1-2B Specialist** | 规避 Small Model Long CoT Degradation |
| 过程监督 | 训练专用 PRM | **推理引擎原生验证 (Code Exec / Regex / Schema)** | <2B PRM F1<50%，不可用 |
| 搜索策略 | MCTS / Best-of-N | **Training-free TTS (DIPA/LATTS) + Engine-Level Backtrack** | 零训练成本；利用 KV Cache 复用降低延迟 |
| 安全机制 | Phase 4 事后评估 | **Router 级 Guardrail + Output Sanitization** | 小模型无法内化 Safety |
| 协作模式 | 无 | **Pipeline / Parallel / Critic-Refiner** | 用架构换智能；用通信换准确率 |
| 适用边界 | 通用数学/代码 | **窄域高精度 + 通用路由** | 承认小模型跨域泛化失效 |

**战略定位**：**"不要试图让 <2B 模型变成一个笨拙的 o1-mini。让它成为一个高效协作系统中的专精组件。"**

---

## 二、6 项事实性错误

### 错误 1【致命】Speculative Decoding 在 <2B 上是**负加速**
**v4 声称**："Speculative Decoding `<1B draft → 1-2B verify` 吞吐 +2-3×"。

**实证（EACL 2026 "An Empirical Study of Speculative Decoding for Small Language Models"）**：

| 配置 | 加速比 | 含义 |
|------|--------|------|
| Qwen2.5-1.5B / Qwen2.5-0.5B（独立 drafting）| **0.83×** | **比无推测还慢** |
| SmolLM2-1.7B / SmolLM2-135M（独立 drafting）| **0.67×** | **比无推测慢 33%** |
| Qwen2.5-1.5B / EAGLE-2（self-draft）| 1.70× | 这才是真加速 |
| SmolLM2-1.7B / EAGLE-2 | 1.81× | 同上 |
| Llama-3.2-1B / EAGLE-2 | 1.44× | 同上 |

**根因**：draft time 16.6ms vs verify time 10.7ms（Qwen），需要 **77.5% accept rate** 才能保本——小模型几乎不可能。

**SpecDecode-Bench**：批大小 1 时 EAGLE 加速比 ≤1.96×，batch=128 时仅 1.21×。

### 错误 2【严重】DIPA / LATTS 引用错位
**v4 声称**："DIPA / LATTS 在 <2B 模型上延迟 -30-50%"。

**实证**：
- **DIPA**（ICLR 2026 under review）：是 test-time compute allocation bandit，**与"小→大切模型"协作无关**，只调整采样预算，**报告指标是"more problems solved under fixed budget"，不是延迟数字**
- **LATTS**（arXiv 2509.20368）：基于 **Llama-3.2-1B** + Qwen2.5-7B verifier，是同一小模型内的 step-level resampling/backtrack/restart，**不切换模型**。LATTS 报告的是"减少 token 数"，不是 wall-clock 延迟

### 错误 3【严重】KV Cache Rollback 开销 -80% **无出处**
**v4 声称**："Engine-Level KV Rollback → 回溯开销 -80%"。

**实证**：
- ❌ **vLLM / SGLang / TensorRT-LLM 均无原生 per-segment rollback API**
- SGLang `pause`/`resume`/`flush_cache` 是 RL 训练抽象，不是 per-segment rollback
- TensorRT-LLM "semantic KV cache reuse" 在 **RFC #14918** 阶段，**未实现**
- 主流粒度是 per-block（16-token block），**per-token snapshot + rollback 没有生产级引擎支持**
- "D-LLM KV 存储开销 -45%" 是 layer-skipping 的存储节省，**不是回溯开销**

### 错误 4【中等】Llama-Guard-3-1B 多语种 Recall 远低于 90%
**v4 声称**：Guardrail-as-a-Router 可"消除 90%+ 的 agentic misalignment 风险"。

**实证**（Llama-Guard-3 官方报告）：
| 数据集 | Llama-Guard-3-1B F1 | Llama-Guard-3-1B FPR |
|--------|---------------------|---------------------|
| 英文 | 0.899 | 0.090 |
| 法语 | 0.939 | 0.012 |
| 德语 | 0.845 | 0.036 |
| Hindi | **0.680** | 0.057 |

- **英文 F1 = 0.899**（接近但未达 90%）
- **Hindi F1 = 0.680**
- **Portuguese F1 = 0.763**
- 跨语种 FRR：医学/法律/性教育等边界话题 **10-25%**

### 错误 5【中等】"Math-Shepherd-1.5B 极简版"不存在
**v4 声称**："Math-Shepherd-1.5B 的极简版，专为 engine integration 设计"。

**实证**：
- ❌ **官方 Math-Shepherd 只有 7B 版本**（基于 LLaMA2-7B）
- Skywork-PRM-1.5B 存在，但**不是 Math-Shepherd 蒸馏**，是 Skywork 独立训练
- GenPRM-1.5B（AAAI 2026）用 23K MATH 数据训练，**比 Math-Shepherd-7B 在 ProcessBench 上显著更好**，但不是"Math-Shepherd 的极简版"

### 错误 6【中等】"小+大异构 ensemble vs 同尺寸投票"的对比无第一手数据
**v4 声称**："异构 ensemble（小+大）比固定策略 +5-10% 准确率"。

**实证**：
- FrugalGPT 是 cascade，**不是 voting**，且只在 5 个分类/QA benchmark 上验证
- LLM-Blender 主要用同尺寸开源 7B+ 模型
- 异构（小+大）vs 同尺寸多模型投票的 head-to-head 对比**无第一手数据**

---

## 三、3 个战略性失败模式

### 失败模式 1【致命】Router 分类错误是结构性失败模式
**问题**：Router 错误率约 5-15%，且不可观测。

| 错误类型 | 概率 | 后果 |
|----------|------|------|
| 完全无关的 specialist | 5-15% | 输出"看似专业但答非所问"——**最危险，用户难发现** |
| 部分相关但非最优 | 20-35% | 输出质量低但可接受 |
| 路由到通用 specialist | 10-20% | 退化为"普通模型" |

**根本问题**：Router 错误的失败模式**比单模型错误更隐蔽**。用户在 ChatGPT/Gemini 上最反感的"自信地胡说八道"在 v4 上**更严重**。

**Anthropic、OpenAI 内部报告**：这种"伪自信"是 NPS 下降的第一大原因。

### 失败模式 2【致命】生产化先例为零
| 案例 | 现状 |
|------|------|
| FrugalGPT (Stanford 2023) | 学术原型，2024 团队承认"production deployment at scale"未公开 |
| RouteLLM (Microsoft 2024) | 仅在内部 Bing Copilot 实验，未大规模生产化 |
| Cascade (Yue 2024) | 学术原型，无生产案例 |
| Notion AI (2023-2024) | 早期 cascade **6 个月内改回单模型** |
| Bing Chat (2023) | 早期多模型路由 **6 个月内回退到统一 GPT-4** |
| 阿里 DMR 报告 | 多模型级联系统维护成本是单模型 2-3x |

### 失败模式 3【严重】维护成本是单模型的 5-10×
| 维度 | 单模型 | 异构 v4 |
|------|--------|---------|
| 基础模型升级（如 Qwen2.5 → Qwen3）| 1× 人月 | **5-10× 人月** |
| 添加新 specialist | N/A | 2-4 人月 |
| 替换 specialist | N/A | 1-2 人月 |
| Incident 数量 | 1× | **4-6×** |
| 监控/调试工具链 | 成熟 | **需自建 3-6 人月** |

**在快速迭代的 LLM 领域（每 3-6 个月一次基础模型升级），v4 的"6 个独立模型生命周期"会成为迭代速度的枷锁**。

---

## 四、能力上限分析

### 4.1 <2B Specialist 真实能力天花板
| 任务 | <2B 模型典型上限 | 数据来源 |
|------|------------------|----------|
| GSM8K 一步算术应用题 | 35-55% | SmolLM2-1.7B: 54.4% |
| GSM8K hard / multi-step | 15-30% | Qwen2.5-1.5B: 30.4% |
| MATH (5-7 步形式化) | 5-15% | TinyLlama-1.1B: 4.5% |
| HumanEval pass@1 | 15-30% | DeepSeek-Coder-1.3B: 28.6% |
| 多步 Agentic Tool Use | 40-60% (单步)；20-35% (4 步以上) | READ 论文 2024 |

### 4.2 训练后的真实掉点风险
| 风险 | 量化数据 |
|------|----------|
| **灾难性遗忘** | 数学 SFT 1 epoch 后，MMLU 一般掉 5-15%；某些 <1B 模型掉 20%+ |
| **领域过拟合 (OOD 掉点)** | 同分布 acc 70% → OOD acc 30-40% |
| **Reasoning Integrity** | <2B 模型在多步推理中 50-69% 正确答案是"巧合答对" |
| **格式过拟合** | Tool-call JSON 训练 100% 格式合规，但同时**拒绝非工具调用** |

### 4.3 三个死结
**死结 1：能力-遗忘的零和**。1.5B 的容量根本装不下"领域知识 + 通用对话 + tool_call 协议 + safety 知识"四套不冲突的能力。

**死结 2：OOD 能力塌方**。v4 在分布内任务上接近 7B，但 OOD 上**比 7B 差得多**。

**死结 3：维护成本吞噬收益**。每加一个 specialist，故障点 ×1.5，维护成本 ×2。

---

## 五、延迟优势是"理论账"

| 指标 | 7B 单模型 (INT4) | 7B 单模型 (FP16) | v4 (3×<2B Parallel) | v4 (Pipeline) |
|------|-------------------|------------------|---------------------|---------------|
| P50 延迟 | 500ms | 1500ms | 600-1200ms | 2000-4000ms |
| 显存/请求 | 5GB (INT4) | 10GB | 6GB | 6GB |
| 能力上限 | 85% GSM8K | 88% | 75-80% | 同左 |
| OOD 鲁棒性 | 高 | 高 | **低** | **低** |
| 运维复杂度 | 1× | 1× | 3-5× | 5-10× |

**v4 在纯推理成本上有 30% 优势（仅在 Parallel Voting + 简单任务上），在延迟上仅在 Parallel 模式有微弱优势，在能力和鲁棒性上全面落后**。

---

## 六、整体评估

| 维度 | 评分 | 备注 |
|------|------|------|
| 战略眼光 | **A** | "用架构换智能"在 <2B 尺度下是合理战略 |
| 架构创新 | **C+** | 异构协作已有 FrugalGPT/Cascade 等学术原型 |
| 组件选型 | **C** | 6 项事实性错误 |
| 工程可行性 | **C-** | KV Rollback、跨模型 KV、Constrained Decoding 嵌套 >3 层都不成熟 |
| 生产化证据 | **F** | 业界没有成功先例 |
| 文档精度 | **B-** | 比 v3 更诚实承认不确定性 |

---

## 七、可借鉴的真实价值

虽然 v4 整体有结构性风险，但其中有几个洞察是**真正有价值**的：

1. **Engine-Native Verification** 是 <2B 的能力倍增器（Python REPL 让数学/代码从 65% → 95%+）
2. **Short-CoT SFT**（≤2K tokens）对小模型的方向正确（严禁 Long CoT）
3. **三层 Safety**（L1 Router + L2 Output + L3 HITL）是真实可用的安全策略
4. **"用架构换智能"在窄场景下确实成立**（内部代码 review、批量 SQL 生成、智能音箱指令）

但这些洞察**不需要"三模型异构协作"的复杂架构**——v4.5 用单模型 + Engine Integration 已经能实现。

---

## 八、对后续迭代的影响

v4 的失败直接催生了 v4.5 的"务实收敛"路线：
- 放弃异构协作主线 → 聚焦 1.5B 单模型
- 放弃 Router 决策 → L1/L2/L3 三层 Safety
- 放弃 KV Rollback / Speculative Decoding 等不成熟技术 → 仅保留 Engine-Native Verification
- 放弃"通用方案"定位 → 明确窄场景边界

---

## 九、引用

1. EACL 2026 "An Empirical Study of Speculative Decoding for Small Language Models"
2. DIPA (ICLR 2026 under review)
3. LATTS (arXiv 2509.20368)
4. FrugalGPT (arXiv 2305.05176)
5. RouteLLM (arXiv 2406.18665)
6. Cascade (arXiv 2401.10819)
7. Meta Llama-Guard-3 报告 (2024-12)
8. Anthropic Agentic Misalignment (Lynch et al. 2025)
9. OpenAI rStar (GitHub)
10. MiniMind README 中实际部署经验

---

## 十、一句话评价

**v4 是 v1→v2→v3→v4 迭代中战略最清晰、但工程风险最大的一次。它正确识别了 <2B 的能力边界并主动转向，但把 2023 年 FrugalGPT/Cascade 走不通的失败模式在 2026 年用 <2B Specialist 重新包装——Speculative Decoding 反向、DIPA/LATTS 引用错位、KV Rollback 无原生支持、生产化先例为零、维护成本是单模型的 5-10×。** 真正的 v4.5 应该是 v4 中"Short-CoT SFT + Engine-Native Verification + Constrained Decoding"的子集，放弃"Router + 异构协作"的主架构——这才有工程可行性。
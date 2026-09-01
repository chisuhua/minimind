# Test-Time Compute 研究

> **目录定位**:`docs/research/test-time-compute/` —— 围绕 **测试时计算（test-time compute）的自适应分配** 研究文档。
>
> **核心问题**:现有 test-time scaling 方法要么静态分配计算，要么从固定分布采样。能否联合适配"where 计算花费"与"how 生成进行"？
>
> **关联度**:⭐⭐⭐ **中** —— 直接关联 SOCA v3-Micro-Final 的 04 §1.2 W4 "LATTS-style 早停" 设计；为 architectures v4.5 LATTS-style 早停提供升级选项。

---

## 📁 论文清单（1 篇，2026 年）

| arXiv ID | 标题 | 作者 | v1 日期 | 模型规模 | 问题角度 |
|---|---|---|---|---|---|
| [2604.21018](https://arxiv.org/abs/2604.21018) | Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations | Bowen Zuo, Dongruo Zhou, Yinglun Zhu | 2026-04-22 | TBD | **联合适配**：where + how + 演化的 in-context demonstrations |

> ⚠️ **重要更正**：本论文在原 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md)（v1.7 之前）和 [`../../architectures/99-final-recommendation.md`](../../architectures/99-final-recommendation.md) §九（99 §九 #25）中被**错误标注为 "DIPA"**。2026-08-30 核验发现：
> - 实际标题为 **"Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations"**
> - "DIPA" 在 arxiv 全字段搜索无独立结果——**本文 ID 对应论文被误标**
> - 已在 99 §九 #25 与 00 §七 #24 同步修订为正确标题

---

## 🎯 关键发现

### 核心问题

现有 TTS 方法的两类局限：
1. **静态分配计算**：对所有 query 分配相同 compute，浪费在 easy queries
2. **从固定分布采样**：即使分配动态 compute，generation distribution 不变

### 方法（两阶段）

1. **Warm-up phase**：识别 easy queries，从**测试集自身**组装初始的 question-response pairs pool
2. **Adaptive phase**：把更多计算集中在 unresolved queries，并通过 **evolving in-context demonstrations** 重塑生成分布
   - 条件化每个生成于"语义相关问题的成功响应"而非"从固定分布重采样"

### 核心贡献

- **联合适配 where + how**（位置 + 分布）
- **利用测试集本身作为示范池**（无需额外标注）
- **通过语义相关 in-context demonstrations 进行上下文学习**
- **更少 inference compute 下超越 baselines**

### 实验

- 跨数学、代码、推理基准测试
- "consistently outperforms baselines while consuming substantially less inference-time compute"

---

## 📐 对 AgenticMind 项目借鉴

### 对 architectures 决策线（`../../architectures/`）

- **v4.5 LATTS-style 早停的细化可能**：04b §1.2 W4 用 "LATTS-style 早停"——本文提供**更精细的 TTS 方案**（联合适配 where + how），可作为 W4 任务的升级选项
- **训练-free TTS 的价值再确认**：99 §八 把"Efficient TTS"列为方向 #3——本文证明"在更少 compute 下超越 baselines"，**支持该方向**
- **DIPA 误标的反思**：本次核验暴露了 architectures 文档的"事实性错误"——v3/v4 已有 6 项此类错误，本次修订**清除其中 1 项**。剩余 5 项应在后续审计中清理

### 对 AgenticDSL 训练链路

- **AgenticDSL 训练中的"自适应示范"灵感**：本文方法"用同类 query 的成功 response 作 ICL demo"——**可直接应用于 AgenticDSL 的 few-shot prompt 工程**
- **"测试集即知识" 的洞见**：本工作用测试集自身的成功响应做演示池，**隐含"分布内数据是最好示范"的发现**——对 AgenticDSL 的 Agent loop 中的 ICL 设计有直接借鉴价值
- **HydraForge 验证器的 TTS 适配**：HydraForge 4 层验证器调用可能因任务复杂度差异巨大——本文方法可作为**动态分配验证器计算**的参考

### 对 SOCA v3-Micro-Final

- **04 §1.2 W4 LATTS-style 早停**：本文可作为 W4 设计的具体参考——**从"静态早停"升级为"动态 compute 分配 + 演化 in-context"**
- **SOCA 04 §9 24 项架构消融可加入"test-time compute adaptive" 维度**：新增消融项 "SOCA-A24b: LATTS-static vs Zuo-2026-adaptive"

---

## 📊 与项目决策的对应

| AGENTS.md 决策 | 本目录论文 | 行动建议 |
|---|---|---|
| **AgenticMind 推理服务优化** | #4 Zuo | 不只是 "Constrained Decoding"——可加 "evolving in-context demo" 提升 few-shot 效果 |
| **SOCA 04 §1.2 W4 LATTS 升级** | #4 Zuo | 用 evolving in-context 替代 static early-stopping |
| **fact-check SOP** | #4 Zuo（误标案例）| 每季度核验所有 arXiv ID——避免 DIPA 误标再次发生 |

---

## 🚀 推荐阅读顺序

| 读者 | 阅读路径 |
|---|---|
| **架构师 / 决策者** | Abstract → §3 Method（warm-up + adaptive）→ §4 Experiments |
| **AgenticDSL Agent loop 设计者** | Abstract → §3 Method（evolving in-context）→ §4 Experiments（跨任务泛化） |
| **SOCA 04 W4 设计者** | Abstract → §3 Method → §4 Experiments → 对照 04b §1.2 W4 |
| **架构文档核验者** | 本 README §"重要更正" → 99 §九 #25 修订 → 00 §七 #24 修订 |

---

## 📅 调研版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-30 | 初版，1 篇论文调研完成（原 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) §4 拆分）+ 修正"DIPA"误标 |

---

> **本目录定位**:作为 [`../soca/2026-arxiv-survey.md`](../soca/2026-arxiv-survey.md) 的**分主题深度版**——survey 提供全局视角与跨论文对比，本目录提供单篇论文的完整内容/创新/借鉴分析。

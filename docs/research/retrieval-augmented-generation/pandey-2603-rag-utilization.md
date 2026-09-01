# [2603.11513] Can Small Language Models Use What They Retrieve?

> **来源**:Pandey, S. (BITS Pilani). "Can Small Language Models Use What They Retrieve? An Empirical Study of Retrieval Utilization Across Model Scale." arXiv:2603.11513, v1 2026-03-12.
> **调研日期**:2026-08-30
> **调研者**:Sisyphus + Oracle 交叉核验
> **本文档归属**:`docs/research/retrieval-augmented-generation/`（RAG 问题领域）
> **关联文档**:`../soca/2026-arxiv-survey.md`（全局视角与跨论文对比）

---

## 1. 元信息

- **作者**：Sanchit Pandey (BITS Pilani, Hyderabad Campus, India)
- **arXiv**：v1 提交 2026-03-12 (78 KB)，10 页 + 5 figures，CC BY 4.0
- **配套资源**：代码与评估数据已发布（anonymous.4open.science/r/rag-utilization-study-C67F）；另有 Zenodo 早期版本
- **被引用状态**：本套架构文档（00 §四.5、06 §2.2、99 §九）均已引用并通过核验

---

## 2. 核心内容

### 2.1 研究问题

≤7B 的小语言模型能否有效利用 RAG 检索到的信息？

### 2.2 实验设计（控制变量非常严格）

- **模型**：5 个规模（SmolLM2-360M / Qwen2.5-1.5B / Qwen2.5-3B / Qwen2.5-7B / Llama-3.1-8B），3 个架构家族
- **检索条件**：4 种（无检索 / BM25 / Dense E5-large-v2 / Oracle）—— **Oracle 是关键创新**（保证段落含答案）
- **数据集**：NQ 500 题 + HotpotQA 500 题 = 1000 题；额外 PopQA 长尾（500K 语料 0% 命中率）
- **核心方法**：**Parametric Knowledge Split**（每个 (模型, 问题) 标记 Known/Unknown，分离利用失败 vs 检索失败）
- **语料**：50 万 Wikipedia 段落（≈ 2% 全量），100 词/段
- **评估**：EM 主指标 + F1 副指标 + 95% bootstrap CI + McNemar + Bonferroni 校正

### 2.3 三大核心结论

1. **Oracle 利用率**：即使 oracle 检索（保证答案在段落里），7B 模型对 Unknown 问题仅 14.6% EM；1.5B 仅 10.0%；360M 0.0%。**85-100% 检索工作被浪费**。
2. **Parametric 知识摧毁**：Known 问题无检索时 100% 正确，加入 oracle 后：7B 失 41.6pp、1.5B 失 57.0pp。**presence 而非 quality 驱动 distraction effect**。
3. **主导失败模式**：2588 例 oracle 失败分析，**61-100% 是 "irrelevant generation"**（模型完全忽略提供的上下文）。

### 2.4 净效应计算（表 5）

对所有模型 `ΔEM_net = p_unk × ΔEM_unk + p_kn × ΔEM_kn` **均为负**——RAG 在 ≤7B 上**净减分**。

---

## 3. 创新点

| 创新 | 与之前工作对比 |
|---|---|
| **Oracle 检索条件** | 之前工作（Lewis 2020、REALM）只在大模型 ≥10B 上测试，且未隔离利用 vs 检索 |
| **Parametric Knowledge Split** | 之前 Mallen 2023 在实体级别切分；本工作在 (模型, 问题) 对级别切分 |
| **5 模型 × 4 检索 × 2 数据集 × 3 prompt** 的控制矩阵 | 此前最大规模 RAG 评估无此严格控制 |
| **错误分类法（6 类）**：irrelevant generation (61-100%) / refusal / partial match / wrong entity / format / verbose correct | 此前仅按"对/错"二分 |
| **正交化提示词测试**：3 种 prompt × 3B，证实质量（context）≠ 干扰（distraction） | 这是核心洞见——oracle 与 noisy 检索的 distraction 不可区分 |

---

## 4. 对项目借鉴

### 4.1 对 SOCA v3-Micro-Final（`../../soca/`）

- **直接打击 SOCA 的 24 项架构消融**：SOCA 的"三区域架构 + 多维 MoE"假设部分基于"复杂结构能提升小模型利用检索上下文的能力"。本文证伪了这一假设——7B 以下 utilization 是**架构无关**的硬天花板，复杂的 SAE 总线/MoE 路由**无法突破**。
- **SOCA 消融结论应包含"retrieval distraction 评估"**：04 §九 24 项 SOCA-A0~A24 消融注册表中，应新增 **SOCA-A24b: "全架构 with RAG vs no-RAG 对照"**——若 SOCA 155M 也复现 Pandey 的 distraction 模式，则进一步支持架构无关 utilization 上限论。
- **CausalGate M3 的合理化**：SOCA §三 M3 的"replace/freeze/noise"模式中的 freeze 模式，恰好对应"抑制 retrieval context 利用"的实验范式——可作为 M3 干预实验的 baseline。

### 4.2 对 architectures 决策线（`../../../architectures/`）

- **直接支持 v4.5 "无检索 + Engine Verify"路线**：00 §三 教训 1"用架构换智能是幻觉"——本文给出**第一个 2026 年的实证背书**。架构文档可直接引用本文作为最新证据。
- **对 v4.6/v4.7 的反证**：04b §2.6 的 "Hallucination Amplification" 与本文的 distraction effect 高度同构——1.5B + RAG 不仅不帮忙反而摧毁 57pp 已知答案，**再次证明 04b 的"必须含 4 道防线"判断的紧迫性**。
- **F-01 基模型决策的实证支撑**：若 F-01 选定 Qwen2.5-1.5B（architectures 04b §1.2 默认 Specialist），本文数据表明 RAG 路线在该尺寸下**净损失 ~3pp**——**强烈建议绕开 RAG 路线，优先 v4.5 的 Engine-Native Verification 路线**。

### 4.3 对 AgenticDSL 训练链路（`../../../../AGENTS.md`）

- **AgenticDSL 选 7B 而非 1.5B 的实证支撑**：本文证明 ≤7B 模型利用 oracle 检索仅 14.6%，但 ≥7B 数据点缺失——AGENTS.md F-01 选 7B（vs 1.5B）正是为绕开 utilization 硬天花板，符合本文证据。
- **AgenticDSL 训练应包含 "no-RAG baseline"**：本文表明 RAG 在小模型上不净增分；AgenticDSL 的 4 层验证器本身就是 no-RAG 路线的增强——**进一步支持"符号验证 > 检索增强"的产品架构选择**。
- **HydraForge 4 层验证器的对照价值**：本文的 "distraction effect" 可作为 HydraForge L1 grammar 验证 + L2 signature 验证的**功能等价物**——通过形式化结构约束，绕开小模型"被无关上下文带偏"的失败模式。

---

## 5. 对 AgenticMind 项目整体启示

| 启示 | 落地动作 |
|---|---|
| **小模型 + RAG 不是免费午餐** | 任何 ≤3B 子项目的 RAG 集成**必须**先跑"Oracle 检索利用率"评估；不要假定 RAG 总是能加性能 |
| **模型规模仍是硬天花板** | F-01 决策若倾向小模型（0.5B/1.5B），应主动放弃 RAG 路线，转向 AgenticDSL 显式结构 + 形式化验证 |
| **检索质量 ≠ 利用能力** | F-02（若规划 AgenticDSL 训练中加入 retrieval）应拆为两个子目标：召回率 vs 利用率，**利用率是真正瓶颈** |
| **指令微调放大 distraction** | 06 §十四 已暗示的"instruction tuning 抑制 parametric knowledge"现象在本文得到独立验证（Observation 1 vs Table 1）——提示 AgenticDSL 训练流程需保留 base model 路径 |

---

## 6. 核心数据快查

| 模型 | None | Noisy Dense | Oracle | Known→Oracle 损失 |
|---|---|---|---|---|
| SmolLM2-360M | 100% Known | 0% | 0% | -100% |
| Qwen2.5-1.5B | 100% / 9% Known | 36.0% / 4.6% | 43.0% / 10.0% | -57.0% / -∞ |
| Qwen2.5-3B | 100% / 13.6% Known | 46.4% / 6.2% | 54.4% / 12.8% | -45.6% / -∞ |
| Qwen2.5-7B | 100% / 18.5% Known | 48.8% / 8.0% | 58.4% / 14.6% | -41.6% / -∞ |
| Llama-3.1-8B (FP16) | 24.0% EM | 17.5% | — | -6.5pp |

**净效应**：`ΔEM_net = p_unk × ΔEM_unk + p_kn × ΔEM_kn` 对所有 ≤7B 模型均为负。

# 2026 年 arXiv 论文综合调研报告（索引版）

> **调研日期**：2026-08-30
> **调研者**：Sisyphus（直读 arxiv.org 全文 + 交叉对照项目主线）+ Oracle 评审
> **v1.1 修订（2026-08-30）**：原 v1.0 单文件版（594 行）已拆分为 6 篇独立论文分析 + 5 个问题领域子目录，本文档降级为**索引与跨论文综合对比**

---

## 📋 文档定位

本文档是 [`../../architectures/` 8 月 30 日修订 StageB-#7 任务](.) 的延伸——在确认 6 篇 2026 arXiv ID 真实存在后，进一步展开每篇论文的内容摘要/创新点/项目借鉴分析。

**v1.1 拆分结构**（6 篇论文 → 5 个问题领域 → 6 个独立分析文件）：
- 每个问题领域有独立目录 + README（说明该领域问题 + 论文清单 + 跨论文对比 + 项目借鉴）
- 每篇论文有独立文件（内容摘要 + 创新点 + 对项目借鉴 + 核心数据快查）
- 本索引文档提供**全局视角 + 跨论文对比 + 项目决策矩阵**

---

## 0. 调研总览

| # | arXiv ID | 标题 | 作者/机构 | v1 日期 | 模型规模 | 与项目主线关联度 | 独立分析文件 | 归属目录 |
|---|---|---|---|---|---|---|---|---|
| 1 | [2603.11513](https://arxiv.org/abs/2603.11513) | Can Small Language Models Use What They Retrieve? | Sanchit Pandey, BITS Pilani | 2026-03-12 | 360M-8B | ⭐⭐⭐⭐⭐ **极高**——直接验证 SOCA/architectures 关键论断 | [`../retrieval-augmented-generation/pandey-2603-rag-utilization.md`](../retrieval-augmented-generation/pandey-2603-rag-utilization.md) | `retrieval-augmented-generation/` |
| 2 | [2604.01457](https://arxiv.org/abs/2604.01457) | Wired for Overconfidence | Tianyi Zhao et al., U. Virginia (COLM 2026) | 2026-04-01 (v3 2026-07-27) | 3B | ⭐⭐⭐⭐ **高**——SOCAMonitor/M15 + 元认知闭环置信度机制 | [`../confidence-calibration/zhao-2604-wired-overconfidence.md`](../confidence-calibration/zhao-2604-wired-overconfidence.md) | `confidence-calibration/` |
| 3 | [2606.00437](https://arxiv.org/abs/2606.00437) | EST-PRM | Ibne Farabi Shihab et al., Iowa State U. | 2026-05-30 | 7-8B PRMs | ⭐⭐⭐⭐ **高**——PRM 路线反证（直接支持 architectures 教训） | [`../reward-model-robustness/shihab-2606-est-prm.md`](../reward-model-robustness/shihab-2606-est-prm.md) | `reward-model-robustness/` |
| 4 | [2604.21018](https://arxiv.org/abs/2604.21018) | Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations（原误标"DIPA"） | Bowen Zuo et al. | 2026-04-22 | TBD | ⭐⭐⭐ **中**——AgenticDSL 训练-free TTS 设计参考 | [`../test-time-compute/zuo-2604-adaptive-test-time-compute.md`](../test-time-compute/zuo-2604-adaptive-test-time-compute.md) | `test-time-compute/` |
| 5 | [2605.22620](https://arxiv.org/abs/2605.22620) | Two is better than one: A Collapse-free Multi-Reward RLIF | Shourov Joarder et al. (BUET/WVU/UCL) | 2026-05-21 | 1.5B-3B | ⭐⭐⭐⭐ **高**——Multi-Reward RLIF 与 AgenticMind F-03 RLIF 路线强相关 | [`../reinforcement-learning/joarder-2605-multi-reward-rlif.md`](../reinforcement-learning/joarder-2605-multi-reward-rlif.md) | `reinforcement-learning/` |
| 6 | [2602.20091](https://arxiv.org/abs/2602.20091) | How Retrieved Context Shapes Internal Representations in RAG | Samuel Yeh, Sharon Li (UW-Madison) | 2026-02-23 (v2 2026-04-16) | 17B-80B | ⭐⭐⭐⭐ **高**——SOCA 隐表示可观测性的实证论据 | [`../retrieval-augmented-generation/yeh-2602-rag-representations.md`](../retrieval-augmented-generation/yeh-2602-rag-representations.md) | `retrieval-augmented-generation/` |

---

## 1. 跨论文综合对比（主题矩阵）

| 主题 | 论文 #1 Pandey | #2 Wired | #3 EST-PRM | #4 ATC | #5 RLIF | #6 RAG-Rep |
|---|---|---|---|---|---|---|
| RAG 利用 | ★ 主轴 | — | — | 部分 | — | ★ 主轴 |
| 置信度机制 | — | ★ 主轴 | — | — | — | 部分 |
| 训练稳定性 | — | — | ★ 主轴 | — | ★ 主轴 | — |
| Test-time compute | — | — | — | ★ 主轴 | — | — |
| 内部表示可观测性 | 部分 | — | — | — | — | ★ 主轴 |
| 跨数据集泛化 | ★ | ★ | ★ | ★ | ★ | ★ |
| 小模型 (≤3B) | ★ | — | 部分 | — | ★ | — |
| 大模型 (≥7B) | 部分 | — | ★ | — | — | ★ |
| 与 PRM 路线 | 反证 | — | ★ 反证 | — | 替代 | — |

**详细对比见各目录的 README：**
- [`../retrieval-augmented-generation/README.md`](../retrieval-augmented-generation/README.md) —— 论文 #1 + #6
- [`../confidence-calibration/README.md`](../confidence-calibration/README.md) —— 论文 #2
- [`../reward-model-robustness/README.md`](../reward-model-robustness/README.md) —— 论文 #3
- [`../test-time-compute/README.md`](../test-time-compute/README.md) —— 论文 #4
- [`../reinforcement-learning/README.md`](../reinforcement-learning/README.md) —— 论文 #5

---

## 2. 6 篇论文交叉揭示的 4 大共性主题

### 主题 A：模型规模仍是硬天花板

- **#1 Pandey**：≤7B 利用率 ≤14.6%，无 RAG 经常优于有 RAG
- **#2 Wired**：verbalized confidence 在 3B 已严重未校准
- **#5 RLIF**：1.5B 仍可达到 GSM8K 68%（vs 监督 75%）——但只能通过 multi-reward + KL-Cov

**结论**：F-01 决策不应倾向 ≤3B（除非配套 multi-reward RLIF 训练）。

### 主题 B：内部表示是诊断与干预的关键接口

- **#2 Wired**：circuit-level intervention 可校准 ECE（无需 retraining）
- **#6 RAG-Rep**：hidden states 编码 retrieval 相关信号
- **#3 EST-PRM**：PRM 失败的根源是**结构而非参数**

**结论**：AgenticMind 应在 HydraForge 4 层验证器之上加 "L5 representation verification" 层。

### 主题 C：互补多信号优于单一强信号

- **#5 RLIF**：multi-reward 比任一单 reward 都稳定
- **#3 EST-PRM**：5 个 PRM 漏洞画像各异，无单一 PRM 稳健
- **#1 Pandey**：4 种检索 × 5 个模型对比，无单一最优

**结论**：HydraForge 4 层验证器（grammar + signature + execution + task）已是 "互补多信号" 架构——架构选择正确。

### 主题 D：label-preserving 攻击是稳健性的真正测试

- **#3 EST-PRM**：保持答案不变的攻击比改变答案更难
- **#1 Pandey**：oracle 检索下 distraction 仍是问题

**结论**：HydraForge 验证器应在 test set 上做 EST-PRM 风格的 fuzzing。

---

## 3. 与项目决策矩阵

| 项目决策 | 论文依据 | 行动建议 |
|---|---|---|
| **F-01 基模型选择** | #1, #5（Pandey 利用率、RLIF 稳定性） | 1.5B 仅在 multi-reward RLIF 训练时可行；7B+ 更安全 |
| **F-03 RLIF 训练路线** | #5（Multi-Reward RLIF） | 采用 multi-reward + GDPO + KL-Cov 框架；避免单 reward |
| **F-04 不依赖 PRM** | #3（EST-PRM 漏洞） | 强化；HydraForge 4 层验证器路线优先于 PRM |
| **HydraForge 验证器设计** | #6（表示分析）+ #2（电路干预） | 加 L5 representation verification 层 |
| **v4.7 元认知闭环** | #2 + #5 | circuit-level ECE 校准 + multi-reward RLIF 训练 |
| **AgenticMind 产品置信度** | #2 | 不用 verbalized confidence；用 hidden state 模式 |

---

## 4. 调研结论

### 4.1 核心结论

1. **6 篇论文全部真实存在且标题与文档引用一致（除 #4 DIPA 误标已修订）**
2. **6 篇论文全部与 AgenticMind 项目主线有可执行借鉴**
3. **最关键借鉴**（按优先级）：
   - **#1 Pandey**：直接证明 ≤7B 模型利用 RAG 是净损失——**支持 F-01 选 ≥7B + 无 RAG 路线**
   - **#5 Multi-Reward RLIF**：直接提供 AgenticMind F-03 RLIF 路线的具体实施方案
   - **#3 EST-PRM**：直接支持 HydraForge 验证器路线优先于 PRM
   - **#2 Wired**：提供电路级 ECE 校准的具体方法
   - **#6 RAG-Rep**：提供内部表示作为诊断接口的科学依据
   - **#4 ATC**：通过 ICL demo 提供 few-shot prompt 工程启示

### 4.2 调研方法论局限

- **6 篇论文均为 2026 年新文**，未来可能有大版本更新
- **未读的论文细节**：详见各目录 README 的"调研方法论局限"小节
- **DIPA 误标已修订**：本文已记录该事实性错误的清除过程

### 4.3 后续建议

1. **重新审计架构文档剩余 5 项事实性错误**（v3/v4 各 6 项，本次修了 1 项）
2. **将本报告内容同步到 AGENTS.md**：在 §3 自循环机制或 §4 训练路线中加入"2026 SOTA 借鉴"段落
3. **将 #5 Multi-Reward RLIF 实施到 F-03 路线**：作为 AgenticMind RLIF 训练的具体技术栈
4. **将 #2 circuit-level calibration 实施到 HydraForge**：作为 L5 验证器层的备选技术
5. **定期复核引用真实性**：每季度对架构文档中所有 arXiv ID 做一次核验

---

## 5. 调研产出（v1.1 拆分结构）

### 5.1 5 个问题领域子目录

```
docs/research/
├── soca/ (existing - SOCA v3-Micro)
├── retrieval-augmented-generation/   [NEW] 论文 #1, #6
│   ├── README.md
│   ├── pandey-2603-rag-utilization.md
│   └── yeh-2602-rag-representations.md
├── confidence-calibration/           [NEW] 论文 #2
│   ├── README.md
│   └── zhao-2604-wired-overconfidence.md
├── reward-model-robustness/           [NEW] 论文 #3
│   ├── README.md
│   └── shihab-2606-est-prm.md
├── test-time-compute/                 [NEW] 论文 #4
│   ├── README.md
│   └── zuo-2604-adaptive-test-time-compute.md
└── reinforcement-learning/            [NEW] 论文 #5
    ├── README.md
    └── joarder-2605-multi-reward-rlif.md
```

### 5.2 文件大小与定位

| 文件 | 大小 | 定位 |
|---|---|---|
| 本文档（soca/2026-arxiv-survey.md） | 索引版 | 全局视角 + 跨论文对比 + 决策矩阵 |
| 各目录 README（5 份） | 200-300 行 | 该领域问题 + 论文清单 + 跨论文对比 + 项目借鉴 |
| 各论文分析文件（6 份） | 150-250 行 | 单篇论文完整内容摘要 + 创新点 + 对项目借鉴 + 核心数据 |

### 5.3 阅读路径

| 读者 | 阅读路径 |
|---|---|
| **架构师 / 决策者** | 本文档 §1-§3（跨论文对比）→ 各目录 README（领域问题）→ 各论文分析（具体借鉴） |
| **算法工程师（AgenticDSL 训练）** | `../reinforcement-learning/` + `../confidence-calibration/` |
| **HydraForge 验证器设计者** | `../reward-model-robustness/` + `../retrieval-augmented-generation/README.md` |
| **SOCA M15 SOCAMonitor 设计者** | `../confidence-calibration/` + `../retrieval-augmented-generation/yeh-2602-rag-representations.md` |

---

## 6. 关联文档

- 上游：[`../../architectures/00-iteration-timeline.md`](../../architectures/00-iteration-timeline.md) §七（引用清单）
- 上游：[`../../architectures/04b-v4.5-and-v4.6.md`](../../architectures/04b-v4.5-and-v4.6.md) §五
- 上游：[`../../architectures/06-metacognitive-closed-loop.md`](../../architectures/06-metacognitive-closed-loop.md) §四 §十一
- 上游：[`../../architectures/99-final-recommendation.md`](../../architectures/99-final-recommendation.md) §九
- 平级：[`../../AGENTS.md`](../../AGENTS.md) §3 自循环机制 / §4 训练路线 / §5 F-01
- 平级：[`../soca/`](.)（SOCA v3-Micro-Final 研究）

---

## 7. 附录：v1.0 → v1.1 修订记录

### v1.0（2026-08-30 上午）

- 单文件版（594 行）
- 含 9 节：调研总览 + 6 篇论文逐一分析（每篇 5 子节）+ 跨论文综合对比 + 调研结论 + 附录
- 核验 + DIPA 误标修订

### v1.1（2026-08-30 下午）

- 拆分结构：将 594 行单文件拆为 **5 个问题领域目录 + 6 篇独立论文分析 + 1 个索引文档（本文件）**
- 每个领域目录含 README（领域定位 + 论文清单 + 跨论文对比 + 项目借鉴）
- 每篇论文含完整分析（元信息 + 内容摘要 + 创新点 + 对项目借鉴 + 核心数据快查）
- 本索引文档保留全局视角与跨论文对比，便于决策者快速浏览

**修订理由**：
- 单文件 594 行过大，单篇论文细节淹没在跨论文对比中
- 按"要解决的问题"分类后，每个领域有独立 README 便于专项深读
- 论文分析文件独立后，可单独引用而不必引用整篇 survey
- 索引文档保持全局视角，避免读者反复跳转

---

## 8. 文档版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-30 上午 | 初版单文件（594 行），含 6 篇论文全部内容 |
| v1.1 | 2026-08-30 下午 | **拆分结构**：5 个问题领域子目录 + 6 篇独立论文分析 + 本索引文档 |

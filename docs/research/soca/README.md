# agenticsom · SOCA v3-Micro 验证模型研究

> **目录定位**:`docs/research/agenticsom/` —— 围绕 **SOCA v3-Micro 验证模型** 的设计审查、参数量甜点、层数甜点、工程实现路线图、通用预训练消融计划、核心模块规格、模块化设计扩展、路由器训练配方、SAE/总线信息论九份研究文档。
>
> **目的**:在 <1B 激活参数约束下,为 AgenticMind 项目寻找一个能"以最小代价验证关键架构假设"的验证模型。
>
> **关联度**:⭐ **RESEARCH(研究性,非工程主线)** —— 这些文档是**设计前期调研**,不是当前训练/部署路径;但其结论会直接影响后续 0.8B-1.2B 正式模型的架构决策。

---

## 📁 文档结构

> **⭐ 机器可读参数真源**:`./soca_micro_final_config.yaml` (v1.0, 2026-08-28)——`03-sweet-spot-layers.md` §五.6 的 YAML 副本,包含 SOCAMicroConfig + SOCAConfig 双视角字段、派生量、注意力分布、消融引用、信息论引用、实施路径。**代码实现建议从此文件加载,杜绝文档间抄数字漂移**。

| 文档 | 关联度 | 内容摘要 |
|---|---|---|
| ⭐ [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) | REVIEW | SOCA v3-Micro(14 层、~106M)**全面审查报告**——三视角审查:原生可解释性(J-Space × J-Lens × SAE)、SOTA 架构匹配(Kimi K3 / DeepSeek V4 / MAGI-2 等 7 个 2026 SOTA 对标)、多维 MoE 完整性;识别 12 个审查问题 + 8 项修正,得到 v3-Micro-REV 配置 |
| ⭐ [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) | ANALYSIS | **参数量甜点评估**——从 106M → 148M(等效 ~150M),多约束优化分析(训练时间 / 效应量 / 组件最小容量 / 验证信噪比 / 扩展可预测性);结论:130M-170M 是甜点区间,**~148M 为最优点**(消融显著性 87%、统计功效 91%) |
| ⭐ [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) | ANALYSIS | **层数甜点评估**——14 / 16 / 18 / 20 层联合优化;结论:**16 层(物理)/ 18 层(等效) × ~155M** 是联合甜点;相比 14 层,消融显著性从 87% → 96%;相比 18 层,避免 16% 额外成本 |
| ⭐⭐ [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) | IMPLEMENTATION | **工程实现路线图与代码骨架**——把 03 锁定的 16 层 × ~155M 配置落地为代码:项目结构、`config.py`、MonitorSlot+广播总线、Gated DeltaNet、Linear Attention、4 种 MoE(Soft/MH/Fine/LoRA)、Joint SAE、WorkspaceMid 循环、完整模型组装、多损失联合训练(4 阶段调度)、**24 个 SOCA 架构消融注册表**、8 个工程陷阱与解决方案、4 周里程碑计划 |
| ⭐ [`05-pretraining-ablation-plan.md`](./05-pretraining-ablation-plan.md) | PLAN | **24 项通用预训练消融详细执行计划**——与 04 的 SOCA 架构消融(`SOCA-A0` ~ `SOCA-A24`)**互补**,共同构成 **48 项完整消融体系**(SOCA 架构 24 + 通用预训练 24,通用侧 SOCA 适配后实做 **22 项**);6 类 × 4 项(数据 `GENERAL-A1-A4` / 架构 `GENERAL-B1-B4` / 训练策略 `GENERAL-C1-C4` / 正则化 `GENERAL-D1-D4` / 优化器 `GENERAL-E1-E4` / 基础设施 `GENERAL-F1-F4`);包含 Phase 1-6 执行顺序、依赖关系、资源估算(~6 天 / 155M 适配)、评估指标加权、判定阈值与决策树、SOCA 特定调整说明 |
| ⭐⭐ [`06-architecture-modules.md`](./06-architecture-modules.md) | SPEC | **SOCA 核心模块规格 M1-M16**——可干预/可观测/可推理的 16 个核心模块,分为 6 类(全局基础设施/计算区/区间接口/工作空间内部/注意力变体/运行时系统);关键创新:M3 CausalGate(4 种模式:normal/replace/freeze/noise)、M2 MonitorSlot(3 种模式:OBSERVE/INJECT/BYPASS)、M1 每层独立 read/write gates、M11 GatedAggregator(跨专家交互注意力)、M6 微专家 + 软门控(替代 LoRA);为 04 的代码提供精确契约 |
| ⭐⭐ [`07-module-extensions.md`](./07-module-extensions.md) | SPEC | **SOCA 完整模块规格扩展 M17-M24**——在 06 基础上补充 8 个辅助子模块(M17 RMSNorm / M18 RoPE / M19 PhaseScheduler / M20 ExpertDispatcher / M21 HealthDashboard / M22 SnapshotRecorder / M23 SOCAConfig / M24 SOCAModel);**关键发现:SOCA 框架是规模无关的**——`SOCAConfig` 支撑 **4 个规模预设**:`config_tiny`(~20M,概念验证)/ **`config_micro`(~155M,SOCA v3-Micro-Final 验证模型)/** `config_medium`(~7B)/ `config_large`(~120B,生产);含 5 阶段 12 周实施优先级(最小可运行→监控→训练→可解释→生产化) |
| ⭐⭐⭐ [`08-router-stability-training.md`](./08-router-stability-training.md) | TRAINING | **路由器稳定性与全局训练策略**——SOCA 训练的"命门"(M9 LatentRouter 在隐空间路由 + 联合 SAE 耦合 + 总线写入依赖三重重压下):**F1-F7 失效模式全谱系**(专家坍缩/熵坍缩/隐空间坍缩/路由震荡/门控震荡/路由退化/SAE-路由冲突);**七重防护**(初始化策略→渐进式路由启用→自适应温度 PID→多目标梯度平衡→专家容量保护→隐空间健康监控→运行时自愈);完整训练循环 + 三档规模超参速查表 |
| ⭐⭐⭐ [`09-sae-bus-information-theory.md`](./09-sae-bus-information-theory.md) | THEORY | **联合 SAE 字典学习动态 与 广播总线信息论性质**——两部分理论深化:①M12 JointSAE:联合 vs 事后训练的"表征塑造器"本质、字典共演化动力学方程、4 阶段生命周期(诞生/特化/稳态/衰退重生)、稀疏度-重建权衡、字典-路由器耦合与 detach 解耦、4 阶段 SAE 训练配方;②M1 BroadcastBus:信息瓶颈/信道容量/信息流三阶段/充分统计量(>0.8)/率失真/全局工作空间理论(GWT)映射/异常检测信息论基础/开销-精度权衡 |

---

## 🎯 一句话目标

> **SOCA(Self-Organizing Cognitive Architecture) v3-Micro 验证模型** —— 在 **16 层 / ~155M 参数** 内,通过 **三区域架构(感知 × 工作空间 × 动作) × 三维 MoE(激活方式 × 计算空间 × 专家粒度) × 三视角可解释性(J-Space × J-Lens × SAE)**,把 0.8B-1.2B 正式模型的关键架构假设提前压缩验证。

---

## 🎯 消费者契约(消融结论的预期去向)

> ⚠️ **本节为 SOCA 与项目其他部分的接口契约**——明确 SOCA 消融结论的预期消费者、对应的决策点,以及在主决策翻转时的降级路径。**修订 2026-08-28**(基于 [`docs/architectures/` 融合分析 Oracle 评审](#))。

### 当前 AGENTS.md F-01 决策状态

[`../../AGENTS.md` §12.10 F-01](../../AGENTS.md):**基模型选择 = Qwen2.5-Coder-7B-Instruct(现成权重,非自训)**。

这意味着 AgenticMind 训练链路**不自训架构**——直接消费上游现成模型 + AgenticDSL SFT/RL 训练。

### SOCA 消融结论的消费路径

| 路径 | 触发条件 | SOCA 作用 | 当前状态 |
|---|---|---|---|
| **A. architectures/99 决策输入** | AGENTS.md F-01 翻转至<1B 自训架构 | SOCA 24 项架构消融(SOCA-A0~A24)为自训架构的模块选型提供实证 | ⚠️ **悬空**(v4.5 用现成 Qwen,不消费 155M 消融) |
| **B. architectures/06 v4.7 内层 hook** | 元认知闭环 v4.7 启动 + CausalGate 验证通过 | M3 CausalGate 作为 per-step hook;M15 SOCAMonitor 作为置信度辅助 | ⚠️ **条件性**(v4.7 自身处"4 周未达 KPI 立即降级"缓刑) |
| **C. architectures/00 教训库** | 任何 SOCA 消融产出实测数据时 | 阴性结果("复杂架构在 155M 无显著增益")是 architectures 教训 1("用架构换智能是幻觉")的**最强支撑** | ✅ **有效路径**(无需依赖 F-01 翻转) |

### 若 F-01 永不翻转的降级路径

**条件**:若 AGENTS.md F-01 在未来 12 个月内**永不翻转至自训<1B 架构**,SOCA 文档集应主动降级:

```
🔴 降级执行:
  - 取消 README §"一句话目标" 中的"前置压缩验证"叙事
  - 全部里程碑 4-7 状态从 🔨/⏳ 改为 ⛔ (研究性归档,不再追求实施)
  - 03 §十 决策落定表锁定项加上"⚠️ 暂无消费者"标注
  - 04/06/08 中的"为正式模型服务"措辞改为"为研究档案保留"
  - 04 §十二 4 周里程碑改为 ⛔ 状态
```

**触发动作**:任一 SOCA 维护者发现 F-12 个月内无翻转迹象 → 启动降级 PR。

### 与 `docs/architectures/` 的真实关系

**不是**"互补的工程方案";**而是**"**研究假设验证器 vs 工程决策收敛点**"。

- SOCA 的价值 = 提供"哪些架构组件在小模型上有/无价值"的**实证输入**
- architectures v4.5/v4.6 的价值 = 工程决策收敛点
- 两者关系 = **文档层面的相互引用**(本 README §📐 已声明),不应被工程化为代码集成,除非上述三个消费路径至少一个被激活

---

## 🧩 SOCA v3-Micro 核心机制速览

### 三区域架构(4+6+4 → 5+6+5)

```
Token Input
   ↓
┌─────────────────────────────────┐
│  PERCEPTION (4-5 层,Dense)     │ ← 1 Std + 3 Gated DeltaNet + 1 Gated Std
│  L1 稀疏惩罚 λ=0.001            │   (或 5 层版本:1 Std + 3 DeltaNet + 1 Gated Std)
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│  WORKSPACE FRONT (2 层)         │ ← Soft × Standard × Coarse
│  dispatch: sigmoid(J-Lens 友好) │   4 slots × 4 experts
└─────────────────────────────────┘
   ↓ Down Interface (d_model → d_bus=256)
┌─────────────────────────────────┐
│  WORKSPACE MID ★ (2 层 × 2 循环)│ ← Sparse × Latent × Multi-Head
│  4h × 12 experts/head = 48 专家 │   组合空间 ~(C₁₂⁴)⁴ ≈ 6×10¹⁰
│  Joint SAE: dict=2048, K=12     │   等效 4 层,有 Gate₁ + Gate₂
└─────────────────────────────────┘
   ↓ Up Interface (d_bus → d_model)
┌─────────────────────────────────┐
│  WORKSPACE BACK (2 层)          │ ← Sparse × Standard × Fine
│  Shared + Routed(20, Top-5)     │   组合空间 C₂₀⁵ = 15504
│  + Device                        │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│  ACTION (4-5 层,Dense × MH-LoRA)│ ← 3 DeltaNet + 2 Gated Std
│  LoRA-MoE: 2h × 10 LoRA, rank=8 │   FFN 1.8× (1612)
└─────────────────────────────────┘
   ↓
OUTPUT HEAD: 32000 (tied)
```

### 三维 MoE 覆盖(5 种组合)

| 区域 | 激活方式 | 计算空间 | 专家粒度 |
|---|---|---|---|
| 感知区 | Dense | Standard | N/A |
| 工作空间前段 | Soft | Standard | Coarse(4 slots) |
| 工作空间中★段 | Sparse | Latent | Multi-Head(4h×12e) |
| 工作空间后段 | Sparse | Standard | Fine(20 routed + 1 shared + 1 device) |
| 动作区 | Dense | Standard | Multi-Head LoRA(2h×10) |

### 三视角可解释性

| 视角 | 验证目标 | 关键设计 |
|---|---|---|
| **J-Space** | 感知→工作空间→动作 三区边界清晰 | Down/Up Interface + 注意力类型切换 |
| **J-Lens** | 中间层→输出 映射近似线性 | 中段使用 Linear Attention(无 softmax) |
| **SAE** | 激活空间天然稀疏 | 三段 SAE(中段 dict=2048 + 前后段 dict=1024) |

---

## 📊 九份研究文档的结论链路

```
01-v3-micro-14l-review.md
  ├── 在 14 层 / 106M 基础上,识别 12 个审查问题
  ├── 修正后 v3-Micro-REV:14 层 / ~106M
  └── 开放问题:106M 是否足以验证关键架构假设?
         │
         ▼
02-sweet-spot-params.md
  ├── 106M → 150M 能解决什么?哪些不能解决?
  ├── 多约束优化:130M-170M 是甜点区间
  ├── 最终推荐:~148M(消融显著性 87%、统计功效 91%)
  └── 开放问题:14 层是否最优?消融效应是否过大?
         │
         ▼
03-sweet-spot-layers.md
  ├── 14 / 16 / 18 / 20 层联合优化
  ├── 多目标评分:16 层得分 8.4(最高)
  ├── 最终推荐:16 层(5P+6W+5A) × ~155M
  └── 消融显著性从 87% → 96%,边际收益最优
         │
         ▼
04-implementation-roadmap.md
  ├── 16 层 × ~155M 配置 → 代码骨架
  ├── 4 周里程碑计划(烟雾测试 → 主训练 → 24 消融 → 分析)
  ├── 8 个工程陷阱与解决方案
  └── 24 个 SOCA 架构消融配置注册表(`SOCA-A0` ~ `SOCA-A24`)
         │
         ▼
05-pretraining-ablation-plan.md
  ├── 与 04 的 `SOCA-A0`~`A24` 互补:通用预训练组件消融(`GENERAL-A1`~`F4`)
  ├── 6 类 × 4 项(数据/架构/训练策略/正则化/优化器/基础设施)
  ├── Phase 1-6 执行顺序 + 资源估算 + 判定决策树
  └── SOCA v3-Micro-Final 调整后:22 项通用消融
         │
         ▼
06-architecture-modules.md
  ├── SOCA 核心模块规格:M1-M16(可干预/可观测/可推理)
  ├── 6 类(全局基础设施/计算区/区间接口/工作空间内部/注意力变体/运行时系统)
  ├── 关键创新:M3 CausalGate(运行时干预)、M2 MonitorSlot(三种模式)
  └── 为 04 的代码骨架提供精确契约
         │
         ▼
07-module-extensions.md
  ├── 辅助子模块:M17-M24(归一化/位置编码/阶段管理/调度/可视化/配置/组装)
  ├── SOCAConfig + **4 个规模预设**:tiny(~20M)/ **micro(~155M)**/ medium(~7B)/ large(~120B)
  ├── 关键洞察:SOCA 框架规模无关(20M → 120B 共 6 个数量级)
  └── 5 阶段 12 周实施优先级(最小可运行→监控→训练→可解释→生产)
         │
         ▼
08-router-stability-training.md
  ├── M9 LatentRouter 深化:失效模式 F1-F7 全谱系
  ├── 七重防护:初始化→渐进启用→自适应温度→梯度平衡→容量保护→隐空间监控→自愈
  ├── 完整训练循环(与 M19 PhaseScheduler 协作)
  └── 三档规模超参速查表(Tiny/Medium/Large)
         │
         ▼
09-sae-bus-information-theory.md
  ├── M12 JointSAE 深化:字典共演化动力学 + 4 阶段生命周期 + 4 阶段训练配方
  ├── M1 BroadcastBus 深化:信息瓶颈/信道容量/充分性/率失真/GWT 映射
  ├── 字典-路由器耦合(detach 解耦,补全 08 的 F7)
  └── 总线异常检测的信息论基础(为什么充分)
         │
         ▼
★ 完整消融体系 = 04 的 24 项 SOCA 架构(`SOCA-A0~A24`)+ 05 的 22 项通用预训练(`GENERAL-A1~F4` 中 SOCA 适配后实做 22 项) = **46 项**
★ 完整模块规格 = 06 的 M1-M16 核心 + 07 的 M17-M24 辅助 = 24 个模块
★ 完整规模覆盖 = 07 的 **4 个预设**(20M/**155M**/7B/120B)
★ 完整训练配方 = 08 路由器七重防护 + 09 SAE/总线理论
★ 实施:沿 MiniMind 训练链路,在 PyTorch 中原生实现
```

---

## 🎯 最终推荐配置(SOCA v3-Micro-Final)

| 维度 | 推荐值 | 关键参数 |
|---|---|---|
| **物理层数** | **16** | 5P + 6W + 5A |
| **等效层数** | **18** | 中段 2×2=4 |
| **总参数** | **~155M** | 上下限 130M-170M |
| **d_model** | 896 | 14 heads × 64 d_head |
| **d_bus** | 256 | 4 heads × 64 d_head |
| **n_vocab** | 32000 | tied embedding |
| **训练数据** | 6B token | 5B→6B(更大模型需更多数据) |
| **训练时间** | ~19h(8×A100-80G) | 主模型单次训练 |
| **消融时间** | ~4.5 天(4 并行) | 24 个消融实验 |
| **统计功效** | 94% | 1 次训练即可得出可靠结论 |
| **消融显著性** | 96%(23/24) | 仅 1 个边缘显著 |

### 区域配置

| 区域 | 层数 | 注意力组成 | 专家配置 |
|---|---|---|---|
| 感知区 | 5 | 1 Std + 3 Gated DeltaNet + 1 Gated Std | Dense + L1(λ=0.001) |
| 工作空间前段 | 2 | Gated Std Attn + Soft MoE | 4 slots × 4 experts, sigmoid dispatch |
| 工作空间中★段 | 2×2 循环 | Linear Attn + MH-MoE | 4h × 12e/head, Top-4/head |
| 工作空间后段 | 2 | Gated Std Attn + Sparse MoE | Shared(1) + Routed(20, Top-5) + Device(1) |
| 动作区 | 5 | 3 Gated DeltaNet + 2 Gated Std | Dense × MH-LoRA(2h × 10 LoRA, rank=8) |

### 训练超参(相比 106M 版调整)

| 参数 | 106M 版 | **155M 版** | 原因 |
|---|---|---|---|
| 总训练 token | 5B | **6B** | 更大模型需更多数据 |
| 学习率 | 3e-4 | **2.5e-4** | 更大模型用更小 LR |
| Warmup 步数 | 2000 | **2500** | 更稳定预热 |
| Batch size | 256×2048 | **256×2048** | 不变 |
| 总步数 | ~9500 | **~11500** | 6B/524K |
| 消融训练步数 | 6000 | **7000** | 消融也需更多步 |

---

## 🔬 与 SOTA 2026 架构的对标

| 模型 | 时间 | 核心架构特征 | 与 SOCA v3-Micro 的关联 |
|---|---|---|---|
| **Kimi K3** | 2026.07 | 2.8T 参数,896 专家激活 16,**KDA + Gated MLA 交替**,Latent MoE | 混合注意力 + Latent MoE |
| **DeepSeek V4** | 2026.04 | 1.6T/49B 激活,**mHC + Engram + DSA**,MoE | 条件记忆 + 稀疏注意力 |
| **MAGI-2 Preview** | 2026.08 | 114B/6B 激活,**12 heads × 256 experts/head, Top-6/head = 72 激活** | Multi-Head MoE 极致 |
| **Qwen3.5** | 2026.02 | **Gated DeltaNet + Gated Attention 3:1 混合**,MoE | 混合注意力比例 |
| **OpenAI Circuit Sparsity** | 2025.12 | 0.4B,**99.9% 权重为零**,电路级可解释 | 原生可解释性标杆 |
| **Gemma 4** | 2026.04 | 31B Dense,**5:1 局部:全局注意力** | 注意力比例设计 |
| **GLM-5** | 2026 | 744B/40B 激活,**MLA + 稀疏注意力** | 潜空间注意力 |

**关键匹配**:
- SOCA 中段 **Multi-Head MoE(4h × 12e)** 与 MAGI-2 同构(MAGI-2 是放大版)
- SOCA 中段 **Latent MoE** 与 Kimi K3 同思路
- SOCA 感知区/动作区 **Gated DeltaNet + Gated Std** 与 Qwen3.5 一致
- SOCA **三视角可解释性**比 Circuit Sparsity 更结构化,但 Circuit Sparsity 提供"权重稀疏"消融对照

**扩展性预测**:
- 16 层 → 22 层(0.8B)= 1.375× ✅
- 16 层 → 26 层(1.2B)= 1.625× ✅
- 155M → 800M = 5.2× ✅
- 155M → 1200M = 7.7× ✅

均处于可靠外推区间,验证结论可直接外推到正式模型设计。

---

## 📐 与项目现有架构体系的关系

| 项目现有架构 | 与 SOCA v3-Micro 的关系 |
|---|---|
| [`docs/architectures/00-iteration-timeline.md`](../../architectures/00-iteration-timeline.md) | **前置**:7 轮推理架构迭代史(v1 → v4.6 → AGI → 元认知闭环),SOCA 是其后的探索方向 |
| [`docs/architectures/06-metacognitive-closed-loop.md`](../../architectures/06-metacognitive-closed-loop.md) | **互补**:元认知闭环(运行时)vs SOCA(架构本体);前者研究"自循环机制",后者研究"自循环的承载架构" |
| [`docs/architectures/99-final-recommendation.md`](../../architectures/99-final-recommendation.md) | **决策依据**:v4.5/v4.6 务实收敛路线(64M-198M 极小模型)为 SOCA 验证模型提供了"小模型必要性"的基础 |

**关键定位差异**:
- AgenticMind 主线架构(`architectures/`)关注 **64M-198M 极小模型 + 运行时验证**
- SOCA v3-Micro(`research/agenticsom/`)关注 **~155M 验证模型 + 原生可解释性**

两条路径不冲突,但解决的问题不同:AgenticMind 主线是"工程务实路线",SOCA 是"架构假设验证路线"。

---

## ⚠️ 基线与外推边界(修订 2026-08-28)

> **本节为 SOCA 结论的适用范围声明——防止 SOCA 数字被错误外推到 architectures 线**。

| 维度 | SOCA v3-Micro-Final | architectures/99 v4.5/v4.6 | AgenticDSL 训练链路(AGENTS.md F-01) |
|---|---|---|---|
| 模型规模 | **155M(自训)** | **1.5B Qwen2.5(现成权重)** | **7B Qwen2.5-Coder-Instruct(现成权重)** |
| 训练数据 | MiniMind 6B token 自训 | Qwen 自带 + MiCoTA SFT | Qwen 自带 + AgenticDSL SFT |
| 训练目标 | 三区域架构假设验证 + 可解释性监控 | 结构化任务能力 + 工程引擎集成 | AgenticDSL 生成与执行 |
| 评估指标 | J-Space/J-Lens/SAE 监控信号 | GSM8K / MATH / HumanEval / P99 / $0.0003/query | AgenticDSL 格式合规 / 任务成功率 / 拒答率 |
| 架构干预 | **零工程集成**(纯研究验证) | 三层 Safety / Engine Verify / Constrained Decoding | 4 层验证器(grammar + signature + execution + task) |
| 维护承诺 | 6 天一次性训练 | 持续 1.5B 生产 | 持续 7B 生产 |

### 边界声明

1. **SOCA 文档集中的消融结论**(96% 显著性、94% 统计功效、充分性 >0.8、24 项 SOCA-A0~A24、22 项 GENERAL-A1~F4)**仅适用于 SOCA 训练链路**(155M + MiniMind 数据 + 自训)。

2. **禁止跨线引用**:任何 "v3-Micro 消融支持/反对 architectures 某结论" 的主张,**必须在目标基线上重测**。例如:
   - ❌ "SOCA 155M 上 SAE 无显著增益" → 推不出 "Qwen2.5-1.5B 上 SAE 也无显著增益"
   - ❌ "SOCA 155M 上 Multi-Head MoE 提升 8%" → 推不出 "Qwen2.5-1.5B 上 Multi-Head MoE 也提升 8%"
   - ✅ "SOCA 155M 上 SAE 无显著增益" → 仅指 "SOCA 训练链路下,155M 模型 SAE 监控信号未超过 [阈值]"

3. **规模外推边界**(03 §九 + 修订 2026-08-28):
   - SOCA 文档建议的外推:155M → 0.8B-1.2B 范围,**外推比 5.2×-7.7×(在文献安全区间内,但 SOCA 未实测)**
   - **禁止外推**:155M → 64M(向下)/ 7B+(向上)
   - **禁止跨模型族外推**:SOCA 训练链路(MiniMind 6B 数据)→ Qwen2.5-Coder-7B(完全不同训练分布)

4. **监督数据来源**(对 architectures 教训 4 的响应):
   - SOCA 中"96% 显著性""94% 统计功效""充分性 >0.8" 均为**推导值,不是测量值**——见 03 §六统计功效分析与 08 §三训练循环中的公式推演
   - **架构简化原则**:任何跨线引用 SOCA 数字时,必须找到对应原文推导过程并标注"[estimated, not measured]"

---

## 🔬 SOCA 实测数据空白声明

> **截至 2026-08-28,本文档集的全部 9 份文档 + soca_micro_final_config.yaml 仍为设计文档,零实验数据**。README §"🗓️ 验证周期规划" 里程碑 4-7 状态为 🔨(实施中)/ ⏳(待开始)——**主模型训练 + 24 项 SOCA 架构消融 + 22 项通用预训练消融均未执行**。

| 文档 | 实测数据状态 |
|---|---|
| 01-09 全部 9 份 | ❌ 零实验数据;所有结论为推导/预估 |
| soca_micro_final_config.yaml | ❌ 配置真源已锁定(03 §五.6),但未用其训练任何模型 |
| 24 项 SOCA-A0~A24 消融 | ❌ 全部 ⏳ 待执行(04 §九) |
| 22 项 GENERAL-A1~F4 消融 | ❌ 全部 ⏳ 待执行(05 §二) |

**影响**:任何依赖 SOCA 数字的决策(无论是 architectures 还是 AGENTS.md F-01)目前都基于**假设而非证据**。

---

## 🗓️ 验证周期规划

```
主模型训练              24 消融实验            分析+报告
(~19h, 8×A100)         (~4.5 天, 4 并行)     (1 天)
[═══════════]          [═════════════]      [══════]
                                                      ↓
                                                 总计 ~6 天
```

**关键里程碑**:
1. ✅ 14 层审查与 12 项问题修正(本文档集 §01)
2. ✅ 参数量甜点确定 ~148M(本文档集 §02)
3. ✅ 层数甜点确定 16 层(本文档集 §03)
4. 🔨 实施:编写 v3-Micro-Final 配置代码(PyTorch 原生实现,沿用 MiniMind 训练链路)
5. ⏳ 实施:24 个消融实验的脚本与超参表
6. ⏳ 验证:在 5B-6B token 数据上完成主模型训练 + 全部消融
7. ⏳ 决策:基于消融结论,确定 0.8B-1.2B 正式模型的架构参数

---

## 🚫 不在范围内的事项

- ❌ 0.8B-1.2B 正式模型的完整架构设计(那是下一步)
- ❌ SOCA v3-Micro 的实际代码实现(本文档仅研究结论,未涉及代码)
- ❌ 训练数据的具体构造与合成(那是 `agenticdsl-training/01-training-data-pipeline.md` 的范围)
- ❌ 与 HydraForge C++ 引擎的集成(那是 `agenticinference/` 的范围)
- ❌ AgenticDSL LLM 训练(那是 `agenticdsl-training/` 主干,与 SOCA 是**平行项目**而非上下游)

---

## 🚀 推荐阅读顺序

| 读者 | 推荐路径 |
|---|---|
| **架构师 / 决策者** | 本 README → [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §八(最终对比总结) |
| **算法工程师** | [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) §二、三、四(三个审查维度)→ [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §五(16 层 × 155M 配置) |
| **训练工程师** | [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §五(训练稳定性与准确度)→ [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §五.7(训练超参) |
| **训练稳定性工程师** | [`08-router-stability-training.md`](./08-router-stability-training.md) 全文(七重防护 + F1-F7 失效模式 + 训练循环)+ [`09-sae-bus-information-theory.md`](./09-sae-bus-information-theory.md) §七(SAE 配方) |
| **理论研究者** | [`09-sae-bus-information-theory.md`](./09-sae-bus-information-theory.md) 全文(字典共演化 + 总线信息论 + GWT 映射) |
| **实施工程师** | [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) 全文(项目结构 + 代码骨架 + 4 周里程碑)+ [`08-router-stability-training.md`](./08-router-stability-training.md) §三(完整训练循环) |
| **模块设计师** | [`06-architecture-modules.md`](./06-architecture-modules.md) M1-M16 + [`07-module-extensions.md`](./07-module-extensions.md) M17-M24(完整 24 模块规格) |
| **可扩展性研究者** | [`07-module-extensions.md`](./07-module-extensions.md) §五(M23 SOCAConfig + **4 个规模预设**:tiny / **micro 155M** / medium / large) |
| **消融实验协调者** | [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) §九(SOCA 消融)+ [`05-pretraining-ablation-plan.md`](./05-pretraining-ablation-plan.md) §八(代码实现)+ §九(SOCA 调整) |
| **项目协调者** | 本 README → [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §三.3(关键发现)→ [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) §十二(里程碑) |

---

## 📅 文档版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-28 | 初版,三份研究文档完成(14L 审查 + 参数量甜点 + 层数甜点) |
| v1.1 | 2026-08-28 | 新增第 4 份文档 `04-implementation-roadmap.md`,把 §03 锁定的 16 层 × ~155M 配置落地为代码骨架、4 周里程碑计划与 8 个工程陷阱应对;README 同步更新:目录定位、文档结构、结论链路(扩展到四份)、推荐阅读顺序(新增"实施工程师"路径)、版本表 |
| v1.2 | 2026-08-28 | 新增第 5 份文档 `05-pretraining-ablation-plan.md`(24 项通用预训练消融详细执行计划),与 04 的 24 项 SOCA 架构消融形成 **48 项完整消融体系**;README 同步:文档结构、结论链路(扩展到五份)、推荐阅读顺序(新增"消融实验协调者"路径) |
| v1.3 | 2026-08-28 | 新增第 6、7 份文档 `06-architecture-modules.md`(M1-M16 核心模块规格)与 `07-module-extensions.md`(M17-M24 辅助子模块 + 完整 24 模块清单 + 5 阶段 12 周实施优先级 + 3 个规模预设:tiny/medium/large);README 同步:文档结构(7 份)、结论链路(扩展到七份,新增"模块化设计→规模扩展"分支)、推荐阅读顺序(新增"模块设计师"与"可扩展性研究者"路径) |
| v1.4 | 2026-08-28 | 新增第 8、9 份文档 `08-router-stability-training.md`(路由器稳定性七重防护 + F1-F7 失效模式 + 完整训练循环 + 三档超参)与 `09-sae-bus-information-theory.md`(联合 SAE 字典共演化动力学 + 广播总线信息论:GWT 映射/充分性/率失真);07 增加训练配方交叉引用;README 同步:文档结构(9 份)、结论链路(扩展到九份)、推荐阅读顺序(新增"训练稳定性工程师"与"理论研究者"路径) |
| v1.5 | 2026-08-28 | **矛盾审查与对齐修订**(Oracle 评审 8.5-9/10):① 01 加 v3-Micro-REV 历史声明;② 02 加 14L 历史配置声明 + 缩放预测改以 155M 为基线;③ 03 §五.6 加 ⭐ 参数真源声明 + §五.4/§五.5 加"非实施依据"注;④ 04 §九消融编号加 `SOCA-` 前缀 + §二 ablation dict 补全为 22 字段 + §八.1 加注指向 07 M19;⑤ 05 消融编号加 `GENERAL-` 前缀 + §十一加"编号引用约定";⑥ 06 §四 M4 重写为 Gated DeltaNet + Standard 混合 + §三 M1 d_bus 比例修订;⑦ 07 §三 M19 加 ⭐ 训练阶段真源声明 + §五新增 `config_micro()` 155M 预设 + 重构项清单;⑧ 08 §五拆分为 5.1/5.2(v3-Micro top_k=4 明确);⑨ 09 §五.2 示例代码 `k=top_k` + §十.2 γ 加实践推荐;⑩ README 同步(预设改 4 个 / 消融编号补前缀 / 消融总数 46 项);⑪ **新增 `soca_micro_final_config.yaml` v1.0**——机器可读参数真源副本(从源头消除文档间抄数字漂移) |
| v1.6 | 2026-08-28 | **Oracle 评审后修订**(`docs/architectures/` 融合可行性分析 → 评分 4/10 → 7/10 需 3 前置条件):① **09 §13.2 降级**——"总线 = 意识"四象限映射改为 `[HYPOTHESIS — 未经验证]` + 列出 probe 验证方法 + 引用 architectures/06 §2.1 utilization 10% 上限(拆除 C1 涌现声明红旗);② **09 §十八 一句话总结降级**——"意识的数学化身"改为"认知科学隐喻 + architectures 1.5B 上限 +2pp 引用";③ **README 新增"🎯 消费者契约"节**——明确 SOCA 消融输出对应 AGENTS.md F-01(Qwen2.5-Coder-7B 现成权重)路径 A 悬空 + 12 个月降级触发器;④ **README 新增"⚠️ 基线与外推边界"节**——SOCA(155M/MiniMind 6B)vs architectures(1.5B Qwen 现成)vs AgenticDSL(7B Qwen 现成)三向对照 + 禁止跨线引用规则 + 派生值标注;⑤ **README 新增"🔬 SOCA 实测数据空白声明"**——明确标注全部 9 份文档零实验数据 + 24 项 SOCA-A0~A24 全部 ⏳ + 22 项 GENERAL-A1~F4 全部 ⏳ + 任何依赖 SOCA 数字的决策当前基于假设而非证据 |

---

> **核心承诺**:本目录的研究不是为了追求"更大的模型",而是为了在最小验证代价下,**对一组明确的架构假设**给出 **可重复、可外推** 的判断。

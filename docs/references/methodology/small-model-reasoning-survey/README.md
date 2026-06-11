# 1B 小模型推理能力建设 - 综合调研与审查

> **调研时间**:2026 年 6 月
> **调研方法**:6 个深度调研 agent 并行执行,覆盖数据工程、训练策略、架构创新、推理时扩展、循环深度模型、系统部署 6 大方向
> **核心理念**:Skeptical Verification + Evidence-Based Recommendations
> **任务来源**:用户提出 25+ 技术方向 + "Cog-Tina-RAG-Loop 黄金组合" + "CRV + 循环深度模型" 方案,需要独立审查并给出 SOTA 建议

---

## 📁 目录结构

### 1. 调研底稿(6 个并行 agent 的原始报告)

| 文档 | 内容 | 关键发现 |
|---|---|---|
| [`01-data-engineering-survey.md`](./01-data-engineering-survey.md) | 数据工程方向(CogPO/CRV、MRPV、Self-Play、PRM、Ouro、RELAY、TinyStories) | **Top 3: R1-Distill 蒸馏、rStar-Math MCTS、CRV 后处理** |
| [`02-training-strategy-survey.md`](./02-training-strategy-survey.md) | 训练策略方向(Tina、LoRA-RL、DPO/KTO、GRPO、RED、课程 RL) | **Top 3: GRPO+R1-Distill 起点+课程、LUFFY、Tina** |
| [`03-architecture-innovation-survey.md`](./03-architecture-innovation-survey.md) | 架构创新方向(UT、mHC、DeltaNet、Mamba、RWKV、MoE、Ouro、Neuro-Symbolic) | **Top 3: Ouro、PAL/Neuro-Symbolic、MoE Upcycling** |
| [`04-inference-time-scaling-survey.md`](./04-inference-time-scaling-survey.md) | 推理时扩展(Self-RAG、CA-TTS、Speculative、DES、Agent、BoN、ToT) | **Top 3: BoN+PRM、DeepConf、Agent Loop+Tool** |
| [`05-loop-model-deepdive.md`](./05-loop-model-deepdive.md) | 循环深度模型专项(Ouro、RELAY、Saunshi、retrofit、CogPO) | **CRV+LoopLM 组合无任何实证,概念混杂** |
| [`06-system-deployment-survey.md`](./06-system-deployment-survey.md) | 系统部署(AWQ、FP8、PowerInfer-2、vLLM、SnapKV、EAGLE-3、QAT) | **Top 3: FP8+vLLM、EAGLE-3、PagedAttention+SnapKV** |

### 2. 用户原始提议(待修正)

| 文档 | 内容 |
|---|---|
| [`10-user-original-proposal.md`](./10-user-original-proposal.md) | 用户原始提出的 25+ 技术方向清单 + "Cog-Tina-RAG-Loop" 黄金组合 + "CRV+循环深度模型" 方案 |

### 3. 修正后的提议(基于综合报告)

| 文档 | 内容 |
|---|---|
| [`20-revised-proposal.md`](./20-revised-proposal.md) | 基于调研证据修正后的技术方向 + 替代方案 + 5 步实施路线 |

### 4. 综合评审报告(主文档)

| 文档 | 内容 |
|---|---|
| [`00-comprehensive-review-report.md`](./00-comprehensive-review-report.md) | 主报告:用户 25+ 方向逐一审查 + 黄金组合审查 + CRV+LoopLM 方案审查 + SOTA 建议 |

---

## 🎯 核心结论速读

### 用户方案的关键问题

1. **"Cog-Tina-RAG-Loop" 黄金组合**: 多个组件需要替换
   - Self-RAG 在 1B 失败 → 用 Pleias-RAG-1B / SeaKR
   - CogPO 在 1B 未验证 → 用 DistilQwen2.5-R1 路线
   - mHC 在 1B 无对比 → 用 mHC-lite 简化版或 Ouro retrofit
   - Tina 必须从 R1-Distill 起点开始

2. **"CRV + 循环深度模型" 方案**: 概念混杂,缺乏实证
   - 混淆了 Meta CRV(白盒 interpretability)和 阿里 CRV(数据生成)
   - "implicit latent reasoning 在 1B 工作"无实证支持
   - "7.7T tokens 是 implicit reasoning 必需"是错误的(只对 Ouro 特定训练)
   - 1B LoopLM 真正可行路径是 **mcleish7 retrofit(50B tokens → 49.9% GSM8K)**

### 真正的 1B SOTA 路径

| 阶段 | 推荐技术 | 1B 实证 |
|---|---|---|
| **数据** | R1-Distill + OpenR1/Mixture-of-Thoughts 精炼 | MATH-500 83.9% |
| **训练** | DeepScaleR/FastCuRL/OpenRS-Star 风格(GRPO + 课程) | AIME24 49.6% |
| **架构(可选)** | Neuro-Symbolic(PAL)/ Ouro retrofit | GSM8K 81.5% / 78.9% |
| **推理时** | BoN + 1.5B PRM + Self-Calibration + DeepConf + Tool Use | MATH-500 26.8→59.6 |
| **系统** | FP8 + vLLM + EAGLE-3 + SnapKV | batch=1 1.4-6.5× |

### 必须拒绝的 5 件事

1. ❌ **CogPO/CRV 在 1B 赌博**(论文未验证 1B,只承诺 ≥3B)
2. ❌ **原始 Self-RAG 端到端训练**(< 3B 反思 token 训练失败)
3. ❌ **Long CoT R1-style 蒸馏 1B 模型**(EMNLP 2025 证实永久性 -75% 退化)
4. ❌ **ToT/GoT/AoT**(1B 生成器瓶颈)
5. ❌ **CogPO + Looped Transformer 组合**(无任何工作做过)

---

## 📊 文档间关系图

```
00-comprehensive-review-report.md (主报告,顶层)
    │
    ├──► 01-06 调研底稿(证据来源)
    │
    ├──► 10-user-original-proposal.md(待修正的原始提议)
    │
    └──► 20-revised-proposal.md(基于 00+01-06 修正后的最终建议)
```

**阅读建议**:
- 想快速理解结论:读 `00-comprehensive-review-report.md`
- 想看具体技术细节:读 `01-06` 对应方向
- 想看用户原始思路:读 `10`
- 想看最终推荐路径:读 `20`

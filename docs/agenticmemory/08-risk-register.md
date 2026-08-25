# 08 · 风险登记册 — 所有已识别风险集中跟踪

> **文档 ID**: MEM-008-RISK-REGISTER
> **生成日期**: 2026-08-25
> **状态**: 草案 v0.1
> **目的**: 集中登记 agenticmemory 项目所有已识别风险,统一跟踪缓解措施与责任人

---

## 0. 使用说明

本文档是所有风险文档的**集中入口**。风险分为以下类别:

| 风险类别 | 主要来源文档 | 典型风险 |
|---|---|---|
| **核心能力风险** | [`01-memory-model.md`](01-memory-model.md) | 能力边界误判、Wiki DAG 不完整 |
| **训练风险** | [`02-training-design.md`](02-training-design.md) | 灾难性遗忘、数据污染、RL 不稳定 |
| **评估风险** | [`03-evaluation.md`](03-evaluation.md) | B/A 假阳性、评估成本失控 |
| **对话特化风险** | [`04-dialogue-extension.md`](04-dialogue-extension.md) | 增量更新不一致、长对话信息丢失 |
| **本体涌现风险** | [`05-schema-emergence.md`](05-schema-emergence.md) | Schema 漂移、噪音淹没信号、本体无限膨胀 |
| **评估方法论风险** | [`06-evaluation-methodology.md`](06-evaluation-methodology.md) | Probe Model 假阳性、评估不显著 |
| **生产部署风险** | [`07-production-deployment.md`](07-production-deployment.md) | 数据飞轮停滞、回归测试失败、Schema 演化失控 |
| **工程实现风险** | `agenticmemory_training/` + `agenticmind/extraction/` | Wiki DAG 构建算法、LoRA 训练失败 |

---

## 1. 核心能力风险(01)

| ID | 风险 | 严重度 | 缓解措施 | 责任人 | 状态 |
|---|---|---|---|---|---|
| **R-C01** | Wiki DAG 构建算法无法保证完整性(节点去重、边合并、层级推断) | 🔴 高 | P1 启动前必须决策算法(见 [`01-memory-model.md`](01-memory-model.md) §9 O1) | 架构组 | 🔴 待解决 |
| **R-C02** | 五步漏斗判定错误导致记忆/推理层误分 | 🟡 中 | 多层验证 + 人工抽检(见 [`01-memory-model.md`](01-memory-model.md) §3) | 训练组 | 🟡 监控中 |
| **R-C03** | `needs_reasoning_model_verification` 标注不准确 | 🟡 中 | 专门 Type E 训练样本(见 [`02-training-design.md`](02-training-design.md) §6.5) | 训练组 | 🟡 监控中 |
| **R-C04** | 数值/条件/否定信息丢失导致推理失败 | 🔴 高 | 关键信息加权损失(数值 5×/条件 4×/否定 4×,见 [`02-training-design.md`](02-training-design.md) §8) | 训练组 | 🔴 监控中 |

---

## 2. 训练风险(02)

| ID | 风险 | 严重度 | 缓解措施 | 责任人 | 状态 |
|---|---|---|---|---|---|
| **R-T01** | 训练数据污染(训练集/测试集重叠) | 🔴 高 | 四层去重 + 时间切分(见 [`02-training-design.md`](02-training-design.md) §9.3) | 数据组 | 🟡 已缓解 |
| **R-T02** | 灾难性遗忘(推理训练遗忘记忆能力) | 🟡 中 | Replay Buffer 30% + EWC + 回归测试(见 [`03-evaluation.md`](03-evaluation.md) §7.3) | 训练组 | 🟡 监控中 |
| **R-T03** | RL 训练不稳定(奖励黑客) | 🟡 中 | KL 散度约束 + 逻辑校验器(见 [`05-schema-emergence.md`](05-schema-emergence.md) §2.2) | 训练组 | 🟡 监控中 |
| **R-T04** | 1B 模型能力上限(记忆提取召回率不达标) | 🟡 中 | 回退到 3B 模型或引入蒸馏(见 [`05-schema-emergence.md`](05-schema-emergence.md) §4) | 训练组 | 🟢 备选方案已备 |
| **R-T05** | Wiki 输出序列过长导致训练不稳定 | 🟡 中 | 分节训练(先 core_facts,再逐步增加其他节,见 [`02-training-design.md`](02-training-design.md) §7) | 训练组 | 🟡 监控中 |
| **R-T06** | 训练配比敏感(各任务配比不当) | 🟡 中 | 消融实验 + 验证集动态调整(见 [`02-training-design.md`](02-training-design.md) §7) | 训练组 | 🟡 监控中 |
| **R-T07** | 推理训练导致记忆能力下降 | 🟡 中 | 推理任务配比 ≤ 60%,保留 ≥ 40% 记忆任务(见 [`02-training-design.md`](02-training-design.md) §9) | 训练组 | 🟡 监控中 |

---

## 3. 评估风险(03 + 06)

| ID | 风险 | 严重度 | 缓解措施 | 责任人 | 状态 |
|---|---|---|---|---|---|
| **R-E01** | B/A 比值假阳性(SOTA 答题器脑补) | 🔴 高 | 使用 Probe Model(base model)而非 SOTA(见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) §1) | 评估组 | 🔴 已识别 |
| **R-E02** | 评估成本失控(教师 API 调用过多) | 🟡 中 | 评估集固定规模 + 预算上限(见 [`03-evaluation.md`](03-evaluation.md) §8) | 评估组 | 🟡 已缓解 |
| **R-E03** | B/A 统计显著性不足(样本量不够) | 🟡 中 | 明确统计显著性检验方法(见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) §5) | 评估组 | 🟡 待解决 |
| **R-E04** | 在线监控指标与离线指标不一致 | 🟡 中 | 明确两者差异并监控(见 [`03-evaluation.md`](03-evaluation.md) §7.1) | 评估组 | 🟡 待解决 |
| **R-E05** | Probe Model 选型不当导致评估失效 | 🟡 中 | 必须是 base model,禁止 instruction-tuned(见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) §2.3) | 评估组 | 🟡 已识别 |

---

## 4. 对话特化风险(04)

| ID | 风险 | 严重度 | 缓解措施 | 责任人 | 状态 |
|---|---|---|---|---|---|
| **R-D01** | 增量更新与全量重建不一致 | 🟡 中 | 增量一致性验证(≥ 98% 一致,见 [`04-dialogue-extension.md`](04-dialogue-extension.md) §7.3) | 架构组 | 🟡 待解决 |
| **R-D02** | 长对话(100+ 轮)信息丢失 | 🟡 中 | N 轮活跃窗口 + 索引层压缩(见 [`04-dialogue-extension.md`](04-dialogue-extension.md) §4.3) | 架构组 | 🟡 待解决 |
| **R-D03** | 指代消解错误率过高 | 🟡 中 | Type J 专项训练 + 指代消解准确率监控(见 [`04-dialogue-extension.md`](04-dialogue-extension.md) §6.2) | 训练组 | 🟡 监控中 |
| **R-D04** | 约束累积遗漏 | 🟡 中 | Type L 专项训练 + 约束累积完整率监控(见 [`04-dialogue-extension.md`](04-dialogue-extension.md) §6.4) | 训练组 | 🟡 监控中 |

---

## 5. 本体涌现风险(05)

| ID | 风险 | 严重度 | 缓解措施 | 责任人 | 状态 |
|---|---|---|---|---|---|
| **R-S01** | 噪音淹没信号(过度提取) | 🔴 高 | 稀疏性惩罚 + L1 正则 + 提取数量上限(见 [`05-schema-emergence.md`](05-schema-emergence.md) §5) | 训练组 | 🔴 已识别 |
| **R-S02** | Schema 漂移(涌现的类型不受控) | 🟡 中 | Schema 版本管理 + 漂移检测 + 增量更新(见 [`03-evaluation.md`](03-evaluation.md) §7.2) | 架构组 | 🟡 监控中 |
| **R-S03** | 本体无限膨胀(新类型过多) | 🟡 中 | 设置类型上限 + 定期合并剪枝(见 [`05-schema-emergence.md`](05-schema-emergence.md) §2.4) | 架构组 | 🟡 监控中 |
| **R-S04** | 伪推理陷阱(复杂记忆误判为推理) | 🟡 中 | 逻辑算子校验(causes/depends_on/if-then,见 [`05-schema-emergence.md`](05-schema-emergence.md) §4.4) | 训练组 | 🟡 监控中 |
| **R-S05** | 重构损失导致过度压缩 | 🟡 中 | 监控重构 Loss 与信息保留率的平衡(见 [`05-schema-emergence.md`](05-schema-emergence.md) §2.3) | 训练组 | 🟡 监控中 |

---

## 6. 评估方法论风险(06)

| ID | 风险 | 严重度 | 缓解措施 | 责任人 | 状态 |
|---|---|---|---|---|---|
| **R-M01** | Probe Model 假阳性(弱模型碰巧答对) | 🟡 中 | 多个不同弱模型交叉验证 + 人工抽检(见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) §1.3) | 评估组 | 🟡 监控中 |
| **R-M02** | 效用增益阈值设置不当(过松或过严) | 🟡 中 | 初始阈值 + 训练中校准(见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) §2.2) | 评估组 | 🟡 待解决 |
| **R-M03** | QA 题库质量不高(合成问题过于简单) | 🟡 中 | 混合来源:合成 + 公开数据集 + 人工标注(见 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) §4.4) | 评估组 | 🟡 待解决 |

---

## 7. 生产部署风险(07)

| ID | 风险 | 严重度 | 缓解措施 | 责任人 | 状态 |
|---|---|---|---|---|---|
| **R-P01** | 数据飞轮停滞(长期没有触发重训) | 🟡 中 | 设置停滞告警(14 天无重训触发,见 [`07-production-deployment.md`](07-production-deployment.md) §5.2) | 部署组 | 🟡 已识别 |
| **R-P02** | 回归测试失败(新模型退化) | 🔴 高 | 强制回归测试 + 任何指标退化 > 5% 拒绝部署(见 [`07-production-deployment.md`](07-production-deployment.md) §6.3) | 部署组 | 🔴 已识别 |
| **R-P03** | Schema 演化失控(新类型无序增长) | 🟡 中 | 审批流程 + 版本管理 + 监控告警(见 [`07-production-deployment.md`](07-production-deployment.md) §5.2) | 架构组 | 🟡 已识别 |
| **R-P04** | 多 LoRA 并发显存超限 | 🟡 中 | S-LoRA 最大并发数限制 + KV 量化(见 [`07-production-deployment.md`](07-production-deployment.md) §2.1) | 部署组 | 🟡 已识别 |
| **R-P05** | KV 缓存量化精度损失 | 🟡 中 | 根据使用频率动态调整精度(hot fp16 / warm int8 / cold int4,见 [`07-production-deployment.md`](07-production-deployment.md) §3.2) | 部署组 | 🟡 已识别 |

---

## 8. 工程实现风险(agenticmemory_training + agenticmind)

| ID | 风险 | 严重度 | 缓解措施 | 责任人 | 状态 |
|---|---|---|---|---|---|
| **R-E01** | Wiki DAG 构建算法无法按时交付 | 🔴 高 | P1 启动前必须决策(见 [`01-memory-model.md`](01-memory-model.md) §9 O1 + [`../agenticmemory_training/08d-wiki-dag-construction.md`](../agenticmemory_training/08d-wiki-dag-construction.md) §5 待解决清单 O1.1/O1.2/O1.3) | 架构组 | 🔴 待解决 |
| **R-E02** | LoRA 训练失败(双 LoRA 切换延迟 > 50ms) | 🟡 中 | 如果延迟超标,改多任务单权重(见 [`../agenticmind/context-management/README.md`](../agenticmind/context-management/README.md) §7 R8) | 训练组 | 🟡 监控中 |
| **R-E03** | 数据合成管线成本超预算 | 🟡 中 | 教师 API 调用预算上限 + 每日告警(见 [`../agenticmemory_training/08a-capacity-gap-design.md`](../agenticmemory_training/08a-capacity-gap-design.md) §8 + 训练侧独立风险 R-02 监控) | 数据组 | 🟡 已缓解 |
| **R-E04** | Schema 融合边界被误解(13 字段与涌现 schema 混合) | 🟡 中 | 明确双轨分离,不混合(见 [`../agenticmemory_training/08b-seed-schema-fusion.md`](../agenticmemory_training/08b-seed-schema-fusion.md)) | 架构组 | 🟡 已缓解 |

---

## 9. 风险统计总览

| 严重度 | 数量 | 状态分布 |
|---|---|---|
| 🔴 高(必须解决) | 5 | 3 待解决,2 已识别,0 已缓解 |
| 🟡 中(监控中) | 19 | 8 待解决,10 监控中,1 已缓解 |
| 🟢 低(备选方案) | 1 | 1 备选方案已备 |
| **总计** | **25** | **12 待解决,13 监控中,2 已缓解/备选** |

**关键关注**:
- 🔴 **R-C01**(Wiki DAG 构建算法)是 P1 启动前的最大障碍,必须优先解决;详见 [`../agenticmemory_training/08d-wiki-dag-construction.md`](../agenticmemory_training/08d-wiki-dag-construction.md) §5 待解决清单 O1.1/O1.2/O1.3
- 🔴 **R-C04**(关键信息丢失)是训练质量的核心风险,贯穿整个训练过程
- 🔴 **R-E01**(B/A 假阳性)是评估方法论的核心风险,已在 [`06-evaluation-methodology.md`](06-evaluation-methodology.md) 中给出解决方案
- 🔴 **R-P02**(回归测试失败)是生产部署的核心风险,必须建立强制回归测试流程

**与训练侧风险的对应关系**(避免双副本漂移):
- 本文档集中登记 25 项风险(架构 + 训练 + 评估 + 部署 + 工程实现)
- 工程实现类风险(R-E01..R-E04)与训练侧 [`../agenticmemory_training/README.md`](../agenticmemory_training/README.md) §8 的 R-01..R-08 有重叠(R-E03 ≈ R-02 成本控制、R-E04 ≈ R-06 Schema 漂移)
- **单一真源原则**:架构级风险(本文档)+ 数据工程级风险(训练侧 README)互为索引,**不重复登记**;新增风险时先确认归属再登记

---

## 10. 风险管理流程

### 10.1 风险登记流程

```
新风险识别 → 分类(类别 + 严重度)→ 登记到本文档 → 分配责任人 → 跟踪状态
```

### 10.2 风险状态流转

```
🔴 待解决 → 🟡 监控中(已实施缓解措施)→ 🟢 已缓解(风险消除)→ 🔵 已关闭(确认无影响)
```

### 10.3 定期审视

- **每周**:检查 🔴 高风险的状态,确保按计划推进
- **每月**:审视 🟡 中风险,决定是否需要升级或降级
- **每季度**:全面审视所有风险,更新文档

---

## 11. 修订记录

| 版本 | 日期 | 变更 | 作者 |
|---|---|---|---|
| v0.1 | 2026-08-25 | 初始版本:集中登记所有已识别风险(25 项),按类别分组 | Sisyphus(AI 助手) |

---

**文档版本**: v0.1
**Owner**: AgenticMind 架构组 + 各风险责任人
**下一步**: 优先解决 R-C01(Wiki DAG 构建算法),同时推进 R-E01(Probe Model 评估方法论)
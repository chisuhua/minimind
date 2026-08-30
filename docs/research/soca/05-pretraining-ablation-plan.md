# SOCA v3-Micro 通用预训练消融实验执行计划(24 项)

> **目的**:在 [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) §九的 **24 项 SOCA 架构消融**(编号 `SOCA-A0` ~ `SOCA-A24`:循环/SAE/MH-MoE 等)之外,**补充 24 项通用预训练消融**(编号 `GENERAL-A1` ~ `GENERAL-F4`:数据/架构/训练策略/正则化/优化器/基础设施),共同构成 SOCA v3-Micro-Final 的 **48 项完整消融体系**。
>
> **消融编号约定**(避免歧义):
> - 04 §九: `SOCA-A0_baseline` / `SOCA-A1_no_cycles` / ... / `SOCA-A24_standard_moe`
> - 本文: `GENERAL-A1_dedup` / `GENERAL-A2_quality_filter` / ... / `GENERAL-F4_data_loader`
> - 两套编号**完全独立**,合并报告时**必须**带前缀以区分
>
> **默认上下文**:大语言模型预训练阶段。若实际场景为 CV、推荐系统或 RLHF,可替换对应模块但整体框架与判定逻辑保持不变。
>
> **关联文档**:
> - 上一阶段(实施路线图):[`04-implementation-roadmap.md`](./04-implementation-roadmap.md)
> - 上一阶段(层数甜点):[`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md)
> - 上一阶段(参数量甜点):[`02-sweet-spot-params.md`](./02-sweet-spot-params.md)
> - 上一阶段(14L 审查):[`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md)

---

## 一、与 04 SOCA 架构消融(SOCA-A0~A24)的互补关系

| 维度 | 04 的 SOCA 架构消融 | **05 的通用预训练消融**(本文档) |
|---|---|---|
| **消融编号** | `SOCA-A0` ~ `SOCA-A24`(24 项) | **`GENERAL-A1` ~ `GENERAL-F4`**(24 项) |
| **消融目标** | 验证 SOCA 三区域/三视角可解释性架构的有效性 | 验证预训练管线的基础组件选择 |
| **典型实验** | 去掉循环 / 去掉 SAE / MH→标准 MoE | RoPE / SwiGLU / AdamW→Sophia |
| **基线对照** | 完整 SOCA 架构 | 标准 Transformer (无 SOCA 创新) |
| **总项数** | 24 项(SOCA-A0~A24) | 24 项(GENERAL-A1-A4, B1-B4, C1-C4, D1-D4, E1-E4, F1-F4) |
| **执行方式** | 同模型内单点修改(架构组件) | 训练 pipeline 单点修改 |
| **组合维度** | 架构内组合(Phase 6) | 训练侧组合(Phase 6) |
| **整体消融数** | **46-48 项**(24 SOCA + 22-24 通用) |  |

> **重要原则**:本文档的消融是 04 的"前置必要条件"——如果通用预训练组件选择不当(例如优化器不合适),04 的 SOCA 架构消融结果会受噪声污染。**因此本计划应优先于 04 的 SOCA 消融执行**(但与 04 的 Phase 1 基础设施消融合并)。

---

## 二、消融配置总表(24 项)

> ⚠️ **所有编号前缀为 `GENERAL-`**(对应 04 的 `SOCA-`),合并报告时必须使用完整前缀。

### 数据类(GENERAL-A1~A4)

| 编号 | 消融内容 | 改什么 | 不改什么 | 预期效应量 |
|------|---------|--------|---------|-----------|
| `GENERAL-A1` | 去重策略 | 启用 MinHash+LSH 去重(阈值 0.8) | 其余 pipeline 不变 | PPL ↓ 0.05–0.15 |
| `GENERAL-A2` | 质量过滤 | 增加困惑度阈值过滤(PPL>500 剔除) | 数据量保持 ±5% | PPL ↓ 0.03–0.08 |
| `GENERAL-A3` | 数据配比 | 代码:网页:书籍 = 2:5:3 → 1:6:3 | 总 token 数不变 | 代码下游 ↑ 3–5% |
| `GENERAL-A4` | 课程学习 | 按难度排序(短→长、简单→复杂) | 数据内容不变 | 收敛速度 ↑ 10–15% |

### 架构类(GENERAL-B1~B4)

| 编号 | 消融内容 | 改什么 | 不改什么 | 预期效应量 |
|------|---------|--------|---------|-----------|
| `GENERAL-B1` | RoPE 位置编码 | 启用 RoPE(θ=10000) | 其余 attention 不变 | 长文本 PPL ↓ 0.1–0.3 |
| `GENERAL-B2` | SwiGLU 激活 | GELU → SwiGLU(FFN 维度×2/3 补偿) | 层数/头数不变 | PPL ↓ 0.02–0.05 |
| `GENERAL-B3` | RMSNorm | LayerNorm → RMSNorm(无 bias) | 归一化位置不变 | 训练速度 ↑ 5–8% |
| `GENERAL-B4` | GQA 注意力 | MHA → GQA(8 KV heads) | 模型宽度不变 | 推理吞吐 ↑ 20–30% |

### 训练策略类(GENERAL-C1~C4)

| 编号 | 消融内容 | 改什么 | 不改什么 | 预期效应量 |
|------|---------|--------|---------|-----------|
| `GENERAL-C1` | 学习率调度 | cosine → linear decay | warmup/peak LR 不变 | 末期 loss ↓ 0.01–0.03 |
| `GENERAL-C2` | Warmup 步数 | 2000 steps → 500 steps | 其余超参不变 | 早期 loss ↑ 0.02(可接受) |
| `GENERAL-C3` | 梯度累积 | 累积步数 4 → 8 | 有效 batch size 不变 | 梯度方差 ↓ 15% |
| `GENERAL-C4` | 序列长度 | 2048 → 4096(位置外推) | 模型参数不变 | 长文本任务 ↑ 5–10% |

### 正则化类(GENERAL-D1~D4)

| 编号 | 消融内容 | 改什么 | 不改什么 | 预期效应量 |
|------|---------|--------|---------|-----------|
| `GENERAL-D1` | Dropout | 0.1 → 0.0(关闭) | 其余正则不变 | PPL ↓ 0.01–0.03 |
| `GENERAL-D2` | Weight Decay | 0.1 → 0.01 | 优化器其余不变 | 过拟合信号 ↓ |
| `GENERAL-D3` | 梯度裁剪 | max_norm=1.0 → 0.5 | 优化器不变 | 训练稳定性 ↑ |
| `GENERAL-D4` | Label Smoothing | ε=0.0 → 0.05 | 损失函数类型不变 | 校准误差 ↓ 2–4% |

### 优化器类(GENERAL-E1~E4)

| 编号 | 消融内容 | 改什么 | 不改什么 | 预期效应量 |
|------|---------|--------|---------|-----------|
| `GENERAL-E1` | 优化器选择 | AdamW → Sophia | LR/调度不变 | 收敛速度 ↑ 15–25% |
| `GENERAL-E2` | β 参数 | AdamW β₂=0.95 → 0.999 | 其余不变 | 训练稳定性 ↑ |
| `GENERAL-E3` | 学习率峰值 | 3e-4 → 6e-4 | 调度曲线不变 | PPL ↓ 0.02–0.06 |
| `GENERAL-E4` | 精度策略 | BF16 → FP16+动态 loss scaling | 模型不变 | 显存占用 ↓ 10% |

### 基础设施类(GENERAL-F1~F4)

| 编号 | 消融内容 | 改什么 | 不改什么 | 预期效应量 |
|------|---------|--------|---------|-----------|
| `GENERAL-F1` | 并行策略 | FSDP → 3D 并行(TP+DP+PP) | 模型不变 | 扩展效率 ↑ 30% |
| `GENERAL-F2` | 通信后端 | NCCL → 自定义 ring-allreduce | 拓扑不变 | 通信开销 ↓ 20% |
| `GENERAL-F3` | Checkpoint 频率 | 每 5000 步 → 每 1000 步 | 存储策略不变 | 恢复时间 ↓ 80% |
| `GENERAL-F4` | 数据加载 | 单进程 → 8 worker prefetch | 数据源不变 | GPU 利用率 ↑ 5–10% |

---

## 三、执行顺序与依赖关系

### 3.1 并行组(可同时启动)

```
Phase 1(基础设施先行,串行):
  F4 → F1 → F2 → F3
  (必须先稳定训练基础设施,否则后续消融结果不可信)

Phase 2(架构类,全并行):
  B1 ‖ B2 ‖ B3 ‖ B4
  (互不依赖,可用同一 baseline checkpoint 分叉)

Phase 3(数据类,全并行):
  A1 ‖ A2 ‖ A3 ‖ A4
  (各自独立修改数据 pipeline)

Phase 4(训练策略+正则化,分组并行):
  [C1 ‖ C2 ‖ C3] ‖ [D1 ‖ D2 ‖ D3 ‖ D4]
  (C4 依赖 B1 的 RoPE 结果,需串行等待)

Phase 5(优化器类,全并行):
  E1 ‖ E2 ‖ E3 ‖ E4

Phase 6(汇总验证):
  将 Phase 2–5 中"通过"的改动合并,做联合消融验证
```

### 3.2 关键依赖链

- **F 系列必须最先完成**:如果训练基础设施不稳定,所有后续消融的方差都会膨胀,结果不可信
- **B1 → C4**:序列长度外推依赖位置编码方案
- **E1/E3 需在 B 系列确定后执行**:不同架构的最优 LR 不同
- **Phase 6 联合消融必须串行**:组合效应不可并行拆分

### 3.3 与 04 SOCA 消融的交叉依赖

```
Phase 0(并行): 04 Phase 1 基础设施消融 ‖ 05 Phase 1 基础设施消融
              (F1-F4 与 04 消融开关 reset 是同一基础设施,可合并)

Phase 1: 04 的 A0 基线模型训练(~19h)
         ‖ 05 Phase 1 基础设施验证(F1-F4,~2-3天)
         (并行:04 训练主模型的同时,验证训练栈稳定性)

Phase 2-5: 04 架构消融 A1-A24 ‖ 05 通用消融 A1-F4
           (并行:两个消融集互不干扰,但共享 GPU 资源)

Phase 6: 04 架构联合消融 ‖ 05 通用联合消融
         (各自组合后,再做"架构+通用"的总组合验证)
```

---

## 四、资源估算(以 7B 模型为例)

| 阶段 | 并行度 | 预计 GPU·hours | 墙钟时间 |
|------|--------|--------------|---------|
| Phase 1 | 1 job | ~2,000 | 3 天 |
| Phase 2+3 | 8 jobs 并行 | ~16,000 | 4 天 |
| Phase 4 | 7 jobs 并行 | ~14,000 | 3.5 天 |
| Phase 5 | 4 jobs 并行 | ~8,000 | 2 天 |
| Phase 6 | 1 job(联合) | ~5,000 | 2 天 |
| **合计** | | **~45,000** | **~15 天** |

> **SOCA v3-Micro-Final 调整**:由于模型仅 ~155M(见 [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md)),单次训练时间从 7B 的 ~2000 GPU·hours 降至 ~155M 的 ~19 GPU·hours(8×A100-80G)。**总消融周期从 15 天压缩至 ~6 天**(4 并行时)。

---

## 五、评估指标统一标准

### 5.1 核心指标体系

| 指标类别 | 具体指标 | 权重 | 说明 |
|---------|---------|------|------|
| **语言建模** | Validation PPL | 30% | 主信号,每 1000 步记录 |
| **下游任务** | MMLU (5-shot) | 20% | 通用知识 |
| | HellaSwag (10-shot) | 10% | 常识推理 |
| | HumanEval (0-shot) | 10% | 代码生成 |
| **监控信号** | 训练 loss 曲线平滑度 | 10% | 无 spike/震荡 |
| | 梯度范数稳定性 | 5% | 无爆炸/消失 |
| | 训练吞吐(tokens/s) | 5% | 不低于 baseline 的 90% |
| | 显存峰值 | 5% | 不超过硬件上限 |
| | 校准误差(ECE) | 5% | 概率输出可靠性 |

### 5.2 加权综合评分公式

```
Score = 0.30 × ΔPPL_norm
      + 0.20 × ΔMMLU_norm
      + 0.10 × ΔHellaSwag_norm
      + 0.10 × ΔHumanEval_norm
      + 0.10 × Stability_score
      + 0.05 × Throughput_score
      + 0.05 × Memory_score
      + 0.05 × Calibration_score
      + 0.05 × Gradient_score
```

其中各 `Δ*_norm` 为相对于 baseline 的归一化改善幅度(映射到 [0,1] 区间)。

---

## 六、"通过/失败"判定阈值与决策树

### 6.1 判定阈值

| 判定等级 | 条件 | 动作 |
|---------|------|------|
| **强通过 ✅** | Score ≥ 0.7 且 PPL 显著下降(p<0.01) | 纳入最终配置 |
| **弱通过 ⚠️** | 0.4 ≤ Score < 0.7 或 PPL 无显著变化但下游提升 | 保留,进入联合消融验证 |
| **边界 🟡** | 0.2 ≤ Score < 0.4 | 延长训练步数(×2)后复测 |
| **失败 ❌** | Score < 0.2 或 PPL 显著上升(p<0.01) | 弃用,记录原因 |
| **异常 🔴** | 训练发散 / loss spike / NaN | 立即终止,排查 bug |

### 6.2 决策树

```
消融实验完成
│
├─ 训练是否正常完成？
│   ├─ 否 → 🔴 异常,排查后重跑
│   └─ 是 ↓
│
├─ PPL 变化是否统计显著？(paired t-test, p<0.05)
│   ├─ 显著下降 → 进入 Score 计算
│   ├─ 无显著变化 → 检查下游任务
│   │   ├─ 下游任务有提升(≥2%)→ Score 加权时 PPL 项置零
│   │   └─ 下游也无提升 → Score 计算
│   └─ 显著上升 →
│       ├─ 上升幅度 < 0.05 → 进入 Score 计算(惩罚项)
│       └─ 上升幅度 ≥ 0.05 → ❌ 直接失败
│
├─ Score 计算完成
│   ├─ Score ≥ 0.7 → ✅ 强通过,纳入配置
│   ├─ 0.4 ≤ Score < 0.7 → ⚠️ 弱通过,进入 Phase 6 联合验证
│   ├─ 0.2 ≤ Score < 0.4 → 🟡 边界
│   │   ├─ 效应量方向正确？
│   │   │   ├─ 是 → 延长训练步数×2 后复测
│   │   │   └─ 否 → ❌ 失败
│   │   └─ 复测后仍边界 → ❌ 失败
│   └─ Score < 0.2 → ❌ 失败
│
└─ 最终输出:通过清单 + 弃用清单 + 联合消融方案
```

### 6.3 "不显著"的特殊处理规则

- **样本量不足导致的不显著**:若标准差过大,增加 3 个随机种子重跑,取均值后重新判定
- **效应量小但方向一致**:若多次实验均呈同方向改善(即使 p>0.05),标记为"弱信号",在 Phase 6 联合消融中验证是否有累积效应
- **与已有消融冲突**:若 A1 通过、A2 通过,但 A1+A2 联合后 Score 下降,说明存在负交互,需单独记录并排除组合

---

## 七、执行 Checklist 总结

1. **Day 0**:跑通 baseline,记录所有指标的基准值
2. **Day 1–3**:Phase 1 基础设施消融,确认训练栈稳定
3. **Day 4–7**:Phase 2+3 并行启动 8 个 job(架构+数据)
4. **Day 8–11**:Phase 4 训练策略+正则化(7 个 job 并行)
5. **Day 12–13**:Phase 5 优化器类(4 个 job 并行)
6. **Day 14–15**:Phase 6 联合消融验证(仅合并"通过"项)
7. **Day 16**:输出最终配置 + 消融报告

> **SOCA v3-Micro-Final 调整**:由于模型仅 ~155M,单次训练约 11 小时(消融版),并行 4 组 GPU 的总墙钟时间约 6 天。每个消融实验建议至少跑 **3 个随机种子**,报告均值 ± 标准差,确保结论的统计可靠性。

---

## 八、消融实验的代码实现要点

### 8.1 消融配置注册表(SOCA 扩展版)

> ⚠️ **所有编号前缀 `GENERAL-`**(与 04 §九 的 `SOCA-A0~A24` 完全独立)。

```python
# ablation/registry_general.py
"""通用预训练消融注册表(与 04 的 SOCA 架构消融并存)"""

GENERAL_ABLATION_REGISTRY = {
    # ═══ 数据类(GENERAL-A1~A4)═══
    "GENERAL-A1_dedup": {
        "desc": "启用 MinHash+LSH 去重",
        "category": "data",
        "overrides": {
            "data_pipeline.enable_dedup": True,
            "data_pipeline.dedup_threshold": 0.8,
        }
    },
    "GENERAL-A2_quality_filter": {
        "desc": "增加困惑度阈值过滤(PPL>500 剔除)",
        "category": "data",
        "overrides": {
            "data_pipeline.enable_quality_filter": True,
            "data_pipeline.ppl_threshold": 500,
        }
    },
    "GENERAL-A3_data_ratio": {
        "desc": "代码:网页:书籍 = 1:6:3",
        "category": "data",
        "overrides": {
            "data_pipeline.ratios": {"code": 1, "web": 6, "book": 3},
        }
    },
    "GENERAL-A4_curriculum": {
        "desc": "按难度排序的课程学习",
        "category": "data",
        "overrides": {
            "data_pipeline.curriculum": "difficulty_sorted",
        }
    },

    # ═══ 架构类(GENERAL-B1~B4)═══
    "GENERAL-B1_rope": {
        "desc": "启用 RoPE 位置编码(θ=10000)",
        "category": "architecture",
        "overrides": {
            "model.attention.position_encoding": "rope",
            "model.attention.rope_theta": 10000,
        }
    },
    "GENERAL-B2_swiglu": {
        "desc": "GELU → SwiGLU",
        "category": "architecture",
        "overrides": {
            "model.ffn.activation": "swiglu",
            "model.ffn.hidden_mult": 2 / 3,  # 补偿维度
        }
    },
    "GENERAL-B3_rmsnorm": {
        "desc": "LayerNorm → RMSNorm",
        "category": "architecture",
        "overrides": {
            "model.norm.type": "rmsnorm",
        }
    },
    "GENERAL-B4_gqa": {
        "desc": "MHA → GQA(8 KV heads)",
        "category": "architecture",
        "overrides": {
            "model.attention.kv_heads": 8,
        }
    },

    # ═══ 训练策略类(GENERAL-C1~C4)═══
    "GENERAL-C1_lr_schedule": {
        "desc": "cosine → linear decay",
        "category": "training_strategy",
        "overrides": {
            "training.lr_scheduler": "linear_decay",
        }
    },
    "GENERAL-C2_warmup": {
        "desc": "warmup 2000 → 500 步",
        "category": "training_strategy",
        "overrides": {
            "training.warmup_steps": 500,
        }
    },
    "GENERAL-C3_grad_accum": {
        "desc": "梯度累积 4 → 8",
        "category": "training_strategy",
        "overrides": {
            "training.grad_accum_steps": 8,
        }
    },
    "GENERAL-C4_seq_len": {
        "desc": "序列长度 2048 → 4096(依赖 GENERAL-B1 RoPE)",
        "category": "training_strategy",
        "overrides": {
            "training.seq_len": 4096,
        },
        "depends_on": ["GENERAL-B1_rope"]
    },

    # ═══ 正则化类(GENERAL-D1~D4)═══
    "GENERAL-D1_dropout": {
        "desc": "Dropout 0.1 → 0.0",
        "category": "regularization",
        "overrides": {
            "training.dropout": 0.0,
        }
    },
    "GENERAL-D2_weight_decay": {
        "desc": "Weight Decay 0.1 → 0.01",
        "category": "regularization",
        "overrides": {
            "training.weight_decay": 0.01,
        }
    },
    "GENERAL-D3_grad_clip": {
        "desc": "Grad clip 1.0 → 0.5",
        "category": "regularization",
        "overrides": {
            "training.grad_clip": 0.5,
        }
    },
    "GENERAL-D4_label_smoothing": {
        "desc": "Label smoothing 0 → 0.05",
        "category": "regularization",
        "overrides": {
            "training.label_smoothing": 0.05,
        }
    },

    # ═══ 优化器类(GENERAL-E1~E4)═══
    "GENERAL-E1_optimizer": {
        "desc": "AdamW → Sophia",
        "category": "optimizer",
        "overrides": {
            "training.optimizer": "sophia",
        }
    },
    "GENERAL-E2_beta2": {
        "desc": "AdamW β₂=0.95 → 0.999",
        "category": "optimizer",
        "overrides": {
            "training.beta2": 0.999,
        }
    },
    "GENERAL-E3_lr_peak": {
        "desc": "LR 峰值 3e-4 → 6e-4",
        "category": "optimizer",
        "overrides": {
            "training.lr": 6e-4,
        },
        "depends_on": ["GENERAL-B2_swiglu", "GENERAL-B3_rmsnorm"]
    },
    "GENERAL-E4_precision": {
        "desc": "BF16 → FP16+动态 loss scaling",
        "category": "optimizer",
        "overrides": {
            "training.precision": "fp16",
            "training.loss_scaling": "dynamic",
        }
    },

    # ═══ 基础设施类(GENERAL-F1~F4)═══
    "GENERAL-F1_parallelism": {
        "desc": "FSDP → 3D 并行",
        "category": "infrastructure",
        "overrides": {
            "infra.parallelism": "tp_dp_pp",
        }
    },
    "GENERAL-F2_communication": {
        "desc": "NCCL → 自定义 ring-allreduce",
        "category": "infrastructure",
        "overrides": {
            "infra.comm_backend": "custom_ring",
        }
    },
    "GENERAL-F3_checkpoint_freq": {
        "desc": "Checkpoint 频率 5000 → 1000 步",
        "category": "infrastructure",
        "overrides": {
            "infra.checkpoint_freq": 1000,
        }
    },
    "GENERAL-F4_data_loader": {
        "desc": "数据加载 → 8 worker prefetch",
        "category": "infrastructure",
        "overrides": {
            "infra.num_workers": 8,
        }
    },
}
```

### 8.2 类别感知执行器

```python
# ablation/general_runner.py
from enum import Enum

class AblationCategory(Enum):
    DATA = "data"
    ARCHITECTURE = "architecture"
    TRAINING_STRATEGY = "training_strategy"
    REGULARIZATION = "regularization"
    OPTIMIZER = "optimizer"
    INFRASTRUCTURE = "infrastructure"


class GeneralAblationRunner:
    """通用预训练消融执行器(与 04 的 SOCA 消融分离但共享基础设施)"""

    # 类别 → Phase 映射
    CATEGORY_PHASE = {
        AblationCategory.INFRASTRUCTURE: 1,
        AblationCategory.ARCHITECTURE: 2,
        AblationCategory.DATA: 3,
        AblationCategory.TRAINING_STRATEGY: 4,
        AblationCategory.REGULARIZATION: 4,  # 与训练策略并行
        AblationCategory.OPTIMIZER: 5,
    }

    def __init__(self, base_config, output_dir="./ablation_general_results"):
        self.base_config = base_config
        self.output_dir = output_dir
        self.results = {}

    def run_phase(self, phase_num: int, parallel: int = 4):
        """执行一个 phase 的所有消融"""
        # 找出该 phase 的所有消融
        phase_ablations = [
            (ab_id, ab) for ab_id, ab in GENERAL_ABLATION_REGISTRY.items()
            if self.CATEGORY_PHASE[AblationCategory(ab["category"])] == phase_num
        ]

        print(f"Phase {phase_num}: Running {len(phase_ablations)} ablations")
        for ab_id, ab in phase_ablations:
            # 检查依赖
            deps = ab.get("depends_on", [])
            if deps and not self._deps_satisfied(deps):
                print(f"  ⏸  Skipping {ab_id}: waiting for {deps}")
                continue

            self.run_single(ab_id)

    def run_single(self, ablation_id: str):
        ab = GENERAL_ABLATION_REGISTRY[ablation_id]
        print(f"\n▶ Running {ablation_id}: {ab['desc']}")

        # 应用覆盖
        config = self._apply_overrides(ab["overrides"])

        # 训练(简化版:7000 步,与 04 同步)
        model = SOCAMicro(config)
        trainer = SOCATrainer(model, config, ...)
        trainer.train(max_steps=7000)

        # 评估
        results = trainer.full_evaluate()
        results["category"] = ab["category"]
        results["score"] = self._compute_score(results, baseline=self.baseline_results)

        # 判定
        decision = self._decision_tree(results)
        results["decision"] = decision

        # 保存
        self.results[ablation_id] = results
        torch.save(results, f"{self.output_dir}/{ablation_id}.pt")

    def _compute_score(self, results, baseline):
        """计算加权综合评分"""
        score = 0.0
        # PPL 改善(30%)
        ppl_improve = (baseline["ppl"] - results["ppl"]) / baseline["ppl"]
        score += 0.30 * max(0, min(1, ppl_improve * 10))  # 归一化到 [0,1]
        # 下游任务改善(20+10+10=40%)
        score += 0.20 * max(0, min(1, (results["mmlu"] - baseline["mmlu"]) / 0.1))
        score += 0.10 * max(0, min(1, (results["hellaswag"] - baseline["hellaswag"]) / 0.1))
        score += 0.10 * max(0, min(1, (results["humaneval"] - baseline["humaneval"]) / 0.1))
        # 监控信号(30%)
        score += 0.10 * results.get("stability", 0.5)
        score += 0.05 * results.get("throughput", 0.5)
        score += 0.05 * results.get("memory", 0.5)
        score += 0.05 * results.get("calibration", 0.5)
        score += 0.05 * results.get("gradient", 0.5)
        return score

    def _decision_tree(self, results):
        """实现 §六 的决策树"""
        score = results["score"]
        ppl_change = results["ppl_change_ratio"]
        ppl_significant = results.get("ppl_significant", False)

        if results.get("training_failed", False):
            return "🔴 异常"

        if ppl_significant and ppl_change > 0.05:
            return "❌ 失败"

        if score >= 0.7 and ppl_significant and ppl_change < 0:
            return "✅ 强通过"
        elif score >= 0.4:
            return "⚠️ 弱通过"
        elif score >= 0.2:
            return "🟡 边界"
        else:
            return "❌ 失败"

    def run_all_phases(self, parallel: int = 4):
        """按 phase 顺序执行所有消融"""
        for phase in [1, 2, 3, 4, 5]:
            self.run_phase(phase, parallel=parallel)
        # Phase 6: 联合消融
        self._run_joint_ablation()

    def _run_joint_ablation(self):
        """合并所有"通过"的消融,做最终联合验证"""
        passed = [
            ab_id for ab_id, r in self.results.items()
            if r["decision"] in ["✅ 强通过", "⚠️ 弱通过"]
        ]
        print(f"\n🎯 Phase 6: Joint ablation with {len(passed)} components:")
        for ab_id in passed:
            print(f"  + {ab_id}")
        # 应用所有通过的 overrides
        joint_overrides = {}
        for ab_id in passed:
            joint_overrides.update(GENERAL_ABLATION_REGISTRY[ab_id]["overrides"])
        # 训练完整版
        self.run_single("PHASE6_JOINT")  # 单一消融 ID 标记为联合实验
```

---

## 九、SOCA v3-Micro-Final 特定的消融调整

由于 SOCA v3-Micro-Final 是 **混合架构**(三区域 + 多维 MoE + 混合注意力),并非标准 Transformer,需要对本计划做以下调整:

| 类别 | 调整项 | 原因 |
|---|---|---|
| **数据类(A1-A4)** | 全适用 | 数据 pipeline 与架构正交 |
| **架构类(B1-B4)** | 部分适用 | SOCA 已有 Gated DeltaNet/Linear Attn/RMSNorm;B1-B4 应在 SOCA 基线 vs SOCA+RoPE 等组合上做 |
| **训练策略类(C1-C4)** | 全适用 | 训练策略与架构正交 |
| **正则化类(D1-D4)** | 全适用 | 正则化与架构正交 |
| **优化器类(E1-E4)** | E1(Sophia)需先验证 SOCA 多区域梯度的二阶信息是否可用;其他适用 | SOCA 的稀疏 MoE 可能破坏 Hessian 估计 |
| **基础设施类(F1-F4)** | F1(FSDP→3D)在 155M 规模下不适用(单卡足够);其他适用 | SOCA v3-Micro 在 8×A100-80G 上单卡可容纳 |

**调整后的 155M 总消融项数**:

```
数据(4) + 架构(4, SOCA 调整) + 训练策略(4) + 正则化(4)
+ 优化器(3, E1 改为可选) + 基础设施(3, F1 不适用)
= 22 项(去掉 E1 和 F1,共 2 项不适用)
+ 联合消融 = 23 项
```

---

## 十、与其他文档的关系

| 文档 | 与本文档的关系 |
|---|---|
| [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) | **审查基础**:SOCA 架构审查中识别的 12 个问题为本文档的架构类消融(B1-B4)提供依据 |
| [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) | **超参基准**:148M 甜点版的 LR(2.5e-4)、warmup(2500)等是 E3/C2 消融的 base 值 |
| [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) | **架构基准**:16 层 × ~155M 配置为所有消融提供统一基线 |
| [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) | **互补**:本文档的 24 项通用消融 + 04 的 24 项 SOCA 架构消融 = 48 项完整消融 |
| [`README.md`](./README.md) | **索引**:本研究文档集的总索引 |

---

## 十一、决策交接清单

| 锁定项 | 值 |
|---|---|
| 消融编号 | 本文 24 项 = `GENERAL-A1`~`GENERAL-F4`;04 24 项 = `SOCA-A0`~`SOCA-A24` |
| 消融总数 | 通用 24 项 + SOCA 24 项 = **48 项** |
| 默认场景 | 大语言模型预训练(SOCA v3-Micro-Final 适配见 §九) |
| Phase 数 | 6 个(基础设施→架构→数据→训练→优化器→联合) |
| 判定阈值 | Score ≥ 0.7 强通过 / 0.4-0.7 弱通过 / 0.2-0.4 边界 / <0.2 失败 |
| 随机种子 | 每个消融 ≥ 3 个,报告均值 ± 标准差 |
| SOCA 调整 | 155M 模型下 E1(F1 不适用),实际通用消融项 = **22 项** |
| 联合消融 | Phase 6 串行执行,只合并"通过"的改动 |
| 与 04 消融关系 | **互补不重叠**:本文档测通用组件,04 测 SOCA 架构组件 |
| 总周期(SOCA) | ~6 天(4 并行,8×A100-80G) |
| **编号引用约定** | 任何引用必须使用完整前缀(如 `SOCA-A5` `GENERAL-B2`),禁止省略 |

---

## 📅 文档版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-28 | 初版,完成 24 项通用预训练消融详细执行计划(6 类 × 4 项) |

---

> **下一步**:阅读 [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) §九的 24 项 SOCA 架构消融,与本文档的 24 项通用消融合并,形成完整的 48 项消融体系;然后按本文档 Phase 1-6 的执行顺序启动消融。

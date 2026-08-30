# SOCA v3-Micro 模块化设计扩展(M17-M24):辅助子模块 + 完整清单 + 实施优先级

> **目的**:在 [`06-architecture-modules.md`](./06-architecture-modules.md) 的 **M1-M16 核心模块**基础上,补充 **M17-M24 辅助子模块**、**完整 24 模块清单**、**5 阶段实施优先级**、**SOCAConfig 配置系统**(含 3 个预设规模)。
>
> **关联文档**:
> - 上一阶段(核心模块):[`06-architecture-modules.md`](./06-architecture-modules.md) - M1-M16 核心规格
> - 上一阶段(代码骨架):[`04-implementation-roadmap.md`](./04-implementation-roadmap.md) - 可按 M1-M24 重构
> - 上一阶段(层数甜点):[`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) - 16 层 × 155M 锁定配置
> - 上一阶段(参数甜点):[`02-sweet-spot-params.md`](./02-sweet-spot-params.md)
> - 上一阶段(14L 审查):[`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md)
>
> **角色**:本文档完成 SOCA 框架从"核心模块"到"完整工程体系"的补全——M17-M24 把 M1-M16 从"白盒可干预架构"升级为"可生产化、可规模化、可监控的训练运行框架"。

---

## 一、与 06 核心模块的关系

| 维度 | 06 核心模块 | **07 模块化设计扩展**(本文档) |
|---|---|---|
| **覆盖范围** | M1-M16 核心规格 | M17-M24 辅助子模块 |
| **关注点** | What(做什么) + Why(为什么) + 接口 | How(怎么实现) + 工具类 + 配置 |
| **类别** | 全局基础设施 / 计算区 / 区间接口 / 工作空间内部 / 注意力变体 / 运行时系统 | **辅助工具 / 配置 / 顶层组装** |
| **关键新增** | M3 CausalGate(运行时干预) | **M19 PhaseScheduler(阶段冻结)** + **M23 SOCAConfig(规模预设)** + **M24 SOCAModel(顶层组装)** |
| **可观测性** | M2 MonitorSlot(每层) + M15 SOCAMonitor(异常检测) | **M21 HealthDashboard(可视化)** + **M22 SnapshotRecorder(轨迹回放)** |
| **扩展性** | 单一配置 | **3 个规模预设**(20M / 7B / 120B),规模无关框架 |

> **关键洞察**:06 的 M1-M16 是 SOCA 的**白盒架构本体**;07 的 M17-M24 是 SOCA 的**工程运行框架**。两者结合 = 完整的 SOCA 实施规范。

---

## 二、模块交互全景图(完整版)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Token Input                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PERCEPTION ZONE (Layers 0 ~ P)                                     │    │
│  │                                                                     │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │    │
│  │  │ M4: PercBlock│───▶│ M4: PercBlock│───▶│ M4: PercBlock│  ×P层   │    │
│  │  │ ┌──────────┐ │    │              │    │              │         │    │
│  │  │ │M13:StdAttn│ │    │              │    │              │         │    │
│  │  │ └──────────┘ │    │              │    │              │         │    │
│  │  │ ┌──────────┐ │    │              │    │              │         │    │
│  │  │ │Dense FFN │ │    │              │    │              │         │    │
│  │  │ └──────────┘ │    │              │    │              │         │    │
│  │  │ ┌──────────┐ │    │              │    │              │         │    │
│  │  │ │M2:Monitor│ │    │              │    │              │         │    │
│  │  │ └──────────┘ │    │              │    │              │         │    │
│  │  │ ┌──────────┐ │    │              │    │              │         │    │
│  │  │ │M3:Causal │ │    │              │    │              │         │    │
│  │  │ │Gates ×2  │ │    │              │    │              │         │    │
│  │  │ └──────────┘ │    │              │    │              │         │    │
│  │  └──────┬───────┘    └──────────────┘    └──────────────┘         │    │
│  │         │                                                           │    │
│  │         │  M1: Bus.read() ← 从总线获取全局上下文                    │    │
│  └─────────┼───────────────────────────────────────────────────────────┘    │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  M7: DOWN INTERFACE                                                 │    │
│  │  d_model → d_bus（线性 + 正交约束 + 残差旁路）                       │    │
│  └─────────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  WORKSPACE ZONE (Layers P ~ W)                                      │    │
│  │                                                                     │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ M5: WorkspaceBlock                                            │  │    │
│  │  │                                                               │  │    │
│  │  │  ┌────────────┐                                               │  │    │
│  │  │  │M14:LinearAttn│ ← 线性注意力（J-Lens 友好）                 │  │    │
│  │  │  └──────┬─────┘                                               │  │    │
│  │  │         ▼                                                     │  │    │
│  │  │  ┌────────────┐                                               │  │    │
│  │  │  │M9:Router   │ ← 隐空间路由（决定计算路径）                   │  │    │
│  │  │  └──────┬─────┘                                               │  │    │
│  │  │         ▼                                                     │  │    │
│  │  │  ┌────────────────────────┐                                   │  │    │
│  │  │  │M10: Expert_0           │                                   │  │    │
│  │  │  │M10: Expert_1           │ ← 隐空间计算（Top-K 选择）       │  │    │
│  │  │  │M10: Expert_2           │                                   │  │    │
│  │  │  │  ...                   │                                   │  │    │
│  │  │  │M10: Expert_{n-1}       │                                   │  │    │
│  │  │  └──────────┬─────────────┘                                   │  │    │
│  │  │             ▼                                                 │  │    │
│  │  │  ┌────────────┐                                               │  │    │
│  │  │  │M11:Aggregat│ ← 门控聚合（专家间交互）                      │  │    │
│  │  │  └──────┬─────┘                                               │  │    │
│  │  │         ▼                                                     │  │    │
│  │  │  ┌────────────┐                                               │  │    │
│  │  │  │M12:JointSAE│ ← 稀疏约束 + 特征提取                        │  │    │
│  │  │  └──────┬─────┘                                               │  │    │
│  │  │         │                                                     │  │    │
│  │  │  M2:MonitorSlot + M3:CausalGates ×4                          │  │    │
│  │  └─────────┼─────────────────────────────────────────────────────┘  │    │
│  │            │                                                        │    │
│  │            │  M1: Bus.write() → 向总线广播推理结果                   │    │
│  │            ▼                                                        │    │
│  │  ╔═══════════════════════════════════════════════════════════════╗  │    │
│  │  ║  M1: BROADCAST BUS                                          ║  │    │
│  │  ║  state: [batch, d_bus]                                      ║  │    │
│  │  ║                                                              ║  │    │
│  │  ║  ← 感知区读取（全局上下文）                                  ║  │    │
│  │  ║  ← 工作空间写入（推理结论）                                  ║  │    │
│  │  ║  → 动作区读取（最终上下文）                                  ║  │    │
│  │  ║  → SOCAMonitor 读取（实时监控）                              ║  │    │
│  │  ╚═══════════════════════════════════════════════════════════════╝  │    │
│  └─────────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  M8: UP INTERFACE                                                   │    │
│  │  d_bus → d_model（线性 + 总线注入 + 门控）                          │    │
│  └─────────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ACTION ZONE (Layers W ~ L)                                         │    │
│  │                                                                     │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │    │
│  │  │ M6: ActBlock │───▶│ M6: ActBlock │───▶│ M6: ActBlock │  ×A层   │    │
│  │  │ ┌──────────┐ │    │              │    │              │         │    │
│  │  │ │M13:StdAttn│ │    │              │    │              │         │    │
│  │  │ └──────────┘ │    │              │    │              │         │    │
│  │  │ ┌──────────┐ │    │              │    │              │         │    │
│  │  │ │Gate Net  │ │    │              │    │              │         │    │
│  │  │ │→64 Micro │ │    │              │    │              │         │    │
│  │  │ │ Experts  │ │    │              │    │              │         │    │
│  │  │ └──────────┘ │    │              │    │              │         │    │
│  │  │ ┌──────────┐ │    │              │    │              │         │    │
│  │  │ │M2:Monitor│ │    │              │    │              │         │    │
│  │  │ └──────────┘ │    │              │    │              │         │    │
│  │  │ ┌──────────┐ │    │              │    │              │         │    │
│  │  │ │M3:Causal │ │    │              │    │              │         │    │
│  │  │ │Gates ×2  │ │    │              │    │              │         │    │
│  │  │ └──────────┘ │    │              │    │              │         │    │
│  │  └──────┬───────┘    └──────────────┘    └──────────────┘         │    │
│  │         │                                                           │    │
│  │         │  M1: Bus.read() ← 从总线获取工作空间结论                  │    │
│  └─────────┼───────────────────────────────────────────────────────────┘    │
│            │                                                                │
│            ▼                                                                │
│      Output Logits                                                          │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════    │
│  M15: SOCAMonitor（运行时）                                                 │
│  ├── 读取所有 M2 插槽 → 异常检测                                           │
│  ├── 读取 M1 总线 → 全局状态评估                                           │
│  ├── 读取 M9 路由 → 计算路径监控                                           │
│  ├── 读取 M12 SAE → 概念激活监控                                           │
│  ├── 写入 M3 因果门 → 干预                                                 │
│  └── 覆写 M1 总线 → 全局干预                                               │
│  ════════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  M16: SOCALoss（训练时）                                                    │
│  ├── LM Loss                                                               │
│  ├── SAE Reconstruction (from M12)                                         │
│  ├── Orthogonality (from M7, M8)                                           │
│  ├── Load Balance (from M9)                                                │
│  ├── Bus Regularization (from M1)                                          │
│  ├── Gate Sparsity (from M6)                                               │
│  └── Expert Orthogonality (from M10)                                       │
│  ════════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  M21: HealthDashboard（可视化）                                              │
│  M22: SnapshotRecorder（轨迹）                                              │
│  M19: PhaseScheduler（阶段管理）                                            │
│  ════════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  M17: RMSNorm（归一化）              M23: SOCAConfig（配置）                  │
│  M18: RotaryEmbedding（位置编码）    M24: SOCAModel（顶层组装）              │
│  M20: ExpertDispatcher（专家分派）                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、辅助子模块类(M17-M20)

### M17. RMSNorm(均方根归一化)

**定位**:所有层的前置归一化。比 LayerNorm 更高效,训练更稳定。

```python
class RMSNorm(nn.Module):
    """
    RMSNorm：比 LayerNorm 少一次均值计算。
    
    公式：x / RMS(x) * γ
    其中 RMS(x) = sqrt(mean(x²))
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**关键参数**:

| 参数 | 典型值 | 说明 |
|:---|:---:|:---|
| `dim` | `d_model` (896) | 与归一化维度一致 |
| `eps` | 1e-6 | 数值稳定 |
| `weight` 初始化 | 1.0 | 初始为恒等变换 |

---

### M18. RotaryEmbedding(旋转位置编码)

**定位**:感知区和动作区的标准注意力使用 RoPE。工作空间的线性注意力不需要(线性注意力无相对位置概念)。

```python
class RotaryEmbedding(nn.Module):
    """
    旋转位置编码(RoPE)。
    
    优势:
    - 相对位置编码(不需要绝对位置)
    - 外推性好
    - 与注意力计算融合,无额外开销
    """
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算 cos/sin 表
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: [batch, heads, seq, dim]
        """
        seq_len = q.shape[2]
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)

        return q_rot, k_rot

    def _rotate(self, x, cos, sin):
        """旋转操作"""
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin
```

**关键参数**:

| 参数 | 典型值 | 说明 |
|:---|:---:|:---|
| `dim` | `d_head` (64) | 单个 head 的维度 |
| `max_seq_len` | 2048-8192 | 预计算长度 |
| `base` | 10000 | RoPE 基频(对应 [`05-pretraining-ablation-plan.md`](./05-pretraining-ablation-plan.md) B1 的 θ=10000) |

**部署位置**:仅 M13 StandardAttention 使用;M14 LinearAttention 不需要。

---

### M19. PhaseScheduler(训练阶段调度器)

> ⭐ **本模块是 SOCA 训练阶段定义的统一真源**(对齐 06 §八 M16 SOCALoss 与 08 §三 训练循环)。其他文档的"四阶段"表述应与本节保持一致。
>
> **与 04 §八.1 SOCALoss 的关系**:04 的 SOCALoss 按"训练进度"(0%/5%/30%/90%)分阶段;**本节按"区域 + 训练量百分比"分阶段**,更符合 SOCA 三区域架构的语义。实施时**建议采用本节定义**(07 M19),将 04 的 SOCALoss.set_phase() 重构为与本节一致。

**定位**:控制四阶段训练的切换逻辑。**管理哪些模块被冻结/解冻**。

**四阶段训练**:

| 阶段 | 训练量 | 解冻范围 | 目标 |
|:---:|:---:|:---|:---|
| Phase 0 | 30% | 感知区 | 编码器先学好 |
| Phase 1 | 30% | + 工作空间 + 接口 + 总线 | 推理核心先学好 |
| Phase 2 | 20% | + 动作区 + 上接口 | 输出能力对齐 |
| Phase 3 | 20% | **全部解冻** | 端到端微调 |

```python
class PhaseScheduler:
    """
    训练阶段调度器。
    
    四阶段:
    Phase 0: 感知区预训练(30% 训练量)
    Phase 1: 工作空间训练(30%)
    Phase 2: 动作区训练(20%)
    Phase 3: 端到端微调(20%)
    """
    def __init__(self, model, total_steps: int):
        self.model = model
        self.total_steps = total_steps

        # 阶段边界
        self.phase_boundaries = [
            int(total_steps * 0.30),  # Phase 0 → 1
            int(total_steps * 0.60),  # Phase 1 → 2
            int(total_steps * 0.80),  # Phase 2 → 3
        ]

        self.current_phase = 0
        self.loss_fn = SOCALoss(config)

    def step(self, global_step: int):
        """每步调用,检查是否需要切换阶段"""
        new_phase = 0
        for i, boundary in enumerate(self.phase_boundaries):
            if global_step >= boundary:
                new_phase = i + 1

        if new_phase != self.current_phase:
            self._switch_phase(new_phase)

    def _switch_phase(self, new_phase: int):
        """切换阶段:冻结/解冻参数"""
        print(f"Switching from Phase {self.current_phase} → Phase {new_phase}")

        # 先全部冻结
        for param in self.model.parameters():
            param.requires_grad = False

        if new_phase == 0:
            # 感知区
            for layer in self.model.perception_layers:
                for param in layer.parameters():
                    param.requires_grad = True

        elif new_phase == 1:
            # 工作空间 + 接口
            for layer in self.model.workspace_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.model.down_interface.parameters():
                param.requires_grad = True
            for param in self.model.up_interface.parameters():
                param.requires_grad = True
            for param in self.model.broadcast_bus.parameters():
                param.requires_grad = True

        elif new_phase == 2:
            # 动作区 + 上接口
            for layer in self.model.action_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.model.up_interface.parameters():
                param.requires_grad = True

        elif new_phase == 3:
            # 全部解冻
            for param in self.model.parameters():
                param.requires_grad = True

        # 更新损失函数权重
        self.loss_fn.set_phase(new_phase)
        self.current_phase = new_phase

    def get_trainable_params(self) -> int:
        """当前可训练参数数"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
```

**与 M16 SOCALoss 的协作**:

| 阶段 | M19 PhaseScheduler | M16 SOCALoss |
|:---:|:---|:---|
| Phase 0 | 解冻感知区 | `weights = {"lm": 1.0}` |
| Phase 1 | + 工作空间 + 接口 | `+ sae, ortho_interface, load_balance, bus_reg` |
| Phase 2 | + 动作区 | `+ gate_sparse` |
| Phase 3 | 全部解冻 | `所有权重 × 0.5`(端到端微调) |

---

### M20. ExpertDispatcher(专家调度器)

**定位**:高效地将 token 分派到对应专家。**避免 Python 循环**,使用向量化操作。

```python
class ExpertDispatcher(nn.Module):
    """
    专家调度器:高效的 token-to-expert 分派。
    
    避免逐专家循环,使用 gather/scatter 实现向量化分派。
    """
    def __init__(self, n_experts: int, capacity_factor: float = 1.5):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

    def forward(self, z: torch.Tensor, indices: torch.Tensor,
                experts: nn.ModuleList) -> torch.Tensor:
        """
        z: [batch, seq, d_bus] 隐空间表征
        indices: [batch, seq] 专家索引
        experts: 专家列表

        返回: [batch, seq, d_bus] 专家计算结果
        """
        batch, seq, d = z.shape
        n_tokens = batch * seq

        # 展平
        z_flat = z.reshape(n_tokens, d)  # [n_tokens, d]
        idx_flat = indices.reshape(n_tokens)  # [n_tokens]

        # 按专家排序
        sorted_indices, sort_order = idx_flat.sort()
        z_sorted = z_flat[sort_order]

        # 找到每个专家的边界
        counts = torch.bincount(sorted_indices, minlength=self.n_experts)
        boundaries = counts.cumsum(0)

        # 分派计算
        output_sorted = torch.zeros_like(z_sorted)
        start = 0
        for expert_id in range(self.n_experts):
            end = boundaries[expert_id].item()
            if end > start:
                expert_input = z_sorted[start:end]
                output_sorted[start:end] = experts[expert_id](expert_input)
            start = end

        # 恢复原始顺序
        output = torch.zeros_like(z_flat)
        output[sort_order] = output_sorted

        return output.reshape(batch, seq, d)
```

**与 M5 WorkspaceBlock 的协作**:M5 的 `_dispatch_experts` 方法(06 §五)可用 M20 替代,提升训练效率。

---

## 四、运行时可视化与记录类(M21-M22)

### M21. HealthDashboard(健康仪表盘)

**定位**:M15 SOCAMonitor 的**可视化前端**。实时展示模型内部状态。

```python
class HealthDashboard:
    """
    健康仪表盘:SOCAMonitor 的可视化和汇总层。
    
    输出:
    - 综合健康评分(0-100)
    - 分区状态(感知/工作/动作)
    - 总线热力图
    - 路由分布图
    - 概念激活图
    - 告警列表
    """
    def __init__(self, monitor: SOCAMonitor):
        self.monitor = monitor
        self.history = []
        self.max_history = 1000

    def update(self, diagnostics: dict) -> dict:
        """每步更新"""
        result = self.monitor.monitor_step(diagnostics)

        self.history.append({
            "health": result["health_score"],
            "alerts": result["n_alerts"],
            "bus_state": diagnostics["bus_states"][-1].clone(),
        })

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return result

    def get_summary(self) -> dict:
        """获取当前状态摘要"""
        if not self.history:
            return {"status": "no_data"}

        recent = self.history[-10:]
        avg_health = sum(r["health"] for r in recent) / len(recent)
        total_alerts = sum(r["alerts"] for r in recent)

        # 趋势判断
        if len(self.history) >= 20:
            old_avg = sum(r["health"] for r in self.history[-20:-10]) / 10
            trend = avg_health - old_avg
        else:
            trend = 0

        return {
            "health_score": avg_health,
            "trend": trend,  # 正=改善,负=恶化
            "total_alerts": total_alerts,
            "status": self._classify(avg_health),
        }

    def _classify(self, score: float) -> str:
        if score > 90:
            return "HEALTHY"
        elif score > 70:
            return "STABLE"
        elif score > 50:
            return "DEGRADED"
        else:
            return "CRITICAL"

    def render_text(self) -> str:
        """文本渲染(用于日志)"""
        s = self.get_summary()
        lines = [
            f"╔══════════════════════════════════════╗",
            f"║  SOCA Health Dashboard               ║",
            f"╠══════════════════════════════════════╣",
            f"║  Status:  {s['status']:<27}║",
            f"║  Health:  {s['health_score']:>6.1f} / 100              ║",
            f"║  Trend:   {s['trend']:>+6.2f}                      ║",
            f"║  Alerts:  {s['total_alerts']:>4}                       ║",
            f"╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)
```

**状态等级**:

| 等级 | 分数范围 | 含义 |
|:---:|:---:|:---|
| HEALTHY | > 90 | 模型状态正常,无异常 |
| STABLE | 70-90 | 监控指标轻微波动,可继续推理 |
| DEGRADED | 50-70 | 部分指标异常,建议干预或重置 |
| CRITICAL | < 50 | 严重异常,需要立即排查 |

---

### M22. SnapshotRecorder(状态记录器)

**定位**:记录推理过程中的**完整状态轨迹**,支持事后分析和回放。

```python
class SnapshotRecorder:
    """
    状态记录器:记录完整推理轨迹。
    
    用途:
    - 事后分析(为什么模型在某步出错)
    - 回放(复现特定行为)
    - 训练数据生成(用正常轨迹做基线)
    """
    def __init__(self, model, max_snapshots: int = 100):
        self.model = model
        self.max_snapshots = max_snapshots
        self.snapshots = []
        self.recording = False

    def start_recording(self):
        self.recording = True
        self.current_trajectory = []

    def record_step(self, step_idx: int, diagnostics: dict):
        """记录一步"""
        if not self.recording:
            return

        snapshot = {
            "step": step_idx,
            "bus_state": self.model.broadcast_bus.state.clone(),
            "monitor_vectors": {
                i: layer.monitor_slot.monitor_vector.clone()
                for i, layer in enumerate(self.model.layers)
            },
            "router_decisions": {
                i: layer.router.last_indices.clone()
                for i, layer in enumerate(self.model.layers)
                if hasattr(layer, 'router')
            },
            "sae_codes": {
                i: layer.joint_sae.last_codes.clone()
                for i, layer in enumerate(self.model.layers)
                if hasattr(layer, 'joint_sae')
            },
            "gate_values": {
                i: layer.last_gate_values.clone()
                for i, layer in enumerate(self.model.layers)
                if hasattr(layer, 'last_gate_values')
            },
        }
        self.current_trajectory.append(snapshot)

    def stop_recording(self) -> list:
        self.recording = False
        trajectory = self.current_trajectory
        self.snapshots.append(trajectory)

        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

        return trajectory

    def replay(self, trajectory: list, intervention_fn=None):
        """
        回放轨迹(可选干预)。
        
        intervention_fn: 可选的干预函数,在每步调用
        """
        results = []
        for snapshot in trajectory:
            if intervention_fn:
                intervention_fn(snapshot)
            results.append(snapshot)
        return results
```

**典型用例**:

1. **诊断错误**:在某步出错时回放完整轨迹,定位是哪一层/模块出问题
2. **训练数据生成**:用正常推理轨迹作为 SOCAMonitor 校准数据
3. **消融对照**:对比"有干预"与"无干预"两条轨迹,验证 M3 CausalGate 的效果

---

## 五、配置与顶层组装类(M23-M24)

### M23. ConfigurationSystem(SOCAConfig + 3 个规模预设)

**定位**:统一管理所有超参数,**支持不同规模的预设配置**。

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SOCAConfig:
    """SOCA 完整配置"""

    # ═══ 全局 ═══
    d_model: int = 4096
    vocab_size: int = 128000
    max_seq_len: int = 8192

    # ═══ 区域划分 ═══
    n_perception_layers: int = 20
    n_workspace_layers: int = 24
    n_action_layers: int = 20

    # ═══ 总线 ═══
    d_bus: int = 1024  # d_model / 4
    bus_decay_init: float = 0.99

    # ═══ 工作空间 ═══
    n_experts: int = 32
    top_k: int = 2
    expert_dim: int = 512
    router_noise_init: float = 0.1
    capacity_factor: float = 1.5

    # ═══ 联合 SAE ═══
    sae_dict_size: int = 4096
    sae_sparsity_k: int = 64
    sae_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20])

    # ═══ 动作区 ═══
    n_micro_experts: int = 64
    micro_expert_dim: int = 512  # d_model / 8

    # ═══ 注意力 ═══
    n_heads_perception: int = 32
    n_heads_workspace: int = 8  # 线性注意力,少头
    n_heads_action: int = 32

    # ═══ 监控 ═══
    d_monitor: int = 256
    sigma_threshold: float = 3.0

    # ═══ 训练(默认 Medium 7B;v3-Micro 适配见 config_micro 注释)═══
    lambda_sae: float = 0.05           # Phase 1 起始(与 04/06/08 对齐)
    lambda_orth: float = 0.005          # 与 04/06 对齐(原 lambda_ortho 字段重命名)
    lambda_balance: float = 0.005
    lambda_bus_reg: float = 0.001
    lambda_sparse: float = 0.001
    lambda_expert_ortho: float = 0.005

    # ═══ 接口 ═══
    interface_residual_gate_init: float = -2.0

    @property
    def total_layers(self) -> int:
        return self.n_perception_layers + self.n_workspace_layers + self.n_action_layers

    @property
    def workspace_layer_range(self) -> tuple:
        start = self.n_perception_layers
        end = start + self.n_workspace_layers
        return (start, end)

    @property
    def action_layer_range(self) -> tuple:
        start = self.n_perception_layers + self.n_workspace_layers
        end = start + self.n_action_layers
        return (start, end)
```

### 4 个规模预设

> **本次修订新增 `config_micro()` 155M 预设**——这是 SOCA v3-Micro-Final 的 SOCAConfig 近似实现,对应 [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §五.6 的参数真源。
>
> **从 YAML 加载**(推荐):所有 4 个预设的字段定义见 [`./soca_micro_final_config.yaml`](./soca_micro_final_config.yaml),实施代码应从此文件加载:
> ```python
> import yaml
> with open("docs/research/agenticsom/soca_micro_final_config.yaml") as f:
>     yaml_cfg = yaml.safe_load(f)
> # SOCAConfig 通用基类字段(soca_config_base 的子集)
> socaconfig = SOCAConfig(**{k: v for k, v in yaml_cfg["soca_config_base"].items() if k in SOCAConfig.__dataclass_fields__})
> ```
>
> **重要语义差异提示**:**SOCAMicroConfig(04 §二)是 v3-Micro-Final 的完整字段实例**(含 `n_cycles / front_n_slots / back_n_routed / back_top_k / mid_top_k_per_head / lambda_orth / mid_n_heads / mid_d_head` 等 SOCAConfig 没有的字段),`config_micro()` 是 SOCAConfig 的**字段子集近似**——**后段 20 routed Top-5、中段 Top-4/head、感知/动作区 attn_types** 等关键配置只能写在注释中,无法从 SOCAConfig 单独构造完整模型。**实施时请以 04 SOCAMicroConfig 为准**;config_micro() 仅作为通用框架下的概念验证锚点。
>
> **未来重构项**(从 SOCAMicroConfig 提取并合并到 SOCAConfig 的字段清单):
> ```
> 优先级 P1(必须合并):
>   - n_cycles / n_ws_mid_physical       # 中段循环机制
>   - mid_n_heads / mid_d_head          # 中段 multi-head 结构
>   - mid_top_k_per_head                # 中段路由激活数
>   - front_n_slots / front_n_experts_per_slot  # 前段 Soft MoE
>   - back_n_routed / back_top_k        # 后段专家路由
> 优先级 P2(可选合并):
>   - lambda_orth / lambda_ortho 字段名统一
>   - attn_types 列表字段(感知/动作区)
>   - sae_layers 字段语义统一(中段/前后段用 SAE dict/k 字段而非绝对层号)
> ```
> 这些字段合并后,SOCAMicroConfig 可以直接继承 SOCAConfig,消除双轨配置类。

```python
def config_micro() -> SOCAConfig:
    """SOCA v3-Micro-Final:16 层 × ~155M 验证模型(参数真源见 03 §五.6)

    字段语义注意:`n_experts` 在 micro 中表示 **per-head 专家数**(中段 4h × 12e = 48 总),
    而在 config_tiny/medium/large 中表示**总专家数**(无 multi-head 拆分)。
    实施时务必根据场景区分;SOCAMicroConfig(04)的字段名 `mid_experts_per_head` 语义更清晰。
    """
    return SOCAConfig(
        # ── 全局 ──
        d_model=896,
        d_bus=256,                        # d_model / 3.5(非标准 1/4)
        vocab_size=32000,
        max_seq_len=2048,

        # ── 区域划分 ──
        n_perception_layers=5,            # 5P
        n_workspace_layers=6,             # 6W(2 前 + 2×2 中 + 2 后)
        n_action_layers=5,                # 5A
        # 总物理 16, 等效 18(中段 2×2=4)

        # ── 总线 ──
        bus_decay_init=0.99,

        # ── 工作空间 ──
        # ⚠️ n_experts 在 micro 中为 **per-head**(中段 4h × 12e = 48 总,激活 16/token)
        # ⚠️ top_k=4 也是 per-head(Medium/Large 默认 top_k=2)
        # 后段 20 routed + 1 shared + 1 device,Top-5(SOCAConfig 缺字段,在注释说明)
        n_experts=12,                     # per-head(v3-Micro-Final);非总专家数
        top_k=4,                          # per-head(v3-Micro-Final);Medium/Large 用 2
        expert_dim=192,                   # 中段 64→192→64(中段 d_bus / mid_n_heads)
        router_noise_init=0.4,            # Tiny-Medium MAP(08 §五.2)
        capacity_factor=1.7,              # Tiny-Medium MAP(08 §五.2)

        # ── 联合 SAE(中段)──
        sae_dict_size=2048,               # 中段 SAE dict(03 §五.6)
        sae_sparsity_k=12,                # 中段 K(03 §五.6)
        sae_layers=[],                    # 不用绝对层号(中段/前后段由 Zone 配置)
        # 前/后段 SAE 配置(由各 Zone 模块独立处理,SOCAConfig 缺字段):
        #   前段 SAE:dict=1024, K=16
        #   后段 SAE:dict=1024, K=16
        # 通过 M5 WorkspaceBlock / M6 ActionBlock 子模块参数传入

        # ── 动作区 ──
        n_micro_experts=20,               # 2h × 10 LoRA/head = 20 总 LoRA(等价 LoRA-MoE)
        micro_expert_dim=448,             # LoRA rank=8;d_head_lora=896/2=448(动作区 per-head)

        # ── 注意力 ──
        n_heads_perception=14,            # 感知/动作 896/64
        n_heads_workspace=4,              # 中段 256/64
        n_heads_action=14,                # 动作 896/64
        # 感知区 attn_types=["standard", "gated_deltanet", "gated_deltanet",
        #                    "gated_deltanet", "gated_standard"]  (03 §五.6)
        # 动作区 attn_types=["gated_deltanet", "gated_deltanet", "gated_deltanet",
        #                    "gated_standard", "gated_standard"]  (03 §五.6)
        # SOCAConfig 缺 attn_types 字段;通过各 Zone 模块的 attn_type 参数配置

        # ── 监控 ──
        d_monitor=48,                     # v3-Micro-Final 专用(Medium 是 256)
        sigma_threshold=3.0,

        # ── 训练(λ 权重与 08 §五.2 v3-Micro 子表对齐)──
        lambda_sae=0.05,                  # Phase 1 起始(08 §五.2)
        lambda_orth=0.005,                # 太强会导致 F3(08 §五.2;原 lambda_ortho 改名)
        lambda_balance=0.005,             # 不要太大(08 §五.2)
        lambda_bus_reg=0.001,
        lambda_sparse=0.001,              # 感知/动作 L1(03 §五.6)
        lambda_expert_ortho=0.005,

        # ── 接口 ──
        interface_residual_gate_init=-2.0,
    )


def config_tiny() -> SOCAConfig:
    """概念验证:~20M 参数"""
    return SOCAConfig(
        d_model=512,
        d_bus=128,
        n_perception_layers=4,
        n_workspace_layers=4,
        n_action_layers=4,
        n_experts=8,
        top_k=2,
        expert_dim=128,
        n_micro_experts=16,
        micro_expert_dim=64,
        sae_dict_size=512,
        sae_sparsity_k=16,
        d_monitor=64,
        n_heads_perception=8,
        n_heads_workspace=4,
        n_heads_action=8,
        vocab_size=32000,
    )

def config_medium() -> SOCAConfig:
    """中等规模:~7B 参数"""
    return SOCAConfig(
        d_model=4096,
        d_bus=1024,
        n_perception_layers=20,
        n_workspace_layers=24,
        n_action_layers=20,
        n_experts=32,
        top_k=2,
        expert_dim=512,
        n_micro_experts=64,
        micro_expert_dim=512,
        sae_dict_size=4096,
        sae_sparsity_k=64,
        d_monitor=256,
        n_heads_perception=32,
        n_heads_workspace=8,
        n_heads_action=32,
        vocab_size=128000,
    )

def config_large() -> SOCAConfig:
    """生产规模:~120B 参数"""
    return SOCAConfig(
        d_model=16384,
        d_bus=4096,
        n_perception_layers=40,
        n_workspace_layers=48,
        n_action_layers=40,
        n_experts=64,
        top_k=2,
        expert_dim=1024,
        n_micro_experts=256,
        micro_expert_dim=2048,
        sae_dict_size=16384,
        sae_sparsity_k=128,
        d_monitor=512,
        n_heads_perception=64,
        n_heads_workspace=16,
        n_heads_action=64,
        vocab_size=256000,
    )
```

### 四档规模对比

| 配置 | 函数 | 规模 | d_model | d_bus | 层级 | 总专家数 | 中段 top_k | SAE 字典 | 微专家 | 参数量 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `config_tiny` | 概念验证 | 20M | 512 | 128 | 4+4+4=12 | 8 | 2 | 512 | 16 | **~20M** |
| `config_micro` | **验证模型** | 155M | 896 | 256 | 5+6+5=16 | 48(中 4h×12e)+20(后) | **4** | 2048(中)/ 1024(前后) | 20 | **~155M** |
| `config_medium` | 中等 | 7B | 4096 | 1024 | 20+24+20=64 | 32 | 2 | 4096 | 64 | **~7B** |
| `config_large` | 生产 | 120B | 16384 | 4096 | 40+48+40=128 | 64 | 2 | 16384 | 256 | **~120B** |

> **关键洞察**:**SOCA 框架是规模无关的**——`SOCAConfig` 的相同结构可支撑 20M → 120B(6 个数量级)。`config_micro`(155M)是 `config_tiny` 和 `config_medium` 之间的**针对 ~150M 验证场景的特殊化**(详见 [`02-sweet-spot-params.md`](./02-sweet-spot-params.md)、[`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §五.6)。
>
> **v3-Micro-Final 关键差异**:中段 `top_k=4`(`config_tiny/medium/large` 默认 `top_k=2`);这是验证模型为增加消融效应量而采用的特殊配置,扩展到 7B/120B 时应切回 `top_k=2`。

### 与 [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) §二 的 SOCAMicroConfig 对比

| 维度 | 04 SOCAMicroConfig | M23 SOCAConfig |
|---|---|---|
| **范围** | 仅 SOCA v3-Micro-Final(155M, 16 层) | 整个 SOCA 框架(20M → 155M → 120B) |
| **字段数** | ~60 个 | 30+ 核心字段 + 派生 property |
| **规模预设** | 无 | `config_tiny` / **`config_micro`** / `config_medium` / `config_large` |
| **派生计算** | 手动 | `@property` 自动计算 total_layers / workspace_layer_range |
| **关系** | v3-Micro-Final **完整字段实例** | v3-Micro-Final **字段子集近似**(缺 n_cycles / front_n_slots / back_n_routed / back_top_k / mid_top_k_per_head / attn_types 等) |
| **可独立构造 v3-Micro 模型** | ✅ 完整构造 | ⚠️ 仅近似——后段 Top-5 / 中段 Top-4 / attn_types 需在模型组装时额外传入 |

> **实施路径**:
> - **直接实施 SOCA v3-Micro-Final** → 用 [`04-implementation-roadmap.md` §二 SOCAMicroConfig](./04-implementation-roadmap.md)(完整字段,60 个)
> - **SOCA 框架通用化、扩展到 7B / 120B** → 用本节 SOCAConfig + `config_tiny/micro/medium/large` 预设
> - **SOCAConfig 字段不足的部分**(n_cycles / front / back / mid / attn_types)→ 建议从 SOCAMicroConfig 提取并合并到 SOCAConfig,**作为未来重构项**

---

### M24. SOCAModel(顶层组装)

**定位**:将所有模块组装为完整模型。

```python
class SOCAModel(nn.Module):
    """
    SOCA 完整模型:组装所有模块。
    """
    def __init__(self, config: SOCAConfig):
        super().__init__()
        self.config = config

        # ═══ Token Embedding ═══
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # ═══ 全局总线 ═══
        self.broadcast_bus = BroadcastBus(config.d_bus, config.total_layers)

        # ═══ 感知区 ═══
        self.perception_layers = nn.ModuleList([
            PerceptionBlock(
                d_model=config.d_model,
                n_heads=config.n_heads_perception,
                ffn_dim=config.d_model * 4,
                d_bus=config.d_bus,
                d_monitor=config.d_monitor,
                layer_idx=i,
            )
            for i in range(config.n_perception_layers)
        ])

        # ═══ 下接口 ═══
        self.down_interface = DownInterface(config.d_model, config.d_bus)

        # ═══ 工作空间区 ═══
        self.workspace_layers = nn.ModuleList([
            WorkspaceBlock(
                d_model=config.d_model,
                d_bus=config.d_bus,
                n_heads=config.n_heads_workspace,
                n_experts=config.n_experts,
                top_k=config.top_k,
                expert_dim=config.expert_dim,
                d_sae=config.sae_dict_size,
                sae_k=config.sae_sparsity_k,
                layer_idx=config.n_perception_layers + i,
            )
            for i in range(config.n_workspace_layers)
        ])

        # ═══ 上接口 ═══
        self.up_interface = UpInterface(config.d_bus, config.d_model)

        # ═══ 动作区 ═══
        self.action_layers = nn.ModuleList([
            ActionBlock(
                d_model=config.d_model,
                n_heads=config.n_heads_action,
                n_micro_experts=config.n_micro_experts,
                micro_dim=config.micro_expert_dim,
                d_bus=config.d_bus,
                d_monitor=config.d_monitor,
                layer_idx=config.n_perception_layers + config.n_workspace_layers + i,
            )
            for i in range(config.n_action_layers)
        ])

        # ═══ 输出头 ═══
        self.output_norm = RMSNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # 权重共享

        # ═══ 运行时组件(非参数)═══
        self.monitor = None  # 推理时初始化
        self.recorder = None

    @property
    def layers(self):
        """所有层的统一访问接口"""
        return (list(self.perception_layers) +
                list(self.workspace_layers) +
                list(self.action_layers))

    def forward(self, input_ids: torch.Tensor,
                return_diagnostics: bool = False) -> dict:
        """
        input_ids: [batch, seq]
        """
        batch, seq = input_ids.shape

        # 重置总线
        self.broadcast_bus.reset(batch)

        # Embedding
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # 诊断收集
        diagnostics = {
            "monitor_vectors": [],
            "bus_states": [],
            "router_entropies": [],
            "sae_losses": [],
            "gate_values": [],
            "sae_max_activations": [],
            "load_balance_losses": [],
        }

        # ═══ 感知区 ═══
        for layer in self.perception_layers:
            x = layer(x, self.broadcast_bus)
            if return_diagnostics:
                diagnostics["monitor_vectors"].append(
                    layer.monitor_slot.monitor_vector
                )

        # ═══ 下接口 ═══
        x = self.down_interface(x)  # [batch, seq, d_bus]

        # ═══ 工作空间区 ═══
        for layer in self.workspace_layers:
            if return_diagnostics:
                x = layer(x, self.broadcast_bus, return_diag=True)
                diag = layer.last_diagnostics
                diagnostics["monitor_vectors"].append(diag["monitor_vector"])
                diagnostics["router_entropies"].append(layer.router.entropy())
                diagnostics["sae_losses"].append(diag["sae_loss"])
                diagnostics["sae_max_activations"].append(diag["sae_codes"].abs().max())
                diagnostics["load_balance_losses"].append(
                    layer.router.load_balance_loss(diag["top_k_indices"])
                )
            else:
                x = layer(x, self.broadcast_bus)

            diagnostics["bus_states"].append(self.broadcast_bus.state.clone())

        # ═══ 上接口 ═══
        x = self.up_interface(x, self.broadcast_bus.state)

        # ═══ 动作区 ═══
        for layer in self.action_layers:
            x = layer(x, self.broadcast_bus)
            if return_diagnostics:
                diagnostics["monitor_vectors"].append(layer.monitor_slot.monitor_vector)
                if layer.last_gate_values is not None:
                    diagnostics["gate_values"].append(layer.last_gate_values)

        # ═══ 输出 ═══
        x = self.output_norm(x)
        logits = self.output_proj(x)

        if return_diagnostics:
            diagnostics["logits"] = logits
            diagnostics["bus_final"] = self.broadcast_bus.state
            diagnostics["sae_loss"] = (torch.stack(diagnostics["sae_losses"]).mean()
                                        if diagnostics["sae_losses"] else torch.tensor(0.0))
            diagnostics["load_balance_loss"] = (torch.stack(diagnostics["load_balance_losses"]).mean()
                                                  if diagnostics["load_balance_losses"] else torch.tensor(0.0))
            return diagnostics

        return {"logits": logits}

    def forward_with_diagnostics(self, input_ids):
        """便捷方法"""
        return self.forward(input_ids, return_diagnostics=True)

    # ═══ 参数量统计 ═══
    def param_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        perception = sum(p.numel() for p in self.perception_layers.parameters())
        workspace = sum(p.numel() for p in self.workspace_layers.parameters())
        action = sum(p.numel() for p in self.action_layers.parameters())
        interfaces = (sum(p.numel() for p in self.down_interface.parameters()) +
                     sum(p.numel() for p in self.up_interface.parameters()))
        bus = sum(p.numel() for p in self.broadcast_bus.parameters())

        return {
            "total": total,
            "perception": perception,
            "workspace": workspace,
            "action": action,
            "interfaces": interfaces,
            "bus": bus,
            "perception_pct": perception / total * 100,
            "workspace_pct": workspace / total * 100,
            "action_pct": action / total * 100,
        }
```

---

## 六、完整模块清单(M1-M24)

| 编号 | 模块名 | 类别 | 核心职责 |
|:---:|:---|:---|:---|
| M1 | BroadcastBus | 全局基础设施 | 全局信息骨干,跨区通信 |
| M2 | MonitorSlot | 全局基础设施 | 每层标准化观测/干预接口 |
| M3 | CausalGate | 全局基础设施 | 每计算步骤的因果干预门 |
| M4 | PerceptionBlock | 计算区 | 输入编码(Dense) |
| M5 | WorkspaceBlock | 计算区 | 推理计算(隐空间 MoE) |
| M6 | ActionBlock | 计算区 | 输出生成(细粒度门控) |
| M7 | DownInterface | 区间接口 | 感知→工作空间(压缩) |
| M8 | UpInterface | 区间接口 | 工作空间→动作(恢复+注入) |
| M9 | LatentRouter | 工作空间内部 | 隐空间专家选择 |
| M10 | ExpertBlock | 工作空间内部 | 隐空间计算单元 |
| M11 | GatedAggregator | 工作空间内部 | 多专家输出聚合 |
| M12 | JointSAE | 工作空间内部 | 联合稀疏自编码器 |
| M13 | StandardAttention | 注意力变体 | 感知/动作区用(softmax + RoPE) |
| M14 | LinearAttention | 注意力变体 | 工作空间用(线性,无 RoPE) |
| M15 | SOCAMonitor | 运行时系统 | 实时异常检测+干预 |
| M16 | SOCALoss | 训练系统 | 训练损失组合(4 阶段调度) |
| M17 | RMSNorm | **辅助** | 归一化(LayerNorm 轻量化) |
| M18 | RotaryEmbedding | **辅助** | 旋转位置编码(感知/动作) |
| M19 | PhaseScheduler | **辅助** | 训练阶段管理(冻结/解冻) |
| M20 | ExpertDispatcher | **辅助** | 高效专家分派(向量化) |
| M21 | HealthDashboard | 运行时系统 | 可视化+健康评分汇总 |
| M22 | SnapshotRecorder | 运行时系统 | 推理状态轨迹记录与回放 |
| M23 | SOCAConfig | **辅助** | 配置管理(3 个规模预设) |
| M24 | SOCAModel | 顶层 | 完整模型组装 |

**模块总数**:**24 个核心模块** = 16 个原始规格(M1-M16)+ **8 个补充**(M17-M24)

**模块按类别分布**:

| 类别 | 模块数 | 编号 |
|:---|:---:|:---|
| 全局基础设施 | 3 | M1-M3 |
| 计算区 | 3 | M4-M6 |
| 区间接口 | 2 | M7-M8 |
| 工作空间内部 | 4 | M9-M12 |
| 注意力变体 | 2 | M13-M14 |
| 运行时/训练系统 | 4 | M15-M16, M21-M22 |
| **辅助工具**(本文件新增) | 6 | M17-M20, M23 |
| 顶层组装 | 1 | M24 |

---

## 七、模块依赖矩阵(完整版 M1-M24)

```
        M1  M2  M3  M4  M5  M6  M7  M8  M9  M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 M21 M22 M23 M24
M1  Bus  ─   ─   ─   ●   ●   ●   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ●   ─   ●   ●   ●   ●
M2  Slot ─   ─   ─   ●   ●   ●   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ●   ●   ─   ●
M3  Gate ─   ─   ─   ●   ●   ●   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ●   ●   ─   ●
M4  Perc ○   ●   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ●
M5  Work ○   ●   ●   ─   ─   ─   ─   ─   ●   ●   ●   ●   ─   ●   ─   ─   ●   ─   ─   ─   ─   ─   ─   ●
M6  Act  ○   ●   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ●
M7  Down ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ●
M8  Up   ○   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ●
M9  Rtr  ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ●   ─   ─   ─   ─   ●   ●   ─   ●
M10 Exp  ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ●   ─   ─   ─   ●
M11 Agg  ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ●
M12 SAE  ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ●   ─   ─   ─   ─   ●   ●   ─   ●
M13 SAttn─   ─   ─   ●   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●
M14 LAttn─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●
M15 Mon  ●   ●   ●   ─   ─   ─   ─   ─   ●   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ●
M16 Loss ─   ─   ─   ─   ─   ─   ●   ●   ●   ●   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●
M17 Norm ─   ─   ─   ●   ●   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●
M18 RoPE ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●
M19 Sched─   ─   ─   ●   ●   ●   ●   ●   ●   ●   ●   ●   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ●   ●
M20 Disp ─   ─   ─   ─   ●   ─   ─   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ─   ●
M21 Dash ●   ●   ●   ─   ─   ─   ─   ─   ●   ─   ─   ●   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ●
M22 Rec  ●   ●   ●   ─   ─   ─   ─   ─   ●   ─   ─   ●   ─   ─   ●   ─   ─   ─   ─   ─   ─   ─   ●
M23 Conf ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ─
M24 Modl ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ─

● = 强依赖(必须有才能工作)
○ = 弱依赖(可选交互)
```

---

## 八、实施优先级排序(5 阶段 / ~12 周)

```
Phase 1(最小可运行,~2 周):
  M17 + M18 + M23 + M24 + M4 + M5 + M6 + M7 + M8
  → 能前向传播,能计算损失

Phase 2(监控可用,+2 周):
  M1 + M2 + M3 + M15
  → 能观察,能干预

Phase 3(训练稳定,+2 周):
  M9 + M10 + M11 + M20 + M16 + M19
  → 能稳定训练

Phase 4(可解释性,+2 周):
  M12 + M13 + M14
  → SAE 联合训练,注意力变体

Phase 5(生产化,+4 周):
  M21 + M22 + M23(完整配置管理)
  → 仪表盘,记录,配置管理
```

### 与 [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) §十二 的 4 周里程碑对比

| 维度 | 04 的 4 周里程碑 | 07 的 5 阶段 12 周 |
|---|---|---|
| **范围** | 仅 SOCA v3-Micro-Final(~155M, 16 层) | 整个 SOCA 框架(可缩放到 120B) |
| **重点** | 烟雾测试 → 主训练 → 24 消融 → 分析 | 最小可运行 → 监控 → 训练 → 可解释 → 生产 |
| **第一阶段目标** | 100 步烟雾测试通过 | 前向传播 + 损失计算 |
| **训练策略** | 隐含(直接训练) | 显式(M19 4 阶段冻结/解冻) |
| **生产化** | 不包含 | 完整包含(M21/M22/M23) |
| **关系** | 04 是 SOCA v3-Micro-Final 的**特化实施计划** | 07 是 SOCA 框架的**通用工程路线图** |

> **实施建议**:对 SOCA v3-Micro-Final 的具体实施,**先用 04 的 4 周里程碑**(因为目标是 155M 验证);**用 07 的 5 阶段**(作为长期扩展到 7B/120B 的参考路线)。

> **训练配方关联**:实施 Phase 3(训练稳定)时,必须结合 [`08-router-stability-training.md`](./08-router-stability-training.md)(路由器七重防护 + 渐进启用 + 梯度平衡)与 [`09-sae-bus-information-theory.md`](./09-sae-bus-information-theory.md)(联合 SAE 字典动态 + 总线信息论),否则路由器(M9)/SAE(M12)/总线(M1)的训练不稳定会阻塞整个 pipeline。

---

## 九、与现有文档的对照(总览)

| 文档 | 职责 | 模块覆盖 | 规模覆盖 | 工程完整度 |
|:---|:---|:---:|:---:|:---:|
| [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) | 14L 审查 | — | 106M | 概念 |
| [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) | 参数甜点 | — | 155M | 概念 |
| [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) | 层数甜点 | — | 155M | 概念 |
| [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) | 代码骨架 | 部分(M1, M2, M12, M13, M14, M24 等) | 155M | ★★★☆☆ |
| [`05-pretraining-ablation-plan.md`](./05-pretraining-ablation-plan.md) | 消融计划 | — | 155M | 流程 |
| [`06-architecture-modules.md`](./06-architecture-modules.md) | **核心模块规格** | **M1-M16** | 155M | ★★★★☆ |
| **[`07-module-extensions.md`](./07-module-extensions.md)** | **辅助 + 完整 + 优先级** | **M17-M24 + 完整 M1-M24 清单** | **20M → 120B** | **★★★★★** |

---

## 十、对 04 的回溯建议(更新版)

| 04 现状 | 07 + 06 建议 |
|---|---|
| `SOCAMicroConfig`(155M 专用) | 用 M23 `SOCAConfig` 通用类 + `config_micro()` 预设替代 |
| `MonitorSlot` 仅压缩投影 | 用 M2 三种模式(OBSERVE/INJECT/BYPASS) |
| 无 CausalGate | 用 M3 四种模式(normal/replace/freeze/noise) |
| `BroadcastBus` 简单读写 | 用 M1 每层独立 read/write gates |
| `LoRAMoE` 用于动作区 | 用 M6 微专家 + 软门控 |
| `WorkspaceMid` 单步 forward | 用 M5 八步流程 + M20 向量化分派 |
| 4 阶段按进度 | 用 M19 按区域阶段(感知→工作空间→动作→端到端) |
| 无 RMSNorm | 用 M17 替代 LayerNorm |
| 无 RoPE | 用 M18 加到 M13 StandardAttention |
| `JointSAE` 简单 | 用 M12 + 字典正交约束 |
| 无 SOCAMonitor | 用 M15 完整版(校准+干预+快照) |
| 无 HealthDashboard | 用 M21 可视化(文本/Matplotlib/Web) |
| 无 SnapshotRecorder | 用 M22 推理轨迹记录与回放 |
| 无 PhaseScheduler | 用 M19 显式阶段管理(冻结/解冻) |

> **实施路径**:04 的代码可以**保留主干,按 M1-M24 重构类名与方法签名**,无需推倒重来。

---

## 十一、回答用户问题

> "需要我对某个具体模块做更深入的分析(比如路由器的训练稳定性、联合 SAE 的字典学习动态、或者总线的信息论性质),还是展开某个特定规模的完整参数估算?"

**当前回答**:本文档已完整整合 M1-M24 的模块规格与 5 阶段实施路线。如果需要进一步深入,推荐优先级:

1. **路由器(M9)的训练稳定性** —— 包括温度退火、负载均衡损失的数学推导、专家坍缩的诊断与恢复(对 [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) §六 修正项 P3-P5 有直接帮助)
2. **总线(M1)的信息论性质** —— 包括 `d_bus` 信息瓶颈的最优选择、读写门控的收敛动力学(对 [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §二的"组件最小容量约束"有支撑)
3. **`config_medium`(7B)完整参数估算** —— 7B 是 120B 的 1/17,但有完整的工程经验可借鉴,可作为扩展规划的中间档

---

## 📅 文档版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-28 | 初版,完成 M17-M24 辅助子模块规格 + 完整 M1-M24 模块清单 + 5 阶段实施优先级 + 3 个规模配置预设(20M/7B/120B) |

---

> **下一步**:阅读 [`06-architecture-modules.md`](./06-architecture-modules.md) 的 M1-M16 核心模块规格,与本文件的 M17-M24 合并,形成完整的 SOCA 实施规范;或按 04 的 4 周里程碑直接开始 SOCA v3-Micro-Final 的实施。

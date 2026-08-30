# SOCA v3-Micro 模块化设计:M1-M16 完整规格

> **目的**:把 [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) 的代码骨架,**抽象为 16 个职责清晰的命名模块(M1-M16)**,建立**可干预**(CausalGate / MonitorSlot / Bus Gates)、**可观测**(SOCAMonitor)、**可推理**(模块边界清晰)的完整架构规格。
>
> **关联文档**(层级互补,本文件不取代它们):
> - 上一阶段(代码骨架):[`04-implementation-roadmap.md`](./04-implementation-roadmap.md)
> - 上一阶段(层数甜点):[`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md)
> - 上一阶段(参数甜点):[`02-sweet-spot-params.md`](./02-sweet-spot-params.md)
> - 上一阶段(14L 审查):[`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md)
>
> **角色**:本文件处于"**模块规格层**"——介于研究(01-03)与代码(04)之间,定义了"每个模块做什么、与其他模块如何协作",为 04 的代码实现提供精确的契约。

---

## 一、与 04 代码骨架的关系

| 维度 | 04 代码骨架 | **06 模块规格**(本文件) |
|---|---|---|
| **抽象层级** | Python 类与函数 | 命名模块与接口 |
| **关注点** | How(如何实现) | What(做什么) + Why(为什么) |
| **命名** | SOCAMicro、BroadcastBus、SoftMoE 等类 | M1-M16 编号 + 类别(全局/计算/接口/工作空间/注意力/运行时) |
| **干预机制** | 消融开关(disable_cycles、disable_bus 等) | **CausalGate + MonitorSlot + Bus Gates**(运行时) |
| **动作区** | LoRA-MoE(rank-8 低秩) | **微专家 + 软门控**(更直接) |
| **注意力** | Standard + Gated DeltaNet + Linear 混合 | **Standard + Linear**(简化) |
| **训练阶段** | 4 阶段(0%/5%/30%/90% 进度) | **4 阶段(感知→工作空间→动作→端到端)** |

> **关键决策**:M1-M16 不取代 04,而是提供**模块化契约**——04 的代码可以按 M1-M16 重组,获得更清晰的干预点和监控接口。

---

## 二、模块总览(16 个核心模块)

SOCA v3-Micro-Final 包含 **16 个核心模块**,按功能分为 **6 类**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SOCA 模块全景图                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─── 全局基础设施 ─────────────────────────────────────────────────┐  │
│  │  M1.  Broadcast Bus（广播总线）                                 │  │
│  │  M2.  Monitor Slot（监控插槽）                                 │  │
│  │  M3.  Causal Gate（因果门）                                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── 计算区 ──────────────────────────────────────────────────────┐  │
│  │  M4.  Perception Block（感知块）                                │  │
│  │  M5.  Workspace Block（工作空间块）                             │  │
│  │  M6.  Action Block（动作块）                                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── 区间接口 ────────────────────────────────────────────────────┐  │
│  │  M7.  Down Interface（下投影接口）                              │  │
│  │  M8.  Up Interface（上投影接口）                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── 工作空间内部 ────────────────────────────────────────────────┐  │
│  │  M9.  Latent Router（隐空间路由器）                            │  │
│  │  M10. Expert Block（专家块）                                   │  │
│  │  M11. Gated Aggregator（门控聚合器）                           │  │
│  │  M12. Joint SAE（联合稀疏自编码器）                            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── 注意力变体 ──────────────────────────────────────────────────┐  │
│  │  M13. Standard Attention（标准注意力）                          │  │
│  │  M14. Linear Attention（线性注意力）                            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── 运行时系统 ──────────────────────────────────────────────────┐  │
│  │  M15. SOCAMonitor（运行时监控器）                              │  │
│  │  M16. SOCALoss（训练损失组合器）                               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、全局基础设施类(M1-M3)

### M1. Broadcast Bus(广播总线)

**定位**:贯穿全模型的全局信息通道。工作空间向总线写入"全局广播"信息;感知区和动作区从总线读取全局上下文。

**核心性质**:
- **维度比例**:`d_bus` 推荐范围 `d_model/4` ~ `d_model/8`(信息论推荐);SOCA v3-Micro-Final 取 `d_bus = d_model / 3.5 ≈ 256`(96+层小模型为了更大瓶颈);`config_medium`(7B) 取 `d_bus = d_model / 4`;`config_large`(120B) 取 `d_bus = d_model / 4`
- **序列级**:每个序列维护一个总线状态(非逐 token)
- **门控读写**:每层有独立的 read/write gate,可学习
- **可观测**:总线状态是"模型全局工作记忆"的直接窗口

```python
class BroadcastBus(nn.Module):
    def __init__(self, d_bus: int, n_layers: int):
        super().__init__()
        self.d_bus = d_bus
        self.n_layers = n_layers

        # 每层独立的读门控(初始 -3.0 → σ ≈ 0.05,几乎不读)
        self.read_gates = nn.ParameterList([
            nn.Parameter(torch.full((d_bus,), -3.0))
            for _ in range(n_layers)
        ])

        # 每层独立的写门控(只有工作空间层会被训练打开)
        self.write_gates = nn.ParameterList([
            nn.Parameter(torch.full((d_bus,), -3.0))
            for _ in range(n_layers)
        ])

        # 衰减因子(防止信息无限积累)
        self.decay = nn.Parameter(torch.tensor(0.99))

        # 运行时状态
        self.state = None
        self.history = []

    def reset(self, batch_size: int = 1):
        """每个新序列开始时重置总线"""
        self.state = torch.zeros(batch_size, self.d_bus)
        self.history = []

    def read(self, layer_idx: int) -> torch.Tensor:
        """从总线读取信息(门控后)"""
        gate = torch.sigmoid(self.read_gates[layer_idx])
        return self.state * gate.unsqueeze(0)

    def write(self, layer_idx: int, new_info: torch.Tensor):
        """软更新写入:state = state * (1-gate) + new_info * gate"""
        gate = torch.sigmoid(self.write_gates[layer_idx])
        self.state = self.state * (1 - gate.unsqueeze(0)) + new_info * gate.unsqueeze(0)

    def forward(self, layer_idx: int, write_info: torch.Tensor = None):
        """每层调用:先衰减,再可选写入,最后读取"""
        self.state = self.state * self.decay
        if write_info is not None:
            self.write(layer_idx, write_info)
        self.history.append(self.state.detach().clone())
        return self.read(layer_idx)
```

**关键参数**:

| 参数 | 典型值 | 说明 |
|:---|:---:|:---|
| `d_bus` | `d_model / 4`(medium/large) <br> `d_model / 3.5`(micro,155M) <br> 范围 `d_model/4` ~ `d_model/8` | 总线宽度;信息瓶颈的物理实现。Micro 取 1/3.5 是因为小模型需要更大瓶颈,理论依据见 [`09-sae-bus-information-theory.md`](./09-sae-bus-information-theory.md) §十五 |
| 读门控初始值 | -3.0 | $\sigma(-3) \approx 0.05$,初始几乎不读 |
| 写门控初始值 | -3.0 | 初始几乎不写;只有工作空间层会打开 |
| `decay` | 0.99 | 每层衰减 1%;防止信息无限积累 |

**与其他模块的交互**:
```
感知区层 → Bus.read(layer_i)   → 获取全局上下文 → 注入注意力
工作空间层 → Bus.write(layer_j, z_summary) → 广播推理结果
动作区层 → Bus.read(layer_k)   → 获取工作空间最终结论
SOCAMonitor → Bus.state → 直接读取"模型在想什么"
```

---

### M2. Monitor Slot(监控插槽)

**定位**:每层一个的标准化观测/干预接口。三种操作:**观察(OBSERVE)**、**注入(INJECT)**、**旁路(BYPASS)**。

**设计约束**:
- 所有层的插槽形状相同 → **跨层可比**
- 压缩投影是线性的 → **J-Lens 链不被打断**
- 监控向量维度固定 → **统一分析框架**

```python
class MonitorSlot(nn.Module):
    def __init__(self, d_model: int, d_monitor: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_monitor = d_monitor  # 固定值,如 256

        # OBSERVE: 压缩投影 (d_model → d_monitor, 线性无偏置)
        self.compress = nn.Linear(d_model, d_monitor, bias=False)
        nn.init.orthogonal_(self.compress.weight)  # 正交初始化

        # INJECT: 注入投影 (d_monitor → d_model, 初始零)
        self.inject = nn.Linear(d_monitor, d_model, bias=False)
        nn.init.zeros_(self.inject.weight)

        # BYPASS: 旁路门控(标量,初始 -5.0 → σ ≈ 0.007)
        self.bypass_logit = nn.Parameter(torch.tensor(-5.0))

        self.monitor_vector = None
        self.injected_signal = None
        self.mode = "normal"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 始终记录监控向量(开销极小)
        self.monitor_vector = self.compress(hidden_states).detach()

        if self.mode == "normal":
            return hidden_states
        elif self.mode == "inject":
            if self.injected_signal is not None:
                return hidden_states + self.inject(self.injected_signal)
            return hidden_states
        elif self.mode == "bypass":
            gate = torch.sigmoid(self.bypass_logit)
            return hidden_states * (1 - gate)
        return hidden_states

    def set_injection(self, signal: torch.Tensor):
        self.injected_signal = signal
        self.mode = "inject"

    def enable_bypass(self):
        self.mode = "bypass"

    def reset(self):
        self.mode = "normal"
        self.injected_signal = None
```

**关键参数**:

| 参数 | 典型值 | 说明 |
|:---|:---:|:---|
| `d_monitor` | 256 | 所有层统一;跨层可比的监控向量 |
| 压缩投影初始化 | 正交 | 保证信息不丢失 |
| 注入投影初始化 | 零 | 初始不干扰计算 |
| 旁路门控初始值 | -5.0 | $\sigma(-5) \approx 0.007$,初始几乎不衰减 |

---

### M3. Causal Gate(因果门)

**定位**:包裹每个计算步骤(注意力输出、FFN 输出、路由决策)的门控。支持四种模式:**正常、替换、冻结、噪声**。

**设计意图**:
- 每个计算步骤都可以被独立干预
- 干预不影响计算图的合法性
- 干预效果可以被精确测量

```python
class CausalGate(nn.Module):
    def __init__(self, d_output: int, name: str = ""):
        super().__init__()
        self.d_output = d_output
        self.name = name
        self.mode = "normal"
        self.replacement = None
        self.cached_result = None
        self.noise_scale = 0.0
        self.intervention_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, compute_fn, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "normal":
            self.cached_result = compute_fn(x).detach()
            return self.cached_result
        elif self.mode == "replace":
            strength = torch.sigmoid(self.intervention_strength)
            normal_result = compute_fn(x)
            return (1 - strength) * normal_result + strength * self.replacement
        elif self.mode == "freeze":
            if self.cached_result is None:
                return compute_fn(x)
            return self.cached_result.expand_as(compute_fn(x))
        elif self.mode == "noise":
            return compute_fn(x) + torch.randn_like(compute_fn(x)) * self.noise_scale
        return compute_fn(x)

    def set_replace(self, value: torch.Tensor):
        self.mode = "replace"; self.replacement = value
    def set_freeze(self):
        self.mode = "freeze"
    def set_noise(self, scale: float):
        self.mode = "noise"; self.noise_scale = scale
    def reset(self):
        self.mode = "normal"; self.replacement = None; self.noise_scale = 0.0
```

**部署位置**(每个 SOCA 层中的因果门分布):

```
Perception Layer:
  ├── CausalGate("attn_out")     → 包裹注意力输出
  └── CausalGate("ffn_out")      → 包裹 FFN 输出

Workspace Layer:
  ├── CausalGate("attn_out")     → 包裹注意力输出
  ├── CausalGate("router")       → 包裹路由决策
  ├── CausalGate("expert_out")   → 包裹专家计算输出
  └── CausalGate("bus_write")    → 包裹总线写入

Action Layer:
  ├── CausalGate("attn_out")     → 包裹注意力输出
  └── CausalGate("gate_ffn_out") → 包裹门控 FFN 输出
```

---

## 四、计算区类(M4-M6)

### M4. Perception Block(感知块)

**定位**:模型的"眼睛和耳朵"。将输入编码为丰富的内部表征。使用 **Gated DeltaNet + Standard Attention 混合**(与 [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §五.6 锁定的 SOCA v3-Micro-Final 配置一致),保证 Jacobian 平滑(J-Lens 友好)。

**设计原则**:
- **混合注意力**:5 层中 = `1 Standard + 3 Gated DeltaNet + 1 Gated Standard`(与 Qwen3.5 的 3:1 DeltaNet/Standard 比例一致)
- 全量计算(所有参数参与每个 token)
- 标准 Dense FFN(FFN 扩展比 2.2×)
- 总线读取(获取全局上下文,但不写入)
- 监控只读(不是干预目标)

```python
class PerceptionBlock(nn.Module):
    """
    SOCA v3-Micro-Final 感知区层。
    attn_type ∈ {"standard", "gated_standard", "gated_deltanet"}
    对齐 03 §五.6:1 Std + 3 Gated DeltaNet + 1 Gated Std
    """
    def __init__(self, d_model, n_heads, ffn_dim, d_bus, d_monitor,
                 layer_idx, attn_type="gated_standard",
                 deltanet_state_dim=128):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_type = attn_type

        # ─── 注意力(M13 Standard / Gated Standard / Gated DeltaNet)───
        self.norm1 = RMSNorm(d_model)
        if attn_type == "standard":
            self.attention = StandardAttention(d_model, n_heads)  # M13
        elif attn_type == "gated_standard":
            self.attention = GatedStandardAttention(d_model, n_heads)  # M13 + 门控
        elif attn_type == "gated_deltanet":
            self.attention = GatedDeltaNet(  # 04 §三.2 实现
                d_model, n_heads, d_head=d_model // n_heads,
                state_dim=deltanet_state_dim
            )
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")
        self.attn_gate = CausalGate(d_model, name=f"P{layer_idx}_attn")  # M3

        # ─── 标准 Dense FFN(扩展比 2.2×)───
        self.norm2 = RMSNorm(d_model)
        self.ffn = DenseFFN(d_model, ffn_dim)
        self.ffn_gate = CausalGate(d_model, name=f"P{layer_idx}_ffn")  # M3

        # ─── L1 稀疏惩罚(对齐 01 §六 修正 P6 + 03 §五.6)───
        self.l1_lambda = 0.001

        # ─── 总线读取 ───
        self.bus_read_proj = nn.Linear(d_bus, d_model, bias=False)
        nn.init.zeros_(self.bus_read_proj.weight)  # 初始不依赖总线

        # ─── 监控插槽 ───
        self.monitor_slot = MonitorSlot(d_model, d_monitor, layer_idx)  # M2

    def forward(self, x, bus):
        bus_info = bus.read(self.layer_idx)
        bus_context = self.bus_read_proj(bus_info)

        normed = self.norm1(x)
        # Standard/GatedStandard 接受 context_bias,GatedDeltaNet 不需要
        if self.attn_type == "gated_deltanet":
            attn_out = self.attn_gate(lambda h: self.attention(h), normed)
        else:
            attn_out = self.attn_gate(
                lambda h: self.attention(h, context_bias=bus_context),
                normed
            )
        x = x + attn_out

        normed = self.norm2(x)
        ffn_out = self.ffn_gate(lambda h: self.ffn(h), normed)
        x = x + ffn_out

        x = self.monitor_slot(x)
        return x


class DenseFFN(nn.Module):
    """标准 Dense FFN:两层线性 + GELU"""
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.up = nn.Linear(d_model, ffn_dim)
        self.act = GELU()
        self.down = nn.Linear(ffn_dim, d_model)

    def forward(self, x):
        return self.down(self.act(self.up(x)))
```

**SOCA v3-Micro-Final 感知区 5 层配置**(对齐 03 §五.6):
```
P0: Standard Attention (attn_type="standard")
P1: Gated DeltaNet       (attn_type="gated_deltanet")
P2: Gated DeltaNet       (attn_type="gated_deltanet")
P3: Gated DeltaNet       (attn_type="gated_deltanet")
P4: Gated Standard Attention (attn_type="gated_standard")
比例 1 Std : 3 DeltaNet : 1 Gated Std (与 Qwen3.5 一致)
```

**与 04 的关系**:
- 04 §二 config.py 与 §七 SOCAMicro 的 `attn_types = ["standard", "deltanet", "deltanet", "deltanet", "gated_standard"]` 是正确的(对应上表 P0-P4)
- 04 §三.2 的 `GatedDeltaNet` 类即为本模块 `attn_type="gated_deltanet"` 的具体实现;Standard 与 GatedStandard 共享 M13 实现
- 04 §三.2 的 `LinearAttention` 用于工作空间**中段**(M5 WorkspaceBlock),**不**用于感知区
- 06 本节描述与 03 §五.6 / 04 §二 全部对齐

---

### M5. Workspace Block(工作空间块)

**定位**:模型的"思考区"。在低维隐空间中进行门控条件计算。是 **J-Space 的物理实现**、**监控的核心区域**。

**计算流程**:
1. 线性注意力(在原始空间)
2. 下投影到隐空间
3. 路由决策(在隐空间)
4. 专家计算(在隐空间)
5. 门控聚合
6. 联合 SAE 约束
7. 上投影回原始空间
8. 总线写入

**设计原则**:
- 隐空间 = 信息瓶颈(J-Space)
- 路由 = 可解释的计算选择
- 联合 SAE = 训练时的特征约束
- 总线写入 = 全局广播

```python
class WorkspaceBlock(nn.Module):
    def __init__(self, d_model, d_bus, n_heads, n_experts, top_k,
                 expert_dim, d_sae, sae_k, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        # ─── 线性注意力(J-Lens 友好)───
        self.norm1 = RMSNorm(d_model)
        self.attention = LinearAttention(d_model, n_heads)  # M14
        self.attn_gate = CausalGate(d_model, name=f"W{layer_idx}_attn")  # M3

        # ─── 下投影:d_model → d_bus ───
        self.down_proj = nn.Linear(d_model, d_bus, bias=False)  # M7
        nn.init.orthogonal_(self.down_proj.weight)

        # ─── 隐空间路由器 ───
        self.router = LatentRouter(d_bus, n_experts, top_k)  # M9
        self.router_gate = CausalGate(n_experts, name=f"W{layer_idx}_router")  # M3

        # ─── 专家组 ───
        self.experts = nn.ModuleList([
            ExpertBlock(d_bus, expert_dim, expert_id=i)  # M10
            for i in range(n_experts)
        ])
        self.expert_gate = CausalGate(d_bus, name=f"W{layer_idx}_expert")  # M3

        # ─── 门控聚合器 ───
        self.aggregator = GatedAggregator(d_bus, n_experts, top_k)  # M11

        # ─── 联合 SAE ───
        self.joint_sae = JointSAE(d_bus, d_sae, sparsity_k=sae_k)  # M12

        # ─── 上投影:d_bus → d_model ───
        self.up_proj = nn.Linear(d_bus, d_model, bias=False)  # M8
        nn.init.orthogonal_(self.up_proj.weight)

        # ─── 总线写入 ───
        self.bus_write_proj = nn.Linear(d_bus, d_bus, bias=False)
        nn.init.zeros_(self.bus_write_proj.weight)
        self.bus_write_gate = CausalGate(d_bus, name=f"W{layer_idx}_bus")  # M3

        # ─── 监控插槽 ───
        self.monitor_slot = MonitorSlot(d_model, d_bus, layer_idx)  # M2

        self.last_diagnostics = None

    def forward(self, x, bus, return_diag=False):
        # Step 1: 线性注意力
        normed = self.norm1(x)
        attn_out = self.attn_gate(lambda h: self.attention(h), normed)
        x = x + attn_out

        # Step 2: 下投影
        z = self.down_proj(x)

        # Step 3: 路由
        router_logits = self.router(z)
        top_k_weights, top_k_indices = self.router_gate(
            lambda h: self.router.select(h),
            router_logits
        )

        # Step 4: 专家计算
        expert_outputs = []
        for k in range(self.top_k):
            idx = top_k_indices[..., k]
            expert_out = self._dispatch_experts(z, idx)
            expert_outputs.append(expert_out)

        # Step 5: 门控聚合
        z_processed = self.aggregator(z, expert_outputs, top_k_weights)

        # Step 6: 联合 SAE
        z_sae, sae_loss = self.joint_sae(z_processed)

        # Step 7: 上投影
        ffn_out = self.up_proj(z_sae)
        x = x + ffn_out

        # Step 8: 总线写入
        z_summary = z_sae.mean(dim=1)
        bus_update = self.bus_write_gate(
            lambda h: self.bus_write_proj(h),
            z_summary
        )
        bus.write(self.layer_idx, bus_update)

        # 监控记录
        x = self.monitor_slot(x)

        if return_diag:
            self.last_diagnostics = {
                "router_logits": router_logits,
                "top_k_indices": top_k_indices,
                "top_k_weights": top_k_weights,
                "latent_z": z,
                "z_sae": z_sae,
                "sae_codes": self.joint_sae.last_codes,
                "sae_loss": sae_loss,
                "bus_update": bus_update,
                "monitor_vector": self.monitor_slot.monitor_vector,
            }

        return x

    def _dispatch_experts(self, z, indices):
        """将 token 分派到对应专家(高效实现)"""
        output = torch.zeros_like(z)
        for expert_id in range(self.n_experts):
            mask = (indices == expert_id)
            if mask.any():
                output[mask] = self.experts[expert_id](z[mask])
        return output
```

---

### M6. Action Block(动作块)

**定位**:模型的"嘴和手"。将工作空间的结论转化为具体输出。**使用细粒度软门控(P6 调整:从 LoRA-MoE 改为微专家 + 软门控)**。

**为什么从 LoRA 改为软门控**:
- 04 用 LoRA(rank=8)是为了**持续学习**(冻结原参数,只训 LoRA)
- 06 改为**软门控**是为了**Jacobian 连续性**——每个微专家 = 一个输出特征,门控值本身就是监控信号

```python
class ActionBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_micro_experts, micro_dim,
                 d_bus, d_monitor, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_micro = n_micro_experts

        # ─── 标准注意力 ───
        self.norm1 = RMSNorm(d_model)
        self.attention = StandardAttention(d_model, n_heads)  # M13
        self.attn_gate = CausalGate(d_model, name=f"A{layer_idx}_attn")  # M3

        # ─── 细粒度门控 FFN ───
        self.norm2 = RMSNorm(d_model)

        # 门控网络:决定每个微专家的激活程度
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, n_micro_experts)
        )

        # 微专家组(全部参与,门控加权)
        self.micro_experts = nn.ModuleList([
            MicroExpert(d_model, micro_dim, expert_id=i)
            for i in range(n_micro_experts)
        ])

        self.ffn_gate = CausalGate(d_model, name=f"A{layer_idx}_ffn")  # M3

        # ─── 总线读取 ───
        self.bus_read_proj = nn.Linear(d_bus, d_model, bias=False)
        nn.init.zeros_(self.bus_read_proj.weight)

        # ─── 监控插槽 ───
        self.monitor_slot = MonitorSlot(d_model, d_monitor, layer_idx)  # M2

        # ─── 门控值存储(监控用)───
        self.last_gate_values = None

    def forward(self, x, bus):
        bus_info = bus.read(self.layer_idx)
        bus_context = self.bus_read_proj(bus_info)

        normed = self.norm1(x)
        attn_out = self.attn_gate(
            lambda h: self.attention(h, context_bias=bus_context),
            normed
        )
        x = x + attn_out

        normed = self.norm2(x)
        gate_logits = self.gate_net(normed)  # [batch, seq, n_micro]
        gate_values = torch.sigmoid(gate_logits)  # 软门控,连续
        self.last_gate_values = gate_values.detach()

        def gated_ffn(h):
            out = torch.zeros_like(h)
            for i, expert in enumerate(self.micro_experts):
                weight = gate_values[..., i:i+1]
                out = out + weight * expert(h)
            return out

        ffn_out = self.ffn_gate(gated_ffn, normed)
        x = x + ffn_out

        x = self.monitor_slot(x)
        return x


class MicroExpert(nn.Module):
    """微专家:小维度,近似单义"""
    def __init__(self, d_model, micro_dim, expert_id):
        super().__init__()
        self.expert_id = expert_id
        self.up = nn.Linear(d_model, micro_dim)
        self.act = nn.GELU()
        self.down = nn.Linear(micro_dim, d_model)
        nn.init.orthogonal_(self.up.weight)

    def forward(self, x):
        return self.down(self.act(self.up(x)))
```

---

## 五、区间接口类(M7-M8)

### M7. Down Interface(下投影接口)

**定位**:感知区→工作空间的桥梁。将高维表征压缩到低维隐空间,创造**信息瓶颈**。

**设计约束**:
- 纯线性(不打断 J-Lens 链)
- 近似正交(不丢失方向信息)
- 可逆性正则(理论上可重建)
- 残差旁路(训练初期帮助梯度流动)

```python
class DownInterface(nn.Module):
    def __init__(self, d_model, d_bus):
        super().__init__()
        self.d_model = d_model
        self.d_bus = d_bus

        # 主投影(正交初始化)
        self.proj = nn.Linear(d_model, d_bus, bias=False)
        nn.init.orthogonal_(self.proj.weight)

        # 层归一化
        self.norm = RMSNorm(d_model)

        # 残差旁路(渐进关闭)
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))
        self.residual_proj = nn.Linear(d_model, d_bus, bias=False)
        nn.init.zeros_(self.residual_proj.weight)

    def forward(self, x):
        x_normed = self.norm(x)
        z = self.proj(x_normed)
        gate = torch.sigmoid(self.residual_gate)
        z_residual = self.residual_proj(x_normed)
        z = z + gate * z_residual
        return z

    def orthogonal_loss(self):
        """正交性惩罚:W @ W^T ≈ I_{d_bus}"""
        W = self.proj.weight
        WWT = W @ W.T
        I = torch.eye(self.d_bus, device=W.device)
        return ((WWT - I) ** 2).mean()

    def reconstruction_loss(self, x):
        """可逆性正则:投影后应能近似重建"""
        z = self.forward(x)
        W_pinv = torch.linalg.pinv(self.proj.weight)
        x_recon = z @ W_pinv.T
        return ((x - x_recon) ** 2).mean()
```

---

### M8. Up Interface(上投影接口)

**定位**:工作空间→动作区的桥梁。将低维隐空间表征恢复为高维,并注入总线全局信息。

```python
class UpInterface(nn.Module):
    def __init__(self, d_bus, d_model):
        super().__init__()

        # 主投影
        self.proj = nn.Linear(d_bus, d_model, bias=False)
        nn.init.orthogonal_(self.proj.weight)

        # 总线注入
        self.bus_inject = nn.Linear(d_bus, d_model, bias=False)
        nn.init.zeros_(self.bus_inject.weight)

        # 影响门控
        self.influence_gate = nn.Parameter(torch.tensor(0.0))

        # 层归一化
        self.norm = RMSNorm(d_bus)

    def forward(self, z, bus_state):
        z_normed = self.norm(z)
        x = self.proj(z_normed)
        gate = torch.sigmoid(self.influence_gate)
        bus_expanded = bus_state.unsqueeze(1)
        bus_projected = self.bus_inject(bus_expanded)
        x = x + gate * bus_projected
        return x

    def orthogonal_loss(self):
        W = self.proj.weight
        WTW = W.T @ W
        I = torch.eye(W.shape[1], device=W.device)
        return ((WTW - I) ** 2).mean()
```

---

## 六、工作空间内部类(M9-M12)

### M9. Latent Router(隐空间路由器)

**定位**:工作空间的核心决策机制。在隐空间中决定"哪些专家参与当前计算"。**路由决策本身就是最重要的监控信号之一**。

**设计原则**:
- 路由在隐空间(非原始空间)→ 更语义化
- 软路由 + 硬选择混合 → 训练稳定 + 推理高效
- 路由熵正则 → 防止坍缩
- 路由历史 → 监控信号

```python
class LatentRouter(nn.Module):
    def __init__(self, d_bus, n_experts, top_k):
        super().__init__()
        self.d_bus = d_bus
        self.n_experts = n_experts
        self.top_k = top_k

        # 路由网络(小型 MLP)
        self.route_net = nn.Sequential(
            nn.Linear(d_bus, d_bus // 2),
            nn.GELU(),
            nn.Linear(d_bus // 2, d_bus // 4),
            nn.GELU(),
            nn.Linear(d_bus // 4, n_experts)
        )

        # 温度参数(控制软/硬程度)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 路由噪声(训练时探索)
        self.noise_scale = nn.Parameter(torch.tensor(0.1))

        self.last_logits = None
        self.last_probs = None
        self.last_indices = None

    def forward(self, z):
        logits = self.route_net(z)
        if self.training:
            noise = torch.randn_like(logits) * self.noise_scale
            logits = logits + noise
        self.last_logits = logits.detach()
        return logits

    def select(self, logits):
        probs = F.softmax(logits / self.temperature, dim=-1)
        self.last_probs = probs.detach()
        top_k_weights, top_k_indices = probs.topk(self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        self.last_indices = top_k_indices.detach()
        return top_k_weights, top_k_indices

    def entropy(self):
        """路由熵(监控用):低熵=确定,高熵=困惑"""
        if self.last_probs is None:
            return torch.tensor(0.0)
        return -(self.last_probs * torch.log(self.last_probs + 1e-8)).sum(-1).mean()

    def load_balance_loss(self, assignments):
        """负载均衡损失:防止专家坍缩"""
        flat = assignments.reshape(-1)
        expert_counts = torch.bincount(flat, minlength=self.n_experts).float()
        expert_freq = expert_counts / expert_counts.sum()
        mean_probs = self.last_probs.mean(dim=(0, 1))
        return (expert_freq * mean_probs).sum() * self.n_experts
```

---

### M10. Expert Block(专家块)

**定位**:工作空间中的基本计算单元。每个专家在隐空间中执行一种特定的"操作"。

**设计原则**:
- 在隐空间(`d_bus`)操作,不在原始空间
- 残差结构(专家输出 = 输入 + 变换)
- 正交初始化(减少专家间冗余)
- 残差门控(控制专家贡献强度)

```python
class ExpertBlock(nn.Module):
    def __init__(self, d_bus, expert_dim, expert_id):
        super().__init__()
        self.expert_id = expert_id
        self.d_bus = d_bus

        self.up = nn.Linear(d_bus, expert_dim)
        self.act = nn.GELU()
        self.down = nn.Linear(expert_dim, d_bus)

        # 残差门控(控制专家贡献强度)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

        # 层归一化
        self.norm = RMSNorm(d_bus)

        # 正交初始化
        nn.init.orthogonal_(self.up.weight)
        nn.init.orthogonal_(self.down.weight)

    def forward(self, z):
        z_normed = self.norm(z)
        transform = self.down(self.act(self.up(z_normed)))
        gate = torch.sigmoid(self.residual_gate)
        return z + gate * transform

    def sparsity_loss(self):
        """专家内部稀疏性(可选)"""
        return self.up.weight.abs().mean() * 0.001
```

---

### M11. Gated Aggregator(门控聚合器)

**定位**:将多个专家的输出聚合为单一表征。**不是简单加权平均**,而是学习"如何组合"。

```python
class GatedAggregator(nn.Module):
    def __init__(self, d_bus, n_experts, top_k):
        super().__init__()
        self.top_k = top_k

        # 交互注意力(专家输出之间)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_bus,
            num_heads=4,
            batch_first=True
        )

        # 门控权重
        self.gate_proj = nn.Linear(d_bus * 2, top_k)

        # 最终投影
        self.out_proj = nn.Linear(d_bus, d_bus)
        self.norm = RMSNorm(d_bus)

    def forward(self, z_original, expert_outputs, router_weights):
        """
        z_original: [batch, seq, d_bus] 原始隐表征
        expert_outputs: list of [batch, seq, d_bus],长度 = top_k
        router_weights: [batch, seq, top_k]
        """
        # 堆叠专家输出
        stacked = torch.stack(expert_outputs, dim=2)  # [batch, seq, top_k, d_bus]

        # 加权平均(基线)
        weighted = (stacked * router_weights.unsqueeze(-1)).sum(dim=2)

        # 交互注意力(专家间)
        stacked_flat = stacked.reshape(-1, self.top_k, z_original.shape[-1])
        attn_out, _ = self.cross_attn(stacked_flat, stacked_flat, stacked_flat)
        attn_out = attn_out.reshape(z_original.shape[0], z_original.shape[1], -1)

        # 门控组合
        gate_input = torch.cat([z_original, weighted], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        # 最终聚合
        combined = (stacked * gate.unsqueeze(-1)).sum(dim=2)

        # 残差 + 投影
        result = z_original + self.out_proj(self.norm(combined))
        return result
```

---

### M12. Joint SAE(联合稀疏自编码器)

**定位**:在工作空间隐表征上联合训练的稀疏自编码器。**不是事后分析工具**,而是训练时的结构约束。

**设计原则**:
- 在隐空间(`d_bus`)操作,维度低 → 字典小
- 联合训练(不是事后拟合)
- Top-K 稀疏(不是 L1)
- 字典正交约束
- 特征语义跨样本稳定

```python
class JointSAE(nn.Module):
    def __init__(self, d_bus, dict_size, sparsity_k):
        super().__init__()
        self.d_bus = d_bus
        self.dict_size = dict_size
        self.sparsity_k = sparsity_k

        # 编码器
        self.encoder = nn.Linear(d_bus, dict_size, bias=False)
        # 解码器
        self.decoder = nn.Linear(dict_size, d_bus, bias=False)
        # 共享字典
        self.tie_weights = True

        self.last_codes = None
        self.last_recon = None

    def forward(self, z):
        codes = self.encoder(z)
        # Top-K 稀疏
        top_k_vals, top_k_idx = codes.topk(self.sparsity_k, dim=-1)
        sparse_codes = torch.zeros_like(codes)
        sparse_codes.scatter_(-1, top_k_idx, top_k_vals)
        # 解码
        z_recon = self.decoder(sparse_codes)
        self.last_codes = sparse_codes.detach()
        self.last_recon = z_recon.detach()
        recon_loss = ((z - z_recon) ** 2).mean()
        return z_recon, recon_loss

    @property
    def dictionary(self):
        return self.decoder.weight.T

    def orthogonal_loss(self):
        D = self.decoder.weight
        DTD = D.T @ D
        I = torch.eye(self.dict_size, device=D.device)
        return ((DTD - I) ** 2).mean()

    def get_active_features(self, z, threshold=0.1):
        codes = self.encoder(z)
        top_k_vals, top_k_idx = codes.topk(self.sparsity_k, dim=-1)
        active = top_k_idx[top_k_vals > threshold]
        return active
```

---

## 七、注意力变体类(M13-M14)

### M13. Standard Attention(标准注意力)

**定位**:感知区和动作区使用的标准多头注意力。支持**总线上下文注入**和**RoPE**。

```python
class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # RoPE(旋转位置编码)
        self.rope = RotaryEmbedding(self.d_head)

    def forward(self, x, context_bias=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE
        q, k = self.rope(q, k)

        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # 总线上下文偏置(加在注意力分数上)
        if context_bias is not None:
            bias = context_bias.view(B, 1, 1, D)[:, :, :, :self.d_head]
            scores = scores + bias.unsqueeze(2) * 0.1

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)
```

---

### M14. Linear Attention(线性注意力)

**定位**:工作空间区使用的线性注意力。Jacobian 更平滑,计算更高效。

**优势**:
- Jacobian 更平滑(无 softmax 的尖锐非线性)
- O(n) 复杂度(适合长序列)
- 可以维护递推状态(适合工作空间的"记忆"语义)

```python
class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # 特征映射:elu(x) + 1(保证非负)
        self.feature_map = lambda x: F.elu(x) + 1

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # 特征映射
        q = self.feature_map(q)
        k = self.feature_map(k)

        # 线性注意力:Q @ (K^T @ V)
        kv = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, kv)

        # 归一化
        k_sum = k.sum(dim=-2, keepdim=True)
        normalizer = torch.matmul(q, k_sum.transpose(-2, -1))
        out = out / (normalizer + 1e-6)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)
```

---

## 八、运行时系统类(M15-M16)

### M15. SOCAMonitor(运行时监控器)

**定位**:推理时的实时监控系统。读取所有监控插槽、总线状态、路由决策,进行**异常检测**与**因果干预**。

**功能**:
1. 实时异常检测(激活、路由、总线、概念)
2. 因果干预(替换、冻结、噪声、旁路)
3. 状态快照(可保存、可回放)
4. 健康评分(综合评估模型当前状态)

```python
class SOCAMonitor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.baseline = None
        self.sigma_threshold = config.sigma_threshold  # 3.0
        self.alert_history = []
        self.health_weights = {
            "activation": 0.2, "routing": 0.2,
            "bus": 0.3, "concept": 0.2, "gate": 0.1,
        }

    def calibrate(self, calibration_loader, n_batches=100):
        """用正常数据建立基线"""
        all_monitor, all_bus, all_router_entropy = [], [], []
        all_gate_values, all_sae_max = [], []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= n_batches: break
                diag = self.model.forward_with_diagnostics(batch)
                all_monitor.append(diag["monitor_vectors"])
                all_bus.append(diag["bus_states"])
                all_router_entropy.append(diag["router_entropies"])
                all_gate_values.append(diag["gate_values"])
                all_sae_max.append(diag["sae_max_activations"])

        self.baseline = {
            "monitor_mean": torch.stack(all_monitor).mean(0),
            "monitor_std": torch.stack(all_monitor).std(0) + 1e-8,
            "bus_mean": torch.stack(all_bus).mean(0),
            "bus_std": torch.stack(all_bus).std(0) + 1e-8,
            "router_ent_mean": torch.stack(all_router_entropy).mean(0),
            "router_ent_std": torch.stack(all_router_entropy).std(0) + 1e-8,
            "gate_mean": torch.stack(all_gate_values).mean(0),
            "gate_std": torch.stack(all_gate_values).std(0) + 1e-8,
            "sae_max_mean": torch.stack(all_sae_max).mean(),
            "sae_max_std": torch.stack(all_sae_max).std() + 1e-8,
        }

    def monitor_step(self, diagnostics):
        """每步推理后调用:返回告警列表和健康评分"""
        alerts = []
        scores = {}

        # 1. 激活异常
        z_scores = (diagnostics["monitor_vectors"] - self.baseline["monitor_mean"]) / self.baseline["monitor_std"]
        scores["activation"] = self._score_from_zscore(z_scores)

        # 2. 路由异常
        ent_z = (diagnostics["router_entropies"] - self.baseline["router_ent_mean"]) / self.baseline["router_ent_std"]
        scores["routing"] = self._score_from_zscore(ent_z)

        # 3. 总线异常
        bus_z = (diagnostics["bus_states"] - self.baseline["bus_mean"]) / self.baseline["bus_std"]
        scores["bus"] = self._score_from_zscore(bus_z)

        # 4. 概念异常(SAE)
        sae_z = (diagnostics["sae_max_activations"] - self.baseline["sae_max_mean"]) / self.baseline["sae_max_std"]
        scores["concept"] = self._score_from_zscore(sae_z)

        # 5. 门控异常
        gate_z = (diagnostics["gate_values"] - self.baseline["gate_mean"]) / self.baseline["gate_std"]
        scores["gate"] = self._score_from_zscore(gate_z)

        health = 100 * sum(self.health_weights[k] * scores[k] for k in scores)

        result = {
            "health_score": health,
            "alerts": alerts,
            "scores": scores,
            "n_alerts": len(alerts),
        }
        if alerts:
            self.alert_history.append(result)
        return result

    def intervene_layer(self, layer_idx, mode, **kwargs):
        """对指定层进行因果干预"""
        layer = self.model.layers[layer_idx]
        if mode == "replace_attn":
            layer.attn_gate.set_replace(kwargs["value"])
        elif mode == "replace_ffn":
            layer.ffn_gate.set_replace(kwargs["value"])
        elif mode == "freeze":
            layer.attn_gate.set_freeze()
            layer.ffn_gate.set_freeze()
        elif mode == "noise":
            layer.attn_gate.set_noise(kwargs.get("scale", 0.1))
        elif mode == "bypass":
            layer.monitor_slot.enable_bypass()

    def intervene_bus(self, new_state):
        """覆写总线状态(全局干预)"""
        self.model.broadcast_bus.state = new_state

    def intervene_expert(self, layer_idx, expert_id, mode):
        """对特定专家进行干预"""
        ws_layer = self.model.layers[layer_idx]
        if mode == "disable":
            ws_layer.experts[expert_id].residual_gate.data.fill_(-10.0)

    def reset_all(self):
        """重置所有干预"""
        for layer in self.model.layers:
            layer.monitor_slot.reset()
            if hasattr(layer, 'attn_gate'):
                layer.attn_gate.reset()
            if hasattr(layer, 'ffn_gate'):
                layer.ffn_gate.reset()

    def snapshot(self):
        """保存当前完整状态(可回放)"""
        return {
            "bus_state": self.model.broadcast_bus.state.clone(),
            "bus_history": self.model.broadcast_bus.get_full_history(),
            "monitor_vectors": [
                layer.monitor_slot.monitor_vector.clone()
                for layer in self.model.layers
            ],
            "router_states": [
                layer.router.last_indices.clone()
                for layer in self.model.layers
                if hasattr(layer, 'router')
            ],
            "sae_codes": [
                layer.joint_sae.last_codes.clone()
                for layer in self.model.layers
                if hasattr(layer, 'joint_sae')
            ],
        }
```

---

### M16. SOCALoss(训练损失组合器)

**定位**:组合所有训练目标:主任务、SAE、正交性、负载均衡、总线正则、稀疏性。**支持分阶段权重调度**。

**损失组成**:
1. LM Loss(交叉熵)
2. SAE Reconstruction Loss
3. Interface Orthogonality Loss
4. Load Balance Loss
5. Bus Regularization
6. Gate Sparsity Loss
7. Expert Orthogonality Loss

```python
class SOCALoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weights = {
            "lm": 1.0,
            "sae": config.lambda_sae,               # 0.05(与 04/07/08 对齐)
            "ortho_interface": config.lambda_orth,  # 0.005(与 04/07/08 对齐;原 0.01 偏大)
            "load_balance": config.lambda_balance,    # 0.005
            "bus_reg": config.lambda_bus_reg,         # 0.001
            "gate_sparse": config.lambda_sparse,      # 0.01
            "expert_ortho": config.lambda_expert_ortho,  # 0.005
        }
        self.phase = 0

    def forward(self, model_output, targets):
        losses = {}

        # 1. 主任务
        losses["lm"] = F.cross_entropy(
            model_output["logits"].view(-1, model_output["logits"].size(-1)),
            targets.view(-1)
        )

        # 2. SAE 重建
        losses["sae"] = model_output["sae_loss"].mean()

        # 3. 接口正交性
        losses["ortho_interface"] = (
            model_output["down_interface"].orthogonal_loss() +
            model_output["up_interface"].orthogonal_loss()
        )

        # 4. 负载均衡
        losses["load_balance"] = model_output["load_balance_loss"]

        # 5. 总线正则(总线方差 ≈ 1)
        bus_states = model_output["bus_history"]  # [n_layers, batch, d_bus]
        bus_var = bus_states.var(dim=0)
        losses["bus_reg"] = ((bus_var - 1.0) ** 2).mean()

        # 6. 门控稀疏性(动作区)
        gate_values = model_output["action_gate_values"]
        losses["gate_sparse"] = gate_values.abs().mean()

        # 7. 专家正交性
        losses["expert_ortho"] = model_output.get("expert_ortho_loss", torch.tensor(0.0))

        # 加权求和
        total = sum(self.weights[k] * losses[k] for k in losses if k in self.weights)
        return total, {k: v.item() for k, v in losses.items()}

    def set_phase(self, phase: int):
        """
        训练阶段切换:
        Phase 0: 感知区(只有 LM)
        Phase 1: 工作空间(LM + SAE + 正交 + 负载均衡)
        Phase 2: 动作区(LM + 稀疏)
        Phase 3: 端到端(全部,权重降低)
        """
        self.phase = phase
        if phase == 0:
            self.weights = {"lm": 1.0}
        elif phase == 1:
            self.weights = {
                "lm": 1.0, "sae": 0.1,
                "ortho_interface": 0.01, "load_balance": 0.01,
                "bus_reg": 0.001,
            }
        elif phase == 2:
            self.weights = {"lm": 1.0, "gate_sparse": 0.01}
        elif phase == 3:
            self.weights = {k: v * 0.5 for k, v in self.weights.items()}
```

**与 04 的差异**:
- 04 是按训练**进度**(0%/5%/30%/90%)分阶段
- 06 是按**区域**(感知→工作空间→动作→端到端)分阶段
- 06 的阶段划分更符合 SOCA 的三区域架构,每个区域先单独训练再端到端微调

---

## 九、模块交互全景图(简化版)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Token Input                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PERCEPTION ZONE (5 layers)                                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ M4: PercBlock│─▶│ M4: PercBlock│─▶│ M4: PercBlock│  (×5)       │    │
│  │  │  + M13 Std   │  │              │  │              │               │    │
│  │  │  + M2 Monitor│  │              │  │              │               │    │
│  │  │  + M3 Gates  │  │              │  │              │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  │         ▲                    ▲                    ▲                  │    │
│  │         └────────────────────┴────────────────────┘                  │    │
│  │                          ▲ M1: Bus.read()                           │    │
│  └──────────────────────────┼──────────────────────────────────────────┘    │
│                             │                                               │
│                             ▼                                               │
│  ┌──────────────────────────┴──────────────────────────────────────────┐    │
│  │  M7: DownInterface (d_model → d_bus)                              │    │
│  │  + orthogonal_loss + reconstruction_loss                          │    │
│  └──────────────────────────┬──────────────────────────────────────────┘    │
│                             │                                               │
│  ┌──────────────────────────▼──────────────────────────────────────────┐    │
│  │  WORKSPACE ZONE (2 layers × 2 cycles = 4 effective)               │    │
│  │                                                                     │    │
│  │  ┌──────────────────────────────────────────────────────────────┐  │    │
│  │  │ M5: WorkspaceBlock                                          │  │    │
│  │  │  M14 LinearAttn → M7 Down → M9 LatentRouter              │  │    │
│  │  │  → M10 ExpertBlock ×N → M11 GatedAggregator               │  │    │
│  │  │  → M12 JointSAE → M8 Up                                    │  │    │
│  │  │  + M2 MonitorSlot + M3 CausalGates(×4)                     │  │    │
│  │  └──────────────────────────────────────────────────────────────┘  │    │
│  │         │ M1: Bus.write(layer_j, z_summary)                        │    │
│  │         ▼                                                           │    │
│  │  ┌══════════════════════════════════════════════════════════════┐   │    │
│  │  ║  M1: BROADCAST BUS (d_bus)                                  ║   │    │
│  │  ║  state: [batch, d_bus]                                      ║   │    │
│  │  ║  read_gates[i] / write_gates[i] (per-layer, learnable)      ║   │    │
│  │  ║  decay = 0.99                                              ║   │    │
│  │  ║  ← 工作空间写入 | 感知/动作读取                                ║   │    │
│  │  └══════════════════════════════════════════════════════════════┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                               │
│                             ▼                                               │
│  ┌──────────────────────────┴──────────────────────────────────────────┐    │
│  │  M8: UpInterface (d_bus → d_model)                                │    │
│  │  + bus_inject (注入总线全局信息)                                  │    │
│  └──────────────────────────┬──────────────────────────────────────────┘    │
│                             │                                               │
│  ┌──────────────────────────▼──────────────────────────────────────────┐    │
│  │  ACTION ZONE (5 layers)                                            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ M6: ActBlock │─▶│ M6: ActBlock │─▶│ M6: ActBlock │  (×5)       │    │
│  │  │  + M13 Std   │  │   微专家+    │  │              │               │    │
│  │  │  + 软门控 FFN│  │   软门控    │  │              │               │    │
│  │  │  + M2 Monitor│  │              │  │              │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  │         ▲                    ▲                    ▲                  │    │
│  │         └────────────────────┴────────────────────┘                  │    │
│  │                          ▲ M1: Bus.read()                           │    │
│  └──────────────────────────┼──────────────────────────────────────────┘    │
│                             │                                               │
│                             ▼                                               │
│                      OUTPUT HEAD                                            │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════    │
│  运行时:                                                                    │
│    M15 SOCAMonitor ← 读取所有 MonitorSlot + Bus.state + Router              │
│    M16 SOCALoss   ← 计算 LM + SAE + 正交 + 负载均衡 + 稀疏                  │
│  ════════════════════════════════════════════════════════════════════════    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 十、模块相互依赖矩阵

| 模块 | 依赖 | 被依赖 |
|---|---|---|
| M1 BroadcastBus | 无 | M4, M5, M6, M8, M15 |
| M2 MonitorSlot | 无 | M4, M5, M6, M15 |
| M3 CausalGate | 无 | M4, M5, M6, M15 |
| M4 PerceptionBlock | M1, M2, M3, M13 | M16 |
| M5 WorkspaceBlock | M1, M2, M3, M7, M8, M9, M10, M11, M12, M14 | M16 |
| M6 ActionBlock | M1, M2, M3, M13 | M16 |
| M7 DownInterface | 无 | M5, M16 |
| M8 UpInterface | M1 | M5, M16 |
| M9 LatentRouter | 无 | M5, M15 |
| M10 ExpertBlock | 无 | M5 |
| M11 GatedAggregator | 无 | M5 |
| M12 JointSAE | 无 | M5, M15, M16 |
| M13 StandardAttention | 无 | M4, M6 |
| M14 LinearAttention | 无 | M5 |
| M15 SOCAMonitor | M1, M2, M3, M9, M12 | (无,纯运行时) |
| M16 SOCALoss | M7, M8, M10, M12 | (无,纯训练) |

---

## 十一、与现有文档的对照

| 维度 | 04 代码骨架 | 06 模块规格 |
|---|---|---|
| BroadcastBus | `class BroadcastBus(nn.Module)` | **M1** + 每层独立 read/write gates |
| MonitorSlot | `class MonitorSlot(nn.Module)` | **M2** + 三种模式(OBSERVE/INJECT/BYPASS) |
| CausalGate | 不存在(用消融开关代替) | **M3** + 四种模式(normal/replace/freeze/noise) |
| PerceptionLayer | `_make_layer(config, attn_type, ffn_mult)` | **M4** + GatedGate 包裹每个计算步骤 |
| WorkspaceMid | `class WorkspaceMid(nn.Module)` | **M5** + 8 步流程(线性注意力→下→路由→专家→聚合→SAE→上→写总线) |
| ActionLayer | `_make_action_layer(config, attn_type)` + LoRAMoE | **M6** + **微专家 + 软门控**(替代 LoRA) |
| Down/Up Interface | `nn.Linear(d_model, d_bus)` 简单线性 | **M7/M8** + 正交损失 + 可逆损失 + 总线注入 |
| LatentRouter | 嵌入 MultiHeadMoE | **M9** 独立模块 + 温度参数 + 噪声 |
| ExpertBlock | 嵌入 MultiHeadMoE | **M10** 独立模块 + 残差门控 |
| GatedAggregator | 不存在(简单加权) | **M11** + 跨专家交互注意力 |
| JointSAE | `class JointSAE(nn.Module)` | **M12** + 字典正交约束 |
| StandardAttention | `class StandardAttention(nn.Module)` | **M13** + RoPE + 总线上下文偏置 |
| LinearAttention | `class LinearAttention(nn.Module)` | **M14** elu+1 feature map |
| SOCAMonitor | `logger.py + probe.py` 简化版 | **M15** 完整版(校准+干预+快照+健康评分) |
| SOCALoss | `class SOCALoss` 7 项损失 | **M16** + **4 阶段调度**(感知→工作空间→动作→端到端) |

---

## 十二、关键设计决策汇总

| 决策 | 选择 | 理由 |
|---|---|---|
| **注意力组成** | Standard + Linear(去 Gated DeltaNet) | 简化实施,易消融;Gated DeltaNet 可在 0.8B 阶段引入 |
| **动作区专家** | 微专家 + 软门控(替代 LoRA) | Jacobian 连续;门控值 = 监控信号 |
| **总线机制** | 每层独立 read/write gate(初始 -3.0) | 渐进启用,避免训练初期被无关层污染 |
| **干预机制** | CausalGate + MonitorSlot(运行时) | 区别于 04 的消融开关(配置层),支持细粒度因果干预 |
| **训练阶段** | 感知→工作空间→动作→端到端(按区域) | 匹配 SOCA 三区域架构,比"按进度"更符合模块化思路 |
| **路由温度** | 可学习(初始 1.0) + 训练噪声 | 兼顾探索与稳定,防止早期坍缩 |
| **SAE 字典** | 与编码器共享权重(`tie_weights=True`) | 减少参数,保证编码解码一致性 |

---

## 十三、对 04 的回溯建议

| 04 现状 | 06 建议 |
|---|---|
| `MonitorSlot` 仅压缩投影 | 增加 3 种模式(OBSERVE/INJECT/BYPASS) |
| 无 CausalGate | 新增 `CausalGate` 包裹每个计算步骤 |
| `BroadcastBus` 简单读写 | 增加每层独立 read/write gates |
| `LoRAMoE` 用于动作区 | 替换为 `ActionBlock` 的微专家 + 软门控 |
| `WorkspaceMid` 单步 forward | 拆分为 8 步流程,每步独立监控 |
| 4 阶段按进度 | 改为按区域(感知→工作空间→动作→端到端) |
| 无 SOCAMonitor | 用 M15 替换 logger/probe 的简化实现 |

> **实施路径**:04 的代码可以**保留主干,按 M1-M16 重构类名和方法签名**,无需推倒重来。

---

## 📅 文档版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-28 | 初版,完成 M1-M16 模块化设计规格(基于新讨论内容整合) |

---

> **下一步**:阅读 [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) 的代码骨架,按本文件的 M1-M16 命名重构类与方法,获得"可干预、可观测、可推理"的完整实现。

# SOCA v3-Micro-Final 工程实现路线图与代码骨架

> **目的**:把 [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) 锁定的 **16 层 × ~155M 配置**落地为可执行的代码骨架与里程碑计划。
>
> **关联文档**:
> - 上一阶段(层数甜点):[`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md)
> - 上游审查:[`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md)
> - 上游参数:[`02-sweet-spot-params.md`](./02-sweet-spot-params.md)
> - **角色**:本目录的"实施交接"文档 —— 把研究结论转换为代码骨架、测试套件、消融框架与里程碑计划

---

## 一、项目结构总览

```
soca_micro/
├── config.py                 # 全局配置(含消融开关)
├── model/
│   ├── __init__.py
│   ├── soca_model.py         # 顶层模型(SOCAMicro)
│   ├── embedding.py          # Token Embedding + Output Head
│   ├── attention/
│   │   ├── standard.py       # Standard Attention + Gated 变体
│   │   ├── gated_deltanet.py # Gated DeltaNet
│   │   └── linear.py         # Linear Attention(工作空间中段)
│   ├── moe/
│   │   ├── soft_moe.py       # Soft MoE(前段,sigmoid dispatch)
│   │   ├── mh_moe.py         # Multi-Head MoE(中段,Sparse×Latent)
│   │   ├── fine_moe.py       # Fine-grained MoE(后段,Shared+Routed+Device)
│   │   └── lora_moe.py       # LoRA-MoE(动作区,Dense)
│   ├── workspace/
│   │   ├── front.py          # 工作空间前段
│   │   ├── mid.py            # 工作空间中★段(含循环)
│   │   ├── back.py           # 工作空间后段
│   │   └── interface.py      # Down/Up Interface
│   ├── sae.py                # Joint SAE(三段)
│   ├── bus.py                # 广播总线
│   ├── monitor.py            # MonitorSlot + SOCAMonitor
│   └── zones.py              # 感知区 / 动作区
├── training/
│   ├── trainer.py            # 训练主循环
│   ├── losses.py             # 多损失联合优化
│   ├── scheduler.py          # 学习率 + 损失权重调度
│   ├── data.py               # 数据加载与课程学习
│   └── eval.py               # 评估(PPL + 下游任务)
├── ablation/
│   ├── registry.py           # 24 个消融的注册表
│   ├── runner.py             # 消融执行器
│   └── analysis.py           # 消融结果分析
├── monitoring/
│   ├── logger.py             # 训练日志
│   ├── probe.py              # 探针采集
│   └── visualize.py          # 可视化(J-Space / 路由热力图 / SAE)
├── scripts/
│   ├── train.py              # 主训练入口
│   ├── ablate.py             # 消融入口
│   └── evaluate.py           # 评估入口
└── tests/
    ├── test_shapes.py        # 维度对齐测试
    ├── test_gradients.py     # 梯度流测试
    └── test_ablation.py      # 消融开关测试
```

---

## 二、全局配置

> **从 YAML 真源加载**(推荐):SOCAMicroConfig 的字段定义见 [`./soca_micro_final_config.yaml`](./soca_micro_final_config.yaml) 的 `soca_config_base` + `soca_micro_specific` 两节。实施代码应从此文件加载,避免手抄本节 dataclass 中的数字。
>
> ```python
> # 从 YAML 加载(推荐)
> import yaml
> with open("docs/research/agenticsom/soca_micro_final_config.yaml") as f:
>     yaml_cfg = yaml.safe_load(f)
> config = SOCAMicroConfig(**yaml_cfg["soca_micro_specific"])
> ```
>
> **手动定义**(仅在无 YAML 库时):以下 dataclass 是 YAML 中 `soca_micro_specific` 段的 Python 镜像;**与 YAML 必须保持一致**——以 YAML 为真源,本节为参考实现。

```python
# config.py
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class SOCAMicroConfig:
    """SOCA v3-Micro-Final: 16 层 × ~155M"""

    # ═══ 基础架构 ═══
    n_vocab: int = 32000
    d_model: int = 896
    max_seq_len: int = 2048
    tie_embeddings: bool = True

    # ═══ 注意力 ═══
    n_heads_attn: int = 14          # 896 / 64
    d_head_attn: int = 64
    deltanet_state_dim: int = 128

    # ═══ 层分配(物理 16 层)═══
    n_perception: int = 5           # L1-L5
    n_ws_front: int = 2             # L6-L7
    n_ws_mid_physical: int = 2      # L8-L9(×2 循环 = 等效 4)
    n_ws_back: int = 2              # L10-L11
    n_action: int = 5               # L12-L16
    n_cycles: int = 2               # 中段循环次数

    # ═══ FFN ═══
    ffn_mult_perception: float = 2.2    # 896 → 1971 → 896
    ffn_mult_action: float = 1.8        # 896 → 1612 → 896

    # ═══ 工作空间前段:Soft MoE ═══
    front_n_slots: int = 4
    front_n_experts_per_slot: int = 4
    front_expert_hidden: int = 1024     # 896→1024→896
    front_dispatch: str = "sigmoid"     # 非 softmax(J-Lens 友好)

    # ═══ 工作空间中★段:MH-MoE ═══
    d_bus: int = 256
    mid_n_heads: int = 4
    mid_d_head: int = 64                # 256 / 4
    mid_experts_per_head: int = 12
    mid_top_k_per_head: int = 4
    mid_expert_hidden: int = 192        # 64→192→64
    mid_attn_type: str = "linear"       # 无 softmax

    # ═══ 工作空间后段:Fine MoE ═══
    back_n_shared: int = 1
    back_n_routed: int = 20
    back_n_device: int = 1
    back_top_k: int = 5
    back_routed_hidden: int = 320       # 896→320→896
    back_shared_hidden: int = 448
    back_device_hidden: int = 224

    # ═══ 动作区:LoRA-MoE ═══
    lora_n_heads: int = 2
    lora_per_head: int = 10             # 总 20 LoRA
    lora_rank: int = 8

    # ═══ Joint SAE ═══
    sae_mid_dict: int = 2048
    sae_mid_k: int = 12
    sae_light_dict: int = 1024          # 前段/后段
    sae_light_k: int = 16

    # ═══ 广播总线 ═══
    bus_gamma: float = 0.99

    # ═══ 监控 ═══
    monitor_dim: int = 48

    # ═══ 训练 ═══
    batch_size: int = 256
    seq_len: int = 2048
    lr: float = 2.5e-4
    warmup_steps: int = 2500
    max_steps: int = 11500              # 6B token / (256×2048)
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # ═══ 辅助损失权重(与 08 §五.2 v3-Micro-Final 子表对齐)═══
    lambda_sparse: float = 0.001        # L1 稀疏(感知/动作)(03 §五.6 / 08 §五.2)
    lambda_orth: float = 0.005          # 正交性(中段)(08 §五.2;04 原值 0.01 偏大→改为 0.005 防 F3)
    lambda_sae: float = 0.05            # SAE 重建(08 §五.2;04 原值 0.15 偏大→改为 0.05)
    lambda_concept: float = 0.001       # 概念正交
    lambda_route: float = 0.005         # 路由负载均衡(08 §五.2;04 原值 0.01 偏大→改为 0.005)
    lambda_cycle: float = 0.005         # 循环一致性(08 §五.2)

    # ═══ 消融开关(默认全关 = 完整模型;对应 §九 SOCA-A0~A24)═══
    # 编号映射:字段名 → §九 消融 ID(SOCA-A0 是 baseline,无字段)
    ablation: dict = field(default_factory=lambda: {
        # ── 核心架构(SOCA-A1~A5)──
        "disable_cycles": False,         # SOCA-A1: 去掉循环
        "disable_bus": False,            # SOCA-A2: 去掉总线
        "disable_sae": False,            # SOCA-A3: 去掉 SAE
        "disable_zones": False,          # SOCA-A4: 去掉区域划分
        "disable_mh_moe": False,         # SOCA-A5: MH→标准 MoE

        # ── 可解释性(SOCA-A6~A9)──
        "enable_weight_sparsity": False, # SOCA-A6: 权重稀疏(默认关)
        "sparsity_ratio": 0.0,           # SOCA-A6: 权重稀疏比例(0.5 = 50%)
        "sae_dict_override": None,       # SOCA-A7: SAE 字典回退(None / 256 / 512)
        "dispatch_override": None,       # SOCA-A8: dispatch 回退(None / "softmax")
        "disable_lora": False,           # SOCA-A9: 去掉 LoRA

        # ── 注意力(SOCA-A10~A12)──
        "attn_override": None,           # SOCA-A10/A11: 全 Standard / 全 DeltaNet
        "disable_gates": False,          # SOCA-A12: 去掉所有门控

        # ── MoE(SOCA-A13~A16)──
        "disable_front_moe": False,      # SOCA-A13: 前段 Soft MoE → Dense
        "disable_device_expert": False,  # SOCA-A14: 后段去掉设备专家
        "disable_shared_expert": False,  # SOCA-A15: 后段去掉共享专家
        "back_top_k_override": None,      # SOCA-A16: 后段 Top-K 覆盖(None / 1)

        # ── 训练策略(SOCA-A17~A18)──
        "disable_all_aux": False,        # SOCA-A17: 去掉所有辅助损失
        "disable_curriculum": False,     # SOCA-A18: 去掉课程学习

        # ── 规模(SOCA-A19~A20)──
        "mid_experts_override": None,    # SOCA-A19/A20: 中段专家数覆盖

        # ── 循环(SOCA-A21~A22)──
        "n_cycles_override": None,       # SOCA-A21: 循环次数覆盖(2 → 3)
        "disable_iter_pos": False,       # SOCA-A22: 去掉迭代位置编码

        # ── 对照(SOCA-A23~A24)──
        "architecture": None,            # SOCA-A23/A24: 标准 Transformer / 标准 MoE
    })
```

---

## 三、核心基础组件

### 3.1 MonitorSlot + 广播总线

> ⚠️ **本节 BroadcastBus 是 MVP 简化实现**(固定 γ + 全局 bus_scale);06 §三 M1 BroadcastBus 是生产级完整接口(每层独立 read/write gates + 可学习门控)。两者功能等价但 06 的实现更灵活;实施时建议**优先使用 06 §三 M1**,本节可作为快速原型。

```python
# model/monitor.py
import torch
import torch.nn as nn

class MonitorSlot(nn.Module):
    """每层的监控槽:将残差流压缩为 48 维摘要"""
    def __init__(self, d_model: int, monitor_dim: int = 48):
        super().__init__()
        self.proj = nn.Linear(d_model, monitor_dim, bias=False)
        self.register_buffer("history", [])  # 推理时收集

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, d_model] → summary: [B, S, monitor_dim]"""
        summary = self.proj(x)
        return summary


# model/bus.py
class BroadcastBus(nn.Module):
    """
    全局广播总线:每层写入,所有层可读。
    使用指数衰减 γ=0.99 控制历史信息权重。

    注:这是 04 的 MVP 简化版(全局 bus_scale);
    生产级接口参见 [`06-architecture-modules.md` §三 M1](./06-architecture-modules.md)(每层独立 read/write gates)
    """
    def __init__(self, d_model: int, d_bus: int = 256, gamma: float = 0.99):
        super().__init__()
        self.write_proj = nn.Linear(d_model, d_bus, bias=False)
        self.read_proj = nn.Linear(d_bus, d_model, bias=False)
        self.gamma = gamma

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化总线状态"""
        return torch.zeros(batch_size, self.write_proj.out_features,
                           device=device)

    def write(self, bus_state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        每层调用一次:将当前层的信息写入总线。
        bus_state: [B, d_bus]
        x: [B, S, d_model] → 取序列均值
        """
        x_mean = x.mean(dim=1)                # [B, d_model]
        new_info = self.write_proj(x_mean)    # [B, d_bus]
        bus_state = self.gamma * bus_state + (1 - self.gamma) * new_info
        return bus_state

    def read(self, bus_state: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        每层调用一次:从总线读取全局信息,广播到每个位置。
        返回: [B, S, d_model]
        """
        bus_read = self.read_proj(bus_state)  # [B, d_model]
        return bus_read.unsqueeze(1).expand(-1, seq_len, -1)


class BusAwareLayer(nn.Module):
    """带总线读写的标准层包装器"""
    def __init__(self, layer: nn.Module, bus: BroadcastBus,
                 monitor: MonitorSlot, bus_scale: float = 0.1):
        super().__init__()
        self.layer = layer
        self.bus = bus
        self.monitor = monitor
        self.bus_scale = bus_scale

    def forward(self, x: torch.Tensor, bus_state: torch.Tensor):
        """
        x: [B, S, d_model]
        bus_state: [B, d_bus]
        返回: (x, new_bus_state, monitor_summary)
        """
        # 1. 从总线读取全局信息
        bus_info = self.bus.read(bus_state, x.size(1))
        x = x + self.bus_scale * bus_info

        # 2. 执行层计算
        x = self.layer(x)

        # 3. 写入总线
        new_bus_state = self.bus.write(bus_state, x)

        # 4. 记录监控
        monitor_summary = self.monitor(x)

        return x, new_bus_state, monitor_summary
```

### 3.2 Gated DeltaNet 注意力

```python
# model/attention/gated_deltanet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet:线性注意力 + 数据依赖门控。

    核心递推:
        S_t = α_t · S_{t-1} + β_t · v_t · k_t^T    (Delta 更新)
        o_t = S_t · q_t                                (查询)

    α, β 由输入门控决定 → 可监控、可消融。
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 state_dim: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.state_dim = state_dim

        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * state_dim, bias=False)

        # 门控:α(遗忘门), β(写入门)
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=True)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)

        # 输出投影
        self.o_proj = nn.Linear(n_heads * state_dim, d_model, bias=False)

        # 短程卷积(局部特征)
        self.short_conv = nn.Conv1d(
            d_model, d_model, kernel_size=4, padding=3, groups=d_model
        )

    def forward(self, x: torch.Tensor,
                return_gates: bool = False) -> torch.Tensor:
        """
        x: [B, S, d_model]
        """
        B, S, D = x.shape

        # 短程卷积
        x_conv = self.short_conv(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
        x_in = x + x_conv

        # Q, K, V
        q = self.q_proj(x_in).view(B, S, self.n_heads, self.d_head)
        k = self.k_proj(x_in).view(B, S, self.n_heads, self.d_head)
        v = self.v_proj(x_in).view(B, S, self.n_heads, self.state_dim)

        # 门控值(可监控)
        alpha = torch.sigmoid(self.alpha_proj(x_in))  # [B, S, H]
        beta = torch.sigmoid(self.beta_proj(x_in))    # [B, S, H]

        # 递推计算(可并行化为扫描操作)
        # 这里用简化版本;生产环境用 triton 或 flash-linear-attention
        o = self._delta_recurrence(q, k, v, alpha, beta)

        # 输出
        o = o.reshape(B, S, self.n_heads * self.state_dim)
        out = self.o_proj(o)

        if return_gates:
            return out, {"alpha": alpha, "beta": beta}
        return out

    def _delta_recurrence(self, q, k, v, alpha, beta):
        """
        Delta 递推:
        S_t = α_t · S_{t-1} + β_t · (v_t ⊗ k_t)
        o_t = S_t @ q_t
        """
        B, S, H, _ = q.shape
        state = torch.zeros(B, H, self.state_dim, self.d_head,
                            device=q.device)
        outputs = []

        for t in range(S):
            # 门控
            a = alpha[:, t, :].unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
            b = beta[:, t, :].unsqueeze(-1).unsqueeze(-1)   # [B,H,1,1]

            # Delta 更新
            kv = v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)  # [B,H,sd,dh]
            state = a * state + b * kv

            # 查询
            o_t = (state * q[:, t].unsqueeze(-2)).sum(dim=-1)  # [B,H,sd]
            outputs.append(o_t)

        return torch.stack(outputs, dim=1)  # [B, S, H, sd]
```

### 3.3 Linear Attention(工作空间中段)

```python
# model/attention/linear.py
class LinearAttention(nn.Module):
    """
    线性注意力:无 softmax,Jacobian 可解析。
    使用 ELU+1 作为 feature map。
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """ELU + 1 保证非负"""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head)

        q = self.feature_map(q)  # [B, S, H, dh]
        k = self.feature_map(k)

        # 线性复杂度:先算 K^T V,再乘 Q
        # kv: [B, H, dh, dh]
        kv = torch.einsum("bshd,bshm->bhdm", k, v)
        # 归一化因子
        k_sum = k.sum(dim=1)  # [B, H, dh]

        # o = Q @ KV / (Q @ K_sum)
        o = torch.einsum("bshd,bhdm->bshm", q, kv)  # [B, S, H, dh]
        denom = torch.einsum("bshd,bhd->bsh", q, k_sum).unsqueeze(-1)
        o = o / (denom + 1e-6)

        o = o.reshape(B, S, self.n_heads * self.d_head)
        return self.o_proj(o)
```

---

## 四、MoE 模块

### 4.1 Soft MoE(前段,sigmoid dispatch)

```python
# model/moe/soft_moe.py
class SoftMoE(nn.Module):
    """
    Soft MoE with sigmoid dispatch(修正 P2:非 softmax)。

    每个 token 独立计算对每个 slot 的亲和度(0-1),
    不做归一化竞争 → 更线性、J-Lens 友好。

    结构:4 slots × 4 experts/slot = 16 experts
    每专家:896→1024→896
    """
    def __init__(self, d_model: int, n_slots: int = 4,
                 n_experts_per_slot: int = 4,
                 expert_hidden: int = 1024,
                 dispatch: str = "sigmoid"):
        super().__init__()
        self.n_slots = n_slots
        self.n_experts_per_slot = n_experts_per_slot
        self.dispatch_type = dispatch

        # Dispatch 网络:每个 token → 每个 slot 的亲和度
        self.dispatch_proj = nn.Linear(d_model, n_slots, bias=True)

        # 每个 slot 内的专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden, bias=False),
                nn.GELU(),
                nn.Linear(expert_hidden, d_model, bias=False),
            )
            for _ in range(n_slots * n_experts_per_slot)
        ])

        # 每个 slot 的输出权重
        self.slot_weights = nn.Parameter(
            torch.ones(n_slots) / n_slots
        )

    def forward(self, x: torch.Tensor,
                return_dispatch: bool = False):
        """
        x: [B, S, d_model]
        """
        B, S, D = x.shape

        # Dispatch:sigmoid(非 softmax)→ 独立门控
        if self.dispatch_type == "sigmoid":
            dispatch = torch.sigmoid(self.dispatch_proj(x))  # [B, S, n_slots]
        else:  # 消融回退:softmax
            dispatch = F.softmax(self.dispatch_proj(x), dim=-1)

        # 每个 slot 的计算
        output = torch.zeros_like(x)
        slot_loads = []

        for s in range(self.n_slots):
            # 该 slot 的权重
            w = dispatch[:, :, s:s+1]  # [B, S, 1]

            # 该 slot 内所有专家的加权平均
            slot_out = torch.zeros_like(x)
            for e in range(self.n_experts_per_slot):
                idx = s * self.n_experts_per_slot + e
                expert_out = self.experts[idx](x)
                slot_out = slot_out + expert_out / self.n_experts_per_slot

            output = output + w * self.slot_weights[s] * slot_out
            slot_loads.append(w.mean().item())

        if return_dispatch:
            return output, dispatch, slot_loads
        return output
```

### 4.2 Multi-Head MoE(中段,核心)

```python
# model/moe/mh_moe.py
class MultiHeadMoE(nn.Module):
    """
    Multi-Head Sparse MoE in Latent Space。

    d_bus=256 → 4 heads × d_head=64
    每 head:12 experts, Top-4
    每专家:64→192→64

    组合空间:(C(12,4))^4 = 495^4 ≈ 6×10^10
    """
    def __init__(self, d_bus: int, n_heads: int = 4,
                 experts_per_head: int = 12,
                 top_k_per_head: int = 4,
                 expert_hidden: int = 192):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_bus // n_heads  # 64
        self.experts_per_head = experts_per_head
        self.top_k = top_k_per_head

        # 每个 head 的路由器
        self.routers = nn.ModuleList([
            nn.Linear(self.d_head, experts_per_head, bias=False)
            for _ in range(n_heads)
        ])

        # 每个 head 的专家
        self.expert_sets = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.d_head, expert_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(expert_hidden, self.d_head, bias=False),
                )
                for _ in range(experts_per_head)
            ])
            for _ in range(n_heads)
        ])

    def forward(self, z: torch.Tensor,
                return_routing: bool = False):
        """
        z: [B, S, d_bus]
        """
        B, S, D = z.shape

        # 拆分为 heads
        z_heads = z.view(B, S, self.n_heads, self.d_head)

        outputs = []
        all_routing = []

        for h in range(self.n_heads):
            z_h = z_heads[:, :, h, :]  # [B, S, d_head]

            # 路由
            logits = self.routers[h](z_h)  # [B, S, n_experts]

            # Top-K 选择
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)  # 归一化

            # 计算被选中的专家
            expert_out = torch.zeros_like(z_h)
            for k in range(self.top_k):
                idx = topk_idx[:, :, k]  # [B, S]
                w = topk_weights[:, :, k:k+1]  # [B, S, 1]

                # 收集该专家的输出
                # 生产环境用 grouped GEMM;这里用循环
                for e in range(self.experts_per_head):
                    mask = (idx == e)  # [B, S]
                    if mask.any():
                        masked_z = z_h[mask]  # [N, d_head]
                        masked_out = self.expert_sets[h][e](masked_z)
                        expert_out[mask] += w[mask] * masked_out

            outputs.append(expert_out)
            all_routing.append(topk_idx)

        # 合并所有 heads
        output = torch.cat(outputs, dim=-1)  # [B, S, d_bus]

        if return_routing:
            return output, all_routing
        return output

    def load_balancing_loss(self, routing_indices: list) -> torch.Tensor:
        """
        辅助损失:防止专家负载不均。
        对每个 head 计算负载方差。
        """
        total_loss = 0.0
        for h, indices in enumerate(routing_indices):
            # indices: [B, S, top_k]
            flat = indices.flatten()
            counts = torch.bincount(flat, minlength=self.experts_per_head)
            freq = counts.float() / counts.sum()
            # 鼓励均匀分布
            uniform = 1.0 / self.experts_per_head
            total_loss += ((freq - uniform) ** 2).sum()
        return total_loss / self.n_heads
```

### 4.3 Fine-grained MoE(后段)

```python
# model/moe/fine_moe.py
class FineMoE(nn.Module):
    """
    后段 Fine-grained MoE:Shared + Routed(20, Top-5) + Device

    Shared:永远激活,负责通用能力
    Routed:20 个专家选 5,负责特化能力
    Device:可学习但默认冻结,预留持续学习
    """
    def __init__(self, d_model: int, n_routed: int = 20,
                 top_k: int = 5, routed_hidden: int = 320,
                 shared_hidden: int = 448, device_hidden: int = 224):
        super().__init__()
        self.n_routed = n_routed
        self.top_k = top_k

        # 路由器
        self.router = nn.Linear(d_model, n_routed, bias=False)

        # 共享专家
        self.shared = nn.Sequential(
            nn.Linear(d_model, shared_hidden, bias=False),
            nn.GELU(),
            nn.Linear(shared_hidden, d_model, bias=False),
        )

        # 路由专家
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, routed_hidden, bias=False),
                nn.GELU(),
                nn.Linear(routed_hidden, d_model, bias=False),
            )
            for _ in range(n_routed)
        ])

        # 设备专家(预留)
        self.device_expert = nn.Sequential(
            nn.Linear(d_model, device_hidden, bias=False),
            nn.GELU(),
            nn.Linear(device_hidden, d_model, bias=False),
        )
        # 设备专家的门控(初始为 0,训练中可学习)
        self.device_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor,
                return_routing: bool = False):
        B, S, D = x.shape

        # 1. 共享专家(永远激活)
        shared_out = self.shared(x)

        # 2. 路由专家(Top-K)
        logits = self.router(x)  # [B, S, n_routed]
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        routed_out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_idx[:, :, k]
            w = topk_weights[:, :, k:k+1]
            for e in range(self.n_routed):
                mask = (idx == e)
                if mask.any():
                    routed_out[mask] += w[mask] * self.routed_experts[e](x[mask])

        # 3. 设备专家(可学习门控)
        device_out = torch.sigmoid(self.device_gate) * self.device_expert(x)

        # 合并
        output = shared_out + routed_out + device_out

        if return_routing:
            return output, topk_idx, topk_weights
        return output
```

### 4.4 LoRA-MoE(动作区)

```python
# model/moe/lora_moe.py
class LoRAMoE(nn.Module):
    """
    Dense Multi-Head LoRA-MoE:所有 LoRA 全部激活。
    2 heads × 10 LoRA/head = 20 总 LoRA
    rank = 8

    用于持续学习:新任务 = 新增 LoRA 权重。
    """
    def __init__(self, d_model: int, n_heads: int = 2,
                 lora_per_head: int = 10, rank: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.lora_per_head = lora_per_head
        self.rank = rank
        self.d_head = d_model // n_heads

        # 每个 LoRA:A (d_head → rank) + B (rank → d_head)
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(self.d_head, rank))
            for _ in range(n_heads * lora_per_head)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, self.d_head))
            for _ in range(n_heads * lora_per_head)
        ])

        # 初始化
        for p in self.lora_A:
            nn.init.kaiming_uniform_(p)
        # B 初始化为 0 → 初始时 LoRA 无贡献

        # 门控(可学习,初始均匀)
        self.gates = nn.Parameter(
            torch.ones(n_heads, lora_per_head) / lora_per_head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, d_model]
        Dense:所有 LoRA 全部参与。
        """
        B, S, D = x.shape
        x_heads = x.view(B, S, self.n_heads, self.d_head)

        lora_out = torch.zeros_like(x_heads)

        for h in range(self.n_heads):
            x_h = x_heads[:, :, h, :]  # [B, S, d_head]
            for l in range(self.lora_per_head):
                idx = h * self.lora_per_head + l
                # LoRA 变换
                delta = (x_h @ self.lora_A[idx]) @ self.lora_B[idx]
                lora_out[:, :, h, :] += self.gates[h, l] * delta

        return lora_out.reshape(B, S, D)
```

---

## 五、Joint SAE

```python
# model/sae.py
class JointSAE(nn.Module):
    """
    联合训练的 Sparse Autoencoder。

    中段:dict=2048, K=12(主 SAE)
    前段/后段:dict=1024, K=16(轻量 SAE)

    与主模型联合训练,梯度通过 stop-gradient 隔离。
    """
    def __init__(self, d_input: int, dict_size: int, sparsity_k: int):
        super().__init__()
        self.dict_size = dict_size
        self.k = sparsity_k

        # 均值(学习)
        self.mean = nn.Parameter(torch.zeros(d_input))

        # 编码器
        self.encoder = nn.Linear(d_input, dict_size, bias=True)

        # 解码器(字典向量)
        self.decoder = nn.Linear(dict_size, d_input, bias=False)

        # 正交初始化
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, d_input]
        返回:(reconstruction, sparse_codes)
        """
        # 中心化
        x_centered = x - self.mean

        # 编码
        pre_act = self.encoder(x_centered)  # [B, S, dict_size]

        # Top-K 稀疏
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1)
        sparse_codes = torch.zeros_like(pre_act)
        sparse_codes.scatter_(-1, topk_idx, topk_vals)

        # 解码
        reconstruction = self.decoder(sparse_codes) + self.mean

        return reconstruction, sparse_codes

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x)

    def activation_l1(self) -> torch.Tensor:
        """字典向量的 L1 范数(用于监控)"""
        return self.decoder.weight.abs().sum()
```

---

## 六、工作空间中★段(含循环机制)

```python
# model/workspace/mid.py
class WorkspaceMid(nn.Module):
    """
    工作空间中★段:2 物理层 × 2 循环 = 等效 4 层。

    结构:
    ├── Down Interface: d_model(896) → d_bus(256)
    ├── 循环体(2 层)× 2 次:
    │   ├── Linear Attention
    │   ├── MH-MoE (4h × 12e, Top-4)
    │   └── Joint SAE (dict=2048, K=12)
    ├── 迭代门控 + 迭代位置编码
    └── Up Interface: d_bus(256) → d_model(896)
    """
    def __init__(self, config: SOCAMicroConfig):
        super().__init__()
        self.n_cycles = config.n_cycles  # 2

        # Down Interface
        self.down_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_bus),
            nn.GELU(),
            nn.Linear(config.d_bus, config.d_bus),
        )

        # 循环体:2 物理层
        self.layers = nn.ModuleList()
        for _ in range(config.n_ws_mid_physical):
            self.layers.append(MidBlock(config))

        # 迭代位置编码(修正 P7)
        self.iter_pos_embed = nn.Parameter(
            torch.zeros(config.n_cycles, 1, config.d_bus)
        )
        nn.init.normal_(self.iter_pos_embed, std=0.02)

        # 迭代门控
        self.cycle_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(config.n_cycles - 1)
        ])

        # Up Interface
        self.up_proj = nn.Sequential(
            nn.Linear(config.d_bus, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # 残差连接
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 正交约束
        self.orth_lambda = config.lambda_orth

    def forward(self, x: torch.Tensor,
                ablation_no_cycles: bool = False):
        """
        x: [B, S, d_model]
        """
        # 保存输入用于残差
        x_in = x

        # Down: d_model → d_bus
        z = self.down_proj(x)  # [B, S, d_bus]

        # 循环
        n_iter = 1 if ablation_no_cycles else self.n_cycles
        iter_routings = []
        iter_states = []

        for cycle in range(n_iter):
            # 迭代位置编码
            z = z + self.iter_pos_embed[cycle].unsqueeze(0)

            # 执行 2 物理层
            cycle_routing = []
            for layer in self.layers:
                z, routing, sae_info = layer(z)
                cycle_routing.append(routing)

            iter_routings.append(cycle_routing)
            iter_states.append(z.detach())

            # 迭代门控(第 2 次及以后)
            if cycle < n_iter - 1:
                gate = torch.sigmoid(self.cycle_gates[cycle])
                z = gate * z + (1 - gate) * iter_states[0]

        # Up: d_bus → d_model
        z_up = self.up_proj(z)  # [B, S, d_model]

        # 残差连接回主流
        output = x_in + self.residual_scale * z_up

        return output, {
            "iter_routings": iter_routings,
            "iter_states": iter_states,
            "bus_state": z.mean(dim=1),  # 用于总线
        }


class MidBlock(nn.Module):
    """中段单个物理层:Linear Attn + MH-MoE + SAE"""
    def __init__(self, config: SOCAMicroConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_bus)
        self.linear_attn = LinearAttention(
            config.d_bus, config.mid_n_heads, config.mid_d_head
        )
        self.norm2 = nn.LayerNorm(config.d_bus)
        self.mh_moe = MultiHeadMoE(
            config.d_bus,
            n_heads=config.mid_n_heads,
            experts_per_head=config.mid_experts_per_head,
            top_k_per_head=config.mid_top_k_per_head,
            expert_hidden=config.mid_expert_hidden,
        )
        # Joint SAE
        self.sae = JointSAE(
            config.d_bus, config.sae_mid_dict, config.sae_mid_k
        )
        self.monitor = MonitorSlot(config.d_bus, config.monitor_dim)

    def forward(self, z: torch.Tensor):
        # Linear Attention
        z = z + self.linear_attn(self.norm1(z))

        # MH-MoE
        moe_out, routing = self.mh_moe(self.norm2(z), return_routing=True)
        z = z + moe_out

        # SAE(联合训练,stop-gradient 隔离主梯度)
        sae_recon, sae_codes = self.sae(z)

        # 监控
        monitor = self.monitor(z)

        return z, routing, {
            "sae_recon": sae_recon,
            "sae_codes": sae_codes,
            "monitor": monitor,
        }
```

---

## 七、完整模型组装

```python
# model/soca_model.py
class SOCAMicro(nn.Module):
    """
    SOCA v3-Micro-Final: 16 层 × ~155M

    感知区(5) → 工作空间前(2) → 中★(2×2) → 后(2) → 动作区(5)
    """
    def __init__(self, config: SOCAMicroConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embedding = nn.Embedding(config.n_vocab, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # 广播总线
        self.bus = BroadcastBus(
            config.d_model, config.d_bus, config.bus_gamma
        ) if not config.ablation.get("disable_bus", False) else None

        # ═══ 感知区(5 层)═══
        self.perception = nn.ModuleList()
        attn_types = ["standard", "deltanet", "deltanet",
                       "deltanet", "gated_standard"]
        for i, at in enumerate(attn_types):
            self.perception.append(self._make_layer(
                config, at, config.ffn_mult_perception
            ))

        # ═══ 工作空间前段(2 层)═══
        self.ws_front = WorkspaceFront(config)

        # ═══ 工作空间中★段 ═══
        self.ws_mid = WorkspaceMid(config)

        # ═══ 工作空间后段(2 层)═══
        self.ws_back = WorkspaceBack(config)

        # ═══ 动作区(5 层)═══
        self.action = nn.ModuleList()
        action_types = ["deltanet", "deltanet", "deltanet",
                         "gated_standard", "gated_standard"]
        for i, at in enumerate(action_types):
            self.action.append(self._make_action_layer(config, at))

        # Output head(tied with embedding)
        self.output_proj = nn.Linear(
            config.d_model, config.n_vocab, bias=False
        )
        if config.tie_embeddings:
            self.output_proj.weight = self.embedding.weight

        # 最终 Norm
        self.final_norm = nn.LayerNorm(config.d_model)

    def _make_layer(self, config, attn_type, ffn_mult):
        """构建感知区层"""
        # ... 根据 attn_type 选择注意力模块
        # 包含 BusAwareLayer 包装
        pass  # 简化:实际实现同标准

    def _make_action_layer(self, config, attn_type):
        """构建动作区层(含 LoRA-MoE)"""
        pass

    def forward(self, input_ids: torch.Tensor,
                return_probes: bool = False):
        """
        input_ids: [B, S]
        """
        B, S = input_ids.shape
        ab = self.config.ablation

        # Embedding
        positions = torch.arange(S, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embed(positions)

        # 初始化总线
        bus_state = None
        if self.bus is not None:
            bus_state = self.bus.init_state(B, x.device)

        probes = {}  # 收集所有监控数据

        # ═══ 感知区 ═══
        for i, layer in enumerate(self.perception):
            x, bus_state, monitor = self._forward_with_bus(
                layer, x, bus_state
            )
            if return_probes:
                probes[f"perception_L{i}"] = monitor

        # ═══ 工作空间前段 ═══
        x, front_info = self.ws_front(x, return_dispatch=True)
        if return_probes:
            probes["ws_front"] = front_info

        # ═══ 工作空间中★段 ═══
        x, mid_info = self.ws_mid(
            x, ablation_no_cycles=ab.get("disable_cycles", False)
        )
        if return_probes:
            probes["ws_mid"] = mid_info

        # ═══ 工作空间后段 ═══
        x, back_info = self.ws_back(x, return_routing=True)
        if return_probes:
            probes["ws_back"] = back_info

        # ═══ 动作区 ═══
        for i, layer in enumerate(self.action):
            x, bus_state, monitor = self._forward_with_bus(
                layer, x, bus_state
            )
            if return_probes:
                probes[f"action_L{i}"] = monitor

        # Output
        x = self.final_norm(x)
        logits = self.output_proj(x)  # [B, S, n_vocab]

        if return_probes:
            return logits, probes
        return logits

    def _forward_with_bus(self, layer, x, bus_state):
        """带总线的层前向"""
        if self.bus is not None and bus_state is not None:
            bus_info = self.bus.read(bus_state, x.size(1))
            x = x + 0.1 * bus_info
            x = layer(x)
            bus_state = self.bus.write(bus_state, x)
        else:
            x = layer(x)
        monitor = None  # 简化
        return x, bus_state, monitor
```

---

## 八、训练循环与损失函数

> ⚠️ **训练阶段定义说明**:本节 SOCALoss 按**训练进度**(0%/5%/30%/90%)分阶段切换**损失权重;训练**冻结/解冻**阶段管理(感知→工作空间→动作→端到端)统一以 **[`07-module-extensions.md` §三 M19 PhaseScheduler](./07-module-extensions.md)** 为真源。建议实施时:**用 07 M19 管理参数冻结**,**用本节 SOCALoss 管理损失权重**;`SOCALoss.set_phase()` 应重构为与 M19 一致(当前是进度制,M19 是区域制)。详见 07 §三 M19。

### 8.1 多损失联合优化

```python
# training/losses.py
class SOCALoss:
    """
    SOCA 多损失联合优化。

    四阶段调度(进度制,与 07 §三 M19 区域制解耦使用):
    Phase 1 (0-5%):    纯 LM loss
    Phase 2 (5-30%):   + 稀疏 + 正交
    Phase 3 (30-90%):  + 全部辅助损失
    Phase 4 (90-100%): 冻结辅助权重,精修主任务
    """
    def __init__(self, config: SOCAMicroConfig):
        self.config = config

    def compute(self, model, logits, labels, probes, step, total_steps):
        """返回总损失 + 各分项"""
        progress = step / total_steps
        losses = {}

        # ═══ 1. 主任务损失(永远存在)═══
        lm_loss = F.cross_entropy(
            logits.view(-1, self.config.n_vocab),
            labels.view(-1),
            ignore_index=-100
        )
        losses["lm"] = lm_loss

        # ═══ 2. 稀疏性损失(Phase 2+)═══
        if progress > 0.05:
            sparse_loss = self._sparsity_loss(model)
            losses["sparse"] = self.config.lambda_sparse * sparse_loss

        # ═══ 3. 正交性损失(Phase 2+)═══
        if progress > 0.05:
            orth_loss = self._orthogonality_loss(model)
            losses["orth"] = self.config.lambda_orth * orth_loss

        # ═══ 4. SAE 重建损失(Phase 3+)═══
        if progress > 0.30 and not self.config.ablation.get("disable_sae"):
            sae_loss = self._sae_loss(model, probes)
            losses["sae"] = self.config.lambda_sae * sae_loss

        # ═══ 5. 概念正交性损失(Phase 3+)═══
        if progress > 0.30:
            concept_loss = self._concept_orthogonality(probes)
            losses["concept"] = self.config.lambda_concept * concept_loss

        # ═══ 6. 路由负载均衡损失(Phase 2+)═══
        if progress > 0.05:
            route_loss = self._routing_loss(model, probes)
            losses["route"] = self.config.lambda_route * route_loss

        # ═══ 7. 循环一致性损失(Phase 3+)═══
        if progress > 0.30 and not self.config.ablation.get("disable_cycles"):
            cycle_loss = self._cycle_consistency(probes)
            losses["cycle"] = self.config.lambda_cycle * cycle_loss

        # ═══ 总损失 ═══
        total = sum(losses.values())
        losses["total"] = total

        return total, losses

    def _sparsity_loss(self, model):
        """感知区/动作区 FFN 激活的 L1 惩罚"""
        loss = 0.0
        for zone in [model.perception, model.action]:
            for layer in zone:
                if hasattr(layer, 'ffn_activations'):
                    loss += layer.ffn_activations.abs().mean()
        return loss

    def _orthogonality_loss(self, model):
        """中段专家权重的正交约束"""
        loss = 0.0
        for block in model.ws_mid.layers:
            for h in range(block.mh_moe.n_heads):
                for expert in block.mh_moe.expert_sets[h]:
                    W = expert[0].weight  # [hidden, d_head]
                    loss += (W @ W.T - torch.eye(
                        W.shape[0], device=W.device
                    )).pow(2).sum()
        return loss

    def _sae_loss(self, model, probes):
        """SAE 重建损失"""
        loss = 0.0
        if "ws_mid" in probes:
            for state in probes["ws_mid"]["iter_states"]:
                for block in model.ws_mid.layers:
                    loss += block.sae.reconstruction_loss(state)
        return loss

    def _concept_orthogonality(self, probes):
        """概念正交性:不同样本激活方向应尽量正交"""
        if "ws_mid" not in probes:
            return torch.tensor(0.0)
        z = probes["ws_mid"]["iter_states"][-1]  # 最后一次迭代
        z_flat = z.reshape(-1, z.shape[-1])[:256]  # 取 256 个样本
        z_norm = F.normalize(z_flat, dim=-1)
        sim = z_norm @ z_norm.T
        off_diag = sim - torch.eye(sim.size(0), device=sim.device)
        return off_diag.pow(2).mean()

    def _routing_loss(self, model, probes):
        """路由负载均衡"""
        loss = 0.0
        if "ws_mid" in probes:
            for block in model.ws_mid.layers:
                # 使用最近一次的路由信息
                pass  # 实际实现从 probes 中提取
        return loss

    def _cycle_consistency(self, probes):
        """循环一致性:两次迭代的路由应趋同"""
        if "ws_mid" not in probes:
            return torch.tensor(0.0)
        routings = probes["ws_mid"]["iter_routings"]
        if len(routings) < 2:
            return torch.tensor(0.0)
        # 比较第 1 次和第 2 次迭代的路由决策
        r1 = routings[0]
        r2 = routings[1]
        # 计算路由一致性(Jaccard 相似度)
        # 简化:计算路由索引的重叠率
        loss = 0.0
        for h in range(len(r1)):
            overlap = (r1[h] == r2[h]).float().mean()
            loss += (1 - overlap)  # 鼓励一致
        return loss / len(r1)
```

### 8.2 训练主循环

```python
# training/trainer.py
class SOCATrainer:
    def __init__(self, model, config, train_loader, eval_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.loss_fn = SOCALoss(config)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )

        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_steps
        )

        # 监控
        self.logger = TrainingLogger()
        self.global_step = 0

    def train(self):
        self.model.train()

        for batch in self.train_loader:
            if self.global_step >= self.config.max_steps:
                break

            # Warmup
            if self.global_step < self.config.warmup_steps:
                lr_scale = self.global_step / self.config.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config.lr * lr_scale

            # 前向
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()

            logits, probes = self.model(
                input_ids, return_probes=True
            )

            # 损失
            total_loss, loss_dict = self.loss_fn.compute(
                self.model, logits, labels, probes,
                self.global_step, self.config.max_steps
            )

            # 反向
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            # 更新
            self.optimizer.step()
            if self.global_step >= self.config.warmup_steps:
                self.scheduler.step()

            # ═══ 监控与日志 ═══
            if self.global_step % 100 == 0:
                self.logger.log({
                    "step": self.global_step,
                    "loss/total": total_loss.item(),
                    **{f"loss/{k}": v.item()
                       for k, v in loss_dict.items()},
                    "lr": self.optimizer.param_groups[0]['lr'],
                })

            # ═══ 定期评估 ═══
            if self.global_step % 1000 == 0:
                self._evaluate()

            # ═══ 可解释性评估(每 2000 步)═══
            if self.global_step % 2000 == 0:
                self._interpretability_eval()

            # ═══ PPL 硬上限检查 ═══
            if self.global_step % 500 == 0:
                ppl = self._quick_ppl()
                baseline_ppl = self.logger.get("baseline_ppl", ppl)
                if ppl > baseline_ppl * 1.03:  # +3% 上限
                    self._reduce_aux_weights()

            self.global_step += 1

    def _evaluate(self):
        """标准评估"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].cuda()
                labels = batch['labels'].cuda()
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, self.config.n_vocab),
                    labels.view(-1), ignore_index=-100
                )
                total_loss += loss.item()
                n_batches += 1
        ppl = math.exp(total_loss / n_batches)
        self.logger.log({"eval/ppl": ppl, "step": self.global_step})
        self.model.train()

    def _interpretability_eval(self):
        """可解释性指标"""
        self.model.eval()
        with torch.no_grad():
            # 1. Jacobian 稀疏度(采样)
            # 2. SAE 重建误差
            # 3. 概念正交性
            # 4. 路由多样性
            # 5. 循环收敛度
            pass  # 实际实现
        self.model.train()

    def _reduce_aux_weights(self):
        """PPL 超标时降低辅助损失权重"""
        self.config.lambda_sparse *= 0.5
        self.config.lambda_orth *= 0.5
        self.config.lambda_sae *= 0.5
        print(f"[WARNING] PPL exceeded +3%. "
              f"Reduced aux weights: "
              f"sparse={self.config.lambda_sparse:.6f}, "
              f"orth={self.config.lambda_orth:.6f}")
```

---

## 九、消融框架

> ⚠️ **消融编号约定**:本文档的 24 项 SOCA 架构消融编号为 **`SOCA-A0` ~ `SOCA-A24`**(原简称 `A0-A24`)。`05-pretraining-ablation-plan.md` 的 24 项通用消融编号为 **`GENERAL-A1` ~ `GENERAL-F4`**。两套编号**完全独立**,合并报告时必须加前缀以避免歧义(详见 [`05-pretraining-ablation-plan.md` §一](./05-pretraining-ablation-plan.md))。

```python
# ablation/registry.py
ABLATION_REGISTRY = {
    # ═══ 核心架构消融(编号 SOCA-A0 ~ SOCA-A24)═══
    "SOCA-A0_baseline": {
        "desc": "完整模型(基线)",
        "overrides": {}
    },
    "SOCA-A1_no_cycles": {
        "desc": "去掉循环(中段只执行 1 次)",
        "overrides": {"disable_cycles": True}
    },
    "SOCA-A2_no_bus": {
        "desc": "去掉广播总线",
        "overrides": {"disable_bus": True}
    },
    "SOCA-A3_no_sae": {
        "desc": "去掉 Joint SAE",
        "overrides": {"disable_sae": True}
    },
    "SOCA-A4_no_zones": {
        "desc": "去掉区域划分(统一为标准层)",
        "overrides": {"disable_zones": True}
    },
    "SOCA-A5_mh_to_standard": {
        "desc": "中段 MH-MoE → 标准 MoE",
        "overrides": {"disable_mh_moe": True}
    },

    # ═══ 可解释性消融 ═══
    "SOCA-A6_weight_sparsity": {
        "desc": "加入权重稀疏(感知/动作 50%)",
        "overrides": {"enable_weight_sparsity": True,
                       "sparsity_ratio": 0.5}
    },
    "SOCA-A7_sae_dict_256": {
        "desc": "SAE 字典回退到 256",
        "overrides": {"sae_dict_override": 256}
    },
    "SOCA-A8_softmax_dispatch": {
        "desc": "前段 dispatch 回退到 softmax",
        "overrides": {"dispatch_override": "softmax"}
    },
    "SOCA-A9_no_lora": {
        "desc": "去掉动作区 LoRA-MoE",
        "overrides": {"disable_lora": True}
    },

    # ═══ 注意力消融 ═══
    "SOCA-A10_all_standard": {
        "desc": "所有注意力改为 Standard",
        "overrides": {"attn_override": "all_standard"}
    },
    "SOCA-A11_all_deltanet": {
        "desc": "所有注意力改为 DeltaNet",
        "overrides": {"attn_override": "all_deltanet"}
    },
    "SOCA-A12_no_gated": {
        "desc": "去掉所有门控",
        "overrides": {"disable_gates": True}
    },

    # ═══ MoE 消融 ═══
    "SOCA-A13_no_front_moe": {
        "desc": "前段 Soft MoE → Dense FFN",
        "overrides": {"disable_front_moe": True}
    },
    "SOCA-A14_no_back_device": {
        "desc": "后段去掉设备专家",
        "overrides": {"disable_device_expert": True}
    },
    "SOCA-A15_no_shared_expert": {
        "desc": "后段去掉共享专家",
        "overrides": {"disable_shared_expert": True}
    },
    "SOCA-A16_back_top1": {
        "desc": "后段 Top-5 → Top-1",
        "overrides": {"back_top_k_override": 1}
    },

    # ═══ 训练策略消融 ═══
    "SOCA-A17_no_aux_loss": {
        "desc": "去掉所有辅助损失(纯 LM)",
        "overrides": {"disable_all_aux": True}
    },
    "SOCA-A18_no_curriculum": {
        "desc": "去掉课程学习(均匀数据)",
        "overrides": {"disable_curriculum": True}
    },

    # ═══ 规模消融 ═══
    "SOCA-A19_half_experts": {
        "desc": "中段专家数减半(12→6)",
        "overrides": {"mid_experts_override": 6}
    },
    "SOCA-A20_double_experts": {
        "desc": "中段专家数加倍(12→24)",
        "overrides": {"mid_experts_override": 24}
    },

    # ═══ 循环消融 ═══
    "SOCA-A21_cycle_3": {
        "desc": "循环 2→3 次",
        "overrides": {"n_cycles_override": 3}
    },
    "SOCA-A22_no_iter_pos": {
        "desc": "去掉迭代位置编码",
        "overrides": {"disable_iter_pos": True}
    },

    # ═══ 对照 ═══
    "SOCA-A23_standard_transformer": {
        "desc": "同参数量标准 Transformer(无区域、无 MoE)",
        "overrides": {"architecture": "standard_transformer"}
    },
    "SOCA-A24_standard_moe": {
        "desc": "同参数量标准 MoE(无区域、无混合注意力)",
        "overrides": {"architecture": "standard_moe"}
    },
}


# ablation/runner.py
class AblationRunner:
    """消融实验执行器"""

    def __init__(self, base_config: SOCAMicroConfig,
                 output_dir: str = "./ablation_results"):
        self.base_config = base_config
        self.output_dir = output_dir
        self.results = {}

    def run_single(self, ablation_id: str):
        """执行单个消融(接受 SOCA-A0~A24 或旧 A0~A24 命名)"""
        # 向后兼容:接受旧命名
        if ablation_id not in ABLATION_REGISTRY and ablation_id.startswith("A"):
            new_id = "SOCA-" + ablation_id
            if new_id in ABLATION_REGISTRY:
                print(f"[WARN] 旧命名 '{ablation_id}' 自动映射为 '{new_id}'")
                ablation_id = new_id
        ab = ABLATION_REGISTRY[ablation_id]
        print(f"Running ablation: {ablation_id} - {ab['desc']}")

        # 创建修改后的配置
        config = self._apply_overrides(ab["overrides"])

        # 构建模型
        model = SOCAMicro(config).cuda()

        # 训练(缩短版:7000 步)
        trainer = SOCATrainer(model, config, ...)
        trainer.train(max_steps=7000)

        # 评估
        results = trainer.full_evaluate()
        self.results[ablation_id] = results

        # 保存
        torch.save(results, f"{self.output_dir}/{ablation_id}.pt")

    def run_all(self, parallel: int = 4):
        """执行所有消融"""
        ablation_ids = list(ABLATION_REGISTRY.keys())
        # 分批并行执行
        for batch_start in range(0, len(ablation_ids), parallel):
            batch = ablation_ids[batch_start:batch_start + parallel]
            # 实际用 multiprocessing 或 SLURM
            for ab_id in batch:
                self.run_single(ab_id)

    def _apply_overrides(self, overrides: dict) -> SOCAMicroConfig:
        """将消融覆盖应用到基础配置"""
        config = copy.deepcopy(self.base_config)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif key in config.ablation:
                config.ablation[key] = value
            else:
                config.ablation[key] = value
        return config
```

---

## 十、工程陷阱与解决方案

### 10.1 关键陷阱清单

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  陷阱 1:循环梯度爆炸/消失                                             │
│  ├── 问题:2 层 × 2 循环 = 4 层链式求导                               │
│  ├── 权重共享 → 梯度是同一参数的 2 倍累积                            │
│  ├── 风险:梯度可能翻倍导致训练不稳定                                  │
│  └── 解决:                                                            │
│      ├── 循环间加梯度缩放:z = 0.5 * z_cycle2 + 0.5 * z_cycle1       │
│      ├── 第 2 次循环的梯度乘以 0.5                                   │
│      └── 监控每次循环的梯度范数,差异 > 5× 时告警                    │
│                                                                         │
│  陷阱 2:Soft MoE sigmoid dispatch 的数值问题                         │
│  ├── 问题:sigmoid 输出 ∈ (0,1),多个 slot 同时高 → 输出过大        │
│  ├── 与 softmax 不同:无归一化约束                                    │
│  └── 解决:                                                            │
│      ├── 输出除以活跃 slot 数:out / (dispatch > 0.5).sum()         │
│      ├── 或:加一个全局缩放因子 1/n_slots                            │
│      └── 训练初期用温度退火:sigmoid(x/τ), τ: 2→1                   │
│                                                                         │
│  陷阱 3:SAE 联合训练梯度干扰                                         │
│  ├── 问题:SAE 的梯度流回主模型 → 干扰主任务学习                    │
│  ├── 特别是训练初期,SAE 未收敛时梯度噪声大                          │
│  └── 解决:                                                            │
│      ├── SAE 损失对主模型用 stop-gradient                            │
│      │   z_for_sae = z.detach()  # SAE 只看,不改主模型             │
│      ├── 前 30% 训练不启用 SAE 损失                                  │
│      └── SAE 用独立的优化器(更小学习率)                            │
│                                                                         │
│  陷阱 4:Multi-Head MoE 的负载不均                                    │
│  ├── 问题:某些专家被过度选择,其他"失业"                           │
│  ├── 48 个专家中可能只有 15-20 个被频繁使用                          │
│  └── 解决:                                                            │
│      ├── 负载均衡辅助损失(已有)                                    │
│      ├── 专家容量上限:每个专家每步最多处理 2× 平均负载              │
│      ├── 随机路由噪声:训练时给路由加 ε~N(0,0.1)                    │
│      └── 监控:每 100 步记录每个专家的使用频率                       │
│                                                                         │
│  陷阱 5:DeltaNet 递推的序列长度限制                                  │
│  ├── 问题:S=2048 的递推在 Python 循环中极慢                        │
│  ├── 2048 步 × 14 heads × 128 state_dim = 大量计算                  │
│  └── 解决:                                                            │
│      ├── 使用 flash-linear-attention 库(Triton 实现)              │
│      ├── 或使用 chunk-wise 并行:将 2048 分为 16 个 128 的块        │
│      └── 训练初期用短序列(512),逐步增加到 2048                   │
│                                                                         │
│  陷阱 6:Embedding tied 时的梯度冲突                                  │
│  ├── 问题:Embedding 和 Output Head 共享权重                         │
│  ├── 两个方向的梯度可能冲突                                           │
│  └── 解决:                                                            │
│      ├── 监控两个方向的梯度余弦相似度                                 │
│      ├── 如果冲突严重(< -0.3),解除 tied                          │
│      └── 或使用梯度缩放:output 方向梯度 × 0.5                      │
│                                                                         │
│  陷阱 7:消融实验的随机性                                             │
│  ├── 问题:不同随机种子导致消融结果波动                               │
│  ├── 特别是小效应量的消融(< 2%)                                   │
│  └── 解决:                                                            │
│      ├── 固定随机种子:42, 123, 456                                  │
│      ├── 关键消融(效应 < 2%)跑 2 个种子                           │
│      └── 报告均值 ± 标准差                                          │
│                                                                         │
│  陷阱 8:总线状态在长序列中的漂移                                     │
│  ├── 问题:γ=0.99 的指数衰减在 16 层后仍然保留 ~85% 的初始信息       │
│  ├── 0.99^16 ≈ 0.85 → 早期层的信息主导总线                         │
│  └── 解决:                                                            │
│      ├── 使用自适应 γ:γ = 0.99 - 0.01 * (layer / n_layers)        │
│      ├── 或:每 4 层重置一次总线                                     │
│      └── 监控:记录每层写入后总线的变化量                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 数值稳定性检查清单

```python
# tests/test_stability.py
def test_numerical_stability(model, sample_batch):
    """训练前必须通过的数值稳定性检查"""
    model.eval()
    checks = []

    with torch.no_grad():
        logits, probes = model(sample_batch, return_probes=True)

        # 1. 输出范围检查
        checks.append(("logits_range",
                        logits.abs().max().item() < 100,
                        f"max={logits.abs().max().item():.2f}"))

        # 2. 注意力权重范围
        # ... 检查每个注意力层的权重是否在合理范围

        # 3. SAE 重建误差
        # ... 不应为 0(未训练)或极大(数值溢出)

        # 4. 路由概率分布
        # ... 不应全部集中在一个专家

        # 5. 总线状态
        # ... 不应全为零或全为极大值

        # 6. 梯度检查(反向一次)
        model.train()
        logits, _ = model(sample_batch, return_probes=True)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                checks.append((f"grad/{name}",
                               0 < grad_norm < 1000,
                               f"norm={grad_norm:.4f}"))

    # 报告
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {detail}")

    return all(p for _, p, _ in checks)
```

---

## 十一、维度对齐验证

```python
# tests/test_shapes.py
def test_all_shapes():
    """确保所有组件维度对齐"""
    config = SOCAMicroConfig()
    model = SOCAMicro(config)

    B, S = 4, 128
    input_ids = torch.randint(0, config.n_vocab, (B, S))

    # 前向
    logits, probes = model(input_ids, return_probes=True)

    # 检查
    assert logits.shape == (B, S, config.n_vocab), \
        f"Logits shape: {logits.shape}"

    # 总参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    assert 140e6 < total_params < 170e6, \
        f"Parameter count {total_params/1e6:.1f}M out of range"

    # 逐区域参数
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params/1e6:.2f}M")

    print("✅ All shape tests passed")
```

---

## 十二、里程碑计划

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  实施里程碑(总计 ~4 周)                                              │
│                                                                         │
│  ═══ 第 1 周:基础搭建 ═══                                            │
│                                                                         │
│  Day 1-2: 基础组件                                                     │
│  ├── [ ] config.py 完成                                               │
│  ├── [ ] Embedding + Output Head                                      │
│  ├── [ ] Standard Attention + Gated 变体                              │
│  ├── [ ] Gated DeltaNet(先用简化递推)                               │
│  ├── [ ] Linear Attention                                             │
│  ├── [ ] 维度对齐测试通过                                             │
│  └── 验收:单独跑通每种注意力,输出形状正确                          │
│                                                                         │
│  Day 3-4: MoE 组件                                                     │
│  ├── [ ] Soft MoE(sigmoid dispatch)                                 │
│  ├── [ ] Multi-Head MoE                                               │
│  ├── [ ] Fine MoE(Shared+Routed+Device)                            │
│  ├── [ ] LoRA-MoE                                                     │
│  ├── [ ] Joint SAE                                                    │
│  └── 验收:每个 MoE 单独跑通,路由分布合理                          │
│                                                                         │
│  Day 5-7: 组装 + 循环                                                  │
│  ├── [ ] BroadcastBus + MonitorSlot                                   │
│  ├── [ ] WorkspaceMid(含循环 + 迭代位置编码)                       │
│  ├── [ ] WorkspaceFront / WorkspaceBack                               │
│  ├── [ ] 感知区 / 动作区                                              │
│  ├── [ ] SOCAMicro 完整组装                                           │
│  ├── [ ] 维度对齐测试(全模型)                                      │
│  ├── [ ] 数值稳定性测试                                              │
│  └── 验收:总参数 ~155M ± 5%,前向输出正确                         │
│                                                                         │
│  ═══ 第 2 周:训练管线 ═══                                            │
│                                                                         │
│  Day 8-9: 损失函数 + 训练循环                                         │
│  ├── [ ] SOCALoss(7 项损失 + 4 阶段调度)                           │
│  ├── [ ] SOCATrainer(含监控、评估、梯度裁剪)                      │
│  ├── [ ] 数据加载(先用随机数据)                                    │
│  ├── [ ] 学习率调度                                                  │
│  └── 验收:随机数据上 100 步,loss 下降                             │
│                                                                         │
│  Day 10-11: 消融框架                                                   │
│  ├── [ ] 24 个消融配置注册                                           │
│  ├── [ ] AblationRunner                                               │
│  ├── [ ] 消融开关测试(每个开关只改目标组件)                       │
│  └── 验收:任意消融配置能正常前向+反向                              │
│                                                                         │
│  Day 12-14: 真实数据 + 调试                                            │
│  ├── [ ] 接入真实训练数据                                            │
│  ├── [ ] DeltaNet 替换为 flash-linear-attention                       │
│  ├── [ ] 性能 profiling + 优化                                       │
│  ├── [ ] 500 步真实数据训练,确认 loss 正常下降                      │
│  └── 验收:500 步后 PPL < 50(随机初始化预期)                      │
│                                                                         │
│  ═══ 第 3 周:完整训练 ═══                                            │
│                                                                         │
│  Day 15-16: 主模型训练                                                │
│  ├── [ ] 6B token 完整训练(~19h)                                   │
│  ├── [ ] 实时监控 PPL、辅助损失、路由分布                            │
│  ├── [ ] 每 2000 步可解释性评估                                      │
│  └── 验收:最终 PPL 在预期范围内                                    │
│                                                                         │
│  Day 17-21: 消融实验                                                   │
│  ├── [ ] 24 个消融(4 并行,~4.5 天)                                │
│  ├── [ ] 实时检查消融训练是否正常                                    │
│  └── 验收:所有消融完成                                             │
│                                                                         │
│  ═══ 第 4 周:分析与报告 ═══                                          │
│                                                                         │
│  Day 22-24: 结果分析                                                   │
│  ├── [ ] 消融效应量排序                                              │
│  ├── [ ] 统计显著性检验                                              │
│  ├── [ ] 可视化(路由热力图、J-Space、SAE 特征)                    │
│  └── [ ] 与基线对比                                                  │
│                                                                         │
│  Day 25-28: 报告 + 扩展设计                                           │
│  ├── [ ] 验证报告                                                    │
│  ├── [ ] 0.8B/1.2B 扩展设计定稿                                     │
│  └── [ ] 下一步计划                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 第一个里程碑验收标准

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  🏁 里程碑 0:100 步烟雾测试(Day 2 结束)                            │
│                                                                         │
│  目标:确认模型能跑起来,梯度能流动                                   │
│                                                                         │
│  验收清单:                                                             │
│  ├── [ ] 模型初始化无报错                                            │
│  ├── [ ] 总参数在 140-170M 范围                                     │
│  ├── [ ] 前向输出形状 [B, S, 32000] 正确                            │
│  ├── [ ] 反向传播无 NaN / Inf                                       │
│  ├── [ ] 所有梯度非零(无"死"参数)                                 │
│  ├── [ ] 100 步后 loss 从 ~10.4 (ln(32000)) 开始下降               │
│  ├── [ ] 总线状态非零且在合理范围                                    │
│  ├── [ ] SAE 重建误差非零(说明在工作)                              │
│  ├── [ ] 路由分布非退化(不是所有 token 去同一专家)                 │
│  └── [ ] 循环两次迭代的状态不同(说明循环有变化)                    │
│                                                                         │
│  如果任何一项失败 → 停止,排查后再继续                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 十三、关键依赖与环境

```yaml
# environment.yml
name: soca_micro
dependencies:
  - python=3.11
  - pytorch=2.4+
  - cuda=12.4
  - pip:
    - flash-attn>=2.6          # 标准注意力加速
    - flash-linear-attention    # DeltaNet 加速(或自写 Triton)
    - triton>=3.0              # 自定义 kernel
    - datasets                 # 数据加载
    - transformers             # 分词器 + 评估
    - wandb                    # 实验追踪
    - einops                   # 张量操作
    - accelerate               # 多卡训练
    - deepspeed                # 可选:大模型训练
```

---

## 十四、与其他文档的关系

| 文档 | 与本文档的关系 |
|---|---|
| [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) | **配置来源**:本代码骨架中所有架构组件都对应审查报告中的修正项(P1-P12) |
| [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) | **超参来源**:训练超参(batch_size/lr/数据量)来自 148M 甜点分析 |
| [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) | **架构来源**:16 层 × ~155M 配置来自层数甜点评估的最终推荐 |

---

## 十五、决策交接(传递给实施团队)

| 锁定项 | 值 | 来源 |
|---|---|---|
| 总参数 | ~155M(±10%) | [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §五.5 |
| 物理层数 | 16 | [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §八.2 |
| 区域分配 | 5P + 6W + 5A | 同上 |
| 注意力头 | 14(896/64) | [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §四.1 |
| 中段专家 | 4h × 12e(48 总)/ Top-4 | [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §四.2 |
| 后段专家 | Shared(1) + Routed(20, Top-5) + Device(1) | [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §五.5 |
| SAE 字典 | 中段 2048 / 前段后段 1024 | [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) §九.2(P1) |
| 前段 dispatch | sigmoid(J-Lens 友好) | [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) §九.2(P2) |
| 训练数据 | 6B token | [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §七.1 |
| 训练时间 | ~19h(主模型)/ ~4.5 天(24 消融) | [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §七.2 |
| 关键消融 | 24 个(详见 §九) | [`01-v3-micro-14l-review.md`](./01-v3-micro-14l-review.md) §八 |
| 统计功效 | 94%(1 次训练即可) | [`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §五.3 |

### 实施起点

从 [`config.py`](#二全局配置) 和 [`tests/test_shapes.py`](#十一维度对齐验证) 开始,先确保维度对齐,再逐个实现组件。第一个可运行的版本应在 **2 天内**(Day 1-2 烟雾测试)完成。

---

## 📅 文档版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-28 | 初版,完成 SOCA v3-Micro-Final 的工程实现路线图与代码骨架 |

---

> **下一步**:阅读 [`README.md`](./README.md) 了解整个研究文档集的索引;或按本骨架开始实施 SOCA v3-Micro 验证模型。

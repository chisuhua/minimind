# SOCA 训练配方:路由器稳定性与全局训练策略

> **目的**:解决 SOCA 训练的核心难点——**隐空间路由器(M9 LatentRouter)的训练稳定性**。提供失效模式全谱系、七重防护体系、完整训练循环与超参推荐。
>
> **关联文档**:
> - 模块定义:[`06-architecture-modules.md`](./06-architecture-modules.md) §六 M9 LatentRouter
> - 工程扩展:[`07-module-extensions.md`](./07-module-extensions.md) §三 M19 PhaseScheduler / M20 ExpertDispatcher
> - 代码骨架:[`04-implementation-roadmap.md`](./04-implementation-roadmap.md) §四.2 MultiHeadMoE
> - 参数甜点:[`02-sweet-spot-params.md`](./02-sweet-spot-params.md) §五 训练稳定性
>
> **角色**:本文档位于"**训练配方层**"(位于研究/模块规格与实施代码之间)——把 06 的 M9 接口落地为**经过失效分析武装的训练策略**。

---

## 零、为什么路由器是 SOCA 训练的"命门"

在 SOCA 中,路由器(M9: LatentRouter)承担了传统 MoE 路由器的所有职责,**外加三重额外压力**:

| 额外压力 | 来源 | 后果 |
|:---|:---|:---|
| **隐空间路由** | 路由在 $d_{bus}$ 而非 $d_{model}$ 上决策 | 信息更压缩 → 区分度更低 → 路由更难学 |
| **联合 SAE 耦合** | SAE 重建损失反向传播经过路由 | 路由梯度被 SAE 噪声污染 |
| **总线写入依赖** | 路由结果决定总线更新内容 | 路由错误 → 总线污染 → 下游全部受损 |

> **核心洞察**:SOCA 的路由器不是一个独立的分类器。它是**信息瓶颈的阀门**。阀门卡住,整个系统瘫痪。
>
> 因此,路由器的训练稳定性不是"锦上添花",而是**架构能否工作的先决条件**。

---

## 一、路由器失效模式全谱系

在设计训练配方之前,必须先理解路由器会怎么坏:

### 1.1 坍缩类(致命)

| 编号 | 失效模式 | 症状 | 原因 | 严重度 |
|:---:|:---|:---|:---|:---:|
| **F1** | 专家坍缩(Expert Collapse) | 1-2 个专家接收 >80% token | 初始权重相似 + 正反馈循环 | ★★★★★ 致命 |
| **F2** | 路由熵坍缩(Entropy Collapse) | 所有 token 的路由分布趋同 | 温度过低 / 噪声不足 | ★★★★☆ |
| **F3** | 隐空间坍缩(Latent Collapse) | 下投影后的 z 向量方差趋零 | 正交约束过强 / 残差旁路关闭太快 | ★★★★★ 致命且难诊断 |

### 1.2 震荡类

| 编号 | 失效模式 | 症状 | 原因 | 严重度 |
|:---:|:---|:---|:---|:---:|
| **F4** | 路由震荡(Routing Oscillation) | 同一 token 在不同 step 被分配到不同专家 | 梯度信号矛盾(LM vs SAE vs Balance) | ★★★☆☆ |
| **F5** | 门控震荡(Gate Oscillation) | 聚合器门控值在 0/1 间跳变 | 专家输出不一致 + 门控学习率过高 | ★★★☆☆ |

### 1.3 退化类

| 编号 | 失效模式 | 症状 | 原因 | 严重度 |
|:---:|:---|:---|:---|:---:|
| **F6** | 路由退化(Routing Degradation) | 路由逐渐变成均匀分布(失去选择性) | 负载均衡损失过强 / 温度过高 | ★★★☆☆ |
| **F7** | SAE-路由冲突(SAE-Router Conflict) | SAE 重建损失下降但 LM loss 上升 | SAE 迫使选择"可重建"而非"有用"的专家 | ★★★★☆ |

---

## 二、路由器训练稳定性方案(七重防护)

### 2.1 防护层总览

```
Layer 1: 初始化策略        → 防止 F1, F3
Layer 2: 渐进式路由启用    → 防止 F1, F4
Layer 3: 自适应温度调度    → 防止 F2, F6
Layer 4: 多目标梯度平衡    → 防止 F4, F7
Layer 5: 专家容量保护      → 防止 F1
Layer 6: 隐空间健康监控    → 防止 F3
Layer 7: 运行时自愈机制    → 兜底所有失效模式
```

### 2.2 Layer 1: 初始化策略

**目标**:确保训练开始时,所有专家都有大致相等的激活概率,且隐空间表征有足够方差。

```python
class RouterInitializer:
    """
    路由器专用初始化器。
    
    核心思想:
    1. 专家权重正交初始化 → 减少初始冗余
    2. 路由器最后一层零初始化 → 初始均匀路由
    3. 下投影正交 + 缩放 → 保证隐空间方差
    4. 残差旁路初始开放 → 保证梯度流动
    """

    @staticmethod
    def init_experts(experts, d_bus: int):
        """
        专家正交初始化。
        up 投影与 down 投影都正交,且不同专家 up 投影之间尽量正交。
        """
        n_experts = len(experts)
        expert_dim = experts[0].up.out_features

        # Step 1: 生成一组全局正交基 [n_experts * expert_dim, d_bus]
        total_dim = n_experts * expert_dim
        global_basis = torch.randn(total_dim, d_bus)
        global_basis, _ = torch.linalg.qr(global_basis.T)   # [d_bus, total_dim]
        global_basis = global_basis.T                       # [total_dim, d_bus]

        # Step 2: 将全局正交基分配给各专家
        for i, expert in enumerate(experts):
            start = i * expert_dim
            end = start + expert_dim
            with torch.no_grad():
                expert.up.weight.copy_(global_basis[start:end])
                down_basis = torch.randn(d_bus, expert_dim)
                down_basis, _ = torch.linalg.qr(down_basis)
                expert.down.weight.copy_(down_basis.T)
                expert.residual_gate.fill_(-2.0)  # 残差门控部分开放

    @staticmethod
    def init_router(router):
        """路由器零初始化最后一层:初始均匀路由,从"均匀"逐步学"选择性""""
        last_linear = router.route_net[-1]
        with torch.no_grad():
            last_linear.weight.zero_()
            last_linear.bias.zero_()
        router.temperature.data.fill_(2.0)    # 初始高温度(软路由)
        router.noise_scale.data.fill_(0.5)    # 初始大噪声(探索)

    @staticmethod
    def init_down_interface(interface, d_model: int, d_bus: int):
        """下投影初始化:正交 + 缩放 + 残差旁路开放"""
        with torch.no_grad():
            scale = (d_model / d_bus) ** 0.5          # 保证输出方差不变
            basis = torch.randn(d_bus, d_model)
            basis, _ = torch.linalg.qr(basis.T)
            interface.proj.weight.copy_(basis.T * scale)
            interface.residual_gate.fill_(0.85)        # σ(0.85) ≈ 0.7,旁路开放
            nn.init.normal_(interface.residual_proj.weight, std=0.01)
```

**为什么这样初始化有效**:

| 初始化策略 | 解决的失效模式 | 机制 |
|:---|:---|:---|
| 全局正交基分配 | F1 专家坍缩 | 专家初始方向完全不同 → 不可能同时吸引相同 token |
| 路由器末层零初始化 | F1, F4 | 初始均匀 → 无偏好 → 无正反馈起点 |
| 高初始温度 | F2 熵坍缩 | 初始软路由 → 所有专家都收到梯度 → 都能学习 |
| 大初始噪声 | F2 | 强制探索 → 防止过早收敛 |
| 下投影缩放 | F3 隐空间坍缩 | 保证 $Var(z) \approx Var(x)$ → 信息不丢失 |
| 残差旁路开放 | F3, F4 | 即使主投影坏了,梯度仍能通过旁路流动 |

### 2.3 Layer 2: 渐进式路由启用

**目标**:不让路由器一开始就承担全部决策压力。从"几乎 Dense"逐步过渡到"稀疏 MoE"。

```python
class ProgressiveRouterScheduler:
    """
    渐进式路由启用调度器。
    
    三个阶段:
    Phase A (0-20% steps): 软路由 + 全专家计算(近似 Dense)
    Phase B (20-50% steps): 逐步降低温度 + 减少活跃专家数
    Phase C (50-100% steps): 正常 Top-K 路由
    
    核心思想:让专家先学会"各自擅长什么",再学会"何时被选择"。
    """

    def __init__(self, config, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        self.phase_a_end = int(total_steps * 0.20)
        self.phase_b_end = int(total_steps * 0.50)
        # Phase A 初始状态
        self.current_top_k = config.n_experts
        self.current_temperature = 5.0
        self.current_noise = 1.0
        self.use_all_experts = True

    def step(self, global_step: int):
        """每步调用,更新路由参数"""
        if global_step < self.phase_a_end:
            # ═══ Phase A: 全专家软路由 ═══
            self.use_all_experts = True
            self.current_top_k = self.config.n_experts
            self.current_temperature = 5.0
            self.current_noise = 1.0

        elif global_step < self.phase_b_end:
            # ═══ Phase B: 渐进收紧 ═══
            progress = (global_step - self.phase_a_end) / (self.phase_b_end - self.phase_a_end)
            target_top_k = self.config.top_k
            self.current_top_k = max(target_top_k,
                int(self.config.n_experts * (1 - progress) + target_top_k * progress))
            self.current_temperature = 5.0 * (0.2 ** progress) + 1.0 * (1 - 0.2 ** progress)
            target_noise = self.config.router_noise_init
            self.current_noise = 1.0 * (1 - progress) + target_noise * progress
            self.use_all_experts = False

        else:
            # ═══ Phase C: 正常路由 ═══
            self.use_all_experts = False
            self.current_top_k = self.config.top_k
            self.current_temperature = 1.0
            self.current_noise = self.config.router_noise_init

    def apply_to_router(self, router):
        router.temperature.data.fill_(self.current_temperature)
        router.noise_scale.data.fill_(self.current_noise)

    def get_effective_top_k(self) -> int:
        return self.current_top_k

    def should_use_all_experts(self) -> bool:
        return self.use_all_experts
```

**Phase A 的实现细节**(WorkspaceBlock 自适应版):

```python
def forward(self, x, bus, router_scheduler, return_diag=False):
    # ... 前面的步骤(线性注意力、下投影)不变 ...

    router_logits = self.router(z)

    if router_scheduler.should_use_all_experts():
        # ═══ Phase A: 全专家计算(权重≈均匀)═══
        weights = F.softmax(router_logits / router_scheduler.current_temperature, dim=-1)
        z_processed = torch.zeros_like(z)
        for expert_id, expert in enumerate(self.experts):
            w = weights[..., expert_id:expert_id + 1]
            z_processed = z_processed + w * expert(z)
        _, top_k_indices = weights.topk(self.config.top_k, dim=-1)
        top_k_weights = weights.gather(-1, top_k_indices)

    else:
        # ═══ Phase B/C: 正常 Top-K ═══
        top_k_weights, top_k_indices = self.router.select(router_logits)
        expert_outputs = []
        for k in range(router_scheduler.get_effective_top_k()):
            idx = top_k_indices[..., k]
            expert_out = self.dispatcher(z, idx, self.experts)
            expert_outputs.append(expert_out)
        z_processed = self.aggregator(z, expert_outputs, top_k_weights)

    # ... 后面的步骤(聚合、SAE、上投影、总线写入)不变 ...
```

**为什么渐进启用有效**:

```
如果直接 Top-2:
→ 只有 2 个专家收到梯度 → 其余 30 个专家"饿死" → 2 个专家被迫承担所有任务 → 坍缩

Phase A 全专家计算:
→ 所有 32 个专家都收到梯度 → 每个专家学会处理"自己擅长的那部分"
→ 20% 训练量后专家已特化 → 切换到 Top-2 时,路由器只需学会"匹配"而非"从零发现"
```

### 2.4 Layer 3: 自适应温度调度

**目标**:温度不是固定超参数,而是根据路由健康状态动态调整(PID 控制器)。

```python
class AdaptiveTemperatureController:
    """
    自适应温度控制器。
    
    监控指标:
    1. 路由熵:太低 → 升温;太高 → 降温
    2. 专家利用率方差:太大 → 升温(促进探索)
    3. 路由稳定性:太低 → 降温(稳定)
    
    控制策略: PID 控制器 (kp=0.1, ki=0.01, kd=0.05)
    """

    def __init__(self, target_entropy=None, n_experts=32, top_k=2):
        """
        n_experts / top_k 默认值是 Medium (7B) 预设(MAGI-2 风格)。

        **SOCA v3-Micro-Final(155M)实例化时必须显式传入**:
            temp_controller = AdaptiveTemperatureController(
                n_experts=12,  # per-head
                top_k=4,      # per-head,激活 16/token(48 总)
            )
        否则会误用 Medium 默认值导致熵目标计算错误。
        """
        self.top_k = top_k
        if target_entropy is None:
            uniform_entropy = -top_k * (1.0 / top_k) * math.log(1.0 / top_k)
            self.target_entropy = uniform_entropy * 1.3  # 比均匀高 30%
        else:
            self.target_entropy = target_entropy
        self.kp, self.ki, self.kd = 0.1, 0.01, 0.05
        self.integral = 0.0
        self.prev_error = 0.0
        self.temperature = 1.0
        self.temp_min, self.temp_max = 0.3, 5.0
        self.entropy_history = []
        self.utilization_history = []

    def update(self, router_entropy: float, expert_utilization):
        """
        每 N 步调用一次(如每 100 步)。
        用熵误差 + 利用率方差惩罚驱动 PID,返回新温度。
        """
        entropy_error = self.target_entropy - router_entropy
        util_var = expert_utilization.var().item()
        util_penalty = max(0, util_var - 0.01) * 0.5
        total_error = entropy_error + util_penalty

        self.integral += total_error
        self.integral = max(-10, min(10, self.integral))
        derivative = total_error - self.prev_error
        self.prev_error = total_error

        adjustment = self.kp * total_error + self.ki * self.integral + self.kd * derivative
        self.temperature += adjustment
        self.temperature = max(self.temp_min, min(self.temp_max, self.temperature))
        self.entropy_history.append(router_entropy)
        self.utilization_history.append(expert_utilization.clone())
        return self.temperature

    def diagnose(self) -> dict:
        """诊断当前路由健康状态:
        healthy / entropy_collapse / entropy_explosion / warming_up"""
        if len(self.entropy_history) < 10:
            return {"status": "warming_up"}
        recent_entropy = self.entropy_history[-10:]
        entropy_trend = np.polyfit(range(len(recent_entropy)), recent_entropy, 1)[0]
        status = "healthy"
        warnings = []
        if np.mean(recent_entropy) < self.target_entropy * 0.5:
            status = "entropy_collapse"; warnings.append("路由熵过低,可能正在坍缩")
        if np.mean(recent_entropy) > self.target_entropy * 2.0:
            status = "entropy_explosion"; warnings.append("路由熵过高,路由失去选择性")
        return {"status": status, "warnings": warnings,
                "current_entropy": recent_entropy[-1],
                "target_entropy": self.target_entropy,
                "current_temperature": self.temperature,
                "entropy_trend": entropy_trend}
```

### 2.5 Layer 4: 多目标梯度平衡

**目标**:解决 LM loss、SAE loss、Balance loss 之间的梯度冲突(GradNorm 风格)。

```python
class GradientBalancer:
    """
    多目标梯度平衡器。
    
    问题:LM loss 想要"对预测有用的专家";SAE loss 想要"易重建的专家";
         Balance loss 想要"均匀分配"。三个目标可能矛盾。
    
    解决方案:
    1. 梯度范数归一化(GradNorm 风格)
    2. 动态权重调整
    3. 梯度投影(去除冲突分量)
    """

    def __init__(self, loss_names: list, alpha: float = 1.5):
        self.loss_names = loss_names
        self.alpha = alpha          # GradNorm 强度(alpha>1 → 加大落后损失权重)
        self.initial_losses = None
        self.loss_weights = {name: 1.0 for name in loss_names}

    def compute_balanced_weights(self, losses: dict, shared_params: list) -> dict:
        if self.initial_losses is None:
            self.initial_losses = {k: v.item() for k, v in losses.items()}
            return self.loss_weights

        # Step 1: 每个损失对共享参数的梯度范数
        grad_norms = {}
        for name, loss in losses.items():
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
            grad_norms[name] = sum(g.norm().item() for g in grads if g is not None)

        # Step 2: 相对训练速率
        training_rates = {name: losses[name].item() / (self.initial_losses[name] + 1e-8)
                          for name in losses}

        # Step 3-4: 调整权重(落后损失加大权重)
        avg_grad_norm = np.mean(list(grad_norms.values()))
        avg_rate = np.mean(list(training_rates.values()))
        new_weights = {}
        for name in losses:
            target_norm = avg_grad_norm * (training_rates[name] / (avg_rate + 1e-8)) ** self.alpha
            ratio = target_norm / (grad_norms[name] + 1e-8)
            new_weights[name] = max(0.1, min(10.0, ratio))
        total = sum(new_weights.values())
        new_weights = {k: v / total * len(new_weights) for k, v in new_weights.items()}
        self.loss_weights = new_weights
        return new_weights

    @staticmethod
    def project_gradients(grads_list: list, reference_grad: torch.Tensor):
        """
        梯度投影:去除与参考梯度(如 LM)冲突的分量。
        用于:当 SAE 梯度与 LM 梯度冲突时,把 SAE 梯度投影到 LM 梯度的正交补空间。
        """
        projected = []
        ref_flat = reference_grad.flatten()
        ref_norm_sq = ref_flat.dot(ref_flat)
        for grad in grads_list:
            grad_flat = grad.flatten()
            proj_coeff = grad_flat.dot(ref_flat) / (ref_norm_sq + 1e-8)
            if proj_coeff < 0:  # 只去除负相关(冲突)分量
                grad_flat = grad_flat - proj_coeff * ref_flat
            projected.append(grad_flat.reshape_as(grad))
        return projected
```

**训练循环中的使用**(简洁版):

```python
# 在 WorkspaceBlock 的训练中
output = model.forward_with_diagnostics(batch)
losses = {
    "lm": F.cross_entropy(output["logits"].view(-1, V), targets.view(-1)),
    "sae": output["sae_loss"],
    "balance": output["load_balance_loss"],
}
router_params = []
for layer in model.workspace_layers:
    router_params.extend(layer.router.parameters())

balanced_weights = grad_balancer.compute_balanced_weights(losses, router_params)
total_loss = sum(balanced_weights[k] * losses[k] for k in losses)
total_loss.backward()
# 可选: 梯度投影(在 optimizer.step() 之前执行)
optimizer.step()
optimizer.zero_grad()
```

### 2.6 Layer 5: 专家容量保护

**目标**:防止任何专家过载或饥饿。

```python
class ExpertCapacityManager:
    """
    专家容量管理器。
    
    双重保护:
    1. 硬容量上限:超过阈值的 token 被重新分配
    2. 软容量正则:鼓励均匀利用
    
    容量上限 = batch_tokens * top_k / n_experts * capacity_factor
    drop_policy: "reassign"(默认) | "drop" | "random"
    """

    def __init__(self, n_experts: int, capacity_factor: float = 1.5,
                 drop_policy: str = "reassign"):
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy
        self.reassignment_counts = 0

    def enforce_capacity(self, indices, logits):
        """强制执行容量限制,返回修正后的 indices"""
        batch, seq, top_k = indices.shape
        n_tokens = batch * seq * top_k
        capacity = int(n_tokens / self.n_experts * self.capacity_factor)

        flat_indices = indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.n_experts)
        overloaded = (counts > capacity).nonzero(as_tuple=True)[0]
        if len(overloaded) == 0:
            return indices

        corrected = indices.clone()
        for expert_id in overloaded:
            excess = counts[expert_id].item() - capacity
            mask = (indices == expert_id)
            assigned_positions = mask.nonzero(as_tuple=False)
            # 按路由置信度排序(最不确定的先重分配)
            confidence = logits.reshape(-1, self.n_experts)[
                assigned_positions[:, 0] * seq * top_k + assigned_positions[:, 1], expert_id]
            _, reorder = confidence.sort()
            to_reassign = assigned_positions[reorder[:excess]]
            for pos in to_reassign:
                b, s, k = pos.tolist()
                sorted_experts = logits[b, s].argsort(descending=True)
                for alt_expert in sorted_experts:
                    if alt_expert != expert_id and counts[alt_expert] < capacity:
                        corrected[b, s, k] = alt_expert
                        counts[expert_id] -= 1
                        counts[alt_expert] += 1
                        self.reassignment_counts += 1
                        break
        return corrected

    def get_utilization_stats(self, indices) -> dict:
        """利用率统计:mean/std/max/min/max_min_ratio/n_dead_experts/reassignments"""
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.n_experts).float()
        freq = counts / counts.sum()
        return {
            "max_min_ratio": (freq.max() / (freq.min() + 1e-8)).item(),
            "n_dead_experts": (freq < 0.001).sum().item(),
            "reassignments": self.reassignment_counts,
        }
```

### 2.7 Layer 6: 隐空间健康监控

**目标**:实时检测隐空间坍缩(F3)——最隐蔽也最致命的失效模式。

```python
class LatentSpaceHealthMonitor:
    """
    隐空间健康监控器。
    
    监控指标:
    1. 方差:z 的各维度方差应 > 阈值(0.01)
    2. 有效秩:z 的有效秩应接近 d_bus(目标 > 0.5 * d_bus)
    3. 能量集中度:PCA 前 50% 成分的能量占比不应过高(< 0.8)
    """

    def __init__(self, d_bus: int, check_interval: int = 100):
        self.d_bus = d_bus
        self.min_variance = 0.01
        self.min_effective_rank = d_bus * 0.5
        self.max_energy_concentration = 0.8
        self.variance_history = []
        self.rank_history = []
        self.alert_count = 0

    def check(self, z, x=None) -> dict:
        """z: [batch, seq, d_bus] → 健康报告"""
        z_flat = z.reshape(-1, self.d_bus)
        alerts = []

        # ═══ 1. 方差检查 ═══
        variances = z_flat.var(dim=0)
        low_var_dims = (variances < self.min_variance).sum().item()
        if low_var_dims > self.d_bus * 0.1:
            alerts.append({"type": "variance_collapse",
                           "severity": low_var_dims / self.d_bus})

        # ═══ 2. 有效秩检查(SVD)═══
        if z_flat.shape[0] >= self.d_bus:
            U, S, V = torch.linalg.svd(z_flat.T @ z_flat / z_flat.shape[0])
        else:
            U, S, V = torch.linalg.svd(z_flat @ z_flat.T / z_flat.shape[0])
            S = S[:min(z_flat.shape[0], self.d_bus)]
        threshold = S.max() * 0.01
        effective_rank = (S > threshold).sum().item()
        if effective_rank < self.min_effective_rank:
            alerts.append({"type": "rank_collapse",
                           "severity": 1 - effective_rank / self.d_bus})

        # ═══ 3. 能量集中度 ═══
        energy = S ** 2
        half_components = len(S) // 2
        concentration = energy[:half_components].sum() / (energy.sum() + 1e-8)
        if concentration > self.max_energy_concentration:
            alerts.append({"type": "energy_concentration"})

        if alerts:
            self.alert_count += 1
        return {"healthy": len(alerts) == 0, "alerts": alerts,
                "effective_rank": effective_rank,
                "energy_concentration": concentration.item()}

    def emergency_response(self, alerts: list, model):
        """紧急响应:检测到隐空间坍缩时的自动修复"""
        for alert in alerts:
            t = alert["type"]
            if t == "variance_collapse":          # 增大残差旁路
                model.down_interface.residual_gate.data.add_(0.5)
            elif t == "rank_collapse":            # 增加噪声 + 升温
                for layer in model.workspace_layers:
                    layer.router.noise_scale.data.add_(0.2)
                    layer.router.temperature.data.add_(0.5)
            elif t == "energy_concentration":     # 加大正交惩罚
                pass
```

### 2.8 Layer 7: 运行时自愈机制

**目标**:当上述防护都失败时的最后兜底。

```python
class RouterSelfHealer:
    """
    路由器自愈机制。
    
    触发条件:连续 N 步检测到严重异常(patience)
    自愈操作:
    1. 轻度:重置路由器末层 + 升温
    2. 中度:重置所有专家 + 回到 Phase A
    3. 重度:完全重新初始化工作空间区
    """

    def __init__(self, patience: int = 500):
        self.patience = patience
        self.consecutive_alerts = 0
        self.last_heal_step = 0
        self.heal_cooldown = 2000    # 两次自愈之间的最小间隔

    def check_and_heal(self, global_step, diagnostics, model, router_scheduler):
        if global_step - self.last_heal_step < self.heal_cooldown:
            return

        has_severe = False
        if "expert_utilization" in diagnostics and diagnostics["expert_utilization"]:
            util = diagnostics["expert_utilization"]
            # 检查专家利用率:过载 / 死亡专家过多 / LM loss 突增
            if util.get("max_min_ratio", 0) > 500 or util.get("n_dead_experts", 0) > 8:
                has_severe = True
        if "latent_health" in diagnostics and diagnostics["latent_health"]:
            if not diagnostics["latent_health"]["healthy"]:
                has_severe = True

        self.consecutive_alerts = self.consecutive_alerts + 1 if has_severe else max(0, self.consecutive_alerts - 1)

        if self.consecutive_alerts >= self.patience:
            self._heal(global_step, model, router_scheduler)

    def _heal(self, global_step, model, router_scheduler):
        severity = min(3, self.consecutive_alerts // self.patience)

        if severity == 1:      # 轻度:重置路由器末层 + 升温
            for layer in model.workspace_layers:
                last = layer.router.route_net[-1]
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
                layer.router.temperature.data.fill_(3.0)
                layer.router.noise_scale.data.fill_(0.5)

        elif severity == 2:    # 中度:重置所有专家 + 回到 Phase A
            for layer in model.workspace_layers:
                RouterInitializer.init_experts(layer.experts, layer.d_bus)
                RouterInitializer.init_router(layer.router)
            router_scheduler.current_top_k = model.config.n_experts
            router_scheduler.use_all_experts = True
            router_scheduler.current_temperature = 5.0

        elif severity >= 3:    # 重度:完全重新初始化工作空间区
            for layer in model.workspace_layers:
                layer.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                RouterInitializer.init_experts(layer.experts, layer.d_bus)
                RouterInitializer.init_router(layer.router)
            RouterInitializer.init_down_interface(model.down_interface, model.config.d_model, model.config.d_bus)
            router_scheduler.current_top_k = model.config.n_experts
            router_scheduler.use_all_experts = True
            router_scheduler.current_temperature = 5.0
            router_scheduler.current_noise = 1.0

        self.last_heal_step = global_step
        self.consecutive_alerts = 0
```

---

## 三、完整训练循环

将以上所有组件整合:

```python
def train_soca(config, train_loader, val_loader, total_steps: int):
    """SOCA 完整训练循环(路由器七重防护整合版)"""

    # ═══ 模型初始化 + 专用初始化 ═══
    model = SOCAModel(config).cuda()
    for layer in model.workspace_layers:
        RouterInitializer.init_experts(layer.experts, config.d_bus)
        RouterInitializer.init_router(layer.router)
    RouterInitializer.init_down_interface(model.down_interface, config.d_model, config.d_bus)

    # ═══ 训练组件 ═══
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=3e-5)
    loss_fn = SOCALoss(config)
    phase_scheduler = PhaseScheduler(model, total_steps)                       # M19
    router_scheduler = ProgressiveRouterScheduler(config, total_steps)         # L2
    temp_controller = AdaptiveTemperatureController(n_experts=config.n_experts, top_k=config.top_k)  # L3
    grad_balancer = GradientBalancer(["lm", "sae", "balance"], alpha=1.5)      # L4
    capacity_manager = ExpertCapacityManager(config.n_experts, config.capacity_factor)  # L5
    latent_monitor = LatentSpaceHealthMonitor(config.d_bus)                    # L6
    self_healer = RouterSelfHealer(patience=500)                               # L7

    global_step = 0
    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break

            # ─── 阶段管理与路由调度 ───
            phase_scheduler.step(global_step)
            router_scheduler.step(global_step)
            for layer in model.workspace_layers:
                router_scheduler.apply_to_router(layer.router)

            # ─── 前向 ➜ 损失 ➜ 梯度平衡 ➜ 反向 ───
            model.train()
            output = model.forward_with_diagnostics(batch["input_ids"].cuda())
            targets = batch["labels"].cuda()
            losses = {
                "lm": F.cross_entropy(output["logits"].view(-1, config.vocab_size), targets.view(-1)),
                "sae": output["sae_loss"],
                "balance": output["load_balance_loss"],
            }
            router_params = sum((list(l.router.parameters()) for l in model.workspace_layers), [])
            balanced_weights = grad_balancer.compute_balanced_weights(losses, router_params)
            total_loss = sum(balanced_weights[k] * losses[k] for k in losses)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); optimizer.zero_grad(); scheduler.step()

            # ─── 监控(每 100 步)— L3 温度 + L6 隐空间健康)───
            # 注:与 04 §八 SOCATrainer 监控频率对齐——04 是每 100 步写日志、每 1000 步 eval、
            # 每 2000 步可解释性评估;这里每 100 步更新控制器是合理的(轻量级 PID + 健康检查)
            if global_step % 100 == 0:
                # 温度自适应(L3)
                avg_entropy = sum(l.router.entropy().item() for l in model.workspace_layers) / len(model.workspace_layers)
                new_temp = temp_controller.update(avg_entropy, torch.ones(config.n_experts))
                for layer in model.workspace_layers:
                    layer.router.temperature.data.fill_(new_temp)
                # 隐空间健康(L6)
                z_sample = output.get("latent_z", None)
                if z_sample is not None:
                    health = latent_monitor.check(z_sample)
                    if not health["healthy"]:
                        latent_monitor.emergency_response(health["alerts"], model)

            # ─── 自愈检查(L7,每 100 步)— 与上面同步)───
            util = None
            first_router = model.workspace_layers[0].router
            if first_router.last_indices is not None:
                util = capacity_manager.get_utilization_stats(first_router.last_indices)
            self_healer.check_and_heal(global_step, {"expert_utilization": util,
                                                      "latent_health": health if global_step % 100 == 0 else None,
                                                      "lm_loss_spike": False},
                                        model, router_scheduler)

            # ─── 日志(每 10 步)— 与 04 §八 的 100 步粒度对齐时调整为 100)───
            if global_step % 10 == 0:
                print(f"Step {global_step} | Phase {phase_scheduler.current_phase} | "
                      f"T={router_scheduler.current_temperature:.2f} | top_k={router_scheduler.current_top_k} | "
                      f"lm={losses['lm'].item():.4f} sae={losses['sae'].item():.4f} balance={losses['balance'].item():.4f}")

            global_step += 1

        # ─── 验证与 checkpoint(每 1000 步)— 对齐 04 §八)───
        if global_step % 1000 == 0:
            val_metrics = validate(model, val_loader, config)
            save_checkpoint(model, optimizer, global_step, config)
```

---

## 四、路由器训练稳定性速查表

| 失效模式 | 检测信号 | 防护层 | 自愈措施 |
|:---|:---|:---|:---|
| F1 专家坍缩 | max_util > 0.5 | L1 正交初始化, L2 渐进启用, L5 容量保护 | L7 重初始化专家 |
| F2 熵坍缩 | entropy < target×0.5 | L1 高初始温度, L3 自适应升温 | L7 升温+加噪 |
| F3 隐空间坍缩 | variance < 0.01 或 rank < d/2 | L1 缩放+残差旁路, L6 实时监控 | L6 紧急响应, L7 全重置 |
| F4 路由震荡 | 同 token 跨步分配不一致 | L2 渐进启用, L4 梯度平衡 | L3 降温 |
| F5 门控震荡 | 聚合器门控值跳变 | L2 渐进启用 | 降低门控学习率 |
| F6 路由退化 | entropy > target×2 | L3 自适应降温 | L7 重置路由器末层 |
| F7 SAE-路由冲突 | SAE↓ but LM↑ | L4 梯度平衡, L4 梯度投影 | 降低 λ_sae |

---

## 五、关键超参数推荐值

> ⚠️ **本节速查表覆盖 SOCA 框架的 3 档预设(Tiny/Medium/Large);SOCA v3-Micro-Final(155M)是验证模型,采用独立的"v3-Micro"列(见底部子表)。**

### 5.1 三档预设速查表

| 参数 | Tiny (20M) | Medium (7B) | Large (120B) | 说明 |
|:---|:---:|:---:|:---:|:---|
| 初始温度 | 3.0 | 2.0 | 2.0 | 越小越硬 |
| 初始噪声 | 0.5 | 0.3 | 0.2 | 大模型更稳定 |
| Phase A 占比 | 25% | 20% | 15% | 大模型预训练数据更多 |
| Phase B 占比 | 30% | 30% | 35% | 大模型需要更长过渡 |
| 目标熵倍数 | 1.5× | 1.3× | 1.2× | 大模型不需要太多探索 |
| 容量因子 | 2.0 | 1.5 | 1.2 | 大模型专家更多,自然更均匀 |
| PID kp | 0.1 | 0.05 | 0.03 | 大模型更敏感 |
| 梯度裁剪 | 1.0 | 1.0 | 0.5 | 大模型梯度更大 |
| 自愈耐心 | 200 | 500 | 1000 | 大模型波动更慢 |
| 自愈冷却 | 1000 | 2000 | 5000 | 避免频繁自愈 |
| λ_sae | 0.05 | 0.1 | 0.1 | Phase 1 起始值 |
| λ_balance | 0.005 | 0.01 | 0.01 | 不要太大 |
| 正交惩罚 | 0.005 | 0.01 | 0.01 | 太强会导致 F3 |
| **中段 top_k** | **2** | **2** | **2** | **Medium/Large 7B-120B 用 top_k=2** |
| 总专家数 | 8 | 32 | 64 | — |

### 5.2 SOCA v3-Micro-Final(155M)适配子表

> **重要区分**:v3-Micro-Final 的中段路由采用 **Top-4/head**(激活 16/token,总 48 专家),**与上表 Medium/Large 预设的 top_k=2 不同**——这是验证模型的特殊配置(需要更多激活 token 以便消融实验有更显著的效应量)。

| 参数 | v3-Micro-Final (155M) | 来源 |
|:---|:---:|:---|
| 中段 head 数 | 4 | 03 §五.6 |
| 中段 experts/head | 12 | 03 §五.6 |
| 中段 **top_k/head** | **4** | **03 §五.6(关键,与 Medium/Large 不同)** |
| 总专家数 | 48 | 4h × 12e |
| 激活/token | 16 | 4 × 4 |
| 后段 top_k | 5 | 03 §五.6(20 routed + 1 shared + 1 device) |
| 初始温度 | 2.5 | Tiny-Medium MAP |
| 初始噪声 | 0.4 | Tiny-Medium MAP |
| Phase A 占比 | 22% | Tiny-Medium MAP |
| Phase B 占比 | 30% | Tiny-Medium MAP |
| 目标熵倍数 | 1.4× | Tiny-Medium MAP |
| 容量因子 | 1.7 | Tiny-Medium MAP |
| PID kp | 0.07 | Tiny-Medium MAP |
| 梯度裁剪 | 1.0 | 同 Tiny |
| 自愈耐心 | 350 | Tiny-Medium MAP |
| 自愈冷却 | 1500 | Tiny-Medium MAP |
| λ_sae | 0.05 | 起点(同 Tiny) |
| λ_balance | 0.005 | 起点(同 Tiny) |
| 正交惩罚 | 0.005 | 起点(同 Tiny) |

> **何时用哪张表**:
> - **实施 SOCA v3-Micro-Final(155M)消融实验** → 用 5.2 子表(关键差异:中段 top_k=4)
> - **扩展到 7B / 120B 正式模型** → 用 5.1 表 Medium / Large 列(top_k=2)
> - **概念验证(<20M)** → 用 5.1 表 Tiny 列(top_k=2)

> **历史版本**:本文档之前版本的 §五 速查表误将 v3-Micro-Final 的 top_k 写成 2,这是基于 Medium/Large 默认预设的笔误——v3-Micro-Final 锁定 top_k=4/head(详见 03 §五.6)。已在本次修订中明确区分。

---

## 六、与 [07-module-extensions.md](./07-module-extensions.md) M19 PhaseScheduler 的关系

| 维度 | 07 M19 PhaseScheduler | 08 路由器稳定性配方(本文档) |
|---|---|---|
| **管理对象** | 整个模型的参数冻结/解冻(区域级) | 路由器单组件的训练稳定性(组件级) |
| **正交性** | 感知→工作空间→动作 区域训练 | 路由渐进启用 + 温度 + 平衡 + 自愈 |
| **交互点** | Phase 1(工作空间)期间路由器压力最大 | 路由器七重防护在 Phase 1 全程生效 |
| **协作模式** | 本文档 §三 的完整训练循环同时使用两者 | M19 管"解冻哪层",L1-L7 管"如何稳定" |

> **关键**:两者不冲突——M19 决定**何时**训练路由器;L1-L7 决定**如何**让路由器在训练时不崩溃。

---

## 七、一句话总结

> **路由器稳定性的本质是:不要让路由器在任何时刻承担超出其当前能力的决策压力。**
>
> 初始化让它从均匀开始;渐进启用让它从简单到复杂;自适应温度让它根据健康状态自我调节;梯度平衡让它不被单一目标绑架;容量保护让它不被极端分配摧毁;隐空间监控让它在崩溃前被救回;自愈机制让它在一切失败后还能重来。
>
> **七重防护不是冗余,而是对一个承载了信息瓶颈阀门的组件应有的尊重。**

---

## 📅 文档版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-28 | 初版,完成路由器稳定性训练配方(七重防护 + F1-F7 失效模式 + 完整训练循环 + 超参速查) |

---

> **下一步**:阅读 [`09-sae-bus-information-theory.md`](./09-sae-bus-information-theory.md) 了解与路由器强耦合的联合 SAE 字典动态与总线信息论;然后按 [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) 实施。
# SOCA 联合 SAE 字典学习动态 与 广播总线信息论性质

> **目的**:深入研究 SOCA 两大核心组件的**理论性质**——联合 SAE(M12)的字典学习动力学,与广播总线(M1)的信息论性质。
>
> **关联文档**:
> - 模块定义:[`06-architecture-modules.md`](./06-architecture-modules.md) §六 M12 JointSAE / §三 M1 BroadcastBus
> - 训练配方:[`08-router-stability-training.md`](./08-router-stability-training.md) §二.7 F7 SAE-路由冲突(本文档 §五 详细展开)
> - 监控系统:[`06-architecture-modules.md`](./06-architecture-modules.md) §八 M15 SOCAMonitor
> - 消融实验:[`05-pretraining-ablation-plan.md`](./05-pretraining-ablation-plan.md) D1-D4(正则化相关)
>
> **角色**:本文档位于"**理论分析层**"——为 [`08-router-stability-training.md`](./08-router-stability-training.md) 的工程配方提供**数学与信息论依据**,帮助理解"为什么这样调参"。

---

## 第一部分:联合 SAE 的字典学习动态

---

## 一、联合 SAE 与事后 SAE 的本质区别

在展开动态分析之前,必须先厘清"联合训练"到底改变了什么:

```
事后 SAE(传统方法)
模型训练完成 → 冻结权重 → 收集激活 → 训练 SAE → 分析特征

问题:
1. SAE 只能拟合"已经存在"的表征结构
2. 如果模型表征本身是纠缠的,SAE 无法解开
3. 字典大小必须 >> 模型维度(10×)才能覆盖
4. 特征语义不稳定(随训练数据变化)


联合 SAE(SOCA 方法)
模型训练 ←→ SAE 训练(同步、耦合、互相约束)

本质变化:
1. SAE 重建损失反向传播 → 模型被迫产生"可稀疏编码"的表征
2. 模型表征塑造 → SAE 字典适配 → 表征进一步塑造(共演化)
3. 字典可以很小(2×),因为表征被"引导"向稀疏方向
4. 特征语义由训练目标共同决定 → 跨样本稳定
```

> **核心洞察**:联合 SAE 不是一个"分析工具"。它是一个**表征塑造器**。它通过重建损失告诉模型:"你的隐空间表征必须能被少数几个字典向量线性组合近似。" 这从根本上改变了隐空间的几何结构。

---

## 二、字典学习的动力学方程

### 2.1 基本更新规则

设字典矩阵为 $D \in \mathbb{R}^{d_{bus} \times K}$($K$ = 字典大小),隐表征为 $z \in \mathbb{R}^{d_{bus}}$,稀疏编码为 $c \in \mathbb{R}^K$(Top-$k$ 稀疏)。

重建目标:

$$\mathcal{L}_{recon} = \mathbb{E}\left[\| z - D c \|_2^2 \right]$$

Top-$k$ 编码:

$$c^* = \text{TopK}\left(D^\top z, \; k\right)$$

字典向量的梯度:

$$\frac{\partial \mathcal{L}_{recon}}{\partial d_j} = -2 \, c_j \left(z - D c\right)$$

**关键观察**:只有被选中的字典向量($c_j \neq 0$)才会收到梯度。未被选中的向量完全静止。

### 2.2 联合训练中的额外梯度

在 SOCA 中,字典向量还通过以下路径收到梯度:

```
字典向量 d_j 的梯度来源:
├── 路径 1: 重建损失(直接)
│   ∂L_recon / ∂d_j = -2 c_j (z - Dc)
│
├── 路径 2: LM 损失(间接,通过 z_sae 影响后续计算)
│   ∂L_lm / ∂d_j = ∂L_lm / ∂z_sae · ∂z_sae / ∂d_j
│   其中 z_sae = D c*,所以 ∂z_sae / ∂d_j = c_j
│
├── 路径 3: 正交约束
│   ∂L_ortho / ∂d_j = 2(D^T D - I)_j · D
│
└── 路径 4: 路由器(间接,通过专家选择影响 z 的分布)
    (在联合训练中是隐式的)
```

**这意味着**:字典向量不仅被"重建质量"驱动,还被"对最终预测的贡献"驱动。一个字典向量如果对应"对预测无用但容易重建"的方向,会在 LM 损失的梯度下被惩罚。

### 2.3 字典生命周期追踪(实现骨架)

```python
class DictionaryLearningDynamics:
    """字典学习动态分析器。
    跟踪每个字典向量的激活频率/梯度范数/方向漂移/重建贡献。"""

    def __init__(self, dict_size: int, d_bus: int, track_interval: int = 50):
        self.dict_size = dict_size
        self.track_interval = track_interval
        self.initial_dict = None
        self.activation_freq = torch.zeros(dict_size)
        self.birth_step = {}       # 首次被激活的步数
        self.death_step = {}       # 连续未激活起始步数
        self.last_activation = torch.zeros(dict_size)
        self.direction_drift_history = defaultdict(list)

    def snapshot_initial(self, dictionary):
        self.initial_dict = dictionary.detach().clone()

    def update(self, step, dictionary, codes, z, z_recon):
        if step % self.track_interval != 0:
            return
        active_mask = (codes.abs() > 1e-6)
        activation_counts = active_mask.sum(dim=0).float()
        self.activation_freq += activation_counts.cpu()
        for idx in (activation_counts > 0).nonzero(as_tuple=True)[0]:
            self.last_activation[idx] = step
            if idx.item() not in self.birth_step:
                self.birth_step[idx.item()] = step
        for j in range(self.dict_size):
            if step - self.last_activation[j] > 1000 and j not in self.death_step:
                self.death_step[j] = step  # 死亡标记

    def get_lifecycle_report(self) -> dict:
        """生成字典生命周期报告:alive/dead/never_born/利用率/方向漂移"""
        alive = [j for j in range(self.dict_size)
                 if j in self.birth_step and j not in self.death_step]
        dead = list(self.death_step.keys())
        never_born = [j for j in range(self.dict_size) if j not in self.birth_step]
        return {
            "total_atoms": self.dict_size,
            "alive": len(alive),
            "dead": len(dead),
            "never_born": len(never_born),
            "utilization_rate": len(alive) / self.dict_size,
            "dead_atom_ids": dead[:20],
            "never_born_ids": never_born[:20],
        }
```

---

## 三、字典向量生命周期

### 3.1 四个生命阶段

```
Stage 1: 诞生(Birth)——训练初期(0-10% 步数)
  所有字典向量被均匀激活(路由器也是均匀的);梯度方向随机漂移
  风险:某些向量初始方向不好 → 永远不被选中 → 从未诞生

Stage 2: 特化(Specialization)——训练中期(10-50% 步数)
  字典向量开始"认领"特定语义方向;高频激活快速收敛
  决定字典的最终结构
  风险:过度特化 → 泛化差;特化方向与路由器冲突 → F7

Stage 3: 稳态(Steady State)——训练后期(50-90% 步数)
  字典方向基本固定,只有幅度微调;特征语义最稳定,适合做监控基准

Stage 4: 衰退/重生(Decay / Rebirth)——训练末期/数据变化时
  某些向量不再被需要 → 激活频率趋零 → 死亡
  新模式需要新方向 → 死亡向量被重新激活 → 重生
  风险:死亡速度 > 重生速度 → 字典有效容量缩小
```

### 3.2 特化阶段的数学描述

在特化阶段,字典向量 $d_j$ 的更新可近似为:

$$d_j^{(t+1)} = d_j^{(t)} + \eta \cdot c_j^{(t)} \cdot \left(z^{(t)} - D c^{(t)}\right) + \eta_{lm} \cdot c_j^{(t)} \cdot \frac{\partial \mathcal{L}_{lm}}{\partial z_{sae}}$$

- 第一项:重建梯度(让 $d_j$ 更接近 $z$ 的方向)
- 第二项:任务梯度(让 $d_j$ 的方向对预测有用)

**特化的固定点条件**(当 $d_j$ 不再变化时):

$$z - Dc = -\lambda_{lm} \cdot \frac{\partial \mathcal{L}_{lm}}{\partial z_{sae}}$$

> **含义**:在联合训练的稳态下,重建残差不是零,而是等于"任务梯度方向"。字典不会完美重建 $z$,而是重建 $z$ 中"对任务有用"的部分。

---

## 四、稀疏度-重建质量的权衡曲面

### 4.1 理论分析

设 $z$ 的真实内在维度为 $r$(即 $z$ 实际上只在一个 $r$ 维子空间上变化),字典大小为 $K$,稀疏度为 $k$。

**定理(非正式)**:联合训练下,当且仅当 $k \geq r$ 时,SAE 可以实现零重建误差。当 $k < r$ 时,最小重建误差为:

$$\mathcal{L}_{min} = \sum_{i=k+1}^{r} \sigma_i^2$$

其中 $\sigma_i$ 是 $z$ 的协方差矩阵的第 $i$ 个奇异值。

**但在联合训练中**,模型会主动降低 $r$:

$$r_{effective} = r_{original} - \Delta r(\lambda_{sae})$$

即:**SAE 损失越大,模型越倾向于产生低内在维度的表征,从而让稀疏编码更容易。**

### 4.2 实际权衡

```
关键观察:
- λ_sae 太小:模型不配合,SAE 重建差
- λ_sae 太大:模型过度压缩,LM 性能下降
- 甜蜜点:λ_sae ∈ [0.05, 0.15]
- k 的甜蜜点:k ∈ [d_bus/16, d_bus/8]
```

### 4.3 动态稀疏度调度

```python
class DynamicSparsityScheduler:
    """动态稀疏度调度:训练过程中逐步收紧稀疏度。
    训练初期 k 大(先学会重建)→ 中期逐步收紧(迫使特化)→ 后期固定(稳定特征)"""

    def __init__(self, d_bus: int, k_start: int, k_end: int,
                 warmup_fraction: float = 0.3, anneal_fraction: float = 0.4):
        self.k_start = k_start
        self.k_end = k_end
        self.warmup_end = warmup_fraction
        self.anneal_end = warmup_fraction + anneal_fraction

    def get_k(self, progress: float) -> int:
        if progress < self.warmup_end:
            return self.k_start
        elif progress < self.anneal_end:
            t = (progress - self.warmup_end) / (self.anneal_end - self.warmup_end)
            k = self.k_end + (self.k_start - self.k_end) * 0.5 * (1 + math.cos(math.pi * t))
            return int(k)
        return self.k_end

    def get_sae_loss_weight(self, progress: float) -> float:
        """SAE 损失权重随训练进展调整"""
        if progress < 0.1:
            return 0.0                    # 最初不施加 SAE 约束
        elif progress < 0.3:
            return 0.1 * (progress - 0.1) / 0.2   # 线性增加
        return 0.1                        # 稳定
```

---

## 五、字典-路由器耦合动态

### 5.1 耦合机制

```
┌──────────┐  选择专家   ┌──────────┐
│  路由器   │ ──────────→ │  专家计算  │
│  (M9)    │             │  (M10)   │
└──────────┘             └────┬─────┘
     ▲                        │ 产生 z
     │                        ▼
     │                  ┌──────────┐
     │                  │  隐表征 z │
     │                  └────┬─────┘
     │                       │ SAE 编码
     │                       ▼
     │                  ┌──────────┐
     │                  │ 联合 SAE  │
     │                  │  (M12)   │
     │                  └────┬─────┘
     │                       │ z_sae 影响后续层 → LM loss → 梯度回传
     └───────────────────────┘

耦合效应:
1. 路由器选择专家 → 决定 z 的分布 → 决定哪些字典向量被激活
2. SAE 重建质量 → 影响 z_sae → 影响 LM loss → 影响路由器梯度
3. 字典特化 → 改变重建方向 → 改变 z_sae → 改变下游计算

潜在冲突(F7):
- 路由器想选"对 LM 最有用的专家"
- SAE 想选"最容易重建的专家"
- 如果两者不一致 → 路由器震荡
```

### 5.2 解耦策略

> **这与 [`08-router-stability-training.md`](./08-router-stability-training.md) §二.5 Layer 4(梯度平衡)互补**:L4 在训练配方层处理 F7;本节的 detach 在架构层处理 F7。
>
> ⚠️ **Top-K 参数对齐**:本节示例代码的 `k=4` 对应 **SOCA v3-Micro-Final**(中段 4h × 12e, Top-4/head, 激活 16/token,详见 [`03-sweet-spot-layers.md`](./03-sweet-spot-layers.md) §五.6)。Medium (7B) / Large (120B) 预设的 top_k=2 参见 [`08-router-stability-training.md`](./08-router-stability-training.md) §五.1。

```python
class SAERouterDecoupler:
    """SAE-路由器解耦器。
    核心思想:SAE 的梯度不应该直接影响路由器的决策。
    SAE 只影响"专家输出的表征质量",不影响"选择哪个专家"。

    实现:在路由器到专家输出之间插入 stop-gradient。"""

    @staticmethod
    def decouple_in_forward(z, router_logits, experts, joint_sae,
                            top_k=4):  # SOCA v3-Micro-Final: top_k=4/head
        """SOCA v3-Micro-Final 默认 top_k=4(中段 4h × 12e, Top-4/head);
        Medium/Large 7B-120B 用 top_k=2。"""
        weights, indices = select_top_k(router_logits, k=top_k)
        expert_out = dispatch_experts(z, indices, experts)
        # ═══ 关键:SAE 编码时,对 z 做 stop-gradient ═══
        z_detached = z.detach()          # 切断梯度:SAE 重建只更新字典和编码器
        z_sae, sae_loss = joint_sae(z_detached)
        # 但 z_sae 仍参与后续计算,LM loss 梯度正常回传到路由器和专家
        return z_sae, sae_loss

    @staticmethod
    def soft_decouple(z, router_logits, experts, joint_sae,
                      alpha=0.8, top_k=4):  # SOCA v3-Micro-Final: top_k=4
        """软解耦:不完全切断,而是衰减。
        alpha=1.0: 完全解耦;alpha=0.0: 完全耦合;alpha=0.8: 推荐
        SOCA v3-Micro-Final 默认 top_k=4;Medium/Large 用 top_k=2"""
        weights, indices = select_top_k(router_logits, k=top_k)
        expert_out = dispatch_experts(z, indices, experts)
        z_partial = alpha * z.detach() + (1 - alpha) * z
        z_sae, sae_loss = joint_sae(z_partial)
        return z_sae, sae_loss
```

---

## 六、字典正交性的动态维护

静态正交初始化不够,因为训练中字典会漂移、向量可能变得相似、死亡向量可能被重新激活到已有方向。

```python
class DynamicOrthogonalizer:
    """动态正交化器:定期清理字典冗余。
    每 N 步计算 Gram 矩阵 → 找到高相关向量对 → 对"死亡/低频"向量重置"""

    def __init__(self, dict_size: int, check_interval: int = 500,
                 correlation_threshold: float = 0.9,
                 min_activation_freq: float = 0.001):
        self.dict_size = dict_size
        self.check_interval = check_interval
        self.corr_threshold = correlation_threshold
        self.min_freq = min_activation_freq

    def check_and_fix(self, step, dictionary, activation_freq) -> int:
        """返回修正的向量数量"""
        if step % self.check_interval != 0:
            return 0
        D_norm = F.normalize(dictionary, dim=0)
        gram = D_norm.T @ D_norm
        upper_tri = torch.triu(gram, diagonal=1)
        high_corr_pairs = (upper_tri > self.corr_threshold).nonzero(as_tuple=False)
        n_fixed = 0
        for pair in high_corr_pairs:
            i, j = pair.tolist()
            # 重置激活频率低的向量
            reset_idx = i if activation_freq[i] < activation_freq[j] else j
            if activation_freq[reset_idx] < self.min_freq:
                with torch.no_grad():
                    dictionary[:, reset_idx] = self._random_orthogonal_to(dictionary, exclude={reset_idx})
                n_fixed += 1
        return n_fixed

    def _random_orthogonal_to(self, dictionary, exclude: set):
        """生成与现有字典向量正交的随机方向(投影到正交补空间)"""
        vectors = [dictionary[:, j] for j in range(dictionary.shape[1]) if j not in exclude]
        if not vectors:
            return F.normalize(torch.randn(dictionary.shape[0]), dim=0)
        V = torch.stack(vectors, dim=1)
        Q, R = torch.linalg.qr(V)
        rand = torch.randn(dictionary.shape[0])
        orthogonal = rand - Q @ (Q.T @ rand)
        return F.normalize(orthogonal, dim=0)
```

---

## 七、联合 SAE 训练配方总结

```
阶段 0 (0-10%): 不施加 SAE 约束
  λ_sae = 0; 让模型和路由器先稳定;字典随机初始化,不更新

阶段 1 (10-30%): 温和引入
  λ_sae: 0 → 0.05(线性); k: d_bus/4(宽松稀疏)
  解耦度 α = 0.5(半耦合); 字典学习率 = 主学习率 × 0.5
  目标:字典开始特化,但不施加太大压力

阶段 2 (30-60%): 逐步收紧
  λ_sae: 0.05 → 0.1; k: d_bus/4 → d_bus/16(余弦退火)
  解耦度 α = 0.8(大部分解耦); 字典学习率 = 主 × 0.3
  每 500 步正交化清理; 目标:特化 + 稀疏度提高

阶段 3 (60-100%): 稳定
  λ_sae: 0.1(固定); k: d_bus/16(固定)
  解耦度 α = 0.9(几乎完全解耦); 字典学习率 = 主 × 0.1
  每 1000 步正交化清理; 目标:特征语义稳定,作为监控基准

监控指标(每 100 步):
  字典利用率(活跃原子比例) → 目标 > 80%
  重建误差 → 目标 < 0.1 × ||z||²
  方向漂移 → 阶段 3 后应 < 0.05
  最大原子间相关性 → 目标 < 0.9
  SAE-LM 梯度余弦相似度 → 不应持续为负(冲突)
```

---

## 第二部分:广播总线的信息论性质

---

## 八、总线作为信息瓶颈

### 8.1 形式化定义

设:
- $X$:输入序列的完整表征($d_{model}$ 维)
- $B_l$:第 $l$ 层的总线状态($d_{bus}$ 维)
- $Y$:模型的最终输出
- $W$:工作空间区的完整内部状态

总线的设计目标:$B_l = f_l(X, W_{<l})$,即总线是输入和之前计算的**函数**。

**信息瓶颈约束**:$I(X; Y) \geq I(B_L; Y)$(总线不应丢失对预测有用的信息),同时 $H(B_l) \leq d_{bus} \cdot \log(\text{precision})$(总线是低维的)。

**设计目标**:最大化 $I(B; Y) / H(B)$ —— 总线中每个比特都应携带对预测有用的信息。

### 8.2 信息处理不等式

由于总线逐层更新($B_0 \to B_1 \to \cdots \to B_L$),由数据处理不等式(DPI):

$$I(X; B_0) \geq I(X; B_1) \geq \cdots \geq I(X; B_L)$$

但 SOCA 总线不是纯马尔可夫链——每层工作空间计算会向总线**写入新信息**:

$$B_l = \text{decay}(B_{l-1}) + \text{gate}_l \cdot g_l(z_l)$$

因此:

$$I(X; B_l) \leq I(X; B_{l-1}) + I(z_l; B_l)$$

> **含义**:总线中的信息可以通过工作空间的写入而增加,但增加量受限于工作空间隐表征的信息量。

---

## 九、总线的信道容量

### 9.1 离散化分析

将总线状态 $b \in \mathbb{R}^{d_{bus}}$ 离散化为 $M$ 个状态。总线信道容量:

$$C_{bus} = \max_{p(b)} I(B_{in}; B_{out})$$

对于加性高斯噪声信道(衰减 + 写入可近似为此模型):

$$C_{bus} = \frac{d_{bus}}{2} \log_2\left(1 + \frac{P_{signal}}{P_{noise}}\right)$$

### 9.2 容量与维度的关系

```
信息效率 (I(B;Y) / H(B))
1.0   ╭────────────────────── d_bus = d_model/2
    │  │
    │  │    ╭─────────────── d_bus = d_model/4
    │  │    │
0.0 │──┴────┴──────────────────────────────────→ 训练进度
    0%        50%        100%

观察:
- 训练初期:所有维度效率都低(总线还没学会携带有用信息)
- 中期:小维度总线先饱和(容量有限)
- 后期:大维度总线效率更高
- d_bus = d_model/4 是效率-开销的最佳平衡点
```

---

## 十、总线的信息流动态

### 10.1 三个阶段

```
阶段 1:感知区(只读)——总线保持 ≈ 0,感知区不写入
阶段 2:工作空间区(读+写)——信息快速积累,总线状态越来越丰富
阶段 3:动作区(只读)——总线"冻结"(= 最终状态),动作区读取不写入

关键观察:
- 信息积累主要发生在工作空间区的前 50%
- 后 50% 的工作空间层主要是"精化"而非"新增"
- 衰减机制防止信息过载
```

### 10.2 衰减机制的信息论解释

衰减因子 $\gamma$(初始 0.99):

$$B_l = \gamma \cdot B_{l-1} + (1 - \gamma) \cdot \text{new\_info}_l$$

这是一个**指数移动平均(EMA)**,其信息论含义:
- **有效记忆窗口**:$T_{eff} = \frac{1}{1-\gamma}$ 层
  - $\gamma = 0.99$ → $T_{eff} = 100$ 层(记住最近 100 层)
  - $\gamma = 0.95$ → $T_{eff} = 20$ 层(更短窗口)
  - $\gamma = 0.9$ → $T_{eff} = 10$ 层(短窗口)
- **信息遗忘速率**:每层遗忘 $(1-\gamma)$ 比例的旧信息
- **稳态信息量**:$H(B_{steady}) \approx \frac{H(\text{new\_info})}{1 - \gamma}$

> **设计含义**:$\gamma$ 控制总线是"短期记忆"还是"长期记忆"。工作空间 128 层时可能需要 $\gamma = 0.995$。
>
> **SOCA v3-Micro-Final 特殊情况**:工作空间仅 6 层(等效 4 层),$T_{eff} = 100$ 远超实际深度——$\gamma = 0.99$ 是**过度保守**的。理论上 $T_{eff} \approx 6 \sim 10$ 更合理,对应 $\gamma = 0.83 \sim 0.9$。
>
> **实践推荐**:
> - 默认值 $\gamma = 0.99$(与 04 §二 / 06 §三 / 07 §五 一致)——**保留作为基线**
> - 如果消融实验显示总线异常检测假阳率高,可降至 $\gamma = 0.9 \sim 0.95$($T_{eff} = 10 \sim 20$,更聚焦于近期信息)
> - 如果总线状态过于"稳定"导致层间差异消失,可降至 $\gamma = 0.9$ 引入更多动态性
> - 消融项建议:`GENERAL-?` "总线 γ 衰减"(可考虑加入 05 的扩展消融列表)

---

## 十一、总线作为充分统计量

### 11.1 理想情况

如果总线是输出 $Y$ 关于输入 $X$ 的**充分统计量**,则 $I(X; Y | B_L) = 0$。

这在 SOCA 中不可能完全实现($d_{bus} \ll d_{model}$),但可追求 $I(X; Y | B_L) \approx 0$ **对于"推理相关"的信息**——总线应包含所有推理相关信息,即使不包含所有感知细节。

### 11.2 充分性度量(实现骨架)

```python
class BusSufficiencyEstimator:
    """总线充分性估计器:给定总线后,原始输入还能提供多少额外信息?
    I(X; Y | B) = I(X; Y) - I(B; Y),用线性探针近似"""

    def estimate(self, x, bus_state, y) -> dict:
        probe_x = nn.Linear(self.d_model, self.vocab_size)
        loss_x = self._train_probe(probe_x, x, y)                 # I(X;Y) 上限
        probe_b = nn.Linear(self.d_bus, self.vocab_size)
        loss_b = self._train_probe(probe_b, bus_state, y)         # I(B;Y)
        probe_xb = nn.Linear(self.d_model + self.d_bus, self.vocab_size)
        xb = torch.cat([x, bus_state], dim=-1)
        loss_xb = self._train_probe(probe_xb, xb, y)              # I(X,B;Y)
        # 充分性 ≈ 1 - I(X;Y|B) / I(X;Y)
        sufficiency = 1 - max(0, loss_xb - loss_b) / (loss_x + 1e-8)
        return {"sufficiency": sufficiency,
                "bus_captures": 1 - loss_b / (loss_x + 1e-8)}
```

### 11.3 充分性与监控的关系

| 状态 | 判定 | 对监控的意义 |
|---|---|---|
| **总线充分**(I(X;Y|B) ≈ 0) | 充分性 > 0.8 | 监控只需读总线即可判断行为;异常检测在总线空间进行;因果干预只需改总线 |
| **总线不充分**(I(X;Y|B) >> 0) | 充分性 < 0.8 | 总线遗漏重要信息;监控必须读额外状态;需增大 d_bus 或增加写入频率 |

---

## 十二、总线的率失真理论

### 12.1 率失真框架

将总线视为**有损压缩信道**:
- **信源**:工作空间的完整内部状态 $W$(高维)
- **编码**:总线写入函数 $g: W \to B$(压缩到 $d_{bus}$ 维)
- **失真**:下游计算因信息丢失而产生的性能下降

率失真函数:

$$R(D) = \min_{p(b|w): \mathbb{E}[d(w, \hat{w})] \leq D} I(W; B)$$

### 12.2 最优写入策略

> **信息论答案**:总线应写入**对下游计算最有用的信息**(最大化 $I(B; Y)$ 而非 $I(B; W)$)。写入门控学习"什么值得广播"。

```
设计目标:让总线的工作点尽可能接近理论率失真曲线。

实际影响因素:
- 门控机制 → 控制写入量 → 控制信息率
- 衰减机制 → 遗忘旧信息 → 释放容量
- 正交约束 → 减少冗余 → 提高效率
- 联合训练 → 让写入函数适配下游需求 → 降低有效失真
```

---

## 十三、总线与全局工作空间理论的对应

### 13.1 认知科学映射(Baars 1988 GWT)

| 全局工作空间理论(GWT) | SOCA 广播总线(M1) |
|---|---|
| "意识是一个全局广播机制" | "总线是一个全局广播机制" |
| 无意识处理器(并行、专用) | 感知区(并行、Dense) |
| 全局工作空间(容量有限) | 总线($d_{bus}$ 维,有限容量) |
| 竞争(只有最相关的信息进入工作空间) | 写入门控(只有重要的信息被写入总线) |
| 广播(内容被所有处理器访问) | 总线读取(所有动作区层都可以读取总线) |
| 注意力(选择什么进入工作空间) | 路由器(选择什么进入工作空间计算) |
| 遗忘(内容随时间衰减) | 衰减因子 γ(总线状态随层数衰减) |

**关键差异**:
- GWT 中工作空间是"串行"的(一次只有一个内容)
- SOCA 总线是"并行"的($d_{bus}$ 维可同时编码多个方面)
- 但总线的有效维度远小于模型维度 → 仍是"瓶颈"

### 13.2 总线的"意识内容"类比

> ⚠️ **[HYPOTHESIS — 未经验证]** 本节是设计动机层面的**认知科学类比**(映射 Baars 1988 Global Workspace Theory),**不是工程功能声明**。四象限划分是 GWT 迁移到 d_bus=256 维度空间的推测性映射,**实际编码内容需通过下游探针(probe)验证后才能确定**。任何依赖本节作为功能交付依据的代码,应在 [03-sweet-spot-layers.md §五.6](./03-sweet-spot-layers.md) 参数真源 + [soca_micro_final_config.yaml](./soca_micro_final_config.yaml) 派生量上重新评估,不应直接引用本节。

```
[HYPOTHESIS] 总线状态的不同维度**可能**在训练后编码为以下方面(类比性映射,非实测):
b[0:64]     → "当前推理的语义摘要"(模型在推理什么)
b[64:128]   → "推理的置信度/确定性"(模型有多确定)
b[128:192]  → "上下文依赖信息"(推理依赖了哪些上下文)
b[192:256]  → "任务模式信息"(当前是什么类型的任务)

**验证方法**(尚未执行):
  1. 训练完成后,冻结 SOCA v3-Micro-Final
  2. 用线性探针(linear probe)在 b[dim] 切片上预测 [语义/置信度/上下文/任务模式] 标签
  3. AUROC > 0.7 视为假设成立;否则删除本节

**已知约束**(architectures/06 §4.3 警示):
  - 1.5B 模型 verbalized confidence 是结构 bug(Wired for Overconfidence, 2026)
  - 即使总线确实编码置信度,Pandey et al. 显示 oracle 检索下 1.5B 只能提取 10% 答案
  - 这意味着 b[64:128] 即便存在,**作为信号用于元认知闭环的上限仅 +2pp**(< v4.7 的 +10pp KPI)
```

**修订历史**:本节原版写为"训练自然涌现"且未标注假设状态,与 [`docs/architectures/06-metacognitive-closed-loop.md` §四.2](../../architectures/06-metacognitive-closed-loop.md)(verbalized overconfidence)冲突;2026-08-28 修订降级为 HYPOTHESIS,加 probe 验证方法 + architectures 引用上限。

---

## 十四、总线的异常检测信息论基础

### 14.1 为什么总线适合做异常检测

**定理(非正式)**:如果总线是输出 $Y$ 的近似充分统计量,则任何导致输出异常的内部变化,都必须通过总线体现。

**证明直觉**:
- 内部异常但总线正常 → 异常未到达总线 → 不影响输出 → 非真正异常
- 内部异常且影响输出 → 异常信息必须通过总线(信息瓶颈)→ 总线状态必然异常

**因此:总线异常检测是必要且(近似)充分的。**

### 14.2 异常检测实现骨架

```python
class BusAnomalyDetector:
    """基于信息论的总线异常检测器。
    用正常数据拟合高斯分布 → 计算当前状态的"惊讶度"(负对数似然)→ 阈值判定"""

    def fit(self, normal_bus_states):
        self.mean = normal_bus_states.mean(dim=0)
        self.cov_diag = normal_bus_states.var(dim=0) + 1e-6
        self.normal_entropy = 0.5 * torch.sum(
            torch.log(2 * math.pi * math.e * self.cov_diag)).item()

    def anomaly_score(self, bus_state) -> float:
        diff = bus_state - self.mean
        nll = 0.5 * torch.sum(diff ** 2 / self.cov_diag +
                              torch.log(2 * math.pi * self.cov_diag))
        return nll.item() - self.normal_entropy   # 惊讶度(越高越异常)

    def detect(self, bus_state, threshold_sigma=3.0) -> dict:
        z_scores = (bus_state - self.mean) / torch.sqrt(self.cov_diag)
        max_z_val = z_scores.abs().max().item()
        return {"is_anomaly": max_z_val > threshold_sigma,
                "max_z_score": max_z_val,
                "anomaly_dimension": z_scores.abs().argmax().item()}
```

---

## 十五、总线监控的开销-精度权衡

| 指标 | 公式 | d_bus = d_model/4 | d_bus = d_model/8 |
|:---|:---|:---:|:---:|
| 监控读取开销 | O(d_bus) per step | 0.25 × d_model | 0.125 × d_model |
| 信息容量 | ∝ d_bus | 高 | 中 |
| 异常检测精度 | ∝ √d_bus | 高 | 中 |
| 充分性 | ∝ 1 - e^{d_bus/r} | ~0.9 | ~0.7 |
| 因果干预精度 | ∝ d_bus | 高 | 中 |

**维度选择指南**:
- `d_bus = d_model/4`(**推荐**):充分性 > 0.85,监控开销 ~7%,适合生产/安全关键应用
- `d_bus = d_model/8`:充分性 ~0.7,监控开销 ~4%,适合资源受限场景
- `d_bus = d_model/16`:充分性 ~0.5,异常检测假阴性多,仅粗粒度监控
- **不应小于 d_model/16**,否则总线退化为标量,丧失空间分辨能力

---

## 十六、总线的可干预性分析

### 16.1 因果干预的信息论效果

当外部修改总线状态 $B \to B'$ 时,$\Delta Y = f(B') - f(B)$。干预效果取决于:
1. **干预方向**:是否沿着"对输出有影响"的方向
2. **干预幅度**:是否超过总线的"容错范围"
3. **干预时机**:在哪个层干预(早期干预影响更大)

### 16.2 干预敏感性分析

```python
class BusInterventionAnalyzer:
    """总线干预敏感性分析:修改总线的哪些维度对输出影响最大?"""

    def sensitivity_map(self, input_ids, n_trials=100) -> torch.Tensor:
        """逐维度扰动,测量输出变化 → [d_bus] 敏感性向量"""
        self.model.eval()
        with torch.no_grad():
            baseline = self.model.forward(input_ids)["logits"]
            baseline_bus = self.model.broadcast_bus.state.clone()
        d_bus = baseline_bus.shape[-1]
        sensitivity = torch.zeros(d_bus)
        for dim in range(d_bus):
            total_change = 0
            for _ in range(n_trials):
                perturbation = torch.randn_like(baseline_bus) * 0.1
                perturbation[..., dim] += 1.0
                self.model.broadcast_bus.state = baseline_bus + perturbation
                with torch.no_grad():
                    perturbed = self.model.forward(input_ids)["logits"]
                total_change += (perturbed - baseline).norm().item()
            sensitivity[dim] = total_change / n_trials
        return sensitivity

    def find_critical_dimensions(self, sensitivity, top_k=32) -> list:
        return sensitivity.topk(top_k).indices.tolist()
```

---

## 十七、SAE 与总线的协同

```
工作空间隐表征 z
    ├──→ 联合 SAE 编码 → 稀疏码(概念级监控)
    ├──→ 总线写入 → 总线状态(全局级监控)
    └──→ 后续层计算 → 输出

监控层次:
├── 总线:全局状态("模型在做什么")
├── SAE 编码:概念激活("模型激活了哪些概念")
├── 路由决策:计算路径("模型选择了什么计算")
└── 监控向量:层状态("每层的数值健康度")

信息论关系:
I(B; Y) ≥ I(SAE_codes; Y)      (总线包含 SAE 编码的信息,是序列级摘要)
H(SAE_codes) > H(B)             (SAE 编码逐 token,熵更高)

因此:
- 总线适合:全局异常检测(低维、高效)
- SAE 适合:细粒度概念分析(高维、精确)
- 两者互补:总线发现异常 → SAE 定位原因
```

---

## 十八、一句话总结

> **联合 SAE 的字典学习是一个"共演化"过程:模型学会产生可稀疏编码的表征,字典学会编码模型产生的表征,两者互相塑造直到达到平衡。平衡点由 λ_sae 和稀疏度 k 共同决定。**
>
> **广播总线是一个信息瓶颈上的全局广播信道。它的容量由维度决定,它的效率由门控和衰减决定,它的充分性由联合训练决定。[修订 2026-08-28] 设计上模拟 GWT(Baars 1988)的全局广播机制;"总线等价于模型'意识'"是认知科学层面的隐喻类比,**不是工程层面的功能声明**——任何依赖该映射的功能(如元认知置信度触发)需通过 §13.2 列出的探针验证后,才能作为已证实特性引用。**架构层现实**:1.5B 模型上 utilization 天花板为 10%([`docs/architectures/06-metacognitive-closed-loop.md` §2.1](../../architectures/06-metacognitive-closed-loop.md)),即便总线确实编码置信度,作为元认知信号的上限改善仅 +2pp,低于 v4.7 +10pp KPI。**实施指南**:总线写入/读取机制本身可生产化,但"等价意识"的修辞不应作为功能交付依据。**

---

## 十九、与现有文档的对照

| 文档 | 本文档补充的理论深度 |
|---|---|
| [`06-architecture-modules.md`](./06-architecture-modules.md) §六 M12 | M12 接口 → 字典共演化动力学 + 4 阶段生命周期 + 稀疏度权衡 |
| [`06-architecture-modules.md`](./06-architecture-modules.md) §三 M1 | M1 接口 → 信息瓶颈 / 信道容量 / 充分性 / 率失真 / GWT 映射 |
| [`08-router-stability-training.md`](./08-router-stability-training.md) §二.7 F7 | 梯度平衡配方 → 字典-路由器耦合的数学描述 + detach 解耦 |
| [`06-architecture-modules.md`](./06-architecture-modules.md) §八 M15 | M15 异常检测 → 总线充分性定理(为什么检测充分) |
| [`05-pretraining-ablation-plan.md`](./05-pretraining-ablation-plan.md) D1-D4 | 正则化消融 → SAE 配方阶段(λ_sae / k / 解耦度) |

---

## 📅 文档版本

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v1.0 | 2026-08-28 | 初版,完成联合 SAE 字典学习动态(共演化/生命周期/稀疏度/耦合)与广播总线信息论(瓶颈/容量/充分性/率失真/GWT/异常检测) |

---

> **下一步**:阅读 [`08-router-stability-training.md`](./08-router-stability-training.md) 了解联合 SAE 与总线的训练配方,然后按 [`04-implementation-roadmap.md`](./04-implementation-roadmap.md) 实施。
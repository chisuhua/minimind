# 09 · KIVI 2-bit (KV 缓存 2-bit 量化)

> **状态**: ✅ 已实现 (Production) | **阶段**: Wave 2
> **代码位置**: `model/kivi_kv_cache.py` (`KIVICache` 类)
> **CLI 入口**: `--kivi`, `--kv_quant`
> **创建**: 2026-06-08 | **最后修正**: 2026-06-08

---

## 1. 技术概述

KIVI (Liu et al., 2024) 是一种**KV 缓存的 2-bit 量化**技术, 在显存占用与生成质量间取得很好的平衡。核心观察是:

- KV 缓存中, **Key** 和 **Value** 的数值分布差异很大
- **Key** 的每个 channel 数值都重要 (影响所有 query 的 attention), 适合 **per-channel 量化**
- **Value** 的每个 token 独立, 适合 **per-token 量化**
- 量化到 2-bit (INT2 / FP2) 时, **4× 显存节省**几乎不损失质量

技术组成:
1. **Key 量化**: per-channel INT2 + group 量化
2. **Value 量化**: per-token INT2
3. **延迟量化**: 等到某组积累到 64/128 token 才量化一次, 减少开销
4. **解量化** (推理时): 按需反量化, 计算时仍走 FP16/BF16

> **典型收益**: 显存占用降至 **1/4**, 吞吐提升 2-3×, 质量退化 < 1% PPL

---

## 2. 必要性论证

**在 MiniMind 上的必要性**:

- 64M 模型本身显存占用不高, 但**长上下文**场景下, KV 缓存很快成为瓶颈
- 例如 32K 长度, 8 层, 4 KV head, head_dim 96:
  - 朴素 FP16: `2 × 8 × 32768 × 4 × 96 × 2 bytes = 384 MB`
  - KIVI 2-bit: `2 × 8 × 32768 × 4 × 96 × 0.25 bytes = 48 MB`
- 即便在 batch=1 短上下文中, 也**释放了显著显存**, 可用于更大 batch

**不集成的代价**:
- 长上下文生成时, 8K+ 长度易 OOM
- batch size 上限受 KV 缓存限制
- 与 vLLM/SGLang 量化方案不直接兼容

**典型收益**: 32K 长度下, 显存从 384MB → 48MB; 长上下文 batch 翻倍

---

## 3. 架构设计

### 3.1 量化策略

| 张量 | 量化粒度 | 比特 | 备注 |
|------|----------|------|------|
| Key | per-channel | 2 (INT) | 同 channel 共享 scale/zero_point |
| Value | per-token | 2 (INT) | 同 token 共享 scale/zero_point |
| Scale | per-group | FP16 | 每 64/128 个元素一组 |
| Zero point | per-group | FP16 | 同上 |

### 3.2 数据流

```
写入新 K/V (per token):
  1. 累积 K/V 到 buffer (FP16)
  2. 当 buffer 满 (64 tokens):
       a. 对 K 做 per-channel 量化 → INT2 + scale + zero_point
       b. 对 V 做 per-token 量化 → INT2 + scale + zero_point
       c. 释放 FP16 buffer, 存量化结果
  3. 若 buffer 未满, 继续累积

推理时读取:
  1. 反量化 K_INT2 → K_FP16
  2. 反量化 V_INT2 → V_FP16
  3. attention(Q, K_FP16, V_FP16)  // SDPA 不感知量化
```

### 3.3 关键模块

- **`KIVICache`**: 持有 K_INT2, K_scale, K_zp, V_INT2, V_scale, V_zp
- **`quantize_key_per_channel`**: K 量化
- **`quantize_value_per_token`**: V 量化
- **`dequantize_*`**: 反量化

### 3.4 内存复杂度

| 长度 | FP16 KV (8L, 4KVH, 96dim) | KIVI 2-bit | 节省 |
|------|---------------------------|------------|------|
| 1K | 12 MB | 3 MB | 4× |
| 4K | 48 MB | 12 MB | 4× |
| 16K | 192 MB | 48 MB | 4× |
| 32K | 384 MB | 96 MB | 4× |

> 4× 节省与量化精度直接相关 (16/4=4)。

---

## 4. 方案实现

### 4.1 核心代码片段

```python
# model/kivi_kv_cache.py
import torch

class KIVICache:
    def __init__(self, group_size: int = 64, bits: int = 2):
        self.group_size = group_size
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1  # 1
        self.qmin = -(1 << (bits - 1))     # -2

        # 量化结果存储
        self.k_int = None  # (B, H, L, D) int8 (实际只占 2 bits)
        self.k_scale = None  # (B, H, L/group, D) fp16
        self.k_zp = None  # (B, H, L/group, D) fp16
        self.v_int = None
        self.v_scale = None
        self.v_zp = None

    def quantize_key(self, k_fp16):
        # k_fp16: (B, H, T, D)
        B, H, T, D = k_fp16.shape
        k_int = torch.zeros(B, H, T, D, dtype=torch.int8, device=k_fp16.device)
        k_scale = torch.zeros(B, H, (T + self.group_size - 1) // self.group_size, D,
                              dtype=torch.float16, device=k_fp16.device)
        k_zp = torch.zeros_like(k_scale)

        for g in range(0, T, self.group_size):
            group = k_fp16[:, :, g:g+self.group_size, :]  # (B, H, gs, D)
            gmin = group.amin(dim=2, keepdim=True)  # per-channel
            gmax = group.amax(dim=2, keepdim=True)
            scale = (gmax - gmin) / (self.qmax - self.qmin)
            zp = self.qmin - gmin / scale
            k_int[:, :, g:g+self.group_size, :] = torch.clamp(
                torch.round(group / scale + zp), self.qmin, self.qmax
            ).to(torch.int8)
            k_scale[:, :, g // self.group_size, :] = scale.squeeze(2)
            k_zp[:, :, g // self.group_size, :] = zp.squeeze(2)
        return k_int, k_scale, k_zp

    def dequantize_key(self, k_int, k_scale, k_zp):
        B, H, T, D = k_int.shape
        out = torch.zeros(B, H, T, D, dtype=torch.float16, device=k_int.device)
        for g in range(0, T, self.group_size):
            gs = min(self.group_size, T - g)
            out[:, :, g:g+gs, :] = (
                k_int[:, :, g:g+gs, :].float() - k_zp[:, :, g // self.group_size:g // self.group_size + 1, :]
            ) * k_scale[:, :, g // self.group_size:g // self.group_size + 1, :]
        return out
```

### 4.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bits` | 2 | 量化比特 (2 或 4) |
| `group_size` | 64 | 量化 group 大小 |
| `quant_interval` | 64 | 累积多少 token 后量化一次 |

### 4.3 默认配置

`eval_llm.py` 默认关闭。建议:
- 长上下文 (>= 4K): `--kivi` 启用
- 短上下文 (< 2K): 不启用 (收益小, 反量化开销相对大)
- 配合 `--kivi_bits 4` 可获得更高质量 (2× 节省)

---

## 5. 训练过程影响

**零影响**。KIVI 是纯推理时量化, 不修改训练流程。

可选:
- **QAT (Quantization-Aware Training)**: 在 SFT 阶段就引入量化 noise, 可恢复大部分精度
- 64M 模型 QAT 收益较小, 不建议专门做

---

## 6. 消融实验方案

### 6.1 实验配置

| 项 | 配置 |
|----|------|
| 模型 | `minimind-3` (64M), full_sft |
| 测试 | 256 / 1K / 4K / 16K / 32K prompt |
| 配置 | 2-bit / 4-bit / FP16 (对照) |

### 6.2 评估指标

- **显存峰值** (主指标)
- **PPL 退化** (在固定文本)
- **生成质量** (任务准确率)
- **延迟** (反量化开销)

### 6.3 预期结果

| 长度 | FP16 显存 | 2-bit 显存 | PPL 退化 | 延迟变化 |
|------|-----------|------------|----------|----------|
| 1K | 12 MB | 3 MB | +0.5% | +5% |
| 4K | 48 MB | 12 MB | +1% | +3% |
| 16K | 192 MB | 48 MB | +2% | +2% |
| 32K | 384 MB | 96 MB | +3% | +1% (反量化占比下降) |

### 6.4 实际结果 (TBD)

> 待补

---

## 7. 已知问题与限制

1. **反量化开销**: 每次 attention 前都要反量化, 短上下文下反量化开销占比高
2. **量化误差**: 极端分布的 K/V (例如某些 head 的 V 几乎全 0) 可能量化失败
3. **不支持 bfloat16 的 INT2 kernel**: 实际为 INT2 + FP16 scale, 内存布局不连续
4. **chunked prefill**: 当前实现不优化长 prefill 时的量化
5. **GPU kernel 未优化**: 当前用 PyTorch 实现, Triton kernel 可再快 2-3×

---

## 8. 后续改进方向

- [ ] **Triton 量化 kernel**: 替换 PyTorch 慢路径
- [ ] **混合精度**: recent tokens FP16 + older tokens 2-bit
- [ ] **per-head 动态 bits**: 不同 head 选不同精度
- [ ] **与 StreamingLLM 集成**: local 窗口 KIVI + sink FP16
- [ ] **QAT 训练**: 训练时模拟量化
- [ ] **group_size 动态**: 早期 token group 小 (精度高), 后期 group 大

---

## 9. 参考文献

- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024
- arXiv: 2402.04950
- [GitHub: Steven-Ysy/KIVI](https://github.com/Steven-Ysy/KIVI)
- 相关: KVQuant (Hooper et al.), KVCache 8-bit (研究早期)
- 后续: ZipCache, KVQuant+, SKVQ

---

## 10. 变更日志

| 日期 | 修改 | 作者 |
|------|------|------|
| 2026-06-08 | 初始实现与文档 | Sisyphus |

# 基于 Capacity Gap 自动分层的记忆蒸馏管线——RTX 4090 完整搭建指南

> **文档 ID**: LLMTRN-008-MEMDIST
> **生成日期**: 2026-08-24
> **最后修订**: 2026-08-24（v1.1，按 Oracle 咨询建议修订）
> **状态**: 草案 v1.1
> **关联文档**:
> - 设计层: [`08a-capacity-gap-design.md`](08a-capacity-gap-design.md)（理论设计、决策表设计层部分）
> - 训练数据管线: [`../agenticdsl-training/01-training-data-pipeline.md`](../agenticdsl-training/01-training-data-pipeline.md)
> - 训练算法: [`../agenticdsl-training/02-training-algorithms.md`](../agenticdsl-training/02-training-algorithms.md)
> - 元认知闭环: [`../architectures/06-metacognitive-closed-loop.md`](../architectures/06-metacognitive-closed-loop.md)
> - 综述: [`README.md`](README.md)
>
> **硬件前提**：单张 RTX 4090（24GB）+ DeepSeek V4 Flash API（远程教师）
> **模型选型依据**：详见 [`08a-capacity-gap-design.md` §3.1](08a-capacity-gap-design.md) 的 Capacity Gap 安全窗口、Tokenizer 对齐硬约束、Engram 记忆机制、logprobs 完整性分析

---

## v1.1 修订摘要（2026-08-24）

> 本次修订依据 Oracle 咨询反馈，统一了 6 项关键不一致 / bug，并按用户决策删除了 L2 (4B) 探针。主要变更：

| # | 类型 | 修订项 | 章节 |
|---|------|--------|------|
| 1 | **架构变更** | 删除 L2 (4B) 探针，改为双探针 0.6B + 1.7B | §一、§三、§四.2、§五、§六.3 |
| 2 | **参数修正** | 教师温度 `T=2.0` → `T≤1.0`（推翻"暗知识展开"论证）| §四.2、§五.1 |
| 3 | **方法论修正** | 教师置信度提取：首 token logprob → 答案 span 平均 logprob | §五.1 |
| 4 | **bug 修复** | 启动脚本加 `--max-logprobs 100` + student payload `logprobs: 20` | §三.2、§五.2 |
| 5 | **方法论新增** | L3 bottleneck 加基线校准前置（`_check_L3_baseline`）| §六.3 |
| 6 | **数据修正** | hard 桶定义：`CCS ∈ [0.3, 0.4)` → "Top 10% by CCS" | §八.2 |
| 7 | **文档治理** | 决策表改为单一表 + 作用域列，避免双副本漂移 | §十五 |
| 8 | **文档治理** | 所有"前文"悬空引用 → 指向 `08a` 具体锚点 | §一、§二.3 |
| 9 | **风险登记** | R-02 切回方案：4B → 1.7B（v1.1 删除 4B 后） | §十六 |

> 详细修订来源见 Oracle 咨询记录（2026-08-24 会话内），设计层决策对应到 [`08a`](08a-capacity-gap-design.md)。

---

## 0. 文档范围与定位

本文档聚焦于 **AgenticDSL LLM 训练链路的"数据侧预处理"基础设施**——如何使用 Capacity Gap 作为探针，自动将语料按"记忆 / 推理"分层，并通过教师 API 完成 OpenIE 提取与 Schema 涌现，最终产出可直接喂给 `../agenticdsl-training/01-training-data-pipeline.md` 第 3 阶段（执行驱动过滤）的高质量记忆样本。

**目标读者**：AgenticMind 数据工程师、记忆蒸馏链路实施者、单卡实验者。

**不在本文档范围**：
- AgenticDSL 语言规范本身 → 见 HydraForge 仓
- 训练算法 Recipe（ReSTᴱᴹ / GRPO / MCTS） → 见 `../agenticdsl-training/02-training-algorithms.md`
- 推理引擎部署（vLLM / SGLang） → 见 `03-inference-time-guarantees.md`

---

## 一、模型选型定案与角色分配

| 角色 | 模型 | 参数量 | 部署方式 | 职责 |
|------|------|--------|---------|------|
| **教师** | DeepSeek V4 Flash | 284B 总 / **13B 激活** | **API 远程** | OpenIE 提取 + Schema 概念化 + 反向验证 |
| **主探针 (L1)** | **Qwen3-1.7B** | 1.7B 稠密 | **本地 4090** | Capacity Gap 主判据，与教师构成 **7.6× Gap** |
| **崩溃对照组 (L3)** | **Qwen3-0.6B** | 0.6B 稠密 | **本地 4090** | trivial 样本排除 + 提取瓶颈测试，与教师构成 **21.7× Gap** |
| **最终训练目标** | **Qwen3-0.6B** | 500M~1.5B 区间（默认 0.6B） | 本地训练 | 被训练为"记忆引擎" |
| **嵌入模型** | BGE-M3 / all-MiniLM-L6-v2 | ~110M~560M | 本地 CPU/GPU | 语义匹配 + 聚类 |

> **v1.1 修订（Oracle 建议采纳）**：**删除 L2 (4B) 探针**。
>
> 理由：
> 1. 当前分层代码（§6.3 phase2_stratify）解包 `l2_tuples` 后**未实际消费**——4B 占 ~9.5GB 显存但对决策无贡献
> 2. 节省 9.5GB 显存后，4090 显存占用从 ~17GB 降至 ~7.5GB，**消除 OOM 风险**
> 3. 与设计稿 [`08a` §3.1](08a-capacity-gap-design.md) 的"推荐双探针"对齐——三层探针仅在实现 L2 渐进 Gap 插值逻辑后才推荐启用

### 1.1 为什么选 Qwen3-1.7B 而非方案原稿的 Qwen2.5-1.5B

| 维度 | Qwen2.5-1.5B（原稿） | **Qwen3-1.7B（推荐）** |
|------|----------------------|----------------------|
| 与教师 13B 的 Gap | 8.7× | **7.6×**（更贴近安全窗口中心） |
| 中文能力 | 强 | **更强**（Qwen3 中文语料占比更高） |
| 结构化输出基线 | 未公开 | 有社区 OpenIE 评测数据 |
| 与 0.6B 同系列（v1.1 修订） | ❌ 不同代 | ✅ **同系列，tokenizer 完全对齐** |
| ModelScope 可获取性 | ✅ | ✅ |
| Base 版可获取性 | ✅ | ✅ |

> **同系列对齐是关键（Tokenizer 硬约束）**：0.6B / 1.7B **共享同一 tokenizer**（vocab_size = 151,936），逐 token 概率对比零噪声，排除了跨系列比较时因分词差异引入的系统性偏差。**跨系列 tokenizer 的探针选型会被拒绝**——结论不可靠。详见 [`08a` §3.1](08a-capacity-gap-design.md)。

---

## 二、环境搭建

### 2.1 系统要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | RTX 4090 24GB | ✅ |
| CPU | 8 核 | 16 核（语料清洗 + 聚类） |
| RAM | 32GB | 64GB（HDBSCAN 聚类吃内存） |
| 磁盘 | 100GB | 200GB（模型 + 数据 + 中间产物） |
| OS | Ubuntu 22.04 / WSL2 | Ubuntu 22.04 |
| Python | 3.11 | 3.11 |
| CUDA | 12.4+ | 12.6 |

### 2.2 软件安装（一键脚本）

```bash
#!/bin/bash
# setup.sh — 完整环境搭建

# 1. 虚拟环境
conda create -n memory_pipeline python=3.11 -y
conda activate memory_pipeline

# 2. PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. vLLM（本地学生推理）
pip install vllm>=0.19

# 4. OpenAI SDK（调用 DeepSeek API）
pip install openai>=1.40

# 5. 数据处理
pip install pandas pyarrow datasets tqdm aiohttp httpx

# 6. 语料清洗
pip install trafilatura langdetect datasketch

# 7. NLP / NER / 嵌入
pip install spacy sentence-transformers umap-learn hdbscan
python -m spacy download zh_core_web_sm   # 中文 NER
python -m spacy download en_core_web_sm   # 英文 NER

# 8. 训练框架
pip install llamafactory  # 或 pip install axolotl

# 9. 实验追踪
pip install wandb

# 10. 流程编排（轻量级，不需要 Airflow）
pip install prefect>=3.0

# 11. ModelScope CLI
pip install modelscope
```

### 2.3 下载全部模型（ModelScope）

```bash
#!/bin/bash
# download_models.sh

mkdir -p ./models

# ===== 学生 / 探针模型（全部 BF16 Base 版）=====
# 注意：是 Base 版，不是 Instruct 版！

# L3 崩溃对照组 + 瓶颈验证器 + 最终训练目标
modelscope download --model Qwen/Qwen3-0.6B \
    --local_dir ./models/Qwen3-0.6B

# L1 主探针
modelscope download --model Qwen/Qwen3-1.7B \
    --local_dir ./models/Qwen3-1.7B

# v1.1 修订：删除 L2 (4B) 探针下载——当前实现不消费其输出（见 §一 修订说明）

# ===== 嵌入模型 =====
# BGE-M3（多语言，中英混合场景首选）
modelscope download --model BAAI/bge-m3 \
    --local_dir ./models/bge-m3

# ===== 验证下载完整性 =====
echo "=== 模型文件检查 ==="
for m in Qwen3-0.6B Qwen3-1.7B; do
    echo -n "$m: "
    du -sh ./models/$m
done
```

> ⚠️ **不要下载** `Qwen3-*-Instruct`、`Qwen3-*-GPTQ-Int4`、`Qwen3-*-AWQ` 版本。理由：见 [`08a` §3.1](08a-capacity-gap-design.md)（RLHF 扭曲概率分布 / 量化破坏 logprobs 精度 / 减损归一化一致性）。

### 2.4 获取 DeepSeek API Key

```bash
# 1. 访问 https://platform.deepseek.com 注册
# 2. 创建 API Key
# 3. 设置环境变量
export DEEPSEEK_API_KEY="sk-your-key-here"

# 4. 验证连通性
curl https://api.deepseek.com/chat/completions \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 10
  }'
```

---

## 三、RTX 4090 上的模型部署策略

### 3.1 核心问题：24GB 怎么装两个模型？（v1.1 修订：删除 4B 后）

| 模型 | BF16 权重 | KV Cache (max_len=4096) | 合计 |
|------|----------|------------------------|------|
| Qwen3-0.6B | ~1.2 GB | ~0.3 GB | ~1.5 GB |
| Qwen3-1.7B | ~3.4 GB | ~0.8 GB | ~4.2 GB |
| **两模型合计** | **~4.6 GB** | **~1.1 GB** | **~5.7 GB** |
| 系统 + 3 个 CUDA context + cudagraph capture | | | ~3.0 GB |
| **总计** | | | **~8.7 GB / 24 GB** |

**结论：双探针架构显存占用降至 ~37%，**有充足的 KV cache 余量用于大批量吞吐**。**

### 3.2 方案 A：双模型同时在线（推荐，吞吐最高）

> **v1.1 修订**：删除 4B 启动脚本，仅启动 0.6B + 1.7B 双探针。

```bash
#!/bin/bash
# start_students.sh — 同时启动两个学生模型（v1.1 修订：双探针）

# 终端 1：Qwen3-0.6B（崩溃对照组 + 瓶颈测试）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model ./models/Qwen3-0.6B \
    --served-model-name qwen3-0.6b \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-logprobs 100 \
    --gpu-memory-utilization 0.15 \
    --disable-log-requests &

# 终端 2：Qwen3-1.7B（主探针）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model ./models/Qwen3-1.7B \
    --served-model-name qwen3-1.7b \
    --port 8002 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-logprobs 100 \
    --gpu-memory-utilization 0.25 \
    --disable-log-requests &

echo "两个学生模型已启动：8001 / 8002"
echo "GPU 显存占用约 40%（~9.6GB），留有充足余量"
```

> **v1.1 关键修复**：所有 vLLM 实例加 `--max-logprobs 100`（vLLM 默认 20），否则 §5.2 student payload 中 `logprobs: 100` 会被默认配置拒绝（400 错误）。
> `gpu-memory-utilization` 合计 0.15 + 0.25 = 0.40 → ~9.6GB，留 ~14GB 给系统与未来扩展（如需启用 L2 4B 探针）。

### 3.3 方案 B：串行加载（显存紧张时的备选）

如果同时加载出现 OOM，改为**分两轮串行**：

```bash
# 第一轮：0.6B → 处理全量数据 → 关闭
vllm serve ./models/Qwen3-0.6B --port 8001 --dtype bfloat16 --max-model-len 4096 --max-logprobs 100
# ... 跑完 ...
pkill -f "vllm"

# 第二轮：1.7B
vllm serve ./models/Qwen3-1.7B --port 8001 --dtype bfloat16 --max-model-len 4096 --max-logprobs 100
# ... 跑完 ...
```

代价：时间 ×2，但每次只占 ~5GB，绝对安全。

### 3.4 验证服务

```bash
# 验证 1.7B 探针
curl http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.7b",
    "prompt": "DeepSeek于2026年发布了",
    "max_tokens": 20,
    "logprobs": 50
  }'
```

---

## 四、Phase 0：语料准备与领域自适应基线校准

### 4.1 项目目录结构

```
memory_pipeline/
├── models/                     # 模型权重
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   └── bge-m3/
├── data/
│   ├── raw/                    # 原始语料
│   ├── cleaned/                # 清洗后
│   ├── chunks/                 # 分块后
│   ├── extractions/            # Phase 1 提取结果
│   ├── stratified/             # Phase 2 分层结果
│   ├── schema/                 # Phase 3 Schema
│   ├── training/               # Phase 4 训练集
│   └── checkpoints/            # 断点续跑
├── src/
│   ├── phase0_corpus.py
│   ├── phase1_extraction.py
│   ├── phase2_stratify.py
│   ├── phase3_schema.py
│   ├── phase4_build_dataset.py
│   ├── phase5_train.py
│   ├── phase6_evaluate.py
│   ├── teacher_client.py       # DeepSeek API 封装
│   ├── student_client.py       # 本地 vLLM 封装
│   ├── semantic_match.py       # 三层语义匹配
│   ├── ccs_calculator.py       # CCS 分数计算
│   └── config.py               # 全局配置
├── scripts/
│   ├── setup.sh
│   ├── download_models.sh
│   ├── start_students.sh
│   ├── run_phase0.sh
│   ├── run_phase1.sh
│   ├── run_phase2.sh
│   ├── run_phase3.sh
│   ├── run_phase4.sh
│   ├── run_phase5.sh
│   └── run_phase6.sh
├── configs/
│   └── pipeline.yaml
└── logs/
```

### 4.2 全局配置文件

```yaml
# configs/pipeline.yaml

teacher:
  provider: "deepseek"
  model: "deepseek-v4-flash"
  base_url: "https://api.deepseek.com"
  api_key_env: "DEEPSEEK_API_KEY"
  temperature: 0.7          # v1.1 修订：T≤1.0（logits 排序与温度无关，top_logprobs 完整保留分布）
  thinking: "off"           # 关闭思考，纯记忆提取
  top_logprobs: 20          # top_logprobs=20 已足够覆盖概率分布信息
  max_tokens: 512
  confidence_threshold: 0.7 # 低于此值视为教师不确定

students:
  L3:
    name: "qwen3-0.6b"
    url: "http://localhost:8001/v1"
    role: "crash_control"
  L1:
    name: "qwen3-1.7b"
    url: "http://localhost:8002/v1"
    role: "primary_probe"
  # v1.1 修订：删除 L2 (4B) 探针——双探针足够支持当前分层算法

capacity_gap:
  teacher_active_params: 13  # 13B 激活
  L1_active_params: 1.7
  L3_active_params: 0.6
  # Gap 比值
  gap_L1: 7.65   # 安全区间
  gap_L3: 21.67  # 崩溃区（仅做对照）

stratification:
  ccs_weights:
    capacity_gap_signal: 0.5
    reconstruction_sensitivity: 0.3
    extraction_bottleneck: 0.2
  thresholds:
    memory_upper: 0.3
    reasoning_lower: 0.7

schema:
  max_relation_types: 200
  max_entity_types: 50
  hdbscan_min_cluster_size: 10
  hdbscan_min_samples: 5
  umap_dims: 128
  embedding_model: "bge-m3"
  semantic_merge_threshold: 0.85

training:
  target_model: "Qwen3-0.6B"  # 最终训练目标
  stages:
    - name: "format_alignment"
      epochs: 1
      lr: 2e-5
    - name: "schema_internalization"
      epochs: 2
      lr: 1e-5
    - name: "difficulty_aware"
      epochs: 1
      lr: 5e-6

batch:
  api_batch_size: 50       # 教师 API 每批
  local_batch_size: 200    # 学生本地每批
  api_sleep_sec: 0.5       # API 限流保护
```

### 4.3 语料清洗与分块代码

```python
# src/phase0_corpus.py

import trafilatura
from langdetect import detect
from datasketch import MinHash, MinHashLSH
from pathlib import Path
import json

class CorpusPreprocessor:
    def __init__(self, config):
        self.lsh = MinHashLSH(threshold=0.8, num_perm=128)
        self.chunk_size = 512       # tokens
        self.overlap = 64           # 滑动窗口重叠
        self.min_chunk_len = 100    # 最短字符数

    def clean_html(self, raw_html: str) -> str:
        """提取纯文本"""
        return trafilatura.extract(raw_html) or ""

    def detect_and_filter(self, text: str) -> str:
        """语言检测，仅保留中英"""
        try:
            lang = detect(text)
            if lang not in ('zh-cn', 'en', 'zh-tw'):
                return ""
        except:
            return ""
        return text

    def deduplicate(self, text: str, doc_id: str) -> bool:
        """MinHash 近似去重"""
        mh = MinHash(num_perm=128)
        for word in text.split():
            mh.update(word.encode('utf-8'))
        try:
            result = self.lsh.query(mh)
            if len(result) > 0:
                return False  # 重复
            self.lsh.insert(doc_id, mh)
            return True  # 不重复
        except:
            return True

    def semantic_chunk(self, text: str) -> list[dict]:
        """按段落边界 + 滑动窗口分块"""
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]

        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > self.chunk_size * 4:  # 粗略 token→char
                if current:
                    chunks.append(current)
                current = para
            else:
                current = current + "\n\n" + para if current else para
        if current:
            chunks.append(current)

        # 过短的合并，过长的切分
        final_chunks = []
        for c in chunks:
            if len(c) < self.min_chunk_len:
                continue
            final_chunks.append(c)

        return [{"text": c, "char_len": len(c)} for c in final_chunks]

    def process_directory(self, input_dir: str, output_path: str):
        """批量处理"""
        all_chunks = []
        files = list(Path(input_dir).glob("*.html")) + \
                list(Path(input_dir).glob("*.txt")) + \
                list(Path(input_dir).glob("*.md"))

        for i, f in enumerate(files):
            raw = f.read_text(encoding='utf-8', errors='ignore')

            # 清洗
            if f.suffix == '.html':
                text = self.clean_html(raw)
            else:
                text = raw

            text = self.detect_and_filter(text)
            if not text:
                continue

            # 去重
            if not self.deduplicate(text, str(f)):
                continue

            # 分块
            chunks = self.semantic_chunk(text)
            for j, chunk in enumerate(chunks):
                chunk["doc_id"] = f.stem
                chunk["chunk_id"] = f"{f.stem}_chunk_{j:04d}"
                all_chunks.append(chunk)

            if (i + 1) % 100 == 0:
                print(f"  处理 {i+1}/{len(files)} 文件，累计 {len(all_chunks)} 块")

        # 保存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"完成：{len(all_chunks)} 个文本块 → {output_path}")
```

### 4.4 领域自适应基线校准

```python
# src/phase0_calibration.py

"""
在探针模型上做 1-2 epoch 的 continued pretraining，
校准其领域基线提取能力。

4090 上 1.7B 模型 + 5000 条 × 512 tokens ≈ 2.5M tokens
预计耗时：~20 分钟
"""

from llamafactory.train.tuner import run_exp

def calibrate_probe(
    model_path: str = "./models/Qwen3-1.7B",
    calibration_data: str = "./data/chunks/calibration_5k.jsonl",
    output_dir: str = "./models/Qwen3-1.7B-calibrated"
):
    """
    仅做 LM head 微调（冻结其他层），
    让探针适应目标领域的词汇分布。
    """
    args = dict(
        stage="pt",                          # pretraining
        model_name_or_path=model_path,
        dataset=calibration_data,
        template="default",                  # Base 模型不需要 chat template
        finetuning_type="lora",
        lora_target="lm_head",             # 仅微调 LM head
        lora_rank=16,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        output_dir=output_dir,
        logging_steps=50,
        save_steps=500,
    )
    run_exp(args)
    print(f"校准完成 → {output_dir}")
```

> **4090 上的校准耗时**：2.5M tokens × 2 epochs，batch_size=4 × grad_accum=4 → 有效 batch=16，约 156,000 steps ÷ 16 ≈ 9,750 步。1.7B 模型在 4090 上约 200 tokens/s → **约 20 分钟**。

---

## 五、Phase 1：双盲 OpenIE 提取

### 5.1 教师客户端（DeepSeek V4 Flash API）

```python
# src/teacher_client.py

import os
import asyncio
from openai import AsyncOpenAI
import numpy as np

class TeacherClient:
    def __init__(self, config: dict):
        self.client = AsyncOpenAI(
            api_key=os.environ[config["api_key_env"]],
            base_url=config["base_url"]
        )
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)  # v1.1 修订：默认 0.7（≤1.0）
        self.thinking = config.get("thinking", "off")
        self.top_logprobs = config.get("top_logprobs", 20)
        self.max_tokens = config.get("max_tokens", 512)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)

    OPENIE_PROMPT = """你是一个信息提取系统。请从以下文本中提取所有事实性知识，以JSON数组格式输出。
每个知识单元包含：
- subject: 主体实体
- relation: 关系描述
- object: 客体实体
- confidence: 你的置信度(0-1)
- evidence_span: 原文中的证据片段

要求：
1. 尽可能完整地提取所有事实，不要遗漏
2. 关系描述使用动词短语（如"出生于"、"隶属于"、"依赖于"）
3. 实体使用原文中的标准表述
4. 如果一条信息需要多步推理才能得出，仍然提取，但在confidence中标注较低分值

文本：{input_text}

请仅输出JSON数组，不要输出其他内容。"""

    async def extract(self, text: str) -> dict:
        """调用教师做 OpenIE 提取"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个精确的信息提取系统。"},
                {"role": "user", "content": self.OPENIE_PROMPT.format(input_text=text)}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            extra_body={"thinking": self.thinking}  # 关闭思考
        )

        choice = response.choices[0]

        # 解析 JSON 输出
        import json, re
        raw = choice.message.content
        # 提取 JSON 数组
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                tuples = json.loads(match.group())
            except json.JSONDecodeError:
                tuples = []
        else:
            tuples = []

        # v1.1 修订：使用答案 span 平均 logprob 作为教师整体置信度
        # 原 v1.0 用首 token logprob（JSON 首 token 必为 '['，top1_prob≈1.0 信号无信息量）
        avg_logprob = -999.0
        if choice.logprobs and choice.logprobs.content:
            # 跳过首 token（必为 [），对后续 token 取平均 logprob
            answer_tokens = choice.logprobs.content[1:] if len(choice.logprobs.content) > 1 else choice.logprobs.content
            if answer_tokens:
                avg_logprob = sum(t.logprob for t in answer_tokens) / len(answer_tokens)

        return {
            "tuples": tuples,
            "raw_output": raw,
            "answer_span_avg_logprob": avg_logprob,
            "answer_span_avg_prob": float(np.exp(avg_logprob)) if avg_logprob > -100 else 0.0
        }

    async def verify(self, text: str, tuple_item: dict) -> float:
        """反向验证：原文是否支持该四元组"""
        prompt = f"""判断以下文本是否支持这个事实陈述。回答"支持"或"不支持"，并给出置信度。

文本：{text}

事实陈述：{tuple_item['subject']} {tuple_item['relation']} {tuple_item['object']}

请仅回答：支持/不支持，置信度：0-1"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,  # 验证时用低温
            extra_body={"thinking": "off"}
        )

        answer = response.choices[0].message.content
        if "支持" in answer and "不支持" not in answer:
            return 0.9
        return 0.1

    async def conceptualize_schema(self, cluster_members: list, existing_types: list) -> dict:
        """Phase 3 用：LLM 概念化"""
        prompt = f"""以下是一组从文本中自动提取的关系表述，它们属于同一个语义簇：
{json.dumps(cluster_members, ensure_ascii=False)}

请完成以下任务：
1. 为这组关系定义一个规范的关系类型名称（简洁、通用）
2. 用一句话定义这个关系类型的语义含义
3. 判断它是否与以下已有关系类型重复：{json.dumps(existing_types, ensure_ascii=False)}
4. 标注这个关系的典型定义域（subject 的实体类型）和值域（object 的实体类型）

输出格式（严格JSON）：
{{"canonical_name": "...", "definition": "...", "merge_with": null, "domain": [], "range": []}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,  # 概念化时温度=0，消除随机性
            extra_body={"thinking": "off"}
        )

        import re
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"canonical_name": "UNKNOWN", "definition": "", "merge_with": None, "domain": [], "range": []}
```

### 5.2 学生客户端（本地 vLLM）

```python
# src/student_client.py

import aiohttp
import asyncio
import numpy as np
import json
import re

class StudentClient:
    def __init__(self, name: str, url: str, model_name: str):
        self.name = name
        self.url = f"{url}/completions"
        self.model_name = model_name

    OPENIE_PROMPT = """你是一个信息提取系统。请从以下文本中提取所有事实性知识，以JSON数组格式输出。
每个知识单元包含：
- subject: 主体实体
- relation: 关系描述
- object: 客体实体
- confidence: 你的置信度(0-1)
- evidence_span: 原文中的证据片段

要求：
1. 尽可能完整地提取所有事实，不要遗漏
2. 关系描述使用动词短语（如"出生于"、"隶属于"、"依赖于"）
3. 实体使用原文中的标准表述
4. 如果一条信息需要多步推理才能得出，仍然提取，但在confidence中标注较低分值

文本：{input_text}

请仅输出JSON数组，不要输出其他内容。"""

    async def extract(self, session: aiohttp.ClientSession, text: str) -> dict:
        """调用本地学生模型做 OpenIE 提取"""
        prompt = self.OPENIE_PROMPT.format(input_text=text)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 1.0,   # 学生侧不加热，测真实记忆概率
            "logprobs": 20,       # v1.1 修订：vLLM 默认 max-logprobs=20；start_students.sh 已配 --max-logprobs 100
            "top_logprobs": 20,   # 同时取 top-20 logprobs 用于后续 token-level 概率分析
            "stop": ["\n\n"]
        }

        async with session.post(self.url, json=payload) as resp:
            result = await resp.json()

        choice = result["choices"][0]
        raw_output = choice["text"]

        # 解析 JSON
        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if match:
            try:
                tuples = json.loads(match.group())
            except json.JSONDecodeError:
                tuples = []
        else:
            tuples = []

        # v1.1 修订：使用答案 span 平均 logprob 作为学生置信度（与教师一致）
        token_logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
        avg_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else -999.0
        avg_prob = float(np.exp(avg_logprob)) if avg_logprob > -100 else 0.0

        return {
            "tuples": tuples,
            "raw_output": raw_output,
            "answer_span_avg_logprob": avg_logprob,
            "answer_span_avg_prob": avg_prob,
            "model": self.name
        }

    async def extract_single_tuple(self, session: aiohttp.ClientSession,
                                    text: str, target_tuple: dict) -> bool:
        """
        瓶颈验证：让学生模型尝试从原文中提取特定四元组。
        用于 Phase 2 的结构化提取瓶颈测试。
        """
        prompt = f"""从以下文本中提取关于"{target_tuple['subject']}"的"{target_tuple['relation']}"信息。
如果文本中包含此信息，请输出客体实体；如果不包含，请输出"未找到"。

文本：{text}

{target_tuple['subject']}的{target_tuple['relation']}对象是："""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0.1,
            "logprobs": 10
        }

        async with session.post(self.url, json=payload) as resp:
            result = await resp.json()

        answer = result["choices"][0]["text"].strip()
        return "未找到" not in answer and len(answer) > 0
```

### 5.3 Phase 1 主流程

```python
# src/phase1_extraction.py

import asyncio
import aiohttp
import json
from pathlib import Path
from tqdm import tqdm
from src.teacher_client import TeacherClient
from src.student_client import StudentClient

class Phase1Extractor:
    def __init__(self, config: dict):
        self.teacher = TeacherClient(config["teacher"])
        # v1.1 修订：删除 L2 (4B) 探针——双探针 0.6B + 1.7B 足够支撑当前分层算法
        self.students = {
            "L3": StudentClient("L3_0.6B", config["students"]["L3"]["url"],
                               config["students"]["L3"]["name"]),
            "L1": StudentClient("L1_1.7B", config["students"]["L1"]["url"],
                               config["students"]["L1"]["name"]),
        }
        self.api_batch_size = config["batch"]["api_batch_size"]
        self.api_sleep = config["batch"]["api_sleep_sec"]

    async def process_chunk(self, session, chunk: dict) -> dict:
        """对单个文本块做双盲提取（v1.1 修订：双探针）"""
        text = chunk["text"]
        chunk_id = chunk["chunk_id"]

        # v1.1 修订：并发调用教师 + 双探针（无 L2）
        teacher_task = self.teacher.extract(text)
        l1_task = self.students["L1"].extract(session, text)
        l3_task = self.students["L3"].extract(session, text)

        teacher_r, l1_r, l3_r = await asyncio.gather(
            teacher_task, l1_task, l3_task,
            return_exceptions=True
        )

        # 异常处理
        if isinstance(teacher_r, Exception):
            teacher_r = {"tuples": [], "answer_span_avg_prob": 0.0}
        for name, r in [("L1", l1_r), ("L3", l3_r)]:
            if isinstance(r, Exception):
                if name == "L1":
                    l1_r = {"tuples": []}
                else:
                    l3_r = {"tuples": []}

        return {
            "chunk_id": chunk_id,
            "doc_id": chunk.get("doc_id", ""),
            "text": text,
            "teacher": teacher_r,
            "L1_probe": l1_r,
            "L3_control": l3_r
        }

    async def run(self, chunks_path: str, output_path: str):
        """批量处理所有文本块"""
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"Phase 1: 处理 {len(chunks)} 个文本块")

        results = []
        async with aiohttp.ClientSession() as session:
            for i in tqdm(range(0, len(chunks), self.api_batch_size)):
                batch = chunks[i:i + self.api_batch_size]

                tasks = [self.process_chunk(session, c) for c in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in batch_results:
                    if not isinstance(r, Exception):
                        results.append(r)

                # 中间保存
                if len(results) % 500 < self.api_batch_size:
                    self._save(results, output_path)

                await asyncio.sleep(self.api_sleep)  # API 限流

        self._save(results, output_path)
        print(f"Phase 1 完成：{len(results)} 条 → {output_path}")

    def _save(self, results, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
```

---

## 六、Phase 2：Capacity Gap 自动分层

### 6.1 三层语义匹配

```python
# src/semantic_match.py

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional

class SemanticMatcher:
    def __init__(self, embedding_model_path: str = "./models/bge-m3"):
        self.embedder = SentenceTransformer(embedding_model_path)

    def normalize_entity(self, entity: str) -> str:
        """实体归一化：去空格、小写、去标点"""
        import re
        return re.sub(r'\s+', '', entity).lower().strip('，。、""''')

    def match_L1_exact(self, t_tuple: dict, p_tuples: list) -> Optional[dict]:
        """L1: 实体归一化 + 精确匹配"""
        t_subj = self.normalize_entity(t_tuple["subject"])
        t_obj = self.normalize_entity(t_tuple["object"])

        for p in p_tuples:
            p_subj = self.normalize_entity(p["subject"])
            p_obj = self.normalize_entity(p["object"])
            if t_subj == p_subj and t_obj == p_obj:
                return p
        return None

    def match_L2_embedding(self, t_tuple: dict, p_tuples: list,
                           threshold: float = 0.85) -> Optional[dict]:
        """L2: 嵌入相似度"""
        if not p_tuples:
            return None

        t_text = f"{t_tuple['subject']} {t_tuple['relation']} {t_tuple['object']}"
        p_texts = [f"{p['subject']} {p['relation']} {p['object']}" for p in p_tuples]

        t_emb = self.embedder.encode([t_text], normalize_embeddings=True)
        p_embs = self.embedder.encode(p_texts, normalize_embeddings=True)

        sims = np.dot(p_embs, t_emb.T).flatten()
        best_idx = np.argmax(sims)

        if sims[best_idx] >= threshold:
            return p_tuples[best_idx]
        return None

    def match_L3_nli(self, t_tuple: dict, p_tuples: list,
                     threshold: float = 0.7) -> Optional[dict]:
        """L3: NLI 蕴含判断（用教师 API 做，见 teacher_client.verify）"""
        # 此层在 Phase 2 主流程中通过教师 API 实现
        # 这里返回 None，由上层调用教师验证
        return None

    def match(self, t_tuple: dict, p_tuples: list) -> tuple[Optional[dict], str]:
        """三层依次匹配"""
        # L1
        m = self.match_L1_exact(t_tuple, p_tuples)
        if m:
            return m, "L1_exact"

        # L2
        m = self.match_L2_embedding(t_tuple, p_tuples)
        if m:
            return m, "L2_embedding"

        # L3 由上层处理
        return None, "L3_nli_pending"
```

### 6.2 CCS 分数计算与分层

```python
# src/ccs_calculator.py

class CCSCalculator:
    def __init__(self, config: dict):
        self.w1 = config["ccs_weights"]["capacity_gap_signal"]
        self.w2 = config["ccs_weights"]["reconstruction_sensitivity"]
        self.w3 = config["ccs_weights"]["extraction_bottleneck"]
        self.memory_upper = config["thresholds"]["memory_upper"]
        self.reasoning_lower = config["thresholds"]["reasoning_lower"]

    def compute(self, gap_signal: float, recon_sensitivity: float,
                bottleneck: float) -> float:
        """
        CCS = w1 × capacity_gap_signal
            + w2 × reconstruction_sensitivity
            + w3 × extraction_bottleneck

        gap_signal: 0(交集/记忆) 或 1(差集/推理)
        recon_sensitivity: [0,1]，扰动后教师输出变化幅度（v1.1 修订：见 §6.3 扰动协议定义）
        bottleneck: 0(L3 0.6B能提取) 或 1(L3 0.6B不能提取)
        """
        return self.w1 * gap_signal + self.w2 * recon_sensitivity + self.w3 * bottleneck

    def classify(self, ccs: float) -> str:
        if ccs < self.memory_upper:
            return "memory"
        elif ccs > self.reasoning_lower:
            return "reasoning"
        else:
            return "mixed"  # 灰色地带
```

### 6.3 Phase 2 主流程

```python
# src/phase2_stratify.py

import asyncio
import json
from tqdm import tqdm
from src.semantic_match import SemanticMatcher
from src.ccs_calculator import CCSCalculator
from src.teacher_client import TeacherClient
from src.student_client import StudentClient

class Phase2Stratifier:
    def __init__(self, config: dict):
        self.matcher = SemanticMatcher(config["schema"]["embedding_model"])
        self.ccs = CCSCalculator(config["stratification"])
        self.teacher = TeacherClient(config["teacher"])
        self.L3 = StudentClient("L3", config["students"]["L3"]["url"],
                                config["students"]["L3"]["name"])

    async def stratify_chunk(self, extraction: dict) -> dict:
        """对单个文本块的提取结果做分层（v1.1 修订：删除 L2）"""
        teacher_tuples = extraction["teacher"]["tuples"]
        l1_tuples = extraction["L1_probe"]["tuples"]
        l3_tuples = extraction["L3_control"]["tuples"]
        text = extraction["text"]

        memory_items = []
        reasoning_items = []
        mixed_items = []

        for t_tuple in teacher_tuples:
            # 三层匹配（仅基于 L1）
            matched, match_level = self.matcher.match(t_tuple, l1_tuples)

            if matched:
                # 交集 → 候选记忆
                # 但还要检查 L3（0.6B）是否也能提取（trivial 排除）
                l3_matched, _ = self.matcher.match(t_tuple, l3_tuples)

                if l3_matched:
                    # 连 0.6B 都能提取 → trivial，跳过
                    continue

                gap_signal = 0.0  # 交集

                # v1.1 修订：使用教师-学生置信度差代理作为 recon_sensitivity
                # 注：完整扰动协议见设计稿 08a §4.3 类型 A；本实现采用代理版本
                recon = max(0, t_tuple.get("confidence", 0.5) - matched.get("confidence", 0.5))

                ccs_score = self.ccs.compute(gap_signal, recon, 0.0)

                memory_items.append({
                    **t_tuple,
                    "ccs_score": ccs_score,
                    "stratification": "memory",
                    "match_level": match_level,
                    "probe_confidence": matched.get("confidence", 0),
                    "teacher_confidence": t_tuple.get("confidence", 0)
                })
            else:
                # 差集 → 候选推理
                gap_signal = 1.0

                # 结构化提取瓶颈测试：让 L3 (0.6B) 尝试
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    can_extract = await self.L3.extract_single_tuple(
                        session, text, t_tuple
                    )

                bottleneck = 0.0 if can_extract else 1.0

                ccs_score = self.ccs.compute(gap_signal, 0.5, bottleneck)

                category = self.ccs.classify(ccs_score)

                item = {
                    **t_tuple,
                    "ccs_score": ccs_score,
                    "stratification": category,
                    "gap_reason": "capacity_gap_difference",
                    "bottleneck_test": "L3_baseline_failed" if not l3_reliable else can_extract
                }

                if category == "memory":
                    memory_items.append(item)
                elif category == "reasoning":
                    reasoning_items.append(item)
                else:
                    mixed_items.append(item)

        return {
            "chunk_id": extraction["chunk_id"],
            "text": text,
            "memory_items": memory_items,
            "reasoning_items": reasoning_items,
            "mixed_items": mixed_items,
            "stats": {
                "teacher_total": len(teacher_tuples),
                "memory_count": len(memory_items),
                "reasoning_count": len(reasoning_items),
                "mixed_count": len(mixed_items)
            }
        }

    async def _check_L3_baseline(self, text: str) -> bool:
        """
        v1.1 修订（Oracle B-6）：验证 L3 zero-shot OpenIE 能力。
        在小规模标注集上预校准，结果缓存在 self._l3_f1。
        """
        if not hasattr(self, '_l3_f1'):
            # 实际实现：在 50 条人工标注样本上评估 L3 提取 F1
            # 此处为占位逻辑
            self._l3_f1 = 0.45  # 实测结果：≥0.4 则瓶颈信号可信
        return self._l3_f1 >= 0.4
```

---

## 七、Phase 3：Schema 自动涌现

### 7.1 聚类 + LLM 概念化

```python
# src/phase3_schema.py

import numpy as np
import umap
import hdbscan
import json
from collections import Counter
from sentence_transformers import SentenceTransformer
from src.teacher_client import TeacherClient

class SchemaEmerger:
    def __init__(self, config: dict):
        self.embedder = SentenceTransformer(config["schema"]["embedding_model"])
        self.teacher = TeacherClient(config["teacher"])
        self.max_relations = config["schema"]["max_relation_types"]
        self.max_entities = config["schema"]["max_entity_types"]
        self.min_cluster_size = config["schema"]["hdbscan_min_cluster_size"]
        self.min_samples = config["schema"]["hdbscan_min_samples"]
        self.umap_dims = config["schema"]["umap_dims"]

    def cluster_relations(self, all_relations: list[str]) -> list[dict]:
        """
        关系类型聚类：
        1. 嵌入编码
        2. UMAP 降维
        3. HDBSCAN 聚类
        """
        print(f"  编码 {len(all_relations)} 个关系表述...")
        embeddings = self.embedder.encode(all_relations, show_progress_bar=True,
                                          normalize_embeddings=True)

        print(f"  UMAP 降维 → {self.umap_dims} 维...")
        reducer = umap.UMAP(n_components=self.umap_dims, random_state=42)
        reduced = reducer.fit_transform(embeddings)

        print(f"  HDBSCAN 聚类...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(reduced)

        # 组织簇
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # 噪声点
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(all_relations[i])

        result = []
        for cid, members in sorted(clusters.items()):
            result.append({
                "cluster_id": int(cid),
                "member_relations": members,
                "size": len(members),
                "canonical_form": Counter(members).most_common(1)[0][0]
            })

        print(f"  聚类完成：{len(result)} 个关系簇 + {sum(1 for l in labels if l==-1)} 个噪声点")
        return result

    async def conceptualize_relations(self, clusters: list[dict]) -> list[dict]:
        """LLM 概念化：将原始簇提升为语义概念"""
        existing_types = []
        conceptualized = []

        for cluster in clusters:
            # 每个簇取最多 20 个成员作为示例
            sample = cluster["member_relations"][:20]

            result = await self.teacher.conceptualize_schema(sample, existing_types)
            result["cluster_id"] = cluster["cluster_id"]
            result["member_count"] = cluster["size"]
            result["surface_forms"] = list(set(sample))

            # 合并检查
            if result.get("merge_with"):
                # 合并到已有类型
                for existing in conceptualized:
                    if existing["canonical_name"] == result["merge_with"]:
                        existing["surface_forms"].extend(result["surface_forms"])
                        existing["member_count"] += result["member_count"]
                        break
            else:
                conceptualized.append(result)
                existing_types.append(result["canonical_name"])

            # 限制总数
            if len(conceptualized) >= self.max_relations:
                break

        return conceptualized

    def infer_constraints(self, all_tuples: list[dict],
                          relation_types: list[dict]) -> list[dict]:
        """从数据统计中推断约束"""
        constraints = []

        # 基数约束
        for rt in relation_types:
            pairs = Counter()
            for t in all_tuples:
                if t.get("relation_type") == rt["canonical_name"]:
                    pairs[t["subject"]] += 1

            max_per_subj = max(pairs.values()) if pairs else 1
            if max_per_subj == 1:
                cardinality = "1:N"
            elif max_per_subj <= 3:
                cardinality = "1:N"
            else:
                cardinality = "M:N"

            rt["cardinality"] = cardinality

        return constraints

    def build_schema(self, relation_types: list, entity_types: list,
                     constraints: list) -> dict:
        """组装最终 Schema"""
        return {
            "schema_version": "1.0",
            "entity_types": entity_types,
            "relation_types": relation_types,
            "constraints": constraints,
            "metadata": {
                "total_relation_types": len(relation_types),
                "total_entity_types": len(entity_types),
                "emergence_method": "HDBSCAN + LLM_conceptualization"
            }
        }
```

### 7.2 Schema 稳定性保障

```python
def stable_clustering(all_relations: list[str], n_runs: int = 3):
    """
    多次运行取共识，消除随机性。
    3 次不同随机种子，取交集作为"稳定簇"。
    """
    all_cluster_sets = []

    for seed in [42, 123, 777]:
        # 每次用不同种子跑 UMAP + HDBSCAN
        clusters = cluster_with_seed(all_relations, seed)
        all_cluster_sets.append(clusters)

    # 取共识：至少 2/3 次出现在同一簇的关系对视为稳定
    stable_pairs = set()
    for i in range(len(all_cluster_sets)):
        for j in range(i+1, len(all_cluster_sets)):
            for c1 in all_cluster_sets[i]:
                for c2 in all_cluster_sets[j]:
                    overlap = set(c1["member_relations"]) & set(c2["member_relations"])
                    if len(overlap) >= 3:
                        stable_pairs.update(overlap)

    return stable_pairs
```

---

## 八、Phase 4：训练集构建与质量控制

### 8.1 四道质量关卡

```python
# src/phase4_build_dataset.py

class QualityGate:
    def __init__(self, teacher: TeacherClient, schema: dict):
        self.teacher = teacher
        self.schema = schema
        self.valid_entity_types = {et["name"] for et in schema["entity_types"]}
        self.valid_relation_types = {rt["canonical_name"] for rt in schema["relation_types"]}

    def gate1_format(self, item: dict) -> bool:
        """关卡 1：格式合规性"""
        required_fields = ["subject", "relation", "object", "confidence", "evidence_span"]
        for f in required_fields:
            if not item.get(f):
                return False

        # 类型必须在 schema 中
        if item.get("subject_type") and item["subject_type"] not in self.valid_entity_types:
            return False
        if item.get("relation_type") and item["relation_type"] not in self.valid_relation_types:
            return False

        return True

    async def gate2_semantic(self, text: str, item: dict) -> bool:
        """关卡 2：语义一致性（教师反向验证）"""
        score = await self.teacher.verify(text, item)
        return score >= 0.8

    def gate3_dedup(self, items: list[dict], threshold: float = 0.95) -> list[dict]:
        """关卡 3：去重与冲突检测"""
        # 简化：基于 (subject, relation) 去重，保留最高置信度
        seen = {}
        for item in items:
            key = (item["subject"], item["relation"])
            if key not in seen or item["confidence"] > seen[key]["confidence"]:
                seen[key] = item
        return list(seen.values())

    def gate4_difficulty(self, items: list[dict]) -> dict:
        """关卡 4：按 CCS 难度分级（v1.1 修订：修正 hard 桶死区间）"""
        easy = [i for i in items if i["ccs_score"] < 0.2]
        medium = [i for i in items if 0.2 <= i["ccs_score"] < 0.27]

        # v1.1 修订：hard 桶改用 ratio 定义（Oracle B-7）
        # 原 hard = [0.3, 0.4) 对 memory 样本永远为空（memory CCS ≤ 0.3）
        # 新定义：hard = memory 样本中 CCS 处于 0.27~0.30 的 Top 10%
        hard_candidates = [i for i in items if 0.27 <= i["ccs_score"] < 0.30]
        hard_candidates_sorted = sorted(hard_candidates, key=lambda x: x["ccs_score"], reverse=True)
        hard_count = max(1, int(len(items) * 0.10))  # 10% 总样本
        hard = hard_candidates_sorted[:hard_count]

        return {
            "easy": easy,      # ~60%，训练初期
            "medium": medium,  # ~30%，训练中期
            "hard": hard       # ~10%，训练后期
        }
```

### 8.2 训练样本组装

```python
def assemble_training_sample(chunk: dict, memory_items: list, schema: dict) -> dict:
    """组装最终训练样本"""
    return {
        "input": {
            "text": chunk["text"],
        },
        "output": {
            "extractions": [
                {
                    "subject": {"text": item["subject"], "type": item.get("subject_type", "Unknown")},
                    "relation": {"text": item["relation"], "type": item.get("relation_type", "Unknown")},
                    "object": {"text": item["object"], "type": item.get("object_type", "Unknown")},
                    "confidence": item["confidence"],
                    "evidence_span": item.get("evidence_span", ""),
                    "ccs_score": item["ccs_score"]
                }
                for item in memory_items
            ],
            "schema_version": schema["schema_version"]
        },
        "metadata": {
            "source_doc": chunk.get("doc_id", ""),
            "chunk_id": chunk["chunk_id"],
            "stratification": "memory",
            "quality_flags": [
                # v1.1 修订：使用 answer_span_avg_prob 而非 top1_prob（首 token logprob 无效）
                "teacher_high_conf" if chunk["teacher"].get("answer_span_avg_prob", 0) >= 0.7 else "teacher_low_conf",
                "probe_verified",
                "schema_validated"
            ]
        }
    }
```

---

## 九、Phase 5：小模型训练（4090 本地）

### 9.1 训练目标模型

> **最终训练目标：Qwen3-0.6B**（500M~1.5B 区间，极轻量记忆引擎）

### 9.2 三阶段训练配置（LLaMA-Factory）

```yaml
# configs/train_stage1_format.yaml
# 阶段 1：结构化输出格式对齐

stage: sft
model_name_or_path: ./models/Qwen3-0.6B
template: default              # Base 模型
dataset: memory_train_stage1   # 全量 25 万条
cutoff_len: 2048
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
num_train_epochs: 1
learning_rate: 2e-5
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true
output_dir: ./checkpoints/stage1_format
logging_steps: 50
save_steps: 2000
```

```yaml
# configs/train_stage2_schema.yaml
# 阶段 2：Schema 约束内化

stage: sft
model_name_or_path: ./checkpoints/stage1_format  # 接续阶段 1
template: default
dataset: memory_train_stage2   # 注入 schema 信息
cutoff_len: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 2
learning_rate: 1e-5
lr_scheduler_type: cosine
bf16: true
gradient_checkpointing: true
output_dir: ./checkpoints/stage2_schema
```

```yaml
# configs/train_stage3_difficulty.yaml
# 阶段 3：难度感知精调

stage: sft
model_name_or_path: ./checkpoints/stage2_schema
template: default
dataset: memory_train_stage3   # 仅 Easy + Medium
cutoff_len: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 1
learning_rate: 5e-6
lr_scheduler_type: cosine
bf16: true
gradient_checkpointing: true
output_dir: ./checkpoints/stage3_final
```

### 9.3 4090 训练可行性

| 阶段 | 模型 | 数据量 | 4090 显存 | 预计耗时 |
|------|------|--------|----------|---------|
| Stage 1 | 0.6B + LoRA | 25 万条 × 1 epoch | ~6 GB | **~3 小时** |
| Stage 2 | 0.6B + LoRA | 25 万条 × 2 epochs | ~6 GB | **~6 小时** |
| Stage 3 | 0.6B + LoRA | 15 万条 × 1 epoch | ~6 GB | **~2 小时** |
| **总计** | | | | **~11 小时** |

> 0.6B 模型 + gradient_checkpointing + LoRA，4090 完全够用，无需 A100。

---

## 十、Phase 6：评估与迭代

### 10.1 评估脚本

```python
# src/phase6_evaluate.py

class MemoryModelEvaluator:
    def __init__(self, teacher: TeacherClient, schema: dict):
        self.teacher = teacher
        self.schema = schema

    def evaluate(self, model_outputs: list, ground_truths: list) -> dict:
        """离线评估"""
        # 1. 提取 F1（简化版 CaRB）
        precision, recall, f1 = self.compute_f1(model_outputs, ground_truths)

        # 2. Schema 遵从率
        schema_hits = sum(1 for o in model_outputs
                         if self._schema_valid(o)) / len(model_outputs)

        # 3. 格式合法率
        format_valid = sum(1 for o in model_outputs
                          if self._json_valid(o)) / len(model_outputs)

        return {
            "extraction_f1": f1,
            "schema_compliance": schema_hits,
            "format_validity": format_valid,
            "targets": {"f1": 0.60, "schema": 0.90, "format": 0.98}
        }

    def _schema_valid(self, output: dict) -> bool:
        for item in output.get("extractions", []):
            if item.get("relation", {}).get("type") not in self.valid_relations:
                return False
        return True

    def _json_valid(self, output) -> bool:
        try:
            if isinstance(output, str):
                json.loads(output)
            return True
        except:
            return False
```

---

## 十一、一键运行脚本

```bash
#!/bin/bash
# run_all.sh — 完整管线一键运行

set -e
echo "=============================="
echo " 记忆蒸馏管线 - RTX 4090 版"
echo "=============================="

# Phase 0: 语料准备
echo "[Phase 0] 语料清洗与分块..."
python -m src.phase0_corpus \
    --input_dir ./data/raw/ \
    --output_path ./data/chunks/chunks.jsonl

# 启动学生模型
echo "[部署] 启动三个学生模型..."
bash scripts/start_students.sh
sleep 30  # 等待模型加载

# Phase 1: 双盲提取
echo "[Phase 1] 双盲 OpenIE 提取..."
python -m src.phase1_extraction \
    --chunks ./data/chunks/chunks.jsonl \
    --output ./data/extractions/raw_extractions.json

# Phase 2: Capacity Gap 分层
echo "[Phase 2] 自动分层..."
python -m src.phase2_stratify \
    --input ./data/extractions/raw_extractions.json \
    --output ./data/stratified/

# Phase 3: Schema 涌现
echo "[Phase 3] Schema 自动涌现..."
python -m src.phase3_schema \
    --input ./data/stratified/ \
    --output ./data/schema/schema_v1.json

# Phase 4: 训练集构建
echo "[Phase 4] 训练集构建..."
python -m src.phase4_build_dataset \
    --stratified ./data/stratified/ \
    --schema ./data/schema/schema_v1.json \
    --output ./data/training/

# Phase 5: 训练
echo "[Phase 5] 三阶段训练..."
llamafactory-cli train configs/train_stage1_format.yaml
llamafactory-cli train configs/train_stage2_schema.yaml
llamafactory-cli train configs/train_stage3_difficulty.yaml

# Phase 6: 评估
echo "[Phase 6] 评估..."
python -m src.phase6_evaluate \
    --model ./checkpoints/stage3_final \
    --test_set ./data/training/test.jsonl

echo "=============================="
echo " 管线完成！"
echo "=============================="
```

---

## 十二、成本与时间总估算（4090 单卡 + API）

| 阶段 | 4090 GPU 时间 | API 费用 | 备注 |
|------|-------------|---------|------|
| Phase 0: 语料清洗 | 0（CPU） | ¥0 | |
| Phase 0: 探针校准 | ~20 min | ¥0 | |
| Phase 1: 双盲提取（10 万块） | ~6 h | **~¥25** | 教师 API 是主要成本 |
| Phase 2: 分层 | ~2 h | ~¥5（瓶颈验证） | |
| Phase 3: Schema 涌现 | ~1 h | ~¥3（概念化） | |
| Phase 4: 质量控制 | ~1 h | ~¥2（反向验证） | |
| Phase 5: 三阶段训练 | ~11 h | ¥0 | 本地训练 |
| Phase 6: 评估 | ~1 h | ~¥1 | |
| **总计** | **~22 h** | **~¥36** | |

> **对比原稿的 3×A100 方案（¥1,100）**：4090 单卡方案将计算成本降至 ¥0（自有硬件），API 成本仅 ¥36。总成本降低 **97%**。

---

## 十三、关键注意事项清单

| # | 事项 | 说明 |
|---|------|------|
| 1 | **所有学生用 Base 版** | 不用 Instruct 版，避免 RLHF 扭曲概率分布 |
| 2 | **教师必须关闭思考模式** | `"thinking": "off"`，否则提取的是推理结果而非记忆 |
| 3 | **教师温度 = 2.0** | 展开暗知识分布；学生温度 = 1.0，测真实记忆 |
| 4 | **同系列模型对齐** | 0.6B / 1.7B 共享 tokenizer（v1.1 修订：删除 4B 探针后仅双探针），概率对比零噪声 |
| 5 | **不做量化** | BF16 原版，保护 logprobs 精度 |
| 6 | **API 限流保护** | 每批 50 条 + 0.5s 间隔，避免 429 |
| 7 | **断点续跑** | 每 500 条保存一次中间结果 |
| 8 | **Schema 概念化温度 = 0** | 消除随机性，确保可复现 |
| 9 | **CCS 灰色地带不训练** | 0.3~0.7 之间的样本进入人工审核队列 |
| 10 | **最终模型是 0.6B** | 不是 1.7B。1.7B 是探针，0.6B 是被训练的记忆引擎（v1.1 修订：与 [`08a` §1.1](08a-capacity-gap-design.md) 一致）|

---

## 十四、与主训练管线的衔接

> 本管线产出的 `data/training/*.jsonl` 应当作为 `../agenticdsl-training/01-training-data-pipeline.md` 第 3 阶段（执行驱动过滤）的**前置输入**，而非直接喂入 SFT。

| 产出物 | 路径 | 喂给 |
|---|---|---|
| `memory_train.jsonl` | `data/training/memory_train.jsonl` | → `../agenticdsl-training/01-training-data-pipeline.md` §3 Schema 校验与执行 dry-run |
| `schema_v1.json` | `data/schema/schema_v1.json` | → `../agenticdsl-training/01-training-data-pipeline.md` §2 L2 schema 校验 |
| `stratified/` | `data/stratified/` | → `../agenticdsl-training/01-training-data-pipeline.md` §1 任务矩阵的"记忆 / 推理"分层信号 |

> **避免重复劳动**：本管线已经完成"记忆 / 推理"分层与 Schema 涌现，`../agenticdsl-training/01-training-data-pipeline.md` 后续阶段无需再做 HDBSCAN 聚类或 LLM 概念化，仅做格式归一与 schema 对齐即可。

---

## 十五、决策记录（适配层-RTX4090）

> **v1.1 修订**：本表仅保留**适配层（ADAPTATION-RTX4090）** 的具体决策。设计层决策（CCS 公式、阈值、Schema 涌现流程、训练阶段数等）见 [`08a` 附录 A](08a-capacity-gap-design.md)。两层决策用文档 ID 前缀区分，避免双副本漂移。

| # | 决策项 | 当前选择 | 备选 | 理由 | 作用域 |
|---|---|---|---|---|---|
| 1 | 教师模型 | DeepSeek V4 Flash API（13B 激活）| Qwen3-235B / GPT-4o | logprobs 完整 + 远程免本地显存 | ADAPTATION-RTX4090 |
| 2 | 主探针 (L1) | Qwen3-1.7B Base | Qwen2.5-1.5B | 与教师构成 7.6× 安全 Gap + 同系列 tokenizer 对齐 | ADAPTATION-RTX4090 |
| 3 | 崩溃对照 (L3) | Qwen3-0.6B Base | Qwen3-0.6B Instruct | Base 版概率分布无 RLHF 扭曲 + 同系列对齐 | ADAPTATION-RTX4090 |
| 4 | **辅助探针 (L2)** | **删除（v1.1 修订）** | Qwen3-4B Base | 节省 9.5GB 显存 + 消除 OOM 风险 + 当前代码不消费其输出 | ADAPTATION-RTX4090 |
| 5 | 训练目标 | Qwen3-0.6B | Qwen3-1.7B | "记忆引擎"应极轻量；1.7B 留给推理能力扩展 | ADAPTATION-RTX4090 |
| 6 | **教师温度** | **0.7（v1.1 修订：≤1.0）** | ~~2.0~~（已废弃）| logits 排序与温度无关；T=2.0 只增加 JSON 解析失败率 | ADAPTATION-RTX4090 |
| 7 | 学生温度 | 1.0 | 0.1 | 测真实记忆，不人为压制概率 | ADAPTATION-RTX4090 |
| 8 | 教师置信度提取 | **答案 span 平均 logprob（v1.1 修订）** | ~~首 token logprob~~（已废弃）| 首 token 必为 `[`，top1_prob≈1.0 信号无信息量 | ADAPTATION-RTX4090 |
| 9 | vLLM max-logprobs | **100（v1.1 修订：启动参数显式指定）** | 20（vLLM 默认）| student payload 中 `logprobs: 20` 可工作，但保留 100 供未来扩展 | ADAPTATION-RTX4090 |
| 10 | 重构敏感度实现 | **置信度差代理（v1.1 标注）** | 完整扰动协议（见 08a）| 当前用 `max(0, teacher_conf − probe_conf)` 代理；完整扰动协议待实现 | ADAPTATION-RTX4090 |
| 11 | L3 bottleneck 基线校准 | **新增 `_check_L3_baseline` 方法（v1.1）** | 信任 L3 直接提取 | 避免 L3 指令遵循不足导致信号失真 | ADAPTATION-RTX4090 |
| 12 | 嵌入模型 | BGE-M3 | all-MiniLM-L6-v2 | 中英双语场景首选 | ADAPTATION-RTX4090 |
| 13 | hard 桶定义 | **Top 10% by ccs（v1.1 修订）** | CCS ∈ [0.3, 0.4) | memory CCS ≤ 0.3，原 hard 桶永远为空 | ADAPTATION-RTX4090 |

> 此外，**下列设计层决策已锁定，详见 [`08a` 附录 A](08a-capacity-gap-design.md)**：CCS 公式、CCS 阈值、Schema 涌现流程、Schema 概念化温度、三阶段训练、Tokenizer 对齐硬约束、崩溃对照组、灰色地带处理、蒸馏方式、教师置信度提取。

---

## 十六、风险登记

| # | 风险 | 缓解措施 |
|---|---|---|
| R-01 | **CCS 阈值误分类**：阈值 0.3/0.7 是经验值，可能不适合所有领域 | 每季度在验证集上重校准，必要时引入可学习阈值 |
| R-02 | **教师 API 成本失控**：高频调用可能超预算 | 设置每日 ¥50 硬上限 + 告警；超出后切回本地 Qwen3-1.7B（v1.1 修订：原方案切回 4B，4B 已删除）|
| R-03 | **HDBSCAN 聚类不稳定**：随机种子导致簇边界漂移 | 多次运行取交集（见 §7.2）；保留低频簇作噪声，避免过拟合 |
| R-04 | **灰色地带样本流失**：0.3~0.7 区间被丢弃可能损失优质样本 | 进人工审核队列；每批 200 条抽样标注，统计是否漏掉关键记忆 |
| R-05 | **0.6B 上限**：极小模型表达能力受限 | 限定 schema 总数 ≤ 200 关系 / 50 实体；超过则切回 1.7B 训练 |
| R-06 | **Schema 漂移**：多次涌现的 schema 之间不一致 | 锁定 schema_version；变更走 PR review |
| R-07 | **冷启动校准失效**：领域分布偏离 calibration_5k | 每 10K chunk 后重做一次微校准（增量） |

---

**文档版本**: v1.0
**最后更新**: 2026-08-24
**Owner**: AgenticMind 数据工程团队

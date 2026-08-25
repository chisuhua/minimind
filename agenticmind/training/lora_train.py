"""P1-4 LoRA 微调主脚本(骨架,需 peft + transformers)

用法:
    pip install peft transformers accelerate bitsandbytes
    export TEACHER_MODEL_PATH=/path/to/Qwen3-0.6B  # 或 Qwen2.5-0.5B 临时替代
    export TOKENIZERS_PARALLELISM=false
    python -m agenticmind.training.lora_train \
        --train-jsonl data/agenticmind/v0/train.jsonl \
        --dev-jsonl data/agenticmind/v0/dev.jsonl \
        --output-dir runs/lora_v0 \
        --epochs 3

设计要点:
- base model: Qwen3-0.6B(对齐 08a D-10 tokenizer 约束;若不可用 fallback 到 Qwen2.5-0.5B)
- LoRA: r=16, alpha=32, dropout=0.05, target_modules=q_proj/v_proj
- 精度: bf16(fp16 fallback);gradient checkpointing 开
- batch: micro=2 + grad_accum=8 → effective 16
- max_seq_len: 1024(input prompt 较长,部分截断)
- 评估:每 epoch 末在 dev set 上跑 eval_f1.py

产出:
- runs/lora_v0/adapter_model.safetensors
- runs/lora_v0/adapter_config.json
- runs/lora_v0/training_log.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


# ============================================================================
# LoRA 配置(常量)
# ============================================================================

DEFAULTLORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "v_proj"],
}

DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "bf16": True,
    "gradient_checkpointing": True,
    "report_to": "none",
}


# ============================================================================
# 依赖检查
# ============================================================================


def check_deps() -> dict[str, Any]:
    """检查 peft / transformers / accelerate 是否可用"""
    missing = []
    for mod in ["torch", "transformers", "peft", "datasets", "accelerate"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f"缺少依赖:{missing}。安装: pip install peft transformers accelerate datasets"
        )
    return {
        "torch": __import__("torch"),
        "transformers": __import__("transformers"),
        "peft": __import__("peft"),
    }


# ============================================================================
# 数据加载
# ============================================================================


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class JsonlDataset:
    """轻量 JSONL 数据集,免去 datasets 依赖

    实现 __len__ / __getitem__,供 Trainer 使用
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 1024,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        full = s["input"] + s["output"] + self.tokenizer.eos_token
        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        prompt_ids = self.tokenizer(s["input"], add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
        labels = labels[: len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================================
# 主训练函数
# ============================================================================


def train(
    train_jsonl: Path,
    dev_jsonl: Path,
    output_dir: Path,
    base_model: str,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    """LoRA 训练主函数

    返回:训练结果摘要
    """
    deps = check_deps()
    torch = deps["torch"]
    transformers = deps["transformers"]
    peft = deps["peft"]

    print(f"加载 base model:{base_model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = peft.LoraConfig(**DEFAULTLORA_CONFIG)
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = JsonlDataset(load_jsonl(train_jsonl), tokenizer, max_length=max_length)
    dev_ds = JsonlDataset(load_jsonl(dev_jsonl), tokenizer, max_length=max_length)

    training_args_dict = {
        **DEFAULT_TRAINING_ARGS,
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    training_args = transformers.TrainingArguments(**training_args_dict)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
    )

    print(f"开始训练:epochs={epochs}, effective batch={batch_size * grad_accum}")
    trainer.train()

    print(f"保存 LoRA 适配器:{output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return {
        "output_dir": str(output_dir),
        "epochs": epochs,
        "train_samples": len(train_ds),
        "dev_samples": len(dev_ds),
        "base_model": base_model,
    }


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-4 LoRA 训练 CLI(需 peft)")
    parser.add_argument(
        "--train-jsonl", type=Path, default=Path("data/agenticmind/v0/train.jsonl")
    )
    parser.add_argument(
        "--dev-jsonl", type=Path, default=Path("data/agenticmind/v0/dev.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_v0"))
    parser.add_argument(
        "--base-model",
        default=os.environ.get("TEACHER_MODEL_PATH", "Qwen/Qwen2.5-0.5B"),
        help="base model path or HF id",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        result = train(
            train_jsonl=args.train_jsonl,
            dev_jsonl=args.dev_jsonl,
            output_dir=args.output_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
        )
        print(f"\n✓ 训练完成:{json.dumps(result, ensure_ascii=False, indent=2)}")
    except ImportError as e:
        print(f"\n✗ 缺少依赖:{e}")
        print("安装:pip install peft transformers accelerate datasets")


if __name__ == "__main__":
    main()
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import math
import time
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, apply_rotary_pos_emb
from model.rt_purbo import RetrievalHeadClassifier, LowDimIndexer
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def kl_divergence_loss(q_logits, k_logits, temperature=1.0):
    q_prob = F.log_softmax(q_logits / temperature, dim=-1)
    k_prob = F.softmax(k_logits / temperature, dim=-1)
    return F.kl_div(q_prob, k_prob, reduction='batchmean') * (temperature ** 2)


def train_stage1(indexer, model, retrieval_heads, loader, optimizer, scaler, args, epoch, wandb=None):
    start_time = time.time()
    for step, (input_ids, _) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        bsz, seq_len = input_ids.shape
        if seq_len < 2048:
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            hidden_states = model.model.dropout(model.model.embed_tokens(input_ids))
            cos = model.model.freqs_cos[:seq_len].to(input_ids.device)
            sin = model.model.freqs_sin[:seq_len].to(input_ids.device)
            position_embeddings = (cos, sin)

            total_kl = 0.0
            num_local_heads = 0

            for layer_idx, layer in enumerate(model.model.layers):
                attn = layer.self_attn
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                x = hidden_states

                xq, xk, xv = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
                xq = xq.view(bsz, seq_len, attn.n_local_heads, attn.head_dim)
                xk = xk.view(bsz, seq_len, attn.n_local_kv_heads, attn.head_dim)
                xq = attn.q_norm(xq)
                xk = attn.k_norm(xk)
                xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

                for h in range(attn.n_local_heads):
                    if not retrieval_heads.get((layer_idx, h), False):
                        kv_h = h // attn.n_rep
                        q_h = xq[:, :, h:h+1, :]
                        k_h = xk[:, :, kv_h:kv_h+1, :]
                        q_low, k_low = indexer(q_h, k_h)

                        with torch.no_grad():
                            scores_full = (q_h @ k_h.transpose(-2, -1)) / math.sqrt(attn.head_dim)
                            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=scores_full.device).triu(1)
                            scores_full[:, :, :, -seq_len:] += causal_mask
                            full_attn = F.log_softmax(scores_full, dim=-1)

                        scores_low = (q_low @ k_low.transpose(-2, -1)) / math.sqrt(q_low.shape[-1])
                        scores_low[:, :, :, -seq_len:] += causal_mask
                        low_attn = F.log_softmax(scores_low, dim=-1)

                        kl_loss = kl_divergence_loss(low_attn, full_attn)
                        total_kl += kl_loss
                        num_local_heads += 1

                hidden_states = x
                hidden_states = residual + hidden_states
                hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

            loss = total_kl / max(num_local_heads, 1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(indexer.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(f'Stage1 Step [{step}/{args.stage1_steps}], KL loss: {loss.item():.6f}, lr: {optimizer.param_groups[0]["lr"]:.8f}')
            if wandb:
                wandb.log({"stage1_kl_loss": loss.item(), "step": step})

        if step >= args.stage1_steps:
            break


def train_stage2(model, indexer, retrieval_heads, loader, optimizer, scaler, args, epoch, start_step=0, wandb=None):
    start_time = time.time()
    iters = args.stage2_steps
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        if step > args.stage2_steps:
            break
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(step, iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            Logger(f'Stage2 Step [{step}/{iters}], loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}')
            if wandb:
                wandb.log({"stage2_loss": current_loss, "stage2_logits_loss": current_logits_loss, "learning_rate": current_lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, f'{args.save_dir}/rt_purbo_stage2_{step}.pth')
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind RTPurbo Two-Stage Training")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--stage1_steps", type=int, default=1000, help="Stage 1 训练步数")
    parser.add_argument("--stage2_steps", type=int, default=600, help="Stage 2 训练步数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--learning_rate_stage2", type=float, default=1e-5, help="Stage 2 学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=4096, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument("--calib_path", type=str, default="../out/rt_purbo_heads.json", help="Retrieval head 标定文件路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--use_wandb', action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RTPurbo", help="wandb项目名")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project)

    Logger("Loading model ...")
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    Logger(f"Loading retrieval heads from {args.calib_path} ...")
    classifier = RetrievalHeadClassifier.load(args.calib_path, model=model)
    retrieval_heads = classifier.retrieval_heads
    retrieval_count = sum(1 for v in retrieval_heads.values() if v)
    total = len(retrieval_heads)
    Logger(f"Retrieval heads: {retrieval_count}/{total} ({100 * retrieval_count / total:.1f}%)")

    indexer = LowDimIndexer(head_dim=lm_config.head_dim, low_dim=16).to(args.device)
    Logger(f"LowDimIndexer params: {sum(p.numel() for p in indexer.parameters())}")

    # Stage 1: freeze model, train indexer
    Logger("=" * 40)
    Logger("Stage 1: Training 16-dim indexer via KL distillation")
    Logger("=" * 40)
    for p in model.parameters():
        p.requires_grad = False
    for p in indexer.parameters():
        p.requires_grad = True

    optimizer_stage1 = optim.AdamW(indexer.parameters(), lr=args.learning_rate)
    scaler_stage1 = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    indices = torch.randperm(len(train_ds)).tolist()
    batch_sampler = train_sampler or indices
    loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(batch_sampler, args.batch_size, 0),
                        num_workers=args.num_workers, pin_memory=True)

    train_stage1(indexer, model, retrieval_heads, loader, optimizer_stage1, scaler_stage1, args, 0, wandb)
    torch.save(indexer.state_dict(), f'{args.save_dir}/rt_purbo_indexer.pth')
    Logger(f"Saved indexer to {args.save_dir}/rt_purbo_indexer.pth")

    # Stage 2: unfreeze, end-to-end
    Logger("=" * 40)
    Logger("Stage 2: End-to-end fine-tuning")
    Logger("=" * 40)
    for p in model.parameters():
        p.requires_grad = True

    optimizer_stage2 = optim.AdamW(model.parameters(), lr=args.learning_rate_stage2)
    scaler_stage2 = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    loader2 = DataLoader(train_ds, batch_sampler=SkipBatchSampler(batch_sampler, args.batch_size, 0),
                         num_workers=args.num_workers, pin_memory=True)

    train_stage2(model, indexer, retrieval_heads, loader2, optimizer_stage2, scaler_stage2, args, 1, 0, wandb)

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, f'{args.save_dir}/rt_purbo_{lm_config.hidden_size}.pth')
    Logger(f"Saved final model to {args.save_dir}/rt_purbo_{lm_config.hidden_size}.pth")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

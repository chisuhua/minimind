
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model.model_minimind import MiniMindConfig
from model.model_minimind_hybrid import HybridMiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}.pth'
            raw_model = getattr(model, '_orig_mod', model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            Logger(f'Model saved to {ckp}')
            model.train()
            del state_dict

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-Next Hybrid Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='hybrid', type=str)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument('--head_dim', default=128, type=int)
    parser.add_argument('--partial_rope_dim', default=32, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument('--from_weight', default='none', type=str)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Hybrid")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    setup_seed(42)

    os.makedirs(args.save_dir, exist_ok=True)

    hybrid_pattern = ['d'] * 6 + ['a'] * 2
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        head_dim=args.head_dim,
        partial_rope_dim=args.partial_rope_dim,
        hybrid_pattern=hybrid_pattern,
    )

    Logger(f'Hybrid pattern: {hybrid_pattern} ({sum(1 for t in hybrid_pattern if t == "d")} DeltaNet + {sum(1 for t in hybrid_pattern if t == "a")} GatedAttention)')

    tokenizer = None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model')

    model = HybridMiniMindForCausalLM(lm_config)

    if args.from_weight != 'none':
        weight_path = f'{args.save_dir}/{args.from_weight}_{args.hidden_size}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=args.device)
            model.load_state_dict(weights, strict=False)
            Logger(f'Loaded partial weights from {weight_path}')
        else:
            Logger(f'Warning: {weight_path} not found, training from scratch')

    total = sum(p.numel() for p in model.parameters()) / 1e6
    Logger(f'Model Params: {total:.2f}M')
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable Params: {n_trainable:.3f}M')

    model = model.to(args.device)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        train_epoch(epoch, loader, len(loader), 0)

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets
import argparse
import time
import warnings
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed

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
    parser = argparse.ArgumentParser(description="Gated DeltaNet PoC Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='gated_deltanet_poc', type=str)
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
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument('--replace_layer', type=int, default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Gated-DeltaNet")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    setup_seed(42)

    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        gated_deltanet_layers=[args.replace_layer],
    )

    from trainer.trainer_utils import init_model as _init_utils_model
    model, tokenizer = _init_utils_model(lm_config, args.from_weight, device=args.device)

    trainable_params = []
    for name, param in model.named_parameters():
        if f"layers.{args.replace_layer}.self_attn" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in trainable_params) / 1e6
    Logger(f'Trainable Params: {n_trainable:.3f}M (layer {args.replace_layer} GatedDeltaNet only)')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        train_epoch(epoch, loader, len(loader), 0)

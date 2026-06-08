import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed

warnings.filterwarnings('ignore')


def kl_div_loss(student_logits, teacher_logits, temperature=4.0):
    """KL divergence between teacher and student output distributions"""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits.detach() / temperature, dim=-1),
        reduction='batchmean',
    ) * (temperature ** 2)


def train_stage1(loader, iters, start_step=0):
    """Stage 1: freeze backbone, train indexer only with KL loss"""
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(step, iters, args.stage1_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.no_grad():
            teacher_out = teacher(input_ids, labels=labels)
            teacher_logits = teacher_out.logits

        student_out = model(input_ids, labels=labels)
        student_logits = student_out.logits

        kl_loss = kl_div_loss(student_logits, teacher_logits)
        loss = kl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Stage1 [{step}/{iters}], loss: {current_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_stage1_{args.hidden_size}.pth'
            raw_model = getattr(model, '_orig_mod', model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            Logger(f'Stage1 model saved to {ckp}')
            model.train()
            del state_dict

    return last_step


def train_stage2(loader, iters, start_step=0):
    """Stage 2: unfreeze all, end-to-end training"""
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(step, iters, args.stage2_lr)
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
            Logger(f'Stage2 [{step}/{iters}], loss: {current_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}.pth'
            raw_model = getattr(model, '_orig_mod', model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            Logger(f'Model saved to {ckp}')
            model.train()
            del state_dict

    return last_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightning Indexer PoC Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='lightning_indexer_poc', type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument('--replace_layer', type=int, default=0)
    parser.add_argument("--stage1_lr", type=float, default=5e-4)
    parser.add_argument("--stage2_lr", type=float, default=1e-5)
    parser.add_argument("--stage1_steps", type=int, default=1000)
    parser.add_argument("--stage2_steps", type=int, default=600)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Lightning-Indexer")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    setup_seed(42)

    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        lightning_indexer_layers=[args.replace_layer],
    )

    from trainer.trainer_utils import init_model as _init_utils_model

    # Load student model with LightningIndexer
    model, tokenizer = _init_utils_model(lm_config, args.from_weight, device=args.device)

    # Load teacher model (frozen backbone, standard attention)
    teacher_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    teacher, _ = _init_utils_model(teacher_config, args.from_weight, device=args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Stage 1: freeze all, train indexer only
    trainable_params = []
    for name, param in model.named_parameters():
        if 'indexer' in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in trainable_params) / 1e6
    Logger(f'Stage1 Trainable Params: {n_trainable:.3f}M (indexer only)')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    optimizer = optim.AdamW(trainable_params, lr=args.stage1_lr)

    Logger(f'=== Stage 1: Training indexer only ({args.stage1_steps} steps) ===')
    train_stage1(loader, args.stage1_steps, 0)

    # Stage 2: unfreeze all, end-to-end
    for name, param in model.named_parameters():
        if 'indexer' in name:
            param.requires_grad = True
        else:
            param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=args.stage2_lr)

    Logger(f'=== Stage 2: End-to-end training ({args.stage2_steps} steps) ===')
    train_stage2(loader, args.stage2_steps, 0)

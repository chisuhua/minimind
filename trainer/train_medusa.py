import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from model.model_minimind import MiniMindConfig
from model.medusa_heads import MedusaHeads
from dataset.lm_dataset import SFTDataset
from transformers import AutoTokenizer
from trainer.trainer_utils import Logger, setup_seed, get_model_params, is_main_process


def train_epoch(epoch, loader, medusa_heads, model, optimizer, args):
    model.eval()
    medusa_heads.train()
    total_loss = 0
    start_time = time.time()

    for step, (input_ids, labels) in enumerate(loader):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        bsz, seq_len = input_ids.shape

        with torch.no_grad():
            hidden_states, _, _ = model.model(input_ids, None, None, False)
            backbone_logits = model.lm_head(hidden_states)
            backbone_preds = backbone_logits.argmax(dim=-1)

        loss = 0
        for k in range(args.medusa_heads):
            offset = k + 1
            if offset >= seq_len:
                continue
            head_logits = medusa_heads.heads[k](hidden_states[:, :seq_len - offset])
            target = backbone_preds[:, offset:]
            padding_mask = labels[:, offset:] == -100
            target = target.masked_fill(padding_mask, -100)
            loss += F.cross_entropy(
                head_logits.reshape(-1, head_logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100,
            )

        loss.backward()
        nn.utils.clip_grad_norm_(medusa_heads.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % args.log_interval == 0:
            spend = time.time() - start_time
            eta = spend / max(step, 1) * (len(loader) - step) // 60
            Logger(f'Epoch [{epoch + 1}/{args.epochs}] Step {step}/{len(loader)} Loss: {loss.item():.4f} ETA: {eta:.0f}min')

    return total_loss / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Medusa-1 Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--medusa_heads", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--backbone", type=str, default="full_sft")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    setup_seed(42)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        medusa_heads=args.medusa_heads,
    )
    tokenizer = AutoTokenizer.from_pretrained('../model')

    from model.model_minimind import MiniMindForCausalLM
    model = MiniMindForCausalLM(lm_config)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    backbone_path = f'{args.save_dir}/{args.backbone}_{args.hidden_size}{moe_suffix}.pth'
    Logger(f'Loading backbone from {backbone_path}')
    weights = torch.load(backbone_path, map_location=args.device)
    model.load_state_dict(weights, strict=False)
    model = model.to(args.device).half()

    for param in model.parameters():
        param.requires_grad = False
    get_model_params(model, lm_config)

    medusa_heads = MedusaHeads(lm_config).to(args.device)
    Logger(f'Medusa heads params: {sum(p.numel() for p in medusa_heads.parameters() if p.requires_grad) / 1e6:.3f}M')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.AdamW(medusa_heads.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        avg_loss = train_epoch(epoch, loader, medusa_heads, model, optimizer, args)
        Logger(f'Epoch [{epoch + 1}/{args.epochs}] Avg Loss: {avg_loss:.4f}')

    save_path = f'{args.save_dir}/medusa_{args.medusa_heads}_{args.hidden_size}{moe_suffix}.pth'
    torch.save(medusa_heads.state_dict(), save_path)
    Logger(f'Saved medusa heads to {save_path}')

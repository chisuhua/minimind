import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import json
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.dflash import DFlashDraftModel, FeatureFusion
from dataset.lm_dataset import SFTDataset
from transformers import AutoTokenizer
from trainer.trainer_utils import Logger, setup_seed, get_model_params, is_main_process


def extract_calibration_data(model, loader, fusion_layers, device):
    """Stage 1: Extract target hidden states for calibration"""
    model.eval()
    calib_data = []
    with torch.no_grad():
        for step, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            hidden_list = []

            def hook_fn(layer_id):
                def hook(module, input, output):
                    hidden_list.append(output[0].detach().cpu())
                return hook

            hooks = []
            for lid in fusion_layers:
                h = model.model.layers[lid].register_forward_hook(hook_fn(lid))
                hooks.append(h)

            model(input_ids, use_cache=False)
            for h in hooks:
                h.remove()

            calib_data.append((input_ids.cpu(), hidden_list))
            if step > 200:
                break
    return calib_data


def train_drafter_stage2(draft_model, target_model, calib_data, args):
    """Stage 2: Train drafter with block diffusion loss"""
    draft_model.train()
    target_model.eval()
    optimizer = optim.AdamW(draft_model.parameters(), lr=args.lr_stage2, weight_decay=0.01)
    total_loss = 0
    start_time = time.time()

    for epoch in range(args.epochs_stage2):
        for step, (input_ids, hidden_list) in enumerate(calib_data):
            input_ids = input_ids.to(args.device)
            hidden_list = [h.to(args.device) for h in hidden_list]
            B, T = input_ids.shape

            loss = 0
            for t in range(1, min(T - 1, args.max_seq_len)):
                anchor_pos = t - 1
                target_pos = t

                anchor_emb = target_model.model.embed_tokens(input_ids[:, anchor_pos:anchor_pos+1])
                mask_emb = target_model.model.embed_tokens(
                    torch.full((B, args.dflash_block_size - 1), 3, device=args.device)
                )
                block_input = torch.cat([anchor_emb, mask_emb], dim=1)

                fusion_hidden = [h[:, anchor_pos:anchor_pos+1] for h in hidden_list]
                g_t = draft_model.feature_fusion(fusion_hidden).squeeze(1)
                pred = draft_model(block_input, g_t=g_t)
                logits = target_model.lm_head(pred)

                target_token = input_ids[:, target_pos]
                loss += F.cross_entropy(logits[:, 0], target_token)

            loss.backward()
            nn.utils.clip_grad_norm_(draft_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % args.log_interval == 0:
                spend = time.time() - start_time
                Logger(f'Stage2 Epoch [{epoch+1}/{args.epochs_stage2}] Step {step}/{len(calib_data)} Loss: {loss.item():.4f}')

    return total_loss / max(len(calib_data), 1)


def train_e2e_stage3(draft_model, target_model, loader, args):
    """Stage 3: End-to-end fine-tune"""
    draft_model.train()
    target_model.eval()
    optimizer = optim.AdamW(draft_model.parameters(), lr=args.lr_stage3, weight_decay=0.01)
    total_loss = 0
    start_time = time.time()

    for epoch in range(args.epochs_stage3):
        for step, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            B, T = input_ids.shape

            hidden_list = []
            def hook_fn(layer_id):
                def hook(module, input, output):
                    hidden_list.append(output[0].detach())
                return hook

            hooks = []
            fusion_layers = args.fusion_layers or [1, 3, 5, 7]
            for lid in fusion_layers:
                h = target_model.model.layers[lid].register_forward_hook(hook_fn(lid))
                hooks.append(h)

            with torch.no_grad():
                target_model(input_ids, use_cache=False)
            for h in hooks:
                h.remove()

            loss = 0
            for t in range(1, min(T - 1, args.max_seq_len)):
                anchor_pos = t - 1
                fusion_hidden = [h[:, anchor_pos:anchor_pos+1] for h in hidden_list]
                g_t = draft_model.feature_fusion(fusion_hidden).squeeze(1)

                anchor_emb = target_model.model.embed_tokens(input_ids[:, anchor_pos:anchor_pos+1])
                mask_emb = target_model.model.embed_tokens(
                    torch.full((B, args.dflash_block_size - 1), 3, device=args.device)
                )
                block_input = torch.cat([anchor_emb, mask_emb], dim=1)
                pred = draft_model(block_input, g_t=g_t)
                logits = target_model.lm_head(pred)

                target_token = input_ids[:, t]
                loss += F.cross_entropy(logits[:, 0], target_token)

            loss.backward()
            nn.utils.clip_grad_norm_(draft_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % args.log_interval == 0:
                spend = time.time() - start_time
                Logger(f'Stage3 Epoch [{epoch+1}/{args.epochs_stage3}] Step {step}/{len(loader)} Loss: {loss.item():.4f}')

    return total_loss / max(len(loader), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DFlash Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--backbone", type=str, default="full_sft")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--dflash_block_size", type=int, default=16)
    parser.add_argument("--fusion_layers", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--lr_stage2", type=float, default=1e-3)
    parser.add_argument("--lr_stage3", type=float, default=5e-5)
    parser.add_argument("--epochs_stage2", type=int, default=3)
    parser.add_argument("--epochs_stage3", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    setup_seed(42)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        use_dflash=True,
        dflash_block_size=args.dflash_block_size,
        dflash_fusion_layers=args.fusion_layers,
    )
    target_model = MiniMindForCausalLM(config)
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'{args.save_dir}/{args.backbone}_{args.hidden_size}{moe_suffix}.pth'
    target_model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    target_model = target_model.half().eval().to(args.device)

    draft_model = DFlashDraftModel(config, block_size=args.dflash_block_size).to(args.device)
    get_model_params(draft_model, config)

    tokenizer = AutoTokenizer.from_pretrained('../model')
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    Logger("Stage 1: Extracting calibration data...")
    calib_data = extract_calibration_data(target_model, loader, args.fusion_layers, args.device)

    Logger("Stage 2: Training drafter...")
    loss2 = train_drafter_stage2(draft_model, target_model, calib_data, args)
    Logger(f"Stage 2 complete, avg loss: {loss2:.4f}")

    Logger("Stage 3: End-to-end fine-tuning...")
    loss3 = train_e2e_stage3(draft_model, target_model, loader, args)
    Logger(f"Stage 3 complete, avg loss: {loss3:.4f}")

    save_path = f'{args.save_dir}/dflash_{args.dflash_block_size}_{args.hidden_size}{moe_suffix}.pth'
    torch.save(draft_model.state_dict(), save_path)
    Logger(f"DFlash draft model saved to {save_path}")

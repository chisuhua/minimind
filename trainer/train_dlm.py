import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from model.model_minimind import MiniMindConfig
from model.diffusion_lm import DiffusionMiniMind, MaskedDiffusionScheduler
from dataset.lm_dataset import PretrainDataset
from transformers import AutoTokenizer
from trainer.trainer_utils import Logger, setup_seed


def main():
    parser = argparse.ArgumentParser(description="MiniMind dLM Diffusion Language Model Training")
    parser.add_argument('--epochs', default=2, type=int, help="训练轮数")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size")
    parser.add_argument('--learning_rate', default=3e-4, type=float, help="学习率")
    parser.add_argument('--max_seq_len', default=340, type=int, help="最大序列长度")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="训练设备")
    parser.add_argument('--data_path', default='../dataset/pretrain_t2t_mini.jsonl', type=str, help="训练数据路径")
    parser.add_argument('--save_dir', default='../out', type=str, help="模型保存目录")
    parser.add_argument('--diffusion_steps', default=100, type=int, help="扩散步数")
    parser.add_argument('--log_interval', default=100, type=int, help="日志打印间隔")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    args = parser.parse_args()

    setup_seed(42)
    device = args.device

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = DiffusionMiniMind(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained('../model')
    scheduler = MaskedDiffusionScheduler(num_steps=args.diffusion_steps, mask_token_id=config.vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    Logger(f'DiffusionMiniMind params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    for epoch in range(args.epochs):
        for step, (input_ids, _) in enumerate(loader):
            input_ids = input_ids.to(device)
            B = input_ids.shape[0]

            t = torch.rand(B, device=device)
            noisy_ids, mask = scheduler.add_noise(input_ids, t)
            logits = model(noisy_ids, t)

            loss = F.cross_entropy(logits[mask], input_ids[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_interval == 0:
                Logger(f'Epoch [{epoch + 1}/{args.epochs}] Step {step} Loss {loss.item():.4f}')

        moe_suffix = '_moe' if config.use_moe else ''
        ckp = f'{args.save_dir}/minimind-3-dlm_{args.hidden_size}{moe_suffix}.pth'
        torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, ckp)
        Logger(f'Saved to {ckp}')


if __name__ == '__main__':
    main()

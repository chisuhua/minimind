import os, sys, math, json, torch, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.nsa import replace_attention_with_nsa

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len=512, size=100):
        self.data = [
            "中国的首都是北京。",
            "机器学习是人工智能的重要分支。",
            "Transformer使用自注意力机制建模上下文关系。",
            "上海是中国最大的城市之一。",
            "深度学习推动了人工智能的快速发展。",
            "自然语言处理是计算机科学的重要领域。",
            "太阳系有八大行星。",
            "Python是一种广泛使用的编程语言。",
            "神经网络由大量神经元相互连接而成。",
            "数据是人工智能的基础。",
            "MiniMind是一个极小的语言模型项目。",
            "注意力机制是Transformer的核心创新。",
            "预训练加微调是目前大模型的主流范式。",
            "梯度下降是训练神经网络的基本算法。",
            "反向传播算法计算损失函数对权重的梯度。",
        ] * (size // 15 + 1)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding="max_length")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        return input_ids, labels

def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return input_ids, labels

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("../model/minimind_tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MiniMindConfig(use_nsa=True, nsa_block_l=32, nsa_sliding_w=512, nsa_top_n=16)
    model = MiniMindForCausalLM(config)
    model = replace_attention_with_nsa(model, config)
    model = model.to(device)

    dataset = SimpleDataset(tokenizer, max_seq_len=128, size=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()

    for epoch in range(3):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")

    save_dir = "../out"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "nsa_768.pth")
    torch.save(model.state_dict(), save_path)
    print(f"NSA model saved to {save_path}")

if __name__ == "__main__":
    train()

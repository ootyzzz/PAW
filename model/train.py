"""
train.py
训练主循环，完成 prompt batch 与 LoRA checkpoint 的配对训练
"""
import torch
import torch.optim as optim
from utils.embedding import get_encoder, embed_batch
from utils.metrics import mse_loss
from utils.scheduler import get_scheduler
from lora.flatten import flatten_lora_weights
from lora.checkpoint_utils import load_checkpoint
from model.hyperconv_decoder import HyperConvDecoder
import random

# 构建 same-task, random-pair 配对
def build_pairs(prompt_batches, lora_checkpoints):
    """
    随机将每个 prompt batch 与同数据集的 LoRA checkpoint 配对
    """
    pairs = []
    for batch in prompt_batches:
        ckpt = random.choice(lora_checkpoints)
        pairs.append((batch, ckpt))
    return pairs

def train_loop(prompt_batches, lora_checkpoints, epochs=100, batch_size=32):
    """
    主训练循环
    prompt_batches: List[List[str]]
    lora_checkpoints: List[str]
    """
    encoder = get_encoder()
    model = HyperConvDecoder()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = get_scheduler(optimizer, 'cosine', epochs)
    pairs = build_pairs(prompt_batches, lora_checkpoints)
    best_loss = float('inf')
    for epoch in range(epochs):
        random.shuffle(pairs)
        for batch, ckpt_path in pairs:
            emb = embed_batch(batch, encoder)  # (8, 384, 384)
            lora_model = load_checkpoint(model, ckpt_path)
            target = flatten_lora_weights(lora_model)
            pred = model(emb)
            loss = mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        # 保存最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f'best_model.pt')

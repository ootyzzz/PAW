"""
embedding.py
用于获取 MiniLM-L6-v2 文本编码器，并对 prompt batch 进行 embedding。
"""
from sentence_transformers import SentenceTransformer
import torch

def get_encoder():
    """
    加载 MiniLM-L6-v2 文本编码器
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_batch(batch, encoder):
    """
    对 batch（prompt string 列表）进行 embedding，输出 shape (batch_size, 384, 384)
    """
    emb = encoder.encode(batch, convert_to_tensor=True)  # (batch_size, 384)
    # pad/truncate to (batch_size, 384, 384)
    if emb.dim() == 2:
        emb = emb.unsqueeze(1).expand(-1, 384, -1)
    return emb

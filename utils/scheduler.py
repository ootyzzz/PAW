"""
scheduler.py
定义学习率调度器
"""
import torch.optim as optim

def get_scheduler(optimizer, scheduler_type='cosine', epochs=100):
    """
    获取学习率调度器
    支持 cosine/linear
    """
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'linear':
        return optim.lr_scheduler.LinearLR(optimizer)
    else:
        return None

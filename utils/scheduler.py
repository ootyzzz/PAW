"""
scheduler.py
定义学习率调度器
"""
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import logging

logger = logging.getLogger(__name__)

class TwoStageScheduler(_LRScheduler):
    """
    两阶段学习率调度器
    前75步: lr = 1e-4
    后50步: lr = 1e-5
    """
    
    def __init__(self, optimizer, stage1_steps=75, stage1_lr=1e-4, stage2_lr=1e-5, last_epoch=-1):
        """
        初始化两阶段调度器
        
        Args:
            optimizer: 优化器
            stage1_steps: 第一阶段步数 (默认75)
            stage1_lr: 第一阶段学习率 (默认1e-4)
            stage2_lr: 第二阶段学习率 (默认1e-5)
            last_epoch: 上次epoch数
        """
        self.stage1_steps = stage1_steps
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.total_steps = stage1_steps + 50  # 总计125步
        
        super(TwoStageScheduler, self).__init__(optimizer, last_epoch)
        
        logger.info(f"TwoStageScheduler initialized:")
        logger.info(f"  Stage 1: steps 0-{stage1_steps-1}, lr={stage1_lr}")
        logger.info(f"  Stage 2: steps {stage1_steps}-{self.total_steps-1}, lr={stage2_lr}")
    
    def get_lr(self):
        """获取当前学习率"""
        current_step = self.last_epoch + 1
        
        if current_step < self.stage1_steps:
            # 第一阶段: 前75步
            current_lr = self.stage1_lr
            stage = 1
        else:
            # 第二阶段: 后50步
            current_lr = self.stage2_lr
            stage = 2
        
        if current_step % 10 == 0 or current_step in [self.stage1_steps, self.total_steps]:
            logger.info(f"Step {current_step}: Stage {stage}, LR = {current_lr}")
        
        return [current_lr for _ in self.optimizer.param_groups]
    
    def should_save_checkpoint(self, step):
        """
        判断是否应该保存checkpoint
        
        Args:
            step: 当前步数
            
        Returns:
            bool: 是否保存
        """
        # 前75步不保存，后50步每步都保存
        return step >= self.stage1_steps
    
    def get_stage_info(self, step):
        """
        获取当前阶段信息
        
        Args:
            step: 当前步数
            
        Returns:
            dict: 阶段信息
        """
        if step < self.stage1_steps:
            return {
                'stage': 1,
                'stage_step': step,
                'total_stage_steps': self.stage1_steps,
                'lr': self.stage1_lr,
                'save_checkpoint': False
            }
        else:
            return {
                'stage': 2,
                'stage_step': step - self.stage1_steps,
                'total_stage_steps': 50,
                'lr': self.stage2_lr,
                'save_checkpoint': True
            }

def get_scheduler(optimizer, scheduler_type='cosine', epochs=100):
    """
    获取学习率调度器
    支持 cosine/linear/two_stage
    """
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'linear':
        return optim.lr_scheduler.LinearLR(optimizer)
    elif scheduler_type == 'two_stage':
        return TwoStageScheduler(optimizer)
    else:
        return None

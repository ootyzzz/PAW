"""
train.py
训练主循环，完成 prompt batch 与 LoRA checkpoint 的配对训练
增强版本支持两阶段LoRA训练和动态checkpoint策略
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

from utils.embedding import get_encoder, embed_batch
from utils.metrics import mse_loss
from utils.scheduler import get_scheduler, TwoStageScheduler
from utils.data_processor import DataProcessor
from lora.flatten import flatten_lora_weights
from lora.checkpoint_utils import load_checkpoint, CheckpointManager
from core.hyperconv_decoder import HyperConvDecoder
import random

logger = logging.getLogger(__name__)

class LoRATrainer:
    """LoRA训练器 - 支持两阶段训练策略"""
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        output_dir: str,
        lora_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化LoRA训练器
        
        Args:
            model_path: 基础模型路径
            data_path: 训练数据路径
            output_dir: 输出目录
            lora_config: LoRA配置
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # LoRA配置
        self.lora_config = lora_config or {
            'r': 16,
            'lora_alpha': 32,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            'lora_dropout': 0.1,
            'bias': 'none',
            'task_type': TaskType.CAUSAL_LM,
        }
        
        # 初始化组件
        self._init_model()
        self._init_data()
        self._init_checkpoint_manager()
        
        logger.info(f"LoRATrainer initialized")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  Output: {output_dir}")
    
    def _init_model(self):
        """初始化模型和tokenizer"""
        logger.info("Loading model and tokenizer...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 应用LoRA
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.base_model, lora_config)
        
        # 启用训练模式
        self.model.train()
        
        logger.info(f"Model loaded with LoRA config: {self.lora_config}")
        logger.info(f"Trainable parameters: {self.model.get_nb_trainable_parameters()}")
    
    def _init_data(self):
        """初始化数据处理器和数据加载器"""
        logger.info("Initializing data processor...")
        
        self.data_processor = DataProcessor(self.model_path)
        
        # 验证数据
        validation_result = self.data_processor.validate_data(self.data_path)
        if not validation_result['valid']:
            raise ValueError(f"Data validation failed: {validation_result['message']}")
        
        logger.info(f"Data validation passed: {validation_result['message']}")
    
    def _init_checkpoint_manager(self):
        """初始化checkpoint管理器"""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=50  # 后50步的checkpoint
        )
    
    def train(
        self,
        batch_size: int = 4,
        max_length: int = 512,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        logging_steps: int = 1,
        save_steps: Optional[int] = None
    ):
        """
        执行两阶段LoRA训练
        
        Args:
            batch_size: 批次大小
            max_length: 最大序列长度
            gradient_accumulation_steps: 梯度累积步数
            warmup_steps: 预热步数
            logging_steps: 日志记录间隔
            save_steps: checkpoint保存间隔 (将被调度器覆盖)
        """
        logger.info("Starting two-stage LoRA training...")
        
        # 创建数据加载器
        dataloader = self.data_processor.create_dataloader(
            data_path=self.data_path,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=True
        )
        
        # 初始化优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,  # 初始学习率，将被调度器覆盖
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 初始化两阶段调度器
        scheduler = TwoStageScheduler(
            optimizer=optimizer,
            stage1_steps=75,
            stage1_lr=1e-4,
            stage2_lr=1e-5
        )
        
        # 训练参数
        total_steps = 125  # 75 + 50
        device = next(self.model.parameters()).device
        
        # 训练循环
        global_step = 0
        accumulated_loss = 0.0
        
        self.model.train()
        
        # 创建数据迭代器
        data_iterator = iter(dataloader)
        
        with tqdm(total=total_steps, desc="Training") as pbar:
            while global_step < total_steps:
                try:
                    # 获取批次数据
                    try:
                        batch = next(data_iterator)
                    except StopIteration:
                        # 重新创建迭代器
                        data_iterator = iter(dataloader)
                        batch = next(data_iterator)
                    
                    # 移动到设备
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # 前向传播
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss / gradient_accumulation_steps
                    accumulated_loss += loss.item()
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度累积
                    if (global_step + 1) % gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # 优化器步骤
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        # 获取当前学习率
                        current_lr = scheduler.get_lr()[0]
                        
                        # 获取阶段信息
                        stage_info = scheduler.get_stage_info(global_step)
                        
                        # 记录日志
                        if global_step % logging_steps == 0:
                            logger.info(
                                f"Step {global_step}: Stage {stage_info['stage']}, "
                                f"Loss = {accumulated_loss:.4f}, LR = {current_lr:.2e}"
                            )
                        
                        # 保存checkpoint (根据调度器策略)
                        if scheduler.should_save_checkpoint(global_step):
                            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                                model=self.model,
                                optimizer=optimizer,
                                step=global_step,
                                loss=accumulated_loss,
                                lr=current_lr,
                                additional_info=stage_info
                            )
                            if checkpoint_path:
                                logger.info(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
                        
                        # 更新进度条
                        pbar.set_postfix({
                            'loss': f'{accumulated_loss:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'stage': stage_info['stage']
                        })
                        pbar.update(1)
                        
                        # 重置累积损失
                        accumulated_loss = 0.0
                        global_step += 1
                    
                    if global_step >= total_steps:
                        break
                        
                except Exception as e:
                    logger.error(f"Error during training step {global_step}: {e}")
                    break
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, 'final_model')
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"Training completed! Final model saved to: {final_model_path}")
        
        # 打印checkpoint摘要
        summary = self.checkpoint_manager.get_checkpoint_summary()
        logger.info(f"Checkpoint summary: {summary}")
        
        return {
            'final_model_path': final_model_path,
            'checkpoint_summary': summary,
            'total_steps': global_step
        }

# 构建 same-task, random-pair 配对 (保持向后兼容)
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
    主训练循环 (保持向后兼容)
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

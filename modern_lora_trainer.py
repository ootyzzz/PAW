"""
现代化的LoRA训练框架示例
基于 Hugging Face Transformers + PEFT + Accelerate + Wandb
"""
import os
import yaml
import torch
import wandb
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
import logging

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    cache_dir: str = "./models"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    
@dataclass
class LoRAConfig:
    """LoRA配置"""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "wandb"
    run_name: str = "qwen2.5_lora_training"

class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # 添加pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            torch_dtype=torch.float16 if self.config.torch_dtype == "auto" else self.config.torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def setup_lora_model(self, model, lora_config: LoRAConfig):
        """设置LoRA模型"""
        self.logger.info("Setting up LoRA configuration")
        
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model

class LoRATrainer:
    """LoRA训练器"""
    
    def __init__(self, 
                 model_config: ModelConfig,
                 lora_config: LoRAConfig,
                 training_config: TrainingConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config
        
        # 设置日志
        self.setup_logging()
        
        # 初始化accelerator
        self.accelerator = Accelerator()
        
        # 初始化wandb
        self.setup_wandb()
        
        # 加载模型
        self.model_manager = ModelManager(model_config)
        self.model, self.tokenizer = self.model_manager.load_model_and_tokenizer()
        self.model = self.model_manager.setup_lora_model(self.model, lora_config)
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'./logs/{self.training_config.run_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_wandb(self):
        """设置wandb"""
        wandb.init(
            project="qwen2.5-lora-training",
            name=self.training_config.run_name,
            config={
                "model": self.model_config.__dict__,
                "lora": self.lora_config.__dict__,
                "training": self.training_config.__dict__
            }
        )
        
    def prepare_datasets(self, train_dataset, eval_dataset=None):
        """准备数据集"""
        self.logger.info("Preparing datasets")
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        return train_dataset, eval_dataset, data_collator
    
    def train(self, train_dataset, eval_dataset=None):
        """开始训练"""
        self.logger.info("Starting training")
        
        # 准备数据
        train_dataset, eval_dataset, data_collator = self.prepare_datasets(
            train_dataset, eval_dataset
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_steps=self.training_config.warmup_steps,
            logging_steps=self.training_config.logging_steps,
            eval_steps=self.training_config.eval_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            evaluation_strategy=self.training_config.evaluation_strategy,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            remove_unused_columns=self.training_config.remove_unused_columns,
            report_to=self.training_config.report_to,
            run_name=self.training_config.run_name,
            fp16=True,  # 启用混合精度训练
            gradient_checkpointing=True,  # 启用梯度检查点
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        
        # 完成wandb记录
        wandb.finish()
        
        self.logger.info("Training completed")

# 使用示例
def main():
    # 配置
    model_config = ModelConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        cache_dir="./models"
    )
    
    lora_config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.05
    )
    
    training_config = TrainingConfig(
        output_dir="./checkpoints/qwen2.5_lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        run_name="qwen2.5_lora_exp1"
    )
    
    # 初始化训练器
    trainer = LoRATrainer(model_config, lora_config, training_config)
    
    # 这里需要加载您的数据集
    # train_dataset = load_your_dataset()
    # eval_dataset = load_your_eval_dataset()
    
    # 开始训练
    # trainer.train(train_dataset, eval_dataset)

if __name__ == "__main__":
    main()

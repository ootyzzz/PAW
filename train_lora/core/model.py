#!/usr/bin/env python3
"""
模型模块
包含LoRA Lightning模块、模型初始化、训练步骤等功能
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import swanlab
import warnings
from typing import Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class LoRALightningModule(pl.LightningModule):
    """Lightning LoRA 训练模块"""
    
    def __init__(
        self,
        model_path: str,
        lora_config: Dict[str, Any],
        learning_rate_stage1: float = 1e-4,
        learning_rate_stage2: float = 1e-5,
        stage1_steps: int = 75,
        stage2_steps: int = 50,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_path = model_path
        self.lora_config = lora_config
        self.learning_rate_stage1 = learning_rate_stage1
        self.learning_rate_stage2 = learning_rate_stage2
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.max_length = max_length
        self.total_steps = stage1_steps + stage2_steps
        
        # 初始化模型和 tokenizer
        self._init_model()
        
        # 训练状态跟踪
        self.training_step_count = 0
        
    def _init_model(self):
        """初始化模型和tokenizer"""
        print(f"📦 加载模型: {self.model_path}")

        # 平衡精度和性能，推荐大多数场景
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('medium')  

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型 - 内存优化版本
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None,  # Lightning 会处理设备分配
            trust_remote_code=True,
            # 内存优化选项
            low_cpu_mem_usage=True,  # 减少CPU内存使用
        )
        
        # 应用 LoRA
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.base_model, lora_config)
        
        print(f"✅ 模型加载完成")
        # 获取可训练参数统计
        trainable_params, total_params = self.model.get_nb_trainable_parameters()
        print(f"📊 可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
    def forward(self, input_ids, attention_mask, labels):
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss = self._compute_loss(batch, "train")
        batch_size = len(batch) if isinstance(batch, list) else batch['input_ids'].size(0)
        
        # 记录训练指标
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/step', self.training_step_count, on_step=True, batch_size=batch_size)
        
        # 记录学习率阶段信息
        current_stage = 1 if self.training_step_count < self.stage1_steps else 2
        self.log('train/stage', current_stage, on_step=True, batch_size=batch_size)
        
        # 记录到 SwanLab
        if hasattr(self, '_swanlab_run'):
            swanlab.log({
                "train/loss": loss.item(),
                "train/step": self.training_step_count,
                "train/stage": current_stage,
                "train/epoch": self.current_epoch
            }, step=self.training_step_count)
        
        self.training_step_count += 1
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤 - 用于早停和学习率调度"""
        loss = self._compute_loss(batch, "val")
        batch_size = len(batch) if isinstance(batch, list) else batch['input_ids'].size(0)
        accuracy = self._compute_accuracy(batch)
        perplexity = torch.exp(loss)
        
        # 记录验证指标
        self.log('val/loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, batch_size=batch_size)  # 早停监控这个
        self.log('val/perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return {
            'val_loss': loss,
            'val_accuracy': accuracy,
            'val_perplexity': perplexity
        }
        
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        loss = self._compute_loss(batch, "test")
        batch_size = len(batch) if isinstance(batch, list) else batch['input_ids'].size(0)
        accuracy = self._compute_accuracy(batch)
        perplexity = torch.exp(loss)
        
        # 记录测试指标
        self.log('test/loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test/accuracy', accuracy, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test/perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_perplexity': perplexity
        }
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的回调（已移除）"""
        pass  # 不再需要
    
    def on_test_epoch_end(self):
        """测试epoch结束时的回调"""
        if hasattr(self, '_swanlab_run'):
            # 记录测试指标到SwanLab
            test_loss = self.trainer.callback_metrics.get('test/loss')
            test_acc = self.trainer.callback_metrics.get('test/accuracy') 
            test_ppl = self.trainer.callback_metrics.get('test/perplexity')
            
            if test_loss is not None:
                swanlab.log({
                    "test/loss": test_loss.item(),
                    "test/accuracy": test_acc.item() if test_acc is not None else 0,
                    "test/perplexity": test_ppl.item() if test_ppl is not None else 0,
                    "final_epoch": self.current_epoch
                }, step=self.training_step_count)
    
    def _compute_loss(self, batch, stage: str):
        """计算损失的通用方法"""
        # 处理batch数据
        if isinstance(batch, list):
            # 如果是list，需要tokenize
            inputs = []
            labels = []
            
            for item in batch:
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                # 组合输入和输出
                full_text = f"{input_text}{output_text}"
                
                # Tokenize
                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                inputs.append(encoding['input_ids'].squeeze())
                labels.append(encoding['input_ids'].squeeze())
            
            input_ids = torch.stack(inputs).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            labels = torch.stack(labels).to(self.device)
            
        else:
            # 如果已经是tensor格式
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask'] 
            labels = batch['labels']
        
        # 前向传播
        outputs = self(input_ids, attention_mask, labels)
        return outputs.loss
    
    def _compute_accuracy(self, batch):
        """计算准确率（真实的多选题准确率）"""
        if not isinstance(batch, list):
            # 如果不是列表格式，返回基于loss的代理指标（用于兼容性）
            with torch.no_grad():
                loss = self._compute_loss(batch, "eval")
                accuracy = torch.exp(-loss)
                return torch.clamp(accuracy, 0.0, 1.0)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for item in batch:
                try:
                    # 解析数据项
                    question = item.get('input', '')
                    options = item.get('options', [])
                    correct_answer = item.get('target', 'A')
                    
                    if not options:
                        total += 1
                        continue
                    
                    # 格式化带选项的问题
                    prompt = f"Question: {question}\n"
                    for option in options:
                        prompt += f"{option}\n"
                    prompt += "Answer:"
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=self.max_length,
                        padding=True
                    ).to(self.device)
                    
                    # 生成参数配置（与评估脚本一致）
                    generation_kwargs = {
                        "max_new_tokens": 3,
                        "do_sample": False,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": False,
                        "output_attentions": False,
                        "output_hidden_states": False,
                    }
                    
                    # Gemma模型特殊处理
                    if "gemma" in self.model_path.lower():
                        generation_kwargs.update({
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "repetition_penalty": 1.0,
                        })
                    
                    # 生成答案 (使用警告抑制)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*cache_implementation.*")
                        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
                        outputs = self.model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                    
                    # 解码生成的答案
                    generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    generated_answer = generated_text.strip().upper()
                    
                    # 提取第一个字母 (A, B, C, 或 D)
                    predicted_answer = None
                    for char in generated_answer:
                        if char in ['A', 'B', 'C', 'D']:
                            predicted_answer = char
                            break
                    
                    # 如果没有找到明确答案，尝试匹配选项前缀
                    if predicted_answer is None:
                        for option in options:
                            if option.startswith('A:') and 'A' in generated_answer:
                                predicted_answer = 'A'
                            elif option.startswith('B:') and 'B' in generated_answer:
                                predicted_answer = 'B'
                            elif option.startswith('C:') and 'C' in generated_answer:
                                predicted_answer = 'C'
                            elif option.startswith('D:') and 'D' in generated_answer:
                                predicted_answer = 'D'
                            if predicted_answer:
                                break
                    
                    # 与正确答案比较
                    if predicted_answer == correct_answer:
                        correct += 1
                    
                    total += 1
                    
                except Exception as e:
                    # 在训练过程中，跳过生成错误的样本
                    total += 1
                    continue
        
        if total == 0:
            return torch.tensor(0.0)
        
        accuracy = correct / total
        return torch.tensor(accuracy)
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate_stage1,  # 初始学习率
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 验证准确率监控的学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',           # 监控验证准确率最大值
            factor=0.5,          # 学习率减半
            patience=30,         # 30步无提升就降低
            min_lr=1e-7          # 最小学习率
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",  # 监控验证准确率
                "interval": "epoch",
                "frequency": 1,
            },
        }

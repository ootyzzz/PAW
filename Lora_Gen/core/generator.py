"""
generator.py
完整的 LoRA 参数生成器
Text Encoder (HuggingFace Transformers) -> HyperConv Decoder -> Tokenized Parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple, Union
import numpy as np

from .hyperconv_decoder import HyperConvDecoder


class HuggingFaceTextEncoder(nn.Module):
    """
    基于HuggingFace Transformers的文本编码器
    使用官方方法实现all-MiniLM-L6-v2
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.model_name = model_name
        print(f"🔄 加载HuggingFace模型: {model_name}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 获取embedding维度
        self.embedding_dim = self.model.config.hidden_size  # 384 for all-MiniLM-L6-v2
        print(f"✅ 模型加载成功，embedding维度: {self.embedding_dim}")
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - 考虑attention mask的正确平均
        这是官网提供的标准实现
        """
        token_embeddings = model_output[0]  # 第一个元素包含所有token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, sentences):
        """
        编码句子为embeddings
        
        Args:
            sentences: str或List[str] - 输入句子
            
        Returns:
            sentence_embeddings: torch.Tensor [batch_size, embedding_dim]
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Tokenize句子
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        # 移动到正确的device
        device = next(self.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # 计算token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # 执行pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # 标准化embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings
    
    def forward(self, sentences):
        """前向传播"""
        return self.encode(sentences)


class LoRAParameterGenerator(nn.Module):
    """
    完整的LoRA参数生成器
    
    架构：
    Text Input -> HuggingFace Text Encoder -> HyperConv Decoder -> Tokenized LoRA Parameters
    """
    def __init__(
        self, 
        text_encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2',  # HuggingFace model
        hidden_dim: int = 384,
        max_seq_len: int = 512,
        num_hyperconv_blocks: int = 3,
        output_dim: int = None,
        freeze_text_encoder: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim or hidden_dim
        
        # Text Encoder (HuggingFace Transformers)
        self.text_encoder = HuggingFaceTextEncoder(text_encoder_name)
        self.encoder_dim = self.text_encoder.embedding_dim
        
        # 冻结text encoder参数以节省显存
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # 输入投影层：将HuggingFace输出映射到hidden_dim
        if self.encoder_dim != hidden_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(self.encoder_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        else:
            self.input_proj = nn.Identity()
        
        # 序列扩展层：将单个embedding扩展为序列
        self.sequence_expander = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * max_seq_len),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Parameter Generator (HyperConv Decoder)
        self.parameter_generator = HyperConvDecoder(
            num_blocks=num_hyperconv_blocks,
            dim=hidden_dim,
            max_seq_len=max_seq_len,
            output_dim=self.output_dim
        )
        
        # 最终输出归一化
        self.output_norm = nn.LayerNorm(self.output_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, prompts: Union[List[str], torch.Tensor], return_embeddings: bool = False):
        """
        Args:
            prompts: List[str] 或已编码的tensor [B, encoder_dim]
            return_embeddings: bool - 是否返回中间embeddings
        
        Returns:
            generated_params: torch.Tensor [B, max_seq_len, output_dim] - 生成的参数tokens
        """
        if isinstance(prompts, list):
            # 1. Text Encoding (HuggingFace Transformers)
            with torch.no_grad():
                text_embeddings = self.text_encoder.encode(prompts)
        else:
            text_embeddings = prompts
        
        batch_size = text_embeddings.shape[0]
        
        # 2. Input Projection
        embeddings = self.input_proj(text_embeddings)  # [B, hidden_dim]
        
        # 3. 扩展到序列维度
        expanded = self.sequence_expander(embeddings)  # [B, hidden_dim * max_seq_len]
        expanded = expanded.view(batch_size, self.max_seq_len, self.hidden_dim)  # [B, max_seq_len, hidden_dim]
        
        # 4. Parameter Generation
        generated_params = self.parameter_generator(expanded)  # [B, max_seq_len, output_dim]
        generated_params = self.output_norm(generated_params)
        
        if return_embeddings:
            return generated_params, text_embeddings
        return generated_params
    
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """单独的文本编码方法"""
        with torch.no_grad():
            return self.text_encoder.encode(prompts)


class LoRATokenizer:
    """
    LoRA参数的Tokenizer
    将checkpoint参数转换为固定长度的token序列
    """
    def __init__(self, max_tokens: int = 512, token_dim: int = 384):
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        
    def tokenize_checkpoint(self, checkpoint_path: str, layer_names: List[str] = None) -> torch.Tensor:
        """
        将checkpoint的LoRA参数tokenize
        
        Args:
            checkpoint_path: checkpoint文件路径
            layer_names: 要提取的层名称列表
            
        Returns:
            tokens: torch.Tensor [max_tokens, token_dim] - tokenized参数
        """
        # 加载checkpoint - 处理PyTorch 2.6的安全模式
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception:
            # 回退到旧版本加载方式
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取参数 - 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                params_dict = checkpoint['state_dict']
            else:
                params_dict = checkpoint
        else:
            params_dict = checkpoint
        
        # 提取LoRA参数
        lora_params = []
        for name, param in params_dict.items():
            if 'lora' in name.lower() and (layer_names is None or any(ln in name for ln in layer_names)):
                lora_params.append(param.flatten())
        
        if not lora_params:
            raise ValueError(f"No LoRA parameters found in {checkpoint_path}")
        
        # 拼接所有参数
        all_params = torch.cat(lora_params, dim=0)  # [total_params]
        
        # 分块并重塑为tokens
        total_elements = len(all_params)
        elements_per_token = self.token_dim
        
        # 计算需要的token数量
        num_tokens = (total_elements + elements_per_token - 1) // elements_per_token
        
        # Padding到合适的大小
        if total_elements % elements_per_token != 0:
            padding_size = elements_per_token - (total_elements % elements_per_token)
            all_params = torch.cat([all_params, torch.zeros(padding_size)], dim=0)
        
        # 重塑为token格式
        tokens = all_params.view(-1, elements_per_token)  # [num_tokens, token_dim]
        
        # Padding或截断到max_tokens
        if num_tokens > self.max_tokens:
            tokens = tokens[:self.max_tokens]
        elif num_tokens < self.max_tokens:
            padding = torch.zeros(self.max_tokens - num_tokens, self.token_dim)
            tokens = torch.cat([tokens, padding], dim=0)
        
        return tokens
    
    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        将tokens转换回参数向量
        
        Args:
            tokens: [max_tokens, token_dim] or [batch_size, max_tokens, token_dim]
            
        Returns:
            params: 扁平化的参数向量
        """
        if tokens.dim() == 3:
            # 批处理情况
            return tokens.view(tokens.shape[0], -1)
        else:
            # 单个样本
            return tokens.view(-1)

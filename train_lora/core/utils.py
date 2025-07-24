#!/usr/bin/env python3
"""
工具模块
包含内存优化、环境配置等通用功能
"""

import os
import torch
import warnings
import logging


def configure_memory_optimizations():
    """配置内存优化设置"""
    # 设置环境变量优化内存使用
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # 避免tokenizer并行警告
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')  # 优化CUDA内存分配
    
    # 启用内存高效的注意力机制
    try:
        import torch.backends.cudnn as cudnn
        if torch.cuda.is_available():
            cudnn.benchmark = True  # 优化cudnn性能
            torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
            torch.backends.cudnn.allow_tf32 = True
    except Exception as e:
        print(f"⚠️ 内存优化配置警告: {e}")


def setup_environment():
    """设置训练环境"""
    # 修复MKL线程层冲突
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    configure_memory_optimizations()
    
    # 屏蔽 Transformers 相关警告
    warnings.filterwarnings("ignore", message=".*cache_implementation.*")
    warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
    
    # 设置 Transformers 日志级别为 ERROR 以减少信息输出
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    # 设置 HuggingFace transformers 的日志级别
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except ImportError:
        pass
    
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

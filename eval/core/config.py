"""
环境配置和导入模块
包含所有环境变量设置、PyTorch配置和必要的导入
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import random

# 设置环境变量以解决tokenizers并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 禁用CUDA图优化，解决Gemma模型的deterministic index put问题
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
# 设置transformers日志级别，减少警告信息
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
torch.set_float32_matmul_precision('medium')  # 使用更保守的设置
# 禁用CUDA图以避免deterministic问题
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(False)  # 禁用Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp(False)  # 禁用内存高效Attention

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

import pandas as pd
from tqdm import tqdm

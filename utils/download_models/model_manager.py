#!/usr/bin/env python3
"""
model_manager.py - 增强版模型管理器
支持Qwen2.5模型加载、LoRA配置管理和模型状态检查
"""

import os
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import logging

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelManager:
    """增强的模型管理器"""
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型配置缓存
        self.model_configs = {}
        
        # 检查依赖
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers and/or PEFT not available. Some features will be limited.")
    
    def download_model(self, model_name: str, save_local: bool = True) -> bool:
        """
        下载模型到本地
        
        Args:
            model_name: 模型名称
            save_local: 是否保存到本地
            
        Returns:
            bool: 是否成功
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available. Please install: pip install transformers torch")
            return False
        
        print(f"📥 开始下载模型: {model_name}")
        print(f"📁 保存路径: {self.cache_dir}")
        
        try:
            # 下载配置
            print("📋 下载配置文件...")
            config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # 下载tokenizer
            print("📦 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # 下载模型
            print("🤖 下载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None  # 先不映射到设备
            )
            
            # 如果需要保存到本地目录
            if save_local:
                local_path = self.cache_dir / model_name.replace("/", "_")
                local_path.mkdir(exist_ok=True)
                
                print(f"💾 保存到本地: {local_path}")
                model.save_pretrained(local_path)
                tokenizer.save_pretrained(local_path)
                config.save_pretrained(local_path)
            
            print("✅ 模型下载完成!")
            
            # 显示模型信息
            self._display_model_info(model, model_name)
            
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False
    
    def load_model_for_lora(
        self, 
        model_path: str,
        lora_config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto"
    ) -> Tuple[Any, Any, Any]:
        """
        加载模型并应用LoRA配置
        
        Args:
            model_path: 模型路径
            lora_config: LoRA配置
            device_map: 设备映射策略
            
        Returns:
            tuple: (model, tokenizer, lora_config)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers and PEFT required for LoRA functionality")
        
        print(f"� 加载模型: {model_path}")
        
        try:
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 设置pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # 应用LoRA配置
            if lora_config is None:
                lora_config = self._get_default_lora_config()
            
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(base_model, peft_config)
            
            print("✅ 模型加载完成!")
            print(f"📊 可训练参数: {model.get_nb_trainable_parameters():,}")
            print(f"📊 总参数: {model.num_parameters():,}")
            
            return model, tokenizer, lora_config
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _get_default_lora_config(self) -> Dict[str, Any]:
        """获取默认LoRA配置"""
        return {
            'r': 16,
            'lora_alpha': 32,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            'lora_dropout': 0.1,
            'bias': 'none',
            'task_type': TaskType.CAUSAL_LM,
        }
    
    def check_model_compatibility(self, model_path: str) -> Dict[str, Any]:
        """
        检查模型兼容性
        
        Args:
            model_path: 模型路径
            
        Returns:
            dict: 兼容性检查结果
        """
        result = {
            "path_exists": False,
            "config_valid": False,
            "tokenizer_valid": False,
            "model_valid": False,
            "cuda_compatible": False,
            "lora_compatible": False,
            "details": {}
        }
        
        try:
            # 检查路径
            path = Path(model_path)
            result["path_exists"] = path.exists()
            
            if not result["path_exists"]:
                result["details"]["error"] = f"Model path does not exist: {model_path}"
                return result
            
            if not TRANSFORMERS_AVAILABLE:
                result["details"]["error"] = "Transformers not available"
                return result
            
            # 检查配置文件
            try:
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                result["config_valid"] = True
                result["details"]["model_type"] = config.model_type
                result["details"]["hidden_size"] = getattr(config, 'hidden_size', 'unknown')
            except Exception as e:
                result["details"]["config_error"] = str(e)
            
            # 检查tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                result["tokenizer_valid"] = True
                result["details"]["vocab_size"] = tokenizer.vocab_size
                result["details"]["pad_token"] = tokenizer.pad_token is not None
            except Exception as e:
                result["details"]["tokenizer_error"] = str(e)
            
            # 检查模型加载
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    device_map=None,  # 不映射到设备
                    torch_dtype=torch.float32  # 使用float32避免设备问题
                )
                result["model_valid"] = True
                result["details"]["num_parameters"] = model.num_parameters()
                
                # 检查LoRA兼容性
                try:
                    lora_config = LoraConfig(**self._get_default_lora_config())
                    get_peft_model(model, lora_config)
                    result["lora_compatible"] = True
                except Exception as e:
                    result["details"]["lora_error"] = str(e)
                
            except Exception as e:
                result["details"]["model_error"] = str(e)
            
            # 检查CUDA兼容性
            result["cuda_compatible"] = torch.cuda.is_available()
            result["details"]["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
            result["details"]["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
        except Exception as e:
            result["details"]["general_error"] = str(e)
        
        return result
    
    def _display_model_info(self, model, model_name: str):
        """显示模型信息"""
        print(f"📊 模型信息:")
        print(f"  - 名称: {model_name}")
        print(f"  - 参数量: {model.num_parameters():,}")
        print(f"  - 缓存路径: {self.cache_dir}")
        print(f"  - 设备: {next(model.parameters()).device}")
        print(f"  - 数据类型: {next(model.parameters()).dtype}")
    
    def list_models(self) -> None:
        """列出已下载的模型"""
        print(f"📚 已下载的模型 ({self.cache_dir}):")
        
        if not self.cache_dir.exists():
            print("📁 模型缓存目录不存在")
            return
        
        # 查找模型目录
        model_dirs = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                # 检查是否包含模型文件
                has_model_files = (
                    any(item.glob("*.bin")) or 
                    any(item.glob("*.safetensors")) or
                    (item / "config.json").exists()
                )
                if has_model_files:
                    model_dirs.append(item)
        
        if not model_dirs:
            print("  暂无已下载的模型")
            return
        
        for i, model_dir in enumerate(model_dirs, 1):
            print(f"  {i}. {model_dir.name}")
            
            # 显示模型文件大小
            total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"     大小: {size_mb:.1f} MB")
            
            # 显示兼容性状态
            compat = self.check_model_compatibility(str(model_dir))
            status = "✅" if compat["lora_compatible"] else "❌"
            print(f"     LoRA兼容: {status}")
    
    def save_model_config(self, model_name: str, config: Dict[str, Any]):
        """保存模型配置"""
        config_file = self.cache_dir / f"{model_name.replace('/', '_')}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """加载模型配置"""
        config_file = self.cache_dir / f"{model_name.replace('/', '_')}_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return None

def main():
    parser = argparse.ArgumentParser(description="增强版模型管理工具")
    parser.add_argument("--action", 
                       choices=["download", "list", "check", "test_lora"], 
                       default="list",
                       help="操作类型")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="模型名称")
    parser.add_argument("--cache_dir", type=str, default="./models",
                       help="模型缓存目录")
    parser.add_argument("--model_path", type=str,
                       help="本地模型路径（用于检查和测试）")
    
    args = parser.parse_args()
    
    print("🏗️  Enhanced Model Manager")
    print("=" * 50)
    
    manager = ModelManager(args.cache_dir)
    
    if args.action == "download":
        manager.download_model(args.model)
    
    elif args.action == "list":
        manager.list_models()
    
    elif args.action == "check":
        model_path = args.model_path or args.model
        result = manager.check_model_compatibility(model_path)
        print(f"🔍 兼容性检查结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.action == "test_lora":
        model_path = args.model_path or args.model
        try:
            model, tokenizer, lora_config = manager.load_model_for_lora(model_path)
            print("✅ LoRA测试成功!")
            print(f"📊 LoRA配置: {lora_config}")
        except Exception as e:
            print(f"❌ LoRA测试失败: {e}")

if __name__ == "__main__":
    main()

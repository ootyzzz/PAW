"""
LoRA迁移模块
负责LoRA权重迁移
"""

import os
from typing import Optional
from .config import PipelineConfig
from .utils import ModelUtils, CommandRunner, get_timestamp


class LoRATransfer:
    """LoRA迁移器"""
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.runner = CommandRunner(verbose=verbose)
        self.verbose = verbose
        self.timestamp = get_timestamp()
    
    def transfer_lora(self, source_lora_path: str, source_model: str, 
                     target_model: str, dataset: str) -> Optional[str]:
        """迁移LoRA
        
        Args:
            source_lora_path: 源LoRA路径
            source_model: 源模型路径
            target_model: 目标模型路径
            dataset: 数据集名称
            
        Returns:
            迁移后的LoRA路径 或 None 如果失败
        """
        source_name = ModelUtils.get_model_short_name(source_model)
        target_name = ModelUtils.get_model_short_name(target_model)
        
        if self.verbose:
            print(f"\n🔄 开始迁移 LoRA: {source_name} → {target_name}")
        
        # 创建输出目录
        output_dir = self._create_output_dir(source_name, target_name, dataset)
        
        # 构建迁移命令
        cmd = self._build_transfer_command(
            source_lora_path, source_model, target_model, output_dir
        )
        
        # 执行迁移
        output = self.runner.run_command(
            cmd,
            f"迁移 LoRA ({source_name} → {target_name})",
            cwd=os.path.dirname(self.config.get('paths.transfer_script'))
        )
        
        if output is None:
            return None
        
        return output_dir
    
    def _create_output_dir(self, source_name: str, target_name: str, dataset: str) -> str:
        """创建输出目录"""
        output_dir = os.path.join(
            self.config.get('paths.transferred_lora_dir'),
            dataset,
            f"{source_name}_to_{target_name}",
            self.timestamp
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _build_transfer_command(self, source_lora_path: str, source_model: str, 
                               target_model: str, output_dir: str) -> str:
        """构建迁移命令"""
        transfer_script = os.path.basename(self.config.get('paths.transfer_script'))
        similarity_threshold = self.config.get('transfer.similarity_threshold')
        
        cmd = f"python {transfer_script} " \
              f"--source_lora {source_lora_path} " \
              f"--source_model {source_model} " \
              f"--target_model {target_model} " \
              f"--output {output_dir} " \
              f"--similarity_threshold {similarity_threshold}"
        
        return cmd
    
    def check_existing_transfer(self, source_model: str, target_model: str, dataset: str) -> Optional[str]:
        """检查是否已有迁移结果"""
        source_name = ModelUtils.get_model_short_name(source_model)
        target_name = ModelUtils.get_model_short_name(target_model)
        
        transfer_base_dir = os.path.join(
            self.config.get('paths.transferred_lora_dir'),
            dataset,
            f"{source_name}_to_{target_name}"
        )
        
        if not os.path.exists(transfer_base_dir):
            return None
        
        # 查找最新的迁移结果
        existing_transfers = [d for d in os.listdir(transfer_base_dir) 
                             if os.path.isdir(os.path.join(transfer_base_dir, d))]
        
        if not existing_transfers:
            return None
        
        latest_transfer = sorted(existing_transfers)[-1]
        transfer_path = os.path.join(transfer_base_dir, latest_transfer)
        
        return transfer_path
    
    def check_step_completed(self, source_model: str, target_model: str, dataset: str) -> tuple[bool, Optional[str]]:
        """检查迁移步骤是否已完成
        
        Returns:
            Tuple[是否完成, 迁移路径]
        """
        existing_path = self.check_existing_transfer(source_model, target_model, dataset)
        
        if existing_path:
            return True, existing_path
        
        return False, None

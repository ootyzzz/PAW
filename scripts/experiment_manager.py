"""
实验管理器
用于管理和跟踪训练实验
增强版本支持LoRA训练和Commonsense数据集实验
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import logging

# 导入项目组件
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.train import LoRATrainer
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class ExperimentManager:
    """增强的实验管理器"""
    
    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.experiments_dir / 'experiment_manager.log'),
                logging.StreamHandler()
            ]
        )
        
    def create_experiment(self, 
                         name: str, 
                         config: Dict,
                         description: str = "",
                         tags: List[str] = None) -> str:
        """创建新实验"""
        
        # 创建实验目录
        exp_dir = self.experiments_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "models").mkdir(exist_ok=True)
        
        # 保存配置
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # 保存实验元数据
        metadata = {
            "name": name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "config": config,
            "type": config.get("experiment_type", "general")
        }
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Experiment '{name}' created successfully")
        logger.info(f"Experiment directory: {exp_dir}")
        
        return str(exp_dir)
    
    def create_commonsense_lora_experiment(
        self,
        name: str,
        model_path: str,
        data_path: str,
        description: str = "Commonsense LoRA Training",
        custom_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建Commonsense LoRA训练实验
        
        Args:
            name: 实验名称
            model_path: 模型路径
            data_path: 数据路径
            description: 实验描述
            custom_config: 自定义配置
            
        Returns:
            str: 实验目录路径
        """
        # 默认配置
        config = {
            "experiment_type": "commonsense_lora",
            "model_path": model_path,
            "data_path": data_path,
            "training": {
                "batch_size": 4,
                "max_length": 512,
                "gradient_accumulation_steps": 1,
                "logging_steps": 1,
                "stage1_steps": 75,
                "stage1_lr": 1e-4,
                "stage2_steps": 50,
                "stage2_lr": 1e-5
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_dropout": 0.1,
                "bias": "none"
            },
            "hardware": {
                "device": "auto",
                "mixed_precision": True
            }
        }
        
        # 合并自定义配置
        if custom_config:
            config.update(custom_config)
        
        # 创建实验
        exp_dir = self.create_experiment(
            name=name,
            config=config,
            description=description,
            tags=["lora", "commonsense", "qwen2.5"]
        )
        
        return exp_dir
    
    def run_commonsense_lora_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """
        运行Commonsense LoRA实验
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            dict: 训练结果
        """
        # 获取实验配置
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        if experiment.get("type") != "commonsense_lora":
            raise ValueError(f"Experiment '{experiment_name}' is not a commonsense_lora experiment")
        
        config = experiment["config"]
        exp_dir = self.experiments_dir / experiment_name
        
        # 更新状态
        self.update_experiment_status(experiment_name, "running")
        
        try:
            # 创建训练器
            trainer = LoRATrainer(
                model_path=config["model_path"],
                data_path=config["data_path"],
                output_dir=str(exp_dir / "models"),
                lora_config=config["lora"]
            )
            
            # 执行训练
            training_config = config["training"]
            
            # 适配不同的配置格式
            batch_size = training_config.get("batch_size", training_config.get("per_device_train_batch_size", 4))
            max_length = training_config.get("max_length", config["data"].get("max_length", 512))
            gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
            logging_steps = training_config.get("logging_steps", 1)
            
            results = trainer.train(
                batch_size=batch_size,
                max_length=max_length,
                gradient_accumulation_steps=gradient_accumulation_steps,
                logging_steps=logging_steps
            )
            
            # 保存结果
            results_file = exp_dir / "results" / "training_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # 更新状态
            self.update_experiment_status(experiment_name, "completed")
            
            logger.info(f"Experiment '{experiment_name}' completed successfully")
            return results
            
        except Exception as e:
            # 更新状态为失败
            self.update_experiment_status(experiment_name, "failed")
            logger.error(f"Experiment '{experiment_name}' failed: {e}")
            
            # 保存错误信息
            error_info = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            error_file = exp_dir / "results" / "error.json"
            with open(error_file, "w") as f:
                json.dump(error_info, f, indent=2)
            
            raise
    
    def validate_experiment_data(self, experiment_name: str) -> Dict[str, Any]:
        """
        验证实验数据
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            dict: 验证结果
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            return {"valid": False, "error": "Experiment not found"}
        
        config = experiment["config"]
        
        try:
            # 验证数据
            data_processor = DataProcessor(config["model_path"])
            validation_result = data_processor.validate_data(config["data_path"])
            
            # 保存验证结果
            exp_dir = self.experiments_dir / experiment_name
            validation_file = exp_dir / "results" / "data_validation.json"
            with open(validation_file, "w") as f:
                json.dump(validation_result, f, indent=2)
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    def list_experiments(self) -> List[Dict]:
        """列出所有实验"""
        experiments = []
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != "archived":
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    experiments.append(metadata)
                else:
                    # 兼容旧格式
                    experiments.append({
                        "name": exp_dir.name,
                        "description": "旧实验格式",
                        "created_at": "未知",
                        "status": "未知",
                        "type": "unknown"
                    })
        
        return sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def get_experiment(self, name: str) -> Optional[Dict]:
        """获取实验信息"""
        exp_dir = self.experiments_dir / name
        metadata_file = exp_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, "r") as f:
            return json.load(f)
    
    def update_experiment_status(self, name: str, status: str):
        """更新实验状态"""
        exp_dir = self.experiments_dir / name
        metadata_file = exp_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            metadata["status"] = status
            metadata["updated_at"] = datetime.now().isoformat()
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
    
    def get_experiment_results(self, name: str) -> Optional[Dict]:
        """获取实验结果"""
        exp_dir = self.experiments_dir / name
        results_file = exp_dir / "results" / "training_results.json"
        
        if not results_file.exists():
            return None
        
        with open(results_file, "r") as f:
            return json.load(f)
    
    def delete_experiment(self, name: str, confirm: bool = False):
        """删除实验"""
        exp_dir = self.experiments_dir / name
        
        if not exp_dir.exists():
            logger.warning(f"Experiment '{name}' does not exist")
            return
        
        if not confirm:
            print(f"⚠️  确定要删除实验 '{name}' 吗？")
            response = input("输入 'yes' 确认: ")
            if response.lower() != 'yes':
                print("取消删除")
                return
        
        shutil.rmtree(exp_dir)
        logger.info(f"Experiment '{name}' deleted")
    
    def archive_experiment(self, name: str):
        """归档实验"""
        exp_dir = self.experiments_dir / name
        archive_dir = self.experiments_dir / "archived"
        archive_dir.mkdir(exist_ok=True)
        
        if exp_dir.exists():
            shutil.move(str(exp_dir), str(archive_dir / name))
            logger.info(f"Experiment '{name}' archived")
        else:
            logger.warning(f"Experiment '{name}' does not exist")
    
    def print_experiments_summary(self):
        """打印实验摘要"""
        experiments = self.list_experiments()
        
        if not experiments:
            print("📝 暂无实验")
            return
        
        print(f"📊 实验总数: {len(experiments)}")
        print("=" * 80)
        
        for i, exp in enumerate(experiments, 1):
            print(f"{i}. {exp['name']} ({exp.get('type', 'unknown')})")
            print(f"   描述: {exp.get('description', '无')}")
            print(f"   状态: {exp.get('status', '未知')}")
            print(f"   创建时间: {exp.get('created_at', '未知')}")
            
            # 显示标签
            tags = exp.get('tags', [])
            if tags:
                print(f"   标签: {', '.join(tags)}")
            
            print()

def main():
    """命令行界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实验管理器")
    parser.add_argument("action", 
                       choices=["create", "create_lora", "run_lora", "validate", "list", "delete", "archive", "summary"],
                       help="操作类型")
    parser.add_argument("--name", type=str, help="实验名称")
    parser.add_argument("--description", type=str, default="", help="实验描述")
    parser.add_argument("--tags", type=str, nargs="+", help="实验标签")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--data_path", type=str, help="数据路径")
    parser.add_argument("--confirm", action="store_true", help="确认删除")
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.action == "create":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        
        config = {}
        if args.config and os.path.exists(args.config):
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
        
        manager.create_experiment(
            args.name, 
            config, 
            args.description,
            args.tags
        )
    
    elif args.action == "create_lora":
        if not args.name or not args.model_path or not args.data_path:
            print("❌ 请提供实验名称、模型路径和数据路径")
            return
        
        manager.create_commonsense_lora_experiment(
            name=args.name,
            model_path=args.model_path,
            data_path=args.data_path,
            description=args.description
        )
    
    elif args.action == "run_lora":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        
        try:
            results = manager.run_commonsense_lora_experiment(args.name)
            print(f"✅ 实验 '{args.name}' 完成")
            print(f"结果: {results}")
        except Exception as e:
            print(f"❌ 实验 '{args.name}' 失败: {e}")
    
    elif args.action == "validate":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        
        result = manager.validate_experiment_data(args.name)
        print(f"验证结果: {result}")
    
    elif args.action == "list":
        experiments = manager.list_experiments()
        for exp in experiments:
            print(f"- {exp['name']} ({exp.get('status', '未知')}) [{exp.get('type', 'unknown')}]")
    
    elif args.action == "delete":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        manager.delete_experiment(args.name, args.confirm)
    
    elif args.action == "archive":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        manager.archive_experiment(args.name)
    
    elif args.action == "summary":
        manager.print_experiments_summary()

if __name__ == "__main__":
    main()
            
            metadata["status"] = status
            metadata["updated_at"] = datetime.now().isoformat()
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
    
    def delete_experiment(self, name: str, confirm: bool = False):
        """删除实验"""
        exp_dir = self.experiments_dir / name
        
        if not exp_dir.exists():
            print(f"❌ 实验 '{name}' 不存在")
            return
        
        if not confirm:
            print(f"⚠️  确定要删除实验 '{name}' 吗？")
            response = input("输入 'yes' 确认: ")
            if response.lower() != 'yes':
                print("取消删除")
                return
        
        shutil.rmtree(exp_dir)
        print(f"✅ 实验 '{name}' 已删除")
    
    def archive_experiment(self, name: str):
        """归档实验"""
        exp_dir = self.experiments_dir / name
        archive_dir = self.experiments_dir / "archived"
        archive_dir.mkdir(exist_ok=True)
        
        if exp_dir.exists():
            shutil.move(str(exp_dir), str(archive_dir / name))
            print(f"📦 实验 '{name}' 已归档")
        else:
            print(f"❌ 实验 '{name}' 不存在")
    
    def print_experiments_summary(self):
        """打印实验摘要"""
        experiments = self.list_experiments()
        
        if not experiments:
            print("📝 暂无实验")
            return
        
        print(f"📊 实验总数: {len(experiments)}")
        print("=" * 80)
        
        for i, exp in enumerate(experiments, 1):
            print(f"{i}. {exp['name']}")
            print(f"   描述: {exp.get('description', '无')}")
            print(f"   状态: {exp.get('status', '未知')}")
            print(f"   创建时间: {exp.get('created_at', '未知')}")
            
            # 显示标签
            tags = exp.get('tags', [])
            if tags:
                print(f"   标签: {', '.join(tags)}")
            
            print()

def main():
    """命令行界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实验管理器")
    parser.add_argument("action", choices=["create", "list", "delete", "archive", "summary"],
                       help="操作类型")
    parser.add_argument("--name", type=str, help="实验名称")
    parser.add_argument("--description", type=str, default="", help="实验描述")
    parser.add_argument("--tags", type=str, nargs="+", help="实验标签")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--confirm", action="store_true", help="确认删除")
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.action == "create":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        
        config = {}
        if args.config and os.path.exists(args.config):
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
        
        manager.create_experiment(
            args.name, 
            config, 
            args.description,
            args.tags
        )
    
    elif args.action == "list":
        experiments = manager.list_experiments()
        for exp in experiments:
            print(f"- {exp['name']} ({exp.get('status', '未知')})")
    
    elif args.action == "delete":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        manager.delete_experiment(args.name, args.confirm)
    
    elif args.action == "archive":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        manager.archive_experiment(args.name)
    
    elif args.action == "summary":
        manager.print_experiments_summary()

if __name__ == "__main__":
    main()

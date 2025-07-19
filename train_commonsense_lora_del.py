#!/usr/bin/env python3
"""
train_commonsense_lora.py
主训练脚本 - 整合所有组件实现Commonsense LoRA训练
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from scripts.experiment_manager_enhanced import ExperimentManager
    from scripts.model_manager import ModelManager
    from utils.data_processor import DataProcessor
    from core.train import LoRATrainer
    from utils.scheduler import TwoStageScheduler
    from lora.checkpoint_utils import CheckpointManager
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有依赖已正确安装")
    sys.exit(1)

def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置根日志器
    logger = logging.getLogger('commonsense_lora_training')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def validate_paths(config: Dict[str, Any]) -> bool:
    """验证路径是否存在"""
    paths_to_check = [
        ("模型路径", config['model']['local_path']),
        ("数据文件", config['data']['train_file'])
    ]
    
    all_valid = True
    for name, path in paths_to_check:
        if not os.path.exists(path):
            print(f"❌ {name}不存在: {path}")
            all_valid = False
        else:
            print(f"✅ {name}验证通过: {path}")
    
    return all_valid

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Commonsense LoRA训练脚本")
    parser.add_argument("--config", 
                       default="./configs/training_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--experiment_name", 
                       default=f"commonsense_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="实验名称")
    parser.add_argument("--mode", choices=['mixed'], 
                       help="训练模式: mixed(混合数据集训练)")
    parser.add_argument("--dry_run", 
                       action="store_true",
                       help="仅验证配置，不实际训练")
    parser.add_argument("--resume", 
                       type=str,
                       help="恢复训练的实验名称")
    parser.add_argument("--validate_only", 
                       action="store_true",
                       help="仅验证数据和模型")
    
    args = parser.parse_args()
    
    # 如果没有指定模式，推荐使用增强版脚本
    if not args.mode:
        print("=" * 70)
        print("💡 提示: 现在支持individual datasets训练!")
        print("   - 默认训练混合数据集 (当前脚本)")
        print("   - 训练individual数据集: python train_commonsense_lora_enhanced.py")
        print("   - 训练mixed数据集: python train_commonsense_lora_enhanced.py --mode mixed")
        print("=" * 70)
        print()
    
    print("🚀 Commonsense LoRA Training Script")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"实验名称: {args.experiment_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 1. 加载配置
        print("\n📋 步骤1: 加载配置...")
        config = load_config(args.config)
        print(f"✅ 配置加载成功")
        
        # 2. 设置日志
        print("\n📝 步骤2: 设置日志系统...")
        log_dir = config.get('logging', {}).get('log_dir', './logs')
        logger = setup_logging(log_dir)
        logger.info(f"训练开始 - 实验: {args.experiment_name}")
        print(f"✅ 日志系统已设置: {log_dir}")
        
        # 3. 验证路径
        print("\n🔍 步骤3: 验证路径...")
        if not validate_paths(config):
            logger.error("路径验证失败")
            return False
        logger.info("路径验证通过")
        
        # 4. 初始化管理器
        print("\n🏗️ 步骤4: 初始化管理器...")
        experiment_manager = ExperimentManager("./experiments")
        model_manager = ModelManager(config['model']['cache_dir'])
        print("✅ 管理器初始化完成")
        
        # 5. 验证模型兼容性
        print("\n🔧 步骤5: 验证模型...")
        model_path = config['model']['local_path']
        compatibility = model_manager.check_model_compatibility(model_path)
        
        if not compatibility['model_valid']:
            logger.error(f"模型验证失败: {compatibility}")
            return False
        
        if not compatibility['lora_compatible']:
            logger.warning("LoRA兼容性检查失败，但将继续尝试")
        
        print("✅ 模型验证通过")
        logger.info(f"模型兼容性: {compatibility}")
        
        # 6. 验证数据
        print("\n📊 步骤6: 验证数据...")
        data_processor = DataProcessor(model_path)
        data_validation = data_processor.validate_data(config['data']['train_file'])
        
        if not data_validation['valid']:
            logger.error(f"数据验证失败: {data_validation}")
            return False
        
        print(f"✅ 数据验证通过: {data_validation['total_samples']} 样本")
        logger.info(f"数据验证结果: {data_validation}")
        
        if args.validate_only:
            print("\n✅ 验证完成，退出")
            return True
        
        # 7. 创建实验
        print(f"\n🧪 步骤7: 创建实验 '{args.experiment_name}'...")
        
        if args.resume:
            print(f"📂 恢复实验: {args.resume}")
            experiment = experiment_manager.get_experiment(args.resume)
            if not experiment:
                logger.error(f"实验不存在: {args.resume}")
                return False
            args.experiment_name = args.resume
        else:
            experiment_config = {
                "experiment_type": "commonsense_lora",
                "model_path": model_path,
                "data_path": config['data']['train_file'],
                "training": config['training'],
                "lora": config['lora'],
                "hardware": config.get('hardware', {}),
                "timestamp": datetime.now().isoformat()
            }
            
            exp_dir = experiment_manager.create_commonsense_lora_experiment(
                name=args.experiment_name,
                model_path=model_path,
                data_path=config['data']['train_file'],
                description=config.get('experiment', {}).get('description', ''),
                custom_config=experiment_config
            )
            print(f"✅ 实验创建成功: {exp_dir}")
        
        if args.dry_run:
            print("\n🏃 Dry run完成，未实际训练")
            logger.info("Dry run完成")
            return True
        
        # 8. 执行训练
        print(f"\n🎯 步骤8: 开始训练...")
        logger.info("开始LoRA训练")
        
        results = experiment_manager.run_commonsense_lora_experiment(args.experiment_name)
        
        print("\n🎉 训练完成!")
        print(f"📊 训练结果:")
        print(f"  - 最终模型路径: {results.get('final_model_path', 'N/A')}")
        print(f"  - 总训练步数: {results.get('total_steps', 'N/A')}")
        print(f"  - Checkpoint数量: {results.get('checkpoint_summary', {}).get('total_checkpoints', 'N/A')}")
        
        logger.info(f"训练完成: {results}")
        
        # 9. 生成报告
        print("\n📋 步骤9: 生成训练报告...")
        generate_training_report(
            experiment_manager, 
            args.experiment_name, 
            results, 
            config
        )
        
        print("✅ 所有步骤完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        if 'logger' in locals():
            logger.error(f"训练失败: {e}", exc_info=True)
        return False

def generate_training_report(
    experiment_manager: ExperimentManager, 
    experiment_name: str, 
    results: Dict[str, Any], 
    config: Dict[str, Any]
):
    """生成训练报告"""
    try:
        experiment = experiment_manager.get_experiment(experiment_name)
        exp_dir = Path("./experiments") / experiment_name
        
        report = {
            "experiment_info": {
                "name": experiment_name,
                "created_at": experiment.get('created_at'),
                "status": experiment.get('status'),
                "description": experiment.get('description')
            },
            "configuration": {
                "model_path": config['model']['local_path'],
                "data_path": config['data']['train_file'],
                "lora_config": config['lora'],
                "training_stages": {
                    "stage1": config['training']['stage1'],
                    "stage2": config['training']['stage2']
                }
            },
            "training_results": results,
            "file_locations": {
                "experiment_dir": str(exp_dir),
                "final_model": results.get('final_model_path'),
                "checkpoints_dir": str(exp_dir / "checkpoints"),
                "logs_dir": str(exp_dir / "logs")
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # 保存报告
        report_file = exp_dir / "results" / "training_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 训练报告已保存: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 生成报告时出错: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

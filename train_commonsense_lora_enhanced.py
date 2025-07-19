#!/usr/bin/env python3
"""
train_commonsense_lora_enhanced.py
增强版训练脚本 - 支持individual和mixed数据集训练
"""

import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

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
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"配置文件加载失败: {e}")

def validate_paths(config: Dict[str, Any]) -> bool:
    """验证配置中的路径"""
    model_path = config['model']['local_path']
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    return True

def get_individual_datasets() -> List[str]:
    """获取individual数据集列表"""
    return [
        'arc-challenge', 'arc-easy', 'boolq', 'hellaswag',
        'openbookqa', 'piqa', 'winogrande'
    ]

def create_individual_config(base_config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """为individual数据集创建配置"""
    config = base_config.copy()
    
    # 更新数据路径
    config['data']['train_file'] = f"data_to_lora/cs/{dataset_name}/{dataset_name}_train_formatted.jsonl"
    
    # 生成时间戳用于实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_lora"
    
    # 更新输出目录 - 使用新的结构 experiments/cs/{dataset_name}/{timestamp_lora}/
    config['training']['output_dir'] = f"./experiments/cs/{dataset_name}/{experiment_name}/models"
    config['checkpoint']['dir'] = f"./experiments/cs/{dataset_name}/{experiment_name}/checkpoints"
    config['logging']['log_dir'] = f"./experiments/cs/{dataset_name}/{experiment_name}/logs"
    
    # 更新实验配置
    config['experiment']['name'] = experiment_name
    config['experiment']['description'] = f"LoRA training on {dataset_name} dataset"
    config['experiment']['tags'] = ["lora", "qwen2.5", dataset_name, "two-stage"]
    
    # 设置实验类型为commonsense_lora
    config['experiment_type'] = "commonsense_lora"
    config['dataset_name'] = dataset_name  # 添加数据集名称字段
    
    # 为实验管理器创建扁平化的配置
    flat_config = {
        "model_path": config['model']['local_path'],
        "data_path": config['data']['train_file'], 
        "lora": config['lora'],
        "training": config['training'],
        "experiment_type": "commonsense_lora"
    }
    
    # 保留原始配置用于其他用途
    config['_flat_config'] = flat_config
    
    return config

def create_mixed_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """为mixed数据集创建配置"""
    config = base_config.copy()
    
    # 更新数据路径
    config['data']['train_file'] = "data_to_lora/cs/mixed/cs_mixed_formatted_train.jsonl"
    
    # 生成时间戳用于实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_lora"
    
    # 更新输出目录 - 使用新的结构 experiments/cs/mixed/{timestamp_lora}/
    config['training']['output_dir'] = f"./experiments/cs/mixed/{experiment_name}/models"
    config['checkpoint']['dir'] = f"./experiments/cs/mixed/{experiment_name}/checkpoints"
    config['logging']['log_dir'] = f"./experiments/cs/mixed/{experiment_name}/logs"
    
    # 更新实验配置
    config['experiment']['name'] = experiment_name
    config['experiment']['description'] = f"LoRA training on mixed commonsense datasets"
    config['experiment']['tags'] = ["lora", "qwen2.5", "mixed", "commonsense", "two-stage"]
    
    # 设置实验类型为commonsense_lora
    config['experiment_type'] = "commonsense_lora"
    config['dataset_name'] = "mixed"  # 添加数据集名称字段
    
    # 为实验管理器创建扁平化的配置
    flat_config = {
        "model_path": config['model']['local_path'],
        "data_path": config['data']['train_file'], 
        "lora": config['lora'],
        "training": config['training'],
        "experiment_type": "commonsense_lora"
    }
    
    # 保留原始配置用于其他用途
    config['_flat_config'] = flat_config
    
    return config

def run_single_experiment(config: Dict[str, Any], dataset_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """运行单个实验"""
    print(f"\n{'=' * 60}")
    print(f"🚀 开始训练: {dataset_name}")
    print(f"{'=' * 60}")
    
    try:
        # 验证数据文件
        data_file = config['data']['train_file']
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"训练数据文件不存在: {data_file}")
        
        logger.info(f"开始训练数据集: {dataset_name}")
        logger.info(f"数据文件: {data_file}")
        
        # 初始化管理器
        experiment_manager = ExperimentManager("./experiments")
        model_manager = ModelManager(config['model']['cache_dir'])
        
        # 验证模型
        model_path = config['model']['local_path']
        compatibility = model_manager.check_model_compatibility(model_path)
        
        if not compatibility['model_valid']:
            raise RuntimeError(f"模型验证失败: {compatibility}")
        
        # 创建实验名称
        experiment_name = config['experiment']['name']  # 使用配置中的实验名称
        dataset_name = config['dataset_name']  # 获取数据集名称
        
        print(f"📋 实验名称: {experiment_name}")
        print(f"📁 数据集: {dataset_name}")
        print(f"📁 数据文件: {data_file}")
        print(f"🎯 输出目录: {config['training']['output_dir']}")
        
        # 先创建实验（使用扁平化配置和数据集名称）
        flat_config = config['_flat_config']
        experiment_dir = experiment_manager.create_experiment(
            name=experiment_name,
            config=flat_config,  # 使用扁平化配置创建实验
            description=config['experiment']['description'],
            tags=config['experiment'].get('tags', []),
            dataset_name=dataset_name  # 传递数据集名称
        )
        
        # 执行训练
        results = experiment_manager.run_commonsense_lora_experiment(experiment_name)
        
        print(f"✅ {dataset_name} 训练完成!")
        print(f"📊 训练结果:")
        print(f"  - 最终模型: {results.get('final_model_path', 'N/A')}")
        print(f"  - 训练步数: {results.get('total_steps', 'N/A')}")
        print(f"  - Checkpoint数: {results.get('checkpoint_summary', {}).get('total_checkpoints', 'N/A')}")
        
        logger.info(f"{dataset_name} 训练完成: {results}")
        
        # 生成报告
        generate_training_report(
            experiment_manager,
            experiment_name,
            results,
            config,
            dataset_name
        )
        
        return {
            'dataset': dataset_name,
            'experiment_name': experiment_name,
            'status': 'success',
            'results': results
        }
        
    except Exception as e:
        print(f"❌ {dataset_name} 训练失败: {e}")
        logger.error(f"{dataset_name} 训练失败: {e}", exc_info=True)
        return {
            'dataset': dataset_name,
            'status': 'failed',
            'error': str(e)
        }

def generate_training_report(
    experiment_manager: ExperimentManager,
    experiment_name: str,
    results: Dict[str, Any],
    config: Dict[str, Any],
    dataset_name: str
):
    """生成训练报告"""
    try:
        experiment = experiment_manager.get_experiment(experiment_name, dataset_name)
        exp_dir = Path("./experiments") / "cs" / dataset_name / experiment_name
        
        report = {
            "experiment_info": {
                "name": experiment_name,
                "dataset": dataset_name,
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
                "logs_dir": str(exp_dir / "logs"),
                "models_dir": str(exp_dir / "models")
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # 保存报告
        results_dir = exp_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        report_file = results_dir / "training_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 训练报告已保存: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 生成报告时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强版Commonsense LoRA训练脚本")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--datasets", nargs='*', 
                       help="要训练的数据集/模式列表。默认训练所有7个individual数据集。可包含: arc-challenge, arc-easy, boolq, hellaswag, openbookqa, piqa, winogrande, mixed")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行，不实际训练")
    parser.add_argument("--validate_only", action="store_true",
                       help="仅验证数据和模型")
    
    args = parser.parse_args()
    
    # 所有可用的individual数据集
    all_individual_datasets = ['arc-challenge', 'arc-easy', 'boolq', 'hellaswag', 
                              'openbookqa', 'piqa', 'winogrande']
    
    # 确定要训练的内容
    if args.datasets is None or len(args.datasets) == 0:
        # 默认：训练所有7个individual数据集
        individual_datasets = all_individual_datasets
        train_mixed = False
        print("🚀 Enhanced Commonsense LoRA Training Script")
        print("=" * 70)
        print(f"配置文件: {args.config}")
        print(f"训练模式: individual (默认 - 所有7个数据集)")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    else:
        # 解析用户指定的数据集
        individual_datasets = [d for d in args.datasets if d in all_individual_datasets]
        train_mixed = 'mixed' in args.datasets
        
        print("🚀 Enhanced Commonsense LoRA Training Script")
        print("=" * 70)
        print(f"配置文件: {args.config}")
        print(f"训练内容: Individual数据集: {individual_datasets}, Mixed: {'是' if train_mixed else '否'}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    train_individual = len(individual_datasets) > 0
    
    try:
        # 1. 加载基础配置
        print("\n📋 步骤1: 加载配置...")
        base_config = load_config(args.config)
        print(f"✅ 配置加载成功")
        
        # 2. 设置日志
        print("\n📝 步骤2: 设置日志系统...")
        log_dir = base_config.get('logging', {}).get('log_dir', './logs')
        logger = setup_logging(log_dir)
        logger.info(f"训练开始 - Individual数据集: {individual_datasets}, Mixed: {train_mixed}")
        print(f"✅ 日志系统已设置")
        
        # 3. 验证基础路径
        print("\n🔍 步骤3: 验证路径...")
        if not validate_paths(base_config):
            logger.error("路径验证失败")
            return False
        logger.info("路径验证通过")
        
        if args.validate_only:
            print("\n✅ 验证完成，退出")
            return True
        
        # 4. 根据模式执行训练
        all_results = []
        
        if train_individual:
            print("\n🎯 步骤4: 执行Individual数据集训练...")
            print(f"📊 训练数据集: {individual_datasets}")
            
            # 逐个训练数据集
            for i, dataset_name in enumerate(individual_datasets):
                print(f"\n{'🔄' * 3} 进度: {i+1}/{len(individual_datasets)} {'🔄' * 3}")
                config = create_individual_config(base_config, dataset_name)
                
                if args.dry_run:
                    print(f"🏃 Dry run: {dataset_name}")
                    print(f"  数据文件: {config['data']['train_file']}")
                    print(f"  输出目录: {config['training']['output_dir']}")
                else:
                    result = run_single_experiment(config, dataset_name, logger)
                    all_results.append(result)
        
        if train_mixed:
            print("\n🎯 执行Mixed数据集训练...")
            config = create_mixed_config(base_config)
            
            if args.dry_run:
                print("🏃 Dry run: mixed")
                print(f"  数据文件: {config['data']['train_file']}")
                print(f"  输出目录: {config['training']['output_dir']}")
            else:
                result = run_single_experiment(config, "mixed", logger)
                all_results.append(result)
        
        if args.dry_run:
            print("\n🏃 Dry run完成，未实际训练")
            return True
        
        # 5. 生成总体报告
        print("\n📋 步骤5: 生成总体训练报告...")
        
        # 确定模式字符串用于报告
        if train_individual and train_mixed:
            mode_str = "individual_and_mixed"
        elif train_mixed:
            mode_str = "mixed"
        else:
            mode_str = "individual"
            
        generate_summary_report(all_results, mode_str)
        
        # 6. 打印总结
        print(f"\n🎉 所有训练完成!")
        print(f"📊 总体统计:")
        successful = sum(1 for r in all_results if r['status'] == 'success')
        failed = len(all_results) - successful
        print(f"  - 成功: {successful}")
        print(f"  - 失败: {failed}")
        print(f"  - 总计: {len(all_results)}")
        
        if failed > 0:
            print(f"\n❌ 失败的数据集:")
            for result in all_results:
                if result['status'] == 'failed':
                    print(f"  - {result['dataset']}: {result.get('error', 'Unknown error')}")
        
        logger.info(f"所有训练完成 - 成功: {successful}, 失败: {failed}")
        return failed == 0
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        if 'logger' in locals():
            logger.error(f"训练失败: {e}", exc_info=True)
        return False

def generate_summary_report(results: List[Dict[str, Any]], mode: str):
    """生成总体报告"""
    try:
        summary = {
            "training_mode": mode,
            "total_experiments": len(results),
            "successful": sum(1 for r in results if r['status'] == 'success'),
            "failed": sum(1 for r in results if r['status'] == 'failed'),
            "results": results,
            "generated_at": datetime.now().isoformat()
        }
        
        # 保存总体报告
        report_dir = Path("./experiments/cs")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"training_summary_{mode}_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 总体报告已保存: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 生成总体报告时出错: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
validate_setup.py
验证脚本 - 用于手动验证LoRA训练环境和组件
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    if exists:
        size = os.path.getsize(file_path)
        print(f"    大小: {size:,} bytes")
    return exists

def check_directory_exists(dir_path, description):
    """检查目录是否存在"""
    exists = os.path.exists(dir_path) and os.path.isdir(dir_path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dir_path}")
    if exists:
        files = list(Path(dir_path).iterdir())
        print(f"    包含 {len(files)} 个文件/目录")
    return exists

def test_imports():
    """测试导入"""
    print_section("测试Python包导入")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM进度条"),
        ("json", "JSON (内置)"),
        ("logging", "Logging (内置)")
    ]
    
    optional_imports = [
        ("peft", "PEFT (LoRA支持)"),
        ("wandb", "Weights & Biases"),
        ("tensorboard", "TensorBoard")
    ]
    
    all_success = True
    
    # 必需的导入
    for module, description in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {description}: 可用")
        except ImportError as e:
            print(f"❌ {description}: 不可用 ({e})")
            all_success = False
    
    # 可选的导入
    print(f"\n可选依赖:")
    for module, description in optional_imports:
        try:
            __import__(module)
            print(f"✅ {description}: 可用")
        except ImportError:
            print(f"⚠️ {description}: 不可用 (可选)")
    
    return all_success

def test_project_structure():
    """测试项目结构"""
    print_section("验证项目文件结构")
    
    required_files = [
        ("configs/training_config.yaml", "训练配置文件"),
        ("utils/data_processor.py", "数据处理器"),
        ("utils/scheduler.py", "学习率调度器"),
        ("lora/checkpoint_utils.py", "Checkpoint管理"),
        ("core/train.py", "训练核心"),
        ("scripts/experiment_manager_enhanced.py", "实验管理器"),
        ("scripts/model_manager.py", "模型管理器"),
        ("train_commonsense_lora.py", "主训练脚本")
    ]
    
    required_dirs = [
        ("models", "模型目录"),
        ("raw_datasets/commonsense", "数据集目录"),
        ("utils", "工具目录"),
        ("lora", "LoRA目录"),
        ("core", "核心目录"),
        ("scripts", "脚本目录")
    ]
    
    all_files_exist = True
    all_dirs_exist = True
    
    print("必需文件:")
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    print(f"\n必需目录:")
    for dir_path, description in required_dirs:
        if not check_directory_exists(dir_path, description):
            all_dirs_exist = False
    
    return all_files_exist and all_dirs_exist

def test_model_and_data():
    """测试模型和数据"""
    print_section("验证模型和数据文件")
    
    # 检查模型文件
    model_path = "models/Qwen-Qwen2.5-0.5B"
    model_files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "tokenizer_config.json"
    ]
    
    print("模型文件:")
    model_valid = check_directory_exists(model_path, "Qwen2.5模型目录")
    
    if model_valid:
        for file_name in model_files:
            file_path = os.path.join(model_path, file_name)
            check_file_exists(file_path, f"  {file_name}")
    
    # 检查数据文件
    print(f"\n数据文件:")
    data_path = "raw_datasets/commonsense/cs_all_unbalanced.jsonl"
    data_valid = check_file_exists(data_path, "Commonsense数据集")
    
    if data_valid:
        # 检查数据格式 - 支持两种格式
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    sample = json.loads(first_line)
                    print(f"📝 数据样本调试信息:")
                    print(f"    实际键: {sorted(sample.keys())}")
                    print(f"    样本内容预览: {dict(list(sample.items())[:3])}")
                    
                    # 检查标准格式
                    standard_keys = ['instruction', 'input', 'output']
                    has_standard = all(key in sample for key in standard_keys)
                    
                    # 检查commonsense格式
                    commonsense_keys = ['id', 'dataset', 'task_type', 'input', 'options', 'target']
                    has_commonsense = all(key in sample for key in commonsense_keys)
                    
                    print(f"    标准格式检查 ({standard_keys}):")
                    for key in standard_keys:
                        status = "✅" if key in sample else "❌"
                        print(f"      {status} {key}")
                    
                    print(f"    Commonsense格式检查 ({commonsense_keys}):")
                    for key in commonsense_keys:
                        status = "✅" if key in sample else "❌"
                        print(f"      {status} {key}")
                    
                    if has_standard:
                        print("✅ 数据格式验证通过 (标准格式)")
                    elif has_commonsense:
                        print("✅ 数据格式验证通过 (Commonsense格式)")
                        print(f"    任务类型: {sample.get('task_type', 'N/A')}")
                        print(f"    数据集: {sample.get('dataset', 'N/A')}")
                        print(f"    选项数量: {len(sample.get('options', []))}")
                    else:
                        print("❌ 数据格式验证失败")
                        missing_standard = set(standard_keys) - set(sample.keys())
                        missing_commonsense = set(commonsense_keys) - set(sample.keys())
                        print(f"    缺少标准格式键: {missing_standard}")
                        print(f"    缺少Commonsense格式键: {missing_commonsense}")
                        data_valid = False
                        data_valid = False
        except Exception as e:
            print(f"❌ 数据格式检查失败: {e}")
            data_valid = False
    
    return model_valid and data_valid

def test_configuration():
    """测试配置文件"""
    print_section("验证配置文件")
    
    config_path = "configs/training_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置
        required_sections = [
            'model',
            'data', 
            'lora',
            'training',
            'checkpoint'
        ]
        
        all_sections_present = True
        for section in required_sections:
            if section in config:
                print(f"✅ 配置节 '{section}': 存在")
            else:
                print(f"❌ 配置节 '{section}': 缺失")
                all_sections_present = False
        
        # 检查关键路径
        if 'model' in config and 'local_path' in config['model']:
            model_path = config['model']['local_path']
            if os.path.exists(model_path):
                print(f"✅ 配置中的模型路径: 有效")
            else:
                print(f"❌ 配置中的模型路径: 无效 ({model_path})")
                all_sections_present = False
        
        if 'data' in config and 'train_file' in config['data']:
            data_path = config['data']['train_file']
            if os.path.exists(data_path):
                print(f"✅ 配置中的数据路径: 有效")
            else:
                print(f"❌ 配置中的数据路径: 无效 ({data_path})")
                all_sections_present = False
        
        return all_sections_present
        
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return False

def test_training_components():
    """测试训练组件导入"""
    print_section("测试训练组件")
    
    # 添加项目路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    components_to_test = [
        ("utils.data_processor", "DataProcessor", "数据处理器"),
        ("utils.scheduler", "TwoStageScheduler", "两阶段调度器"),
        ("lora.checkpoint_utils", "CheckpointManager", "Checkpoint管理器"),
        ("scripts.model_manager", "ModelManager", "模型管理器")
    ]
    
    all_components_ok = True
    
    for module_name, class_name, description in components_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            component_class = getattr(module, class_name)
            print(f"✅ {description}: 可导入")
        except ImportError as e:
            print(f"❌ {description}: 导入失败 ({e})")
            all_components_ok = False
        except AttributeError as e:
            print(f"❌ {description}: 类不存在 ({e})")
            all_components_ok = False
        except Exception as e:
            print(f"❌ {description}: 其他错误 ({e})")
            all_components_ok = False
    
    return all_components_ok

def generate_validation_report():
    """生成验证报告"""
    print_section("生成验证报告")
    
    # 运行所有测试
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {
            "imports": test_imports(),
            "project_structure": test_project_structure(),
            "model_and_data": test_model_and_data(),
            "configuration": test_configuration(),
            "training_components": test_training_components()
        }
    }
    
    # 计算总体结果
    all_passed = all(results["tests"].values())
    results["overall_status"] = "PASS" if all_passed else "FAIL"
    
    # 保存报告
    report_file = "validation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print_section("验证总结")
    print(f"总体状态: {'✅ 通过' if all_passed else '❌ 失败'}")
    print(f"详细报告已保存: {report_file}")
    
    for test_name, status in results["tests"].items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {test_name}: {'通过' if status else '失败'}")
    
    # 为失败的测试提供额外信息
    if not all_passed:
        print(f"\n🔍 失败项目详细信息:")
        failed_tests = [name for name, status in results["tests"].items() if not status]
        
        for test_name in failed_tests:
            print(f"\n❌ {test_name} 失败:")
            
            if test_name == "model_and_data":
                print(f"   - 请检查模型文件是否完整（config.json, model.safetensors等）")
                print(f"   - 请确认数据文件格式正确（支持commonsense格式）")
                print(f"   - 数据文件路径: raw_datasets/commonsense/cs_all_unbalanced.jsonl")
                print(f"   - 预期数据键: ['id', 'dataset', 'task_type', 'input', 'options', 'target']")
            
            elif test_name == "imports":
                print(f"   - 请安装缺失的Python包")
                print(f"   - 运行: conda install peft transformers torch")
            
            elif test_name == "project_structure":
                print(f"   - 请检查项目文件结构是否完整")
                print(f"   - 确认所有必需的脚本和配置文件存在")
            
            elif test_name == "configuration":
                print(f"   - 请检查configs/training_config.yaml配置文件")
                print(f"   - 确认所有必需的配置节存在")
            
            elif test_name == "training_components":
                print(f"   - 请检查训练组件是否能正常导入")
                print(f"   - 可能存在Python语法错误或依赖问题")
    
    if all_passed:
        print(f"\n🎉 所有验证通过！可以开始训练。")
        print(f"运行命令: python train_commonsense_lora.py --validate_only")
    else:
        print(f"\n⚠️ 存在问题，请先解决后再开始训练。")
        print(f"💡 提示: 查看上方的详细错误信息，或检查 {report_file} 获取完整报告")
    
    return all_passed

def main():
    """主函数"""
    print("🔍 P2W项目 - LoRA训练环境验证")
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = generate_validation_report()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

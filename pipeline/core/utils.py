"""
工具模块
包含通用的工具函数和命令执行器
"""

import os
import re
import subprocess
import numpy as np
from datetime import datetime
from typing import Any, Optional, Dict


class ModelUtils:
    """模型相关工具函数"""
    
    @staticmethod
    def get_model_short_name(model_path: str) -> str:
        """获取模型简称"""
        model_name = os.path.basename(model_path)
        # 移除常见前缀
        model_name = model_name.replace("models/", "")
        return model_name
    
    @staticmethod
    def create_experiment_id(source_model: str, target_model: str, dataset: str, timestamp: str) -> str:
        """创建实验ID"""
        source_short = ModelUtils.get_model_short_name(source_model)
        target_short = ModelUtils.get_model_short_name(target_model)
        return f"{source_short}_to_{target_short}_{dataset}_{timestamp}"
    
    @staticmethod
    def check_model_exists(model_path: str) -> bool:
        """检查模型是否存在"""
        return os.path.exists(model_path) and os.path.isdir(model_path)


class CommandRunner:
    """命令执行器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def run_command(self, cmd: str, description: str, cwd: str = None) -> Optional[str]:
        """运行命令并返回结果"""
        if self.verbose:
            print(f"\n🚀 {description}")
            print(f"📝 命令: {cmd}")
        
        try:
            # 设置环境变量，保留训练相关输出
            env = os.environ.copy()
            env.update({
                'PYTHONUNBUFFERED': '1',  # 确保Python输出不被缓冲
                'PYTHONIOENCODING': 'utf-8'
            })
            
            # 启动进程，使用实时输出
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并stderr到stdout
                text=True,
                cwd=cwd,
                bufsize=0,  # 无缓冲
                universal_newlines=True,
                env=env
            )
            
            stdout_lines = []
            current_progress_line = None
            
            # 实时处理输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    stdout_lines.append(line)
                    
                    if self.verbose:
                        self._process_output_line(line, current_progress_line)
            
            # 确保最后换行
            if current_progress_line and self.verbose:
                print()
            
            # 等待进程结束
            process.wait()
            
            if process.returncode != 0:
                if self.verbose:
                    print(f"❌ 命令失败 (退出码: {process.returncode})")
                    # 显示最后几行输出作为错误信息
                    if stdout_lines:
                        print("最后的输出:")
                        for line in stdout_lines[-5:]:
                            if line.strip():
                                print(f"   {line}")
                return None
            
            if self.verbose:
                print(f"✅ {description} 完成")
            return '\n'.join(stdout_lines)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 执行命令时出错: {e}")
            return None
    
    def _process_output_line(self, line: str, current_progress_line: Optional[str]):
        """处理输出行，控制显示内容"""
        # 显示重要的SwanLab信息（包括链接和基本状态）
        if any(keyword in line for keyword in [
            'swanlab.cn', 'View run at', 'View project at',
            'Tracking run with swanlab', 'Syncing run'
        ]):
            print(f"🔗 {line}")
            return
        
        # 显示训练开始信息
        if any(keyword in line for keyword in [
            '🚀 Lightning LoRA 训练', '开始Lightning训练',
            'Lightning训练完成', '实验目录:', '最终模型:'
        ]):
            print(f"   {line}")
            return
        
        # 处理进度条 - 只显示最新的进度
        if re.search(r'\d+%\|[█▉▊▋▌▍▎▏ ]*\|', line) or ('it/s' in line and ('Epoch' in line or 'step' in line)):
            if current_progress_line:
                # 清除之前的进度行
                print('\r' + ' ' * len(current_progress_line) + '\r', end='', flush=True)
            print(f"\r📊 {line}", end='', flush=True)
            current_progress_line = line
        else:
            # 如果有之前的进度行，先换行
            if current_progress_line:
                print()  # 换行
                current_progress_line = None
            
            # 显示重要的状态信息，但过滤噪音
            if not any(noise in line for noise in [
                'LOCAL_RANK', 'CUDA_VISIBLE_DEVICES', 'Sanity Checking DataLoader',
                'generation flags are not valid', 'cache_implementation',
                'Using 16bit Automatic Mixed Precision', 'GPU available:', 'TPU available:'
            ]) and line.strip():
                # 显示有用的信息
                if any(useful in line for useful in [
                    '✅', '📊', '🎯', '📁', '⚠️', '❌', 
                    'accuracy', 'loss', 'test_result', 'final_model',
                    'Lightning训练', '模型加载', '可训练参数'
                ]):
                    print(f"   {line}")


class DataCleaner:
    """数据清理工具"""
    
    @staticmethod
    def deep_clean_value(value: Any) -> Any:
        """深度清理数据，确保所有numpy数组都被转换为Python原生类型"""
        try:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return value.item()  # 转换单元素数组为标量
                else:
                    return value.tolist()  # 转换多元素数组为列表
            elif isinstance(value, (np.integer, np.floating)):
                return value.item()  # 转换numpy标量为Python原生类型
            elif isinstance(value, dict):
                return {k: DataCleaner.deep_clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [DataCleaner.deep_clean_value(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(DataCleaner.deep_clean_value(v) for v in value)
            elif hasattr(value, 'tolist'):  # 任何有tolist方法的对象
                return value.tolist()
            elif hasattr(value, 'item'):  # 任何有item方法的对象
                return value.item()
            elif isinstance(value, float) and (value != value):  # NaN
                return None
            elif value is None or isinstance(value, (int, float, str, bool)):
                return value
            else:
                return str(value)
        except Exception as e:
            # 如果转换失败，记录错误并返回字符串表示
            print(f"⚠️ 数据清理警告: {value} ({type(value)}) -> {str(value)}")
            return str(value)
    
    @staticmethod
    def clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """清理字典中的所有值"""
        clean_data = {}
        for key, value in data.items():
            clean_data[key] = DataCleaner.deep_clean_value(value)
        return clean_data


class OutputParser:
    """输出解析器"""
    
    @staticmethod
    def parse_training_accuracy(output: str) -> Optional[float]:
        """解析训练输出，提取准确率"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'test/accuracy' in line and 'test_result' in line:
                    # 提取类似 'test/accuracy': 0.7465870380401611 的信息
                    match = re.search(r"'test/accuracy':\s*([\d.]+)", line)
                    if match:
                        return float(match.group(1))
            return None
        except Exception as e:
            print(f"⚠️ 解析训练输出时出错: {e}")
            return None
    
    @staticmethod
    def parse_evaluation_accuracy(output: str) -> Optional[float]:
        """解析评估输出获取准确率"""
        try:
            # 方法1: 查找表格格式的accuracy
            table_pattern = r"\|\s*test/accuracy\s*\|\s*([\d.]+)\s*\|"
            match = re.search(table_pattern, output, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
                print(f"   📊 从表格提取准确率: {accuracy:.4f}")
                return accuracy
            
            # 方法2: 查找字典格式的accuracy
            dict_pattern = r"['\"]?test/accuracy['\"]?\s*[:\|]\s*([\d.]+)"
            match = re.search(dict_pattern, output, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
                print(f"   📊 从字典提取准确率: {accuracy:.4f}")
                return accuracy
            
            # 方法3: 查找一般accuracy信息
            general_pattern = r"accuracy['\"]?\s*[:\|=]\s*([\d.]+)"
            match = re.search(general_pattern, output, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
                print(f"   📊 从一般格式提取准确率: {accuracy:.4f}")
                return accuracy
            
            # 方法4: 逐行分析
            lines = output.split('\n')
            for line in lines:
                line = line.strip()
                if 'accuracy' in line.lower() and any(char.isdigit() for char in line):
                    # 提取行中的所有数字
                    numbers = re.findall(r'\d+\.?\d*', line)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            # 准确率通常在0-1之间
                            if 0 <= num <= 1:
                                print(f"   📊 从行提取准确率: {num:.4f} (行: {line[:50]}...)")
                                return num
                        except:
                            continue
            
            print(f"   ⚠️ 未能提取准确率，返回None")
            # 调试：显示评估输出的关键部分
            print("   🔍 评估输出关键行:")
            for line in output.split('\n')[-20:]:  # 显示最后20行
                if any(keyword in line.lower() for keyword in ['accuracy', 'test', 'loss']):
                    print(f"     {line.strip()}")
            
        except Exception as e:
            print(f"⚠️ 解析评估输出时出错: {e}")
        
        return None


def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

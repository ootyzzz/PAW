import torch
import platform
import subprocess
import re
import sys
import os

def check_environment_consistency():
    """快速检查环境一致性"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if not conda_env:
        return False, "未激活conda环境"
    
    # 检查Python路径
    expected_path = f"C:\\Users\\feifa\\.conda\\envs\\{conda_env}\\python.exe"
    if sys.executable.lower() != expected_path.lower():
        return False, f"Python路径不匹配，当前: {sys.executable}"
    
    # 检查命令行一致性
    try:
        cmd_version = subprocess.check_output("python --version", shell=True).decode("utf-8").strip()
        script_version = f"Python {platform.python_version()}"
        if cmd_version != script_version:
            return False, f"版本不一致: 命令行({cmd_version}) vs 脚本({script_version})"
    except:
        return False, "无法检查命令行python版本"
    
    return True, "环境一致"

def check_cuda_compatibility():
    """检查RTX 5060 GPU的CUDA兼容性和可用性"""
    print("===== RTX 5060 CUDA环境验证脚本 v2.0 =====")
    
    # 环境一致性快速检查
    print("【环境一致性检查】")
    is_consistent, message = check_environment_consistency()
    if is_consistent:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        print("💡 建议: conda deactivate && conda activate cuda312")
        print("=" * 50)
        return  # 环境有问题就直接返回
    
    # 系统信息
    print(f"系统信息: {platform.system()} {platform.version()} ({platform.machine()})")
    print("=" * 50)
    
    # Python环境详细信息
    print("【Python环境信息】")
    print(f"Python版本: {platform.python_version()}")
    print(f"Python可执行文件路径: {sys.executable}")
    
    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"当前Conda环境: {conda_env}")
        
        # 检查环境一致性
        expected_path = f"C:\\Users\\feifa\\.conda\\envs\\{conda_env}\\python.exe"
        if sys.executable.lower() != expected_path.lower():
            print(f"⚠️  警告: Python路径与conda环境不匹配!")
            print(f"    期望路径: {expected_path}")
            print(f"    实际路径: {sys.executable}")
            print(f"    建议: 运行 'conda deactivate && conda activate {conda_env}' 重新激活环境")
        else:
            print("✅ Python路径与conda环境匹配")
    else:
        print("当前环境: 系统Python (非Conda环境)")
    
    # 检查虚拟环境
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        print(f"虚拟环境路径: {venv_path}")
    
    # 显示Python路径和命令行检查
    print(f"Python库路径: {sys.path[0] if sys.path else '未知'}")
    
    # 检查命令行python版本一致性
    try:
        cmd_python_version = subprocess.check_output("python --version", shell=True).decode("utf-8").strip()
        script_python_version = f"Python {platform.python_version()}"
        
        print(f"命令行python版本: {cmd_python_version}")
        print(f"脚本运行python版本: {script_python_version}")
        
        if cmd_python_version != script_python_version:
            print("⚠️  警告: 命令行python版本与脚本运行版本不一致!")
            print("    这可能导致环境配置问题，建议重新激活conda环境")
        else:
            print("✅ 命令行python版本与脚本运行版本一致")
    except Exception as e:
        print(f"无法检查命令行python版本: {e}")
    print("=" * 50)
    
    # 获取NVIDIA驱动信息
    print("【NVIDIA驱动信息】")
    try:
        if platform.system() == "Windows":
            result = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        else:
            result = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"]).decode("utf-8")
        
        driver_version = re.search(r"Driver Version: (\d+\.\d+)", result)
        if driver_version:
            driver_version = driver_version.group(1)
            print(f"驱动版本: {driver_version}")
            
            # 提取CUDA版本
            cuda_version_match = re.search(r"CUDA Version: (\d+\.\d+)", result)
            cuda_version = cuda_version_match.group(1) if cuda_version_match else "未知"
            print(f"支持的最高CUDA版本: {cuda_version}")
        else:
            print("无法获取驱动版本信息")
    except Exception as e:
        print(f"获取驱动信息失败: {e}")
    
    # 打印GPU列表
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader,nounits", shell=True).decode("utf-8").strip()
        print("GPU列表:")
        gpu_list = gpu_info.split('\n')
        for i, gpu in enumerate(gpu_list):
            name, uuid, memory = gpu.strip().split(', ')
            print(f"  - GPU{i}: {name} | 显存: {memory} MiB | UUID: {uuid}")
    except:
        print("  - 无法获取GPU详细信息")
    print("=" * 50)
    
    # CUDA工具包版本
    print("【CUDA工具包版本】")
    try:
        nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
        cuda_version = re.search(r"release (\d+\.\d+)", nvcc_output).group(1)
        print(f"release {cuda_version}")
    except Exception as e:
        print(f"无法获取CUDA工具包版本: {e}")
    print("=" * 50)
    
    # PyTorch CUDA支持验证
    print("【PyTorch CUDA支持验证】")
    try:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"PyTorch编译时的CUDA版本: {torch.version.cuda}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU设备数量: {torch.cuda.device_count()}")
            
            # 检查每个GPU
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                compute_capability = torch.cuda.get_device_properties(i).major * 10 + torch.cuda.get_device_properties(i).minor
                
                print(f"GPU {i} 详情:")
                print(f"  - 名称: {gpu_name}")
                print(f"  - 显存: {gpu_memory:.2f} GB")
                print(f"  - 计算能力: sm_{compute_capability}")
                
                # 检查RTX 5060特定信息
                if "RTX 5060" in gpu_name:
                    print("  - ✅ 检测到RTX 5060 GPU")
                    
                    # 验证CUDA功能
                    try:
                        # 创建一个简单的张量并在GPU上执行操作
                        x = torch.tensor([1.0], device=f"cuda:{i}")
                        y = torch.tensor([2.0], device=f"cuda:{i}")
                        z = x + y
                        
                        # 检查计算结果
                        if z.item() == 3.0:
                            print("  - ✅ CUDA计算验证成功")
                        else:
                            print(f"  - ❌ CUDA计算结果异常: {z.item()} (预期3.0)")
                    except Exception as e:
                        print(f"  - ❌ GPU运算失败: {e}")
                        print("    提示: 尝试设置环境变量 CUDA_LAUNCH_BLOCKING=1 以获取更详细的错误信息")
    except Exception as e:
        print(f"验证PyTorch CUDA支持时出错: {e}")
    print("=" * 50)
    
    # 结论
    if torch.cuda.is_available():
        print("【验证完成】")
        print("结论: CUDA环境正常，GPU加速可用")
    else:
        print("【验证完成】")
        print("结论: CUDA环境配置存在问题，无法使用GPU加速")
        print("建议: 检查PyTorch版本是否与CUDA工具包兼容，或更新NVIDIA驱动")

if __name__ == "__main__":
    check_cuda_compatibility()
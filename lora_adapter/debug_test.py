#!/usr/bin/env python3
"""
简化的测试脚本
"""

import os
import sys
from pathlib import Path

# 添加正确的路径
lora_adapter_dir = Path("/root/PAW/lora_adapter")
src_dir = lora_adapter_dir / "src"
sys.path.insert(0, str(src_dir))

print("Python路径:")
for p in sys.path[:5]:
    print(f"  {p}")

print(f"\\nsrc目录存在: {src_dir.exists()}")
print(f"文件列表: {list(src_dir.glob('*.py'))}")

# 测试导入
try:
    import model_utils
    print("✅ 成功导入model_utils")
except Exception as e:
    print(f"❌ 导入model_utils失败: {e}")

try:
    import lora_x_core
    print("✅ 成功导入lora_x_core")
except Exception as e:
    print(f"❌ 导入lora_x_core失败: {e}")

# 测试LoRA路径
source_lora_path = "/root/PAW/runs/Qwen_Qwen2.5-1.5B/arc-challenge_lora_20250723_191421/final_model"
print(f"\\nLoRA路径存在: {Path(source_lora_path).exists()}")
if Path(source_lora_path).exists():
    files = list(Path(source_lora_path).glob("*"))
    print(f"LoRA文件: {[f.name for f in files]}")

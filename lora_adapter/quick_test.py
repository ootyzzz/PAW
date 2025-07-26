#!/usr/bin/env python3
"""
LoRA-X实际功能测试
"""

import sys
import logging
from pathlib import Path

# 设置路径
lora_adapter_dir = Path(__file__).parent
src_dir = lora_adapter_dir / "src"
sys.path.insert(0, str(src_dir))

import torch
from model_utils import ModelWeightLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_lora_loading():
    """测试LoRA权重加载"""
    print("🔍 测试LoRA权重加载...")
    
    loader = ModelWeightLoader()
    source_lora_path = "../train_lora/runs/Qwen_Qwen2.5-1.5B/arc-challenge_lora_20250723_191421/final_model"
    
    try:
        lora_weights, config = loader.load_lora_weights(source_lora_path)
        
        print(f"✅ 成功加载LoRA: {len(lora_weights)}个参数")
        print(f"✅ 配置: rank={config.get('r')}, alpha={config.get('lora_alpha')}")
        
        # 显示权重信息
        print("\\n📊 权重详情:")
        for i, (key, weight) in enumerate(lora_weights.items()):
            print(f"  {key}: {weight.shape} ({weight.dtype})")
            if i >= 3:
                print(f"  ... (+{len(lora_weights)-4}个更多权重)")
                break
        
        return lora_weights, config
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_simple_transfer():
    """测试简单的迁移逻辑"""
    print("\\n🔬 测试核心迁移逻辑...")
    
    from lora_x_core import LoRAXCore
    
    # 创建测试数据
    print("创建测试权重矩阵...")
    source_weight = torch.randn(64, 128, dtype=torch.float32)
    target_weight = torch.randn(64, 128, dtype=torch.float32)
    lora_delta = torch.randn(64, 128, dtype=torch.float32) * 0.1  # 小的变化
    
    lora_x = LoRAXCore(rank=32, similarity_threshold=0.1)
    
    try:
        # 测试SVD分解
        U_s, S_s, Vh_s = lora_x.compute_svd_subspace(source_weight)
        U_t, S_t, Vh_t = lora_x.compute_svd_subspace(target_weight)
        
        print(f"✅ SVD分解: U_s={U_s.shape}, U_t={U_t.shape}")
        
        # 测试相似性计算
        similarity = lora_x.compute_subspace_similarity(U_s, U_t)
        print(f"✅ 子空间相似性: {similarity:.3f}")
        
        # 测试单层迁移
        transferred = lora_x._transfer_single_layer(lora_delta, source_weight, target_weight)
        print(f"✅ 层迁移: {lora_delta.shape} -> {transferred.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 LoRA-X核心功能测试")
    print("=" * 60)
    
    # 测试1: LoRA加载
    lora_weights, config = test_lora_loading()
    
    # 测试2: 核心算法
    test_simple_transfer()
    
    print("\\n" + "=" * 60)
    
    if lora_weights is not None:
        print("✅ 基础功能正常，可以尝试实际迁移！")
        print("\\n📋 下一步:")
        print("  1. 运行完整迁移: python scripts/transfer_lora_x.py")
        print("  2. 评估迁移结果")
        print("  3. 记录实验数据")
    else:
        print("❌ 基础功能有问题，需要修复")

if __name__ == "__main__":
    main()

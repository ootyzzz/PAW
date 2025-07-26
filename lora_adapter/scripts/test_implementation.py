#!/usr/bin/env python3
"""
LoRA-X迁移测试脚本
"""

import sys
import logging
from pathlib import Path

# 设置路径
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent / "src"))

# 设置简单日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_lora_loading():
    """测试LoRA权重加载"""
    from model_utils import ModelWeightLoader
    
    loader = ModelWeightLoader()
    
    # 测试路径
    source_lora_path = "../../train_lora/runs/Qwen_Qwen2.5-1.5B/arc-challenge_lora_20250723_191421/final_model"
    
    print("测试LoRA权重加载...")
    try:
        lora_weights, config = loader.load_lora_weights(source_lora_path)
        print(f"✅ 成功加载LoRA权重: {len(lora_weights)}个参数")
        print(f"✅ LoRA配置: {config}")
        
        # 显示一些权重信息
        print("\\n权重详情:")
        for i, (key, weight) in enumerate(lora_weights.items()):
            print(f"  {key}: {weight.shape}")
            if i >= 5:  # 只显示前5个
                print(f"  ... 还有{len(lora_weights)-6}个权重")
                break
                
        return True
    except Exception as e:
        print(f"❌ LoRA加载失败: {e}")
        return False

def test_model_analysis():
    """测试模型结构分析"""
    from model_utils import ModelWeightLoader
    
    loader = ModelWeightLoader()
    
    # 测试路径
    qwen_path = "../../../autodl-tmp/models/Qwen_Qwen2.5-1.5B"
    gemma_path = "../../../autodl-tmp/models/gemma-2-2b-it"
    
    print("\\n测试模型结构分析...")
    
    try:
        print("分析Qwen模型...")
        qwen_attn_layers = loader.get_attention_layer_names(qwen_path)
        print(f"✅ Qwen attention层: {len(qwen_attn_layers)}个")
        
        print("\\n分析Gemma模型...")
        gemma_attn_layers = loader.get_attention_layer_names(gemma_path)
        print(f"✅ Gemma attention层: {len(gemma_attn_layers)}个")
        
        # 显示层名称示例
        print("\\nQwen层名称示例:")
        for layer in qwen_attn_layers[:3]:
            print(f"  {layer}")
            
        print("\\nGemma层名称示例:")
        for layer in gemma_attn_layers[:3]:
            print(f"  {layer}")
            
        return True
    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 LoRA-X实现测试")
    print("=" * 50)
    
    # 测试LoRA加载
    test1 = test_lora_loading()
    
    # 测试模型分析  
    test2 = test_model_analysis()
    
    print("\\n" + "=" * 50)
    if test1 and test2:
        print("✅ 所有测试通过！可以进行迁移实验")
    else:
        print("❌ 部分测试失败，需要修复问题")

if __name__ == "__main__":
    main()

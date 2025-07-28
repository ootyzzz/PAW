#!/usr/bin/env python3
"""
自定义LoRA迁移脚本
从Meta-Llama-3.1-8B-Instruct迁移LoRA到Qwen2.5-7B-Instruct
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import torch

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from lora_x_core import LoRAXCore
from model_utils import ModelWeightLoader, save_transferred_lora

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_timestamp():
    """生成时间戳格式: YYMMDD_HHMMSS"""
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M%S")


def infer_source_model_path(lora_path: str) -> str:
    """根据LoRA路径推断源模型路径"""
    # 从 /root/autodl-tmp/loraed/Meta-Llama-3.1-8B-Instruct/250728_010944/final_model
    # 推断为 /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct
    lora_path = Path(lora_path)
    
    # 如果路径以final_model结尾，向上两级获取模型名
    if lora_path.name == "final_model":
        # /root/autodl-tmp/loraed/Meta-Llama-3.1-8B-Instruct/250728_010944/final_model
        # 向上两级: Meta-Llama-3.1-8B-Instruct
        model_name = lora_path.parent.parent.name  
    else:
        # 否则向上一级获取模型名
        model_name = lora_path.parent.name  
    
    source_model_path = f"/root/autodl-tmp/models/{model_name}"
    return source_model_path


def print_layer_details(lora_key: str, base_key: str, similarity: float, 
                       source_shape: tuple, target_shape: tuple, 
                       lora_shape: tuple, status: str, reason: str = ""):
    """打印层的详细信息"""
    print(f"\n{'='*80}")
    print(f"🔍 层分析: {lora_key}")
    print(f"{'='*80}")
    print(f"📋 基础权重键: {base_key}")
    print(f"📊 相似度分数: {similarity:.4f}")
    print(f"📐 源模型形状: {source_shape}")
    print(f"📐 目标模型形状: {target_shape}")
    print(f"📐 LoRA权重形状: {lora_shape}")
    print(f"🎯 处理状态: {status}")
    if reason:
        print(f"💡 原因: {reason}")
    print(f"{'='*80}")


def print_transfer_summary(stats: dict):
    """打印迁移总结"""
    print(f"\n{'🎉'*20} 迁移完成总结 {'🎉'*20}")
    print(f"{'='*80}")
    print(f"📊 总层数: {stats['total_layers']}")
    print(f"✅ 成功迁移: {stats['transferred_layers']}")
    print(f"❌ 跳过层数: {len(stats['skipped_layers'])}")
    
    if stats['transferred_layers'] > 0:
        success_rate = (stats['transferred_layers'] / stats['total_layers']) * 100
        print(f"📈 迁移成功率: {success_rate:.1f}%")
    else:
        print(f"⚠️  警告: 没有成功迁移任何层!")
    
    print(f"\n🎯 成功迁移的层:")
    for i, layer in enumerate(stats.get('transferred_list', []), 1):
        print(f"  {i:2d}. {layer}")
    
    if stats['skipped_layers']:
        print(f"\n❌ 跳过的层及原因:")
        for i, layer in enumerate(stats['skipped_layers'], 1):
            reason = stats['skipped_reasons'].get(layer, "未知原因")
            print(f"  {i:2d}. {layer}")
            print(f"      原因: {reason}")
    
    # 按层类型统计
    if 'layer_types' in stats:
        print(f"\n📋 按层类型统计:")
        for layer_type, type_stats in stats['layer_types'].items():
            total = type_stats['total']
            transferred = type_stats['transferred']
            rate = (transferred / total * 100) if total > 0 else 0
            print(f"  {layer_type:12s}: {transferred:2d}/{total:2d} ({rate:5.1f}%)")
    
    print(f"{'='*80}")


class VerboseLoRAXCore(LoRAXCore):
    """增强版LoRA-X核心类，提供详细输出"""
    
    def transfer_lora_weights(self, 
                            source_lora: dict,
                            target_base_weights: dict,
                            source_base_weights: dict) -> dict:
        """执行LoRA-X迁移，提供详细输出"""
        transferred_lora = {}
        transfer_stats = {
            'total_layers': 0,
            'transferred_layers': 0, 
            'transferred_list': [],
            'skipped_layers': [],
            'skipped_reasons': {},
            'similarity_stats': [],
            'layer_types': {}
        }
        
        print(f"\n🚀 开始LoRA权重迁移...")
        print(f"📊 源LoRA包含 {len(source_lora)} 个权重")
        print(f"📊 源模型包含 {len(source_base_weights)} 个基础权重")
        print(f"📊 目标模型包含 {len(target_base_weights)} 个基础权重")
        
        # 预计算所有相似度
        print(f"\n🔄 预计算层相似度...")
        similarities = self._precompute_similarities_verbose(source_lora, source_base_weights, target_base_weights)
        
        # 遍历源LoRA权重
        for lora_key in source_lora.keys():
            if not lora_key.endswith('.weight'):
                continue

            transfer_stats['total_layers'] += 1
            
            # 分析层类型
            layer_type = self._classify_layer_type(lora_key)
            if layer_type not in transfer_stats['layer_types']:
                transfer_stats['layer_types'][layer_type] = {'total': 0, 'transferred': 0}
            transfer_stats['layer_types'][layer_type]['total'] += 1

            # 找到对应的基础权重
            base_key = self._map_lora_to_base_key(lora_key)

            if base_key not in source_base_weights or base_key not in target_base_weights:
                reason = "找不到对应的基础权重"
                similarity = 0.0
                source_shape = "N/A"
                target_shape = "N/A"
                lora_shape = tuple(source_lora[lora_key].shape)
                
                print_layer_details(lora_key, base_key, similarity, 
                                  source_shape, target_shape, lora_shape, 
                                  "❌ 跳过", reason)
                
                transfer_stats['skipped_layers'].append(lora_key)
                transfer_stats['skipped_reasons'][lora_key] = reason
                continue

            # 获取权重和形状信息
            source_base = source_base_weights[base_key]
            target_base = target_base_weights[base_key]
            lora_weight = source_lora[lora_key]
            
            source_shape = tuple(source_base.shape)
            target_shape = tuple(target_base.shape)
            lora_shape = tuple(lora_weight.shape)

            # 获取相似度
            similarity = similarities.get(base_key, 0.0)
            transfer_stats['similarity_stats'].append({
                'layer': lora_key,
                'similarity': similarity,
                'layer_type': layer_type
            })

            # 检查维度兼容性
            if not self._check_dimension_compatibility(source_base, target_base, lora_weight):
                print_layer_details(lora_key, base_key, similarity, 
                                  source_shape, target_shape, lora_shape, 
                                  "🔧 Frobenius投影", "维度不兼容，使用投影方法")
                
                try:
                    projected_weight = self._frobenius_projection(lora_weight, source_base, target_base)
                    transferred_lora[lora_key] = projected_weight
                    transfer_stats['transferred_layers'] += 1
                    transfer_stats['transferred_list'].append(lora_key)
                    transfer_stats['layer_types'][layer_type]['transferred'] += 1
                    print(f"✅ 投影成功!")
                except Exception as e:
                    reason = f"Frobenius投影失败: {e}"
                    print_layer_details(lora_key, base_key, similarity, 
                                      source_shape, target_shape, lora_shape, 
                                      "❌ 跳过", reason)
                    transfer_stats['skipped_layers'].append(lora_key)
                    transfer_stats['skipped_reasons'][lora_key] = reason
                continue

            # 相似性过滤
            if similarity < self.similarity_threshold:
                reason = f"相似性过低 ({similarity:.4f} < {self.similarity_threshold})"
                print_layer_details(lora_key, base_key, similarity, 
                                  source_shape, target_shape, lora_shape, 
                                  "❌ 跳过", reason)
                transfer_stats['skipped_layers'].append(lora_key)
                transfer_stats['skipped_reasons'][lora_key] = reason
                continue

            # 执行迁移
            print_layer_details(lora_key, base_key, similarity, 
                              source_shape, target_shape, lora_shape, 
                              "🔄 正在迁移", "相似性通过，执行子空间投影")
            
            try:
                transferred_weight = self._transfer_single_layer(lora_weight, source_base, target_base)
                transferred_lora[lora_key] = transferred_weight
                transfer_stats['transferred_layers'] += 1
                transfer_stats['transferred_list'].append(lora_key)
                transfer_stats['layer_types'][layer_type]['transferred'] += 1
                print(f"✅ 迁移成功!")
            except Exception as e:
                reason = f"迁移过程出错: {e}"
                print_layer_details(lora_key, base_key, similarity, 
                                  source_shape, target_shape, lora_shape, 
                                  "❌ 跳过", reason)
                transfer_stats['skipped_layers'].append(lora_key)
                transfer_stats['skipped_reasons'][lora_key] = reason
        
        print_transfer_summary(transfer_stats)
        return transferred_lora
    
    def _precompute_similarities_verbose(self, source_lora: dict, 
                                       source_base_weights: dict,
                                       target_base_weights: dict) -> dict:
        """预计算相似度，提供详细输出"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        similarities = {}
        
        # 收集所有需要计算的层
        valid_keys = []
        for lora_key in source_lora.keys():
            if not lora_key.endswith('.weight'):
                continue
            base_key = self._map_lora_to_base_key(lora_key)
            if base_key in source_base_weights and base_key in target_base_weights:
                valid_keys.append(base_key)
        
        # 去重
        unique_keys = list(set(valid_keys))
        print(f"📊 需要计算相似度的唯一层数: {len(unique_keys)}")
        
        # 批量计算SVD和相似度
        for i, base_key in enumerate(unique_keys):
            print(f"🔄 计算进度: {i+1}/{len(unique_keys)} - {base_key}")
            
            try:
                source_base = source_base_weights[base_key].to(device)
                target_base = target_base_weights[base_key].to(device)
                
                print(f"   源权重形状: {tuple(source_base.shape)}")
                print(f"   目标权重形状: {tuple(target_base.shape)}")
                
                U_s, _, _ = self.compute_svd_subspace(source_base)
                U_t, _, _ = self.compute_svd_subspace(target_base)
                similarity = self.compute_subspace_similarity(U_s, U_t)
                similarities[base_key] = similarity
                
                print(f"   ✅ 相似度: {similarity:.4f}")
                
            except Exception as e:
                print(f"   ❌ 计算失败: {e}")
                similarities[base_key] = 0.0
        
        return similarities


def main():
    parser = argparse.ArgumentParser(description="自定义LoRA迁移脚本")
    parser.add_argument("--source_lora", type=str, 
                       default="/root/autodl-tmp/loraed/Meta-Llama-3.1-8B-Instruct/251728_010944",
                       help="源LoRA模型路径")
    parser.add_argument("--target_model", type=str, 
                       default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
                       help="目标基础模型路径")
    parser.add_argument("--output_base", type=str, 
                       default="/root/autodl-tmp/shifted/arc-challenge/Llama-3.1-8B_to_Qwen2.5-7B",
                       help="输出基础路径")
    parser.add_argument("--rank", type=int, default=320,
                       help="SVD截断秩")
    parser.add_argument("--similarity_threshold", type=float, default=0.3,
                       help="子空间相似性阈值")
    
    args = parser.parse_args()
    
    # 生成时间戳和输出路径
    timestamp = generate_timestamp()
    output_path = os.path.join(args.output_base, timestamp)
    
    # 推断源模型路径
    source_model_path = infer_source_model_path(args.source_lora)
    
    print(f"\n{'🚀'*20} LoRA迁移开始 {'🚀'*20}")
    print(f"{'='*80}")
    print(f"📂 源LoRA路径: {args.source_lora}")
    print(f"📂 源模型路径: {source_model_path}")
    print(f"📂 目标模型路径: {args.target_model}")
    print(f"📂 输出路径: {output_path}")
    print(f"⚙️  SVD截断秩: {args.rank}")
    print(f"⚙️  相似性阈值: {args.similarity_threshold}")
    print(f"🕐 时间戳: {timestamp}")
    print(f"{'='*80}")
    
    try:
        # 检查路径是否存在
        if not os.path.exists(args.source_lora):
            print(f"❌ 错误: 源LoRA路径不存在: {args.source_lora}")
            return False
        
        if not os.path.exists(source_model_path):
            print(f"❌ 错误: 源模型路径不存在: {source_model_path}")
            return False
            
        if not os.path.exists(args.target_model):
            print(f"❌ 错误: 目标模型路径不存在: {args.target_model}")
            return False
        
        # 初始化组件
        print(f"\n🔧 初始化LoRA-X核心组件...")
        lora_x = VerboseLoRAXCore(rank=args.rank, similarity_threshold=args.similarity_threshold)
        loader = ModelWeightLoader()
        
        # 加载源LoRA权重
        print(f"\n📥 加载源LoRA权重...")
        source_lora_weights, lora_config = loader.load_lora_weights(args.source_lora)
        print(f"✅ 成功加载 {len(source_lora_weights)} 个LoRA权重")
        
        # 加载基础模型权重
        print(f"\n📥 加载源模型权重...")
        source_base_weights = loader.load_base_model_weights(source_model_path)
        print(f"✅ 成功加载 {len(source_base_weights)} 个源模型权重")
        
        print(f"\n📥 加载目标模型权重...")
        target_base_weights = loader.load_base_model_weights(args.target_model)
        print(f"✅ 成功加载 {len(target_base_weights)} 个目标模型权重")
        
        # 执行迁移
        print(f"\n🔄 执行LoRA-X迁移...")
        transferred_lora = lora_x.transfer_lora_weights(
            source_lora=source_lora_weights,
            target_base_weights=target_base_weights,
            source_base_weights=source_base_weights
        )
        
        if not transferred_lora:
            print(f"❌ 迁移失败：没有成功迁移任何层")
            return False
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 保存结果
        print(f"\n💾 保存迁移结果到: {output_path}")
        save_transferred_lora(transferred_lora, lora_config, output_path)
        
        # 保存迁移日志
        log_file = os.path.join(output_path, "transfer_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"LoRA迁移日志\n")
            f.write(f"时间戳: {timestamp}\n")
            f.write(f"源LoRA: {args.source_lora}\n")
            f.write(f"源模型: {source_model_path}\n")
            f.write(f"目标模型: {args.target_model}\n")
            f.write(f"输出路径: {output_path}\n")
            f.write(f"SVD截断秩: {args.rank}\n")
            f.write(f"相似性阈值: {args.similarity_threshold}\n")
        
        print(f"\n🎉 LoRA迁移完成！")
        print(f"📂 结果保存在: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 迁移过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

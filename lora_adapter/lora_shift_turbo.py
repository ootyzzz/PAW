#!/usr/bin/env python3
"""
极致加速版LoRA迁移脚本 - Turbo Mode
优化策略:
1. 批量SVD计算，减少GPU内存传输
2. 简化相似度计算，使用快速近似方法
3. 移除详细日志，只保留关键信息
4. 并行处理多个层
5. 内存优化，及时释放不需要的张量

python /root/PAW/lora_adapter/lora_shift_turbo.py --source_lora /root/autodl-tmp/loraed/Qwen2.5-7B-Instruct/250719_004518/final_model --target_model /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct --batch_size 8

"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import gc

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from lora_x_core import LoRAXCore
from model_utils import ModelWeightLoader, save_transferred_lora

# 设置简化日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def generate_timestamp():
    """生成时间戳格式: YYMMDD_HHMMSS"""
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M%S")


def infer_source_model_path(lora_path: str) -> str:
    """根据LoRA路径推断源模型路径"""
    lora_path = Path(lora_path)
    
    if lora_path.name == "final_model":
        model_name = lora_path.parent.parent.name  
    else:
        model_name = lora_path.parent.name  
    
    source_model_path = f"/root/autodl-tmp/models/{model_name}"
    return source_model_path


class TurboLoRAXCore(LoRAXCore):
    """极致加速版LoRA-X核心类"""
    
    def __init__(self, rank=128, similarity_threshold=0.002, batch_size=16):
        super().__init__(rank, similarity_threshold)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu_device = torch.device('cpu')
        
    def stream_similarity_compute(self, valid_layers: list, source_base_weights: dict, target_base_weights: dict) -> dict:
        """批量并行计算相似度"""
        similarities = {}
        total_layers = len(valid_layers)
        start_time = datetime.now()
        
        print(f"⚡ 计算{total_layers}层相似度 (batch_size={self.batch_size})")
        
        # 按batch_size分批处理
        for batch_start in range(0, total_layers, self.batch_size):
            batch_start_time = datetime.now()
            batch_end = min(batch_start + self.batch_size, total_layers)
            batch_layers = valid_layers[batch_start:batch_end]
            
            # 预加载batch权重到GPU
            batch_data = []
            for lora_key, base_key in batch_layers:
                try:
                    source_weight = source_base_weights[base_key].to(self.device).float()
                    target_weight = target_base_weights[base_key].to(self.device).float()
                    batch_data.append({
                        'source_weight': source_weight,
                        'target_weight': target_weight,
                        'base_key': base_key
                    })
                except Exception as e:
                    similarities[base_key] = 0.0
            
            # 并行SVD计算
            if batch_data:
                try:
                    from concurrent.futures import ThreadPoolExecutor
                    
                    def compute_similarity_on_gpu(data_item):
                        try:
                            source_weight = data_item['source_weight']
                            target_weight = data_item['target_weight']
                            
                            U_s, _, _ = self.compute_svd_subspace(source_weight)
                            U_t, _, _ = self.compute_svd_subspace(target_weight)
                            similarity = self.compute_subspace_similarity(U_s, U_t)
                            
                            del U_s, U_t
                            return similarity
                        except Exception as e:
                            return 0.0
                    
                    max_workers = min(len(batch_data), 8)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(compute_similarity_on_gpu, data_item) for data_item in batch_data]
                        
                        for i, future in enumerate(futures):
                            try:
                                similarity = future.result(timeout=60)
                                similarities[batch_data[i]['base_key']] = similarity
                            except Exception as e:
                                similarities[batch_data[i]['base_key']] = 0.0
                
                except Exception as e:
                    # 回退到串行计算
                    for data_item in batch_data:
                        try:
                            source_weight = data_item['source_weight']
                            target_weight = data_item['target_weight']
                            base_key = data_item['base_key']
                            
                            U_s, _, _ = self.compute_svd_subspace(source_weight)
                            U_t, _, _ = self.compute_svd_subspace(target_weight)
                            similarity = self.compute_subspace_similarity(U_s, U_t)
                            similarities[base_key] = similarity
                            
                            del U_s, U_t
                        except:
                            similarities[base_key] = 0.0
            
            # 清理batch数据
            del batch_data
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # 计算性能指标
            batch_end_time = datetime.now()
            batch_duration = (batch_end_time - batch_start_time).total_seconds()
            batch_size_actual = len(batch_layers)
            items_per_sec = batch_size_actual / batch_duration if batch_duration > 0 else 0
            
            # 计算进度和预估时间
            completed = batch_end
            progress_pct = (completed / total_layers) * 100
            elapsed_total = (batch_end_time - start_time).total_seconds()
            
            if completed > 0:
                avg_time_per_item = elapsed_total / completed
                remaining_items = total_layers - completed
                eta_seconds = remaining_items * avg_time_per_item
                eta_minutes = eta_seconds / 60
            else:
                eta_minutes = 0
            
            print(f"   [{completed:3d}/{total_layers}] {progress_pct:5.1f}% | {items_per_sec:.1f} items/s | 已用时 {elapsed_total/60:.1f}m | 预计剩余 {eta_minutes:.1f}m")
        
        total_time = (datetime.now() - start_time).total_seconds()
        avg_speed = total_layers / total_time if total_time > 0 else 0
        print(f"✅ 相似度计算完成 | 总用时 {total_time/60:.1f}m | 平均速度 {avg_speed:.1f} items/s")
        
        return similarities

    def transfer_lora_weights(self,
                            source_lora: dict,
                            target_base_weights: dict,
                            source_base_weights: dict) -> dict:
        """极致加速的LoRA权重迁移"""
        transferred_lora = {}
        stats = {
            'total_layers': 0,
            'transferred_layers': 0,
            'skipped_layers': 0,
            'transferred_list': [],
            'skipped_list': [],
            'skipped_reasons': {},
            'similarity_stats': [],
            'layer_types': {},
            'processing_times': []
        }
        
        print(f"🚀 Turbo模式启动 - 处理{len(source_lora)}个LoRA权重")
        
        # 收集所有需要处理的层
        valid_layers = []
        for lora_key in source_lora.keys():
            if not lora_key.endswith('.weight'):
                continue
            
            base_key = self._map_lora_to_base_key(lora_key)
            if base_key in source_base_weights and base_key in target_base_weights:
                valid_layers.append((lora_key, base_key))
                stats['total_layers'] += 1
                
                # 统计层类型
                layer_type = self._classify_layer_type(lora_key)
                if layer_type not in stats['layer_types']:
                    stats['layer_types'][layer_type] = {'total': 0, 'transferred': 0}
                stats['layer_types'][layer_type]['total'] += 1
        
        print(f"📊 有效层数: {len(valid_layers)}")
        
        # 流式计算相似度
        similarities = self.stream_similarity_compute(valid_layers, source_base_weights, target_base_weights)
        
        # 流式处理迁移
        print("⚡ 流式迁移权重...")
        
        for i, (lora_key, base_key) in enumerate(valid_layers):
            if i % 50 == 0:
                print(f"   迁移进度: {i}/{len(valid_layers)}")
            
            similarity = similarities.get(base_key, 0.0)
            layer_type = self._classify_layer_type(lora_key)
            
            stats['similarity_stats'].append({
                'layer': lora_key,
                'similarity': similarity,
                'layer_type': layer_type
            })
            
            # 相似度过滤
            if similarity < self.similarity_threshold:
                reason = f"相似度过低 ({similarity:.6f} < {self.similarity_threshold})"
                stats['skipped_layers'] += 1
                stats['skipped_list'].append(lora_key)
                stats['skipped_reasons'][lora_key] = reason
                continue
            
            try:
                start_time = datetime.now()
                
                # 只在GPU上进行迁移计算
                source_base = source_base_weights[base_key].to(self.device)
                target_base = target_base_weights[base_key].to(self.device)
                lora_weight = source_lora[lora_key].to(self.device)
                
                # 检查维度兼容性
                if not self._check_dimension_compatibility(source_base, target_base, lora_weight):
                    transferred_weight = self._frobenius_projection(lora_weight, source_base, target_base)
                else:
                    transferred_weight = self._transfer_single_layer(lora_weight, source_base, target_base)
                
                transferred_lora[lora_key] = transferred_weight.cpu()
                stats['transferred_layers'] += 1
                stats['transferred_list'].append(lora_key)
                stats['layer_types'][layer_type]['transferred'] += 1
                
                # 记录处理时间
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                stats['processing_times'].append(processing_time)
                
                # 及时清理GPU内存
                del source_base, target_base, lora_weight, transferred_weight
                
            except Exception as e:
                reason = f"迁移失败: {str(e)}"
                stats['skipped_layers'] += 1
                stats['skipped_list'].append(lora_key)
                stats['skipped_reasons'][lora_key] = reason
            
            # 定期清理内存
            if i % 5 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        # 计算统计信息
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
        
        self._print_detailed_stats(stats)
        return transferred_lora, stats

    def _print_detailed_stats(self, stats: dict):
        """打印详细统计信息"""
        print(f"\n{'🎉'*20} Turbo迁移完成统计 {'🎉'*20}")
        print(f"{'='*80}")
        print(f"📊 总层数: {stats['total_layers']}")
        print(f"✅ 成功迁移: {stats['transferred_layers']}")
        print(f"❌ 跳过层数: {stats['skipped_layers']}")
        
        if stats['transferred_layers'] > 0:
            success_rate = (stats['transferred_layers'] / stats['total_layers']) * 100
            print(f"📈 迁移成功率: {success_rate:.1f}%")
        
        # 按层类型统计
        if stats['layer_types']:
            print(f"\n📋 按层类型统计:")
            for layer_type, type_stats in stats['layer_types'].items():
                total = type_stats['total']
                transferred = type_stats['transferred']
                rate = (transferred / total * 100) if total > 0 else 0
                print(f"  {layer_type:12s}: {transferred:2d}/{total:2d} ({rate:5.1f}%)")
        
        # 相似度统计
        if stats['similarity_stats']:
            similarities = [s['similarity'] for s in stats['similarity_stats']]
            print(f"\n📊 相似度统计:")
            print(f"  平均相似度: {sum(similarities)/len(similarities):.6f}")
            print(f"  最高相似度: {max(similarities):.6f}")
            print(f"  最低相似度: {min(similarities):.6f}")
        
        # 性能统计
        if 'avg_processing_time' in stats:
            print(f"\n⚡ 性能统计:")
            print(f"  平均处理时间: {stats['avg_processing_time']:.3f}秒/层")
            print(f"  总处理时间: {stats['total_processing_time']:.1f}秒")
        
        print(f"{'='*80}")

    def _fast_svd_subspace(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """快速SVD子空间计算 - 使用截断SVD"""
        # 使用更小的rank进行快速计算
        fast_rank = min(self.rank // 2, 64)
        try:
            U, _, _ = torch.svd_lowrank(weight_matrix.float(), q=fast_rank)
            return U[:, :fast_rank]
        except:
            # 回退到标准方法
            return super().compute_svd_subspace(weight_matrix)[0]


def main():
    parser = argparse.ArgumentParser(description="极致加速版LoRA迁移脚本")
    parser.add_argument("--source_lora", type=str, required=True,
                       help="源LoRA模型路径")
    parser.add_argument("--target_model", type=str, required=True,
                       help="目标基础模型路径")
    parser.add_argument("--output_base", type=str, 
                       default="/root/autodl-tmp/shifted/turbo",
                       help="输出基础路径")
    parser.add_argument("--rank", type=int, default=64,
                       help="SVD截断秩 (降低以提速)")
    parser.add_argument("--similarity_threshold", type=float, default=0.002,
                       help="相似性阈值")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批处理大小")
    
    args = parser.parse_args()
    
    # 生成时间戳和输出路径
    timestamp = generate_timestamp()
    output_path = os.path.join(args.output_base, timestamp)
    
    # 推断源模型路径
    source_model_path = infer_source_model_path(args.source_lora)
    
    print(f"🚀 TURBO LoRA迁移 - 极致加速模式")
    print(f"📂 源LoRA: {args.source_lora}")
    print(f"📂 源模型: {source_model_path}")
    print(f"📂 目标模型: {args.target_model}")
    print(f"📂 输出: {output_path}")
    print(f"⚙️ 参数: rank={args.rank}, threshold={args.similarity_threshold}, batch={args.batch_size}")
    
    try:
        # 检查路径
        for path in [args.source_lora, source_model_path, args.target_model]:
            if not os.path.exists(path):
                print(f"❌ 路径不存在: {path}")
                return False
        
        # 初始化组件
        lora_x = TurboLoRAXCore(
            rank=args.rank, 
            similarity_threshold=args.similarity_threshold,
            batch_size=args.batch_size
        )
        loader = ModelWeightLoader()
        
        # 加载权重
        print("📥 加载LoRA权重...")
        source_lora_weights, lora_config = loader.load_lora_weights(args.source_lora)
        
        print("📥 加载源模型权重...")
        source_base_weights = loader.load_base_model_weights(source_model_path)
        
        print("📥 加载目标模型权重...")
        target_base_weights = loader.load_base_model_weights(args.target_model)
        
        # 执行迁移
        start_time = datetime.now()
        transferred_lora, stats = lora_x.transfer_lora_weights(
            source_lora=source_lora_weights,
            target_base_weights=target_base_weights,
            source_base_weights=source_base_weights
        )
        end_time = datetime.now()
        
        if not transferred_lora:
            print("❌ 迁移失败：没有成功迁移任何层")
            return False
        
        # 保存结果
        os.makedirs(output_path, exist_ok=True)
        save_transferred_lora(transferred_lora, lora_config, output_path)
        
        # 保存详细统计日志
        duration = (end_time - start_time).total_seconds()
        stats_file = os.path.join(output_path, "transfer_stats.json")
        
        # 准备统计数据
        detailed_stats = {
            "timestamp": timestamp,
            "total_duration_seconds": duration,
            "source_lora_path": args.source_lora,
            "target_model_path": args.target_model,
            "output_path": output_path,
            "parameters": {
                "rank": args.rank,
                "similarity_threshold": args.similarity_threshold,
                "batch_size": args.batch_size
            },
            "results": {
                "total_layers": stats['total_layers'],
                "transferred_layers": stats['transferred_layers'],
                "skipped_layers": stats['skipped_layers'],
                "success_rate": (stats['transferred_layers'] / stats['total_layers'] * 100) if stats['total_layers'] > 0 else 0
            },
            "layer_types": stats['layer_types'],
            "similarity_statistics": {
                "similarities": [s['similarity'] for s in stats['similarity_stats']],
                "avg_similarity": sum(s['similarity'] for s in stats['similarity_stats']) / len(stats['similarity_stats']) if stats['similarity_stats'] else 0,
                "max_similarity": max(s['similarity'] for s in stats['similarity_stats']) if stats['similarity_stats'] else 0,
                "min_similarity": min(s['similarity'] for s in stats['similarity_stats']) if stats['similarity_stats'] else 0
            },
            "transferred_layers_list": stats['transferred_list'],
            "skipped_layers_details": {
                "layers": stats['skipped_list'],
                "reasons": stats['skipped_reasons']
            }
        }
        
        # 保存JSON统计文件
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, indent=2, ensure_ascii=False)
        
        print(f"🎉 Turbo迁移完成！用时: {duration:.1f}秒")
        print(f"📂 结果: {output_path}")
        print(f"📊 详细统计: {stats_file}")
        return True
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
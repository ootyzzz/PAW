"""
LoRA训练器包装器，用于添加batch追踪功能
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from core.train import LoRATrainer


@dataclass
class BatchTrackingInfo:
    """Batch追踪信息"""
    step: int
    epoch: int
    batch_size: int
    sample_lines: List[int]
    timestamp: str
    

class TrackingLoRATrainer:
    """带有batch追踪功能的LoRA训练器包装器"""
    
    def __init__(self, trainer: LoRATrainer, output_dir: str):
        self.trainer = trainer
        self.output_dir = Path(output_dir)
        self.tracking_log_file = self.output_dir / "batch_tracking.jsonl"
        self.checkpoint_mappings = {}
        self.batch_records = []
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def track_batch(self, step: int, batch_data: Dict[str, Any]) -> None:
        """记录单个batch的信息"""
        try:
            # 调试信息：检查batch数据结构
            if step == 0:  # 只在第一步打印调试信息
                print(f"🔍 调试batch结构: type={type(batch_data)}, keys={list(batch_data.keys()) if hasattr(batch_data, 'keys') else 'N/A'}")
            
            # 提取sample信息
            samples = []
            if isinstance(batch_data, list):
                for i, sample in enumerate(batch_data):
                    sample_info = {
                        "batch_index": i,
                        "source_line": sample.get('_source_line', -1),
                        "epoch": sample.get('_epoch', -1),
                        "has_input": 'input' in sample,
                        "has_output": 'output' in sample
                    }
                    samples.append(sample_info)
            elif isinstance(batch_data, dict):
                # 处理字典格式的batch（可能来自custom_collate_fn）
                batch_size = len(batch_data.get('input', [])) if 'input' in batch_data else len(next(iter(batch_data.values())))
                for i in range(batch_size):
                    sample_info = {
                        "batch_index": i,
                        "source_line": batch_data.get('_source_line', [-1] * batch_size)[i] if '_source_line' in batch_data else -1,
                        "epoch": batch_data.get('_epoch', [-1] * batch_size)[i] if '_epoch' in batch_data else -1,
                        "has_input": 'input' in batch_data,
                        "has_output": 'output' in batch_data
                    }
                    samples.append(sample_info)
            else:
                # 处理tensor batch的情况
                batch_size = len(batch_data) if hasattr(batch_data, '__len__') else 1
                for i in range(batch_size):
                    sample_info = {
                        "batch_index": i,
                        "source_line": -1,
                        "epoch": -1,
                        "has_input": True,
                        "has_output": True
                    }
                    samples.append(sample_info)
            
            # 创建batch记录
            batch_record = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "batch_size": len(samples),
                "samples": samples
            }
            
            # 保存记录
            self.batch_records.append(batch_record)
            
            # 写入JSONL文件
            with open(self.tracking_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(batch_record, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"⚠️  追踪第{step}步时出错: {e}")
    
    def track_checkpoint(self, checkpoint_step: int, checkpoint_name: str, 
                        batch_range: Tuple[int, int]) -> None:
        """记录checkpoint信息"""
        checkpoint_info = {
            "checkpoint_step": checkpoint_step,
            "checkpoint_name": checkpoint_name,
            "batch_range": {
                "start_step": batch_range[0],
                "end_step": batch_range[1]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.checkpoint_mappings[checkpoint_name] = checkpoint_info
        # 不打印单个checkpoint，在最后统一报告
    
    def train_with_tracking(self, dataloader, config: Dict[str, Any], 
                           checkpoint_steps: List[int]) -> Dict[str, Any]:
        """执行带追踪的训练"""
        
        print(f"\n🚀 开始带追踪的LoRA训练...")
        print(f"📊 配置:")
        print(f"  - 最大步数: {config['training']['max_steps']}")
        print(f"  - Checkpoint步骤: {len(checkpoint_steps)}个 (步骤{min(checkpoint_steps)}-{max(checkpoint_steps)})")
        
        # 预先生成checkpoint映射（静默）
        for checkpoint_step in checkpoint_steps:
            checkpoint_name = f"checkpoint-{checkpoint_step}"
            # 每个checkpoint对应前一步的数据
            self.track_checkpoint(checkpoint_step, checkpoint_name, 
                                (checkpoint_step - 1, checkpoint_step - 1))
        
        # 启动并行batch追踪
        import threading
        import os
        
        # 设置tokenizers环境变量避免fork警告
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        tracking_active = {"active": True}
        
        def parallel_batch_tracking():
            """并行执行batch追踪"""
            step = 0
            batch_iter = iter(dataloader)
            max_steps = config['training']['max_steps']
            
            while step < max_steps and tracking_active["active"]:
                try:
                    batch = next(batch_iter)
                    self.track_batch(step, batch)
                    
                    step += 1
                    time.sleep(0.01)  # 小延迟避免过度占用资源
                    
                except StopIteration:
                    # 重新开始dataloader
                    batch_iter = iter(dataloader)
                    if step < max_steps:
                        batch = next(batch_iter)
                        self.track_batch(step, batch)
                        step += 1
                except Exception as e:
                    print(f"⚠️  Batch追踪异常: {e}")
                    break
        
        # 启动追踪线程
        tracking_thread = threading.Thread(target=parallel_batch_tracking, daemon=True)
        tracking_thread.start()
        
        try:
            # 执行实际训练
            print("🏃‍♂️ 开始实际LoRA训练...")
            
            training_result = self.trainer.train(
                batch_size=config['training']['per_device_train_batch_size'],
                max_length=config.get('data', {}).get('max_length', 512),
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
                warmup_steps=0,
                logging_steps=config['training'].get('logging_steps', 1),
                save_steps=config['training'].get('save_steps', 50)
            )
            
            print("✅ LoRA训练完成!")
            
        except Exception as e:
            print(f"❌ LoRA训练失败: {e}")
            raise
        finally:
            # 停止追踪
            tracking_active["active"] = False
            time.sleep(1)  # 等待追踪线程结束
        
        # 生成映射文件
        self.generate_mapping_file()
        
        return {
            "status": "training_completed",
            "output_dir": str(self.output_dir),
            "total_steps": config['training']['max_steps'],
            "checkpoint_steps": checkpoint_steps,
            "tracking_file": str(self.tracking_log_file),
            "training_result": training_result
        }
    
    def generate_mapping_file(self) -> None:
        """生成完整的line-to-checkpoint映射文件"""
        
        print(f"\n📋 生成line-to-checkpoint映射...")
        
        # 构建映射关系
        line_to_checkpoint = {}
        checkpoint_to_lines = {}
        
        for ckpt_name, ckpt_info in self.checkpoint_mappings.items():
            checkpoint_step = ckpt_info['checkpoint_step']
            start_step = ckpt_info['batch_range']['start_step']
            end_step = ckpt_info['batch_range']['end_step']
            
            # 找到相关的batch记录
            related_batches = [
                record for record in self.batch_records
                if start_step <= record["step"] <= end_step
            ]
            
            lines_in_checkpoint = set()
            
            for batch_record in related_batches:
                for sample in batch_record["samples"]:
                    source_line = sample.get("source_line", -1)
                    if source_line >= 0:
                        line_to_checkpoint[source_line] = ckpt_name
                        lines_in_checkpoint.add(source_line)
            
            checkpoint_to_lines[ckpt_name] = {
                "checkpoint_step": checkpoint_step,
                "lines": sorted(list(lines_in_checkpoint)),
                "line_count": len(lines_in_checkpoint),
                "batch_range": ckpt_info['batch_range']
            }
        
        # 生成完整映射
        mapping_data = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_checkpoints": len(checkpoint_to_lines),
                "total_tracked_lines": len(line_to_checkpoint),
                "total_batch_records": len(self.batch_records)
            },
            "line_to_checkpoint": line_to_checkpoint,
            "checkpoint_to_lines": checkpoint_to_lines
        }
        
        # 保存映射文件
        mapping_file = self.output_dir / "line_to_checkpoint_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 映射文件保存: {mapping_file}")
        print(f"📊 追踪统计:")
        print(f"  - Checkpoints: {mapping_data['generation_info']['total_checkpoints']}")
        print(f"  - 追踪行数: {mapping_data['generation_info']['total_tracked_lines']}")
        print(f"  - Batch记录: {mapping_data['generation_info']['total_batch_records']}")

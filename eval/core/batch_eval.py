"""
批量评估模块
包含批量评估逻辑和结果处理功能
"""

from .config import *
from .data import get_test_file_path, SimpleDataset
from .evaluator import LightningModelEvaluator


def evaluate_models(
    models_list: List[str],
    dataset_name: str,
    output_dir: str = "eval/results",
    base_model_path: str = None,
    sample_ratio: float = 1.0,
    batch_size: int = 1
):
    """评估多个模型并保存结果"""
    print("\n" + "=" * 70)
    print(f"🚀 Lightning 批量模型评估")
    print("=" * 70)
    
    # 准备输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据
    test_file = get_test_file_path(dataset_name)
    test_dataset = SimpleDataset(test_file, sample_ratio=sample_ratio)
    
    print(f"📝 数据集: {dataset_name}")
    print(f"📝 测试文件: {test_file}")
    print(f"📊 样本数量: {len(test_dataset)}")
    print(f"📊 批处理大小: {batch_size}")
    print(f"📊 采样比例: {sample_ratio*100:.1f}%")
    
    results = {}
    start_time = time.time()
    
    # 准备共享数据加载器 - 使用固定的随机种子以确保可比性
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Lightning测试推荐打乱顺序
        num_workers=2,  # 减少worker数量，降低fork带来的警告
        pin_memory=True,
        collate_fn=lambda batch: batch,  # 保持批次格式不变
        generator=torch.Generator().manual_seed(42),  # 固定随机种子
        persistent_workers=True  # 保持worker持续运行，避免频繁fork
    )
    
    # 评估每个模型
    for i, model_path in enumerate(models_list):
        print(f"\n{'='*70}")
        print(f"📊 [{i+1}/{len(models_list)}] 评估模型: {model_path}")
        
        model_name = Path(model_path).name
        if not model_name:  # 处理路径末尾的斜杠
            model_name = Path(model_path).parent.name
            
        try:
            print(f"🔍 开始初始化评估器...")
            print(f"🔍 模型路径: {model_path}")
            print(f"🔍 基础模型路径: {base_model_path}")
            print(f"🔍 模型名称: {model_name}")
            
            # 初始化Lightning评估模块
            evaluator = LightningModelEvaluator(model_path, base_model_path)
            print(f"✅ 评估器初始化成功")
            
            # 模型加载完成，无需调整batch size
            
            # 创建Trainer (无需checkpoint) - 针对Gemma模型优化
            trainer_kwargs = {
                "accelerator": 'auto',
                "devices": 1,  # 强制使用单GPU，避免多进程问题
                "precision": '16-mixed' if torch.cuda.is_available() else 32,
                "logger": False,
                "enable_checkpointing": False,  # 评估不需要检查点
                "enable_model_summary": False,  # 关闭模型摘要
                "enable_progress_bar": True,  # 启用进度条显示累积准确率
                "deterministic": False,  # 对Gemma模型禁用deterministic
                "num_sanity_val_steps": 0,  # 避免sanity检查
                "inference_mode": True,  # 使用推理模式
                "benchmark": False,  # 关闭基准测试
                "strategy": "auto",  # 使用自动策略，但限制为单设备
            }
            
            # 如果是Gemma模型，使用更保守的设置
            if "gemma" in model_path.lower():
                trainer_kwargs.update({
                    "precision": 32,  # 使用32位精度避免数值问题
                    "deterministic": False,  # 完全禁用deterministic
                })
            
            trainer = Trainer(**trainer_kwargs)
            
            # 执行测试
            eval_start = time.time()
            test_results = trainer.test(evaluator, dataloaders=test_loader)
            eval_time = time.time() - eval_start
            
            # 整理结果 - 确保所有值都是Python标量而不是Tensor
            model_results = {}
            if test_results and len(test_results) > 0:
                raw_results = test_results[0]
                # 转换所有的tensor值为Python标量
                for key, value in raw_results.items():
                    if hasattr(value, 'item'):
                        model_results[key] = value.item()
                    else:
                        model_results[key] = value
            
            # 添加时间指标
            model_results['eval_time_seconds'] = eval_time
            model_results['samples_per_second'] = len(test_dataset) / eval_time
            
            # 添加到结果集
            results[model_name] = {
                dataset_name: model_results
            }
            
            # 不再保存单个模型的JSON结果文件，减少文件输出
            # result_file = output_path / f"{model_name}_{dataset_name}_evaluation_results.json"
            # with open(result_file, 'w', encoding='utf-8') as f:
            #     json.dump(model_results, f, indent=4, ensure_ascii=False)
                
            print(f"✅ 评估完成 (用时: {eval_time:.1f}秒, {model_results['samples_per_second']:.1f} 样本/秒)")
            
            # 清理内存
            del evaluator
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            print(f"❌ 异常类型: {type(e).__name__}")
            print(f"❌ 模型路径: {model_path}")
            print(f"❌ 模型名称: {model_name}")
            print(f"❌ 数据集: {dataset_name}")
            print(f"❌ 基础模型路径: {base_model_path}")
            print(f"❌ 当前工作目录: {os.getcwd()}")
            
            # 检查模型路径是否存在
            if os.path.exists(model_path):
                print(f"✅ 模型路径存在")
                try:
                    files = os.listdir(model_path)
                    print(f"🔍 模型目录文件: {files[:5]}...")
                except Exception as list_error:
                    print(f"⚠️ 无法列出模型目录: {list_error}")
            else:
                print(f"❌ 模型路径不存在")
            
            # 内存状态
            try:
                if torch.cuda.is_available():
                    gpu_allocated = torch.cuda.memory_allocated() / 1024**3
                    gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"🔍 GPU内存: {gpu_allocated:.2f}GB / {gpu_reserved:.2f}GB")
            except Exception as mem_error:
                print(f"⚠️ 内存检查失败: {mem_error}")
            
            print(f"❌ 详细错误信息:")
            traceback.print_exc()
            
            results[model_name] = {
                dataset_name: {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "model_path": model_path,
                    "model_exists": os.path.exists(model_path)
                }
            }
    
    # 计算总用时
    total_time = time.time() - start_time
    
    # 简化汇总结果输出 - 不再生成带时间戳的JSON文件
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # summary_file = output_path / f"lightning_evaluation_summary_{timestamp}.json"
    #
    # summary_data = {
    #     "evaluation_summary": {
    #         "dataset": dataset_name,
    #         "total_models": len(models_list),
    #         "sample_ratio": sample_ratio,
    #         "batch_size": batch_size,
    #         "total_samples": len(test_dataset),
    #         "total_time_seconds": total_time,
    #         "timestamp": datetime.now().isoformat()
    #     },
    #     "results": results
    # }
    #
    # with open(summary_file, 'w', encoding='utf-8') as f:
    #     json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    # 保存CSV格式结果
    rows = []
    for model_name, model_results in results.items():
        for dataset_name, dataset_results in model_results.items():
            if 'error' not in dataset_results:
                # 确保所有值都是Python标量
                loss_val = dataset_results.get('test/loss', 0)
                acc_val = dataset_results.get('test/accuracy', 0)
                ppl_val = dataset_results.get('test/perplexity', 0)
                time_val = dataset_results.get('eval_time_seconds', 0)
                samples_val = dataset_results.get('samples_per_second', 0)
                
                # 强制转换为Python原生类型
                try:
                    if hasattr(loss_val, 'item'):
                        loss_val = float(loss_val.item())
                    else:
                        loss_val = float(loss_val)
                except:
                    loss_val = 0.0
                    
                try:
                    if hasattr(acc_val, 'item'):
                        acc_val = float(acc_val.item())
                    else:
                        acc_val = float(acc_val)
                except:
                    acc_val = 0.0
                    
                try:
                    if hasattr(ppl_val, 'item'):
                        ppl_val = float(ppl_val.item())
                    else:
                        ppl_val = float(ppl_val)
                except:
                    ppl_val = 0.0
                    
                try:
                    if hasattr(time_val, 'item'):
                        time_val = float(time_val.item())
                    else:
                        time_val = float(time_val)
                except:
                    time_val = 0.0
                    
                try:
                    if hasattr(samples_val, 'item'):
                        samples_val = float(samples_val.item())
                    else:
                        samples_val = float(samples_val)
                except:
                    samples_val = 0.0
                
                row_data = {
                    'Model': str(model_name),
                    'Dataset': str(dataset_name),
                    'Loss': round(loss_val, 4),
                    'Accuracy': round(acc_val, 4),
                    'Perplexity': round(ppl_val, 4),
                    'Eval_Time(s)': round(time_val, 1),
                    'Samples/Sec': round(samples_val, 1),
                    'Batch_Size': batch_size,
                    'Timestamp': str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                }
                rows.append(row_data)
    
    # 保存CSV格式结果 - 使用原生CSV写入避免pandas问题
    if rows:
        try:
            import csv
            
            # 生成带时间戳的CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = output_path / f"lightning_evaluation_results_{timestamp}.csv"
            
            # 写入CSV文件
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                if rows:
                    fieldnames = rows[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            
            print(f"📁 CSV结果已保存到: {csv_file}")
            
            # 如果是完整数据集评估，也追加到总的实验结果文件
            if sample_ratio == 1.0:
                experiment_csv = Path("results/experiment_results.csv")
                
                # 确保目录存在
                experiment_csv.parent.mkdir(parents=True, exist_ok=True)
                
                # 简单追加到文件
                file_exists = experiment_csv.exists()
                with open(experiment_csv, 'a', newline='', encoding='utf-8') as f:
                    if rows:
                        fieldnames = rows[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerows(rows)
                
                print(f"📁 总结果已追加到: {experiment_csv}")
        except Exception as csv_error:
            print(f"⚠️ 保存CSV结果失败: {csv_error}")
            import traceback
            traceback.print_exc()
    
    print(f"⏱️  总评估时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    
    return results

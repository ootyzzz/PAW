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
    batch_size: int = 8
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
            # 初始化Lightning评估模块
            evaluator = LightningModelEvaluator(model_path, base_model_path)
            
            # 创建Trainer (无需checkpoint) - 针对Gemma模型优化
            trainer_kwargs = {
                "accelerator": 'auto',
                "devices": 'auto',
                "precision": '16-mixed' if torch.cuda.is_available() else 32,
                "logger": False,
                "enable_checkpointing": False,  # 评估不需要检查点
                "enable_model_summary": False,  # 关闭模型摘要
                "enable_progress_bar": True,
                "deterministic": False,  # 对Gemma模型禁用deterministic
                "num_sanity_val_steps": 0,  # 避免sanity检查
                "inference_mode": True,  # 使用推理模式
                "benchmark": False,  # 关闭基准测试
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
            
            # 保存单个模型结果
            result_file = output_path / f"{model_name}_{dataset_name}_evaluation_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, indent=4, ensure_ascii=False)
                
            print(f"✅ 评估完成 (用时: {eval_time:.1f}秒, {model_results['samples_per_second']:.1f} 样本/秒)")
            print(f"📊 结果:")
            print(f"  - Loss: {model_results.get('test/loss', 0):.4f}")
            print(f"  - Accuracy: {model_results.get('test/accuracy', 0):.4f}") 
            print(f"  - Perplexity: {model_results.get('test/perplexity', 0):.4f}")
            
            # 清理内存
            del evaluator
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            print(f"❌ 模型路径: {model_path}")
            print(f"❌ 模型名称: {model_name}")
            print(f"❌ 数据集: {dataset_name}")
            print(f"❌ 详细错误信息:")
            traceback.print_exc()
            results[model_name] = {
                dataset_name: {"error": str(e)}
            }
    
    # 计算总用时
    total_time = time.time() - start_time
    
    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_path / f"lightning_evaluation_summary_{timestamp}.json"
    
    summary_data = {
        "evaluation_summary": {
            "dataset": dataset_name,
            "total_models": len(models_list),
            "sample_ratio": sample_ratio,
            "batch_size": batch_size,
            "total_samples": len(test_dataset),
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
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
                
                # 转换tensor/numpy值为Python标量
                if hasattr(loss_val, 'item'):
                    loss_val = float(loss_val.item())
                if hasattr(acc_val, 'item'):
                    acc_val = float(acc_val.item())
                if hasattr(ppl_val, 'item'):
                    ppl_val = float(ppl_val.item())
                if hasattr(time_val, 'item'):
                    time_val = float(time_val.item())
                if hasattr(samples_val, 'item'):
                    samples_val = float(samples_val.item())
                
                rows.append({
                    'Model': str(model_name),
                    'Dataset': str(dataset_name),
                    'Loss': round(float(loss_val), 4),
                    'Accuracy': round(float(acc_val), 4),
                    'Perplexity': round(float(ppl_val), 4),
                    'Eval_Time(s)': round(float(time_val), 1),
                    'Samples/Sec': round(float(samples_val), 1),
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
    
    csv_file = None  # 初始化变量
    if rows:
        try:
            # 手动写CSV文件，避免pandas问题
            csv_file = output_path / f"lightning_evaluation_results_{timestamp}.csv"
            
            # 获取列名
            headers = ['Model', 'Dataset', 'Loss', 'Accuracy', 'Perplexity', 'Eval_Time(s)', 'Samples/Sec', 'Timestamp']
            
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                f.write(','.join(headers) + '\n')
                for row in rows:
                    values = []
                    for header in headers:
                        val = row.get(header, '')
                        # 确保值是字符串，并处理可能的逗号
                        if isinstance(val, str) and ',' in val:
                            val = f'"{val}"'
                        values.append(str(val))
                    f.write(','.join(values) + '\n')
            
            print(f" 结果已保存到: {csv_file}")
            
            # 尝试pandas方式作为备选
            try:
                # 只有在手动方式成功后才尝试pandas
                import pandas as pd
                df = pd.DataFrame(rows)
                cumulative_csv = output_path / "all_evaluation_results.csv"
                if cumulative_csv.exists():
                    existing_df = pd.read_csv(cumulative_csv, encoding='utf-8-sig')
                    # 移除重复项
                    for _, row_data in df.iterrows():
                        mask = (existing_df['Model'] == row_data['Model']) & (existing_df['Dataset'] == row_data['Dataset'])
                        existing_df = existing_df[~mask]
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv(cumulative_csv, index=False, encoding='utf-8-sig')
                else:
                    df.to_csv(cumulative_csv, index=False, encoding='utf-8-sig')
                print(f"📁 累积结果: {cumulative_csv}")
            except Exception as pandas_error:
                print(f"⚠️ pandas累积CSV更新失败: {pandas_error}")
                # 不影响主要CSV文件的保存
        except Exception as e:
            print(f"⚠️ 保存CSV结果失败: {e}")
            traceback.print_exc()
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print("📊 评估结果汇总")
    print("=" * 80)
    print(f"{'Model':<40} {'Dataset':<15} {'Loss':<8} {'Accuracy':<10} {'Perplexity':<12} {'Time(s)':<8} {'Samples/s':<10}")
    print("-" * 110)
    
    for model_name, model_results in results.items():
        for dataset_name, dataset_results in model_results.items():
            if 'error' not in dataset_results:
                print(f"{model_name:<40} {dataset_name:<15} "
                      f"{dataset_results.get('test/loss', 0):<8.4f} "
                      f"{dataset_results.get('test/accuracy', 0):<10.4f} "
                      f"{dataset_results.get('test/perplexity', 0):<12.4f} "
                      f"{dataset_results.get('eval_time_seconds', 0):<8.1f} "
                      f"{dataset_results.get('samples_per_second', 0):<10.1f}")
            else:
                print(f"{model_name:<40} {dataset_name:<15} {'ERROR':<8} {'ERROR':<10} {'ERROR':<12} {'ERROR':<8} {'ERROR':<10}")
    
    print("\n" + "=" * 80)
    print(f"⏱️  总评估时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"📁 汇总结果: {summary_file}")
    if csv_file:
        print(f"📊 CSV结果: {csv_file}")
    
    return results

#!/usr/bin/env python3
"""
batch_evaluate.py
批量评估不同prompt生成的LoRA权重性能

运行多个validation prompts并比较结果
"""

import subprocess
import json
from pathlib import Path

def run_evaluation(prompt_idx, max_samples=100):
    """运行单个prompt的评估"""
    cmd = [
        "python", "evaluate_generated_lora.py",
        "--prompt_idx", str(prompt_idx),
        "--max_test_samples", str(max_samples)
    ]
    
    print(f"🔄 运行prompt {prompt_idx}的评估...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Prompt {prompt_idx} 评估完成")
        return True
    else:
        print(f"❌ Prompt {prompt_idx} 评估失败: {result.stderr}")
        return False

def collect_results():
    """收集所有结果"""
    results_dir = Path("evaluation_results")
    results = []
    
    # 基线结果
    baseline_file = results_dir / "arc_challenge_evaluation_baseline.json"
    if baseline_file.exists():
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
            results.append({
                'type': 'baseline',
                'prompt_idx': -1,
                'accuracy': baseline_data['metrics']['accuracy'],
                'correct': baseline_data['metrics']['correct'],
                'total': baseline_data['metrics']['total']
            })
    
    # LoRA结果
    for i in range(10):  # 假设最多10个prompts
        lora_file = results_dir / f"arc_challenge_evaluation_prompt_{i}.json"
        if lora_file.exists():
            with open(lora_file, 'r', encoding='utf-8') as f:
                lora_data = json.load(f)
                results.append({
                    'type': 'lora',
                    'prompt_idx': i,
                    'accuracy': lora_data['metrics']['accuracy'],
                    'correct': lora_data['metrics']['correct'],
                    'total': lora_data['metrics']['total']
                })
    
    return results

def plot_results(results):
    """绘制结果对比图"""
    baseline_acc = None
    lora_results = []
    
    for r in results:
        if r['type'] == 'baseline':
            baseline_acc = r['accuracy']
        else:
            lora_results.append((r['prompt_idx'], r['accuracy']))
    
    if not lora_results:
        print("⚠️ 没有找到LoRA评估结果")
        return
    
    # 排序
    lora_results.sort()
    prompt_indices, accuracies = zip(*lora_results)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    # LoRA结果
    bars = plt.bar(range(len(prompt_indices)), accuracies, alpha=0.7, label='LoRA Enhanced')
    
    # 基线
    if baseline_acc is not None:
        plt.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.1%})')
    
    plt.xlabel('Validation Prompt Index')
    plt.ylabel('Accuracy')
    plt.title('LoRA Generator Performance Across Different Validation Prompts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.1%}', ha='center', va='bottom')
    
    plt.xticks(range(len(prompt_indices)), [f'P{i}' for i in prompt_indices])
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('evaluation_results/lora_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 结果可视化已保存到: evaluation_results/lora_performance_comparison.png")

def main():
    """主函数"""
    print("🚀 批量LoRA评估实验")
    print("=" * 50)
    
    # 确保baseline存在
    baseline_file = Path("evaluation_results/arc_challenge_evaluation_baseline.json")
    if not baseline_file.exists():
        print("⚠️ 未找到baseline结果，先运行baseline评估...")
        subprocess.run(["python", "evaluate_generated_lora.py", "--baseline_only", "--max_test_samples", "100"])
    
    # 批量运行不同prompt的评估
    max_prompts = 5  # 测试前5个prompts
    successful_runs = 0
    
    for i in range(max_prompts):
        if run_evaluation(i, max_samples=100):
            successful_runs += 1
    
    print(f"\n📈 完成 {successful_runs}/{max_prompts} 个prompt的评估")
    
    # 收集并分析结果
    results = collect_results()
    
    if results:
        print("\n📊 结果汇总:")
        print("-" * 60)
        for r in results:
            if r['type'] == 'baseline':
                print(f"📋 Baseline:        {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            else:
                print(f"🔥 LoRA Prompt {r['prompt_idx']:2d}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
        
        # 找出最佳LoRA结果
        lora_results = [r for r in results if r['type'] == 'lora']
        if lora_results:
            best_lora = max(lora_results, key=lambda x: x['accuracy'])
            baseline = next((r for r in results if r['type'] == 'baseline'), None)
            
            if baseline:
                improvement = best_lora['accuracy'] - baseline['accuracy']
                relative_improvement = improvement / baseline['accuracy'] * 100
                
                print(f"\n🎯 最佳结果:")
                print(f"   最佳LoRA (Prompt {best_lora['prompt_idx']}): {best_lora['accuracy']:.1%}")
                print(f"   相比Baseline提升: +{improvement:.1%} (相对+{relative_improvement:.1f}%)")
        
        # 绘制结果
        plot_results(results)
    else:
        print("⚠️ 未找到评估结果文件")

if __name__ == "__main__":
    main()

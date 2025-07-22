#!/usr/bin/env python3
"""
inference.py
LoRA参数生成器推理脚本

使用方法：
python inference.py --model experiments/lora_generator_xxx/results/final_model/generator_model.pt --prompt "Question: What is..."
python inference.py --model experiments/lora_generator_xxx/results/final_model/generator_model.pt --prompt_file test_prompts.txt
"""
import os
import sys
from pathlib import Path
import argparse
import torch
import yaml
from typing import List, Dict, Any

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.generator import LoRAParameterGenerator, LoRATokenizer


class LoRAGeneratorInference:
    """LoRA参数生成器推理类"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model, self.config = self._load_model(model_path)
        self.tokenizer = LoRATokenizer(
            max_tokens=self.config['model']['max_seq_len'],
            token_dim=self.config['model']['output_dim']
        )
        
    def _setup_device(self, device: str) -> torch.device:
        """设置设备"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path: str):
        """加载模型"""
        print(f"📦 加载模型: {model_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # 创建模型
        model = LoRAParameterGenerator(
            text_encoder_name=config['model']['text_encoder_name'],
            hidden_dim=config['model']['hidden_dim'],
            max_seq_len=config['model']['max_seq_len'],
            num_hyperconv_blocks=config['model']['num_hyperconv_blocks'],
            output_dim=config['model']['output_dim'],
            freeze_text_encoder=True
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ 模型加载完成 - 设备: {self.device}")
        
        return model, config
    
    def generate_lora_parameters(
        self, 
        prompts: List[str],
        return_raw: bool = False
    ) -> Dict[str, Any]:
        """
        生成LoRA参数
        
        Args:
            prompts: 输入prompt列表
            return_raw: 是否返回原始token格式
            
        Returns:
            结果字典
        """
        with torch.no_grad():
            # 生成参数tokens
            generated_tokens = self.model(prompts)  # [B, max_seq_len, output_dim]
            
            results = {
                'prompts': prompts,
                'generated_tokens': generated_tokens.cpu() if return_raw else None,
                'batch_size': len(prompts),
                'token_shape': generated_tokens.shape,
                'statistics': self._compute_statistics(generated_tokens)
            }
            
            # 如果需要，将tokens转换为参数向量
            if not return_raw:
                param_vectors = []
                for i in range(len(prompts)):
                    param_vec = self.tokenizer.detokenize(generated_tokens[i])
                    param_vectors.append(param_vec.cpu())
                
                results['parameter_vectors'] = param_vectors
                results['parameter_shapes'] = [vec.shape for vec in param_vectors]
            
            return results
    
    def _compute_statistics(self, tokens: torch.Tensor) -> Dict[str, float]:
        """计算生成参数的统计信息"""
        with torch.no_grad():
            stats = {
                'mean': tokens.mean().item(),
                'std': tokens.std().item(),
                'min': tokens.min().item(),
                'max': tokens.max().item(),
                'l1_norm': tokens.abs().mean().item(),
                'l2_norm': torch.norm(tokens).item() / tokens.numel()**0.5
            }
        return stats
    
    def save_generated_parameters(
        self, 
        results: Dict[str, Any], 
        output_path: str,
        save_format: str = 'pt'
    ):
        """保存生成的参数"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format == 'pt':
            torch.save(results, output_path.with_suffix('.pt'))
        elif save_format == 'yaml':
            # 移除tensor数据，只保存元数据
            save_data = {
                'prompts': results['prompts'],
                'batch_size': results['batch_size'],
                'token_shape': list(results['token_shape']),
                'statistics': results['statistics'],
                'config': self.config
            }
            with open(output_path.with_suffix('.yaml'), 'w') as f:
                yaml.dump(save_data, f, default_flow_style=False)
        
        print(f"💾 结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LoRA参数生成器推理脚本")
    parser.add_argument("--model", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--prompt", type=str, default=None,
                       help="单个输入prompt")
    parser.add_argument("--prompt_file", type=str, default=None,
                       help="包含多个prompt的文件")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备 (cuda/cpu/auto)")
    parser.add_argument("--save_format", type=str, default="pt",
                       choices=["pt", "yaml"], help="保存格式")
    parser.add_argument("--return_raw", action="store_true",
                       help="返回原始token格式而不是参数向量")
    
    args = parser.parse_args()
    
    # 验证输入
    if not args.prompt and not args.prompt_file:
        print("❌ 必须提供 --prompt 或 --prompt_file")
        return False
    
    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        return False
    
    try:
        # 初始化推理器
        print("🚀 LoRA参数生成器推理")
        print("=" * 50)
        
        inference = LoRAGeneratorInference(args.model, args.device)
        
        # 准备prompts
        prompts = []
        if args.prompt:
            prompts = [args.prompt]
        elif args.prompt_file:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        
        print(f"📝 输入prompts: {len(prompts)} 个")
        for i, prompt in enumerate(prompts[:3]):  # 只显示前3个
            print(f"  {i+1}. {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        if len(prompts) > 3:
            print(f"  ... (还有 {len(prompts)-3} 个)")
        
        # 生成参数
        print("\\n🧠 生成LoRA参数...")
        results = inference.generate_lora_parameters(prompts, return_raw=args.return_raw)
        
        # 显示结果
        print("\\n📊 生成结果:")
        print(f"  - Token形状: {results['token_shape']}")
        print(f"  - 统计信息:")
        for key, value in results['statistics'].items():
            print(f"    {key}: {value:.6f}")
        
        # 保存结果
        if args.output:
            inference.save_generated_parameters(
                results, args.output, args.save_format
            )
        else:
            # 默认输出路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"Lora_Gen/results/generated_params_{timestamp}.pt"
            inference.save_generated_parameters(results, output_path, args.save_format)
        
        print("\\n✅ 推理完成!")
        return True
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    from datetime import datetime
    success = main()
    sys.exit(0 if success else 1)

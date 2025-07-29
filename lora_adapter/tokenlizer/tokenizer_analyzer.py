#!/usr/bin/env python3
"""
Tokenizer差异分析工具
用于分析Qwen和Llama模型之间的tokenizer差异，为LoRA迁移提供基础数据
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set
import torch
from transformers import AutoTokenizer
import numpy as np
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizerAnalyzer:
    """Tokenizer差异分析器"""
    
    def __init__(self, qwen_path: str, llama_path: str):
        """
        初始化分析器
        
        Args:
            qwen_path: Qwen模型tokenizer路径
            llama_path: Llama模型tokenizer路径
        """
        self.qwen_path = qwen_path
        self.llama_path = llama_path
        
        logger.info(f"加载Qwen tokenizer: {qwen_path}")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        
        logger.info(f"加载Llama tokenizer: {llama_path}")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
        
        logger.info("Tokenizer加载完成")
    
    def analyze_basic_info(self) -> Dict:
        """分析基础信息"""
        qwen_vocab = self.qwen_tokenizer.get_vocab()
        llama_vocab = self.llama_tokenizer.get_vocab()
        
        info = {
            'qwen_vocab_size': len(qwen_vocab),
            'llama_vocab_size': len(llama_vocab),
            'qwen_model_max_length': getattr(self.qwen_tokenizer, 'model_max_length', 'N/A'),
            'llama_model_max_length': getattr(self.llama_tokenizer, 'model_max_length', 'N/A'),
            'qwen_tokenizer_class': self.qwen_tokenizer.__class__.__name__,
            'llama_tokenizer_class': self.llama_tokenizer.__class__.__name__,
        }
        
        # 特殊token分析
        def extract_special_tokens(tokenizer):
            special_tokens = set()
            if hasattr(tokenizer, 'special_tokens_map'):
                for value in tokenizer.special_tokens_map.values():
                    if isinstance(value, list):
                        special_tokens.update(value)
                    else:
                        special_tokens.add(value)
            return special_tokens
        
        qwen_special = extract_special_tokens(self.qwen_tokenizer)
        llama_special = extract_special_tokens(self.llama_tokenizer)
        
        info.update({
            'qwen_special_tokens': list(qwen_special),
            'llama_special_tokens': list(llama_special),
            'common_special_tokens': list(qwen_special & llama_special),
        })
        
        return info
    
    def analyze_vocab_overlap(self) -> Dict:
        """分析词汇表重叠情况"""
        qwen_vocab = set(self.qwen_tokenizer.get_vocab().keys())
        llama_vocab = set(self.llama_tokenizer.get_vocab().keys())
        
        common_tokens = qwen_vocab & llama_vocab
        qwen_only = qwen_vocab - llama_vocab
        llama_only = llama_vocab - qwen_vocab
        
        overlap_stats = {
            'total_unique_tokens': len(qwen_vocab | llama_vocab),
            'common_tokens_count': len(common_tokens),
            'qwen_only_count': len(qwen_only),
            'llama_only_count': len(llama_only),
            'overlap_ratio': len(common_tokens) / len(qwen_vocab | llama_vocab),
            'qwen_coverage': len(common_tokens) / len(qwen_vocab),
            'llama_coverage': len(common_tokens) / len(llama_vocab),
        }
        
        # 分析常见token的类型
        common_token_types = self._classify_tokens(list(common_tokens)[:100])  # 分析前100个
        qwen_only_types = self._classify_tokens(list(qwen_only)[:100])
        llama_only_types = self._classify_tokens(list(llama_only)[:100])
        
        overlap_stats.update({
            'common_token_types': common_token_types,
            'qwen_only_types': qwen_only_types,
            'llama_only_types': llama_only_types,
        })
        
        return overlap_stats
    
    def analyze_encoding_differences(self, test_texts: List[str]) -> List[Dict]:
        """分析编码差异"""
        results = []
        
        for text in test_texts:
            try:
                qwen_tokens = self.qwen_tokenizer.encode(text, add_special_tokens=False)
                llama_tokens = self.llama_tokenizer.encode(text, add_special_tokens=False)
                
                qwen_decoded = [self.qwen_tokenizer.decode([t]) for t in qwen_tokens]
                llama_decoded = [self.llama_tokenizer.decode([t]) for t in llama_tokens]
                
                result = {
                    'text': text,
                    'qwen_tokens': qwen_tokens,
                    'llama_tokens': llama_tokens,
                    'qwen_decoded': qwen_decoded,
                    'llama_decoded': llama_decoded,
                    'qwen_length': len(qwen_tokens),
                    'llama_length': len(llama_tokens),
                    'length_ratio': len(llama_tokens) / len(qwen_tokens) if qwen_tokens else float('inf'),
                    'compression_efficiency': {
                        'qwen': len(text) / len(qwen_tokens) if qwen_tokens else 0,
                        'llama': len(text) / len(llama_tokens) if llama_tokens else 0,
                    }
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"编码文本失败: {text[:50]}... 错误: {e}")
                results.append({
                    'text': text,
                    'error': str(e),
                    'qwen_tokens': [],
                    'llama_tokens': [],
                    'qwen_length': 0,
                    'llama_length': 0,
                    'length_ratio': 0,
                })
        
        return results
    
    def _classify_tokens(self, tokens: List[str]) -> Dict[str, int]:
        """分类token类型"""
        types = {
            'ascii_letters': 0,
            'digits': 0,
            'punctuation': 0,
            'whitespace': 0,
            'chinese': 0,
            'special_chars': 0,
            'subword_pieces': 0,
            'other': 0
        }
        
        for token in tokens:
            if not token:
                continue
                
            if token.startswith('▁') or token.startswith('##') or token.startswith('Ġ'):
                types['subword_pieces'] += 1
            elif token.isalpha() and all(ord(c) < 128 for c in token):
                types['ascii_letters'] += 1
            elif token.isdigit():
                types['digits'] += 1
            elif any('\u4e00' <= c <= '\u9fff' for c in token):
                types['chinese'] += 1
            elif token.isspace():
                types['whitespace'] += 1
            elif all(not c.isalnum() for c in token):
                types['punctuation'] += 1
            elif any(ord(c) > 127 for c in token):
                types['special_chars'] += 1
            else:
                types['other'] += 1
        
        return types
    
    def generate_test_texts(self) -> List[str]:
        """生成测试文本"""
        return [
            # 英文测试
            "Hello, how are you?",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "This is a longer sentence with multiple words and punctuation marks!",
            
            # 中文测试
            "你好，你好吗？",
            "机器学习是人工智能的重要分支。",
            "这是一个包含多个词汇和标点符号的较长句子！",
            "深度学习模型在自然语言处理任务中表现出色。",
            
            # 混合语言测试
            "Hello 你好 world 世界",
            "Machine Learning 机器学习 is amazing!",
            
            # 特殊字符测试
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "Numbers: 1234567890",
            "Mixed: abc123中文!@#",
            
            # 代码测试
            "def hello_world(): print('Hello, World!')",
            "import torch\nfrom transformers import AutoTokenizer",
            
            # 长文本测试
            "This is a very long sentence that contains many words and should help us understand how different tokenizers handle longer sequences of text with various types of content including punctuation and numbers like 123 and special characters.",
        ]
    
    def run_full_analysis(self, output_dir: str = None) -> Dict:
        """运行完整分析"""
        logger.info("开始完整的tokenizer差异分析...")
        
        # 基础信息分析
        logger.info("分析基础信息...")
        basic_info = self.analyze_basic_info()
        
        # 词汇表重叠分析
        logger.info("分析词汇表重叠...")
        vocab_overlap = self.analyze_vocab_overlap()
        
        # 编码差异分析
        logger.info("分析编码差异...")
        test_texts = self.generate_test_texts()
        encoding_diffs = self.analyze_encoding_differences(test_texts)
        
        # 汇总结果
        results = {
            'basic_info': basic_info,
            'vocab_overlap': vocab_overlap,
            'encoding_differences': encoding_diffs,
            'summary': self._generate_summary(basic_info, vocab_overlap, encoding_diffs)
        }
        
        # 保存结果
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def _generate_summary(self, basic_info: Dict, vocab_overlap: Dict, encoding_diffs: List[Dict]) -> Dict:
        """生成分析摘要"""
        # 计算平均长度比例
        length_ratios = [ed['length_ratio'] for ed in encoding_diffs if 'length_ratio' in ed and ed['length_ratio'] != float('inf')]
        avg_length_ratio = np.mean(length_ratios) if length_ratios else 0
        
        # 计算压缩效率差异
        qwen_compressions = [ed['compression_efficiency']['qwen'] for ed in encoding_diffs if 'compression_efficiency' in ed]
        llama_compressions = [ed['compression_efficiency']['llama'] for ed in encoding_diffs if 'compression_efficiency' in ed]
        
        summary = {
            'vocab_size_ratio': basic_info['llama_vocab_size'] / basic_info['qwen_vocab_size'],
            'vocab_overlap_ratio': vocab_overlap['overlap_ratio'],
            'avg_length_ratio': avg_length_ratio,
            'avg_compression_efficiency': {
                'qwen': np.mean(qwen_compressions) if qwen_compressions else 0,
                'llama': np.mean(llama_compressions) if llama_compressions else 0,
            },
            'compatibility_score': self._calculate_compatibility_score(vocab_overlap, length_ratios),
            'recommendations': self._generate_recommendations(basic_info, vocab_overlap, avg_length_ratio)
        }
        
        return summary
    
    def _calculate_compatibility_score(self, vocab_overlap: Dict, length_ratios: List[float]) -> float:
        """计算兼容性分数 (0-1)"""
        # 词汇重叠权重: 40%
        vocab_score = vocab_overlap['overlap_ratio'] * 0.4
        
        # 长度一致性权重: 30%
        if length_ratios:
            length_variance = np.var(length_ratios)
            length_score = max(0, 1 - length_variance) * 0.3
        else:
            length_score = 0
        
        # 覆盖率权重: 30%
        coverage_score = (vocab_overlap['qwen_coverage'] + vocab_overlap['llama_coverage']) / 2 * 0.3
        
        return vocab_score + length_score + coverage_score
    
    def _generate_recommendations(self, basic_info: Dict, vocab_overlap: Dict, avg_length_ratio: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if vocab_overlap['overlap_ratio'] < 0.5:
            recommendations.append("词汇表重叠度较低，建议实施词汇表对齐策略")
        
        if abs(avg_length_ratio - 1.0) > 0.3:
            recommendations.append("编码长度差异较大，可能影响位置编码的迁移效果")
        
        if basic_info['qwen_vocab_size'] > basic_info['llama_vocab_size'] * 1.2:
            recommendations.append("源模型词汇表显著大于目标模型，建议使用降维映射")
        
        if vocab_overlap['qwen_coverage'] < 0.7:
            recommendations.append("Qwen词汇覆盖率较低，嵌入层迁移可能效果有限")
        
        recommendations.append("建议优先迁移非嵌入层（attention和MLP层）")
        recommendations.append("考虑实施tokenizer-aware的嵌入层对齐策略")
        
        return recommendations
    
    def _save_results(self, results: Dict, output_dir: str):
        """保存分析结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        with open(output_path / 'tokenizer_analysis_full.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存摘要报告
        summary_report = self._format_summary_report(results)
        with open(output_path / 'tokenizer_analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        logger.info(f"分析结果已保存到: {output_path}")
    
    def _format_summary_report(self, results: Dict) -> str:
        """格式化摘要报告"""
        basic = results['basic_info']
        overlap = results['vocab_overlap']
        summary = results['summary']
        
        report = f"""
Tokenizer差异分析报告
{'='*50}

基础信息:
- Qwen词汇表大小: {basic['qwen_vocab_size']:,}
- Llama词汇表大小: {basic['llama_vocab_size']:,}
- 词汇表大小比例: {summary['vocab_size_ratio']:.3f}

词汇表重叠分析:
- 总体重叠率: {overlap['overlap_ratio']:.3f}
- Qwen覆盖率: {overlap['qwen_coverage']:.3f}
- Llama覆盖率: {overlap['llama_coverage']:.3f}
- 共同词汇数: {overlap['common_tokens_count']:,}

编码效率:
- 平均长度比例: {summary['avg_length_ratio']:.3f}
- Qwen平均压缩效率: {summary['avg_compression_efficiency']['qwen']:.2f}
- Llama平均压缩效率: {summary['avg_compression_efficiency']['llama']:.2f}

兼容性评估:
- 兼容性分数: {summary['compatibility_score']:.3f} / 1.0

改进建议:
"""
        for i, rec in enumerate(summary['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        return report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tokenizer差异分析工具")
    parser.add_argument("--qwen_path", type=str, 
                       default="/root/autodl-tmp/loraed/Qwen2.5-7B-Instruct/250719_004518/final_model",
                       help="Qwen模型tokenizer路径")
    parser.add_argument("--llama_path", type=str,
                       default="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct", 
                       help="Llama模型tokenizer路径")
    parser.add_argument("--output_dir", type=str,
                       default="/root/PAW/lora_adapter/tokenlizer/analysis_results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.qwen_path):
        logger.error(f"Qwen路径不存在: {args.qwen_path}")
        return
    
    if not os.path.exists(args.llama_path):
        logger.error(f"Llama路径不存在: {args.llama_path}")
        return
    
    # 运行分析
    analyzer = TokenizerAnalyzer(args.qwen_path, args.llama_path)
    results = analyzer.run_full_analysis(args.output_dir)
    
    # 打印摘要
    print("\n" + "="*60)
    print("TOKENIZER差异分析完成")
    print("="*60)
    print(f"兼容性分数: {results['summary']['compatibility_score']:.3f} / 1.0")
    print(f"词汇重叠率: {results['vocab_overlap']['overlap_ratio']:.3f}")
    print(f"平均长度比例: {results['summary']['avg_length_ratio']:.3f}")
    print("\n关键建议:")
    for i, rec in enumerate(results['summary']['recommendations'][:3], 1):
        print(f"{i}. {rec}")
    print(f"\n详细结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
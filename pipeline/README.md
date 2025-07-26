# LoRA Transfer Pipeline

Automated pipeline for training LoRA adapters and transferring weights between language models with performance evaluation.

## Quick Start

```bash
# Run complete transfer experiment
python pipeline/transfer_pipeline.py \
  --source_model gemma-2-2b-it \
  --target_model Qwen_Qwen2.5-1.5B \
  --dataset arc-challenge

# Evaluation only mode
python pipeline/transfer_pipeline.py --eval_only

# Quick test with minimal steps
python pipeline/transfer_pipeline.py --quick_test
```

## Configuration

All paths and parameters are configured in `pipeline/config/pipeline_config.yaml`. The pipeline uses relative paths for portability across different machines.

### Key Configuration Sections

**Paths**: Model directories, script locations, output paths
**Training**: Batch size, max steps, learning rate, datasets
**Evaluation**: Sample ratio, batch size
**Transfer**: Similarity threshold for weight mapping

## Supported Models

The pipeline works with models in `../autodl-tmp/models/`:
- gemma-2-2b-it
- Qwen_Qwen2.5-1.5B
- Qwen-Qwen2.5-0.5B
- Llama-3.2-3B-Instruct

## Supported Datasets

- arc-challenge
- arc-easy
- piqa
- hellaswag
- winogrande

## Pipeline Steps

1. Evaluate source model baseline
2. Train LoRA adapter on source model
3. Evaluate target model baseline
4. Transfer LoRA weights to target model
5. Evaluate transferred LoRA
6. Train native LoRA on target model (optional)

## Directory Structure

```
pipeline/
├── config/pipeline_config.yaml    # Main configuration
├── core/                          # Core pipeline components
├── transfer_pipeline.py           # Main pipeline script
└── README.md

runs/{dataset}/{model}/{timestamp}/ # Training outputs
└── final_model/                   # Trained LoRA adapters

../autodl-tmp/transferred_lora/    # Transfer results
└── {dataset}/{source}_to_{target}/

results/                           # Experiment results
├── experiment_results.csv         # Structured data
└── experiment_summary.md          # Summary report
```

## Command Line Options

```bash
--source_model MODEL     # Source model name
--target_model MODEL     # Target model name  
--dataset DATASET        # Dataset to use
--eval_only             # Skip training, evaluate only
--quick_test            # Use quick test configuration
--config CONFIG_FILE    # Custom configuration file
```

## Results

Results are saved in CSV format for data analysis and Markdown format for human review. The pipeline automatically tracks:
- Base model accuracies
- LoRA training results
- Transfer performance
- Comparison metrics

## Environment Requirements

- PyTorch Lightning
- SwanLab for experiment tracking
- Transformers library
- Standard ML evaluation tools

Working directory should be the PAW root directory when running the pipeline.


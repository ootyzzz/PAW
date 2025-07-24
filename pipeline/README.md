# LoRA Transfer Pipeline

Automated LoRA training and transfer pipeline for cross-model LoRA weight migration and performance evaluation.

## Quick Start

### Basic Usage
```bash
# Run complete A->B transfer experiment
python transfer_pipeline.py \
  --source_model Qwen-Qwen2.5-0.5B \
  --target_model Qwen_Qwen2.5-1.5B \
  --dataset arc-challenge

# Quick test mode
python transfer_pipeline.py --quick_test

# Evaluation only (skip training)
python transfer_pipeline.py \
  --source_model Qwen-Qwen2.5-0.5B \
  --target_model Qwen_Qwen2.5-1.5B \
  --dataset arc-challenge \
  --eval_only
```

## Configuration

Configuration files are located in `config/`:
- `pipeline_config.yaml` - Standard configuration
- `quick_test_config.yaml` - Fast testing with reduced parameters

### Supported Models
```
Available models in /root/autodl-tmp/models/:
- Qwen-Qwen2.5-0.5B
- Qwen_Qwen2.5-1.5B
- Llama-3.2-3B-Instruct
- gemma-2-2b-it
```

### Supported Datasets
- arc-challenge
- arc-easy
- piqa
- boolq

## Pipeline Steps

1. **Train Source LoRA**: Train LoRA adapter on source model
2. **Transfer LoRA**: Migrate LoRA weights to target model
3. **Evaluate Target Base**: Evaluate target model baseline performance
4. **Evaluate Transferred LoRA**: Evaluate migrated LoRA performance
5. **Train Target LoRA**: Train native LoRA on target model (for comparison)
6. **Evaluate Source Base**: Evaluate source model baseline performance

## Directory Structure

```
runs/                              # Training results
├── {dataset}/
│   └── {model}/
│       └── {timestamp}/
│           └── final_model/       # Final LoRA model

transferred_lora/                  # Transfer results
├── {dataset}/
│   └── {source}_to_{target}/
│       └── {timestamp}/

results/                           # Experiment records
├── experiment_results.csv         # Structured data
└── experiment_summary.md          # Human-readable report
```

## Results Management

Results are automatically saved in two formats:
- **CSV**: Machine-readable structured data
- **Markdown**: Human-readable summary with tables

View results:
```bash
cat results/experiment_summary.md
```

## Configuration Files

### Standard Configuration (`pipeline_config.yaml`)
- Default batch size: 8
- Default max steps: 1000
- Full evaluation (100% samples)

### Quick Test Configuration (`quick_test_config.yaml`)
- Reduced batch size: 4
- Reduced max steps: 20
- Sample evaluation (5% samples)

## Error Handling

The pipeline includes:
- Automatic detection of existing training results
- Resume capability for interrupted experiments
- Comprehensive error logging and recovery
- Data type cleaning for pandas compatibility


# Usage Guide - Qwen-7B Fine-tuning Project

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export WANDB_API_KEY=<your_wandb_key>
export HF_TOKEN=<your_huggingface_token>
```

### 2. Run Complete Pipeline
```bash
# Run all steps
python scripts/run_full_pipeline.py

# Run with inference server
python scripts/run_full_pipeline.py --include-server

# Skip certain steps
python scripts/run_full_pipeline.py --skip-training --skip-evaluation
```

## Individual Script Usage

### Data Analysis
```bash
python scripts/data_analysis.py
```
- Analyzes the reasoning traces dataset
- Outputs: `results/data_analysis/`
- Generates cleaning recommendations

### Baseline Benchmarking (Part 1)
```bash
python scripts/benchmark_qwen.py
```
- Evaluates base Qwen-7B-Instruct model
- Outputs: `results/benchmark_qwen/`
- Creates prompting format documentation

### Fine-tuning (Part 2)
```bash
python scripts/fine_tune_qwen.py
```
- Fine-tunes model on reasoning traces
- Outputs: Model saved to `models/qwen-7b-finetuned/`
- Logs training metrics to W&B

### Evaluation (Part 3)
```bash
python scripts/evaluate_finetuned.py
```
- Compares fine-tuned vs baseline performance
- Outputs: `results/comparison/`
- Requires baseline results to exist

### Inference Server (Part 4 - Bonus)
```bash
python scripts/inference_server.py
```
- Starts FastAPI server on port 8000
- Supports both vLLM and transformers backends
- Provides REST API for code generation

## API Usage

### Start Server
```bash
python scripts/inference_server.py
```

### Generate Code
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "problem": "Write a function that finds the maximum element in an array",
    "max_tokens": 512,
    "temperature": 0.1
})

solution = response.json()["solution"]
print(solution)
```

### Batch Generation
```python
problems = [
    {"problem": "Problem 1 description"},
    {"problem": "Problem 2 description"}
]

response = requests.post("http://localhost:8000/generate/batch", json=problems)
solutions = [r["solution"] for r in response.json()]
```

## Configuration

### Training Configuration
Edit `configs/training_config.yaml` to modify:
- Learning rate and batch sizes
- LoRA parameters
- Dataset settings
- Hardware configurations

### Model Configuration
Key parameters in scripts:
```python
# In fine_tune_qwen.py
config = FineTuningConfig(
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    lora_r=16,
    num_train_epochs=3
)
```

## Hardware Requirements

### Minimum Requirements
- **RAM**: 32GB system RAM
- **GPU**: 24GB VRAM (RTX 4090, A6000, etc.)
- **Storage**: 50GB free space

### Recommended Setup
- **GPU**: A100 40GB or better
- **RAM**: 64GB+ system RAM
- **Storage**: SSD with 100GB+ free space

### Memory Optimization
If running into memory issues:
1. Reduce batch size: `per_device_train_batch_size=2`
2. Increase gradient accumulation: `gradient_accumulation_steps=8`
3. Use gradient checkpointing: `gradient_checkpointing=true`
4. Switch to CPU offloading for model layers

## Output Structure

```
├── results/
│   ├── data_analysis/          # Dataset analysis results
│   ├── benchmark_qwen/         # Baseline evaluation
│   └── comparison/             # Final comparison results
├── models/
│   └── qwen-7b-finetuned/     # Fine-tuned model
├── final_report.json          # Pipeline summary
└── PIPELINE_SUMMARY.md        # Human-readable summary
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
per_device_train_batch_size=2
gradient_accumulation_steps=8

# Or use CPU offloading
device_map="auto"
max_memory_per_gpu="20GB"
```

#### 2. Dataset Loading Fails
```python
# Check dataset name in script
dataset_name = "microsoft/orca-math-word-problems-200k"  # Fallback dataset

# Or provide local dataset path
dataset_path = "/path/to/your/350k_reasoning_traces"
```

#### 3. Model Loading Issues
```bash
# Ensure HuggingFace token is set
export HF_TOKEN=<your_token>

# Or use local model path
model_name = "/path/to/local/qwen/model"
```

#### 4. Slow Training
- Enable Flash Attention: `pip install flash-attn`
- Use compiled model: `torch.compile(model)`
- Increase `dataloader_num_workers`

### Performance Tips

1. **Use vLLM for Inference**: Much faster than transformers
2. **Enable W&B Logging**: Monitor training progress
3. **Use SSD Storage**: Faster data loading
4. **Pin Memory**: Speed up data transfers
5. **Gradient Checkpointing**: Trade speed for memory

## Customization

### Adding New Datasets
1. Modify `ReasoningTracesDataProcessor.load_dataset()`
2. Update data formatting in `format_training_examples()`
3. Adjust preprocessing parameters

### Changing Evaluation Metrics
1. Edit `CodeContestsEvaluator.evaluate_solutions()`
2. Add new metrics to results dictionary
3. Update comparison logic in evaluation script

### Custom Prompting
1. Modify `_format_prompt()` in benchmark script
2. Update training data formatting
3. Adjust generation parameters

## Monitoring and Logging

### W&B Integration
- Automatic logging of training metrics
- Hyperparameter tracking
- Model artifacts saving
- Run comparison dashboard

### Local Logging
- Training logs: `models/qwen-7b-finetuned/training_logs.json`
- Configuration: `models/qwen-7b-finetuned/training_config.json`
- Results: Various JSON files in results directories

## Best Practices

1. **Start Small**: Test with limited samples first
2. **Monitor Closely**: Watch for overfitting signs
3. **Save Checkpoints**: Enable regular model saving
4. **Document Changes**: Update configuration files
5. **Version Control**: Track code and config changes
6. **Reproducible Runs**: Use fixed seeds

## Getting Help

### Debugging Steps
1. Check log files for error messages
2. Verify GPU memory availability
3. Confirm dataset accessibility
4. Test with smaller configurations
5. Review requirements.txt versions

### Common Error Patterns
- Memory errors → Reduce batch size
- Loading errors → Check file paths/tokens
- Generation errors → Verify prompting format
- Server errors → Check model loading

For additional support, check the documentation in each results directory and the detailed error logs.
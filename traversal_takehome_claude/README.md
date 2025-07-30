# AI Researcher Interview - Qwen-7B Fine-tuning for Coding Tasks

This project evaluates and fine-tunes the Qwen-7B-Instruct model on coding tasks using Codeforces-style benchmarks and open-source reasoning traces.

## Project Structure

```
├── data/                       # Dataset storage
├── models/                     # Model checkpoints
├── scripts/                    # Training and evaluation scripts
├── results/                    # Evaluation results and logs
├── configs/                    # Configuration files
└── notebooks/                  # Analysis notebooks
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export WANDB_API_KEY=<your_wandb_key>
export HF_TOKEN=<your_huggingface_token>
```

## Project Parts

### Part 1: Benchmark Performance
- Evaluate Qwen-7B-Instruct on Codeforces dataset
- Document prompting format and data processing decisions

### Part 2: Fine-tuning
- Fine-tune on 350K reasoning traces dataset
- Document training hyperparameters and data decisions
- Track training and evaluation metrics

### Part 3: Re-evaluation
- Evaluate fine-tuned model performance
- Compare with baseline results

### Part 4: Inference Endpoint (Bonus)
- Deploy model using HuggingFace or vLLM
- Provide API endpoint for querying

## Usage

See individual scripts in the `scripts/` directory for detailed usage instructions.
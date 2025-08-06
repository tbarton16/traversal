#!/bin/bash

# Memory-optimized launch script for lora_gpt.py
# This script includes comprehensive memory optimizations to prevent OOM errors

set -e  # Exit on any error

echo "Starting memory-optimized LoRA GPT training..."

# Memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false

# NCCL optimizations for multi-GPU training
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Python memory optimizations
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1

# DeepSpeed optimizations
export DEEPSPEED_ACCELERATOR=cuda

# Clear GPU memory before starting
echo "Clearing GPU memory..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --gpu-reset || echo "Warning: Could not reset GPUs"
fi

python3 -c "
import torch
import gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print(f'Cleared memory on {torch.cuda.device_count()} GPU(s)')
"

# Set output directory
OUTPUT_DIR="${1:-/mnt/storage/qwen/lora_gpt_memory_optimized}"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Memory optimizations: ENABLED"
echo "  CPU offloading: ENABLED"
echo "  Gradient checkpointing: ENABLED"
echo "  4-bit quantization: ENABLED"
echo "  Batch size optimizations: ENABLED"
echo "  Single GPU mode: ENABLED"
echo "  Sequence length: 2048 tokens (experimental)"

# Launch training with memory optimizations
echo "Launching DeepSpeed training (single GPU, 2048 tokens)..."
deepspeed --num_gpus=1 \
    --master_port=29500 \
    lora_gpt_2048_flash.py \
    --max_seq_len 2048 \
    --wandb_run_name "test-run-1"
    --deepspeed ds_gpt.json

echo "Training completed successfully!"
echo "Model saved to: $OUTPUT_DIR"

# Final cleanup
echo "Performing final cleanup..."
python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
print('Final cleanup completed')
"

echo "All done!"
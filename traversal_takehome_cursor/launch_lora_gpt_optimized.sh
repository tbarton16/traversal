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
export RUN_NAME="test-run-90percent2048"
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
deepspeed --master_port=2950$CUDA_VISIBLE_DEVICES \
    lora_gpt_2048_flash.py \
    --output_dir /mnt/storage/qwen/$RUN_NAME \
    --max_seq_len 2048 \
    --wandb_run_name $RUN_NAME \
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
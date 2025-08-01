#!/bin/bash

# Launch script for distributed training on 8-node H100 machine
# This script sets up the environment and launches training across all nodes

set -e

# Configuration
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export WORLD_SIZE=${WORLD_SIZE:-8}  # 8 nodes
export NNODES=${NNODES:-8}
export NODE_RANK=${NODE_RANK:-0}

# Model and training configuration
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="./finetuned_model_distributed"
DATASET_PATH=""  # Set to your dataset path if available
NUM_EPOCHS=3
BATCH_SIZE=8  # Per GPU batch size
LEARNING_RATE=1e-5
MAX_LENGTH=2048

# Create output directory
mkdir -p $OUTPUT_DIR

# Set up environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all 8 GPUs per node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1

# Function to launch training
launch_training() {
    echo "Launching distributed training on node $NODE_RANK"
    echo "World size: $WORLD_SIZE"
    echo "Master address: $MASTER_ADDR"
    echo "Master port: $MASTER_PORT"
    
    # Launch with torchrun
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=8 \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        part2_distributed_finetune.py \
        --model_name $MODEL_NAME \
        --dataset_path $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --eval_codeforces \
        --codeforces_data ""
}

# Function to setup single node training (for testing)
setup_single_node() {
    echo "Setting up single node training for testing..."
    export WORLD_SIZE=1
    export NNODES=1
    export NODE_RANK=0
    export MASTER_ADDR="localhost"
    export MASTER_PORT="29500"
    
    launch_training
}

# Function to setup multi-node training
setup_multi_node() {
    echo "Setting up multi-node training..."
    
    # Check if we're on the master node
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "This is the master node (rank 0)"
        echo "Make sure all other nodes are ready before proceeding"
        echo "Press Enter to continue..."
        read
    fi
    
    launch_training
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [single|multi]"
    echo ""
    echo "Options:"
    echo "  single  - Run single node training (for testing)"
    echo "  multi   - Run multi-node training (requires proper setup)"
    echo ""
    echo "Environment variables:"
    echo "  NODE_RANK     - Node rank (0-7 for 8 nodes)"
    echo "  MASTER_ADDR   - Master node address"
    echo "  MASTER_PORT   - Master node port"
    echo "  WORLD_SIZE    - Total number of nodes"
    echo ""
    echo "Example for node 0:"
    echo "  NODE_RANK=0 MASTER_ADDR=192.168.1.100 $0 multi"
    echo ""
    echo "Example for node 1:"
    echo "  NODE_RANK=1 MASTER_ADDR=192.168.1.100 $0 multi"
}

# Main script logic
case "${1:-}" in
    "single")
        setup_single_node
        ;;
    "multi")
        setup_multi_node
        ;;
    *)
        show_usage
        exit 1
        ;;
esac 
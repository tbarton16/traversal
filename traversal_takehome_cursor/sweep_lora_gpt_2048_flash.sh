#!/bin/bash

# Parameter sweep script for lora_gpt_2048_flash.py
# Sweeps over max_seq_len and code_extraction_probability parameters

set -e  # Exit on any error

# Configuration
SCRIPT_PATH="lora_gpt_2048_flash.py"
BASE_OUTPUT_DIR="/mnt/storage/qwen/sweep_experiments"
LOG_DIR="${BASE_OUTPUT_DIR}/logs"
RESULTS_DIR="${BASE_OUTPUT_DIR}/results"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Parameter ranges
MAX_SEQ_LEN_VALUES=(512 1024 1536 2048)
CODE_EXTRACTION_PROB_VALUES=(0.0 0.25 0.5 0.75 1.0)

# Training configuration (conservative for sweeping)
MAX_STEPS=500
BATCH_SIZE=1
GRAD_ACCUMULATION=8
LEARNING_RATE=1e-4

# WandB configuration
WANDB_PROJECT="lora-gpt-sweep"
WANDB_ENTITY=""  # Set this if you have a specific entity

# Function to clean up GPU memory
cleanup_gpu() {
    echo "🧹 Cleaning up GPU memory..."
    python3 -c "
import torch
import gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
gc.collect()
print('GPU memory cleared')
"
    sleep 5
}

# Function to check available GPU memory
check_gpu_memory() {
    python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        total = props.total_memory / 1024**3
        free = total - allocated
        print(f'GPU {i}: {free:.2f}GB free / {total:.2f}GB total')
        if free < 10.0:
            print(f'WARNING: GPU {i} has less than 10GB free memory')
            return False
    return True
else:
    print('No CUDA devices available')
    return False
" || return 1
}

# Function to run a single experiment
run_experiment() {
    local max_seq_len=$1
    local code_extraction_prob=$2
    local run_name="seqlen_${max_seq_len}_codeprob_${code_extraction_prob}"
    local output_dir="${BASE_OUTPUT_DIR}/${run_name}"
    local log_file="${LOG_DIR}/${run_name}.log"
    
    echo "🚀 Starting experiment: ${run_name}"
    echo "   max_seq_len: ${max_seq_len}"
    echo "   code_extraction_probability: ${code_extraction_prob}"
    echo "   Output: ${output_dir}"
    echo "   Log: ${log_file}"
    
    # Clean up before each run
    cleanup_gpu
    
    # Check if we have enough memory
    if ! check_gpu_memory; then
        echo "❌ Insufficient GPU memory for experiment ${run_name}, skipping..."
        echo "SKIPPED: Insufficient GPU memory" > "${log_file}"
        return 1
    fi
    
    # Create experiment-specific output directory
    mkdir -p "$output_dir"
    
    # Record experiment parameters
    cat > "${output_dir}/experiment_config.txt" << EOF
Experiment: ${run_name}
Start time: $(date)
Parameters:
  max_seq_len: ${max_seq_len}
  code_extraction_probability: ${code_extraction_prob}
  max_steps: ${MAX_STEPS}
  per_device_train_batch_size: ${BATCH_SIZE}
  gradient_accumulation_steps: ${GRAD_ACCUMULATION}
  learning_rate: ${LEARNING_RATE}
Output directory: ${output_dir}
EOF
    
    # Run the training script
    local start_time=$(date +%s)
    
    if python3 "$SCRIPT_PATH" \
        --output_dir "$output_dir" \
        --max_seq_len "$max_seq_len" \
        --code_extraction_probability "$code_extraction_prob" \
        --max_steps "$MAX_STEPS" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUMULATION" \
        --learning_rate "$LEARNING_RATE" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$run_name" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_tags "sweep,max_seq_len_${max_seq_len},code_prob_${code_extraction_prob}" \
        > "$log_file" 2>&1; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "✅ Experiment ${run_name} completed successfully in ${duration}s"
        
        # Record success
        echo "SUCCESS: Completed in ${duration}s" >> "${output_dir}/experiment_config.txt"
        echo "End time: $(date)" >> "${output_dir}/experiment_config.txt"
        
        # Extract key metrics from log if available
        if [ -f "$log_file" ]; then
            echo "📊 Extracting metrics..."
            grep -E "(loss|perplexity|GPU memory)" "$log_file" | tail -10 > "${output_dir}/final_metrics.txt" || true
        fi
        
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "❌ Experiment ${run_name} failed after ${duration}s"
        
        # Record failure
        echo "FAILED: After ${duration}s" >> "${output_dir}/experiment_config.txt"
        echo "End time: $(date)" >> "${output_dir}/experiment_config.txt"
        echo "Check log file: ${log_file}" >> "${output_dir}/experiment_config.txt"
        
        # Show last few lines of error log
        echo "Last 20 lines of error log:"
        tail -20 "$log_file" || true
        
        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    local summary_file="${RESULTS_DIR}/sweep_summary.txt"
    
    echo "📋 Generating sweep summary..."
    
    cat > "$summary_file" << EOF
Parameter Sweep Summary
======================
Script: $SCRIPT_PATH
Date: $(date)
Base output directory: $BASE_OUTPUT_DIR

Parameter Ranges:
- max_seq_len: ${MAX_SEQ_LEN_VALUES[*]}
- code_extraction_probability: ${CODE_EXTRACTION_PROB_VALUES[*]}

Training Configuration:
- max_steps: $MAX_STEPS
- batch_size: $BATCH_SIZE
- gradient_accumulation_steps: $GRAD_ACCUMULATION
- learning_rate: $LEARNING_RATE

Results:
========
EOF

    local total_experiments=0
    local successful_experiments=0
    local failed_experiments=0
    
    for max_seq_len in "${MAX_SEQ_LEN_VALUES[@]}"; do
        for code_extraction_prob in "${CODE_EXTRACTION_PROB_VALUES[@]}"; do
            local run_name="seqlen_${max_seq_len}_codeprob_${code_extraction_prob}"
            local output_dir="${BASE_OUTPUT_DIR}/${run_name}"
            local config_file="${output_dir}/experiment_config.txt"
            
            total_experiments=$((total_experiments + 1))
            
            if [ -f "$config_file" ]; then
                if grep -q "SUCCESS:" "$config_file"; then
                    successful_experiments=$((successful_experiments + 1))
                    local duration=$(grep "SUCCESS:" "$config_file" | sed 's/.*in \([0-9]*\)s.*/\1/')
                    echo "✅ $run_name - Completed in ${duration}s" >> "$summary_file"
                else
                    failed_experiments=$((failed_experiments + 1))
                    local reason=$(grep -E "(FAILED:|SKIPPED:)" "$config_file" | head -1)
                    echo "❌ $run_name - $reason" >> "$summary_file"
                fi
            else
                failed_experiments=$((failed_experiments + 1))
                echo "❓ $run_name - No config file found" >> "$summary_file"
            fi
        done
    done
    
    echo "" >> "$summary_file"
    echo "Statistics:" >> "$summary_file"
    echo "- Total experiments: $total_experiments" >> "$summary_file"
    echo "- Successful: $successful_experiments" >> "$summary_file"
    echo "- Failed/Skipped: $failed_experiments" >> "$summary_file"
    echo "- Success rate: $(( (successful_experiments * 100) / total_experiments ))%" >> "$summary_file"
    
    echo "📊 Summary saved to: $summary_file"
    
    # Display summary
    echo ""
    echo "=== SWEEP SUMMARY ==="
    cat "$summary_file"
}

# Main execution
main() {
    echo "🔬 Starting parameter sweep for lora_gpt_2048_flash.py"
    echo "📁 Results will be saved to: $BASE_OUTPUT_DIR"
    echo ""
    
    # Check if script exists
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "❌ Error: Script $SCRIPT_PATH not found!"
        exit 1
    fi
    
    # Check for CUDA
    if ! python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
        echo "❌ Error: PyTorch CUDA not available!"
        exit 1
    fi
    
    # Log sweep configuration
    echo "Parameter sweep configuration:"
    echo "- max_seq_len values: ${MAX_SEQ_LEN_VALUES[*]}"
    echo "- code_extraction_probability values: ${CODE_EXTRACTION_PROB_VALUES[*]}"
    echo "- Total experiments: $((${#MAX_SEQ_LEN_VALUES[@]} * ${#CODE_EXTRACTION_PROB_VALUES[@]}))"
    echo ""
    
    local experiment_count=0
    local total_experiments=$((${#MAX_SEQ_LEN_VALUES[@]} * ${#CODE_EXTRACTION_PROB_VALUES[@]}))
    local successful_count=0
    
    # Record start time
    local sweep_start_time=$(date +%s)
    
    # Run all combinations
    for max_seq_len in "${MAX_SEQ_LEN_VALUES[@]}"; do
        for code_extraction_prob in "${CODE_EXTRACTION_PROB_VALUES[@]}"; do
            experiment_count=$((experiment_count + 1))
            echo ""
            echo "🔄 Experiment $experiment_count of $total_experiments"
            echo "=================================================="
            
            if run_experiment "$max_seq_len" "$code_extraction_prob"; then
                successful_count=$((successful_count + 1))
            fi
            
            # Progress update
            echo "Progress: $experiment_count/$total_experiments completed, $successful_count successful"
            
            # Brief pause between experiments
            echo "⏳ Waiting 10 seconds before next experiment..."
            sleep 10
        done
    done
    
    local sweep_end_time=$(date +%s)
    local sweep_duration=$((sweep_end_time - sweep_start_time))
    
    echo ""
    echo "🏁 Parameter sweep completed!"
    echo "Total time: ${sweep_duration}s ($((sweep_duration / 60)) minutes)"
    echo "Successful experiments: $successful_count/$total_experiments"
    
    # Generate final summary
    generate_summary
    
    echo ""
    echo "📂 All results saved to: $BASE_OUTPUT_DIR"
    echo "📋 Summary report: ${RESULTS_DIR}/sweep_summary.txt"
    echo "📝 Individual logs: $LOG_DIR"
}

# Handle interruption gracefully
trap 'echo ""; echo "🛑 Sweep interrupted by user"; cleanup_gpu; exit 130' INT

# Check command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--dry-run]"
    echo ""
    echo "Parameter sweep script for lora_gpt_2048_flash.py"
    echo ""
    echo "Options:"
    echo "  --dry-run    Show what would be executed without running experiments"
    echo "  --help, -h   Show this help message"
    echo ""
    echo "Configuration:"
    echo "  max_seq_len values: ${MAX_SEQ_LEN_VALUES[*]}"
    echo "  code_extraction_probability values: ${CODE_EXTRACTION_PROB_VALUES[*]}"
    echo "  Total experiments: $((${#MAX_SEQ_LEN_VALUES[@]} * ${#CODE_EXTRACTION_PROB_VALUES[@]}))"
    exit 0
fi

if [ "$1" = "--dry-run" ]; then
    echo "🔍 DRY RUN - Showing planned experiments:"
    echo ""
    count=0
    for max_seq_len in "${MAX_SEQ_LEN_VALUES[@]}"; do
        for code_extraction_prob in "${CODE_EXTRACTION_PROB_VALUES[@]}"; do
            count=$((count + 1))
            echo "$count. max_seq_len=$max_seq_len, code_extraction_probability=$code_extraction_prob"
        done
    done
    echo ""
    echo "Total: $count experiments would be run"
    echo "Estimated time: ~$((count * 15)) minutes (assuming 15 min per experiment)"
    exit 0
fi

# Run the main function
main "$@"
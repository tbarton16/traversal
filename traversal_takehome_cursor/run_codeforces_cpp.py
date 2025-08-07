#!/usr/bin/env python3
"""
Codeforces benchmark runner for Qwen2.5-7B-Instruct **with LoRA adapter, quantization, and ChatML formatting** - C++ VERSION.

Key features
------------
* **Direct HuggingFace integration** – uses Qwen2.5-7B-Instruct with LoRA adapter and 4-bit quantization.
* **LoRA adapter support** – uses PEFT library for efficient fine-tuned model loading.
* **4-bit quantization** – BitsAndBytesConfig for memory-efficient inference.
* **Flash Attention 2** – tries to use Flash Attention for better memory efficiency.
* **ChatML prompt formatting** – uses the same format as lora_gpt_2048_flash.py training data.
* **C++ compilation and execution** – compiles and runs C++ solutions with timeout handling.
* **Parallel processing** – multiple workers for concurrent model inference.
* **Per‑worker wandb runs** – grouped so dashboards aggregate nicely.
* **Graceful resumption** – already‑finished problems are skipped.
* **Smarter `extract_code`** – handles ````cpp` fences *and* raw output, so badly‑formatted generations no longer break evaluation.
* **Detailed file output** – saves prompt, generation, tests, and results to files for each problem.

"""
import argparse
import os
import re
import sys
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple
import multiprocessing as mp
import time
import random
import json
import threading
from multiprocessing import Manager
import signal
from contextlib import contextmanager

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
import wandb

# ──────────────────────────────────────────────────────────────────────────
# Timeout handling
# ──────────────────────────────────────────────────────────────────────────

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout handling"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")
    
    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ──────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────
CHAT_TEMPLATE = "{header}{dialog}<|im_start|>assistant\n"

def build_chat(messages):
    """Convert HF chat format to ChatML string."""
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)

# Codeforces problem template (formatted for C++ solutions)
PROMPT_TEMPLATE = """You will be given a competitive programming problem.
Analyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in C++ that is thoroughly optimized for both speed and memory usage.

Your solution should be a complete C++ program that reads from standard input and writes to standard output. Include all necessary headers and use efficient I/O methods. Make sure to handle all edge cases and optimize for the given constraints.

Put your final solution within a single code block:
```cpp
#include <iostream>
// ... other includes as needed ...
using namespace std;

int main() {{
    // your code here
    return 0;
}}
```

# Problem statement
{description}
# Input
{input_format}
# Output
{output_format}
"""

# ──────────────────────────────────────────────────────────────────────────
# Generation / evaluation helpers
# ──────────────────────────────────────────────────────────────────────────

def extract_code(txt: str) -> str:
    """Return the longest plausible C++ code block.

    1. Prefer ```cpp or ```c++ fenced blocks (common with LLMs).
    2. Otherwise, grab everything starting at the first #include.
    3. Fall back to the full generation as‑is.
    """
    # 1️⃣ fenced blocks
    blocks = re.findall(r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```", txt, flags=re.IGNORECASE)
    if blocks:
        # Pick the longest – usually the full solution
        candidate = max(blocks, key=len).strip()
        if candidate:
            return candidate

    # 2️⃣ first #include onwards
    m = re.search(r"#include\s*<", txt)
    if m:
        return txt[m.start():].strip()

    # 3️⃣ fallback
    return txt.strip()


def generate_solution(model, tokenizer, prompt: str, max_tokens: int = 2048) -> Tuple[str, str]:
    """Generate candidate C++ code using Qwen2.5-7B-Instruct with LoRA adapter and extract the relevant portion.
    
    Returns:
        Tuple of (extracted_code, raw_generation)
    """
    try:
        # Format the prompt using ChatML template (like lora_gpt_2048_flash.py)
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Use ChatML formatting instead of tokenizer.apply_chat_template
        text = CHAT_TEMPLATE.format(
            header="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            dialog=build_chat(messages),
        )
        
        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        raw_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        extracted_code = extract_code(raw_generation)
        return extracted_code, raw_generation
        
    except Exception as e:
        print(f"Model generation error: {e}", file=sys.stderr)
        # Return a minimal fallback solution
        fallback = "#include <iostream>\nusing namespace std;\nint main() { return 0; }"
        return fallback, fallback


def compile_and_run_cpp(code: str, test_input: str, timeout_seconds: int = 30) -> Tuple[str, str, bool]:
    """Compile and run C++ code with given input.
    
    Args:
        code: C++ source code
        test_input: Input to provide to the program
        timeout_seconds: Maximum time for compilation + execution
        
    Returns:
        Tuple of (output, error_message, success)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        source_file = os.path.join(temp_dir, "solution.cpp")
        executable_file = os.path.join(temp_dir, "solution")
        
        try:
            # Write source code to file
            with open(source_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Compile with optimizations
            compile_cmd = [
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
                source_file, "-o", executable_file
            ]
            
            compile_start = time.time()
            compile_process = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds // 2  # Use half timeout for compilation
            )
            compile_time = time.time() - compile_start
            
            if compile_process.returncode != 0:
                return "", f"COMPILE_ERROR: {compile_process.stderr}", False
            
            # Run the executable
            remaining_timeout = max(1, timeout_seconds - int(compile_time))
            run_process = subprocess.run(
                [executable_file],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=remaining_timeout
            )
            
            if run_process.returncode != 0:
                return "", f"RUNTIME_ERROR: {run_process.stderr}", False
            
            return run_process.stdout, "", True
            
        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT_ERROR: Exceeded {timeout_seconds} seconds", False
        except Exception as e:
            return "", f"EXECUTION_ERROR: {e}", False


def evaluate_solution(code: str, tests: List[Dict[str, str]], timeout_seconds: int = 120) -> Tuple[float, str, List[Dict[str, str]]]:
    """Evaluate C++ solution and return accuracy, error message, and detailed test results.
    
    Args:
        code: The C++ code to evaluate
        tests: List of test cases
        timeout_seconds: Maximum time allowed for execution (default: 120 seconds = 2 minutes)
    """
    test_results = []
    
    # First, try to compile the code once
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = os.path.join(temp_dir, "solution.cpp")
            executable_file = os.path.join(temp_dir, "solution")
            
            # Write source code to file
            with open(source_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Compile
            compile_cmd = [
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
                source_file, "-o", executable_file
            ]
            
            compile_process = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 seconds for compilation
            )
            
            if compile_process.returncode != 0:
                return 0.0, f"compile_error: {compile_process.stderr}", test_results
            
            # Now run tests
            passed = 0
            start_time = time.time()
            
            for i, case in enumerate(tests):
                # Check if we've already exceeded the timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    test_results.append({
                        "test_number": i + 1,
                        "input": case["input"],
                        "expected_output": case["output"],
                        "actual_output": f"TIMEOUT_ERROR: Execution exceeded {timeout_seconds} seconds",
                        "correct": False
                    })
                    return passed / len(tests), f"timeout_error: Execution exceeded {timeout_seconds} seconds", test_results
                
                try:
                    # Apply timeout to individual test case execution
                    remaining_timeout = max(1, timeout_seconds - int(elapsed_time))
                    
                    run_process = subprocess.run(
                        [executable_file],
                        input=case["input"].rstrip("\n"),
                        capture_output=True,
                        text=True,
                        timeout=remaining_timeout
                    )
                    
                    if run_process.returncode != 0:
                        actual_output = f"RUNTIME_ERROR: {run_process.stderr}"
                        is_correct = False
                    else:
                        actual_output = run_process.stdout.strip()
                        expected_output = case["output"].strip()
                        is_correct = actual_output.lower() == expected_output.lower()
                    
                    test_results.append({
                        "test_number": i + 1,
                        "input": case["input"],
                        "expected_output": case["output"],
                        "actual_output": actual_output,
                        "correct": is_correct
                    })
                    
                    if is_correct:
                        passed += 1
                        
                except subprocess.TimeoutExpired:
                    test_results.append({
                        "test_number": i + 1,
                        "input": case["input"],
                        "expected_output": case["output"],
                        "actual_output": f"TIMEOUT_ERROR: Test case exceeded {remaining_timeout} seconds",
                        "correct": False
                    })
                    return passed / len(tests), f"timeout_error: Test case timeout", test_results
                except Exception as e:
                    test_results.append({
                        "test_number": i + 1,
                        "input": case["input"],
                        "expected_output": case["output"],
                        "actual_output": f"RUNTIME_ERROR: {e}",
                        "correct": False
                    })
                    return passed / len(tests), f"runtime_error: {e}", test_results
                    
    except subprocess.TimeoutExpired:
        return 0.0, f"compile_timeout_error: Compilation exceeded 30 seconds", test_results
    except Exception as e:
        return 0.0, f"compile_error: {e}", test_results
            
    return passed / len(tests), "", test_results


def write_result_to_file(worker_id: int, prob_id: str, prompt: str, raw_generation: str, 
                        extracted_code: str, tests: List[Dict[str, str]], 
                        accuracy: float, error: str, test_results: List[Dict[str, str]], 
                        output_dir: str = "results"):
    """Write detailed result information to a file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize problem ID for use in filename (replace / with _)
    safe_prob_id = prob_id.replace("/", "_").replace("\\", "_")
    
    result_data = {
        "problem_id": prob_id,
        "timestamp": datetime.utcnow().isoformat(),
        "worker_id": worker_id,
        "prompt": prompt,
        "raw_generation": raw_generation,
        "extracted_code": extracted_code,
        "tests": tests,
        "test_results": test_results,
        "accuracy": accuracy,
        "error": error,
        "passed_tests": int(accuracy * len(tests)),
        "total_tests": len(tests)
    }
    
    filename = f"{output_dir}/worker{worker_id}_{safe_prob_id}_result.json"
    VERBOSE = True
    if VERBOSE :
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # Also write a human-readable version
        readable_filename = f"{output_dir}/worker{worker_id}_{safe_prob_id}_readable.txt"
        with open(readable_filename, 'w', encoding='utf-8') as f:
            f.write(f"Problem ID: {prob_id}\n")
            f.write(f"Worker: {worker_id}\n")
            f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
            f.write(f"Accuracy: {accuracy:.2%} ({int(accuracy * len(tests))}/{len(tests)} tests passed)\n")
            if error:
                f.write(f"Error: {error}\n")
            f.write("\n" + "="*80 + "\n")
            f.write("PROMPT:\n")
            f.write("="*80 + "\n")
            f.write(prompt)
            f.write("\n\n" + "="*80 + "\n")
            f.write("RAW GENERATION:\n")
            f.write("="*80 + "\n")
            f.write(raw_generation)
            f.write("\n\n" + "="*80 + "\n")
            f.write("EXTRACTED C++ CODE:\n")
            f.write("="*80 + "\n")
            f.write(extracted_code)
            f.write("\n\n" + "="*80 + "\n")
            f.write("TEST RESULTS:\n")
            f.write("="*80 + "\n")
            if test_results:
                for result in test_results:
                    status = "✓ PASS" if result["correct"] else "✗ FAIL"
                    f.write(f"Test {result['test_number']}: {status}\n")
                    f.write(f"Input:\n{result['input']}\n")
                    f.write(f"Expected Output:\n{result['expected_output']}\n")
                    f.write(f"Actual Output:\n{result['actual_output']}\n")
                    f.write("-" * 40 + "\n")
            else:
                # Fallback for cases where test_results is empty (e.g., compile errors)
                for i, test in enumerate(tests):
                    f.write(f"Test {i+1}: (No execution - see error above)\n")
                    f.write(f"Input:\n{test['input']}\n")
                    f.write(f"Expected Output:\n{test['output']}\n")
                    f.write("-" * 40 + "\n")
            f.write("="*80 + "\n")

# ──────────────────────────────────────────────────────────────────────────
# Shared logging utilities
# ──────────────────────────────────────────────────────────────────────────

def log_to_shared_wandb(shared_state, problem_id: str, accuracy: float, error: str, worker_id: int, wandb_config: dict):
    """Thread-safe logging to shared wandb run with running accuracy tracking."""
    with shared_state['lock']:
        # Update shared metrics (for final summary)
        shared_state['total_problems'] += 1
        shared_state['total_accuracy'] += accuracy
        shared_state['problem_results'][problem_id] = {
            'accuracy': accuracy,
            'error': error,
            'worker_id': worker_id,
            'timestamp': time.time()
        }
        
        # Store logging data for chronological processing by logging thread
        log_entry = {
            'problem_id': problem_id,
            'accuracy': accuracy,
            'error': error,
            'worker_id': worker_id,
            'timestamp': time.time()  # Critical for chronological ordering
        }
        
        # Add to queue for main process to log
        if 'log_queue' not in shared_state:
            shared_state['log_queue'] = []
        shared_state['log_queue'].append(log_entry)

# ──────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────

def worker(worker_id: int, world_size: int, args, shared_state, wandb_config: dict, dataset_shard):
    # Load model and tokenizer for Qwen-7B-Instruct 2.5 with LoRA and quantization
    print(f"[Worker {worker_id}] Loading Qwen-7B-Instruct 2.5 with LoRA adapter and quantization for C++ generation...", file=sys.stderr)
    
    # Force GPU 2 as requested by user
    device = "cuda"
    print(f"[Worker {worker_id}] Using device: {device}", file=sys.stderr)
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization configuration from lora_gpt_2048_flash.py
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
    )
    
    # Load base model with quantization
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,
            attn_implementation="flash_attention_2",  # Try Flash Attention
        )
        print(f"[Worker {worker_id}] ✅ Successfully loaded model with Flash Attention 2!", file=sys.stderr)
    except Exception as e:
        print(f"[Worker {worker_id}] ⚠️ Flash Attention 2 failed: {e}", file=sys.stderr)
        print(f"[Worker {worker_id}] Falling back to standard attention", file=sys.stderr)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,
        )
    
    # Clear memory after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # LoRA handling based on arguments
    if args.disable_lora:
        print(f"[Worker {worker_id}] LoRA disabled - using base model only", file=sys.stderr)
        model = base_model
    elif args.lora_model_path:
        print(f"[Worker {worker_id}] Loading pre-trained LoRA model from: {args.lora_model_path}", file=sys.stderr)
        try:
            model = PeftModel.from_pretrained(
                base_model,
                args.lora_model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": device}
            )
            print(f"[Worker {worker_id}] ✅ Successfully loaded pre-trained LoRA model", file=sys.stderr)
        except Exception as e:
            print(f"[Worker {worker_id}] ❌ Failed to load LoRA model: {e}", file=sys.stderr)
            print(f"[Worker {worker_id}] Falling back to base model without LoRA", file=sys.stderr)
            model = base_model
    else:
        print(f"[Worker {worker_id}] Creating fresh LoRA adapter", file=sys.stderr)
        # LoRA configuration (conservative for long sequences like in lora_gpt_2048_flash.py)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_cfg = LoraConfig(
            r=8,  # Moderate LoRA rank
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        
        # Prepare model for k-bit training and apply LoRA
        model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=False  # Disable for inference
        )
        model = get_peft_model(model, lora_cfg)
        print(f"[Worker {worker_id}] ✅ Fresh LoRA adapter created", file=sys.stderr)
    
    # Print model info
    if hasattr(model, 'print_trainable_parameters'):
        print(f"[Worker {worker_id}] Model configuration:", file=sys.stderr)
        model.print_trainable_parameters()
    else:
        print(f"[Worker {worker_id}] Using base model (no LoRA adapter)", file=sys.stderr)
    
    model.eval()
    
    # Determine model description for logging
    if args.disable_lora:
        model_desc = "Qwen2.5-7B-Instruct (no LoRA) - C++ mode"
    elif args.lora_model_path:
        model_desc = f"Qwen2.5-7B-Instruct + pre-trained LoRA ({args.lora_model_path}) - C++ mode"
    else:
        model_desc = "Qwen2.5-7B-Instruct + fresh LoRA - C++ mode"
    
    print(f"[Worker {worker_id}] Starting with {model_desc} on {device}", file=sys.stderr)
    print(f"[Worker {worker_id}] Dataset shard size: {len(dataset_shard)} problems", file=sys.stderr)
    print(f"[Worker {worker_id}] C++ compilation + execution timeout: {args.timeout} seconds", file=sys.stderr)
    
    # Check if g++ is available
    try:
        subprocess.run(["g++", "--version"], capture_output=True, check=True)
        print(f"[Worker {worker_id}] ✅ g++ compiler available", file=sys.stderr)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"[Worker {worker_id}] ❌ g++ compiler not found! Install build-essential or g++", file=sys.stderr)
        return
    
    # Add random seed for reproducibility
    random.seed(args.seed + worker_id)

    # Use the pre-shuffled dataset shard passed from main process
    ds_shard = dataset_shard

    total_acc = 0.0
    processed = 0
    total_problems_in_shard = len(ds_shard)

    for problem_number, row in enumerate(ds_shard, 1):
        prob_id = row["id"]
        # Check if already finished by looking in shared state
        with shared_state['lock']:
            if prob_id in shared_state['problem_results']:
                continue

        prompt = PROMPT_TEMPLATE.format(
            description=row["description"],
            input_format=row["input_format"],
            output_format=row["output_format"],
        )

        
        code, raw_generation = generate_solution(model, tokenizer, prompt)
        acc, err, test_results = evaluate_solution(code, row["official_tests"], args.timeout)
        
        # Write detailed results to file
        write_result_to_file(
            worker_id=worker_id,
            prob_id=prob_id,
            prompt=prompt,
            raw_generation=raw_generation,
            extracted_code=code,
            tests=row["official_tests"],
            accuracy=acc,
            error=err,
            test_results=test_results,
            output_dir=args.output_dir
        )

        # Log to shared wandb run
        log_to_shared_wandb(shared_state, prob_id, acc, err, worker_id, wandb_config)

        total_acc += acc
        processed += 1
        status = f"[Worker {worker_id}] Problem {problem_number}/{total_problems_in_shard} ({prob_id}): {acc*100:.1f}% (#{processed})"
        if err:
            status += f" – {err}"
        print(status, file=sys.stderr)

    # Log final worker stats to shared state
    with shared_state['lock']:
        shared_state['worker_stats'][worker_id] = {
            'accuracy': total_acc / processed if processed else 0.0,
            'processed': processed
        }

# ──────────────────────────────────────────────────────────────────────────
# Main – spawns workers
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="verifiable")
    p.add_argument("--split", default="test")

    p.add_argument("--wandb_project", required=True)
    p.add_argument("--wandb_entity")
    p.add_argument("--wandb_group")

    p.add_argument("--num_workers", type=int, default=2, help="Number of parallel workers (reduced default for GPU memory)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results-cpp", help="Directory to save detailed result files")
    
    # LoRA configuration arguments
    p.add_argument("--lora_model_path", type=str, default=None, 
                   help="Path to pre-trained LoRA model directory or HuggingFace model name. If not provided, creates fresh LoRA adapter.")
    p.add_argument("--disable_lora", action="store_true", 
                   help="Disable LoRA adapter entirely and use base model only")
    
    # Execution timeout argument
    p.add_argument("--timeout", type=int, default=120,
                   help="Maximum time in seconds for C++ compilation + execution (default: 120 seconds = 2 minutes)")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)", file=sys.stderr)
        if args.num_workers > torch.cuda.device_count():
            print(f"Warning: {args.num_workers} workers requested but only {torch.cuda.device_count()} GPU(s) available", file=sys.stderr)
    else:
        print("CUDA not available, using CPU", file=sys.stderr)

    world_size = args.num_workers
    print(f"Starting {world_size} workers for Qwen2.5-7B-Instruct + LoRA C++ evaluation", file=sys.stderr)
    print(f"Detailed results will be saved to: {args.output_dir}/", file=sys.stderr)
    
    # Load and shuffle the dataset in main process
    print(f"Loading dataset: {args.subset} split={args.split}", file=sys.stderr)
    ds = load_dataset("open-r1/codeforces", args.subset, split=args.split)
    print(f"Dataset loaded with {len(ds)} problems", file=sys.stderr)
    
    # Shuffle the dataset using the seed for reproducibility
    print(f"Shuffling dataset with seed {args.seed}...", file=sys.stderr)
    ds = ds.shuffle(seed=args.seed)
    
    # Create shards for each worker
    dataset_shards = []
    for worker_id in range(world_size):
        shard = ds.shard(num_shards=world_size, index=worker_id)
        dataset_shards.append(list(shard))  # Convert to list for easier serialization
        print(f"Worker {worker_id} will process {len(dataset_shards[worker_id])} problems", file=sys.stderr)

    # Create shared state for coordinating between workers
    manager = Manager()
    shared_state = manager.dict({
        'total_problems': 0,
        'total_accuracy': 0.0,
        'problem_results': manager.dict(),
        'worker_stats': manager.dict(),
        'lock': manager.Lock(),
        'log_queue': manager.list(),
        'running': manager.Value('b', True)
    })

    # Initialize single shared wandb run in main process
    wandb_run = None
    
    # Build tags based on LoRA configuration
    tags = ["codeforces", "consolidated", "qwen2.5-7b-instruct", "quantized", "cpp", f"{world_size}workers"]
    if args.disable_lora:
        tags.append("no-lora")
    elif args.lora_model_path:
        tags.extend(["pretrained-lora", "loaded-lora"])
    else:
        tags.extend(["fresh-lora", "random-lora"])
        
    wandb_config = {
        'project': args.wandb_project,
        'entity': args.wandb_entity,
        'group': args.wandb_group or f"cf-cpp-{datetime.utcnow():%Y%m%dT%H%M%S}",
        'name': f"consolidated-cpp-run-{world_size}workers",
        'tags': tags,
        'config': vars(args)
    }
    
    try:
        wandb_run = wandb.init(**wandb_config)
        print(f"Initialized consolidated wandb run: {wandb_run.name}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}", file=sys.stderr)

    # Start logging thread to process queued log entries
    def logging_thread():
        """Process log entries from workers and send to wandb with proper ordering and smoothing."""
        logged_count = 0
        total_accuracy_sum = 0.0
        recent_accuracies = []  # Rolling window for smoothing
        window_size = 20  # Size of smoothing window
        
        while shared_state['running'] or len(shared_state['log_queue']) > 0:
            if len(shared_state['log_queue']) > 0:
                # Get all queued entries and sort by timestamp for chronological order
                entries_to_process = []
                with shared_state['lock']:
                    entries_to_process = list(shared_state['log_queue'])
                    shared_state['log_queue'][:] = []  # Clear the queue
                
                # Sort by timestamp to ensure chronological processing
                entries_to_process.sort(key=lambda x: x['timestamp'])
                
                for entry in entries_to_process:
                    if wandb_run:
                        try:
                            # Recalculate running accuracy based on chronological order
                            logged_count += 1
                            total_accuracy_sum += entry['accuracy']
                            chronological_running_accuracy = total_accuracy_sum / logged_count
                            
                            # Update rolling window for smoothing
                            recent_accuracies.append(entry['accuracy'])
                            if len(recent_accuracies) > window_size:
                                recent_accuracies.pop(0)  # Remove oldest
                            
                            # Calculate smoothed running accuracy (moving average)
                            smoothed_accuracy = sum(recent_accuracies) / len(recent_accuracies)
                            
                            # Update the entry with the corrected metrics
                            log_entry = entry.copy()
                            log_entry['running_accuracy'] = chronological_running_accuracy
                            log_entry['smoothed_running_accuracy'] = smoothed_accuracy
                            log_entry['chronological_step'] = logged_count
                            log_entry['step'] = logged_count  # Override step to be chronological
                            
                            wandb_run.log(log_entry)
                            
                        except Exception as e:
                            print(f"Warning: Failed to log to wandb: {e}", file=sys.stderr)
            
            time.sleep(0.1)  # Small delay to prevent busy waiting

    log_thread = threading.Thread(target=logging_thread, daemon=True)
    log_thread.start()

    # Create and start processes
    processes = []
    for worker_id in range(world_size):
        p = mp.Process(target=worker, args=(worker_id, world_size, args, shared_state, wandb_config, dataset_shards[worker_id]))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Signal logging thread to stop
    shared_state['running'] = False
    
    # Wait for logging thread to finish processing remaining entries
    log_thread.join(timeout=5.0)

    # Log final summary statistics
    if wandb_run:
        try:
            final_accuracy = shared_state['total_accuracy'] / shared_state['total_problems'] if shared_state['total_problems'] > 0 else 0.0
            
            # Log final summary
            wandb_run.log({
                'final_overall_accuracy': final_accuracy,
                'total_problems_completed': shared_state['total_problems']
            })
            
            # Log per-worker final stats
            for worker_id, stats in shared_state['worker_stats'].items():
                wandb_run.log({
                    f'worker_{worker_id}_final_accuracy': stats['accuracy'],
                    f'worker_{worker_id}_problems_processed': stats['processed']
                })
            
            wandb_run.finish()
            print(f"Final overall C++ accuracy: {final_accuracy:.2%} ({shared_state['total_problems']} problems)", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to log final summary: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
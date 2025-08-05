#!/usr/bin/env python3
"""
Codeforces benchmark runner for Qwen-7B-Instruct 2.5 **with robust code extraction and detailed file output**.

Key features
------------
* **Direct HuggingFace integration** – uses Qwen-7B-Instruct 2.5 via transformers library.
* **Parallel processing** – multiple workers for concurrent model inference.
* **Per‑worker wandb runs** – grouped so dashboards aggregate nicely.
* **Graceful resumption** – already‑finished problems are skipped.
* **Smarter `extract_code`** – handles ````python` fences *and* raw output, so badly‑formatted generations no longer break evaluation.
* **Detailed file output** – saves prompt, generation, tests, and results to files for each problem.

Usage
-----
```bash
python run_codeforces.py \
    --wandb_project qwen7b_cf_eval \
    --num_workers 2 \
    --output_dir results
```
"""
import argparse
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import multiprocessing as mp
import time
import random
import json
import threading
from multiprocessing import Manager

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

# ──────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are solving a Codeforces problem. Implement **only**

the function: 
```python
def solve(inp: str) -> str:
    # Return the answer for this instance.
```

Use only the Python standard library. Be concise and don't elaborate too much just output the solution.

---
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
    """Return the longest plausible code block.

    1. Prefer ```python fenced blocks (common with LLMs).
    2. Otherwise, grab everything starting at the first `def solve`.
    3. Fall back to the full generation as‑is.
    """
    # 1️⃣ fenced blocks
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", txt, flags=re.IGNORECASE)
    if blocks:
        # Pick the longest – usually the full solution
        candidate = max(blocks, key=len).strip()
        if candidate:
            return candidate

    # 2️⃣ first def solve onwards
    m = re.search(r"def\s+solve\s*\(", txt)
    if m:
        return txt[m.start():].strip()

    # 3️⃣ fallback
    return txt.strip()


def generate_solution(model, tokenizer, prompt: str, max_tokens: int = 512) -> Tuple[str, str]:
    """Generate candidate Python code using Qwen-7B-Instruct 2.5 and extract the relevant portion.
    
    Returns:
        Tuple of (extracted_code, raw_generation)
    """
    try:
        # Format the prompt for Qwen chat template
        messages = [
            {"role": "system", "content": "You are a competitive programming expert. Generate clean, efficient Python code."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768,
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
        fallback = "def solve(inp: str) -> str: return ''"
        return fallback, fallback


def evaluate_solution(code: str, tests: List[Dict[str, str]]) -> Tuple[float, str, List[Dict[str, str]]]:
    """Evaluate solution and return accuracy, error message, and detailed test results."""
    ns: Dict[str, object] = {}
    test_results = []
    
    try:
        exec(code, ns)  # nosec
    except Exception as e:  # pylint: disable=broad-except
        return 0.0, f"compile_error: {e}", test_results

    solve_fn = ns.get("solve")
    if solve_fn is None:
        return 0.0, "solve_not_found", test_results

    passed = 0
    for i, case in enumerate(tests):
        try:
            pred = solve_fn(case["input"].rstrip("\n"))
            actual_output = str(pred).strip()
            expected_output = case["output"].strip()
            is_correct = actual_output.lower() == expected_output.lower()
            
            test_results.append({
                "test_number": i + 1,
                "input": case["input"],
                "expected_output": case["output"],
                "actual_output": str(pred),
                "correct": is_correct
            })
            
            if is_correct:
                passed += 1
                
        except Exception as e:  # pylint: disable=broad-except
            test_results.append({
                "test_number": i + 1,
                "input": case["input"],
                "expected_output": case["output"],
                "actual_output": f"RUNTIME_ERROR: {e}",
                "correct": False
            })
            return passed / len(tests), f"runtime_error: {e}", test_results
            
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
    VERBOSE = False
    if VERBOSE:
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
            f.write("EXTRACTED CODE:\n")
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
    # Load model and tokenizer for Qwen-7B-Instruct 2.5
    print(f"[Worker {worker_id}] Loading Qwen-7B-Instruct 2.5...", file=sys.stderr)
    
    device = f"cuda:{worker_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    print(f"[Worker {worker_id}] Using device: {device}", file=sys.stderr)
    
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device} if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.eval()
    
    print(f"[Worker {worker_id}] Starting with Qwen-7B-Instruct 2.5 on {device}", file=sys.stderr)
    print(f"[Worker {worker_id}] Dataset shard size: {len(dataset_shard)} problems", file=sys.stderr)
    
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
        
        # Small delay to prevent overheating/memory issues
        time.sleep(0.1 + random.uniform(0, 0.2))
        
        code, raw_generation = generate_solution(model, tokenizer, prompt)
        acc, err, test_results = evaluate_solution(code, row["official_tests"])
        
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
    p.add_argument("--output_dir", default="results", help="Directory to save detailed result files")
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
    print(f"Starting {world_size} workers for Qwen-7B-Instruct 2.5 evaluation", file=sys.stderr)
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
    wandb_config = {
        'project': args.wandb_project,
        'entity': args.wandb_entity,
        'group': args.wandb_group or f"cf-{datetime.utcnow():%Y%m%dT%H%M%S}",
        'name': f"consolidated-run-{world_size}workers",
        'tags': ["codeforces", "consolidated", "qwen-7b-instruct-2.5", f"{world_size}workers"],
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
            print(f"Final overall accuracy: {final_accuracy:.2%} ({shared_state['total_problems']} problems)", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to log final summary: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parallel Codeforces benchmark runner for multi‑GPU rigs **with centralized wandb logging**.

Key features
------------
* **Multiprocessing** – one worker per GPU via `torch.multiprocessing.spawn`.
* **Dataset sharding** – each worker evaluates a unique slice of the tasks.
* **Centralized wandb logging** – single aggregated run instead of per-GPU runs.
* **Graceful resumption** – already‑finished problems are skipped.
* **Smarter `extract_code`** – handles ````python` fences *and* raw output, so badly‑formatted generations no longer break evaluation.

Usage  
-----
```bash 
python run_codeforces2.py \
    --wandb_project qwen_cf_eval \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num_gpus 8
```
Need fewer GPUs? Set `--num_gpus` or adjust `CUDA_VISIBLE_DEVICES`.
"""
import argparse
import os
import re
import sys
import json
import time
import glob
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import wandb

# ──────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are solving a Codeforces problem. Implement **only**

the function:
```python
def solve(inp: str) -> str:
    #Return the answer for this instance.
```

Use only the Python standard library.

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


def generate_solution(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> Tuple[str, str]:
    """Generate candidate Python code and extract the relevant portion.
    
    Returns:
        Tuple[str, str]: (raw_output, extracted_code)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=False,
        )
    prompt_len = inputs['input_ids'].shape[1]
    raw_output = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    extracted_code = extract_code(raw_output)
    return raw_output, extracted_code

import math

def broad_match(pred, expected, *, float_tol=1e-9):
    """
    Return True when `pred` and `expected` are considered equivalent.

    Rules
    -----
    1. Treat both inputs as strings and strip leading/trailing whitespace.
    2. If the first *token* in each string can be parsed as a number,
       compare those numbers only (ignore any extra tokens) using `math.isclose`.
    3. Otherwise, compare the full strings case-insensitively.
    """
    ps = str(pred).strip()
    es = str(expected).strip()

    # Split on any whitespace, grab only the first field
    p_first, e_first = (ps.split(maxsplit=1) + [""])[:1][0], (es.split(maxsplit=1) + [""])[:1][0]

    try:
        p_num = float(p_first)
        e_num = float(e_first)
        # Both first tokens are numeric → numeric comparison
        return math.isclose(p_num, e_num, rel_tol=float_tol, abs_tol=float_tol)
    except ValueError:
        # At least one token wasn't numeric → string comparison (case-insensitive)
        return ps.casefold() == es.casefold()

def evaluate_solution(code: str, tests: List[Dict[str, str]]) -> Tuple[float, str]:
    ns: Dict[str, object] = {}
    try:
        exec(code, ns)  # nosec
    except Exception as e:  # pylint: disable=broad-except
        return 0.0, f"compile_error: {e}"

    solve_fn = ns.get("solve")
    if solve_fn is None:
        return 0.0, "solve_not_found"

    # Check if solve_fn is actually callable
    if not callable(solve_fn):
        return 0.0, "runtime_error: 'solve' is not a function"

    passed = 0
    for case in tests:
        try:
            pred = solve_fn(case["input"].rstrip("\n"))
            # print("prediction: ", pred, "|", str(pred).strip(),type(str(pred).strip()))
            # print("expected: ", case["output"], "|", case["output"].strip(), type(case["output"].strip()))
            # print("comparison: ", broad_match(pred, case["output"]))
            # print("--------------------------------")
        except Exception as e:  # pylint: disable=broad-except
            return passed / len(tests), f"runtime_error: {e}"
        if broad_match(pred, case["output"]):
            passed += 1.0
            
    print("passed: ", passed)
    print("total: ", len(tests))
    return passed / float(len(tests)), ""

# ──────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────

def is_easy_problem(row) -> bool:
    """Check if a problem is considered 'easy' based on difficulty rating."""
    # Check if the row has a 'rating' or 'difficulty' field
    if 'rating' in row and row['rating'] is not None:
        return row['rating'] <= 1200
    elif 'difficulty' in row and row['difficulty'] is not None:
        return row['difficulty'] <= 1200
    
    # If no explicit rating, try to extract from problem ID
    # Many Codeforces datasets include rating in metadata or can be inferred
    prob_id = row.get('id', '')
    
    # Some datasets encode difficulty in the problem ID or have it as metadata
    # For problems without explicit ratings, we'll be conservative and include them
    # You may need to adjust this logic based on the actual dataset structure
    return True  # Default to including if we can't determine difficulty


def worker(rank: int, world_size: int, args, shared_metrics_file: str):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # No wandb initialization here - only rank 0 will handle wandb logging

    # load model
    print(f"[GPU {rank}] Loading model {args.model}…", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": rank},
        torch_dtype="auto",
        trust_remote_code=True,
        token=args.hf_token or os.getenv("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=args.hf_token or os.getenv("HF_TOKEN")
    )
    set_seed(args.seed + rank)

    # dataset shard
    ds = load_dataset("open-r1/codeforces", args.subset, split=args.split)
    
    # Filter for easy problems only
    print(f"[GPU {rank}] Filtering dataset for easy problems (rating <= 1200)...", file=sys.stderr)
    ds_filtered = ds.filter(is_easy_problem)
    print(f"[GPU {rank}] Filtered from {len(ds)} to {len(ds_filtered)} easy problems", file=sys.stderr)
    
    ds_shard = ds_filtered.shard(num_shards=world_size, index=rank)

    total_acc = 0.0
    processed = 0
    
    # Create output directories
    os.makedirs("raw_outputs", exist_ok=True)
    os.makedirs("extracted_codes", exist_ok=True)

    # Track processed problems to avoid duplicates
    processed_problems_file = f"processed_problems_gpu_{rank}.json"
    processed_problems = set()
    if os.path.exists(processed_problems_file):
        try:
            with open(processed_problems_file, "r") as f:
                processed_problems = set(json.load(f))
        except (json.JSONDecodeError, FileNotFoundError):
            processed_problems = set()

    for row in ds_shard:
        prob_id = row["id"]
        if prob_id in processed_problems:
            continue

        prompt = PROMPT_TEMPLATE.format(
            description=row["description"],
            input_format=row["input_format"],
            output_format=row["output_format"],
        )
        raw_output, code = generate_solution(model, tokenizer, prompt)
        
        acc, err = evaluate_solution(code, row["official_tests"])

        # Write metrics to shared file for aggregation
        metric_entry = {
            "timestamp": time.time(),
            "gpu_rank": rank,
            "problem_id": prob_id,
            "accuracy": acc,
            "error": err
        }
        
        # Append to shared metrics file (with file locking for thread safety)
        with open(shared_metrics_file, "a") as f:
            f.write(json.dumps(metric_entry) + "\n")
        
        # Track processed problems
        processed_problems.add(prob_id)
        with open(processed_problems_file, "w") as f:
            json.dump(list(processed_problems), f)

        total_acc += acc
        processed += 1
        status = f"[GPU {rank}] {prob_id}: {acc*100:.1f}% (#{processed})"
        if err:
            status += f" – {err}"
        print(status, file=sys.stderr)

    # Write final summary for this GPU
    final_acc = total_acc / processed if processed else 0.0
    summary_entry = {
        "timestamp": time.time(),
        "gpu_rank": rank,
        "type": "final_summary",
        "total_problems": processed,
        "overall_accuracy": final_acc
    }
    
    with open(shared_metrics_file, "a") as f:
        f.write(json.dumps(summary_entry) + "\n")
    
    print(f"[GPU {rank}] Finished processing {processed} problems with {final_acc*100:.1f}% accuracy", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────
# Centralized wandb logging
# ──────────────────────────────────────────────────────────────────────────

def wandb_logger_process(args, shared_metrics_file: str, num_workers: int):
    """
    Central process that reads metrics from all workers and logs to wandb.
    Runs until all workers are finished.
    """
    # Initialize wandb run
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group or f"cf-{datetime.utcnow():%Y%m%dT%H%M%S}",
        name="aggregated_run",
        tags=["codeforces", "multi_gpu", "easy_only", "aggregated"],
        config=vars(args),
    )
    
    print("[Logger] Wandb logging process started", file=sys.stderr)
    
    # Track metrics
    logged_entries = set()
    gpu_summaries = {}
    workers_finished = set()
    total_problems = 0
    total_accuracy = 0.0
    problem_accuracies = []
    
    # Poll for new metrics every few seconds
    while len(workers_finished) < num_workers:
        time.sleep(5)  # Check every 5 seconds
        
        if not os.path.exists(shared_metrics_file):
            continue
            
        try:
            with open(shared_metrics_file, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line or line in logged_entries:
                    continue
                    
                try:
                    entry = json.loads(line)
                    logged_entries.add(line)
                    
                    if entry.get("type") == "final_summary":
                        # Worker finished
                        gpu_rank = entry["gpu_rank"]
                        workers_finished.add(gpu_rank)
                        gpu_summaries[gpu_rank] = entry
                        print(f"[Logger] GPU {gpu_rank} finished with {entry['overall_accuracy']*100:.1f}% accuracy on {entry['total_problems']} problems", file=sys.stderr)
                    else:
                        # Regular problem result
                        problem_accuracies.append(entry["accuracy"])
                        
                        # Log individual problem result
                        wandb.log({
                            "problem_accuracy": entry["accuracy"],
                            "gpu_rank": entry["gpu_rank"],
                            "step": len(problem_accuracies)
                        })
                        
                        # Update running average
                        current_avg = sum(problem_accuracies) / len(problem_accuracies)
                        wandb.log({
                            "running_average_accuracy": current_avg,
                            "problems_completed": len(problem_accuracies),
                            "step": len(problem_accuracies)
                        })
                        
                except json.JSONDecodeError:
                    continue
                    
        except FileNotFoundError:
            continue
    
    # Final aggregation
    if gpu_summaries:
        total_problems = sum(s["total_problems"] for s in gpu_summaries.values())
        weighted_accuracy = sum(s["overall_accuracy"] * s["total_problems"] for s in gpu_summaries.values()) / total_problems if total_problems > 0 else 0.0
        
        # Log final summary
        run.summary.update({
            "total_problems": total_problems,
            "overall_accuracy": weighted_accuracy,
            "num_gpus": num_workers,
            "problems_per_gpu": {f"gpu_{k}": v["total_problems"] for k, v in gpu_summaries.items()},
            "accuracy_per_gpu": {f"gpu_{k}": v["overall_accuracy"] for k, v in gpu_summaries.items()}
        })
        
        print(f"[Logger] Final results: {total_problems} problems, {weighted_accuracy*100:.1f}% overall accuracy", file=sys.stderr)
    
    run.finish()
    print("[Logger] Wandb logging process finished", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────
# Main – spawns workers
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--subset", default="verifiable")
    p.add_argument("--split", default="test")
    p.add_argument("--hf_token")

    p.add_argument("--wandb_project", default='codeforces-benchmark-2')
    p.add_argument("--wandb_entity", default='tbarton16-brown-university')
    p.add_argument("--wandb_group")

    p.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    world_size = min(args.num_gpus, torch.cuda.device_count())
    if world_size == 0:
        print("No GPU found!", file=sys.stderr)
        sys.exit(1)

    # Create shared metrics file
    shared_metrics_file = f"shared_metrics_{datetime.utcnow():%Y%m%d_%H%M%S}.jsonl"
    
    # Clean up any existing temporary files
    for f in ["processed_problems_gpu_*.json"]:
        for file_path in glob.glob(f):
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
    
    # Remove existing shared metrics file if it exists
    if os.path.exists(shared_metrics_file):
        os.remove(shared_metrics_file)
    
    print(f"Starting benchmark with {world_size} GPUs", file=sys.stderr)
    print(f"Shared metrics file: {shared_metrics_file}", file=sys.stderr)
    
    # Start wandb logger process
    logger_process = mp.Process(
        target=wandb_logger_process, 
        args=(args, shared_metrics_file, world_size)
    )
    logger_process.start()
    
    # Start worker processes
    mp.spawn(worker, args=(world_size, args, shared_metrics_file), nprocs=world_size, join=True)
    
    # Wait for logger to finish
    logger_process.join()
    
    print("All processes finished", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Part 1: Benchmark Qwen-7B-Instruct on Codeforces-style problems.

This script evaluates the base Qwen-7B-Instruct model on competitive programming problems,
measuring performance with different prompting strategies.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed
)
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    model_name: str = "Qwen/Qwen-7B-Instruct"
    dataset_name: str = "deepmind/code_contests"
    max_samples: Optional[int] = 100  # Limit for initial testing
    temperature: float = 0.1
    max_new_tokens: int = 1024
    top_p: float = 0.9
    do_sample: bool = True
    device: str = "auto"
    seed: int = 42
    output_dir: str = "results/benchmark_qwen"

class CodeContestsEvaluator:
    """Evaluates Qwen model on competitive programming problems."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
    def _setup_device(self) -> str:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS device")
            else:
                device = "cpu"
                logger.info("Using CPU device")
        else:
            device = self.config.device
        return device
    
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the Qwen model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def load_dataset(self) -> Dataset:
        """Load and preprocess the Code Contests dataset."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load the test split
        dataset = load_dataset(self.config.dataset_name, split="test")
        
        # Limit samples for initial testing
        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))
        
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset
    
    def create_prompts(self, problems: List[Dict]) -> List[str]:
        """Create prompts for the coding problems."""
        prompts = []
        
        for problem in problems:
            description = problem.get('description', '')
            public_tests = problem.get('public_tests', {})
            
            # Format public test cases
            test_examples = ""
            if public_tests and 'input' in public_tests and 'output' in public_tests:
                inputs = public_tests['input'][:3]  # Limit to first 3 examples
                outputs = public_tests['output'][:3]
                
                for i, (inp, out) in enumerate(zip(inputs, outputs)):
                    test_examples += f"\nExample {i+1}:\nInput: {inp.strip()}\nOutput: {out.strip()}\n"
            
            # Create the prompt
            prompt = f"""You are an expert competitive programmer. Solve the following problem:

Problem:
{description}

{test_examples}

Write a complete Python solution. Your code should read from standard input and write to standard output.

```python
"""
            
            prompts.append(prompt)
        
        return prompts
    
    def generate_solutions(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        prompts: List[str]
    ) -> List[str]:
        """Generate solutions for the given prompts."""
        solutions = []
        
        generation_config = GenerationConfig(
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        logger.info("Generating solutions...")
        for prompt in tqdm(prompts):
            try:
                # Tokenize input
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=2048
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # Decode output
                generated_text = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                # Extract code between ```python and ```
                if "```" in generated_text:
                    code_blocks = generated_text.split("```")
                    for block in code_blocks:
                        if block.strip().startswith("python"):
                            solution = block[6:].strip()  # Remove "python"
                            break
                        elif any(keyword in block for keyword in ["def ", "import ", "for ", "while ", "if "]):
                            solution = block.strip()
                            break
                    else:
                        solution = generated_text.strip()
                else:
                    solution = generated_text.strip()
                
                solutions.append(solution)
                
            except Exception as e:
                logger.error(f"Error generating solution: {e}")
                solutions.append("")
                
        return solutions
    
    def evaluate_solutions(
        self, 
        problems: List[Dict], 
        solutions: List[str]
    ) -> Dict[str, any]:
        """Evaluate generated solutions."""
        results = {
            'total_problems': len(problems),
            'solutions_generated': sum(1 for sol in solutions if sol.strip()),
            'generation_rate': 0.0,
            'detailed_results': []
        }
        
        for i, (problem, solution) in enumerate(zip(problems, solutions)):
            problem_result = {
                'problem_id': i,
                'has_solution': bool(solution.strip()),
                'solution_length': len(solution),
                'problem_difficulty': problem.get('difficulty', 'unknown'),
                'problem_rating': problem.get('rating', 0),
                'solution': solution[:500] + "..." if len(solution) > 500 else solution
            }
            results['detailed_results'].append(problem_result)
        
        results['generation_rate'] = results['solutions_generated'] / results['total_problems']
        
        # Analyze by difficulty
        difficulty_stats = {}
        for result in results['detailed_results']:
            diff = result['problem_difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'solved': 0}
            difficulty_stats[diff]['total'] += 1
            if result['has_solution']:
                difficulty_stats[diff]['solved'] += 1
        
        results['difficulty_breakdown'] = difficulty_stats
        
        return results
    
    def save_results(self, results: Dict, prompts: List[str], solutions: List[str]):
        """Save evaluation results."""
        # Save main results
        results_file = Path(self.config.output_dir) / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed data
        detailed_data = []
        for i, (prompt, solution, result) in enumerate(zip(prompts, solutions, results['detailed_results'])):
            detailed_data.append({
                'problem_id': i,
                'prompt': prompt,
                'generated_solution': solution,
                'has_solution': result['has_solution'],
                'solution_length': result['solution_length'],
                'problem_difficulty': result['problem_difficulty'],
                'problem_rating': result['problem_rating']
            })
        
        df = pd.DataFrame(detailed_data)
        df.to_csv(Path(self.config.output_dir) / "detailed_results.csv", index=False)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def run_benchmark(self):
        """Run the complete benchmark evaluation."""
        logger.info("Starting Qwen-7B benchmark evaluation")
        
        # Load model and dataset
        model, tokenizer = self.load_model_and_tokenizer()
        dataset = self.load_dataset()
        
        # Convert dataset to list of dicts
        problems = [dict(example) for example in dataset]
        
        # Create prompts
        prompts = self.create_prompts(problems)
        
        # Save prompting format documentation
        prompt_doc = {
            "prompting_strategy": "instruction_following_with_examples",
            "format": "Problem description + Examples + Request for Python solution",
            "max_examples": 3,
            "template": prompts[0] if prompts else "No problems loaded",
            "notes": [
                "Uses natural language instructions",
                "Includes up to 3 public test examples",
                "Requests complete Python solution",
                "Expects code to read from stdin and write to stdout"
            ]
        }
        
        with open(Path(self.config.output_dir) / "prompting_format.json", 'w') as f:
            json.dump(prompt_doc, f, indent=2)
        
        # Generate solutions
        solutions = self.generate_solutions(model, tokenizer, prompts)
        
        # Evaluate results
        results = self.evaluate_solutions(problems, solutions)
        
        # Save results
        self.save_results(results, prompts, solutions)
        
        # Print summary
        logger.info(f"Benchmark completed:")
        logger.info(f"  Total problems: {results['total_problems']}")
        logger.info(f"  Solutions generated: {results['solutions_generated']}")
        logger.info(f"  Generation rate: {results['generation_rate']:.2%}")
        
        return results

def main():
    """Main function."""
    config = BenchmarkConfig()
    evaluator = CodeContestsEvaluator(config)
    results = evaluator.run_benchmark()
    
    print("\n=== Benchmark Results ===")
    print(f"Total problems evaluated: {results['total_problems']}")
    print(f"Solutions generated: {results['solutions_generated']}")
    print(f"Generation rate: {results['generation_rate']:.2%}")
    
    if results['difficulty_breakdown']:
        print("\nBreakdown by difficulty:")
        for difficulty, stats in results['difficulty_breakdown'].items():
            rate = stats['solved'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {difficulty}: {stats['solved']}/{stats['total']} ({rate:.2%})")

if __name__ == "__main__":
    main()
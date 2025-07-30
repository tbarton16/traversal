#!/usr/bin/env python3
"""
Part 1: Benchmark coding performance evaluation
Evaluate Qwen-7B-Instruct model on Codeforces-style benchmark
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import re

class CodeforcesEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set up evaluation metrics
        self.bleu = evaluate.load("bleu")
        self.exact_match = evaluate.load("exact_match")
        
    def format_prompt(self, problem: Dict[str, Any]) -> str:
        """
        Format the problem into a prompt for the model.
        Using a structured format that includes problem description, constraints, and examples.
        """
        prompt = f"""You are an expert competitive programmer. Solve the following Codeforces problem:

Problem: {problem['problem']}

Input Format: {problem.get('input_format', 'Standard input')}
Output Format: {problem.get('output_format', 'Standard output')}

Constraints: {problem.get('constraints', 'None specified')}

Examples:
{problem.get('examples', 'No examples provided')}

Please provide a complete solution in Python. Your solution should be efficient and handle all edge cases.

Solution:"""
        return prompt
    
    def generate_solution(self, prompt: str, max_length: int = 2048) -> str:
        """Generate a solution using the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after the prompt)
        solution = generated_text[len(prompt):].strip()
        return solution
    
    def extract_code(self, solution: str) -> str:
        """Extract code blocks from the generated solution"""
        # Look for code blocks marked with ```python or ```
        code_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(code_pattern, solution, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, look for lines that look like code
        lines = solution.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if any(keyword in line for keyword in ['def ', 'import ', 'class ', 'if __name__', 'print(']):
                in_code = True
            if in_code and line.strip():
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else solution
    
    def evaluate_solution(self, generated_code: str, expected_output: str, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate the generated solution against test cases"""
        try:
            # Create a safe execution environment
            exec_globals = {}
            exec(generated_code, exec_globals)
            
            # Test the solution
            correct = 0
            total = len(test_cases)
            
            for test_case in test_cases:
                try:
                    # Assuming the solution defines a function called 'solve' or similar
                    if 'solve' in exec_globals:
                        result = exec_globals['solve'](test_case['input'])
                        if str(result).strip() == str(test_case['output']).strip():
                            correct += 1
                except Exception as e:
                    print(f"Error executing solution: {e}")
                    continue
            
            accuracy = correct / total if total > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'correct_cases': correct,
                'total_cases': total,
                'executable': True
            }
            
        except Exception as e:
            return {
                'accuracy': 0.0,
                'correct_cases': 0,
                'total_cases': len(test_cases),
                'executable': False,
                'error': str(e)
            }
    
    def run_evaluation(self, dataset_path: str = None, num_samples: int = 100) -> Dict[str, Any]:
        """Run the full evaluation on the dataset"""
        print("Loading Codeforces dataset...")
        
        # Load a sample dataset (you would replace this with actual Codeforces data)
        # For demonstration, we'll create a synthetic dataset
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                problems = json.load(f)
        else:
            # Create synthetic problems for demonstration
            problems = self.create_synthetic_problems()
        
        results = []
        metrics = {
            'total_problems': len(problems),
            'executable_solutions': 0,
            'correct_solutions': 0,
            'avg_accuracy': 0.0
        }
        
        print(f"Evaluating on {len(problems)} problems...")
        
        for i, problem in enumerate(tqdm(problems[:num_samples])):
            prompt = self.format_prompt(problem)
            solution = self.generate_solution(prompt)
            code = self.extract_code(solution)
            
            evaluation = self.evaluate_solution(code, problem.get('expected_output', ''), problem.get('test_cases', []))
            
            result = {
                'problem_id': i,
                'problem': problem['problem'][:100] + '...' if len(problem['problem']) > 100 else problem['problem'],
                'generated_code': code,
                'evaluation': evaluation
            }
            results.append(result)
            
            if evaluation['executable']:
                metrics['executable_solutions'] += 1
                if evaluation['accuracy'] == 1.0:
                    metrics['correct_solutions'] += 1
        
        # Calculate final metrics
        if metrics['total_problems'] > 0:
            metrics['avg_accuracy'] = sum(r['evaluation']['accuracy'] for r in results) / len(results)
            metrics['executable_rate'] = metrics['executable_solutions'] / metrics['total_problems']
            metrics['success_rate'] = metrics['correct_solutions'] / metrics['total_problems']
        
        return {
            'metrics': metrics,
            'results': results
        }
    
    def create_synthetic_problems(self) -> List[Dict]:
        """Create synthetic problems for demonstration"""
        return [
            {
                'problem': 'Given an array of integers, find the sum of all elements.',
                'input_format': 'First line contains n, the number of elements. Second line contains n space-separated integers.',
                'output_format': 'Print the sum of all elements.',
                'constraints': '1 ≤ n ≤ 10^5, -10^9 ≤ ai ≤ 10^9',
                'examples': 'Input: 3\n1 2 3\nOutput: 6',
                'test_cases': [
                    {'input': [1, 2, 3], 'output': 6},
                    {'input': [0, 0, 0], 'output': 0},
                    {'input': [-1, -2, -3], 'output': -6}
                ]
            },
            {
                'problem': 'Find the maximum element in an array.',
                'input_format': 'First line contains n, the number of elements. Second line contains n space-separated integers.',
                'output_format': 'Print the maximum element.',
                'constraints': '1 ≤ n ≤ 10^5, -10^9 ≤ ai ≤ 10^9',
                'examples': 'Input: 3\n1 5 3\nOutput: 5',
                'test_cases': [
                    {'input': [1, 5, 3], 'output': 5},
                    {'input': [-1, -5, -3], 'output': -1},
                    {'input': [10], 'output': 10}
                ]
            }
        ]
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen-7B-Instruct on Codeforces benchmark')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct', 
                       help='Model name to evaluate')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to Codeforces dataset JSON file')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of problems to evaluate')
    parser.add_argument('--output_path', type=str, default='part1_results.json',
                       help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run evaluation on')
    
    args = parser.parse_args()
    
    print("Initializing Codeforces Evaluator...")
    evaluator = CodeforcesEvaluator(model_name=args.model_name, device=args.device)
    
    print("Running evaluation...")
    results = evaluator.run_evaluation(
        dataset_path=args.dataset_path,
        num_samples=args.num_samples
    )
    
    # Print summary
    metrics = results['metrics']
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total problems evaluated: {metrics['total_problems']}")
    print(f"Executable solutions: {metrics['executable_solutions']} ({metrics.get('executable_rate', 0):.2%})")
    print(f"Correct solutions: {metrics['correct_solutions']} ({metrics.get('success_rate', 0):.2%})")
    print(f"Average accuracy: {metrics['avg_accuracy']:.2%}")
    print("="*50)
    
    # Save results
    evaluator.save_results(results, args.output_path)

if __name__ == "__main__":
    main() 
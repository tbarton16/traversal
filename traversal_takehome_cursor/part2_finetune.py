#!/usr/bin/env python3
"""
Part 2: Fine-tune Qwen-7B-Instruct on reasoning traces dataset
Fine-tune the model on 350K reasoning traces and evaluate performance
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
import evaluate
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from datetime import datetime

class ReasoningTracesTrainer:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_length: int = 2048,
                 device: str = "cuda"):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set up evaluation metrics
        self.bleu = evaluate.load("bleu")
        self.exact_match = evaluate.load("exact_match")
        
    def load_reasoning_traces(self, dataset_path: str = None) -> Dataset:
        """Load and preprocess the reasoning traces dataset"""
        print("Loading reasoning traces dataset...")
        
        if dataset_path and os.path.exists(dataset_path):
            # Load from local file
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        else:
            # Create synthetic reasoning traces for demonstration
            data = self.create_synthetic_reasoning_traces()
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        print(f"Loaded {len(dataset)} reasoning traces")
        return dataset
    
    def create_synthetic_reasoning_traces(self) -> List[Dict]:
        """Create synthetic reasoning traces for demonstration"""
        traces = []
        
        # Math reasoning examples
        math_problems = [
            {
                "question": "What is 15 + 27?",
                "reasoning": "Let me break this down step by step:\n1. First, I'll add the ones: 5 + 7 = 12\n2. I need to carry the 1 to the tens place\n3. Now add the tens: 1 + 1 + 2 = 4\n4. So 15 + 27 = 42",
                "answer": "42"
            },
            {
                "question": "Solve for x: 3x + 5 = 20",
                "reasoning": "Let me solve this equation step by step:\n1. First, subtract 5 from both sides: 3x = 15\n2. Then divide both sides by 3: x = 5\n3. Let me verify: 3(5) + 5 = 15 + 5 = 20 ✓",
                "answer": "x = 5"
            }
        ]
        
        # Coding reasoning examples
        coding_problems = [
            {
                "question": "Write a function to find the factorial of a number",
                "reasoning": "Let me think about this step by step:\n1. Factorial of n is n * (n-1) * (n-2) * ... * 1\n2. I need to handle base cases: factorial(0) = 1, factorial(1) = 1\n3. For n > 1, I can use recursion or iteration\n4. Let me write an iterative solution for efficiency",
                "answer": "def factorial(n):\n    if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"
            },
            {
                "question": "Find the maximum element in an array",
                "reasoning": "Let me think about this:\n1. I need to iterate through the array\n2. Keep track of the maximum value seen so far\n3. Compare each element with the current maximum\n4. Update maximum if current element is larger\n5. Return the final maximum",
                "answer": "def find_max(arr):\n    if not arr:\n        return None\n    max_val = arr[0]\n    for num in arr[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val"
            }
        ]
        
        # Combine and format
        for problem in math_problems + coding_problems:
            trace = {
                "instruction": problem["question"],
                "input": "",
                "output": f"{problem['reasoning']}\n\nFinal Answer: {problem['answer']}"
            }
            traces.append(trace)
        
        # Generate more synthetic data
        for i in range(1000):  # Generate 1000 more synthetic traces
            trace = self.generate_synthetic_trace(i)
            traces.append(trace)
        
        return traces
    
    def generate_synthetic_trace(self, index: int) -> Dict:
        """Generate a synthetic reasoning trace"""
        problems = [
            {
                "question": f"What is {index + 1} * {index + 2}?",
                "reasoning": f"Let me calculate this step by step:\n1. {index + 1} * {index + 2} = {(index + 1) * (index + 2)}\n2. This equals {(index + 1) * (index + 2)}",
                "answer": str((index + 1) * (index + 2))
            },
            {
                "question": f"Find the sum of numbers from 1 to {index + 5}",
                "reasoning": f"Let me solve this:\n1. Sum of numbers from 1 to n is n*(n+1)/2\n2. Here n = {index + 5}\n3. So sum = {index + 5} * {index + 6} / 2 = {(index + 5) * (index + 6) // 2}",
                "answer": str((index + 5) * (index + 6) // 2)
            }
        ]
        
        problem = problems[index % len(problems)]
        return {
            "instruction": problem["question"],
            "input": "",
            "output": f"{problem['reasoning']}\n\nFinal Answer: {problem['answer']}"
        }
    
    def format_training_data(self, dataset: Dataset) -> Dataset:
        """Format the dataset for training"""
        def format_example(example):
            # Format as instruction-following data
            prompt = f"Question: {example['instruction']}\n\nLet me think about this step by step:\n"
            response = example['output']
            
            # Combine prompt and response
            full_text = prompt + response
            
            return {
                "text": full_text,
                "prompt": prompt,
                "response": response
            }
        
        formatted_dataset = dataset.map(format_example)
        return formatted_dataset
    
    def tokenize_function(self, examples):
        """Tokenize the examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, dataset: Dataset, test_size: float = 0.1) -> tuple:
        """Prepare the dataset for training and evaluation"""
        # Split into train and validation
        train_dataset, eval_dataset = train_test_split(
            dataset, 
            test_size=test_size, 
            random_state=42
        )
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_dataset)
        eval_dataset = Dataset.from_list(eval_dataset)
        
        # Tokenize datasets
        train_tokenized = train_dataset.map(
            self.tokenize_function, 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        eval_tokenized = eval_dataset.map(
            self.tokenize_function, 
            batched=True, 
            remove_columns=eval_dataset.column_names
        )
        
        return train_tokenized, eval_tokenized
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # Calculate perplexity
        loss = F.cross_entropy(
            torch.tensor(predictions), 
            torch.tensor(labels), 
            reduction='mean'
        )
        perplexity = torch.exp(loss).item()
        
        return {
            "perplexity": perplexity,
            "eval_loss": loss.item()
        }
    
    def train(self, 
              dataset_path: str = None,
              output_dir: str = "./finetuned_model",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-5,
              warmup_steps: int = 100,
              save_steps: int = 500,
              eval_steps: int = 500,
              gradient_accumulation_steps: int = 4,
              max_grad_norm: float = 1.0,
              weight_decay: float = 0.01,
              logging_steps: int = 10):
        """Train the model on reasoning traces"""
        
        # Load and prepare dataset
        dataset = self.load_reasoning_traces(dataset_path)
        formatted_dataset = self.format_training_data(dataset)
        train_dataset, eval_dataset = self.prepare_dataset(formatted_dataset)
        
        print(f"Training on {len(train_dataset)} examples")
        print(f"Evaluating on {len(eval_dataset)} examples")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if os.getenv("WANDB_PROJECT") else None,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            push_to_hub=False,
            ddp_find_unused_parameters=False,
        )
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Set up trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        
        # Evaluate final performance
        final_metrics = trainer.evaluate()
        print("Final evaluation metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")
        
        return trainer, final_metrics
    
    def evaluate_on_codeforces(self, 
                              model_path: str,
                              codeforces_data: str = None,
                              num_samples: int = 50) -> Dict[str, Any]:
        """Evaluate the fine-tuned model on Codeforces benchmark"""
        print("Loading fine-tuned model for evaluation...")
        
        # Load the fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create evaluator (reuse from Part 1)
        from part1_benchmark import CodeforcesEvaluator
        evaluator = CodeforcesEvaluator(model_name=model_path, device=self.device)
        
        # Run evaluation
        results = evaluator.run_evaluation(
            dataset_path=codeforces_data,
            num_samples=num_samples
        )
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen-7B-Instruct on reasoning traces')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Base model name')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to reasoning traces dataset')
    parser.add_argument('--output_dir', type=str, default='./finetuned_model',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--eval_codeforces', action='store_true',
                       help='Evaluate on Codeforces after training')
    parser.add_argument('--codeforces_data', type=str, default=None,
                       help='Path to Codeforces evaluation data')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ReasoningTracesTrainer(
        model_name=args.model_name,
        max_length=args.max_length,
        device=args.device
    )
    
    # Train the model
    print("Starting fine-tuning process...")
    trainer_instance, final_metrics = trainer.train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save training metrics
    with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Evaluate on Codeforces if requested
    if args.eval_codeforces:
        print("Evaluating fine-tuned model on Codeforces benchmark...")
        codeforces_results = trainer.evaluate_on_codeforces(
            model_path=args.output_dir,
            codeforces_data=args.codeforces_data,
            num_samples=50
        )
        
        # Save Codeforces evaluation results
        with open(os.path.join(args.output_dir, 'codeforces_eval_results.json'), 'w') as f:
            json.dump(codeforces_results, f, indent=2)
        
        print("Codeforces evaluation completed and saved.")
    
    print("Fine-tuning process completed!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Part 3: Re-evaluate benchmark performance after fine-tuning.

This script evaluates the fine-tuned Qwen model and compares it with the baseline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd

from benchmark_qwen import CodeContestsEvaluator, BenchmarkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    base_model_name: str = "Qwen/Qwen-7B-Instruct"
    finetuned_model_path: str = "models/qwen-7b-finetuned"
    baseline_results_path: str = "results/benchmark_qwen/benchmark_results.json"
    output_dir: str = "results/comparison"
    max_samples: int = 100  # Use same samples as baseline for fair comparison
    seed: int = 42

class FineTunedEvaluator(CodeContestsEvaluator):
    """Evaluator for fine-tuned models."""
    
    def __init__(self, config: BenchmarkConfig, finetuned_model_path: str):
        super().__init__(config)
        self.finetuned_model_path = finetuned_model_path
    
    def load_model_and_tokenizer(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading fine-tuned model from: {self.finetuned_model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.finetuned_model_path,
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        
        # Load fine-tuned weights
        model = PeftModel.from_pretrained(base_model, self.finetuned_model_path)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

class ModelComparator:
    """Compares baseline and fine-tuned model performance."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_baseline_results(self) -> Dict:
        """Load baseline evaluation results."""
        logger.info(f"Loading baseline results from: {self.config.baseline_results_path}")
        
        if not Path(self.config.baseline_results_path).exists():
            logger.error("Baseline results not found. Please run benchmark_qwen.py first.")
            return {}
        
        with open(self.config.baseline_results_path, 'r') as f:
            return json.load(f)
    
    def evaluate_finetuned(self) -> Dict:
        """Evaluate the fine-tuned model."""
        logger.info("Evaluating fine-tuned model...")
        
        # Create benchmark config for fine-tuned model
        benchmark_config = BenchmarkConfig(
            max_samples=self.config.max_samples,
            seed=self.config.seed,
            output_dir=f"{self.config.output_dir}/finetuned_results"
        )
        
        # Evaluate fine-tuned model
        evaluator = FineTunedEvaluator(benchmark_config, self.config.finetuned_model_path)
        results = evaluator.run_benchmark()
        
        return results
    
    def compare_results(self, baseline_results: Dict, finetuned_results: Dict) -> Dict:
        """Compare baseline and fine-tuned results."""
        logger.info("Comparing results...")
        
        comparison = {
            "baseline": {
                "total_problems": baseline_results.get("total_problems", 0),
                "solutions_generated": baseline_results.get("solutions_generated", 0),
                "generation_rate": baseline_results.get("generation_rate", 0.0),
                "difficulty_breakdown": baseline_results.get("difficulty_breakdown", {})
            },
            "finetuned": {
                "total_problems": finetuned_results.get("total_problems", 0),
                "solutions_generated": finetuned_results.get("solutions_generated", 0),
                "generation_rate": finetuned_results.get("generation_rate", 0.0),
                "difficulty_breakdown": finetuned_results.get("difficulty_breakdown", {})
            },
            "improvements": {},
            "analysis": {}
        }
        
        # Calculate improvements
        baseline_rate = comparison["baseline"]["generation_rate"]
        finetuned_rate = comparison["finetuned"]["generation_rate"]
        
        comparison["improvements"]["generation_rate_change"] = finetuned_rate - baseline_rate
        comparison["improvements"]["relative_improvement"] = (
            (finetuned_rate - baseline_rate) / baseline_rate if baseline_rate > 0 else 0
        )
        
        # Analyze by difficulty
        baseline_diff = comparison["baseline"]["difficulty_breakdown"]
        finetuned_diff = comparison["finetuned"]["difficulty_breakdown"]
        
        difficulty_comparison = {}
        for difficulty in set(list(baseline_diff.keys()) + list(finetuned_diff.keys())):
            baseline_stats = baseline_diff.get(difficulty, {"total": 0, "solved": 0})
            finetuned_stats = finetuned_diff.get(difficulty, {"total": 0, "solved": 0})
            
            baseline_rate = baseline_stats["solved"] / baseline_stats["total"] if baseline_stats["total"] > 0 else 0
            finetuned_rate = finetuned_stats["solved"] / finetuned_stats["total"] if finetuned_stats["total"] > 0 else 0
            
            difficulty_comparison[difficulty] = {
                "baseline_rate": baseline_rate,
                "finetuned_rate": finetuned_rate,
                "improvement": finetuned_rate - baseline_rate
            }
        
        comparison["improvements"]["difficulty_breakdown"] = difficulty_comparison
        
        # Generate analysis
        analysis = []
        
        if comparison["improvements"]["generation_rate_change"] > 0:
            analysis.append("Fine-tuning improved overall solution generation rate")
        elif comparison["improvements"]["generation_rate_change"] < 0:
            analysis.append("Fine-tuning decreased overall solution generation rate")
        else:
            analysis.append("Fine-tuning had no effect on overall solution generation rate")
        
        # Find best and worst performing difficulties
        improvements_by_diff = {k: v["improvement"] for k, v in difficulty_comparison.items()}
        if improvements_by_diff:
            best_diff = max(improvements_by_diff.keys(), key=lambda k: improvements_by_diff[k])
            worst_diff = min(improvements_by_diff.keys(), key=lambda k: improvements_by_diff[k])
            
            analysis.append(f"Best improvement on {best_diff} problems: {improvements_by_diff[best_diff]:.2%}")
            analysis.append(f"Worst performance on {worst_diff} problems: {improvements_by_diff[worst_diff]:.2%}")
        
        comparison["analysis"]["summary"] = analysis
        
        return comparison
    
    def create_visualizations(self, comparison: Dict):
        """Create comparison visualizations and save as data."""
        # Prepare data for potential visualization
        viz_data = {
            "overall_comparison": [
                {"model": "Baseline", "generation_rate": comparison["baseline"]["generation_rate"]},
                {"model": "Fine-tuned", "generation_rate": comparison["finetuned"]["generation_rate"]}
            ],
            "difficulty_comparison": []
        }
        
        for difficulty, stats in comparison["improvements"]["difficulty_breakdown"].items():
            viz_data["difficulty_comparison"].extend([
                {"difficulty": difficulty, "model": "Baseline", "rate": stats["baseline_rate"]},
                {"difficulty": difficulty, "model": "Fine-tuned", "rate": stats["finetuned_rate"]}
            ])
        
        # Save visualization data
        with open(Path(self.config.output_dir) / "visualization_data.json", 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        # Save as DataFrame for easy analysis
        df_overall = pd.DataFrame(viz_data["overall_comparison"])
        df_difficulty = pd.DataFrame(viz_data["difficulty_comparison"])
        
        df_overall.to_csv(Path(self.config.output_dir) / "overall_comparison.csv", index=False)
        df_difficulty.to_csv(Path(self.config.output_dir) / "difficulty_comparison.csv", index=False)
    
    def run_comparison(self):
        """Run the complete comparison analysis."""
        logger.info("Starting model comparison...")
        
        # Load baseline results
        baseline_results = self.load_baseline_results()
        if not baseline_results:
            logger.error("Cannot proceed without baseline results")
            return None
        
        # Evaluate fine-tuned model
        finetuned_results = self.evaluate_finetuned()
        
        # Compare results
        comparison = self.compare_results(baseline_results, finetuned_results)
        
        # Create visualizations
        self.create_visualizations(comparison)
        
        # Save comparison results
        with open(Path(self.config.output_dir) / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Print summary
        self.print_summary(comparison)
        
        logger.info(f"Comparison results saved to {self.config.output_dir}")
        return comparison
    
    def print_summary(self, comparison: Dict):
        """Print comparison summary."""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        baseline = comparison["baseline"]
        finetuned = comparison["finetuned"]
        improvements = comparison["improvements"]
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Baseline model:")
        print(f"    Solutions generated: {baseline['solutions_generated']}/{baseline['total_problems']}")
        print(f"    Generation rate: {baseline['generation_rate']:.2%}")
        
        print(f"\n  Fine-tuned model:")
        print(f"    Solutions generated: {finetuned['solutions_generated']}/{finetuned['total_problems']}")
        print(f"    Generation rate: {finetuned['generation_rate']:.2%}")
        
        print(f"\n  IMPROVEMENT:")
        print(f"    Absolute change: {improvements['generation_rate_change']:+.2%}")
        print(f"    Relative improvement: {improvements['relative_improvement']:+.2%}")
        
        print(f"\nPERFORMANCE BY DIFFICULTY:")
        for difficulty, stats in improvements["difficulty_breakdown"].items():
            print(f"  {difficulty}:")
            print(f"    Baseline: {stats['baseline_rate']:.2%}")
            print(f"    Fine-tuned: {stats['finetuned_rate']:.2%}")
            print(f"    Change: {stats['improvement']:+.2%}")
        
        print(f"\nANALYSIS:")
        for point in comparison["analysis"]["summary"]:
            print(f"  • {point}")
        
        print("="*60)

def main():
    """Main function."""
    config = EvaluationConfig()
    
    # Check if fine-tuned model exists
    if not Path(config.finetuned_model_path).exists():
        logger.error(f"Fine-tuned model not found at {config.finetuned_model_path}")
        logger.error("Please run fine_tune_qwen.py first")
        return
    
    comparator = ModelComparator(config)
    comparison = comparator.run_comparison()
    
    if comparison:
        print("\nComparison completed successfully!")
        print(f"Results saved to: {config.output_dir}")

if __name__ == "__main__":
    main()
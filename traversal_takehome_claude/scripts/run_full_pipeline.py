#!/usr/bin/env python3
"""
Complete pipeline runner for the Qwen-7B fine-tuning project.

This script runs the entire pipeline from data analysis through evaluation.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Runs the complete ML pipeline."""
    
    def __init__(self, skip_steps: Optional[List[str]] = None):
        self.skip_steps = skip_steps or []
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a command and handle errors."""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✓ {description} completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout[-500:]}")  # Show last 500 chars
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description} failed")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"✗ {description} failed with exception: {e}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if all required files and dependencies exist."""
        logger.info("Checking prerequisites...")
        
        required_files = [
            self.scripts_dir / "data_analysis.py",
            self.scripts_dir / "benchmark_qwen.py",
            self.scripts_dir / "fine_tune_qwen.py",
            self.scripts_dir / "evaluate_finetuned.py",
            self.project_root / "requirements.txt"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.error("Missing required files:")
            for file_path in missing_files:
                logger.error(f"  - {file_path}")
            return False
        
        logger.info("✓ All required files found")
        return True
    
    def run_data_analysis(self) -> bool:
        """Run data analysis step."""
        if "data_analysis" in self.skip_steps:
            logger.info("Skipping data analysis (--skip-data-analysis)")
            return True
        
        return self.run_command(
            [sys.executable, str(self.scripts_dir / "data_analysis.py")],
            "Data analysis"
        )
    
    def run_baseline_benchmark(self) -> bool:
        """Run baseline benchmarking."""
        if "baseline_benchmark" in self.skip_steps:
            logger.info("Skipping baseline benchmark (--skip-baseline)")
            return True
        
        return self.run_command(
            [sys.executable, str(self.scripts_dir / "benchmark_qwen.py")],
            "Baseline benchmarking"
        )
    
    def run_fine_tuning(self) -> bool:
        """Run fine-tuning step."""
        if "fine_tuning" in self.skip_steps:
            logger.info("Skipping fine-tuning (--skip-training)")
            return True
        
        return self.run_command(
            [sys.executable, str(self.scripts_dir / "fine_tune_qwen.py")],
            "Fine-tuning"
        )
    
    def run_evaluation(self) -> bool:
        """Run evaluation of fine-tuned model."""
        if "evaluation" in self.skip_steps:
            logger.info("Skipping evaluation (--skip-evaluation)")
            return True
        
        # Check if fine-tuned model exists
        model_path = self.project_root / "models" / "qwen-7b-finetuned"
        if not model_path.exists():
            logger.error("Fine-tuned model not found. Cannot run evaluation.")
            logger.error("Either run fine-tuning first or use --skip-evaluation")
            return False
        
        return self.run_command(
            [sys.executable, str(self.scripts_dir / "evaluate_finetuned.py")],
            "Fine-tuned model evaluation"
        )
    
    def start_inference_server(self) -> bool:
        """Start inference server (optional)."""
        if "inference_server" in self.skip_steps:
            logger.info("Skipping inference server (--skip-server)")
            return True
        
        # Check if fine-tuned model exists
        model_path = self.project_root / "models" / "qwen-7b-finetuned"
        if not model_path.exists():
            logger.warning("Fine-tuned model not found. Cannot start inference server.")
            logger.warning("Run fine-tuning first or use --skip-server")
            return True  # Don't fail the pipeline for this
        
        logger.info("Starting inference server...")
        logger.info("Note: Server will run in the background. Check logs for status.")
        logger.info("API will be available at http://localhost:8000")
        
        try:
            # Start server in background
            subprocess.Popen(
                [sys.executable, str(self.scripts_dir / "inference_server.py")],
                cwd=self.project_root
            )
            logger.info("✓ Inference server started in background")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to start inference server: {e}")
            return False
    
    def generate_final_report(self) -> bool:
        """Generate a final summary report."""
        logger.info("Generating final report...")
        
        report_data = {
            "pipeline_status": "completed",
            "steps_completed": [],
            "results_locations": {},
            "recommendations": []
        }
        
        # Check what results exist
        results_dir = self.project_root / "results"
        
        if (results_dir / "data_analysis").exists():
            report_data["steps_completed"].append("data_analysis")
            report_data["results_locations"]["data_analysis"] = str(results_dir / "data_analysis")
        
        if (results_dir / "benchmark_qwen").exists():
            report_data["steps_completed"].append("baseline_benchmark")
            report_data["results_locations"]["baseline_benchmark"] = str(results_dir / "benchmark_qwen")
        
        if (self.project_root / "models" / "qwen-7b-finetuned").exists():
            report_data["steps_completed"].append("fine_tuning")
            report_data["results_locations"]["fine_tuned_model"] = str(self.project_root / "models" / "qwen-7b-finetuned")
        
        if (results_dir / "comparison").exists():
            report_data["steps_completed"].append("evaluation")
            report_data["results_locations"]["evaluation"] = str(results_dir / "comparison")
        
        # Add recommendations
        report_data["recommendations"] = [
            "Review data analysis results to understand dataset characteristics",
            "Check baseline benchmark results to establish performance baseline",
            "Examine training logs for fine-tuning convergence and stability",
            "Compare baseline vs fine-tuned performance in evaluation results",
            "Consider hyperparameter tuning based on results",
            "Deploy inference server for practical testing"
        ]
        
        # Save report
        report_path = self.project_root / "final_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create human-readable summary
        summary_lines = [
            "# Qwen-7B Fine-tuning Pipeline - Final Report",
            "",
            f"Pipeline completed with {len(report_data['steps_completed'])} steps.",
            "",
            "## Completed Steps:",
        ]
        
        for step in report_data["steps_completed"]:
            summary_lines.append(f"- {step.replace('_', ' ').title()}")
        
        summary_lines.extend([
            "",
            "## Results Locations:",
        ])
        
        for name, location in report_data["results_locations"].items():
            summary_lines.append(f"- {name.replace('_', ' ').title()}: `{location}`")
        
        summary_lines.extend([
            "",
            "## Next Steps:",
        ])
        
        for rec in report_data["recommendations"]:
            summary_lines.append(f"- {rec}")
        
        summary_lines.extend([
            "",
            "## Usage:",
            "",
            "1. **Review Results**: Check each results directory for detailed analysis",
            "2. **Model Usage**: Load the fine-tuned model from `models/qwen-7b-finetuned`",
            "3. **API Testing**: If inference server is running, test at http://localhost:8000",
            "4. **Further Training**: Adjust hyperparameters based on results and retrain",
            "",
            "For questions or issues, refer to the documentation in each results directory."
        ])
        
        summary_path = self.project_root / "PIPELINE_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_lines))
        
        logger.info(f"✓ Final report saved to {report_path}")
        logger.info(f"✓ Pipeline summary saved to {summary_path}")
        
        return True
    
    def run_pipeline(self, include_server: bool = False) -> bool:
        """Run the complete pipeline."""
        logger.info("="*60)
        logger.info("STARTING QWEN-7B FINE-TUNING PIPELINE")
        logger.info("="*60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Aborting pipeline.")
            return False
        
        steps = [
            ("Data Analysis", self.run_data_analysis),
            ("Baseline Benchmark", self.run_baseline_benchmark),
            ("Fine-tuning", self.run_fine_tuning),
            ("Evaluation", self.run_evaluation),
        ]
        
        if include_server:
            steps.append(("Inference Server", self.start_inference_server))
        
        failed_steps = []
        
        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")
            if not step_func():
                failed_steps.append(step_name)
                logger.error(f"Step '{step_name}' failed!")
                
                # Ask user if they want to continue
                try:
                    response = input(f"\nStep '{step_name}' failed. Continue with remaining steps? (y/n): ")
                    if response.lower() not in ['y', 'yes']:
                        logger.info("Pipeline aborted by user.")
                        return False
                except KeyboardInterrupt:
                    logger.info("\nPipeline interrupted by user.")
                    return False
        
        # Generate final report regardless of failures
        logger.info("\n--- Final Report ---")
        self.generate_final_report()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED")
        logger.info("="*60)
        
        if failed_steps:
            logger.warning(f"Pipeline completed with {len(failed_steps)} failed steps:")
            for step in failed_steps:
                logger.warning(f"  - {step}")
        else:
            logger.info("All steps completed successfully!")
        
        logger.info("Check final_report.json and PIPELINE_SUMMARY.md for details.")
        logger.info("="*60)
        
        return len(failed_steps) == 0

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Run the complete Qwen-7B fine-tuning pipeline")
    
    parser.add_argument("--skip-data-analysis", action="store_true",
                       help="Skip data analysis step")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline benchmarking")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip fine-tuning step")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--skip-server", action="store_true",
                       help="Skip starting inference server")
    parser.add_argument("--include-server", action="store_true",
                       help="Include inference server startup in pipeline")
    
    args = parser.parse_args()
    
    # Build skip list
    skip_steps = []
    if args.skip_data_analysis:
        skip_steps.append("data_analysis")
    if args.skip_baseline:
        skip_steps.append("baseline_benchmark")
    if args.skip_training:
        skip_steps.append("fine_tuning")
    if args.skip_evaluation:
        skip_steps.append("evaluation")
    if args.skip_server:
        skip_steps.append("inference_server")
    
    # Run pipeline
    runner = PipelineRunner(skip_steps=skip_steps)
    success = runner.run_pipeline(include_server=args.include_server)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
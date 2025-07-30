#!/usr/bin/env python3
"""
Part 2: Fine-tune Qwen-7B-Instruct on 350K reasoning traces.

This script implements supervised fine-tuning (SFT) of the Qwen model on
a dataset of multi-domain reasoning traces.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    # Model settings
    model_name: str = "Qwen/Qwen-7B-Instruct"
    dataset_name: str = "microsoft/orca-math-word-problems-200k"  # Placeholder - will use reasoning traces
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Data settings
    max_seq_length: int = 2048
    train_split_ratio: float = 0.95
    max_samples: Optional[int] = None  # Use all available data
    
    # Output settings
    output_dir: str = "models/qwen-7b-finetuned"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Other settings
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False

class ReasoningTracesDataProcessor:
    """Processes reasoning traces dataset for fine-tuning."""
    
    def __init__(self, config: FineTuningConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
    def load_dataset(self) -> Dataset:
        """Load and process the reasoning traces dataset."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # For this example, we'll use a publicly available reasoning dataset
        # In practice, you would load your specific 350K reasoning traces dataset
        try:
            dataset = load_dataset(self.config.dataset_name, split="train")
        except Exception as e:
            logger.warning(f"Could not load {self.config.dataset_name}: {e}")
            logger.info("Loading alternative reasoning dataset...")
            dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        
        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))
        
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset
    
    def format_training_examples(self, examples: List[Dict]) -> List[str]:
        """Format examples for instruction following."""
        formatted = []
        
        for example in examples:
            # Extract problem and solution
            if "question" in example and "answer" in example:
                question = example["question"]
                answer = example["answer"]
            elif "problem" in example and "solution" in example:
                question = example["problem"]
                answer = example["solution"]
            elif "input" in example and "output" in example:
                question = example["input"]
                answer = example["output"]
            else:
                # Try to find any text fields
                text_fields = [k for k, v in example.items() if isinstance(v, str) and len(v) > 10]
                if len(text_fields) >= 2:
                    question = example[text_fields[0]]
                    answer = example[text_fields[1]]
                else:
                    continue
            
            # Format as instruction-following conversation
            formatted_example = f"<|im_start|>system\nYou are a helpful assistant that provides step-by-step reasoning and solutions to problems.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            
            formatted.append(formatted_example)
        
        return formatted
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        formatted_examples = self.format_training_examples(examples)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_examples,
            truncation=True,
            padding=False,
            max_length=self.config.max_seq_length,
            return_tensors=None,
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        logger.info("Tokenizing dataset...")
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing examples"
        )
        
        # Split into train and validation
        train_size = int(len(tokenized_dataset) * self.config.train_split_ratio)
        
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset

class QwenFineTuner:
    """Fine-tunes Qwen model on reasoning traces."""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        
        # Set seed
        set_seed(config.seed)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        wandb.init(
            project="qwen-7b-finetuning",
            config=config.__dict__,
            name=f"qwen-7b-ft-{config.learning_rate}-{config.lora_r}"
        )
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def setup_lora(self, model):
        """Setup LoRA configuration."""
        logger.info("Setting up LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def create_training_arguments(self):
        """Create training arguments."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to="wandb",
            run_name=f"qwen-7b-ft-{self.config.learning_rate}-{self.config.lora_r}",
            seed=self.config.seed,
        )
    
    def fine_tune(self):
        """Run the fine-tuning process."""
        logger.info("Starting fine-tuning process...")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Setup LoRA
        model = self.setup_lora(model)
        
        # Prepare dataset
        data_processor = ReasoningTracesDataProcessor(self.config, tokenizer)
        raw_dataset = data_processor.load_dataset()
        train_dataset, eval_dataset = data_processor.prepare_dataset(raw_dataset)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Save training configuration
        config_path = Path(self.config.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metrics
        logs = trainer.state.log_history
        with open(Path(self.config.output_dir) / "training_logs.json", 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Fine-tuning completed. Model saved to {self.config.output_dir}")
        
        return trainer

def main():
    """Main function."""
    config = FineTuningConfig()
    fine_tuner = QwenFineTuner(config)
    trainer = fine_tuner.fine_tune()
    
    print("\n=== Fine-tuning completed ===")
    print(f"Model saved to: {config.output_dir}")
    print("Training logs and configuration saved alongside the model.")

if __name__ == "__main__":
    main()
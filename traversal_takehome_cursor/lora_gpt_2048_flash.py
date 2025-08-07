#!/usr/bin/env python3
"""LoRA GPT training with 2048 tokens using Flash Attention for memory efficiency"""
import gc
import warnings
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Union
import random
import re

import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Import code extraction function
from run_codeforces_ckpt import extract_code

# Memory optimization settings for long sequences
os.environ["PYTORCH_CUDALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

@dataclass
class FlashAttentionArguments:
    output_dir: str = field(default="/mnt/storage/qwen/flash_2048_lora_no_resume")
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    dataset_name: str = field(default="open-r1/Mixture-of-Thoughts")
    max_seq_len: int = field(default=2048, metadata={"help": "Long sequence with Flash Attention"})
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "High accumulation for 2048 tokens"})
    learning_rate: float = field(default=1e-4)
    max_steps: int = field(default=1000)
    lora_r: int = field(default=8, metadata={"help": "Moderate LoRA rank"})
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    deepspeed: str = field(default="ds_gpt_ultra_low_memory.json")
    # Dataset processing configuration
    code_extraction_probability: float = field(default=0.5, metadata={"help": "Probability of extracting code vs using full conversation (0.0-1.0)"})
    # Wandb configuration
    wandb_project: str = field(default="lora-gpt-2048-flash", metadata={"help": "WandB project name"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "WandB run name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "WandB entity/team name"})
    wandb_tags: Optional[str] = field(default="lora,gpt,2048,flash-attention", metadata={"help": "Comma-separated WandB tags"})

CHAT_TEMPLATE = "{header}{dialog}<|im_start|>assistant\n"

def build_chat(messages):
    """Convert HF chat format to ChatML string."""
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)

def get_long_sequence_dataset(tokenizer, args):
    """Dataset processing optimized for long sequences with conditional code extraction."""
    
    def tokenize_fn(example):
        try:
            messages = example.get("messages", [])
            if not messages or len(messages) < 2:
                return {"input_ids": [], "labels": [], "skip": True}
            
            # Decide whether to extract code based on probability
            should_extract_code = random.random() < args.code_extraction_probability
            
            if should_extract_code:
                # Extract problem prompt (user message) and assistant response
                user_message = None
                assistant_message = None
                
                for msg in messages:
                    if msg["role"] == "user":
                        user_message = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_message = msg["content"]
                
                if not user_message or not assistant_message:
                    return {"input_ids": [], "labels": [], "skip": True}
                
                # Extract code from assistant message
                extracted_code = extract_code(assistant_message)
                
                # Check if meaningful code was extracted
                if not extracted_code:
                    return {"input_ids": [], "labels": [], "skip": True}
                
                # Skip if extract_code just returned the original text (fallback case)
                if extracted_code.strip() == assistant_message.strip():
                    return {"input_ids": [], "labels": [], "skip": True}
                
                # Additional filtering: ensure extracted code looks like actual code
                code_lower = extracted_code.lower()
                has_code_keywords = any(keyword in code_lower for keyword in [
                    'def ', 'import ', 'class ', 'for ', 'while ', 'if ', 'else:', 'elif ',
                    '#include', 'using namespace', 'int main()', 'function', 'var ', 'let ', 'const '
                ])
                
                if not has_code_keywords:
                    return {"input_ids": [], "labels": [], "skip": True}

        
                # Create formatted conversation with extracted code
                formatted_messages = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": extracted_code}
                ]
                
                text = CHAT_TEMPLATE.format(
                    header="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                    dialog=build_chat(formatted_messages),
                )
            else:
                # Use original conversation format
                text = CHAT_TEMPLATE.format(
                    header="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                    dialog=build_chat(messages),
                )
            
            tokens = tokenizer(
                text,
                max_length=args.max_seq_len,
                truncation=True,
                padding=False,  # Dynamic padding
                return_tensors=None,
            )
            
            # Only keep sequences that are reasonably long
            if len(tokens["input_ids"]) < args.max_seq_len // 4:
                return {"input_ids": [], "labels": [], "skip": True}
                
            tokens["labels"] = tokens["input_ids"].copy()
            tokens["skip"] = False
            return tokens
        except Exception as e:
            return {"input_ids": [], "labels": [], "skip": True}

    ds = load_dataset(args.dataset_name, "code", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=2000, seed=42)
    ds = ds.map(tokenize_fn, batched=False, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: not x.get("skip", False))
    ds = ds.remove_columns(["skip"])
    
    return ds

def main():
    parser = argparse.ArgumentParser()
    for field_ in FlashAttentionArguments.__dataclass_fields__.values():
        field_type = field_.type
        # Handle typing.Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            field_type = field_type.__args__[0]  # Get the non-None type
        
        parser.add_argument(f"--{field_.name}", type=field_type, default=field_.default)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args(namespace=FlashAttentionArguments())
    
    # Aggressive memory clearing
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect()
    
    # Initialize wandb
    if args.local_rank <= 0:
        
        # Configure wandb tags
        wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else []
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            tags=wandb_tags,
            config={
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "max_seq_len": args.max_seq_len,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "max_steps": args.max_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "deepspeed_config": args.deepspeed,
                "code_extraction_probability": args.code_extraction_probability,
            }
        )
        print(f"🔗 WandB tracking initialized: {args.wandb_project}")
    
    # Tokenizer with long sequence support
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
        model_max_length=args.max_seq_len
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization for memory savings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
    )
    
    # Try to load model with Flash Attention
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,
            attn_implementation="flash_attention_2",  # Try Flash Attention
        )
        if args.local_rank <= 0:
            print("✅ Successfully loaded model with Flash Attention 2!")
            wandb.log({"flash_attention_enabled": True})
    except Exception as e:
        if args.local_rank <= 0:
            print(f"⚠️ Flash Attention 2 failed: {e}")
            print("Falling back to standard attention (may cause OOM)")
            wandb.log({"flash_attention_enabled": False, "flash_attention_error": str(e)})
        
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,
        )
    
    # Clear memory after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # Conservative LoRA configuration for long sequences
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
        init_lora_weights="gaussian",
    )
    model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True   
    )

    model = get_peft_model(model, lora_cfg)
    
    # Enable gradient checkpointing (essential for long sequences)
    model.gradient_checkpointing_enable()
    
    # Enable input gradients for LoRA compatibility
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    if args.local_rank <= 0:
        model.print_trainable_parameters()
        print(f"Model loaded with max sequence length: {args.max_seq_len}")
        
        # Verify gradient setup
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        frozen_params = [p for p in model.parameters() if not p.requires_grad]
        print(f"Trainable parameters: {len(trainable_params)}")
        print(f"Frozen parameters: {len(frozen_params)}")
        
        if len(trainable_params) == 0:
            print("❌ WARNING: No trainable parameters found!")
        else:
            print(f"✅ Found {len(trainable_params)} trainable parameters")
    
    # Dataset for long sequences
    train_dataset = get_long_sequence_dataset(tokenizer, args)
    
    # Data collator with dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Training arguments optimized for long sequences
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        
        # Memory optimizations (critical for 2048 tokens)
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        
        # Conservative optimizer settings
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-6,
        max_grad_norm=1.0,  # Match DeepSpeed config
        
        # DeepSpeed with aggressive CPU offloading
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        
        # Minimal features to save memory
        report_to=["wandb"] if args.local_rank <= 0 else ["none"],
        eval_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        resume_from_checkpoint=False,  # Don't restart from checkpoint
        
        # Additional memory optimizations
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=1,
        ignore_data_skip=True,  # Skip data consistency checks to save memory
        
        # Gradient and training stability
        skip_memory_metrics=True,  # Reduce memory overhead
        log_on_each_node=False,    # Reduce logging overhead
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Final memory clearing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    if args.local_rank <= 0:
        print("Starting 2048-token training (experimental)...")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory before training: {allocated:.2f}GB / {total:.2f}GB")
            print(f"Memory usage: {(allocated/total)*100:.1f}%")
            
            # Log initial memory stats to wandb
            wandb.log({
                "memory/gpu_allocated_gb": allocated,
                "memory/gpu_total_gb": total,
                "memory/gpu_usage_percent": (allocated/total)*100,
            })
        
        print("\n🚨 MEMORY WARNING:")
        print("2048 tokens require ~64GB GPU memory for activations alone")
        print("This will likely fail without Flash Attention or very large GPUs")
        print("Consider using shorter sequences (512-1024 tokens) for better stability\n")
    
    try:
        trainer.train()
        if args.local_rank <= 0:
            print("🎉 Successfully completed 2048-token training!")
            wandb.log({"training_completed": True})
    except torch.cuda.OutOfMemoryError as e:
        if args.local_rank <= 0:
            print(f"❌ OOM Error as expected: {e}")
            print("\nSuggested fixes:")
            print("1. Reduce sequence length to 512-1024 tokens")
            print("2. Install Flash Attention: pip install flash-attn")
            print("3. Use larger GPUs (A100 80GB, H100)")
            print("4. Reduce batch size further or increase gradient accumulation")
            wandb.log({"training_failed": True, "oom_error": str(e)})
        raise
    
    # Save and cleanup
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    if args.local_rank <= 0:
        print(f"Training completed! Model saved to {args.output_dir}")
        # Final memory logging
        if torch.cuda.is_available():
            final_allocated = torch.cuda.memory_allocated() / 1024**3
            wandb.log({
                "memory/final_gpu_allocated_gb": final_allocated,
                "training/model_saved": True,
                "training/output_dir": args.output_dir
            })
        wandb.finish()

if __name__ == "__main__":
    main()
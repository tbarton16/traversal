#!/usr/bin/env python
"""
Multi-node SFT of Qwen/Qwen2.5-7B-Instruct on Mixture-of-Thoughts.

Launch (DeepSpeed):
deepspeed --num_nodes 1 --num_gpus 8 finetune_qwen.py \
  --deepspeed ds_config.json \
  --output_dir /mnt/storage/traversal__takehome/qwen2.5-mot-7b-sft
"""
import os, json, hashlib, argparse, wandb, torch, gc
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)

CHAT_TEMPLATE = "{header}{dialog}<|im_start|>assistant\n"

def build_chat(messages: list[str|dict]) -> str:
    """Convert HF chat dicts to ChatML string."""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)

def sha256(s): return hashlib.sha256(s.encode()).hexdigest()

def preprocess(example, tokenizer, max_len):
    # Mixture-of-Thoughts dataset uses "messages" 
    messages = example.get("messages", [])
    text = CHAT_TEMPLATE.format(
        header="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        dialog=build_chat(messages),
    )
    tok = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",  # Pad to max_length to ensure consistent tensor sizes
        add_special_tokens=False,
        return_attention_mask=True,
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_name", default="open-r1/Mixture-of-Thoughts")
    parser.add_argument("--dataset_config", default="code")
    parser.add_argument("--max_seq_len", type=int, default=4096)  # Reduced from 2048
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for testing")
    parser.add_argument("--output_dir", required=True)
    args, unknown_args = parser.parse_known_args()

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.distributed.get_world_size()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[DEBUG] Loading dataset {args.dataset_name}...")
    # Use reuse mode to avoid cache compatibility issues but allow fresh download if needed
    raw = load_dataset(
        args.dataset_name, 
        args.dataset_config, 
        split="train",
        streaming=False,
        download_mode="reuse_dataset_if_exists"
    )
    
    print(f"[DEBUG] Original dataset size: {len(raw)}")
    print(f"[DEBUG] Sample columns: {raw.column_names}")
    if len(raw) > 0:
        print(f"[DEBUG] Sample entry: {raw[0]}")
    
    # Optionally limit dataset size for faster testing
    if args.max_samples and len(raw) > args.max_samples:
        print(f"[DEBUG] Limiting dataset to {args.max_samples} samples")
        raw = raw.select(range(args.max_samples))

    # Filter & dedupe - optimize for memory and speed
    print("[DEBUG] Starting filtering and preprocessing...")
    
    # Step 1: Basic filtering (single process to avoid memory issues)
    def basic_filter(ex):
        if "messages" not in ex or not ex["messages"]:
            return False
        messages = ex["messages"]
        return len(messages) >= 2  # Need at least user + assistant
    
    raw = raw.filter(basic_filter, num_proc=1)
    print(f"[DEBUG] After basic filtering: {len(raw)}")
    
    # Step 2: Dedupe in batches to manage memory
    unique = set()
    # def dedupe_batch(examples):
    #     keep_indices = []
    #     for i, ex in enumerate(examples["messages"]):
    #         # Create hash for deduplication
    #         content_parts = []
    #         for msg in ex:
    #             if isinstance(msg, dict) and "content" in msg:
    #                 content_parts.append(msg["content"])
            
    #         if content_parts:
    #             h = sha256("".join(content_parts))
    #             if h not in unique:
    #                 unique.add(h)
    #                 keep_indices.append(i)
        
    #     # Return filtered batch
    #     return {key: [examples[key][i] for i in keep_indices] 
    #             for key in examples.keys()}
    
    # Process in smaller batches to avoid memory issues
    # raw = raw.map(dedupe_batch, batched=True, batch_size=1000, num_proc=1)
    # print(f"[DEBUG] After deduplication: {len(raw)}")
    # Step 3: Tokenization with limited multiprocessing
    max_workers = min(4, os.cpu_count())  # Limit to 4 workers max
    print(f"[DEBUG] Using {max_workers} workers for tokenization...")
    
    proc = raw.map(
        lambda e: preprocess(e, tokenizer, args.max_seq_len),
        batched=False, 
        remove_columns=raw.column_names,
        num_proc=20,
        load_from_cache_file=False  # Disable caching to avoid disk I/O issues
    )
    print(f"[DEBUG] Processed dataset size: {len(proc)}")
    print(f"[DEBUG] Processed dataset columns: {proc.column_names}")
    
    # Clear any existing CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Train
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    
    # Clear cache again after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,          # real batch = this × grad_acc × GPUs
        gradient_accumulation_steps=4,          # Reduced to balance with higher per_device_batch_size
        learning_rate=2e-4,
        warmup_steps=500,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        bf16=True,
        deepspeed="ds_config.json",
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_pin_memory=False,            # Reduce memory pressure
        dataloader_num_workers=0,               # Reduce memory overhead
        dataloader_drop_last=True,              # Drop incomplete batches to avoid size issues
        max_grad_norm=1.0,                      # Explicit gradient clipping
        optim="adamw_torch",                    # Use memory-efficient optimizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=proc,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

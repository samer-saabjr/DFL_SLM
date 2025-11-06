#!/usr/bin/env python
# src/main/main_glue_lora.py
"""
Main entry for GLUE + RoBERTa + LoRA (Wq/Wv only).
- Loads cached models (from your download script) via HF cache
- Uses modular data_prep / models / metrics / utils

This script fine-tunes RoBERTa models on GLUE benchmark tasks using LoRA
(Low-Rank Adaptation), a parameter-efficient fine-tuning technique that
only trains small adapter matrices attached to the query and value weights
in the attention layers, significantly reducing the number of trainable
parameters while maintaining competitive performance.
"""

import os
import argparse
import numpy as np
from transformers import Trainer, TrainingArguments

# Import custom modules for data preparation, metrics, models, and utilities
from data_prep import load_and_tokenize, eval_split_names
from metrics import build_metric_fn, evaluate_mnli_overall
from models import resolve_model_id, build_model, count_trainable_params, build_collator
from utils import set_global_seed, write_manifest

def main():
    """
    Main function to fine-tune RoBERTa on GLUE tasks using LoRA.
    
    Workflow:
    1. Parse command-line arguments
    2. Set up output directory and random seed
    3. Load and tokenize the specified GLUE task dataset
    4. Build RoBERTa model with LoRA adapters
    5. Configure HuggingFace Trainer
    6. Train and/or evaluate the model
    7. Save results and manifest
    """
    # ==================== Argument Parsing ====================
    ap = argparse.ArgumentParser()
    
    # Model configuration
    ap.add_argument("--model_size", choices=["base", "large"], default="base",
                    help="RoBERTa model size: 'base' (~125M params) or 'large' (~355M params)")
    
    # Task selection - GLUE benchmark has 9 different NLP tasks
    ap.add_argument("--task_name", required=True, 
                    help="GLUE task: cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli")
    
    # Output and reproducibility
    ap.add_argument("--output_dir", default="./out",
                    help="Directory to save model checkpoints and results")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")

    # LoRA hyperparameters
    # LoRA works by adding trainable low-rank matrices A and B such that W' = W + BA
    ap.add_argument("--lora_r", type=int, default=8,
                    help="LoRA rank: dimensionality of low-rank matrices (lower = fewer params)")
    ap.add_argument("--lora_alpha", type=int, default=8,
                    help="LoRA scaling factor; typical practice is alpha = r")
    ap.add_argument("--lora_dropout", type=float, default=0.0,
                    help="Dropout probability for LoRA layers")

    # Training hyperparameters
    # Note: Sequence length is fixed at 128 tokens (handled in data_prep module)
    ap.add_argument("--per_device_train_batch_size", type=int, default=32,
                    help="Training batch size per GPU/CPU device")
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32,
                    help="Evaluation batch size per GPU/CPU device")
    ap.add_argument("--learning_rate", type=float, default=2e-4,
                    help="Learning rate for optimizer (2e-4 is typical for LoRA)")
    ap.add_argument("--num_train_epochs", type=float, default=3.0,
                    help="Number of training epochs")
    
    # Execution flags
    ap.add_argument("--do_train", action="store_true",
                    help="Whether to run training")
    ap.add_argument("--do_eval", action="store_true",
                    help="Whether to run evaluation")

    # Save the trained LoRA adapter to this directory
    ap.add_argument("--adapter_output_dir", type=str, default=None,
                    help="Directory to save the trained LoRA adapter. Defaults to <output_dir>/<task_name>_<model_size>/r<lora_r>/adapter")

    # Parse arguments and set up environment
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)  # Create output directory if needed
    set_global_seed(args.seed)  # Set random seeds for NumPy, PyTorch, etc.
    
    # Set default adapter_output_dir if not specified (includes rank for comparison)
    if args.adapter_output_dir is None:
        args.adapter_output_dir = os.path.join(args.output_dir, f"{args.task_name}_{args.model_size}", f"r{args.lora_r}", "adapter")
    
    # Create the training and adapter output directories immediately so they're visible
    training_output_dir = os.path.join(args.output_dir, f"{args.task_name}_{args.model_size}", f"r{args.lora_r}")
    os.makedirs(training_output_dir, exist_ok=True)
    if args.do_train:
        os.makedirs(args.adapter_output_dir, exist_ok=True)
        print(f"📁 Training output: {training_output_dir}")
        print(f"📁 Adapter will be saved to: {args.adapter_output_dir}")

    # ==================== Data Loading & Tokenization ====================
    # Resolve model identifier (e.g., "base" -> "roberta-base")
    model_id = resolve_model_id(args.model_size)
    
    # Load GLUE task dataset and tokenize it
    # Returns:
    # - tokenized: DatasetDict with tokenized train/validation/test splits
    # - num_labels: Number of classes (e.g., 2 for binary, 3 for MNLI, 1 for regression)
    # - is_regression: True for STS-B (regression task), False for classification
    # - tokenizer: The RoBERTa tokenizer used
    tokenized, num_labels, is_regression, tokenizer = load_and_tokenize(
        args.task_name, model_id, max_length=128
    )

    # ==================== Model Setup with LoRA ====================
    # Build RoBERTa model with LoRA adapters applied to query/value attention weights
    # This dramatically reduces trainable parameters while maintaining performance
    model = build_model(
        model_id=model_id,
        num_labels=num_labels,
        is_regression=is_regression,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Count and display trainable vs total parameters
    # LoRA typically trains <1% of total parameters
    trainable, total = count_trainable_params(model)
    print(f"Model: {model_id} | Trainable params (LoRA): {trainable:,} / Total: {total:,}")

    # ==================== Trainer Configuration ====================
    # Build data collator for dynamic padding
    collator = build_collator(tokenizer)
    
    # Get the correct split names for this task (most use "train", but some vary)
    train_split, _ = eval_split_names(args.task_name)
    
    # MNLI is special: has two validation sets (matched and mismatched domains)
    is_mnli = (args.task_name.lower() == "mnli")
    eval_dataset = (tokenized["validation_mismatched"] if is_mnli else tokenized["validation"])

    # Configure training arguments for HuggingFace Trainer
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=args.do_train,
        do_eval=args.do_eval,
        eval_strategy="epoch" if args.do_train else "no",  # Evaluate after each epoch
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,  # Log metrics every 50 steps
        save_strategy="epoch" if args.do_train else "no",  # Save checkpoint after each epoch
        load_best_model_at_end=args.do_train,  # Load best checkpoint at end of training
        report_to="none",  # Don't report to wandb, tensorboard, etc.
        seed=args.seed,
    )

    # Initialize HuggingFace Trainer with model, datasets, and compute_metrics function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized[train_split] if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_metric_fn(args.task_name),  # Task-specific metrics
    )

    # ==================== Training ====================
    if args.do_train:
        print(f"\nStarting training on dataset: {args.task_name}...")
        trainer.train()
        print(f"Training complete for dataset: {args.task_name}!")

    if args.do_train and args.adapter_output_dir:
        # Save ONLY the LoRA adapter weights + config (what fedavg_merge needs)
        model.save_pretrained(args.adapter_output_dir)
        print(f"💾 Saved LoRA adapter to: {args.adapter_output_dir}")
    # ==================== Evaluation ====================
    results = {}
    if args.do_eval:
        print(f"\nStarting evaluation on dataset: {args.task_name}...")
        if is_mnli:
            # MNLI has two validation sets: matched and mismatched
            # Need to evaluate on both and report combined results
            results.update(evaluate_mnli_overall(trainer, tokenized))
        else:
            # Standard evaluation on single validation set
            res = trainer.evaluate()
            results.update(res)

    # ==================== Display Results ====================
    if results:
        print("\n" + "="*50)
        print(f"EVALUATION RESULTS - Dataset: {args.task_name.upper()}")
        print("="*50)
        for k, v in results.items():
            # Format numeric values with 6 decimal places
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")
        print("="*50)

    # ==================== Save Manifest ====================
    # Write a manifest file documenting the experiment configuration
    # Useful for tracking hyperparameters and reproducing results
    manifest = os.path.join(training_args.output_dir, "manifest.txt")
    write_manifest(
        manifest,
        {
            "Model size": args.model_size,
            "Task": args.task_name,
            "Seq len": 128,
            "Uniform batch (train/eval)": f"{args.per_device_train_batch_size}/{args.per_device_eval_batch_size}",
            "LoRA r/alpha/dropout": f"{args.lora_r}/{args.lora_alpha}/{args.lora_dropout}",
        },
        dataset_name=args.task_name,
    )
    print(f"\nWrote manifest for dataset '{args.task_name}': {manifest}")

if __name__ == "__main__":
    main()

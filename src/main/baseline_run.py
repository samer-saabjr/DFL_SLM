#!/usr/bin/env python
# src/main/baseline_run.py
# Evaluate RoBERTa on GLUE WITHOUT LoRA.
# Supports zero-shot (eval-only) and full fine-tuning (train+eval).

from __future__ import annotations
import os
import argparse
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoConfig

from data_prep import load_and_tokenize, eval_split_names
from metrics import build_metric_fn, evaluate_mnli_overall
from models import resolve_model_id, build_collator
from utils import set_global_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", choices=["base","large"], required=True)
    ap.add_argument("--task_name", required=True, help="cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli")
    ap.add_argument("--output_dir", default="./out_baseline")
    ap.add_argument("--seed", type=int, default=42)

    # Fairness knobs (match LoRA discipline: uniform batch across tasks; seq_len=128 is set in data loader)
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)

    # Typical LR for full fine-tuning is lower than LoRA; default 2e-5 (tune if you wish)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)

    ap.add_argument("--do_train", action="store_true", help="Full fine-tuning baseline")
    ap.add_argument("--do_eval", action="store_true", help="Evaluate (zero-shot if no training)")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)

    model_id = resolve_model_id(args.model_size)

    # Tokenize and infer label spaces correctly for the task; seq_len fixed to 128 inside
    tokenized, num_labels, is_regression, tokenizer = load_and_tokenize(args.task_name, model_id, max_length=128)

    # Build plain (non-PEFT) model with the right head
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.num_labels = 1 if is_regression else num_labels
    cfg.problem_type = "regression" if is_regression else "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=cfg)

    # Datasets
    train_split, _ = eval_split_names(args.task_name)
    is_mnli = (args.task_name.lower() == "mnli")
    eval_dataset = tokenized["validation_mismatched"] if is_mnli else tokenized["validation"]

    # Trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.task_name}_{args.model_size}"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=args.do_train,
        do_eval=args.do_eval,
        eval_strategy="epoch" if args.do_train else "no",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_strategy="epoch" if args.do_train else "no",
        load_best_model_at_end=args.do_train,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized[train_split] if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=build_collator(tokenizer),
        compute_metrics=build_metric_fn(args.task_name),
    )

    # Train (full fine-tuning baseline) / Eval (zero-shot or after FT)
    if args.do_train:
        print(f"\nStarting baseline training on dataset: {args.task_name}...")
        trainer.train()
        print(f"Baseline training complete for dataset: {args.task_name}!")

    results = {}
    if args.do_eval:
        print(f"\nStarting baseline evaluation on dataset: {args.task_name}...")
        if is_mnli:
            results.update(evaluate_mnli_overall(trainer, tokenized))
        else:
            res = trainer.evaluate()
            results.update(res)

    if results:
        print(f"\n==== BASELINE DEV RESULTS (no LoRA) - Dataset: {args.task_name.upper()} ====")
        for k, v in results.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")
        print("="*50)

if __name__ == "__main__":
    main()

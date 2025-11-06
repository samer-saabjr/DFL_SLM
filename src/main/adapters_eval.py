#!/usr/bin/env python
# src/main/adapters_eval.py
# Load base + adapter (client or fused) and evaluate on GLUE dev.

from __future__ import annotations
import os
import argparse
import numpy as np
from transformers import Trainer, TrainingArguments
from peft import PeftModel

from .data_prep import load_and_tokenize, eval_split_names
from .metrics import build_metric_fn, evaluate_mnli_overall
from .models import resolve_model_id
from .utils import set_global_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", choices=["base","large"], required=True)
    ap.add_argument("--task_name", required=True)  # cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--output_dir", default="./out_eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)

    model_id = resolve_model_id(args.model_size)

    # Tokenize & infer label spaces correctly for this task
    tokenized, num_labels, is_regression, tokenizer = load_and_tokenize(args.task_name, model_id, max_length=128)

    # Base model with correct head shape for the task
    from transformers import AutoModelForSequenceClassification, AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.num_labels = 1 if is_regression else num_labels
    cfg.problem_type = "regression" if is_regression else "single_label_classification"
    base = AutoModelForSequenceClassification.from_pretrained(model_id, config=cfg)

    # Load adapter
    model = PeftModel.from_pretrained(base, args.adapter_path)

    # Trainer
    is_mnli = (args.task_name.lower() == "mnli")
    eval_dataset = (tokenized["validation_mismatched"] if is_mnli else tokenized["validation"])

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.task_name}_{args.model_size}"),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=False,
        do_eval=True,
        evaluation_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        compute_metrics=build_metric_fn(args.task_name),
    )

    # Evaluate (MNLI overall vs single-split)
    print(f"\nEvaluating adapter on dataset: {args.task_name}...")
    results = {}
    if is_mnli:
        results.update(evaluate_mnli_overall(trainer, tokenized))
    else:
        res = trainer.evaluate()
        results.update(res)

    print(f"\n{'='*50}")
    print(f"DEV EVAL RESULTS - Dataset: {args.task_name.upper()}")
    print(f"{'='*50}")
    for k, v in results.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("="*50)

if __name__ == "__main__":
    main()

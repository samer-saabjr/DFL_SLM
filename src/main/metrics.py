# src/main/metrics.py
"""
Metric computation functions for GLUE benchmark tasks.

This module provides utilities for evaluating model performance on various
GLUE tasks using task-appropriate metrics (accuracy, Matthews correlation, Pearson correlation).
"""

from typing import Dict, Any
import numpy as np
import evaluate
from transformers import Trainer

def build_metric_fn(task_name: str):
    """
    Build a metric computation function for a specific GLUE task.
    
    Creates and returns a metric computation function tailored to the specified
    GLUE task. Different tasks use different evaluation metrics:
    - CoLA: Matthews Correlation Coefficient (MCC)
    - STS-B: Pearson correlation (regression task)
    - Other tasks: Accuracy
    
    Args:
        task_name: Name of the GLUE task (e.g., "cola", "stsb", "mnli", "sst2")
    
    Returns:
        A compute_metrics function that takes eval_pred and returns a dict of metrics
    """
    t = task_name.lower()
    
    # CoLA (Corpus of Linguistic Acceptability) uses Matthews Correlation Coefficient
    # This is a binary classification task where MCC is preferred over accuracy
    # because the dataset is imbalanced
    if t == "cola":
        mcc = evaluate.load("matthews_correlation")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            # Convert logits to class predictions by taking argmax
            preds = np.argmax(logits, axis=-1)
            return {"matthews_correlation": mcc.compute(predictions=preds, references=labels)["matthews_correlation"]}
        return compute_metrics

    # STS-B (Semantic Textual Similarity Benchmark) is a regression task
    # It predicts similarity scores (0-5), so we use Pearson correlation
    if t == "stsb":
        pearson = evaluate.load("pearsonr")
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            # For regression, predictions are already continuous values
            preds = preds.squeeze()
            return {"pearson": pearson.compute(predictions=preds, references=labels)["pearsonr"]}
        return compute_metrics

    # Default metric for most GLUE tasks (MNLI, SST-2, QNLI, QQP, RTE, MRPC, WNLI)
    # These are classification tasks where accuracy is the standard metric
    acc = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Convert logits to class predictions by taking argmax
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": acc.compute(predictions=preds, references=labels)["accuracy"]}
    return compute_metrics

def evaluate_mnli_overall(trainer: Trainer, tokenized) -> Dict[str, Any]:
    """
    Evaluate MNLI on both matched and mismatched validation sets.
    
    MNLI (Multi-Genre Natural Language Inference) has two validation sets:
    - validation_matched: Same genres as training data
    - validation_mismatched: Different genres from training data
    
    This function evaluates on both sets and computes the overall accuracy
    as the average of the two, which is the standard MNLI evaluation protocol.
    
    Args:
        trainer: HuggingFace Trainer object with the model to evaluate
        tokenized: Dictionary containing tokenized validation datasets with keys
                   "validation_matched" and "validation_mismatched"
    
    Returns:
        Dictionary with three metrics:
        - mnli_accuracy_matched: Accuracy on matched validation set
        - mnli_accuracy_mismatched: Accuracy on mismatched validation set
        - mnli_accuracy_overall: Average of matched and mismatched accuracies
    """
    # Temporarily replace the trainer's metric function with MNLI-specific metrics
    original = trainer.compute_metrics
    trainer.compute_metrics = build_metric_fn("mnli")

    # Evaluate on matched validation set (same genre as training)
    eval_mm = trainer.evaluate(eval_dataset=tokenized["validation_matched"])
    
    # Evaluate on mismatched validation set (different genre from training)
    eval_mmm = trainer.evaluate(eval_dataset=tokenized["validation_mismatched"])

    # Restore the original metric function
    trainer.compute_metrics = original

    # Extract accuracy values from evaluation results
    # (evaluation results may have multiple keys, we want the one with "accuracy")
    acc_mm = [v for k, v in eval_mm.items() if "accuracy" in k]
    acc_mmm = [v for k, v in eval_mmm.items() if "accuracy" in k]
    
    # Get the first accuracy value, or NaN if not found
    acc_mm = acc_mm[0] if acc_mm else float("nan")
    acc_mmm = acc_mmm[0] if acc_mmm else float("nan")
    
    # Compute overall accuracy as the mean of matched and mismatched
    # (using nanmean to handle potential NaN values gracefully)
    overall = float(np.nanmean([acc_mm, acc_mmm]))

    return {
        "mnli_accuracy_matched": acc_mm,
        "mnli_accuracy_mismatched": acc_mmm,
        "mnli_accuracy_overall": overall
    }

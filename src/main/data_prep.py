# src/main/data_prep.py
"""
Data preparation module for GLUE benchmark tasks.

This module handles loading and tokenizing GLUE (General Language Understanding Evaluation)
benchmark datasets for fine-tuning language models. It supports all 9 GLUE tasks and handles
both single-sentence and sentence-pair classification/regression tasks.
"""

from typing import Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
import torch  # noqa

# Mapping of GLUE task names to their input field names
# Format: "task_name": (first_sentence_field, second_sentence_field)
# None for second field indicates single-sentence tasks
GLUE_FIELDS = {
    "cola": ("sentence", None),           # Corpus of Linguistic Acceptability (single sentence)
    "sst2": ("sentence", None),           # Stanford Sentiment Treebank (single sentence)
    "mrpc": ("sentence1", "sentence2"),   # Microsoft Research Paraphrase Corpus (sentence pairs)
    "stsb": ("sentence1", "sentence2"),   # Semantic Textual Similarity Benchmark (sentence pairs, regression)
    "qqp": ("question1", "question2"),    # Quora Question Pairs (question pairs)
    "mnli": ("premise", "hypothesis"),    # Multi-Genre Natural Language Inference (sentence pairs)
    "qnli": ("question", "sentence"),     # Question Natural Language Inference (question-sentence pairs)
    "rte": ("sentence1", "sentence2"),    # Recognizing Textual Entailment (sentence pairs)
    "wnli": ("sentence1", "sentence2"),   # Winograd Natural Language Inference (sentence pairs)
}

def load_and_tokenize(
    task_name: str,
    model_id: str,
    max_length: int = 128
):
    """
    Load a GLUE task dataset and tokenize it for model training.
    
    Args:
        task_name: Name of the GLUE task (e.g., "cola", "sst2", "mnli")
        model_id: HuggingFace model identifier for loading the appropriate tokenizer
        max_length: Maximum sequence length for tokenization (default: 128)
    
    Returns:
        tuple: (tokenized_dataset, num_labels, is_regression, tokenizer)
            - tokenized_dataset: Dataset with tokenized inputs and labels
            - num_labels: Number of output labels (1 for regression, 2 or 3 for classification)
            - is_regression: Boolean indicating if task is regression (True for STS-B)
            - tokenizer: The tokenizer instance used for tokenization
    
    Raises:
        ValueError: If task_name is not a valid GLUE task
    """
    # Normalize task name to lowercase
    task = task_name.lower()
    if task not in GLUE_FIELDS:
        raise ValueError(f"Unknown GLUE task: {task}")

    # Load the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Load the raw GLUE dataset from HuggingFace
    raw = load_dataset("glue", task)
    
    # Get the field names for this task (single sentence or sentence pair)
    sent1_key, sent2_key = GLUE_FIELDS[task]

    def _tok_fn(batch):
        """Tokenization function applied to each batch of examples."""
        # Single-sentence tasks (CoLA, SST-2)
        if sent2_key is None:
            return tokenizer(batch[sent1_key], truncation=True, max_length=max_length)
        # Sentence-pair tasks (MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI)
        return tokenizer(batch[sent1_key], batch[sent2_key], truncation=True, max_length=max_length)

    # Remove original text columns, keeping only label/score and tokenized features
    # This reduces memory usage and focuses on model inputs
    to_remove = [c for c in raw["train"].column_names if c not in ("label", "score")]
    tokenized = raw.map(_tok_fn, batched=True, remove_columns=to_remove)

    # Handle STS-B as a regression task (predicting similarity scores 0-5)
    is_regression = (task == "stsb")
    if is_regression:
        # STS-B uses "score" field instead of "label" - handle both cases
        label_col = "label" if "label" in tokenized["train"].features else "score"
        tokenized = tokenized.rename_column(label_col, "labels")
        tokenized.set_format(type="torch")  # Convert to PyTorch tensors
        return tokenized, 1, True, tokenizer  # 1 output for regression

    # Handle all classification tasks (binary or 3-way classification)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")  # Convert to PyTorch tensors
    
    # MNLI has 3 classes (entailment, neutral, contradiction), others are binary
    num_labels = 3 if task == "mnli" else 2
    return tokenized, num_labels, False, tokenizer

def eval_split_names(task_name: str) -> Tuple[str, str]:
    """
    Return the dataset split names for training and evaluation.
    
    Args:
        task_name: Name of the GLUE task
    
    Returns:
        tuple: (train_split_name, eval_split_name)
            - train_split_name: Name of the training split (always "train")
            - eval_split_name: Name of the evaluation split (always "validation")
    
    Note:
        MNLI has two validation sets (matched and mismatched), but both are under
        the "validation" split. Special handling for MNLI evaluation is done in
        the metrics module (see metrics.evaluate_mnli_overall).
    """
    t = task_name.lower()
    if t == "mnli":
        # MNLI will be evaluated on both matched and mismatched validation sets
        # separately in the metrics computation
        return "train", "validation"
    
    # All other GLUE tasks use standard train/validation splits
    return "train", "validation"

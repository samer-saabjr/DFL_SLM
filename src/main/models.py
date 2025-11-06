# src/main/models.py
"""
Model configuration utilities for RoBERTa with LoRA adapters.

This module provides functions to build and configure RoBERTa models with
Low-Rank Adaptation (LoRA) for efficient fine-tuning on sequence classification tasks.
"""

from typing import Tuple
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model


def resolve_model_id(model_size: str) -> str:
    """
    Convert a simple model size identifier to the full HuggingFace model ID.
    
    Args:
        model_size: Either "base" or "large" to specify model variant
        
    Returns:
        Full HuggingFace model identifier (e.g., "roberta-base" or "roberta-large")
        
    Raises:
        ValueError: If model_size is not "base" or "large"
    """
    if model_size not in {"base", "large"}:
        raise ValueError("model_size must be 'base' or 'large'")
    return "roberta-base" if model_size == "base" else "roberta-large"

def build_model(
    model_id: str,
    num_labels: int,
    is_regression: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float
):
    """
    Build a RoBERTa sequence classification model with LoRA adapters.
    
    This function creates a pre-trained RoBERTa model and applies Low-Rank Adaptation (LoRA)
    to the query and value matrices in the attention layers. LoRA dramatically reduces the
    number of trainable parameters while maintaining model performance, making it ideal for
    federated learning and resource-constrained scenarios.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "roberta-base")
        num_labels: Number of output labels for classification (ignored if is_regression=True)
        is_regression: If True, configure for regression task; if False, for classification
        lora_r: Rank of the LoRA update matrices (lower = fewer trainable params)
        lora_alpha: Scaling factor for LoRA updates (typically set to lora_r or 2*lora_r)
        lora_dropout: Dropout probability for LoRA layers
        
    Returns:
        RoBERTa model wrapped with LoRA adapters, ready for training
    """
    # Load the pre-trained model configuration
    cfg = AutoConfig.from_pretrained(model_id)
    
    # Configure the model head based on task type
    if is_regression:
        cfg.num_labels = 1  # Single output for regression
        cfg.problem_type = "regression"
    else:
        cfg.num_labels = num_labels  # Multiple outputs for classification
        cfg.problem_type = "single_label_classification"

    # Load the pre-trained model with sequence classification head
    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=cfg)

    # Configure LoRA parameters
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
        r=lora_r,  # Rank of LoRA matrices (controls adapter capacity)
        lora_alpha=lora_alpha,  # Scaling parameter for LoRA updates
        lora_dropout=lora_dropout,  # Regularization via dropout
        target_modules=["query", "value"],  # Only adapt attention Q and V matrices (Wq/Wv)
        bias="none"  # Don't add LoRA to bias terms
    )
    
    # Wrap the model with LoRA adapters (freezes base model, adds trainable adapters)
    model = get_peft_model(model, peft_cfg)
    return model

def count_trainable_params(model) -> Tuple[int, int]:
    """
    Count the number of trainable and total parameters in a model.
    
    This is particularly useful for verifying that LoRA is working correctly - you should
    see a dramatic reduction in trainable parameters (typically <1% of total) when using LoRA.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Tuple of (trainable_params, total_params)
        
    Example:
        >>> trainable, total = count_trainable_params(model)
        >>> print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    """
    # Sum all parameters that have requires_grad=True (i.e., will be updated during training)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Sum all parameters in the model (frozen + trainable)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_collator(tokenizer):
    """
    Create a data collator for dynamic padding of tokenized sequences.
    
    The collator handles batching of variable-length sequences by padding them to the
    maximum length in each batch (rather than a fixed global maximum). This improves
    efficiency by reducing unnecessary padding tokens.
    
    Args:
        tokenizer: HuggingFace tokenizer instance (must match the model's tokenizer)
        
    Returns:
        DataCollatorWithPadding instance for use with PyTorch DataLoader
    """
    return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

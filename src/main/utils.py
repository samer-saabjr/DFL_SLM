# src/main/utils.py
"""
Utility functions for the GLUE benchmark experiments with RoBERTa and LoRA.

This module provides helper functions for:
- Setting random seeds for reproducibility
- Writing experiment manifests for tracking configurations and results
"""

import os
from typing import Dict, Any
from transformers import set_seed


def set_global_seed(seed: int):
    """
    Set the global random seed for reproducibility.
    
    This ensures consistent results across multiple runs by setting seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator (CPU and CUDA)
    - Transformers library operations
    
    Args:
        seed (int): The random seed value to use
    """
    set_seed(seed)


def write_manifest(path: str, notes: Dict[str, Any], dataset_name: str = "GLUE"):
    """
    Write a manifest file documenting the experiment configuration and results.
    
    Creates a text file containing metadata about the training run, including
    hyperparameters, performance metrics, and other relevant information.
    The manifest helps track experiments and reproduce results.
    
    Args:
        path (str): File path where the manifest should be written
        notes (Dict[str, Any]): Dictionary of key-value pairs to record in the manifest.
                                Typically includes model name, task, learning rate,
                                batch size, accuracy, etc.
        dataset_name (str): Name of the dataset (e.g., "GLUE", "sst2", "mnli")
    
    Example:
        >>> notes = {
        ...     "task": "sst2",
        ...     "model": "roberta-base",
        ...     "accuracy": 0.9234,
        ...     "learning_rate": 2e-4
        ... }
        >>> write_manifest("out/sst2_base/manifest.txt", notes, dataset_name="sst2")
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Write the manifest file with experiment information
    with open(path, "w", encoding="utf-8") as f:
        # Header indicating the experimental setup with dataset name
        f.write(f"Dataset: {dataset_name}\n")
        f.write("GLUE + RoBERTa + LoRA (Wq/Wv only)\n")
        
        # Write all configuration/result parameters
        for k, v in notes.items():
            f.write(f"{k}: {v}\n")
        
        # Add reference to the LoRA implementation
        f.write("LoRA repo: https://github.com/microsoft/LoRA\n")

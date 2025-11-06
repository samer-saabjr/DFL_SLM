# src/main/sharding.py
"""
Data Sharding Module for Federated Learning on GLUE Tasks

This module provides utilities to partition GLUE benchmark datasets across
multiple federated learning clients. It supports three sharding strategies:
- IID: Stratified random splits maintaining label distribution
- Overlap: Partially overlapping data splits across clients
- Category: Non-IID splits assigning entire categories to specific clients
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Sequence
import json
import math
import random
from collections import defaultdict
import numpy as np
from datasets import DatasetDict, load_dataset

# ============================================================================
# Utility Functions
# ============================================================================

def _rng(seed: int):
    """
    Create a seeded random number generator for reproducible shuffling.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Random.Random instance with fixed seed
    """
    r = random.Random(seed)
    return r

def _bin_scores(values: Sequence[float], bin_width: float = 0.2) -> List[int]:
    """
    Convert continuous regression scores into discrete bins for stratification.
    
    Used primarily for STS-B (Semantic Textual Similarity Benchmark) which has
    regression labels in [0.0, 5.0]. Binning allows stratified sampling on
    regression tasks.
    
    Args:
        values: Sequence of continuous scores (e.g., STS-B similarity scores)
        bin_width: Width of each bin (default 0.2 creates 25 bins for range [0,5])
        
    Returns:
        List of bin indices, e.g., [0.0,0.2)→0, [0.2,0.4)→1, [0.4,0.6)→2, etc.
        
    Example:
        >>> _bin_scores([0.15, 0.25, 0.35, 0.45], bin_width=0.2)
        [0, 1, 1, 2]
    """
    return [int(math.floor(v / bin_width)) for v in values]

def _indices_by_key(examples: Dict[str, Any], key: str) -> Dict[Any, List[int]]:
    """
    Group dataset indices by unique values of a specific key (e.g., labels).
    
    This enables stratified sampling by grouping examples with the same
    label/category together, then distributing across clients proportionally.
    
    Args:
        examples: Dictionary with dataset columns (e.g., {"label": [...], "text": [...]})
        key: Column name to group by (e.g., "label", "genre", "bin")
        
    Returns:
        Dictionary mapping each unique value to list of indices
        Example: {0: [1, 5, 9], 1: [2, 3, 7], 2: [4, 6, 8]}
    """
    buckets: Dict[Any, List[int]] = defaultdict(list)
    for i, v in enumerate(examples[key]):
        buckets[v].append(i)
    return buckets

# ============================================================================
# Main Sharding Function
# ============================================================================

def make_client_splits(
    task_name: str,
    k: int,
    mode: str = "iid",           # "iid" | "overlap" | "category"
    seed: int = 42,
    overlap_ratio: float = 0.0,  # only for mode="overlap": fraction of train reused across clients
    category_key: Optional[str] = None,  # only for mode="category": e.g., "label", "genre", or custom column
    bin_stsb: bool = True,       # for STS-B score binning if category_key is "labels" (regression)
) -> Dict[str, Any]:
    """
    Create federated learning data splits for a GLUE task's training set.
    
    This function partitions the training data of a GLUE benchmark task across
    k clients using one of three strategies:
    
    1. **IID Mode (Independent and Identically Distributed)**:
       - Stratified random sampling maintaining label distribution
       - Each client gets proportional representation of all classes
       - Best for simulating homogeneous federated learning scenarios
       
    2. **Overlap Mode**:
       - Clients share a fraction of common training examples
       - Useful for studying convergence with data redundancy
       - Controlled by overlap_ratio parameter
       
    3. **Category Mode (Non-IID)**:
       - Entire categories/labels assigned to specific clients
       - Creates heterogeneous data distribution across clients
       - Simulates real-world federated scenarios with biased local data
    
    Args:
        task_name: GLUE task name (e.g., "sst2", "mnli", "qqp", "stsb")
        k: Number of federated learning clients
        mode: Sharding strategy - "iid", "overlap", or "category"
        seed: Random seed for reproducibility
        overlap_ratio: [overlap mode only] Fraction of data shared across clients (0.0 to 1.0)
        category_key: [category mode only] Column to partition by (e.g., "label", "genre")
        bin_stsb: Whether to bin STS-B regression scores into discrete categories
        
    Returns:
        Dictionary containing:
          - "client_indices": List of k lists, each containing dataset indices for that client
          - "size_per_client": Number of examples per client
          - "total_train": Total training examples in original dataset
          - "task": Task name
          - "mode": Sharding mode used
          - "meta": Additional metadata (seed, overlap_ratio, category_key, bin_stsb)
          
    Example:
        >>> splits = make_client_splits("sst2", k=5, mode="iid", seed=42)
        >>> print(f"Client 0 has {len(splits['client_indices'][0])} examples")
        >>> # Use splits['client_indices'][0] to select data for first client
    """
    # Load the GLUE dataset for the specified task
    task = task_name.lower()
    raw = load_dataset("glue", task)
    train = raw["train"]

    n = len(train)
    rng = _rng(seed)

    # ========================================================================
    # Mode 1: IID (Stratified Random Sampling)
    # ========================================================================
    if mode == "iid":
        # Group data by stratification key to maintain label distribution
        # Different GLUE tasks have different column structures
        
        if task == "mnli":
            # MNLI (Multi-Genre NLI): Stratify by both label and genre
            # This ensures each client gets examples from all genres and labels
            if "genre" in train.column_names:
                # Create composite key: "label|genre" (e.g., "entailment|fiction")
                comp = [f"{lbl}|{gen}" for lbl, gen in zip(train["label"], train["genre"])]
                buckets = _indices_by_key({"comp": comp}, "comp")
            else:
                # Fallback to label-only stratification
                buckets = _indices_by_key({"label": train["label"]}, "label")
                
        elif task in {"mrpc", "qqp", "qnli", "rte", "sst2", "cola", "wnli"}:
            # Standard classification tasks: stratify by label
            if "label" in train.column_names:
                buckets = _indices_by_key({"label": train["label"]}, "label")
            else:
                # No labels found, use all data as single bucket
                buckets = {"_all": list(range(n))}
                
        elif task == "stsb":
            # STS-B (Semantic Textual Similarity): regression task with scores [0, 5]
            # Bin continuous scores into discrete buckets for stratification
            scores = train["label"] if "label" in train.column_names else train["score"]
            bins = _bin_scores(scores, bin_width=0.2) if bin_stsb else scores
            buckets = _indices_by_key({"bin": bins}, "bin")
        else:
            # Unknown task: no stratification
            buckets = {"_all": list(range(n))}

        # Distribute examples from each bucket across clients using round-robin
        # This ensures balanced label distribution across all clients
        client_indices = [[] for _ in range(k)]
        for _, idxs in buckets.items():
            rng.shuffle(idxs)  # Randomize within bucket
            # Round-robin assignment: bucket[0]→client0, bucket[1]→client1, ..., bucket[k]→client0, ...
            for i, idx in enumerate(idxs):
                client_indices[i % k].append(idx)

        # Final shuffle within each client to mix examples from different buckets
        for ci in client_indices:
            rng.shuffle(ci)

    # ========================================================================
    # Mode 2: Overlap (Partially Shared Data)
    # ========================================================================
    elif mode == "overlap":
        # Validate overlap ratio
        assert 0.0 <= overlap_ratio < 1.0, "overlap_ratio must be in [0,1)"
        
        # Calculate split sizes
        # Each client gets: (1-overlap_ratio)/k unique data + overlap_ratio/k shared data
        rng.shuffle_indices = list(range(n))
        base_size = int((1.0 - overlap_ratio) * n / k)  # Size of unique portion per client
        
        # Create pool for overlapping examples (can be sampled multiple times)
        overlap_pool = list(range(n))
        rng.shuffle(overlap_pool)
        
        client_indices = []
        used = set()  # Track unique indices already assigned

        start = 0
        for _ in range(k):
            # Part 1: Assign unique chunk to this client (no overlap with other clients)
            unique_chunk = []
            while len(unique_chunk) < base_size and start < n:
                idx = rng.randrange(n)
                if idx not in used:
                    unique_chunk.append(idx)
                    used.add(idx)  # Mark as used so no other client gets it
                    start += 1
                    
            # Part 2: Add overlap chunk (randomly sampled from entire dataset)
            # These examples may appear in multiple clients' data
            overlap_sz = int(overlap_ratio * n / k)
            overlap_chunk = [overlap_pool[rng.randrange(n)] for __ in range(overlap_sz)]
            
            # Combine unique and overlap portions, remove duplicates within client
            shard = list(set(unique_chunk + overlap_chunk))
            rng.shuffle(shard)
            client_indices.append(shard)

    # ========================================================================
    # Mode 3: Category (Non-IID by Label/Genre)
    # ========================================================================
    elif mode == "category":
        # Validate required parameter
        assert category_key is not None, "Provide category_key for category mode."
        
        # Determine how to group examples by category
        key = category_key
        if key == "labels" and task == "stsb" and bin_stsb:
            # Special case: STS-B with binned regression scores
            scores = train["label"] if "label" in train.column_names else train["score"]
            bins = _bin_scores(scores, bin_width=0.2)
            buckets = _indices_by_key({"bin": bins}, "bin")
        else:
            # General case: use specified column as category
            if key not in train.column_names:
                raise ValueError(f"{key} not found in train columns: {train.column_names}")
            buckets = _indices_by_key(train, key)

        # Assign entire categories to clients in round-robin fashion
        # This creates highly non-IID data: each client specializes in specific categories
        # Example: Client 0 gets all "positive" sentiment, Client 1 gets all "negative"
        cats = list(buckets.keys())
        rng.shuffle(cats)  # Randomize category assignment order
        client_indices = [[] for _ in range(k)]
        for i, c in enumerate(cats):
            # Category i goes to client (i % k)
            client_indices[i % k].extend(buckets[c])
        
        # Shuffle examples within each client
        for ci in client_indices:
            rng.shuffle(ci)

    else:
        raise ValueError("mode must be one of: iid | overlap | category")

    # ========================================================================
    # Return Shard Manifest
    # ========================================================================
    sizes = [len(ci) for ci in client_indices]
    return {
        "client_indices": client_indices,    # List[List[int]]: indices for each client
        "size_per_client": sizes,             # List[int]: number of examples per client
        "total_train": n,                     # int: total training examples
        "task": task,                         # str: task name
        "mode": mode,                         # str: sharding mode used
        "meta": {                             # Additional configuration metadata
            "seed": seed,
            "overlap_ratio": overlap_ratio,
            "category_key": category_key,
            "bin_stsb": bin_stsb,
        }
    }

# ============================================================================
# Persistence Function
# ============================================================================

def save_shard_manifests(out_dir: str, shard_info: Dict[str, Any]):
    """
    Save client data splits to JSON files for later use in federated training.
    
    This function persists the output of make_client_splits() to disk, creating:
    - One JSON file per client containing their assigned indices
    - A summary JSON file with complete sharding information
    
    Args:
        out_dir: Directory path where manifest files will be saved
        shard_info: Output dictionary from make_client_splits()
        
    Output Files:
        - {out_dir}/client_1.json: Indices for first client
        - {out_dir}/client_2.json: Indices for second client
        - ...
        - {out_dir}/client_k.json: Indices for k-th client
        - {out_dir}/summary.json: Complete sharding metadata and statistics
        
    Example:
        >>> splits = make_client_splits("sst2", k=5, mode="iid")
        >>> save_shard_manifests("./data/sst2_splits", splits)
        >>> # Creates: client_1.json, client_2.json, ..., client_5.json, summary.json
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract dataset/task name for logging
    dataset_name = shard_info.get("task", "unknown")
    k = len(shard_info["client_indices"])
    
    print(f"\n{'='*60}")
    print(f"Saving Shard Manifests - Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Number of clients: {k}")
    print(f"Sharding mode: {shard_info.get('mode', 'unknown')}")
    print(f"Output directory: {out_dir}")
    
    # Save individual client manifest files
    # Each file contains the dataset indices assigned to that client
    for i, idxs in enumerate(shard_info["client_indices"]):
        client_file = f"{out_dir}/client_{i+1}.json"
        with open(client_file, "w") as f:
            json.dump({"dataset": dataset_name, "indices": idxs}, f)
        print(f"  Client {i+1}: {len(idxs)} samples → {client_file}")
    
    # Save summary file with complete sharding information
    # Includes: all indices, sizes, task info, mode, and metadata
    summary_file = f"{out_dir}/summary.json"
    with open(summary_file, "w") as f:
        json.dump(shard_info, f, indent=2)
    print(f"  Summary: {summary_file}")
    print(f"{'='*60}")

# DFL_SLM: Federated Learning with Small Language Models (RoBERTa + LoRA)

This repository implements **Federated Averaging (FedAvg)** for fine-tuning RoBERTa models on GLUE benchmark tasks using **Low-Rank Adaptation (LoRA)**. The core research contribution is the aggregation strategy for LoRA adapters across federated clients, supporting both delta-weight averaging with SVD compression and direct A/B matrix averaging.

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [1. Environment Creation](#1-environment-creation)
  - [2. Install Requirements](#2-install-requirements)
  - [3. Download and Cache RoBERTa Models](#3-download-and-cache-roberta-models)
- [Preparing Data Shards for Federated Learning](#preparing-data-shards-for-federated-learning)
- [Running Experiments](#running-experiments)
  - [Baseline: Standard RoBERTa Fine-tuning](#baseline-standard-roberta-fine-tuning)
  - [Standard LoRA Fine-tuning](#standard-lora-fine-tuning)
  - [Multi-Round Federated Averaging (FedAvg)](#multi-round-federated-averaging-fedavg)
  - [Single-Shot FedAvg Merge](#single-shot-fedavg-merge)
- [Understanding the Aggregation Methods](#understanding-the-aggregation-methods)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

This project explores **parameter-efficient federated learning** by combining:

- **LoRA (Low-Rank Adaptation)**: Fine-tunes only small adapter matrices instead of full model weights, reducing trainable parameters by >99%
- **FedAvg**: Aggregates local client updates into a global model through weighted averaging
- **SVD Compression**: Optionally compresses averaged adapters back to rank-r using Singular Value Decomposition

### Key Features

✅ Modular implementation supporting all 9 GLUE tasks  
✅ Three data sharding strategies (IID, overlap, non-IID category-based)  
✅ Two aggregation approaches: ΔW averaging + SVD or naive A/B matrix averaging  
✅ Multi-round federated training with optional warm-start from pre-trained adapters  
✅ Per-client and global model evaluation  

---

## Setup

### 1. Environment Creation

Create a Python 3.11.4 virtual environment:

```bash
# Using pyenv (recommended)
pyenv install 3.11.4
pyenv local 3.11.4
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n dfl_slm python=3.11.4
conda activate dfl_slm
```

### 2. Install Requirements

Install all dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies include:**
- `transformers>=4.45.0` - HuggingFace Transformers for RoBERTa models
- `datasets>=2.19.0` - GLUE benchmark datasets
- `peft>=0.11.1` - Parameter-Efficient Fine-Tuning (LoRA)
- `accelerate>=0.34.2` - Distributed training support
- `evaluate>=0.4.2` - Evaluation metrics
- `scikit-learn>=1.4.0` - ML utilities
- `scipy>=1.11.0` - Scientific computing (SVD)

### 3. Download and Cache RoBERTa Models

Before running experiments, download and cache RoBERTa models locally to avoid repeated downloads:

```bash
cd src/models
python download_roberta_models.py --hf_home ./hf_cache
```

**Arguments:**
- `--hf_home`: (Optional) Custom cache directory for HuggingFace models. If not specified, uses default HF cache location.
- `--revision`: (Optional) Model revision/branch/tag to download (default: `main`).

This will download:
- `roberta-base` (~125M parameters)
- `roberta-large` (~355M parameters)

**Note:** All subsequent training scripts will automatically use these cached models.

---

## Preparing Data Shards for Federated Learning

Before running federated experiments, you must partition the training data across K clients using the `sharding.py` module.

### Example: Create 5 IID Client Shards for SST2

```bash
cd src/main
python - <<'PY'
import os
from sharding import make_client_splits, save_shard_manifests

K = 5
out_dir = f"out/sst2_base/shards_iid_k{K}"
info = make_client_splits(task_name="sst2", k=K, mode="iid", seed=123)
save_shard_manifests(out_dir, info)
print("Client sizes:", info["size_per_client"])
PY
```

This creates:
- `out/sst2_base/shards_iid_k5/client_1.json` - Indices for client 1
- `out/sst2_base/shards_iid_k5/client_2.json` - Indices for client 2
- ... (one file per client)
- `out/sst2_base/shards_iid_k5/summary.json` - Complete sharding metadata

### Sharding Modes

#### 1. **IID (Independent and Identically Distributed)**
Stratified random splits maintaining label distribution across all clients.

```python
info = make_client_splits(task_name="sst2", k=5, mode="iid", seed=42)
```

#### 2. **Overlap**
Clients share a fraction of common training examples (controlled by `overlap_ratio`).

```python
info = make_client_splits(task_name="sst2", k=5, mode="overlap", 
                          overlap_ratio=0.2, seed=42)
```

#### 3. **Category (Non-IID)**
Entire categories/labels assigned to specific clients, creating heterogeneous data distribution.

```python
info = make_client_splits(task_name="sst2", k=5, mode="category", 
                          category_key="label", seed=42)
```

### GLUE Tasks Supported

- **cola**: Linguistic acceptability (2 classes)
- **sst2**: Sentiment analysis (2 classes)
- **mrpc**: Paraphrase detection (2 classes)
- **stsb**: Semantic similarity (regression)
- **qqp**: Question pair similarity (2 classes)
- **mnli**: Natural language inference (3 classes, multi-genre)
- **qnli**: Question-answering NLI (2 classes)
- **rte**: Textual entailment (2 classes)
- **wnli**: Winograd NLI (2 classes)

---

## Running Experiments

All commands below assume you're in the `src/main/` directory:

```bash
cd src/main
```

### Baseline: Standard RoBERTa Fine-tuning

Fine-tune full RoBERTa model **without LoRA** (all parameters trainable).

```bash
python baseline_run.py \
  --model_size base \
  --task_name sst2 \
  --output_dir ./out_baseline \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --seed 42
```

**Key Arguments:**
- `--model_size`: `base` (~125M) or `large` (~355M)
- `--task_name`: Any GLUE task (use `sst2` for quick testing)
- `--do_train`: Enable training
- `--do_eval`: Enable evaluation (zero-shot if `--do_train` omitted)
- `--learning_rate`: Typical for full fine-tuning is `2e-5` (lower than LoRA)

**Expected Output:**
- Model checkpoints: `./out_baseline/sst2_base/`
- Evaluation metrics printed to console

---

### Standard LoRA Fine-tuning

Fine-tune RoBERTa using **LoRA adapters** (parameter-efficient, <1% trainable params).

```bash
python main_glue_lora.py \
  --model_size base \
  --task_name sst2 \
  --output_dir ./out \
  --do_train \
  --do_eval \
  --lora_r 8 \
  --lora_alpha 8 \
  --lora_dropout 0.0 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 3.0 \
  --seed 42
```

**LoRA Hyperparameters:**
- `--lora_r`: Rank of LoRA matrices (default: 8). Lower = fewer parameters.
- `--lora_alpha`: Scaling factor (default: 8). Typical practice: `alpha = r`.
- `--lora_dropout`: Dropout probability for LoRA layers (default: 0.0).
- `--learning_rate`: Typical for LoRA is `2e-4` (higher than full fine-tuning).

**Advanced Options:**
- `--adapter_output_dir`: Custom directory to save trained adapter. Default: `./out/{task}_{model_size}/r{lora_r}/adapter/`

**Expected Output:**
- Adapter weights: `./out/sst2_base/r8/adapter/`
- Evaluation metrics and manifest: `./out/sst2_base/r8/manifest.txt`

---

### Multi-Round Federated Averaging (FedAvg)

Run **multiple rounds** of federated learning: broadcast → local train → aggregate → repeat.

#### Step 1: Create Client Shards (if not done already)

```bash
python - <<'PY'
import os
from sharding import make_client_splits, save_shard_manifests
K = 3
out_dir = f"out/sst2_base/shards_iid_k{K}"
info = make_client_splits(task_name="sst2", k=K, mode="iid", seed=123)
save_shard_manifests(out_dir, info)
print("sizes:", info["size_per_client"])
PY
```

#### Step 2: Run Multi-Round FedAvg

**Example 1: FedAvg with SVD Compression (default)**

```bash
python fedavg_rounds.py \
  --model_size base \
  --task_name sst2 \
  --shard_dir ./out/sst2_base/shards_iid_k3 \
  --rounds 5 \
  --local_epochs 1.0 \
  --batch_size 32 \
  --learning_rate 2e-4 \
  --lora_r 8 \
  --lora_alpha 8 \
  --lora_dropout 0.0 \
  --svd_rank 8 \
  --output_root ./out_fedavg \
  --eval_each_round \
  --eval_batch_size 32 \
  --seed 42
```

**Example 2: Naive FedAvg (average A/B matrices directly, skip SVD)**

```bash
python fedavg_rounds.py \
  --model_size base \
  --task_name sst2 \
  --shard_dir ./out/sst2_base/shards_iid_k3 \
  --rounds 5 \
  --local_epochs 1.0 \
  --batch_size 32 \
  --learning_rate 2e-4 \
  --lora_r 8 \
  --lora_alpha 8 \
  --lora_dropout 0.0 \
  --skip_svd \
  --output_root ./out_fedavg \
  --eval_each_round \
  --seed 42
```

**Example 3: Warm-Start from Pre-Trained Adapter**

```bash
python fedavg_rounds.py \
  --model_size base \
  --task_name sst2 \
  --shard_dir ./out/sst2_base/shards_iid_k3 \
  --rounds 5 \
  --local_epochs 1.0 \
  --init_adapter ./out/sst2_base/r8/adapter \
  --lora_r 8 \
  --lora_alpha 8 \
  --svd_rank 8 \
  --output_root ./out_fedavg \
  --eval_each_round \
  --seed 42
```

**Key Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_size` | `base` or `large` | Required |
| `--task_name` | GLUE task (e.g., `sst2`) | Required |
| `--shard_dir` | Directory with `client_#.json` manifests | Required |
| `--rounds` | Number of federated rounds | 5 |
| `--local_epochs` | Training epochs per client per round | 1.0 |
| `--batch_size` | Batch size for training | 32 |
| `--learning_rate` | Learning rate | 2e-4 |
| `--lora_r` | LoRA rank for local training | 8 |
| `--lora_alpha` | LoRA alpha for local training | 8 |
| `--lora_dropout` | LoRA dropout | 0.0 |
| `--svd_rank` | Rank for SVD compression after aggregation | 8 |
| `--alpha` | Alpha for fused adapter (defaults to `--svd_rank`) | None |
| `--target_modules` | LoRA target modules | `["query", "value"]` |
| `--skip_svd` | Skip SVD; directly average A/B matrices | False |
| `--init_adapter` | Optional starting adapter for round 1 | None |
| `--output_root` | Root directory for outputs | `./out_fedavg` |
| `--eval_each_round` | Evaluate fused model after each round | False |
| `--eval_clients` | Evaluate individual clients after training | False |
| `--eval_batch_size` | Batch size for evaluation | 32 |
| `--seed` | Random seed | 42 |

**Output Structure:**

```
out_fedavg/
└── sst2_base/
    ├── round_1/
    │   ├── client_1/
    │   │   ├── adapter/              # Client 1's trained adapter
    │   │   ├── base_model.pt         # Client 1's base model + classifier
    │   │   └── ckpt/                 # Training checkpoints
    │   ├── client_2/
    │   ├── client_3/
    │   └── fused_r8/                 # Aggregated global model
    │       ├── adapter/              # Fused adapter
    │       ├── base_model.pt         # Fused base model + averaged classifier
    │       └── eval_tmp/             # Evaluation logs (if --eval_each_round)
    ├── round_2/
    │   └── ...
    └── round_5/
```

**Important Notes:**

1. **Classification Head Averaging**: The script now averages both LoRA adapters AND classification head weights across clients.
2. **SVD vs. Naive Averaging**: 
   - Default (without `--skip_svd`): Compute full ΔW = scaling × (B @ A), average across clients, then SVD back to rank `--svd_rank`.
   - With `--skip_svd`: Directly average A matrices and B matrices separately (naive FedAvg).
3. **Warm-Start**: Use `--init_adapter` to continue training from a pre-trained adapter (e.g., centralized baseline).

---

### Single-Shot FedAvg Merge

Aggregate **already-trained** adapters from K clients into a single global adapter (one-shot, no iterative rounds).

#### Use Case
You already have K trained adapters (e.g., from independent LoRA runs) and want to merge them using FedAvg.

#### Example: Merge 3 Pre-Trained Adapters

```bash
python fedavg_merge.py \
  --model_size base \
  --task_name sst2 \
  --adapters \
    ./client1/adapter \
    ./client2/adapter \
    ./client3/adapter \
  --shard_manifests \
    ./out/sst2_base/shards_iid_k3/client_1.json \
    ./out/sst2_base/shards_iid_k3/client_2.json \
    ./out/sst2_base/shards_iid_k3/client_3.json \
  --svd_rank 8 \
  --alpha 8 \
  --save_adapter ./out_merged/sst2_fused_adapter
```

**Key Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_size` | `base` or `large` | Required |
| `--task_name` | GLUE task name (for logging) | `glue` |
| `--adapters` | List of adapter directories to merge | Required |
| `--weights` | Manual weights for each adapter (optional) | None |
| `--shard_manifests` | JSON files with client indices (for auto-weighting) | None |
| `--svd_rank` | Rank for SVD compression | Required |
| `--alpha` | Alpha for fused adapter (defaults to `--svd_rank`) | None |
| `--save_adapter` | Output directory for fused adapter | Required |

**Weighting Options:**

1. **Automatic (from shard manifests)**: Uses sample count per client
   ```bash
   --shard_manifests client_1.json client_2.json client_3.json
   ```

2. **Manual weights**: Explicitly specify weights
   ```bash
   --weights 0.3 0.5 0.2
   ```

3. **Equal weights** (default if neither provided): All clients weighted equally

**Expected Output:**
- Fused adapter: `./out_merged/sst2_fused_adapter/`
- Manifest: `./out_merged/sst2_fused_adapter/fuse_manifest.json`

---

## Understanding the Aggregation Methods

### Method 1: ΔW Averaging + SVD Compression (Default)

**Algorithm:**

1. For each client adapter, compute full delta weight: ΔW = scaling × (B @ A)
2. Weighted average: ΔW_global = Σ(w_i × ΔW_i)
3. SVD factorize: ΔW_global ≈ U_r Σ_r V_r^T
4. Reconstruct rank-r adapter: B_new = U_r Σ_r, A_new = V_r^T

**Advantages:**
- Compresses to lower rank (e.g., clients train at r=16, aggregate to r=8)
- More stable when clients have different ranks
- Better theoretical alignment with FedAvg

**Usage:**
```bash
python fedavg_rounds.py ... --svd_rank 8  # Default behavior
```

---

### Method 2: Naive A/B Matrix Averaging (Skip SVD)

**Algorithm:**

1. For each client adapter, extract A and B matrices directly
2. Weighted average separately: A_global = Σ(w_i × A_i), B_global = Σ(w_i × B_i)
3. Use averaged matrices directly (no SVD)

**Advantages:**
- Faster (no SVD computation)
- Preserves exact rank
- Simpler implementation

**Disadvantages:**
- Cannot compress rank (clients and global must have same r)
- May be less stable with heterogeneous clients

**Usage:**
```bash
python fedavg_rounds.py ... --skip_svd  # No --svd_rank needed
```

---

## Project Structure

```
DFL_SLM/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── src/
│   ├── main/
│   │   ├── __init__.py
│   │   ├── baseline_run.py        # Standard RoBERTa fine-tuning (no LoRA)
│   │   ├── main_glue_lora.py      # Standard LoRA fine-tuning
│   │   ├── fedavg_rounds.py       # Multi-round federated learning
│   │   ├── fedavg_merge.py        # Single-shot adapter merging
│   │   ├── sharding.py            # Data partitioning for FL clients
│   │   ├── data_prep.py           # GLUE data loading and tokenization
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── models.py              # Model construction utilities
│   │   ├── utils.py               # Misc utilities (seeds, manifests)
│   │   └── out/                   # Default output directory
│   └── models/
│       ├── download_roberta_models.py  # Pre-download RoBERTa models
│       └── hf_cache/              # Local HuggingFace cache
└── ...
```

---

## Tips & Best Practices

### 1. **Start Small for Debugging**
Use SST2 (small dataset) with 2-3 clients for initial testing:
```bash
K=2
info = make_client_splits(task_name="sst2", k=K, mode="iid", seed=42)
```

### 2. **Choosing LoRA Rank**
- `r=4`: Very parameter-efficient, may sacrifice accuracy
- `r=8`: Good balance (recommended starting point)
- `r=16`: Higher accuracy, more parameters
- `r=32`: Approaching full fine-tuning performance

### 3. **SVD Rank Strategy**
You can train clients at higher rank and compress during aggregation:
```bash
# Clients train at r=16, aggregate to r=8
python fedavg_rounds.py --lora_r 16 --svd_rank 8 ...
```

### 4. **Evaluating Convergence**
Use `--eval_each_round` to track global model performance across rounds:
```bash
python fedavg_rounds.py --eval_each_round --eval_clients ...
```

### 5. **Data Heterogeneity Experiments**
Compare IID vs. non-IID:
```python
# IID (homogeneous)
make_client_splits(task_name="sst2", k=5, mode="iid")

# Non-IID (heterogeneous)
make_client_splits(task_name="sst2", k=5, mode="category", category_key="label")
```

### 6. **Warm-Starting from Centralized Training**
First train a centralized baseline, then use it to initialize federated learning:
```bash
# Step 1: Train centralized baseline
python main_glue_lora.py --task_name sst2 --do_train --output_dir ./out

# Step 2: Use as warm-start for FedAvg
python fedavg_rounds.py --init_adapter ./out/sst2_base/r8/adapter ...
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solutions:**
- Reduce `--batch_size` (e.g., 16 or 8)
- Use `--model_size base` instead of `large`
- Reduce `--lora_r` to lower rank

### Issue: "No module named 'transformers'"
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: "Dataset not found"
**Solution:** Ensure internet connection for first run (datasets auto-download from HuggingFace).

### Issue: "Adapter path does not exist"
**Solution:** Train adapters first using `main_glue_lora.py` before merging with `fedavg_merge.py`.


#!/usr/bin/env python
# src/main/fedavg_rounds.py
# Multi-round FedAvg over LoRA ΔW: broadcast -> local train -> FedAvg -> repeat.

from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# --- reuse helpers from your project if you prefer ---
from data_prep import load_and_tokenize
from metrics import build_metric_fn, evaluate_mnli_overall
from models import resolve_model_id

GLUE_FIELDS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def _collect_lora_deltas(model: PeftModel) -> Dict[str, torch.Tensor]:
    deltas = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if "default" in module.lora_A.keys():
                A = module.lora_A["default"].weight.data
                B = module.lora_B["default"].weight.data
                if hasattr(module, "scaling"):
                    scaling = module.scaling["default"]
                else:
                    alpha = module.lora_alpha["default"]
                    r = module.r["default"]
                    scaling = alpha / r
                delta = (B @ A) * scaling
                deltas[name] = delta.detach().float().cpu()
    return deltas

def _weighted_average(all_deltas: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    wsum = sum(weights); w = [wi/wsum for wi in weights]
    names = all_deltas[0].keys()
    avg = {}
    for n in names:
        acc = None
        for d, wi in zip(all_deltas, w):
            t = d[n] * wi
            acc = t if acc is None else (acc + t)
        avg[n] = acc
    return avg

def _collect_lora_params(model: PeftModel) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Collect raw A and B matrices (without computing delta)"""
    params = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if "default" in module.lora_A.keys():
                A = module.lora_A["default"].weight.data.detach().float().cpu()
                B = module.lora_B["default"].weight.data.detach().float().cpu()
                params[name] = (A, B)
    return params

def _average_lora_params(all_params: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], 
                         weights: List[float]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Average A and B matrices separately (naive LoRA FedAvg)"""
    wsum = sum(weights); w = [wi/wsum for wi in weights]
    names = all_params[0].keys()
    avg = {}
    for n in names:
        A_acc, B_acc = None, None
        for params, wi in zip(all_params, w):
            A, B = params[n]
            A_weighted = A * wi
            B_weighted = B * wi
            A_acc = A_weighted if A_acc is None else (A_acc + A_weighted)
            B_acc = B_weighted if B_acc is None else (B_acc + B_weighted)
        avg[n] = (A_acc, B_acc)
    return avg

def _svd_factorize_to_lora(M: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(r, U.shape[1], Vh.shape[0])
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
    B = U_r @ torch.diag(S_r)   # (out x r)
    A = Vh_r                    # (r x in)
    return B, A

def _build_base(model_id: str, num_labels: int, is_reg: bool):
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.num_labels = 1 if is_reg else num_labels
    cfg.problem_type = "regression" if is_reg else "single_label_classification"
    return AutoModelForSequenceClassification.from_pretrained(model_id, config=cfg)

def _attach_new_lora(base, r: int, alpha: int, dropout: float, target_modules: List[str]):
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none"
    )
    return get_peft_model(base, cfg)

def _train_one_client(
    model_id: str,
    task: str,
    indices: List[int],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    init_adapter: Optional[str],  # fused adapter from prev round or None
    target_modules: List[str],
    epochs: float,
    bs: int,
    lr: float,
    seed: int,
    out_dir: str
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # Data
    tokenized, num_labels, is_reg, tok = load_and_tokenize(task, model_id, max_length=128)
    train_subset = tokenized["train"].select(indices)
    eval_ds = tokenized["validation_mismatched"] if task=="mnli" else tokenized["validation"]

    # Model
    base = _build_base(model_id, num_labels, is_reg)
    if init_adapter:
        model = PeftModel.from_pretrained(base, init_adapter)
    else:
        model = _attach_new_lora(base, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target_modules=target_modules)

    # Train LoRA only
    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "ckpt"),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        eval_strategy="no",
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_subset, eval_dataset=None, tokenizer=tok)
    trainer.train()

    # Save adapter
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    return adapter_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", choices=["base","large"], required=True)
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--shard_dir", required=True, help="Dir with client_#.json manifests (indices per client).")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local_epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=2e-4)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--target_modules", nargs="*", default=["query","value"])

    ap.add_argument("--svd_rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=None, help="Alpha for fused adapter; default=r.")
    ap.add_argument("--init_adapter", type=str, default=None, help="Optional starting adapter (round 1 broadcast).")
    ap.add_argument("--skip_svd", action="store_true", help="Skip SVD compression; directly average A/B matrices (naive LoRA FedAvg).")

    ap.add_argument("--output_root", type=str, default="./out_fedavg")
    ap.add_argument("--eval_each_round", action="store_true")
    ap.add_argument("--eval_clients", action="store_true", help="Evaluate each client individually after training.")
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_id = resolve_model_id(args.model_size)
    alpha_fused = args.alpha if args.alpha is not None else args.svd_rank

    # Load shard manifests
    client_files = sorted([f for f in os.listdir(args.shard_dir) if f.startswith("client_") and f.endswith(".json")],
                          key=lambda x: int(x.split("_")[1].split(".")[0]))
    K = len(client_files)
    clients = []
    sizes = []
    for f in client_files:
        with open(os.path.join(args.shard_dir, f), "r") as fh:
            data = json.load(fh)
        idxs = data["indices"]
        clients.append(idxs)
        sizes.append(len(idxs))
    print(f"🌱 Loaded {K} clients. Sizes: {sizes}")

    # Optional: tokenization once for eval
    tokenized_all, num_labels, is_reg, tok = load_and_tokenize(args.task_name, model_id, max_length=128)

    fused_prev: Optional[str] = args.init_adapter  # adapter path to broadcast
    for r in range(1, args.rounds + 1):
        print(f"\n===== ROUND {r}/{args.rounds} =====")

        round_dir = os.path.join(args.output_root, f"{args.task_name}_{args.model_size}", f"round_{r}")
        os.makedirs(round_dir, exist_ok=True)

        # Local training per client
        adapter_paths = []
        for i, idxs in enumerate(clients, start=1):
            c_out = os.path.join(round_dir, f"client_{i}")
            apath = _train_one_client(
                model_id=model_id,
                task=args.task_name,
                indices=idxs,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                init_adapter=fused_prev,
                target_modules=args.target_modules,
                epochs=args.local_epochs,
                bs=args.batch_size,
                lr=args.learning_rate,
                seed=args.seed,
                out_dir=c_out
            )
            adapter_paths.append(apath)
            print(f"  ✅ client {i} adapter: {apath}")
            
            # Optional: evaluate individual client
            if args.eval_clients:
                is_mnli = (args.task_name.lower() == "mnli")
                eval_ds = tokenized_all["validation_mismatched"] if is_mnli else tokenized_all["validation"]
                from transformers import TrainingArguments
                targs = TrainingArguments(output_dir=os.path.join(c_out, "eval_tmp"),
                                          per_device_eval_batch_size=args.eval_batch_size,
                                          do_eval=True, report_to="none")
                client_model = PeftModel.from_pretrained(_build_base(model_id, num_labels, is_reg), apath)
                trainer = Trainer(model=client_model, args=targs, eval_dataset=eval_ds, tokenizer=tok,
                                  compute_metrics=build_metric_fn(args.task_name))
                if is_mnli:
                    res = evaluate_mnli_overall(trainer, tokenized_all)
                else:
                    res = trainer.evaluate()
                print(f"     📊 Client {i} metrics:", {k: float(v) if hasattr(v, "__float__") else v for k, v in res.items()})
                del client_model, trainer

        # FedAvg aggregation
        base_tmp = _build_base(model_id, num_labels, is_reg)
        weights = sizes  # sample-count weighting
        
        if args.skip_svd:
            # Naive LoRA FedAvg: directly average A and B matrices
            print("  🔀 Aggregating via naive LoRA averaging (no SVD)...")
            all_params = []
            for apath in adapter_paths:
                m = PeftModel.from_pretrained(base_tmp, apath)
                all_params.append(_collect_lora_params(m))
                del m
            fused_params = _average_lora_params(all_params, weights)
            
            # Build fused model and assign averaged A/B
            fused_dir = os.path.join(round_dir, f"fused_r{args.lora_r}_nosvd")
            os.makedirs(fused_dir, exist_ok=True)
            
            base_write = _build_base(model_id, num_labels, is_reg)
            fused_model = _attach_new_lora(base_write, r=args.lora_r, alpha=args.lora_alpha,
                                           dropout=0.0, target_modules=args.target_modules)
            
            for name, module in fused_model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "default" in module.lora_A.keys():
                    if name not in fused_params:
                        continue
                    A_avg, B_avg = fused_params[name]
                    with torch.no_grad():
                        module.lora_A["default"].weight.copy_(A_avg.to(module.lora_A["default"].weight.dtype))
                        module.lora_B["default"].weight.copy_(B_avg.to(module.lora_B["default"].weight.dtype))
        else:
            # Original: compute full ΔW, average, then SVD back to rank-r
            print("  🔀 Aggregating via ΔW averaging + SVD compression...")
            all_d = []
            for apath in adapter_paths:
                m = PeftModel.from_pretrained(base_tmp, apath)
                all_d.append(_collect_lora_deltas(m))
                del m
            fused = _weighted_average(all_d, weights)

            # Build fused skeleton and assign SVD(B@A) with new scaling
            fused_dir = os.path.join(round_dir, f"fused_r{args.svd_rank}")
            os.makedirs(fused_dir, exist_ok=True)
            scaling_new = alpha_fused / args.svd_rank

            base_write = _build_base(model_id, num_labels, is_reg)
            fused_model = _attach_new_lora(base_write, r=args.svd_rank, alpha=alpha_fused,
                                           dropout=0.0, target_modules=args.target_modules)

            for name, module in fused_model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "default" in module.lora_A.keys():
                    if name not in fused:  # skip non-LoRA modules
                        continue
                    M = fused[name] / scaling_new
                    B, A = _svd_factorize_to_lora(M, r=args.svd_rank)
                    with torch.no_grad():
                        module.lora_A["default"].weight.copy_(A.to(module.lora_A["default"].weight.dtype))
                        module.lora_B["default"].weight.copy_(B.to(module.lora_B["default"].weight.dtype))

        fused_model.save_pretrained(fused_dir)
        fused_prev = fused_dir  # broadcast next round
        print(f"  🔗 fused adapter saved: {fused_dir}")

        # Optional dev eval (quick)
        if args.eval_each_round:
            is_mnli = (args.task_name.lower() == "mnli")
            eval_ds = tokenized_all["validation_mismatched"] if is_mnli else tokenized_all["validation"]
            from transformers import TrainingArguments
            targs = TrainingArguments(output_dir=os.path.join(fused_dir, "eval_tmp"),
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      do_eval=True, report_to="none")
            eval_model = PeftModel.from_pretrained(_build_base(model_id, num_labels, is_reg), fused_dir)
            trainer = Trainer(model=eval_model, args=targs, eval_dataset=eval_ds, tokenizer=tok,
                              compute_metrics=build_metric_fn(args.task_name))
            if is_mnli:
                res = evaluate_mnli_overall(trainer, tokenized_all)
            else:
                res = trainer.evaluate()
            print("  📊 dev metrics:", {k: float(v) if hasattr(v, "__float__") else v for k, v in res.items()})

    print("\n✅ Multi-round FedAvg completed.")

if __name__ == "__main__":
    main()

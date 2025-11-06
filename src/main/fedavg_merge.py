#!/usr/bin/env python
# src/main/fedavg_merge.py
# Weighted FedAvg over LoRA ΔW across k adapters; optional SVD back to rank r.

from __future__ import annotations
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from models import resolve_model_id

def _collect_lora_deltas(model: PeftModel) -> Dict[str, torch.Tensor]:
    """
    For each LoRA target linear module (e.g., query/value), compute
    ΔW = scaling * (B @ A). Returns dict[name] = ΔW (cpu float32).
    """
    deltas: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if "default" in getattr(module, "lora_A").keys():
                A = module.lora_A["default"].weight.data  # [r, in]
                B = module.lora_B["default"].weight.data  # [out, r]
                # scaling could be stored in module.scaling or derived from alpha/r
                if hasattr(module, "scaling"):
                    scaling = module.scaling["default"]
                else:
                    alpha = module.lora_alpha["default"]
                    r = module.r["default"]
                    scaling = alpha / r
                delta = (B @ A) * scaling
                deltas[name] = delta.detach().float().cpu()
    return deltas

def _weighted_average(
    all_deltas: List[Dict[str, torch.Tensor]],
    weights: List[float]
) -> Dict[str, torch.Tensor]:
    wsum = sum(weights)
    norm_w = [w / wsum for w in weights]
    names = all_deltas[0].keys()
    avg: Dict[str, torch.Tensor] = {}
    for n in names:
        acc = None
        for d, w in zip(all_deltas, norm_w):
            t = d[n] * w
            acc = t if acc is None else (acc + t)
        avg[n] = acc
    return avg

def _svd_factorize_to_lora(M: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factorize M (out x in) into B (out x r), A (r x in) such that B@A ≈ M
    using top-r SVD: M ≈ U_r diag(S_r) V_r^T  =>  set B=U_r diag(S_r), A=V_r^T.
    """
    # full_matrices=False gives economic SVD
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(r, U.shape[1], Vh.shape[0])
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    B = U_r @ torch.diag(S_r)           # (out x r)
    A = Vh_r                             # (r x in)
    return B, A

def _build_peft_skeleton(base_model, r: int, alpha: int, target_modules: List[str]) -> PeftModel:
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, cfg)
    return peft_model

def _infer_target_modules_from_adapter(base_model, adapter_path: str) -> List[str]:
    # Load once to read its peft_config
    tmp = PeftModel.from_pretrained(base_model, adapter_path)
    peft_cfg = tmp.peft_config["default"]
    # Commonly it’s ["query","value"], but we'll read it to be safe:
    tmods = list(sorted(set(peft_cfg.target_modules)))
    # detach to free memory
    del tmp
    return tmods

def _load_base(model_id: str, num_labels: int = 2, problem_type: str = "single_label_classification"):
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.num_labels = num_labels
    cfg.problem_type = problem_type
    return AutoModelForSequenceClassification.from_pretrained(model_id, config=cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", choices=["base","large"], required=True)
    ap.add_argument("--task_name", required=False, default="glue")
    ap.add_argument("--adapters", nargs="+", required=True, help="Paths to k adapter dirs (order matters).")
    ap.add_argument("--weights", nargs="*", type=float, default=None, help="Optional weights aligned with --adapters; defaults to equal or read from shard manifests.")
    ap.add_argument("--shard_manifests", nargs="*", default=None, help="Optional JSONs with {indices:[...]} per client; used to derive weights by sample count.")
    ap.add_argument("--svd_rank", type=int, required=True, help="Rank r to compress fused ΔW back into LoRA.")
    ap.add_argument("--alpha", type=int, default=None, help="LoRA alpha for fused adapter (defaults to r).")
    ap.add_argument("--save_adapter", required=True, help="Output dir for fused adapter.")
    args = ap.parse_args()

    model_id = resolve_model_id(args.model_size)
    os.makedirs(args.save_adapter, exist_ok=True)

    # Build a small base model only for adapter ops (labels won't matter for attention weights)
    base_for_read = _load_base(model_id=model_id, num_labels=2)

    # 1) Collect deltas per adapter
    all_deltas: List[Dict[str, torch.Tensor]] = []
    target_modules: Optional[List[str]] = None
    inferred_r = None
    inferred_alpha = None

    for path in args.adapters:
        model = PeftModel.from_pretrained(base_for_read, path)
        deltas = _collect_lora_deltas(model)
        all_deltas.append(deltas)
        # infer rank/alpha from first adapter for defaults
        if inferred_r is None:
            peft_cfg = model.peft_config["default"]
            inferred_r = peft_cfg.r
            inferred_alpha = peft_cfg.lora_alpha
        if target_modules is None:
            target_modules = _infer_target_modules_from_adapter(base_for_read, path)
        # clean up per-adapter weights from memory
        del model

    assert target_modules is not None
    # 2) Weights
    if args.weights is not None and len(args.weights) > 0:
        weights = [float(w) for w in args.weights]
        assert len(weights) == len(args.adapters), "--weights must match --adapters length."
    elif args.shard_manifests:
        counts = []
        for mpath in args.shard_manifests:
            with open(mpath, "r") as f:
                data = json.load(f)
            counts.append(len(data["indices"]))
        weights = counts
    else:
        weights = [1.0] * len(args.adapters)

    # 3) Average
    fused = _weighted_average(all_deltas, weights)

    # 4) Build fused PEFT skeleton (rank r, alpha)
    r = int(args.svd_rank)
    alpha = int(args.alpha) if args.alpha is not None else r
    base_for_write = _load_base(model_id=model_id, num_labels=2)
    fused_model = _build_peft_skeleton(base_for_write, r=r, alpha=alpha, target_modules=target_modules)

    # 5) Write A/B from SVD (per module), accounting for scaling
    # scaling_new = alpha / r; we want B@A = ΔW_avg / scaling_new
    scaling_new = alpha / r
    for name, module in fused_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "default" in module.lora_A.keys():
            if name not in fused:
                # It's possible some non-target got through; just continue
                continue
            M = fused[name] / scaling_new  # target product for B@A
            B, A = _svd_factorize_to_lora(M, r=r)
            # Assign
            with torch.no_grad():
                module.lora_A["default"].weight.copy_(A.to(module.lora_A["default"].weight.dtype))
                module.lora_B["default"].weight.copy_(B.to(module.lora_B["default"].weight.dtype))
                # PEFT keeps alpha/r internally in module.scaling; already set by LoraConfig

    # 6) Save fused adapter
    fused_model.save_pretrained(args.save_adapter)

    # 7) Manifest
    manifest = {
        "dataset": args.task_name,
        "model_id": model_id,
        "task": args.task_name,
        "k_adapters": len(args.adapters),
        "adapters": args.adapters,
        "weights": weights,
        "svd_rank": r,
        "alpha": alpha,
        "target_modules": target_modules,
    }
    with open(os.path.join(args.save_adapter, "fuse_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ FedAvg Merge Complete - Dataset: {args.task_name.upper()}")
    print(f"{'='*60}")
    print(f"Fused adapter saved to: {args.save_adapter}")
    print(f"Number of adapters merged: {len(args.adapters)}")
    print(f"SVD rank: {r}, Alpha: {alpha}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

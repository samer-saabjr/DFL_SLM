#!/usr/bin/env python
# Downloads & caches RoBERTa models so later scripts can load them offline.
# Models: roberta-base, roberta-large

import argparse
import os
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_home",
        type=str,
        default=None,
        help="Optional HF cache root (e.g., ./hf_cache). If set, will export HF_HOME & TRANSFORMERS_CACHE."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch/tag to download (default: main)."
    )
    args = parser.parse_args()

    if args.hf_home:
        os.makedirs(args.hf_home, exist_ok=True)
        os.environ["HF_HOME"] = os.path.abspath(args.hf_home)
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
        os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
        print(f"✔ Using HF cache at: {os.environ['HF_HOME']}")

    model_ids = ["roberta-base", "roberta-large"]
    for mid in model_ids:
        print(f"\n⬇️  Downloading {mid} (revision={args.revision}) …")
        local_path = snapshot_download(
            repo_id=mid,
            repo_type="model",
            revision=args.revision,   # keep 'main' unless you need a specific tag
        )
        print(f"✅ Cached {mid} at: {local_path}")

    print("\nAll set. Your training scripts can now call from_pretrained('roberta-base'|'roberta-large') without re-downloading.")

if __name__ == "__main__":
    main()

"""Prepare AR-GRPO RL data in verl parquet format.

Reads SYSTEM_PROMPT from search.config so the student sees identical
framing at SFT, eval, and RL.

Usage:
  python search/scripts/prepare_rl_data.py \
      --local_dir data/search/rl \
      --data_sources nq,hotpotqa --val_frac 0.01
"""
import argparse
import os
import sys

import datasets

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from search.config import SYSTEM_PROMPT

print(f"[prepare_rl_data] SYSTEM_PROMPT (first 120): {SYSTEM_PROMPT[:120]!r}")


def make_prefix(example):
    q = example["question"].strip()
    if q and q[-1] != "?":
        q += "?"
    return f"{SYSTEM_PROMPT} Question: {q}\n"


def process_fn(example, idx, data_source, split):
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": make_prefix(example)}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": example["golden_answers"]},
        },
        "extra_info": {"split": split, "index": idx},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True)
    ap.add_argument("--data_sources", default="nq,hotpotqa")
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    all_splits = []
    for ds_name in args.data_sources.split(","):
        ds_name = ds_name.strip()
        print(f"Loading {ds_name} train...")
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", ds_name)
        train = ds["train"]
        train = train.map(
            lambda ex, idx, _ds=ds_name: process_fn(ex, idx, _ds, "train"),
            with_indices=True,
            remove_columns=[c for c in train.column_names
                            if c not in ("question", "golden_answers")],
        )
        all_splits.append(train)
        print(f"  {ds_name}: {len(train):,} train examples")

    merged = datasets.concatenate_datasets(all_splits)
    merged = merged.shuffle(seed=args.seed)
    n_val = max(1, int(len(merged) * args.val_frac))
    val_ds = merged.select(range(n_val)).map(
        lambda ex, idx: {**ex, "extra_info": {**ex["extra_info"], "split": "test"}},
        with_indices=True,
    )
    train_ds = merged.select(range(n_val, len(merged)))
    train_ds.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(args.local_dir, "test.parquet"))
    print(f"\nWrote {len(train_ds):,} train + {len(val_ds):,} val to {args.local_dir}")

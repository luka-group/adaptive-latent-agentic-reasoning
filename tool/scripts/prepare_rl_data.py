"""Prepare AR-GRPO RL data in verl parquet format (tool domain).

Source: the SFT teacher-trace JSONL produced by
`tool/scripts/prepare_sft_data.py` (already filtered to score >= 1.5 by
default — i.e., trajectories where the Qwen3-4B rollout actor produced
an exact AST match to the expected ToolMind calls). Reusing this file
guarantees the RL data has the same quality bar and same item set as
SFT, so any gain we measure during AR-GRPO is on top of clean
supervision.

We pre-render the chat-template prompt
    [system + tools-block + user + `<|im_start|>assistant\\n`]
so the agent loop can skip the chat template (which would otherwise
double-render the tools block) and tokenize raw. Expected calls are
flattened across turns into a single JSON-string list, matching the
`score_episode` signature used by `tool/rl/reward.py`.

Usage:
  python -m tool.scripts.prepare_rl_data \\
      --trace_path data/tool/sft_traces.jsonl \\
      --local_dir  data/tool/rl \\
      --tokenizer  Qwen/Qwen3-4B-Thinking-2507 \\
      --max_train 20000 --val_frac 0.01
"""
from __future__ import annotations

import argparse
import json
import os

import datasets
from transformers import AutoTokenizer

from tool.config import SYSTEM_PROMPT


ASSISTANT_OPEN = "<|im_start|>assistant\n"


def _render_prompt(tok, system_msg: str, user_msg: str, tools: list) -> str:
    """Render the prompt exactly the way SFT does:
    `apply_chat_template([{system}, {user}], tools=..., add_generation_prompt=False)`
    + manual assistant opener (no `<think>` prefix).
    """
    prefix = tok.apply_chat_template(
        [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
    )
    return prefix + ASSISTANT_OPEN


def _flat_expected(traj: dict) -> list[str]:
    """Flatten `turns[*].tool_calls` (already JSON strings in the trace
    file) into one list. Final-answer turns contribute no expected calls."""
    out: list[str] = []
    for turn in traj.get("turns") or []:
        out.extend(turn.get("tool_calls") or [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_path", required=True,
                    help="SFT trace JSONL produced by tool/scripts/prepare_sft_data.py "
                         "(already score-filtered).")
    ap.add_argument("--local_dir", required=True,
                    help="Output dir for {train,test}.parquet")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--max_prompt_tok", type=int, default=7000,
                    help="Skip trajectories whose pre-rendered prompt would exceed this "
                         "(no room for a useful response under max_response_length=4096).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"[prepare_rl_data] tokenizer={args.tokenizer} trace={args.trace_path}")
    print(f"[prepare_rl_data] SYSTEM_PROMPT[:60]={SYSTEM_PROMPT[:60]!r}")

    records: list[dict] = []
    n_total = n_no_tools = n_no_calls = n_too_long = 0
    with open(args.trace_path) as f:
        for line in f:
            n_total += 1
            traj = json.loads(line)
            tools = traj.get("tools") or []
            question = (traj.get("question") or "").strip()
            if not tools or not question:
                n_no_tools += 1
                continue
            flat = _flat_expected(traj)
            if not flat:
                n_no_calls += 1
                continue
            prompt_text = _render_prompt(tok, SYSTEM_PROMPT, question, tools)
            n_prompt_tok = len(tok.encode(prompt_text, add_special_tokens=False))
            if n_prompt_tok > args.max_prompt_tok:
                n_too_long += 1
                continue
            records.append({
                "data_source": "toolmind/sft_traces",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "tool-use",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"expected_calls": flat},
                },
                "extra_info": {
                    "split":            "train",  # overridden for val below
                    "signature":        traj.get("signature", ""),
                    "score":            float(traj.get("score", 0.0)),
                    "n_prompt_tok":     n_prompt_tok,
                    "n_expected_calls": len(flat),
                },
            })

    print(f"[prepare_rl_data] scanned={n_total}  kept={len(records)}  "
          f"skipped(no_tools={n_no_tools} no_calls={n_no_calls} too_long={n_too_long})")

    ds = datasets.Dataset.from_list(records).shuffle(seed=args.seed)
    if args.max_train > 0 and len(ds) > args.max_train + max(1, int(len(ds) * args.val_frac)):
        n_val_cap = max(1, int(args.max_train * args.val_frac))
        ds = ds.select(range(args.max_train + n_val_cap))
    n_val = max(1, int(len(ds) * args.val_frac))
    val_ds = ds.select(range(n_val)).map(
        lambda ex: {**ex, "extra_info": {**ex["extra_info"], "split": "test"}},
    )
    train_ds = ds.select(range(n_val, len(ds)))

    train_path = os.path.join(args.local_dir, "train.parquet")
    val_path = os.path.join(args.local_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    print(f"\n[prepare_rl_data] wrote {len(train_ds):,} train + "
          f"{len(val_ds):,} val → {args.local_dir}")


if __name__ == "__main__":
    main()

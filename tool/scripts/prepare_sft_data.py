"""Convert ToolMind rollouts into a teacher-trace JSONL for Stage 1 AASD.

Input layout (one of):
  <src>/*.jsonl   — sharded rollout files; each line is a rollout dict with
                    keys {signature, item_idx, rollout_idx, scores, num_correct,
                          tokens, messages, expected_calls_by_turn, preds_by_turn}.

We keep rollouts whose top score is >= `--min_score` (default 1.5 = perfect
tool-call match) and parse each into a flat trajectory consumed by
`tool.dataset.ToolLatentDataset`. The tools list is parsed out of the
`<tools>...</tools>` block in the rollout's system prompt; the per-rollout
system preamble is dropped (the training-time system prompt comes from
`tool.config.SYSTEM_PROMPT`, with tools rendered by Qwen3's native template).

Output schema (one JSON object per line):
  {
    "signature": str, "score": float, "item_idx": int, "rollout_idx": int,
    "tools": [ {...openapi-ish dict...}, ... ],
    "question": str,
    "turns": [
      {
        "thinking":      str,                    # text inside <think>...</think>
        "free_text":     str,                    # text between </think> and first <tool_call>, if any
        "tool_calls":    [ str, ... ],           # JSON-string per call, in order
        "tool_response": str | null,             # the immediately-following <tool_response> body
      },
      ...
    ],
    "answer_thinking": str,    # thinking of the final assistant turn that has no tool calls
    "answer":          str,    # free text of that final turn (may be empty for short rollouts)
  }
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

TOOLS_BLOCK_RE = re.compile(r"<tools>\s*\n(.*?)\n\s*</tools>", re.DOTALL)
THINK_RE = re.compile(r"<think>\s*\n?(.*?)\n?\s*</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*\n?(.*?)\n?\s*</tool_call>", re.DOTALL)


def parse_tools(system_content: str) -> list[dict]:
    """One JSON object per non-empty line inside <tools></tools>."""
    m = TOOLS_BLOCK_RE.search(system_content)
    if not m:
        return []
    out: list[dict] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # Some specs may be split across lines; skip malformed.
            return []
    return out


def parse_assistant(content: str, tool_calls_field: list | None):
    """Return (thinking, free_text, tool_calls_json_strs)."""
    thinking = ""
    rest = content
    m = THINK_RE.search(content)
    if m:
        thinking = m.group(1).strip()
        rest = content[: m.start()] + content[m.end():]

    # Prefer the structured tool_calls list when present; fall back to
    # parsing <tool_call> blocks out of the text.
    calls: list[str] = []
    if tool_calls_field:
        for tc in tool_calls_field:
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                # Some serializers store arguments as a JSON-encoded string.
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            calls.append(json.dumps({"name": name, "arguments": args}))
    else:
        for tcm in TOOL_CALL_RE.finditer(rest):
            blob = tcm.group(1).strip()
            try:
                obj = json.loads(blob)
                calls.append(json.dumps({
                    "name": obj.get("name"),
                    "arguments": obj.get("arguments", {}),
                }))
            except json.JSONDecodeError:
                continue

    free_text = TOOL_CALL_RE.sub("", rest).strip()
    return thinking, free_text, calls


def parse_tool_response(content: str) -> str:
    """Strip outer <tool_response>...</tool_response> wrapper if present."""
    m = re.search(r"<tool_response>\s*\n?(.*?)\n?\s*</tool_response>",
                  content, re.DOTALL)
    return m.group(1).strip() if m else content.strip()


def convert_rollout(r: dict) -> dict | None:
    conv = r.get("messages")
    # messages is nested: list[ list[ dict ] ]
    if not conv or not isinstance(conv, list):
        return None
    conv = conv[0]
    if not conv or conv[0].get("role") != "system":
        return None
    tools = parse_tools(conv[0].get("content", ""))
    if not tools:
        return None

    question = ""
    for m in conv:
        if m.get("role") == "user":
            question = m.get("content", "").strip()
            break
    if not question:
        return None

    turns: list[dict] = []
    answer_thinking = ""
    answer = ""

    # Walk assistant→tool pairs. Trailing assistant with no tool calls is
    # the final answer turn.
    pending: dict | None = None
    for m in conv:
        role = m.get("role")
        if role == "assistant":
            think, free_text, calls = parse_assistant(
                m.get("content", "") or "", m.get("tool_calls"),
            )
            if calls:
                # Flush any prior pending turn that never got a response.
                if pending is not None:
                    turns.append(pending)
                pending = {
                    "thinking": think,
                    "free_text": free_text,
                    "tool_calls": calls,
                    "tool_response": None,
                }
            else:
                answer_thinking = think
                answer = free_text
        elif role == "tool":
            resp = parse_tool_response(m.get("content", "") or "")
            if pending is not None:
                pending["tool_response"] = resp
                turns.append(pending)
                pending = None
    if pending is not None:
        turns.append(pending)

    # Useful only if we have at least one tool call OR a final answer.
    if not turns and not answer:
        return None

    return {
        "signature": r.get("signature"),
        "score": float(r.get("scores", [0.0])[0] or 0.0),
        "item_idx": r.get("item_idx"),
        "rollout_idx": r.get("rollout_idx"),
        "tools": tools,
        "question": question,
        "turns": turns,
        "answer_thinking": answer_thinking,
        "answer": answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Directory of rollout shard JSONLs (e.g., toolmind_rollouts/full/).")
    ap.add_argument("--out", required=True,
                    help="Output teacher-trace JSONL path.")
    ap.add_argument("--min_score", type=float, default=1.5,
                    help="Keep rollouts whose top score >= this. Default 1.5 = perfect.")
    ap.add_argument("--dedup_by_item", action="store_true",
                    help="If set, keep only one (lowest rollout_idx) rollout per item_idx.")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Optional cap on kept rollouts (debug).")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.src, "*.jsonl")))
    if not shards:
        raise SystemExit(f"no shards under {args.src}")

    n_total = n_pass_score = n_kept = 0
    seen_items: set[int] = set()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fo:
        for path in shards:
            with open(path) as fi:
                for line in fi:
                    n_total += 1
                    r = json.loads(line)
                    scores = r.get("scores") or [0.0]
                    if (scores[0] or 0.0) < args.min_score:
                        continue
                    n_pass_score += 1
                    if args.dedup_by_item:
                        it = r.get("item_idx")
                        if it in seen_items:
                            continue
                        seen_items.add(it)
                    out = convert_rollout(r)
                    if out is None:
                        continue
                    fo.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n_kept += 1
                    if args.limit > 0 and n_kept >= args.limit:
                        break
            if args.limit > 0 and n_kept >= args.limit:
                break

    print(f"[prepare_sft_data] scanned={n_total} pass_score={n_pass_score} "
          f"kept={n_kept} out={args.out}")


if __name__ == "__main__":
    main()

"""Tool-domain reward: AST match on `<tool_call>` blocks + `<think>`
balance gate, with AR-GRPO counters.

Trajectory shape (post-SFT, single-shot):

    [prompt: system + tools + user + <|im_start|>assistant\\n]
    <latent>••••</latent>          # OR <think>...</think>
    <tool_call>{json}</tool_call>  # one or more calls
    <|im_end|>

Reward (binary EM, format-gated):
  - `format_ok=False`  →  score=0  (handled by AR-GRPO as hard -1.0 shaped)
  - else  score = 1.0 iff `score_episode` on the parsed calls returns the
    full bonus (i.e., dense=1.0 + sparse_lam=0.5 → reward >= 1.0).

`format_ok` only checks `<think>` balance. Latent open/close are not
gated — malformed latent would break the vLLM plugin's K-sentinel state
machine before scoring runs.

Counters returned for AR-GRPO shaping (same field names as search):
  em, n_think, n_latent, format_ok, think_chars, n_tool_calls,
  num_correct, unbalanced_latent.
"""
from __future__ import annotations

import json
import re

from tool.rollout.scoring import score_episode


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_LATENT_OPEN_RE = re.compile(r"<latent>")
_LATENT_CLOSE_RE = re.compile(r"</latent>")
_THINK_TAG_RE = re.compile(r"<(/?)(think)>")


def _format_ok(text: str) -> bool:
    """Every `<think>` has a matching `</think>` (no interleaving)."""
    stack = []
    for m in _THINK_TAG_RE.finditer(text):
        is_close = (m.group(1) == "/")
        if not is_close:
            stack.append("think")
        else:
            if not stack or stack[-1] != "think":
                return False
            stack.pop()
    return len(stack) == 0


def _think_chars(text: str) -> int:
    return sum(len(m.group(1)) for m in _THINK_RE.finditer(text))


def _parse_tool_calls(text: str):
    """Extract `<tool_call>{json}</tool_call>` blobs from the assistant
    response, parsed to dicts. Drops unparseable blocks silently."""
    out = []
    for m in _TOOL_CALL_RE.finditer(text):
        blob = m.group(1).strip()
        try:
            out.append(json.loads(blob))
        except json.JSONDecodeError:
            continue
    return out


def compute_score(data_source, solution_str, ground_truth,
                  extra_info=None, **kwargs):
    """verl reward callback.

    `ground_truth` is the dict written by `prepare_rl_data.py`:
        {"expected_calls": [json_str, ...]}
    (Parquet doesn't roundtrip lists-of-dicts cleanly through verl's
    cast layer, so each call is stored as a JSON string and parsed by
    `score_episode` via `parse_expected_call`.)
    """
    if isinstance(ground_truth, dict):
        expected = list(ground_truth.get("expected_calls") or [])
    else:
        expected = list(ground_truth or [])

    n_tool_calls_in_text = len(_TOOL_CALL_RE.findall(solution_str))
    n_think = len(_THINK_RE.findall(solution_str))
    n_latent = len(_LATENT_OPEN_RE.findall(solution_str))
    n_latent_close = len(_LATENT_CLOSE_RE.findall(solution_str))
    unbalanced_latent = float(n_latent != n_latent_close)
    think_chars = _think_chars(solution_str)
    fmt_ok = _format_ok(solution_str)

    base = {
        "format_ok":         float(fmt_ok),
        "n_think":           float(n_think),
        "n_latent":          float(n_latent),
        "unbalanced_latent": unbalanced_latent,
        "think_chars":       float(think_chars),
        "n_tool_calls":      float(n_tool_calls_in_text),
    }

    if not fmt_ok:
        return {"score": 0.0, "em": 0.0, "num_correct": 0.0, **base}

    preds = _parse_tool_calls(solution_str)
    reward, num_correct = score_episode(preds, expected)
    # Binarize: 1.0 iff all expected calls were matched (sparse bonus
    # only fires when correct == len(expected), see score_episode).
    em = 1.0 if reward >= 1.0 else 0.0
    return {
        "score": float(em),
        "em": float(em),
        "num_correct": float(num_correct),
        **base,
    }

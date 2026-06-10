"""Search reward — EM + format gate, returning AR-GRPO counters.

Trajectory layout (per SFT + agent loop):

    [prompt] <latent>••••</latent> <search>q</search>
                                   <information>...</information>
    [...turns with <latent>••••</latent> OR <think>...</think>...]
    <answer>final</answer>

Score = 1.0 if the normalized last `<answer>` matches a gold AND the
trajectory is format-correct; else 0.0.

Format gate (think-only): every `<think>` must have a matching
`</think>`. Closes the reward hack where the model emits `<think>`
without a close to evade the bucket classifier while reaping the
latent-bucket bonus. `<latent>`/`</latent>` are NOT format-gated —
malformed latent blocks would break the vLLM plugin's K-sentinel
state machine anyway, and the bucket classifier reads `n_think`.

Counters returned for AR-GRPO shaping:
  em, n_think, n_latent, format_ok, n_search, think_chars.
"""
from __future__ import annotations

import re
import string


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_LATENT_OPEN_RE = re.compile(r"<latent>")
_LATENT_CLOSE_RE = re.compile(r"</latent>")
_THINK_TAG_RE = re.compile(r"<(/?)(think)>")


def _format_ok(text: str) -> bool:
    stack = []
    for m in _THINK_TAG_RE.finditer(text):
        is_close = (m.group(1) == "/")
        tag = m.group(2)
        if not is_close:
            stack.append(tag)
        else:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
    return len(stack) == 0


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _extract_last_answer(solution_str: str):
    matches = list(_ANSWER_RE.finditer(solution_str))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _em(prediction: str, golden) -> int:
    if isinstance(golden, str):
        golden = [golden]
    p = _normalize(prediction)
    for g in golden:
        if _normalize(g) == p:
            return 1
    return 0


def _think_chars(text: str) -> int:
    return sum(len(m.group(1)) for m in _THINK_RE.finditer(text))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if isinstance(ground_truth, dict):
        golden = ground_truth.get("target")
    else:
        golden = ground_truth

    n_search = len(_SEARCH_RE.findall(solution_str))
    n_think = len(_THINK_RE.findall(solution_str))
    n_latent = len(_LATENT_OPEN_RE.findall(solution_str))
    n_latent_close = len(_LATENT_CLOSE_RE.findall(solution_str))
    # Logged for observability — not currently gated on. Helps detect if
    # the plugin's force-emit is dropping closes on truncation.
    unbalanced_latent = float(n_latent != n_latent_close)
    think_chars = _think_chars(solution_str)
    fmt_ok = _format_ok(solution_str)

    answer = _extract_last_answer(solution_str)
    if answer is None or not fmt_ok:
        return {
            "score": 0.0,
            "em": 0.0,
            "format_ok": float(fmt_ok),
            "n_search": float(n_search),
            "n_think": float(n_think),
            "n_latent": float(n_latent),
            "unbalanced_latent": unbalanced_latent,
            "think_chars": float(think_chars),
        }
    em = _em(answer, golden)
    return {
        "score": float(em),
        "em": float(em),
        "format_ok": 1.0,
        "n_search": float(n_search),
        "n_think": float(n_think),
        "n_latent": float(n_latent),
        "unbalanced_latent": unbalanced_latent,
        "think_chars": float(think_chars),
    }

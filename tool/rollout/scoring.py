"""Tool-call parsing + episode scoring.

Validation and scoring helpers for tool-calling rollouts (adapted from
the Atropos `tool_use_multiturn_server` environment).

Public surface:
  - normalize_tool_call_json(text)             - canonicalise raw assistant reply
  - validate_think_only(text)                  - "no tool call" turn validator
  - validate_think_plus_calls(text)            - "<think>...</think><tool_call>...</tool_call>+" validator
  - parse_expected_call(raw_call)              - parse ground-truth call (handles double-encoding)
  - json_objects_match(model_json, expected_json) - asymmetric subset check w/ coercion
  - score_episode(pred_calls, exp_calls, ...)  - dense + sparse + mismatch penalty
"""
from __future__ import annotations

import ast
import json
import re
from typing import List, Optional, Tuple


# ----- normalization --------------------------------------------------

def normalize_tool_call_json(txt: str) -> str:
    """Canonicalise assistant replies so:
      - leading `<think>...</think>` block is preserved
      - every `<tool_call>...</tool_call>` block is converted to valid JSON
        (handles Python-literal-style payloads via literal_eval + quote sub)
      - each `<tool_call>` / `</tool_call>` tag sits on its own line
    Returns the original text unchanged when normalisation fails.
    """
    m = re.match(r"^\s*(<think>[\s\S]*?</think>)\s*", txt, flags=re.IGNORECASE)
    if not m:
        return txt
    think_part = m.group(1)

    def _convert(match: re.Match) -> str:
        raw = match.group(1).strip()
        try:
            obj = ast.literal_eval(raw)
            return f"<tool_call>{json.dumps(obj, separators=(',', ':'))}</tool_call>"
        except Exception:
            pass
        try:
            json_like = re.sub(r"'([^']*)':", r'"\1":', raw)
            json_like = re.sub(r":\s*'([^']*)'", r':"\1"', json_like)
            json.loads(json_like)
            return f"<tool_call>{json_like}</tool_call>"
        except Exception:
            return match.group(0)

    tail = txt[len(m.group(0)):]
    tail = re.sub(
        r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
        _convert, tail, flags=re.DOTALL | re.IGNORECASE,
    )
    out = think_part + tail
    out = re.sub(r"\s*<tool_call>\s*", "\n<tool_call>\n", out)
    out = re.sub(r"\s*</tool_call>\s*", "\n</tool_call>\n", out)
    return out


def _canonicalise_tool_json(raw: str) -> Optional[str]:
    """Parse `raw` as JSON or Python literal; return canonical `json.dumps`.
    Returns None when both attempts fail."""
    try:
        return json.dumps(json.loads(raw), separators=(",", ":"))
    except Exception:
        pass
    try:
        return json.dumps(ast.literal_eval(raw), separators=(",", ":"))
    except Exception:
        return None


# ----- validators -----------------------------------------------------

def validate_think_only(txt: str) -> bool:
    """A narration turn must start with a single `<think>...</think>`
    block and contain no `<tool_call>` or `<tool_response>` anywhere."""
    txt = normalize_tool_call_json(txt)
    if not isinstance(txt, str):
        return False
    blocks = re.findall(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE)
    if len(blocks) != 1:
        return False
    if not re.match(r"^\s*<think>", txt, flags=re.IGNORECASE):
        return False
    if re.search(r"<tool_call\s*>", txt, flags=re.IGNORECASE):
        return False
    if re.search(r"<tool_response\s*>", txt, flags=re.IGNORECASE):
        return False
    return True


def validate_think_plus_calls(txt: str) -> Optional[List[dict]]:
    """A tool-calling turn must be `<think>...</think>` followed by one+
    `<tool_call>{...}</tool_call>` blocks with only whitespace between.
    Returns the parsed call list on success, else None."""
    txt = normalize_tool_call_json(txt)
    if re.search(r"<tool_response\s*>", txt, flags=re.IGNORECASE):
        return None
    m = re.match(r"\s*(<think>[\s\S]*?</think>)", txt, flags=re.IGNORECASE)
    if not m:
        return None
    rest = txt[len(m.group(1)):]
    tc_re = r"\s*<tool_call>\s*([\s\S]*?)\s*</tool_call>\s*"
    calls: List[dict] = []
    while True:
        m_tc = re.match(tc_re, rest, flags=re.IGNORECASE)
        if not m_tc:
            break
        canon = _canonicalise_tool_json(m_tc.group(1))
        if canon is None:
            return None
        calls.append(json.loads(canon))
        rest = rest[m_tc.end():]
    if not calls or rest.strip():
        return None
    return calls


# ----- expected-call parsing + comparison -----------------------------

def _coerce_jsonlike(val):
    """Best-effort coercion of `JSON-like` payloads (often double-encoded
    in the source dataset). Returns the coerced Python object, or `val`
    unchanged on failure."""
    if not isinstance(val, str):
        return val
    s = val.strip()
    low = s.lower()
    if low == "true":  return True
    if low == "false": return False
    if low in ("null", "none"): return None
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return val


def parse_expected_call(raw_call):
    """Parse a possibly-stringified-and-double-encoded ground-truth call
    into a dict `{name, arguments}`. Returns `{}` on hard failure so
    downstream comparison fails cleanly."""
    obj = raw_call
    if isinstance(raw_call, str):
        try:
            obj = json.loads(raw_call)
        except Exception:
            obj = _coerce_jsonlike(raw_call)
    if not isinstance(obj, dict):
        return {}
    if "arguments" in obj:
        obj["arguments"] = _coerce_jsonlike(obj["arguments"])
    return obj


def json_objects_match(model_json, expected_json) -> bool:
    """Asymmetric subset check: every key/value in `expected_json` must
    appear (recursively) in `model_json`. Robust to string-encoded dicts,
    Python-style booleans, and ordering."""
    model_json = _coerce_jsonlike(model_json)
    expected_json = _coerce_jsonlike(expected_json)
    if isinstance(expected_json, dict):
        if not isinstance(model_json, dict):
            return False
        for k, v in expected_json.items():
            if k not in model_json:
                return False
            if not json_objects_match(model_json[k], v):
                return False
        return True
    return model_json == expected_json


# ----- episode scoring -----------------------------------------------

def score_episode(
    pred_calls: list,
    exp_calls: list,
    lam: float = 0.5,
    wrong_call_penalty: float = -0.2,
) -> Tuple[float, int]:
    """Score one episode against its expected call list.

    Returns `(reward, num_correct_calls)`.
      - dense reward = #correct / #expected
      - sparse bonus = `lam` if every expected call matched
      - mismatch penalty = `wrong_call_penalty` if `__MISMATCH__` appeared

    Relevance episodes (no expected calls) fall back to the
    `__APOLOGY__` / `__INFO__` convention of the upstream environment.
    ToolMind rollouts never produce these sentinels; the branch is kept
    for completeness.
    """
    if len(exp_calls) == 0:
        has_apology = "__APOLOGY__" in pred_calls
        has_info = "__INFO__" in pred_calls
        other_calls = [
            c for c in pred_calls
            if c not in ("__APOLOGY__", "__INFO__", "__MISMATCH__")
        ]
        success = ("__MISMATCH__" not in pred_calls) and not other_calls
        if not success:
            return wrong_call_penalty, 0
        return 1.0 + 0.1 * int(has_apology) + 0.1 * int(has_info), 0

    exp_jsons = [parse_expected_call(r) for r in exp_calls]
    mismatch_penalty = 0.0
    if pred_calls and "__MISMATCH__" in pred_calls:
        pred_calls = [c for c in pred_calls if c != "__MISMATCH__"]
        mismatch_penalty = wrong_call_penalty
    pred_calls = list(pred_calls) + [{}] * (len(exp_jsons) - len(pred_calls))
    correct = sum(
        1 for p, e in zip(pred_calls, exp_jsons) if json_objects_match(p, e)
    )
    dense = correct / max(1, len(exp_jsons))
    bonus = lam if correct == len(exp_jsons) else 0.0
    return dense + bonus + mismatch_penalty, correct

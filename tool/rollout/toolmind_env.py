"""ToolMind dataset loader for turn-synchronized rollout.

Walks each row of `Nanbeige/ToolMind` and emits

    (messages_tuple, expected_calls_by_turn, inter_turns)

triples compatible with `run_turn_sync.ItemState`.

The verifier built on top of these triples works as follows:

  * `messages_tuple` = initial conversation (system + first user) sent to vLLM.
    The system content is a deep-thinking preamble + a Hermes-style
    `<tools>...</tools>` block built from the row's `tools` field + (optionally)
    the dataset's own system policy text.

  * `expected_calls_by_turn[i]` = list of JSON-string tool-call expectations
    for the i-th *assistant* turn of the recorded trajectory.  Empty list means
    the model should reply with `<think>` + plain text (no tool call).

  * `inter_turns[i]` = the tool/user messages recorded between the i-th and
    (i+1)-th assistant turns.  Replayed verbatim into the running conversation
    so the model sees the same context the recorded assistant did.

Two subsets are supported (HF load_dataset split names):

  * `graph_syn_datasets` (163K rows, no system policy, BFCL-style)
  * `open_datasets` (205K rows, Tau-bench retail/airline policies)

The dataset itself is ShareGPT-style; the only translation we do is wrapping
`tool` role content in `<tool_response>...</tool_response>` so the running
conversation matches the Hermes tool-calling surface form.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

from datasets import load_dataset


DATASET_ID = "Nanbeige/ToolMind"

DEEP_THINKING_PREAMBLE = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via "
    "systematic reasoning processes to help come to a correct solution prior "
    "to answering. You should enclose your thoughts and internal monologue "
    "inside <think> </think> tags, and then provide your solution or response "
    "to the problem."
)

TOOL_INSTRUCTION_TEMPLATE = (
    "You have access to the following tools, listed inside <tools></tools> "
    "XML tags:\n"
    "<tools>\n"
    "{tools_block}\n"
    "</tools>\n\n"
    "For each tool call, return a JSON object with the function name and "
    "arguments inside <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
    "</tool_call>\n"
    "Tool results will be returned wrapped in <tool_response></tool_response> "
    "tags. After receiving a tool response, you may either call another tool "
    "or reply with a final answer."
)


def _build_system_content(tools: List[dict], policy: str = "") -> str:
    """Compose the system prompt: deep-thinking preamble + tools block + policy."""
    tools_block = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    parts = [
        DEEP_THINKING_PREAMBLE,
        TOOL_INSTRUCTION_TEMPLATE.format(tools_block=tools_block),
    ]
    if policy:
        parts.append(policy)
    return "\n\n".join(parts)


def _format_expected_calls(tool_calls: list) -> List[str]:
    """ToolMind `tool_calls` -> list[str] suitable for
    `parse_expected_call` (which accepts JSON strings *or* dicts).
    """
    out: List[str] = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        # arguments is already a dict in ToolMind; preserve it as a dict so
        # _json_objects_match can recurse without re-parsing strings.
        out.append(
            json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        )
    return out


def _convert_tool_msg(content: str) -> Dict[str, str]:
    """Wrap a ToolMind `tool` content in Hermes `<tool_response>` tags.

    `_validate_think_*` rejects any model reply that *contains* a
    `<tool_response>` tag, so wrapping these here only affects the message
    we feed to the model (as a user/tool role), not what the model is
    expected to produce.
    """
    return {
        "role": "tool",
        "content": f"<tool_response>\n{content}\n</tool_response>",
    }


def _build_one_item(row: dict):
    """Convert one ToolMind row into the rollout triple, or return None to skip."""
    convo = row.get("conversations") or []
    tools = row.get("tools") or []
    if not tools or not convo:
        return None

    # Capture an existing system message if present, then strip it from the
    # walk -- we'll fold it into our composed system prompt.
    existing_system = ""
    convo_after_sys = []
    for m in convo:
        if m.get("role") == "system" and not existing_system:
            existing_system = m.get("content") or ""
            continue
        convo_after_sys.append(m)

    # Find first user; nothing to rollout without one.
    first_user_idx = next(
        (i for i, m in enumerate(convo_after_sys) if m.get("role") == "user"),
        None,
    )
    if first_user_idx is None:
        return None

    # All assistant turn indices, in order.
    asst_indices = [
        i for i, m in enumerate(convo_after_sys) if m.get("role") == "assistant"
    ]
    if not asst_indices:
        return None
    if asst_indices[0] <= first_user_idx:
        # malformed: assistant before first user
        return None

    # Build expected calls per assistant turn.  Require at least one turn with
    # a tool_call -- a pure-text trajectory has nothing to score.
    expected_calls_by_turn: List[List[str]] = []
    has_any_tc = False
    for ai in asst_indices:
        tc = convo_after_sys[ai].get("tool_calls") or []
        if tc:
            expected_calls_by_turn.append(_format_expected_calls(tc))
            has_any_tc = True
        else:
            expected_calls_by_turn.append([])
    if not has_any_tc:
        return None

    # Build inter_turns[k] = messages between asst[k] and asst[k+1].
    inter_turns: List[List[Dict[str, str]]] = []
    for k in range(len(asst_indices) - 1):
        cur = asst_indices[k]
        nxt = asst_indices[k + 1]
        between: List[Dict[str, str]] = []
        for j in range(cur + 1, nxt):
            m = convo_after_sys[j]
            role = m.get("role", "")
            content = m.get("content") or ""
            if role == "tool":
                between.append(_convert_tool_msg(content))
            elif role in ("user", "system"):
                between.append({"role": role, "content": content})
            # Skip stray assistants (shouldn't happen given asst_indices boundaries)
        inter_turns.append(between)

    # Initial messages: composed system + everything *before* the first
    # assistant turn (i.e., first user, plus any tool/user messages before it
    # -- which is unusual but harmless).
    sys_content = _build_system_content(tools, existing_system)
    initial: List[Dict[str, str]] = [{"role": "system", "content": sys_content}]
    for j in range(asst_indices[0]):
        m = convo_after_sys[j]
        role = m.get("role", "")
        content = m.get("content") or ""
        if role == "tool":
            initial.append(_convert_tool_msg(content))
        elif role in ("user", "system"):
            # We've already folded any system into sys_content; skip duplicates.
            if role == "system":
                continue
            initial.append({"role": role, "content": content})

    return (tuple(initial), expected_calls_by_turn, inter_turns)


def load_toolmind_items(
    subset: str,
    num_shards: int,
    shard_idx: int,
    max_rows: int = -1,
) -> List[Tuple[Tuple[Dict[str, str], ...], List[List[str]], List[List[Dict[str, str]]]]]:
    """Load one shard of a ToolMind subset and return rollout triples.

    Args:
      subset: HF split name -- "graph_syn_datasets" or "open_datasets".
      num_shards / shard_idx: contiguous sharding.  num_shards=1 disables.
      max_rows: cap after sharding (for smoke tests).  -1 = no cap.
    """
    print(f"[toolmind_env] Loading {DATASET_ID} subset={subset}")
    ds = load_dataset(DATASET_ID, split=subset)
    print(f"[toolmind_env] Loaded {len(ds)} rows")
    if num_shards and num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        print(
            f"[toolmind_env] Shard {shard_idx}/{num_shards}: {len(ds)} rows"
        )

    items = []
    n_skipped = 0
    for row in ds:
        triple = _build_one_item(row)
        if triple is None:
            n_skipped += 1
            continue
        items.append(triple)
        if max_rows > 0 and len(items) >= max_rows:
            break

    print(
        f"[toolmind_env] Built {len(items)} items (skipped {n_skipped} rows "
        f"with no usable tool-call trajectory)"
    )
    return items

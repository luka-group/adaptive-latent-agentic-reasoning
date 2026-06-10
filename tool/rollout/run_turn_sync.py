"""Turn-synchronized ToolMind rollout runner.

For each turn we spawn one HTTP request per still-active state and
`asyncio.gather` them; vLLM's continuous batcher fuses them on the GPU,
trading the ~50 tok/s of single-stream serial replay for whatever the
chosen concurrency saturates.

Per-item state machine:
  - load_toolmind_items() returns (messages, expected_calls_by_turn, inter_turns)
  - at turn t we post `messages` to /v1/chat/completions, parse the reply
    with the validators in `tool.rollout.scoring`, and either advance
    (append the asst msg + the next `inter_turn`'s tool/user messages) or
    mark the state inactive on the first mismatch.
  - on max_turns or inactive, `score_episode` aggregates per-turn correctness
    into the final scalar reward we write to disk.

CLI (env vars only):

    SHARD_IDX=0 NUM_SHARDS=4 \
    VLLM_BASE_URL=http://localhost:9004/v1 VLLM_MODEL=Qwen/Qwen3-4B \
    OUT_PATH=data/tool/toolmind_rollouts/graph_syn_datasets.shard0.jsonl \
    MAX_CONCURRENT=128 \
    TOOLMIND_SUBSET=graph_syn_datasets \
    python -m tool.rollout.run_turn_sync
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import aiohttp
from transformers import AutoTokenizer

from tool.rollout.scoring import (
    json_objects_match as _json_objects_match,
    normalize_tool_call_json as _normalize_tool_call_json,
    parse_expected_call as _parse_expected_call,
    score_episode as _score_episode,
    validate_think_only as _validate_think_only,
    validate_think_plus_calls as _validate_think_plus_calls,
)
from tool.rollout.toolmind_env import load_toolmind_items


MODEL_DEFAULT = "Qwen/Qwen3-4B"

logger = logging.getLogger("run_turn_sync")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ---- Per-item state for the turn loop -------------------------------------------------


def _item_signature(messages: List[Dict[str, str]], rollout_idx: int = 0) -> str:
    """Stable per-(item, rollout) key for resume.

    sha256 of (system + first-user) message bodies, plus the rollout index so
    that the N rollouts per item under group_size=N are individually resumable.
    System prompt carries the tools schema; the first user turn is the task.
    """
    sys_txt = ""
    usr_txt = ""
    for m in messages:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value") or ""
        if role == "system" and not sys_txt:
            sys_txt = content
        if role in ("user", "human") and not usr_txt:
            usr_txt = content
            break
    return hashlib.sha256(
        (sys_txt + "\x1f" + usr_txt + "\x1f" + str(rollout_idx)).encode("utf-8", errors="ignore")
    ).hexdigest()


class ItemState:
    __slots__ = (
        "idx",
        "rollout_idx",
        "messages",
        "expected_calls_by_turn",
        "inter_turns",
        "preds_by_turn",
        "active",
        "max_turns",
        "signature",
        "written",
    )

    def __init__(
        self,
        idx: int,
        rollout_idx: int,
        messages_tuple,
        expected_calls_by_turn: List[List[str]],
        inter_turns: List[List[Dict[str, str]]],
        max_turns_cap: Optional[int] = None,
    ):
        self.idx = idx
        self.rollout_idx = rollout_idx
        # messages_tuple may contain frozenset elements; cast each turn to a
        # plain dict so we can extend it.
        self.messages: List[Dict[str, str]] = [dict(m) for m in messages_tuple]
        self.expected_calls_by_turn = expected_calls_by_turn
        self.inter_turns = inter_turns
        self.preds_by_turn: List[List] = [[] for _ in expected_calls_by_turn]
        self.active = True
        full_turns = len(expected_calls_by_turn)
        self.max_turns = (
            min(full_turns, max_turns_cap) if max_turns_cap and max_turns_cap > 0
            else full_turns
        )
        self.signature = _item_signature(self.messages, rollout_idx)
        self.written = False


def load_done_signatures(out_path: Path) -> set:
    """Read an existing output JSONL and return the set of item signatures that
    are already on disk.  Used for resume.
    """
    done: set = set()
    if not out_path.exists():
        return done
    n_bad = 0
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            sig = rec.get("signature")
            if sig:
                done.add(sig)
    if n_bad:
        logger.warning("load_done_signatures: %d malformed lines skipped in %s", n_bad, out_path)
    return done


# ---- Turn-synchronized rollout loop ---------------------------------------------------


async def post_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    sem: asyncio.Semaphore,
    temperature: float = 1.0,
    timeout_s: int = 600,
) -> str:
    """One chat-completions POST, returns the assistant content string."""
    async with sem:
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with session.post(url, json=payload, timeout=timeout_s) as r:
            d = await r.json()
        try:
            return d["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("malformed response %s: %s", e, str(d)[:200])
            return ""


def _advance_one(state: ItemState, turn_idx: int, raw_txt: str) -> None:
    """Validate one assistant reply against the expected calls; mutates state."""
    norm = _normalize_tool_call_json(raw_txt)
    expected = state.expected_calls_by_turn[turn_idx]

    # Always record the assistant message so we have a full conversation at the end.
    state.messages.append({"role": "assistant", "content": raw_txt})

    if expected:
        calls = _validate_think_plus_calls(norm)
        if calls is None or len(calls) != len(expected):
            state.preds_by_turn[turn_idx].append("__MISMATCH__")
            state.active = False
            return
        for mdl, exp_raw in zip(calls, expected):
            exp_obj = _parse_expected_call(exp_raw)
            if not _json_objects_match(mdl, exp_obj):
                state.preds_by_turn[turn_idx].append("__MISMATCH__")
                state.active = False
                return
        state.preds_by_turn[turn_idx].extend(calls)
    else:
        # narration / no-tool-call turn (e.g. final assistant summary or relevance)
        if not _validate_think_only(norm):
            state.active = False


async def run_turn_loop(
    states: List[ItemState],
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
    out_fh: TextIO,
    tokenizer,
    max_token_length: int,
    counts: Dict[str, int],
) -> None:
    """Turn-synchronized rollout loop with truly streaming per-item writes.

    Each turn, we spawn one `asyncio.Task` per active state.  Each task awaits
    its POST and then *immediately* advances + (if the item is done) writes
    the record to disk.  The outer `asyncio.gather` just waits for the turn's
    tasks to all settle before moving to the next turn.

    This is the important detail: writes do NOT batch at the turn boundary --
    they happen as soon as each individual POST returns and the item is
    classified as done (mismatch or last-turn).  For `single` (one turn) we
    therefore see writes streaming in throughout the run rather than all at
    the end, which is what we want for both resume safety and progress
    visibility.
    """
    if not states:
        return
    write_lock = asyncio.Lock()
    global_max_turns = max(s.max_turns for s in states)
    logger.info(
        "turn loop start: %d states, global_max_turns=%d, concurrency=%d",
        len(states), global_max_turns, sem._value,
    )

    async def post_and_finalize(st: ItemState, turn_idx: int) -> str:
        """Post one turn, advance state, write to disk if item done. Returns 'ok' / 'err' / 'done'."""
        try:
            txt = await post_one(session, base_url, model, st.messages, max_tokens, sem)
        except Exception as e:
            logger.warning("post error item=%d rollout=%d turn=%d: %s",
                           st.idx, st.rollout_idx, turn_idx, e)
            st.active = False
            txt = None
        if txt is not None:
            _advance_one(st, turn_idx, txt)

        # Optional cumulative-token cutoff: stop the trajectory if the running
        # conversation has grown past max_token_length, even when the model
        # would still be allowed to continue by turn count.  This is the
        # "long-tail context" guard -- a multi-turn item whose context has
        # blown past the budget can't be tokenized for SFT anyway.
        if st.active and max_token_length and tokenizer is not None:
            try:
                ids = tokenizer.apply_chat_template(
                    st.messages, tokenize=True, add_generation_prompt=False
                )
                if len(ids) > max_token_length:
                    st.active = False
            except Exception:
                pass

        last_turn = (turn_idx + 1) >= st.max_turns
        if not st.written and (not st.active or last_turn):
            async with write_lock:
                _score_and_write_one(
                    st, out_fh,
                    tokenizer=tokenizer,
                    max_token_length=max_token_length,
                    counts=counts,
                )
                st.written = True
            return "done"
        return "ok" if txt is not None else "err"

    for turn_idx in range(global_max_turns):
        # Inject pre-recorded tool response from the dataset into every active item.
        if turn_idx > 0:
            for st in states:
                if not st.active or st.written:
                    continue
                if turn_idx - 1 < len(st.inter_turns):
                    st.messages.extend(st.inter_turns[turn_idx - 1])

        active_states = [
            st for st in states
            if st.active and not st.written and turn_idx < st.max_turns
        ]
        if not active_states:
            logger.info("turn %d: 0 active items, ending early", turn_idx)
            break

        t0 = time.time()
        tasks = [
            asyncio.create_task(post_and_finalize(st, turn_idx))
            for st in active_states
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - t0

        n_ok = sum(1 for r in results if r == "ok")
        n_done = sum(1 for r in results if r == "done")
        n_err = sum(1 for r in results if r == "err" or isinstance(r, BaseException))
        n_still_active = sum(1 for st in states if st.active and not st.written)
        logger.info(
            "turn %d  sent=%d ok=%d done_this_turn=%d err=%d elapsed=%.1fs  "
            "written_total=%d still_active=%d",
            turn_idx, len(active_states), n_ok, n_done, n_err, elapsed,
            counts["written"], n_still_active,
        )


def _score_and_write_one(
    st: ItemState,
    fh: TextIO,
    *,
    tokenizer,
    max_token_length: int,
    counts: Dict[str, int],
    wrong_call_penalty: float = -0.2,
) -> None:
    """Apply episode scoring + binarization + length-decay to one item,
    then write its JSONL record. Updates counts in place.
    """
    expected_flat = [c for turn in st.expected_calls_by_turn for c in turn]
    preds_flat = [c for turn in st.preds_by_turn for c in turn]
    reward, num_correct = _score_episode(
        preds_flat, expected_flat, wrong_call_penalty=wrong_call_penalty
    )
    try:
        input_ids = tokenizer.apply_chat_template(
            st.messages, tokenize=True, add_generation_prompt=False
        )
    except Exception as e:
        logger.warning("tokenize failed item=%d: %s", st.idx, e)
        input_ids = []
    if reward > 0.99 and len(input_ids) > max_token_length * 0.5:
        cutoff = max_token_length * 0.5
        frac = min((len(input_ids) - cutoff) / (max_token_length - cutoff), 1.0)
        reward = max(0.0, reward - frac)
    final_score = reward if reward > 0 else -1.0
    counts["written"] += 1
    if final_score > 0:
        counts["kept_positive"] += 1
    else:
        counts["rejected"] += 1
    rec = {
        "signature": st.signature,
        "item_idx": st.idx,
        "rollout_idx": st.rollout_idx,
        "scores": [final_score],
        "num_correct": num_correct,
        "tokens": [input_ids],
        "messages": [st.messages],
        "expected_calls_by_turn": st.expected_calls_by_turn,
        "preds_by_turn": st.preds_by_turn,
    }
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()


# ---- Entrypoint -----------------------------------------------------------------------


async def main() -> None:
    shard_idx = int(os.environ.get("SHARD_IDX", "0"))
    num_shards = int(os.environ.get("NUM_SHARDS", "4"))
    total_items = int(os.environ.get("TOTAL_ITEMS", "-1"))   # -1 => all in shard
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:9004/v1")
    model = os.environ.get("VLLM_MODEL", MODEL_DEFAULT)
    out_path = Path(os.environ["OUT_PATH"])
    max_concurrent = int(os.environ.get("MAX_CONCURRENT", "128"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
    max_token_length = int(os.environ.get("MAX_TOKEN_LENGTH", "8192"))
    # Hard cap on episode length; 0 / unset => no cap (follow dataset).
    max_turns_cap = int(os.environ.get("MAX_TURNS_CAP", "0"))
    # Best-of-N rollouts per dataset item (group_size). Default 1 => one rollout.
    rollout_n = int(os.environ.get("ROLLOUT_N", "1"))
    toolmind_subset = os.environ.get("TOOLMIND_SUBSET", "graph_syn_datasets")

    # 1. Load items.
    items = load_toolmind_items(
        subset=toolmind_subset,
        num_shards=num_shards,
        shard_idx=shard_idx,
    )

    if total_items > 0:
        items = items[:total_items]

    # 2. Resume: drop items whose signature already exists in the output file.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_sigs = load_done_signatures(out_path)
    if done_sigs:
        logger.info("resume: %d already-done signatures found in %s", len(done_sigs), out_path)

    # Expand each dataset item into N rollouts (group_size=N), each with its own
    # signature so resume is per-(item, rollout_idx).
    all_states: List[ItemState] = []
    for i, (m, e, t) in enumerate(items):
        for r in range(rollout_n):
            all_states.append(ItemState(i, r, m, e, t, max_turns_cap=max_turns_cap))
    states = [s for s in all_states if s.signature not in done_sigs]
    n_skipped = len(all_states) - len(states)
    logger.info(
        "dataset=%s  shard %d/%d  items=%d  rollout_n=%d  total_states=%d  "
        "already_done=%d  to_process=%d  max_turns_cap=%s  "
        "vllm=%s  model=%s  concurrency=%d",
        toolmind_subset, shard_idx, num_shards, len(items), rollout_n, len(all_states),
        n_skipped, len(states), (max_turns_cap or "off"),
        base_url, model, max_concurrent,
    )
    if not states:
        logger.info("nothing to do (all items already in output). exiting.")
        return

    tok = AutoTokenizer.from_pretrained(model)

    # 3. Streaming-write turn loop. Output is opened in append mode so resumed
    # runs add to the existing file instead of overwriting.
    counts = {"written": 0, "kept_positive": 0, "rejected": 0}
    connector = aiohttp.TCPConnector(limit=max_concurrent + 16)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    with out_path.open("a", buffering=1) as out_fh:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            sem = asyncio.Semaphore(max_concurrent)
            await run_turn_loop(
                states,
                session=session,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                sem=sem,
                out_fh=out_fh,
                tokenizer=tok,
                max_token_length=max_token_length,
                counts=counts,
            )
        # Catch any state that somehow never wrote (shouldn't happen, defensive).
        for st in states:
            if not st.written:
                _score_and_write_one(
                    st, out_fh,
                    tokenizer=tok,
                    max_token_length=max_token_length,
                    counts=counts,
                )
                st.written = True

    logger.info("DONE  %s  %s  (skipped=%d)", out_path, counts, n_skipped)


if __name__ == "__main__":
    asyncio.run(main())

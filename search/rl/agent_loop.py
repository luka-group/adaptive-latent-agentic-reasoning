"""Search agent loop for verl GRPO rollouts.

The vLLM plugin (`alar.vllm_plugin`) drives the `<latent>••••</latent>`
state machine autonomously: on `<latent>` emission it splices in K
projector outputs at sentinel positions, then lets the model emit
`</latent>` naturally. So per turn this loop only:

  1. Generates until `</search>` or `</answer>` (stop strings).
  2. If `</answer>`: terminates.
  3. If `</search>`: retrieves, appends `<information>...</information>`
     (loss-masked).

response_mask convention (per token):
  1 = assistant-generated, contributes to PPO loss
  0 = K sentinel positions inside `<latent>` blocks
      AND retrieval observation tokens

Env vars:
  LATENT_RETRIEVAL_URL    default http://127.0.0.1:8000
  LATENT_MAX_TURNS        default 6
  LATENT_TOPK             default 3
  LATENT_MAX_DOC_CHARS    default 500
  LATENT_NUM_LATENT       default 4
  LATENT_DEBUG_FIRST_N    default 0 (verbose log for first N rollouts)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Tuple
from uuid import uuid4

import aiohttp
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

# Where the vLLM plugin (alar.vllm_plugin) wrote per-request z files.
# We read and immediately delete them after each turn's generate().
_LATENT_Z_DIR = os.environ.get("LATENT_Z_DIR", "/tmp/latent_z")

# AgentLoopWorker is a Ray actor process that never imports alar.modeling,
# so the deferred install hook in LatentQwen2ForCausalLM.__init__ doesn't
# fire here. Without install(), the AsyncLLMServerManager.generate patch
# (which threads the caller's request_id to vLLM instead of dropping it
# in favor of a fresh uuid4().hex) is missing, and our z files become
# unreachable from this process. Install on module load.
try:
    from alar.rl.install_patches import install as _alar_install
    _alar_install()
except Exception as _e:
    import traceback
    print(f"[agent] WARN: install_patches failed: {_e}\n{traceback.format_exc()}",
          flush=True)

logger = logging.getLogger(__file__)

LOG_FIRST_N = int(os.environ.get("LATENT_DEBUG_FIRST_N", "0"))
_dbg_counter = 0

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _dbg(msg: str):
    if _dbg_counter < LOG_FIRST_N:
        print(f"[agent] {msg}", flush=True)


def _read_and_clear_z_file(request_id: str) -> torch.Tensor | None:
    """Load the z file written by the vLLM plugin for this request_id and
    delete it. Returns the [N_z, H] tensor of z values for this turn, or
    None if no file is found (turn had no `<latent>` blocks).

    vLLM internally augments request_id with a suffix (e.g. our `uuid.hex`
    becomes `<uuid_hex>-<8_hex_chars>` in `input_batch.req_ids`), so the
    plugin's `_persist_z_seq` writes `<augmented_id>.pt`, not the literal
    `<request_id>.pt` we passed to `generate()`. We glob the prefix to find
    the actual file.
    """
    import glob
    pattern = os.path.join(_LATENT_Z_DIR, f"{request_id}*.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        # Unexpected (each agent loop submits one request per turn) — pick
        # the newest by mtime and warn.
        logger.warning("[agent] %d z files match prefix %s — using newest",
                       len(matches), request_id)
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = matches[0]
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        z_seq = data.get("z_seq") if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("[agent] failed to load %s: %s", path, e)
        z_seq = None
    for p in matches:
        try:
            os.remove(p)
        except OSError:
            pass
    return z_seq


def _build_turn_z_positions(turn_ids: List[int], n_z: int) -> List[int]:
    """Find the positions of K sentinels (SENTINEL_ID) in turn_ids, in order.
    Caller passes n_z so we can validate the count matches the expected K.
    """
    positions = [i for i, t in enumerate(turn_ids) if t == SENTINEL_ID]
    if len(positions) != n_z:
        logger.warning(
            "[agent] sentinel count mismatch: turn has %d • tokens, "
            "vLLM produced %d z values", len(positions), n_z,
        )
    return positions[: min(len(positions), n_z)]


async def _retrieve(session: aiohttp.ClientSession, url: str, query: str,
                    topk: int, max_doc_chars: int) -> str:
    payload = {"queries": [query], "topk": topk}
    try:
        async with session.post(url, json=payload, timeout=60) as resp:
            data = await resp.json()
    except Exception as e:
        logger.warning(f"retrieval failed: {e}")
        return ""
    results = data.get("results") or data.get("result") or []
    if results and isinstance(results[0], list):
        results = results[0]
    parts = []
    for idx, doc in enumerate(results):
        block = f"Doc {idx + 1}(Title: {doc.get('title','')}) {doc.get('text','')}"
        if max_doc_chars > 0:
            block = block[:max_doc_chars]
        parts.append(block)
    return "\n".join(parts)


SENTINEL_ID = 6667  # `•` bullet token


def _build_response_mask(
    turn_ids: List[int],
    latent_open_ids: List[int],
    latent_close_ids: List[int],
    K: int,
) -> List[int]:
    """Mask=1 by default; mask=0 at the K sentinel positions inside each
    `<latent>` block.

    Fast path: pattern-match `<open_ids> K-sentinels <close_ids>`. The
    sentinel positions are dummy tokens whose embeddings the plugin
    replaces with projector outputs; the model never sampled them so
    their logprobs are not meaningful (mask=0). The open/close pieces
    ARE naturally sampled (mask=1).

    Fallback: any run of `SENTINEL_ID` of length ≥1 also gets mask=0,
    so a BPE-split open/close (rare but possible if the model drifts)
    still masks the sentinels and doesn't poison the loss.
    """
    n = len(turn_ids)
    mask = [1] * n
    if K <= 0:
        return mask

    # Fast path: full block pattern match.
    matched = [False] * n
    P = len(latent_open_ids)
    Q = len(latent_close_ids)
    if P > 0 and Q > 0:
        i = 0
        while i + P + K + Q <= n:
            if (turn_ids[i:i + P] == list(latent_open_ids)
                    and turn_ids[i + P + K:i + P + K + Q] == list(latent_close_ids)):
                for j in range(i + P, i + P + K):
                    mask[j] = 0
                    matched[j] = True
                i = i + P + K + Q
                continue
            i += 1

    # Fallback: any remaining SENTINEL_ID position (e.g. if open/close
    # BPE merged differently). Mask them too — they're never validly
    # sampled by the model, so unmasking them is always wrong.
    for j in range(n):
        if turn_ids[j] == SENTINEL_ID and not matched[j]:
            mask[j] = 0
    return mask


@register("latent_agent")
class LatentAgentLoop(AgentLoopBase):
    """Multi-turn search rollout for ALAR latent-reasoning models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length

        base_url = os.environ.get(
            "LATENT_RETRIEVAL_URL", "http://127.0.0.1:8000",
        ).rstrip("/")
        self.retrieval_url = (
            base_url if base_url.endswith("/retrieve") else base_url + "/retrieve"
        )
        self.max_turns = int(os.environ.get("LATENT_MAX_TURNS", "6"))
        self.topk = int(os.environ.get("LATENT_TOPK", "3"))
        self.max_doc_chars = int(os.environ.get("LATENT_MAX_DOC_CHARS", "500"))
        self.K = int(os.environ.get("LATENT_NUM_LATENT", "4"))

        self.latent_open_ids = list(self.tokenizer.encode(
            "<latent>", add_special_tokens=False))
        self.latent_close_ids = list(self.tokenizer.encode(
            "</latent>", add_special_tokens=False))
        if not self.latent_open_ids or not self.latent_close_ids:
            raise RuntimeError(
                f"latent agent loop: <latent>={self.latent_open_ids} "
                f"</latent>={self.latent_close_ids} not encodable"
            )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        global _dbg_counter
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)
        input_len = len(prompt_ids)
        _dbg(f"=== NEW ROLLOUT prompt_len={input_len} sampling={dict(sampling_params)}")

        sp = dict(sampling_params)
        sp.setdefault("stop", [])
        sp["stop"] = list(sp["stop"]) + ["</search>", "</answer>"]
        sp["include_stop_str_in_output"] = True

        response_ids: List[int] = []
        response_mask: List[int] = []
        response_logprobs: List[float] = []
        # Sparse list of (response_position, z_tensor) pairs accumulated across
        # all turns. Densified into a [T, H] tensor below for AgentLoopOutput.
        latent_z_pairs: List[Tuple[int, torch.Tensor]] = []
        metrics: dict[str, Any] = {}
        num_turns = 0
        cur_prompt = list(prompt_ids)

        async with aiohttp.ClientSession() as session:
            for turn in range(self.max_turns):
                if len(response_mask) >= self.response_length:
                    break
                num_turns += 1

                turn_request_id = uuid4().hex
                with simple_timer("generate_sequences", metrics):
                    out: TokenOutput = await self.server_manager.generate(
                        request_id=turn_request_id,
                        prompt_ids=cur_prompt,
                        sampling_params=sp,
                    )
                turn_ids = list(out.token_ids)
                turn_mask = _build_response_mask(
                    turn_ids, self.latent_open_ids, self.latent_close_ids, self.K,
                )
                turn_lp = (list(out.log_probs)
                           if out.log_probs is not None
                           else [0.0] * len(turn_ids))
                if len(turn_lp) != len(turn_ids):
                    turn_lp = [0.0] * len(turn_ids)
                _dbg(f"TURN {turn} stop={out.stop_reason} len={len(turn_ids)} "
                     f"mask_sum={sum(turn_mask)}")

                # Pull z values for this turn (written by the vLLM plugin).
                # Map them to sentinel positions in turn_ids, then translate
                # to positions in the full response by adding the offset.
                turn_z = _read_and_clear_z_file(turn_request_id)
                if turn_z is not None and turn_z.numel() > 0:
                    turn_offset = len(response_ids)
                    sentinel_positions = _build_turn_z_positions(turn_ids, turn_z.shape[0])
                    for k, p_in_turn in enumerate(sentinel_positions):
                        latent_z_pairs.append((turn_offset + p_in_turn, turn_z[k]))

                response_ids.extend(turn_ids)
                response_mask.extend(turn_mask)
                response_logprobs.extend(turn_lp)
                cur_prompt = cur_prompt + turn_ids

                turn_text = self.tokenizer.decode(turn_ids)
                if ANSWER_RE.search(turn_text):
                    _dbg(f"TURN {turn} ANSWER → break")
                    break
                m = SEARCH_RE.search(turn_text)
                if not m:
                    _dbg(f"TURN {turn} no <search>/<answer> → break")
                    break
                query = m.group(1).strip()
                _dbg(f"TURN {turn} q={query[:80]}")

                docs = await _retrieve(
                    session, self.retrieval_url, query,
                    self.topk, self.max_doc_chars,
                )
                info_text = f"\n\n<information>{docs}</information>\n\n"
                info_ids = self.tokenizer.encode(info_text, add_special_tokens=False)
                if len(response_mask) + len(info_ids) >= self.response_length:
                    break
                cur_prompt = cur_prompt + info_ids
                response_ids.extend(info_ids)
                response_mask.extend([0] * len(info_ids))
                response_logprobs.extend([0.0] * len(info_ids))

        response_ids = response_ids[:self.response_length]
        response_mask = response_mask[:self.response_length]
        response_logprobs = response_logprobs[:self.response_length]

        # Densify the captured z pairs into a [response_len, H] tensor for
        # the actor's compute_log_prob shortcut. Drop any pair whose response
        # position fell past the truncation boundary.
        latent_z_dense: torch.Tensor | None = None
        if latent_z_pairs:
            kept = [(p, z) for (p, z) in latent_z_pairs if p < len(response_ids)]
            if kept:
                H = kept[0][1].shape[-1]
                latent_z_dense = torch.zeros(len(response_ids), H, dtype=torch.bfloat16)
                for p, z in kept:
                    latent_z_dense[p] = z.to(torch.bfloat16)

        if _dbg_counter < LOG_FIRST_N:
            full = self.tokenizer.decode(response_ids)
            _dbg(f"=== FULL turns={num_turns} len={len(response_ids)}")
            _dbg(full[:2000])
            _dbg(f"latent_z pairs captured: {len(latent_z_pairs)}, "
                 f"kept: {0 if latent_z_dense is None else int((latent_z_dense.abs().sum(-1) > 0).sum().item())}")
        _dbg_counter += 1

        extra_fields: dict[str, Any] = {}
        if latent_z_dense is not None:
            # Stash in extra_fields so AgentLoopOutput.as_dict() carries it
            # through to the DataProto pipeline. Downstream the engine reads
            # `latent_z` from the micro-batch and passes to model.forward.
            extra_fields["latent_z"] = latent_z_dense

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )

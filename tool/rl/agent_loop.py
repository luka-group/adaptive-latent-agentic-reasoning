"""Tool-domain agent loop for verl GRPO rollouts.

Single-shot generation: one assistant turn, stop on `<|im_end|>`. No
tool execution or multi-turn replay — the BFCL eval distribution we're
optimizing toward is single-shot, and ToolMind training already taught
single-shot patterns. (Multi-turn replay against ToolMind's recorded
`inter_turns` is a clean follow-on, not implemented here.)

The vLLM plugin (`alar.vllm_plugin`) drives the `<latent>••••</latent>`
state machine autonomously: on `<latent>` emission it splices K
projector outputs at the sentinel positions, then lets the model emit
`</latent>` naturally. We capture the per-request z values written by
the plugin and pass them back as `extra_fields["latent_z"]` so the
actor's compute_log_prob can skip the K-iter recompute
(LATENT_CACHED_Z=1 in scripts/train_stage2.sh).

response_mask convention (per token):
  1 = assistant-generated, contributes to PPO loss
  0 = K sentinel positions inside `<latent>` blocks

Prompts arrive pre-rendered: `prepare_rl_data.py` flattens the chat
template (system + tools + user + `<|im_start|>assistant\\n`) into a
single string in the parquet `prompt` field, so we tokenize raw rather
than re-applying the chat template (which would double-render the tools
block).

Env vars:
  LATENT_NUM_LATENT      default 4
  LATENT_DEBUG_FIRST_N   default 0 (verbose log for first N rollouts)
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Tuple
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

_LATENT_Z_DIR = os.environ.get("LATENT_Z_DIR", "/tmp/latent_z")

# Install verl-side patches so the request_id propagation to vLLM survives
# (the deferred install hook in LatentQwenForCausalLM doesn't fire inside
# the AgentLoopWorker process — no alar.modeling import there).
try:
    from alar.rl.install_patches import install as _alar_install
    _alar_install()
except Exception as _e:
    import traceback
    print(f"[tool-agent] WARN: install_patches failed: {_e}\n{traceback.format_exc()}",
          flush=True)

logger = logging.getLogger(__file__)

LOG_FIRST_N = int(os.environ.get("LATENT_DEBUG_FIRST_N", "0"))
_dbg_counter = 0

SENTINEL_ID = 6667  # `•` bullet token


def _dbg(msg: str):
    if _dbg_counter < LOG_FIRST_N:
        print(f"[tool-agent] {msg}", flush=True)


def _read_and_clear_z_file(request_id: str) -> torch.Tensor | None:
    """Load the per-request z file written by the vLLM plugin and remove
    it. vLLM augments our `request_id` with a suffix internally, so we
    glob the prefix."""
    import glob
    pattern = os.path.join(_LATENT_Z_DIR, f"{request_id}*.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning("[tool-agent] %d z files match prefix %s — using newest",
                       len(matches), request_id)
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = matches[0]
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        z_seq = data.get("z_seq") if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("[tool-agent] failed to load %s: %s", path, e)
        z_seq = None
    for p in matches:
        try:
            os.remove(p)
        except OSError:
            pass
    return z_seq


def _build_response_mask(
    turn_ids: List[int],
    latent_open_ids: List[int],
    latent_close_ids: List[int],
    K: int,
) -> List[int]:
    """Mask=1 by default; mask=0 at K sentinel positions inside each
    `<latent>` block. Same logic as search agent_loop._build_response_mask.
    """
    n = len(turn_ids)
    mask = [1] * n
    if K <= 0:
        return mask
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
    for j in range(n):
        if turn_ids[j] == SENTINEL_ID and not matched[j]:
            mask[j] = 0
    return mask


@register("tool_latent_agent")
class ToolLatentAgentLoop(AgentLoopBase):
    """Single-shot tool-calling rollout for ALAR latent-reasoning models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.K = int(os.environ.get("LATENT_NUM_LATENT", "4"))

        self.latent_open_ids = list(self.tokenizer.encode(
            "<latent>", add_special_tokens=False))
        self.latent_close_ids = list(self.tokenizer.encode(
            "</latent>", add_special_tokens=False))
        if not self.latent_open_ids or not self.latent_close_ids:
            raise RuntimeError(
                f"tool agent loop: <latent>={self.latent_open_ids} "
                f"</latent>={self.latent_close_ids} not encodable"
            )
        # Qwen3 EOS = <|im_end|>; we stop here per assistant turn.
        self.eos_token_id = self.tokenizer.eos_token_id

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        global _dbg_counter
        # `raw_prompt` is a list of messages produced by verl. Our
        # prepare_rl_data writes the FULL pre-rendered prompt (system +
        # tools + user + `<|im_start|>assistant\\n`) as the content of a
        # single user-role message. Decode it raw so we don't double-
        # wrap with the chat template.
        messages = list(kwargs["raw_prompt"])
        prerendered = messages[0]["content"] if messages else ""
        prompt_ids = self.tokenizer.encode(prerendered, add_special_tokens=False)
        input_len = len(prompt_ids)
        _dbg(f"=== NEW ROLLOUT prompt_len={input_len} sampling={dict(sampling_params)}")

        sp = dict(sampling_params)
        # Force a single-shot rollout: stop on <|im_end|>. Tokens.
        existing_stop_ids = list(sp.get("stop_token_ids") or [])
        if self.eos_token_id is not None and self.eos_token_id not in existing_stop_ids:
            existing_stop_ids.append(self.eos_token_id)
        sp["stop_token_ids"] = existing_stop_ids
        sp["include_stop_str_in_output"] = True

        metrics: dict[str, Any] = {}
        turn_request_id = uuid4().hex
        with simple_timer("generate_sequences", metrics):
            out: TokenOutput = await self.server_manager.generate(
                request_id=turn_request_id,
                prompt_ids=prompt_ids,
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
        _dbg(f"STOP={out.stop_reason} len={len(turn_ids)} mask_sum={sum(turn_mask)}")

        # Pull and densify z values for the assistant turn.
        latent_z_pairs: List[Tuple[int, torch.Tensor]] = []
        turn_z = _read_and_clear_z_file(turn_request_id)
        if turn_z is not None and turn_z.numel() > 0:
            sentinel_positions = [k for k, t in enumerate(turn_ids) if t == SENTINEL_ID]
            for k, p in enumerate(sentinel_positions[: turn_z.shape[0]]):
                latent_z_pairs.append((p, turn_z[k]))

        response_ids = turn_ids[:self.response_length]
        response_mask = turn_mask[:self.response_length]
        response_logprobs = turn_lp[:self.response_length]

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
            _dbg(f"=== FULL len={len(response_ids)}")
            _dbg(full[:2000])
            _dbg(f"latent_z pairs captured: {len(latent_z_pairs)}")
        _dbg_counter += 1

        extra_fields: dict[str, Any] = {}
        if latent_z_dense is not None:
            extra_fields["latent_z"] = latent_z_dense

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=1,
            metrics=metrics,
            extra_fields=extra_fields,
        )

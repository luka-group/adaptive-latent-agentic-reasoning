"""vLLM plugin for ALAR latent rollouts.

At rollout time the saved ckpt is a stock Qwen2.5/Qwen3 (we sidecar the
projector to `projector.pt` so vLLM loads it as the plain base model).
This plugin attaches the projector to the loaded model and patches
`GPUModelRunner` to splice projector outputs into the K sentinel
positions whenever the model emits the `<latent>` literal.

State per request:

    IDLE
      └─ model decodes `<latent>` 3-piece BPE literal ─┐
                                                       ▼
            step 1 (after detection): forward consumes the literal's
                last token. h = sample_hidden. z_1 = projector(h).
                Stash z_1; replace sampled token with `•` (sentinel,
                id 6667). remaining = K.
                                                       │
                                                       ▼
            step k (2..K): preprocess overrides last position's input
                embedding with the stashed z_{k-1}. Forward consumes
                z_{k-1}. z_k = projector(h). Stash z_k; replace sampled
                token with `•`. remaining -= 1.
                                                       │
                                                       ▼
            step K+1: preprocess overrides last position's embedding
                with z_K. Forward consumes z_K. DO NOT project. FORCE
                output = `</latent>`-piece-0 (522). Schedule the
                remaining close pieces. remaining = 0.
                                                       │
                                                       ▼
            steps K+2, K+3: force output = `</latent>`-pieces 1 (50023)
                and 2 (29). KV cache for each piece is computed by the
                model's normal forward over the forced input_ids.
                                                       │
                                                       ▼
                                                     IDLE

The model's training distribution already conditions it to emit
`</latent>` after the K latent positions most of the time, but the
force-emit makes that 100% so a single missed close doesn't cascade
into a runaway trajectory.

Total positions advanced in the KV cache per latent block: K (with
projector-output embeddings at those positions) + 3 (`</latent>` BPE
pieces, normal sampling). This exactly matches the SFT layout.

Activation: vLLM auto-imports this module via `vllm.general_plugins`
when the package is pip-installed (see pyproject.toml). Set
`LATENT_VLLM=0` to disable.
"""

from __future__ import annotations

import logging
import os

# Workaround: tvm_ffi (a transitive dep of vLLM v1 via flashinfer)
# optionally loads a pre-built `torch_c_dlpack_ext` .so whose C++ ABI can
# mismatch the installed torch wheels, breaking the vLLM import. Skipping
# the optional extension falls back to the default DLPack path (one-time
# JIT cost, no correctness impact). setdefault preserves user override.
os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

LATENT_OPEN_TEXT = "<latent>"
LATENT_CLOSE_TEXT = "</latent>"
SENTINEL_ID = 6667  # '•' — single-piece, non-merging BPE; matches SFT data layer.

# Per-request z capture: vLLM's projector produces K z values per latent block.
# We accumulate them on the request state and persist to disk so the agent_loop
# (a separate process) can read them after generate() returns. The actor's
# compute_log_prob then passes the cached z's to LatentQwen2Model.forward as
# `cached_latent_z`, skipping its own K-iter recompute.
_LATENT_Z_DIR = os.environ.get("LATENT_Z_DIR", "/tmp/latent_z")


def _persist_z_seq(req_id: str, z_seq) -> None:
    """Write the per-request z sequence to /tmp/latent_z/<req_id>.pt.
    Overwrites on each call (one extra small write per K-iter step). Cheap
    (~8 KB × K), no real cost, robust across process boundaries.

    Note: vLLM v1 augments `req_id` with a `-<8hex>` suffix internally
    (visible as `input_batch.req_ids[i]`), so the actual filename is
    `<caller_request_id>-<8hex>.pt`. agent_loop's reader globs the prefix.
    """
    if not z_seq:
        return
    try:
        os.makedirs(_LATENT_Z_DIR, exist_ok=True)
        path = os.path.join(_LATENT_Z_DIR, f"{req_id}.pt")
        stacked = torch.stack([z.contiguous() for z in z_seq], dim=0)
        torch.save({"z_seq": stacked}, path + ".tmp")
        os.replace(path + ".tmp", path)
    except Exception as e:
        log.warning("[latent vllm] _persist_z_seq(%s) failed: %s", req_id, e)


def _env_enabled() -> bool:
    return os.getenv("LATENT_VLLM", "").lower() not in {"0", "false", "no", "n", "off"}


class _LatentProjector(nn.Module):
    """Inference-side mirror of `alar.modeling.LatentProjector`.

    Plain nn modules (no LatentQwen2 config), so vLLM's weight loader
    treats it as a generic MLP attached to the stock Qwen2 model.
    """

    def __init__(self, hidden_size: int, prj_dim: int = 0,
                 dropout: float = 0.0, no_ln: bool = False):
        super().__init__()
        prj_dim = int(prj_dim) if prj_dim else int(hidden_size)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prj_dim),
            nn.GELU(),
            nn.Linear(prj_dim, hidden_size),
        ]
        if not no_ln:
            layers.append(nn.LayerNorm(hidden_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_projector_sidecar(projector: nn.Module, model_path: str) -> bool:
    p = os.path.join(model_path, "projector.pt")
    if not os.path.exists(p):
        log.warning(f"[latent vllm]: no projector.pt at {model_path}")
        return False
    sd = torch.load(p, map_location="cpu")
    dev = next(projector.parameters()).device
    dtype = next(projector.parameters()).dtype
    sd_cast = {k: v.to(device=dev, dtype=dtype) for k, v in sd.items()}
    missing, unexpected = projector.load_state_dict(sd_cast, strict=False)
    if missing or unexpected:
        log.warning(f"[latent vllm]: projector load missing={missing} unexpected={unexpected}")
    log.info(f"[latent vllm]: loaded projector sidecar ({len(sd)} tensors)")
    return True


def _maybe_reload_sync(model_runner) -> None:
    """RL hot-reload: re-read projector weights from the sync file the
    trainer writes after each PPO update. Mtime-gated to skip on idle
    steps."""
    path = os.environ.get("LATENT_SYNC_PATH")
    if not path or not os.path.exists(path):
        return
    try:
        cur_mtime = os.path.getmtime(path)
    except OSError:
        return
    last = getattr(model_runner, "_latent_sync_mtime", None)
    if last is not None and cur_mtime <= last:
        return
    projector = getattr(model_runner, "_latent_projector", None)
    if projector is None:
        return
    try:
        sd = torch.load(path, map_location="cpu")
    except Exception as e:
        log.warning(f"[latent vllm] sync load failed at {path}: {e}")
        return
    proj_sd = sd.get("projector") if isinstance(sd, dict) else sd
    if proj_sd:
        dev = next(projector.parameters()).device
        dtype = next(projector.parameters()).dtype
        cast = {k: v.to(device=dev, dtype=dtype) for k, v in proj_sd.items()}
        projector.load_state_dict(cast, strict=False)
    model_runner._latent_sync_mtime = cur_mtime
    log.info(f"[latent vllm] sync: reloaded projector from {path}")


def _ensure_runner_state(self) -> None:
    if getattr(self, "_latent_ready", False) or getattr(self, "_latent_disabled", False):
        return

    tok_name = getattr(self.model_config, "tokenizer", None) or self.model_config.model
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name, trust_remote_code=self.model_config.trust_remote_code,
    )
    latent_open_ids = tuple(tokenizer.encode(LATENT_OPEN_TEXT, add_special_tokens=False))
    latent_close_ids = tuple(tokenizer.encode(LATENT_CLOSE_TEXT, add_special_tokens=False))
    if not latent_open_ids or not latent_close_ids:
        log.warning("[latent vllm]: `<latent>`/`</latent>` tokenizes empty; disabling plugin.")
        self._latent_disabled = True
        return

    raw_model = (self.get_model() if callable(getattr(self, "get_model", None))
                 else self.model)
    h = int(raw_model.config.hidden_size)
    prj_dim = int(os.getenv("LATENT_PRJ_DIM", "0"))
    prj_dropout = float(os.getenv("LATENT_PRJ_DROPOUT", "0.0"))
    prj_no_ln = os.getenv("LATENT_PRJ_NO_LN", "").lower() in {"1", "true", "yes"}
    dtype = next(raw_model.parameters()).dtype
    projector = _LatentProjector(h, prj_dim=prj_dim,
                                 dropout=prj_dropout, no_ln=prj_no_ln)
    projector = projector.to(device=self.device, dtype=dtype)
    _load_projector_sidecar(projector, self.model_config.model)

    self._latent_projector = projector
    self._latent_open_ids: Tuple[int, ...] = latent_open_ids
    self._latent_close_ids: Tuple[int, ...] = latent_close_ids
    self._latent_num_latent = int(os.getenv("LATENT_NUM_LATENT", "4"))
    self._latent_override_mask: List[bool] = []
    self._latent_step_skip_sampling: List[bool] = []
    self._latent_ready = True
    log.info(
        "[latent vllm]: ready (latent_open=%s latent_close=%s K=%d sentinel=%d)",
        latent_open_ids, latent_close_ids, self._latent_num_latent, SENTINEL_ID,
    )


def _patch_vllm_runner() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GPUModelRunner, "_latent_patched", False):
        return

    orig_preprocess = GPUModelRunner._preprocess
    orig_execute = GPUModelRunner.execute_model
    orig_update_after = GPUModelRunner._update_states_after_model_execute

    def patched_preprocess(self, scheduler_output, num_input_tokens,
                           intermediate_tensors=None):
        out = orig_preprocess(self, scheduler_output, num_input_tokens,
                              intermediate_tensors)
        if not _env_enabled():
            return out
        _ensure_runner_state(self)
        if getattr(self, "_latent_disabled", False):
            return out
        _maybe_reload_sync(self)

        input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec = out
        num_reqs = self.input_batch.num_reqs
        self._latent_override_mask = [False] * num_reqs

        if input_ids is None:
            return out

        override_mask = [False] * num_reqs
        req_ids = self.input_batch.req_ids
        for i in range(num_reqs):
            rs = self.requests[req_ids[i]]
            if getattr(rs, "_latent_next_embed", None) is not None:
                override_mask[i] = True
        self._latent_override_mask = override_mask
        if not any(override_mask):
            return out

        # Prefill re-emits the K sentinel ids; we do NOT recompute the
        # projector at prefill of turn T+1. vLLM prefix-caching reuses the
        # K/V from turn T's decode (where the projector outputs were
        # spliced in). The decode-time per-position override below is the
        # only embedding substitution this plugin does.
        embed_fn = getattr(self.model, "embed_input_ids", None)
        if embed_fn is None:
            embed_fn = getattr(getattr(self.model, "model", None),
                               "embed_input_ids", None)
        if embed_fn is None:
            raise RuntimeError(
                "[latent vllm] requires `embed_input_ids` on the vLLM model.")
        embeds = embed_fn(input_ids)
        qsl = self.query_start_loc.np
        for i in range(num_reqs):
            if not override_mask[i]:
                continue
            rs = self.requests[req_ids[i]]
            e = rs._latent_next_embed
            rs._latent_next_embed = None
            last_idx = int(qsl[i + 1] - 1)
            if last_idx < 0:
                continue
            embeds[last_idx].copy_(
                e.to(device=embeds.device, dtype=embeds.dtype).view(-1)
            )
        self.inputs_embeds.gpu[:num_input_tokens].copy_(embeds)
        return (None, self.inputs_embeds.gpu[:num_input_tokens], positions,
                intermediate_tensors, model_kwargs, ec)

    def patched_execute_model(self, scheduler_output, intermediate_tensors=None):
        out = orig_execute(self, scheduler_output, intermediate_tensors)
        if not _env_enabled():
            return out
        if self.execute_model_state is None:
            return out
        _ensure_runner_state(self)
        if getattr(self, "_latent_disabled", False):
            return out

        state = self.execute_model_state
        sample_hs = state.sample_hidden_states  # [num_reqs, hidden]
        proj = self._latent_projector(sample_hs)

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        override_mask = getattr(self, "_latent_override_mask", [False] * num_reqs)
        skip_mask = [False] * num_reqs

        for i in range(num_reqs):
            rs = self.requests[req_ids[i]]

            if getattr(rs, "_latent_pending_first", False):
                # Just consumed the `<latent>` literal's last token.
                # h = hidden at trigger; z_1 = projector(h).
                rs._latent_pending_first = False
                rs._latent_next_embed = proj[i].detach()
                # Capture for export to actor's compute_log_prob shortcut.
                if not hasattr(rs, "_captured_z_seq") or rs._captured_z_seq is None:
                    rs._captured_z_seq = []
                rs._captured_z_seq.append(proj[i].detach().to(torch.bfloat16).cpu())
                _persist_z_seq(req_ids[i], rs._captured_z_seq)
                skip_mask[i] = True
                continue

            if override_mask[i]:
                remaining = int(getattr(rs, "_latent_remaining", 0) or 0)
                assert remaining > 0, "consumed embed but remaining <= 0"
                remaining -= 1
                rs._latent_remaining = remaining
                if remaining > 0:
                    rs._latent_next_embed = proj[i].detach()
                    if not hasattr(rs, "_captured_z_seq") or rs._captured_z_seq is None:
                        rs._captured_z_seq = []
                    rs._captured_z_seq.append(proj[i].detach().to(torch.bfloat16).cpu())
                    _persist_z_seq(req_ids[i], rs._captured_z_seq)
                else:
                    rs._latent_next_embed = None
                skip_mask[i] = True
                continue

        self._latent_step_skip_sampling = skip_mask
        return out

    def patched_update_after(self, output_token_ids: torch.Tensor, scheduler_output):
        if (not _env_enabled() or getattr(self, "_latent_disabled", False)
                or not getattr(self, "_latent_ready", False)):
            return orig_update_after(self, output_token_ids, scheduler_output)

        if output_token_ids.dim() != 2 or output_token_ids.size(1) != 1:
            raise NotImplementedError(
                "[latent vllm]: only 1-token decode steps supported."
            )

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        skip_mask = (getattr(self, "_latent_step_skip_sampling", None)
                     or [False] * num_reqs)
        open_ids = self._latent_open_ids
        close_ids = self._latent_close_ids
        L = len(open_ids)
        K = self._latent_num_latent
        debug = bool(os.environ.get("LATENT_VLLM_DEBUG"))

        # First pass: handle mid-close-emit, then run open-detection for
        # all non-reasoning, non-close-emit, non-skip-mask steps.
        for i in range(num_reqs):
            rid = req_ids[i]
            rs = self.requests[rid]

            # If mid-close-emit, force the next `</latent>` BPE piece.
            close_remaining = int(getattr(rs, "_latent_close_remaining", 0) or 0)
            if close_remaining > 0:
                piece_idx = len(close_ids) - close_remaining
                output_token_ids[i, 0] = int(close_ids[piece_idx])
                rs._latent_close_remaining = close_remaining - 1
                if debug:
                    log.info("[latent vllm]: force-emit close piece %d/%d (id=%d) on req %s",
                             piece_idx + 1, len(close_ids),
                             int(close_ids[piece_idx]), rid)
                continue

            # Skip detection for any reasoning/sentinel step (skip_mask
            # covers both the post-trigger first step and K-sentinel
            # steps; the K-th of those becomes the close-emit trigger
            # below). Without this guard the buf would record the model's
            # (about-to-be-overridden) sample.
            if skip_mask[i]:
                continue

            just_sampled = int(output_token_ids[i, 0].item())
            in_reasoning = (
                getattr(rs, "_latent_pending_first", False)
                or getattr(rs, "_latent_next_embed", None) is not None
                or int(getattr(rs, "_latent_remaining", 0) or 0) > 0
            )
            if in_reasoning:
                continue

            buf = getattr(rs, "_latent_buf", None)
            if buf is None or buf.maxlen != L:
                buf = deque(maxlen=L)
                rs._latent_buf = buf
            buf.append(just_sampled)
            if len(buf) == L and tuple(buf) == open_ids:
                if debug:
                    log.info("[latent vllm]: <latent> detected on req %s", rid)
                rs._latent_remaining = int(K)
                rs._latent_pending_first = True
                buf.clear()

        # Second pass: dummy `•` for K-sentinel steps; at the K-th step
        # (remaining hit 0 in execute), force-emit close_ids[0] and
        # schedule the remaining close pieces so the next len(close_ids)-1
        # steps emit `</latent>` deterministically.
        for i in range(num_reqs):
            if not skip_mask[i]:
                continue
            rs = self.requests[req_ids[i]]
            remaining = int(getattr(rs, "_latent_remaining", 0) or 0)
            pending_first = getattr(rs, "_latent_pending_first", False)
            if remaining == 0 and not pending_first:
                output_token_ids[i, 0] = int(close_ids[0])
                if len(close_ids) > 1:
                    rs._latent_close_remaining = len(close_ids) - 1
                if debug:
                    log.info("[latent vllm]: K-th step on req %s — "
                             "force-emit close piece 1/%d (id=%d), %d more queued",
                             req_ids[i], len(close_ids), int(close_ids[0]),
                             max(0, len(close_ids) - 1))
            else:
                output_token_ids[i, 0] = SENTINEL_ID

        self._latent_step_skip_sampling = []
        return orig_update_after(self, output_token_ids, scheduler_output)

    GPUModelRunner._preprocess = patched_preprocess
    GPUModelRunner.execute_model = patched_execute_model
    GPUModelRunner._update_states_after_model_execute = patched_update_after
    GPUModelRunner._latent_patched = True


def register() -> None:
    """vLLM `vllm.general_plugins` entry point. Idempotent."""
    if not _env_enabled():
        return
    log.info("[latent vllm] registering runner patches")
    print("[[latent vllm]] register() running, patching GPUModelRunner", flush=True)
    try:
        _patch_vllm_runner()
        print("[[latent vllm]] runner patch applied", flush=True)
    except Exception as e:
        log.warning(f"[latent vllm]: runner patch failed: {e}")
        print(f"[[latent vllm]] runner patch failed: {e}", flush=True)

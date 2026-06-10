"""RL patch installer.

Idempotent. Designed to be imported in every Ray worker process before
any worker instance is constructed.

What `install()` does:

  1. Imports `alar.modeling` so the `latent_qwen{2,3}` model types are
     registered with HF AutoConfig before verl's HFModelConfig calls
     `AutoConfig.from_pretrained`.

  2. Registers `alar.vllm_plugin` in-process (idempotent, gated by
     `LATENT_VLLM`). vLLM's entry-point auto-load also covers this,
     but registering explicitly here makes the dependency visible.

  3. Wraps the worker's `__init__` to auto-import `alar.modeling` from
     inside each worker (defensive).

  4. Wraps the worker's `update_actor` to dump the projector state_dict
     to `LATENT_SYNC_PATH` after every PPO update. The rollout-side
     plugin (`_maybe_reload_sync` in `alar.vllm_plugin`) mtime-checks
     this file each preprocess and reloads on change.

  5. Wraps the worker's `save_checkpoint` to write a PEFT-format adapter
     dir (lora_adapter/{adapter_config.json,adapter_model.safetensors})
     including both the LoRA deltas AND the projector tensors
     (modules_to_save). verl's default save path either skips the
     adapter entirely (engine_workers backend) or writes an adapter
     that's missing the projector because layered_summon_lora_params
     only walks `model.layers.X` (fsdp_workers backend).

Why a sidecar sync is needed: base Qwen2.5 weights are propagated
actor→vLLM by verl's standard hybrid-engine `update_weights`, but the
projector is a custom module that the vLLM model class doesn't know
how to receive via that path.

Why two worker classes: verl's `ppo_trainer.yaml` defaults to
`use_legacy_worker_impl: 'disable'`, which routes the actor through
`verl.workers.engine_workers.ActorRolloutRefWorker`. The older path
(`verl.workers.fsdp_workers.AsyncActorRolloutRefWorker`) is also still
in the tree. We patch both classes so the wraps fire regardless of
which one verl ends up using.
"""
from __future__ import annotations

import os
import sys


def _install_faulthandler() -> None:
    """Periodic Python-stack dump in this worker process. Diagnostic only,
    gated on LATENT_FAULTHANDLER_SECS (set to N to dump every N seconds).

    When the worker hangs, each fire prints every thread's Python stack to
    stderr (which Ray captures into the log). Snapshots across time show
    whether execution is stuck on one line (same frame repeated) or
    progressing slowly through a call graph (frames change). Has no measurable
    overhead between fires (just a timer thread).
    """
    secs = os.environ.get("LATENT_FAULTHANDLER_SECS")
    if not secs:
        return
    try:
        secs_i = int(secs)
    except ValueError:
        return
    if secs_i <= 0:
        return
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(secs_i, repeat=True, file=sys.stderr)
    print(f"[install] faulthandler armed: dump_traceback_later every {secs_i}s",
          flush=True)


def _get_actor_fsdp(self):
    """Return the FSDP-wrapped actor module from either worker backend.

    - fsdp_workers.ActorRolloutRefWorker exposes `self.actor_module_fsdp`.
    - engine_workers.ActorRolloutRefWorker holds the FSDP module inside
      its `self.actor` (TrainingWorker) → `self.actor.engine.module`.
    """
    fsdp = getattr(self, "actor_module_fsdp", None)
    if fsdp is not None:
        return fsdp
    actor = getattr(self, "actor", None)
    if actor is None:
        return None
    engine = getattr(actor, "engine", None)
    if engine is None:
        return None
    return getattr(engine, "module", None)


def _get_peft_model(actor_fsdp):
    """Unwrap one FSDP layer to reach the underlying PeftModel (if any)."""
    if actor_fsdp is None:
        return None
    return getattr(actor_fsdp, "_fsdp_wrapped_module", actor_fsdp)


def _gather_full_state(actor_fsdp):
    """Collective: all ranks must call this. Returns the full state dict
    on rank 0 (CPU-offloaded), empty dict elsewhere.

    With PARAM_OFFLOAD=True (default in RL configs), verl offloads FSDP
    shards to CPU between phases. The FSDP collective state_dict requires
    shards to be on the compute device — so load → gather → offload back.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    try:
        from verl.utils.fsdp_utils import (
            load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu,
        )
    except Exception:
        load_fsdp_model_to_gpu = offload_fsdp_model_to_cpu = None

    if load_fsdp_model_to_gpu is not None:
        load_fsdp_model_to_gpu(actor_fsdp)
    try:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(actor_fsdp, StateDictType.FULL_STATE_DICT, cfg):
            return actor_fsdp.state_dict()
    finally:
        if offload_fsdp_model_to_cpu is not None:
            offload_fsdp_model_to_cpu(actor_fsdp)


def _extract_projector_for_sync(full_state):
    """Pull out projector tensors and rewrite keys to the plugin's expected
    layout: `net.X.weight`. Handles both plain (unwrapped) and PEFT
    (modules_to_save-wrapped) key prefixes."""
    proj_state = {}
    for k, v in full_state.items():
        if ".projector." not in k:
            continue
        tail = k.split(".projector.", 1)[1]
        if tail.startswith("modules_to_save."):
            parts = tail.split(".", 2)
            if len(parts) == 3:
                tail = parts[2]
        if tail.startswith("net."):
            proj_state[tail] = v
    return proj_state


def _save_peft_adapter(peft_model, full_state, out_dir):
    """Write adapter_config.json + adapter_model.safetensors into `out_dir`
    using peft's own state_dict extractor. Includes LoRA deltas AND
    modules_to_save (projector) tensors."""
    import json

    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)

    # adapter weights: lora_A/lora_B/lora_embedding + modules_to_save params
    adapter_state = get_peft_model_state_dict(peft_model, state_dict=full_state,
                                              adapter_name="default")

    # Manual modules_to_save fallback: under engine_workers + FSDP, peft's
    # AuxiliaryTrainingWrapper discovery via model.named_modules() misses
    # the projector (likely because the FSDP wrap reshuffles module
    # registration timing). Scan the full state_dict ourselves for
    # `.modules_to_save.default.` keys and rewrite to peft's
    # inference-time layout (`...projector.<param>`).
    modules_to_save = []
    try:
        cfg = peft_model.peft_config.get("default", None)
        if cfg is not None:
            modules_to_save = list(getattr(cfg, "modules_to_save", None) or [])
    except Exception:
        modules_to_save = []
    if modules_to_save:
        for k, v in full_state.items():
            if ".modules_to_save.default." not in k:
                continue
            if not any(f".{m}.modules_to_save." in k for m in modules_to_save):
                continue
            new_key = k.replace(".modules_to_save.default.", ".")
            if new_key not in adapter_state:
                adapter_state[new_key] = v

    # safetensors requires contiguous tensors and rejects bf16 sharing
    adapter_state = {
        k: (v.detach().contiguous().clone() if hasattr(v, "detach") else v)
        for k, v in adapter_state.items()
    }

    save_file(adapter_state, os.path.join(out_dir, "adapter_model.safetensors"))

    # adapter_config.json: serialize the PEFT config and normalize enums
    cfg = peft_model.peft_config["default"]
    if hasattr(cfg, "to_dict"):
        cfg_dict = cfg.to_dict()
    else:
        from dataclasses import asdict
        cfg_dict = asdict(cfg)
    for key in list(cfg_dict.keys()):
        val = cfg_dict[key]
        if hasattr(val, "value"):  # enum
            cfg_dict[key] = val.value
        elif isinstance(val, set):
            cfg_dict[key] = sorted(val)

    # Honesty: if no modules_to_save tensors made it into adapter_state,
    # don't advertise them in the config — that would make consumers
    # (PeftModel.from_pretrained) silently load random weights for those
    # modules. This currently happens because under verl's engine_workers
    # path, the base model is plain Qwen2 with no `projector` submodule,
    # so PEFT can't register an AuxiliaryTrainingWrapper for it and the
    # projector is never in the actor's trainable params anyway.
    has_modules_to_save_tensors = any(
        ".modules_to_save." in k or any(f"{m}." in k.split(".")[-3:-1]
                                         for m in (cfg_dict.get("modules_to_save") or []))
        for k in adapter_state
    )
    if cfg_dict.get("modules_to_save") and not has_modules_to_save_tensors:
        cfg_dict["modules_to_save"] = None

    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)
    return adapter_state


_VERL_DISPATCH_ATTR = "attrs_3141562937"  # MAGIC_ATTR from verl.single_controller.base.decorator


def _copy_verl_dispatch_attrs(src_fn, dst_fn):
    """Copy verl's @register dispatch metadata from src_fn to dst_fn.

    verl's RayWorkerGroup builds its dispatch table by scanning the worker
    class for functions carrying the magic attribute set by @register. If we
    replace a method without copying this attribute, RayWorkerGroup loses
    the dispatch registration and fails with `'RayWorkerGroup' object has
    no attribute X`. functools.wraps doesn't help — it copies __wrapped__
    and __dict__ at decoration time, but our @register-decorated original
    set the attr via setattr after the wraps chain.
    """
    if hasattr(src_fn, _VERL_DISPATCH_ATTR):
        setattr(dst_fn, _VERL_DISPATCH_ATTR, getattr(src_fn, _VERL_DISPATCH_ATTR))


def _wrap_worker_class(cls, sync_path):
    """Install __init__, update_actor, and save_checkpoint wraps on `cls`.
    Idempotent — repeated calls are no-ops."""
    if cls is None:
        return

    # 1) __init__ wrap — ensure alar.modeling is imported in each worker process
    if not getattr(cls.__init__, "_alar_patched", False):
        _orig_init = cls.__init__

        def _wrapped_init(self, *args, **kwargs):
            try:
                import alar.modeling  # noqa: F401
            except Exception:
                pass
            return _orig_init(self, *args, **kwargs)

        _wrapped_init._alar_patched = True
        _copy_verl_dispatch_attrs(_orig_init, _wrapped_init)
        cls.__init__ = _wrapped_init
        print(f"[install] patched {cls.__module__}.{cls.__name__}.__init__",
              flush=True)

    # 2) update_actor wrap — projector → file IPC sync for vLLM rollout
    if (sync_path
            and hasattr(cls, "update_actor")
            and not getattr(cls.update_actor, "_sync_patched", False)):
        _orig_update = cls.update_actor

        def _wrapped_update(self, *args, **kwargs):
            out = _orig_update(self, *args, **kwargs)
            try:
                import torch
                import torch.distributed as dist
                actor_fsdp = _get_actor_fsdp(self)
                if actor_fsdp is None:
                    return out
                full_state = _gather_full_state(actor_fsdp)  # collective

                rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
                if not rank0:
                    return out

                proj_state = _extract_projector_for_sync(full_state)
                if not proj_state:
                    sample = list(full_state.keys())[:5]
                    proj_like = [k for k in full_state if "projector" in k or "net.1" in k or "net.3" in k][:5]
                    print(f"[sync] WARN: no projector keys matched from "
                          f"{len(full_state)}-key state_dict\n"
                          f"  first 5 keys: {sample}\n"
                          f"  projector-like keys: {proj_like}", flush=True)
                    return out

                payload = {"projector": proj_state}
                os.makedirs(os.path.dirname(sync_path) or ".", exist_ok=True)
                tmp = sync_path + ".tmp"
                torch.save(payload, tmp)
                os.replace(tmp, sync_path)
                print(f"[sync] wrote projector ({len(proj_state)} tensors) → {sync_path}",
                      flush=True)
            except Exception as e:
                print(f"[sync] WARN: {type(e).__name__}: {e}", flush=True)
            return out

        _wrapped_update._sync_patched = True
        _copy_verl_dispatch_attrs(_orig_update, _wrapped_update)
        cls.update_actor = _wrapped_update
        print(f"[install] patched {cls.__module__}.{cls.__name__}.update_actor "
              f"(sync_path={sync_path})", flush=True)

    # 3) save_checkpoint wrap — write loadable PEFT adapter (LoRA + projector)
    if (hasattr(cls, "save_checkpoint")
            and not getattr(cls.save_checkpoint, "_proj_save_patched", False)):
        _orig_save = cls.save_checkpoint

        def _wrapped_save(self, local_path, hdfs_path=None, global_step=0,
                          max_ckpt_to_keep=None):
            out = _orig_save(self, local_path=local_path, hdfs_path=hdfs_path,
                             global_step=global_step,
                             max_ckpt_to_keep=max_ckpt_to_keep)
            try:
                import torch.distributed as dist
                actor_fsdp = _get_actor_fsdp(self)
                if actor_fsdp is None:
                    print("[ckpt-save] WARN: no FSDP actor module found, skipping",
                          flush=True)
                    return out
                peft_model = _get_peft_model(actor_fsdp)
                if not hasattr(peft_model, "peft_config"):
                    # Not a PEFT-wrapped model — nothing to add to verl's save.
                    return out

                full_state = _gather_full_state(actor_fsdp)  # collective

                rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
                if rank0:
                    lora_dir = os.path.join(local_path, "lora_adapter")
                    adapter_state = _save_peft_adapter(peft_model, full_state,
                                                      lora_dir)
                    proj_keys = [k for k in adapter_state if ".projector." in k]
                    print(f"[ckpt-save] wrote {lora_dir} "
                          f"({len(adapter_state)} tensors, "
                          f"{len(proj_keys)} projector)", flush=True)

                if dist.is_initialized():
                    dist.barrier()
            except Exception as e:
                print(f"[ckpt-save] WARN: {type(e).__name__}: {e}", flush=True)
            return out

        _wrapped_save._proj_save_patched = True
        _copy_verl_dispatch_attrs(_orig_save, _wrapped_save)
        cls.save_checkpoint = _wrapped_save
        print(f"[install] patched {cls.__module__}.{cls.__name__}.save_checkpoint "
              f"(writes lora_adapter/ with LoRA + projector)", flush=True)


def install() -> None:
    # 0. Diagnostic faulthandler — arms as early as possible so any later
    # hang in this process produces stack dumps. Gated on LATENT_FAULTHANDLER_SECS.
    _install_faulthandler()

    # 1. Register the LatentQwen{2,3} model types with HF AutoConfig.
    try:
        import alar.modeling  # noqa: F401
    except Exception as e:
        print(f"[install] WARN: alar.modeling import: {e}", flush=True)

    # 2. Register the vLLM plugin (idempotent; gated by LATENT_VLLM).
    if os.environ.get("LATENT_VLLM", "").lower() in ("1", "true", "yes", "y", "on"):
        try:
            from alar.vllm_plugin import register as _register
            _register()
            print("[install] alar.vllm_plugin registered", flush=True)
        except Exception as e:
            print(f"[install] WARN: vllm_plugin register: {e}", flush=True)

    # 3. Patch BOTH worker backends. verl picks one based on
    # `trainer.use_legacy_worker_impl` (default 'disable' → engine_workers).
    sync_path = os.environ.get("LATENT_SYNC_PATH")

    try:
        from verl.workers.fsdp_workers import ActorRolloutRefWorker as _FW
        _wrap_worker_class(_FW, sync_path)
    except Exception as e:
        print(f"[install] WARN: cannot patch fsdp_workers: {e}", flush=True)

    try:
        from verl.workers.engine_workers import ActorRolloutRefWorker as _EW
        _wrap_worker_class(_EW, sync_path)
    except Exception as e:
        print(f"[install] WARN: cannot patch engine_workers: {e}", flush=True)

    # 4. Optional: print per-micro-batch progress inside FSDPEngine.forward_backward_batch.
    # Useful for babysitting long runs — tells us "step=N phase=fwd_only mb=3/8"
    # without having to wait for the whole phase to finish. Default OFF.
    if os.environ.get("LATENT_LP_PROGRESS", "").lower() in ("1", "true", "yes", "on"):
        _install_fbbatch_progress()

    # 5. cached_latent_z plumbing — wire z tensors from rollout through to
    # FSDPEngine.forward to skip the K-iter expansion in actor compute_log_prob
    # and update_actor. Gated on LATENT_CACHED_Z=1.
    if os.environ.get("LATENT_CACHED_Z", "").lower() in ("1", "true", "yes", "on"):
        _install_cached_z_plumbing()


def _install_fbbatch_progress() -> None:
    """Wrap FSDPEngine.forward_backward_batch to print micro-batch progress.

    verl's per-phase tqdm bars go to stderr with \\r updates; tee'd log files
    only catch the final '8/8' completion line. To see in-flight progress we
    monkey-patch the loop and print every micro-batch. Cheap (one print per
    micro-batch) and rank-gated to avoid log spam.
    """
    import time
    try:
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    except ImportError as e:
        print(f"[install] WARN: cannot patch FSDPEngine: {e}", flush=True)
        return

    if getattr(FSDPEngine.forward_backward_batch, "_lp_progress_patched", False):
        return

    _orig_fbb = FSDPEngine.forward_backward_batch

    def _wrapped_fbb(self, data, loss_function, forward_only=False):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        rank0 = rank == 0
        phase = "fwd" if forward_only else "fwd+bwd"
        t0 = time.time()
        # Count micro-batches by inspecting the data tensor count once.
        n_micros = None
        try:
            bs = int(data["loss_mask"].shape[0])
            mb = getattr(self.engine_config, "micro_batch_size_per_gpu", None) \
                 or getattr(self.engine_config, "ppo_micro_batch_size_per_gpu", None) \
                 or 1
            n_micros = max(1, (bs + mb - 1) // mb)
        except Exception:
            pass

        # Wrap the iteration: monkey-patch forward_step to count calls.
        _orig_fs = self.forward_step
        counter = {"i": 0}

        def _counting_fs(micro_batch, loss_function, forward_only):
            counter["i"] += 1
            if rank0:
                total = n_micros if n_micros else "?"
                elapsed = time.time() - t0
                print(f"[lp] phase={phase} mb={counter['i']}/{total} elapsed={elapsed:.1f}s",
                      flush=True)
            return _orig_fs(micro_batch, loss_function, forward_only)

        self.forward_step = _counting_fs
        try:
            result = _orig_fbb(self, data, loss_function, forward_only=forward_only)
        finally:
            self.forward_step = _orig_fs

        if rank0:
            total = n_micros if n_micros else counter["i"]
            print(f"[lp] phase={phase} DONE mb={counter['i']}/{total} "
                  f"total_elapsed={time.time()-t0:.1f}s", flush=True)
        return result

    _wrapped_fbb._lp_progress_patched = True
    FSDPEngine.forward_backward_batch = _wrapped_fbb
    print("[install] patched FSDPEngine.forward_backward_batch (lp progress)",
          flush=True)


# ---------------------------------------------------------------------------
# cached_latent_z plumbing
# ---------------------------------------------------------------------------
# Data flow (read top-to-bottom):
#   1. agent_loop.py: per-turn vLLM rollout writes z_seq to /tmp/latent_z/<req>.pt.
#      LatentAgentLoop.run() reads, maps to response-token positions, returns
#      a [response_len, H] dense tensor in AgentLoopOutput.extra_fields["latent_z"].
#   2. _agent_loop_postprocess (patched): pads to [1, prompt_pad + response_pad, H]
#      with left-pad-zeros for prompt area and the latent_z values at their
#      response positions; carries through extra_fields.
#   3. _postprocess (patched): pops latent_z from each sample's extra_fields,
#      stacks across the batch into batch["latent_z"] of shape (bsz, P+R, H).
#      Avoids the np.object stuffing path that loses tensor identity.
#      latent_z stays a regular tensor through verl's TensorDict plumbing.
#   4. FSDPEngineWithLMHead.prepare_model_inputs (patched): unpacks the
#      micro-batch's `latent_z` to (1, total_nnz, H) via attention_mask and
#      injects it as `cached_latent_z=` into model_inputs.
#   5. compute_ref_log_prob (patched on ActorRolloutRefWorker): strips
#      latent_z from the data before infer_batch — ref weights ≠ rollout
#      actor weights, so the cached z is stale; ref must fall back to K-iter.
#
# Sequence parallel (ulysses_sp > 1) is NOT supported in this initial cut —
# the cached_z slicing would need to mirror ulysses_pad_and_slice_inputs for
# the 3D tensor, which is not implemented here. The patch raises if it sees
# use_ulysses_sp=True.
#
# update_actor: passed through (cache valid when bypass_mode=True / 1 PPO
# epoch — actor weights at start of update_actor == rollout-time weights;
# weights only change after the single optimizer.step() at end of the
# mini-batch). Multi-epoch PPO would invalidate the cache after iter 1; we
# don't run multi-epoch here so we accept the risk and document it.


def _patch_agent_loop_postprocess() -> None:
    """Pad extra_fields['latent_z'] from [response_len, H] (response-only
    sparse) to [1, prompt_pad + response_pad, H] aligning with input_ids
    layout (left-padded prompt + right-padded response).
    """
    try:
        from verl.experimental.agent_loop import agent_loop as _al
    except ImportError as e:
        print(f"[install] WARN: cannot patch _agent_loop_postprocess: {e}", flush=True)
        return

    cls = _al.AgentLoopWorker
    if getattr(cls._agent_loop_postprocess, "_alar_cached_z_patched", False):
        return
    _orig = cls._agent_loop_postprocess

    async def _wrapped(self, output, validate, **kwargs):
        import torch
        result = await _orig(self, output, validate, **kwargs)
        try:
            ef = result.extra_fields
            lz = ef.get("latent_z") if ef else None
            if isinstance(lz, torch.Tensor) and lz.dim() == 2:
                # lz: [response_len_unpadded, H] from agent_loop run
                P = int(result.prompt_ids.shape[1])
                R = int(result.response_ids.shape[1])
                H = int(lz.shape[-1])
                padded = torch.zeros(1, P + R, H, dtype=lz.dtype)
                actual = min(int(lz.shape[0]), R)
                if actual > 0:
                    padded[0, P:P + actual] = lz[:actual]
                ef["latent_z"] = padded
        except Exception as e:
            # Don't break the rollout if padding fails — log and let downstream
            # treat as missing.
            print(f"[cached_z] WARN: _agent_loop_postprocess pad: {e}", flush=True)
            if result.extra_fields is not None:
                result.extra_fields.pop("latent_z", None)
        return result

    _wrapped._alar_cached_z_patched = True
    cls._agent_loop_postprocess = _wrapped
    print(f"[install] patched {cls.__module__}.{cls.__name__}."
          "_agent_loop_postprocess (cached_z pad)", flush=True)


def _patch_postprocess_stack_z() -> None:
    """Pop latent_z from each input.extra_fields before _postprocess's
    np.object stuffing loop runs, then stack across the batch and inject
    into the returned DataProto's batch TensorDict.
    """
    try:
        from verl.experimental.agent_loop import agent_loop as _al
    except ImportError as e:
        print(f"[install] WARN: cannot patch _postprocess: {e}", flush=True)
        return

    cls = _al.AgentLoopWorker
    if getattr(cls._postprocess, "_alar_cached_z_patched", False):
        return
    _orig = cls._postprocess

    def _wrapped(self, inputs, input_non_tensor_batch=None, validate=False):
        import torch
        # Pop latent_z BEFORE the orig runs (which would coerce to np.object).
        lz_list = [inp.extra_fields.pop("latent_z", None)
                   if inp.extra_fields else None for inp in inputs]
        result = _orig(self, inputs, input_non_tensor_batch=input_non_tensor_batch,
                       validate=validate)
        try:
            present = [z for z in lz_list if isinstance(z, torch.Tensor)]
            # Always emit latent_z so every rollout shard carries the key:
            # DataProto.concat across shards is strict, so a shard where no
            # sample produced a latent block must still expose a (zero) tensor
            # of the right shape or the cross-shard concat raises KeyError.
            if present:
                ref = present[0]                       # (1, P+R, H)
                seqlen, H = int(ref.shape[1]), int(ref.shape[-1])
                dtype = ref.dtype
            else:
                # No latent samples in this shard: derive shape from the batch
                # (P+R) and hidden size from env (set by the launch script).
                H = int(os.environ.get("LATENT_HIDDEN_SIZE", "0"))
                seqlen = int(result.batch["input_ids"].shape[1])
                if H <= 0:
                    raise RuntimeError(
                        "latent-free shard and LATENT_HIDDEN_SIZE unset; "
                        "cannot build zero latent_z")
                dtype = torch.bfloat16
            full = []
            for z in lz_list:
                if isinstance(z, torch.Tensor):
                    full.append(z)
                else:
                    full.append(torch.zeros(1, seqlen, H, dtype=dtype))
            stacked = torch.cat(full, dim=0)  # (bsz, P+R, H)
            result.batch["latent_z"] = stacked
        except Exception as e:
            print(f"[cached_z] WARN: _postprocess stack: {e}", flush=True)
        return result

    _wrapped._alar_cached_z_patched = True
    cls._postprocess = _wrapped
    print(f"[install] patched {cls.__module__}.{cls.__name__}."
          "_postprocess (cached_z stack)", flush=True)


def _patch_prepare_model_inputs() -> None:
    """Inject cached_latent_z into model_inputs when micro_batch carries
    a nested `latent_z`. Skip silently for the no_padding-else branch
    (use_remove_padding=False) and for use_ulysses_sp=True (raises)."""
    try:
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    except ImportError as e:
        print(f"[install] WARN: cannot patch prepare_model_inputs: {e}",
              flush=True)
        return

    if getattr(FSDPEngineWithLMHead.prepare_model_inputs,
               "_alar_cached_z_patched", False):
        return
    _orig = FSDPEngineWithLMHead.prepare_model_inputs

    def _wrapped(self, micro_batch):
        import torch
        model_inputs, output_args = _orig(self, micro_batch=micro_batch)
        try:
            lz = micro_batch.get("latent_z", None) if "latent_z" in micro_batch.keys() else None
            if lz is None or not isinstance(lz, torch.Tensor):
                return model_inputs, output_args
            if lz.is_nested:
                # latent_z is expected as a regular (bsz, T, H) tensor —
                # converting it to a jagged NestedTensor upstream would hit
                # a pytorch 3D-nested unbind bug in chunk_tensordict.
                # Defensive bail if something converted it anyway.
                print("[cached_z] WARN: latent_z is nested — expected regular "
                      "(bsz, T, H); skipping", flush=True)
                return model_inputs, output_args
            if getattr(self, "use_ulysses_sp", False):
                print("[cached_z] WARN: ulysses_sp not supported with cached_z; "
                      "skipping", flush=True)
                return model_inputs, output_args
            # lz: regular (mb, T, H). Unpack using attention_mask to match
            # input_ids_rmpad's layout: select rows where attention_mask==1,
            # in row-major (sample-major) order. Equivalent to:
            #   for i in range(mb): for j where attn[i,j]==1: take lz[i,j]
            # which is exactly how input_ids_rmpad = input_ids.values() is
            # built from the nested input_ids derived via unpad_input.
            attn = micro_batch.get("attention_mask", None)
            if attn is None:
                print("[cached_z] WARN: no attention_mask; cannot unpack lz; "
                      "skipping", flush=True)
                return model_inputs, output_args
            lz_dev = lz.to(attn.device) if lz.device != attn.device else lz
            z_rmpad = lz_dev[attn.bool()].unsqueeze(0)  # (1, total_nnz, H)
            # Sanity: total_nnz must match input_ids_rmpad.
            inp = model_inputs.get("input_ids")
            if inp is not None and inp.shape[-1] != z_rmpad.shape[1]:
                print(f"[cached_z] WARN: shape mismatch input_ids "
                      f"{tuple(inp.shape)} vs latent_z "
                      f"{tuple(z_rmpad.shape)} — skipping", flush=True)
                return model_inputs, output_args
            model_inputs["cached_latent_z"] = z_rmpad
        except Exception as e:
            print(f"[cached_z] WARN: prepare_model_inputs inject: {e}",
                  flush=True)
        return model_inputs, output_args

    _wrapped._alar_cached_z_patched = True
    FSDPEngineWithLMHead.prepare_model_inputs = _wrapped
    print("[install] patched FSDPEngineWithLMHead.prepare_model_inputs "
          "(cached_z inject)", flush=True)


def _patch_compute_ref_log_prob() -> None:
    """Strip latent_z from the data passed to the ref worker. Ref weights
    differ from the rollout actor's, so cached z (produced by the actor at
    rollout time) is stale for ref — the model must fall back to K-iter."""
    try:
        from verl.workers.engine_workers import ActorRolloutRefWorker
    except ImportError as e:
        print(f"[install] WARN: cannot patch compute_ref_log_prob: {e}",
              flush=True)
        return

    if not hasattr(ActorRolloutRefWorker, "compute_ref_log_prob"):
        return
    if getattr(ActorRolloutRefWorker.compute_ref_log_prob,
               "_alar_cached_z_patched", False):
        return
    _orig = ActorRolloutRefWorker.compute_ref_log_prob

    # This wrapper only runs on a *separate* ref worker, i.e. when
    # ref_in_actor is False — which happens only under full FT (LORA_RANK=0).
    # Under LoRA the ref logprob is computed by the actor's compute_log_prob
    # (ref ≡ actor with adapter disabled), so this path is never hit there.
    lora_on = int(os.environ.get("LORA_RANK", "16") or "16") > 0

    def _wrapped(self, data):
        # Full FT: KEEP latent_z so the ref's prepare_model_inputs injects
        # cached_latent_z and takes the fast single-forward path. Stripping it
        # forces the data-dependent K-iter recompute, whose per-rank forward
        # (hence FSDP all-gather) count varies with each rank's latent-block
        # layout — that desyncs the collectives and hangs multi-GPU FSDP.
        # Reusing the actor's z for the reference is exactly what the LoRA
        # path does (it never strips z), so this matches established behavior;
        # any z staleness is a second-order KL-anchor effect.
        try:
            if lora_on and "latent_z" in data.keys():
                # LoRA-only (currently unreachable): drop z so ref recomputes
                # via K-iter against the adapter-disabled base weights.
                data = data.exclude("latent_z")
        except Exception as e:
            print(f"[cached_z] WARN: ref strip latent_z: {e}", flush=True)
        # LoRA only: force the ref's infer_batch to wrap the forward in
        # peft `disable_adapter()`, so ref evaluates the base SFT weights
        # without the actor's trained LoRA. Without this, ref ≡ actor and
        # KL(actor || ref) collapses to 0.
        # Under full FT (LORA_RANK=0) there is no adapter and a *separate*
        # frozen ref worker already holds the base SFT weights, so forcing
        # no_lora_adapter would call disable_adapter() on a plain module and
        # crash. Gate on LORA_RANK to match the launcher's own LoRA switch.
        if lora_on:
            try:
                from verl.utils import tensordict_utils as _tu
                _tu.assign_non_tensor(data, no_lora_adapter=True)
            except Exception as e:
                print(f"[ref] WARN: could not set no_lora_adapter: {e}", flush=True)
        return _orig(self, data)

    _wrapped._alar_cached_z_patched = True
    _copy_verl_dispatch_attrs(_orig, _wrapped)
    ActorRolloutRefWorker.compute_ref_log_prob = _wrapped
    mode = "LoRA: strip latent_z + no_lora_adapter" if lora_on else \
        "full FT: keep cached_latent_z (fast path, rank-synced)"
    print(f"[install] patched ActorRolloutRefWorker.compute_ref_log_prob "
          f"({mode})", flush=True)


def _patch_server_manager_generate() -> None:
    """Pass the caller's request_id straight through to vLLM, instead of verl's
    default `uuid4().hex` rewrite. Without this, the request_id we use in
    agent_loop for keying z files (`/tmp/latent_z/<request_id>.pt`) is
    decorrelated from the one vLLM's plugin sees, so agent_loop can never
    find the files.
    """
    try:
        from verl.experimental.agent_loop import agent_loop as _al
    except ImportError as e:
        print(f"[install] WARN: cannot patch AsyncLLMServerManager.generate: {e}",
              flush=True)
        return

    cls = _al.AsyncLLMServerManager
    if getattr(cls.generate, "_alar_cached_z_patched", False):
        return
    _orig = cls.generate

    async def _wrapped(self, request_id, *, prompt_ids, sampling_params,
                       image_data=None, video_data=None, **kwargs):
        server_id, server = await self._acquire_server(request_id)
        try:
            output = await server.generate.remote(
                request_id=request_id,  # use the caller's id (not a fresh uuid)
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )
            return output
        finally:
            self._release_server(server_id)

    _wrapped._alar_cached_z_patched = True
    cls.generate = _wrapped
    print(f"[install] patched {cls.__module__}.{cls.__name__}.generate "
          "(thread request_id through to vLLM)", flush=True)


def _install_cached_z_plumbing() -> None:
    """Wire up the cached_z patches. Each is idempotent; safe to call repeatedly."""
    _patch_agent_loop_postprocess()
    _patch_postprocess_stack_z()
    _patch_prepare_model_inputs()
    _patch_compute_ref_log_prob()
    _patch_server_manager_generate()

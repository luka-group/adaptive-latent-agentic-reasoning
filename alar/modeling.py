"""Qwen2.5 / Qwen3 ForCausalLM with the ALAR latent-block expansion.

Surface form (identical at SFT / RL / eval):

    <latent>••••</latent>

`<latent>` and `</latent>` are multi-piece BPE literals (3 pieces each in
Qwen2.5). The K=4 inner positions are bullet sentinels (`•`, id 6667).
At every step the model finds each contiguous `latent_mask=True` run of
K positions and replaces the K sentinel embeddings with K projector
outputs computed iteratively from the hidden state at the trigger.

For each block:
  1. Forward the preceding text up to (but not including) the K
     positions, with `use_cache=True`. The cache ends right at the
     trigger.
  2. last_h = out.last_hidden_state[:, -1, :]
     z_1 = f_phi(last_h)
  3. For k = 1..K-1:
       out = forward(inputs_embeds=z_k.unsqueeze(1), past_kv)
       z_{k+1} = f_phi(out.last_hidden_state[:, -1, :])
     Each forward extends the cache by one position; the K visible
     output positions in the assembled embedding are exactly z_1..z_K.
  4. Fold z_K into the cache so the trailing text (which starts with
     the first BPE piece of `</latent>`) can attend to it.

After per-sample iteration, a final batched forward over the assembled
embeddings produces hidden states aligned to input positions; gradients
to the projector flow through those hiddens via attention dependencies.

Save/load: projector weights are written to a `projector.pt` sidecar so
the base ckpt directory stays vLLM-compatible.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2PreTrainedModel,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
)
from transformers.utils import can_return_tuple

log = logging.getLogger(__name__)

DEFAULT_K = 4


class LatentProjector(nn.Module):
    """f_phi: hidden_size → hidden_size MLP shared across K iterations.

    `Dropout → Linear(d→prj) → GELU → Linear(prj→d) → LayerNorm`.
    With prj_dim=hidden_size that's ~2*d^2 params (~16.8M at d=2048).
    """

    def __init__(
        self,
        hidden_size: int,
        prj_dim: Optional[int] = None,
        dropout: float = 0.0,
        no_ln: bool = False,
    ):
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


def _reconstruct_latent_mask(
    input_ids: torch.LongTensor,
    latent_open_ids: List[int],
    latent_close_ids: List[int],
    num_latent: int,
) -> torch.BoolTensor:
    """Rebuild latent_mask from token ids by matching the surface pattern
    `<latent>` open + K sentinel positions + `</latent>` close. Used on
    the RL recompute path where the dataloader's latent_mask is not
    available. Stays on GPU; no CPU sync.
    """
    bs, T = input_ids.shape
    K = int(num_latent)
    mask = torch.zeros((bs, T), dtype=torch.bool, device=input_ids.device)
    if K <= 0 or not latent_open_ids or not latent_close_ids:
        return mask
    P = len(latent_open_ids)
    Q = len(latent_close_ids)
    if T < P + K + Q:
        return mask
    L = T - P - K - Q + 1
    valid = torch.ones((bs, L), dtype=torch.bool, device=input_ids.device)
    for j, tok in enumerate(latent_open_ids):
        valid &= input_ids[:, j : j + L] == int(tok)
    for j, tok in enumerate(latent_close_ids):
        valid &= input_ids[:, P + K + j : P + K + j + L] == int(tok)
    for k in range(P, P + K):
        mask[:, k : k + L] |= valid
    return mask


def _expand_latent_blocks(
    *,
    input_ids: torch.LongTensor,
    latent_mask: torch.BoolTensor,
    embed_fn: Callable[[torch.LongTensor], torch.Tensor],
    base_forward_fn: Callable[..., object],
    projector_fn: Callable[[torch.Tensor], torch.Tensor],
    forward_kwargs: dict,
) -> torch.Tensor:
    """Iterative latent generation; returns [1, T, H] with sentinels
    replaced by K projector outputs. Cache grows across the whole pass.
    """
    assert input_ids.shape[0] == 1, "expects batch=1; use the batched wrapper"
    mask_1d = latent_mask[0]
    T = int(input_ids.shape[1])

    pad = torch.zeros(1, dtype=torch.bool, device=mask_1d.device)
    padded = torch.cat([pad, mask_1d, pad])
    diff = padded.to(torch.int8).diff()
    starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()
    ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()
    blocks: List[Tuple[int, int]] = list(zip(starts, ends))

    all_embeds: List[torch.Tensor] = []
    cur_past = None
    processed = 0

    for start, end in blocks:
        K = end - start
        assert processed < start, (
            "latent block must be preceded by the `<latent>` literal; "
            "got block at sequence start with no preceding text."
        )
        text_ids = input_ids[:, processed:start]
        text_embeds = embed_fn(text_ids)
        all_embeds.append(text_embeds)
        out = base_forward_fn(
            input_ids=None,
            inputs_embeds=text_embeds,
            past_key_values=cur_past,
            use_cache=True,
            **forward_kwargs,
        )
        cur_past = out.past_key_values
        last_h = out.last_hidden_state[:, -1, :]

        target_dtype = text_embeds.dtype
        for k in range(K):
            z = projector_fn(last_h)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z = z.to(target_dtype)
            all_embeds.append(z.unsqueeze(1))
            out = base_forward_fn(
                input_ids=None,
                inputs_embeds=z.unsqueeze(1),
                past_key_values=cur_past,
                use_cache=True,
                **forward_kwargs,
            )
            cur_past = out.past_key_values
            if k < K - 1:
                # Detach severs the K-unroll backward chain so each z_k
                # only carries grad through one projector application
                # rather than k-1 stacked LM forwards (which would blow
                # the gradient norm up under bf16).
                last_h = out.last_hidden_state[:, -1, :].detach()

        processed = end

    if processed < T:
        rest_ids = input_ids[:, processed:]
        rest_embeds = embed_fn(rest_ids)
        all_embeds.append(rest_embeds)

    return torch.cat([e.to(all_embeds[0].device) for e in all_embeds], dim=1)


@contextlib.contextmanager
def _maybe_summon_full_params(module: Optional[nn.Module]):
    """Gather full FSDP-sharded params for the K-iter block.

    Without this, each `base_forward_fn` call inside the K-loop triggers its
    own per-layer FSDP all-gather; wrapping the loop in
    `FSDP.summon_full_params` gathers once and reuses across all bs ×
    (1 text-prefix + K iter) forwards. Math is unchanged — same params, just
    resident longer. Peak GPU memory rises by the size of the unsharded
    portion for the duration of the loop.

    OPT-IN via `LATENT_KITER_SUMMON=1`. Default OFF because verl's
    `param_offload=True` + log_prob recompute path can leave FSDP flat_param
    handles in a state that `summon_full_params` can't recover from
    (AssertionError in `_check_unsharded`). We catch any error on context
    entry and silently fall back to nullcontext so a failure here can never
    break training.

    Returns a no-op context when `module` is None, the env var is unset,
    FSDP is unavailable, or `module` has no FSDP descendants.
    """
    if module is None or not os.environ.get("LATENT_KITER_SUMMON"):
        yield
        return
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        yield
        return
    if not any(isinstance(m, FSDP) for m in module.modules()):
        yield
        return
    ctx = FSDP.summon_full_params(module, writeback=False, recurse=True)
    try:
        ctx.__enter__()
    except (AssertionError, RuntimeError, ValueError) as e:
        log.warning("[kiter-summon] disabled for this call (%s)", type(e).__name__)
        yield
        return
    try:
        yield
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass


def _expand_latent_blocks_batched(
    *,
    input_ids: torch.LongTensor,
    latent_mask: torch.BoolTensor,
    embed_fn: Callable[[torch.LongTensor], torch.Tensor],
    base_forward_fn: Callable[..., object],
    projector_fn: Callable[[torch.Tensor], torch.Tensor],
    forward_kwargs: dict,
    fsdp_summon_module: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Per-sample wrapper around `_expand_latent_blocks` for bs > 1.

    The iterative loop is inherently sequential per sample (its past_kv
    chain spans the whole sequence). The win at bs > 1 comes from the
    outer batched forward over assembled embeddings amortizing weight
    loads across B samples.

    `LATENT_DETACH_KITER=1` runs the entire K-iter expansion under
    `torch.no_grad()`. Set this for RL where the projector is frozen
    (PEFT default with `modules_to_save=None`): the K-iter forwards then
    produce no gradient back-path, so PyTorch skips their activations
    and FSDP skips their backward all-gathers. The math change is that
    LoRA on text-prefix layers loses the auxiliary gradient that flowed
    back via the projector chain at iter 0 — Forward B (the final
    assembled-embeds forward in `LatentQwen2Model.forward`) is unchanged
    and remains the dominant gradient signal.

    Do NOT set during SFT: Stage 1 / mode warmup train the projector
    via `modules_to_save=["projector"]`, and the projector needs full
    gradient flow back through `_expand_latent_blocks`.

    `fsdp_summon_module` (typically the LatentQwen2Model itself) gates the
    `LATENT_KITER_SUMMON=1` all-gather amortization. See
    `_maybe_summon_full_params`. Composable with `LATENT_DETACH_KITER`:
    summon kills forward gathers; detach kills backward gathers.
    """
    bs = int(input_ids.shape[0])
    embeds_list: List[torch.Tensor] = []
    grad_ctx = torch.no_grad() if os.environ.get("LATENT_DETACH_KITER") else contextlib.nullcontext()
    with _maybe_summon_full_params(fsdp_summon_module), grad_ctx:
        for b in range(bs):
            embeds_list.append(_expand_latent_blocks(
                input_ids=input_ids[b : b + 1],
                latent_mask=latent_mask[b : b + 1],
                embed_fn=embed_fn,
                base_forward_fn=base_forward_fn,
                projector_fn=projector_fn,
                forward_kwargs=forward_kwargs,
            ))
    return torch.cat(embeds_list, dim=0)


def _latent_model_forward(
    *,
    super_forward,
    embed_tokens,
    projector,
    config,
    fsdp_summon_module,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds,
    use_cache,
    output_attentions,
    output_hidden_states,
    cache_position,
    latent_mask,
    cached_latent_z,
    **kwargs,
) -> BaseModelOutputWithPast:
    """Shared latent-aware forward for `LatentQwen{2,3}Model`.

    The two HF base models have identical forward signatures, so this
    helper consolidates the latent-expansion logic and the per-variant
    Latent class only has to thread `super().forward` into it.
    """
    if input_ids is None or inputs_embeds is not None:
        return super_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

    if latent_mask is None and config.latent_open_ids and config.latent_close_ids:
        latent_mask = _reconstruct_latent_mask(
            input_ids,
            config.latent_open_ids,
            config.latent_close_ids,
            int(config.num_latent),
        )

    has_latent = latent_mask is not None and bool(latent_mask.any().item())
    if not has_latent:
        return super_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

    assert latent_mask.shape == input_ids.shape, (
        f"latent_mask {tuple(latent_mask.shape)} must match "
        f"input_ids {tuple(input_ids.shape)}"
    )

    if cached_latent_z is not None:
        token_embs = embed_tokens(input_ids)
        if cached_latent_z.shape != token_embs.shape:
            raise ValueError(
                f"cached_latent_z {tuple(cached_latent_z.shape)} must match "
                f"token_embs {tuple(token_embs.shape)} (dense [B, T, H])"
            )
        z = cached_latent_z.to(dtype=token_embs.dtype, device=token_embs.device)
        full_embs = torch.where(
            latent_mask.unsqueeze(-1).expand_as(token_embs),
            z,
            token_embs,
        )
    else:
        full_embs = _expand_latent_blocks_batched(
            input_ids=input_ids,
            latent_mask=latent_mask,
            embed_fn=embed_tokens,
            base_forward_fn=super_forward,
            projector_fn=projector,
            forward_kwargs=kwargs,
            fsdp_summon_module=fsdp_summon_module,
        )
    return super_forward(
        inputs_embeds=full_embs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=None,
        **kwargs,
    )


def _latent_lm_forward(
    *,
    model,
    lm_head,
    config,
    loss_function,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds,
    labels,
    use_cache,
    output_attentions,
    output_hidden_states,
    cache_position,
    logits_to_keep,
    latent_mask,
    cached_latent_z,
    **kwargs,
) -> CausalLMOutputWithPast:
    """Shared LM-head wrapper used by both Latent variants."""
    outputs: BaseModelOutputWithPast = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        latent_mask=latent_mask,
        cached_latent_z=cached_latent_z,
        **kwargs,
    )
    hidden = outputs.last_hidden_state
    slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
    logits = lm_head(hidden[:, slice_idx, :])

    loss = None
    if labels is not None:
        loss = loss_function(
            logits=logits, labels=labels,
            vocab_size=config.vocab_size, **kwargs,
        )
    return CausalLMOutputWithPast(
        loss=loss, logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class LatentQwen2Config(Qwen2Config):
    model_type = "latent_qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        num_latent: int = DEFAULT_K,
        prj_dim: int = 0,
        prj_dropout: float = 0.0,
        prj_no_ln: bool = False,
        latent_open_ids: Optional[List[int]] = None,
        latent_close_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_latent = int(num_latent)
        self.prj_dim = int(prj_dim)
        self.prj_dropout = float(prj_dropout)
        self.prj_no_ln = bool(prj_no_ln)
        self.latent_open_ids = list(latent_open_ids) if latent_open_ids else []
        self.latent_close_ids = list(latent_close_ids) if latent_close_ids else []
        if not hasattr(self, "layer_types") or self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


class LatentQwen2PreTrainedModel(Qwen2PreTrainedModel):
    config_class = LatentQwen2Config


class LatentQwen2Model(Qwen2Model):
    config_class = LatentQwen2Config

    def __init__(self, config: LatentQwen2Config):
        super().__init__(config)
        self.projector = LatentProjector(
            hidden_size=config.hidden_size,
            prj_dim=(config.prj_dim or config.hidden_size),
            dropout=config.prj_dropout,
            no_ln=config.prj_no_ln,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        latent_mask: Optional[torch.BoolTensor] = None,
        cached_latent_z: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        return _latent_model_forward(
            super_forward=super().forward,
            embed_tokens=self.embed_tokens,
            projector=self.projector,
            config=self.config,
            fsdp_summon_module=self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            latent_mask=latent_mask,
            cached_latent_z=cached_latent_z,
            **kwargs,
        )


def _load_projector_state_dict(projector: nn.Module, sd: dict) -> None:
    """ZeRO-3 aware projector load."""
    target_params = list(projector.parameters())
    is_zero3 = (
        len(target_params) > 0
        and target_params[0].numel() == 0
        and target_params[0].dim() > 0
    )
    if is_zero3:
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        with GatheredParameters(target_params, modifier_rank=0):
            missing, unexpected = projector.load_state_dict(sd, strict=False)
    else:
        missing, unexpected = projector.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log.warning(f"projector load: missing={missing} unexpected={unexpected}")


class LatentQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = LatentQwen2Config

    def __init__(self, config: LatentQwen2Config):
        super().__init__(config)
        self.model = LatentQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # Deferred RL-side install hook: fires the FIRST time this model
        # class is instantiated in any process (driver or Ray worker).
        # By the time the model is constructed, verl is fully imported
        # → safe to patch ActorRolloutRefWorker / DataParallelPPOActor.
        # install() is idempotent (no-op if already patched).
        if not getattr(LatentQwen2ForCausalLM, "_alar_install_done", False):
            try:
                from alar.rl.install_patches import install as _alar_install
                _alar_install()
                LatentQwen2ForCausalLM._alar_install_done = True
            except Exception as _e:
                pass  # fail silent; driver-side install still ran

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        latent_mask: Optional[torch.BoolTensor] = None,
        cached_latent_z: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return _latent_lm_forward(
            model=self.model,
            lm_head=self.lm_head,
            config=self.config,
            loss_function=self.loss_function,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            latent_mask=latent_mask,
            cached_latent_z=cached_latent_z,
            **kwargs,
        )

    def save_pretrained(self, save_directory, **kwargs):
        """Write base weights with `architectures=['Qwen2ForCausalLM']` so
        vLLM loads them as stock Qwen2.5, and dump the projector to a
        sidecar `projector.pt`. The latent expansion is then reattached
        at vLLM load by the plugin in `alar.vllm_plugin`.
        """
        import json
        state_dict = kwargs.pop("state_dict", None) or self.state_dict()
        proj_prefix = "model.projector."
        base_state = {k: v for k, v in state_dict.items() if not k.startswith(proj_prefix)}
        proj_state = {k[len(proj_prefix):]: v for k, v in state_dict.items() if k.startswith(proj_prefix)}
        super().save_pretrained(save_directory, state_dict=base_state, **kwargs)
        os.makedirs(save_directory, exist_ok=True)
        torch.save(proj_state, os.path.join(save_directory, "projector.pt"))
        cfg_path = os.path.join(save_directory, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            # Keep `model_type: latent_qwen2` + latent fields so verl /
            # HF AutoConfig dispatches to LatentQwen2ForCausalLM (giving
            # the actor a projector that PPO + sync hook can act on).
            # vLLM picks its model class from `architectures`, so it
            # still loads this dir as stock Qwen2ForCausalLM and lets
            # the vLLM plugin re-attach the projector at runtime.
            cfg["architectures"] = ["Qwen2ForCausalLM"]
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if isinstance(pretrained_model_name_or_path, str) and os.path.isdir(pretrained_model_name_or_path):
            p = os.path.join(pretrained_model_name_or_path, "projector.pt")
            if os.path.exists(p):
                sd = torch.load(p, map_location="cpu")
                _load_projector_state_dict(model.model.projector, sd)
        return model


# ----------------------------------------------------------------------
# Qwen3 mirror (used by the tool domain, e.g. Qwen3-4B-Thinking-2507).
# Forward logic is identical and lives in `_latent_model_forward` /
# `_latent_lm_forward`; we just plug the Qwen3 base classes.
# ----------------------------------------------------------------------


class LatentQwen3Config(Qwen3Config):
    model_type = "latent_qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        num_latent: int = DEFAULT_K,
        prj_dim: int = 0,
        prj_dropout: float = 0.0,
        prj_no_ln: bool = False,
        latent_open_ids: Optional[List[int]] = None,
        latent_close_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_latent = int(num_latent)
        self.prj_dim = int(prj_dim)
        self.prj_dropout = float(prj_dropout)
        self.prj_no_ln = bool(prj_no_ln)
        self.latent_open_ids = list(latent_open_ids) if latent_open_ids else []
        self.latent_close_ids = list(latent_close_ids) if latent_close_ids else []
        if not hasattr(self, "layer_types") or self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


class LatentQwen3PreTrainedModel(Qwen3PreTrainedModel):
    config_class = LatentQwen3Config


class LatentQwen3Model(Qwen3Model):
    config_class = LatentQwen3Config

    def __init__(self, config: LatentQwen3Config):
        super().__init__(config)
        self.projector = LatentProjector(
            hidden_size=config.hidden_size,
            prj_dim=(config.prj_dim or config.hidden_size),
            dropout=config.prj_dropout,
            no_ln=config.prj_no_ln,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        latent_mask: Optional[torch.BoolTensor] = None,
        cached_latent_z: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        return _latent_model_forward(
            super_forward=super().forward,
            embed_tokens=self.embed_tokens,
            projector=self.projector,
            config=self.config,
            fsdp_summon_module=self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            latent_mask=latent_mask,
            cached_latent_z=cached_latent_z,
            **kwargs,
        )


class LatentQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = LatentQwen3Config

    def __init__(self, config: LatentQwen3Config):
        super().__init__(config)
        self.model = LatentQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if not getattr(LatentQwen3ForCausalLM, "_alar_install_done", False):
            try:
                from alar.rl.install_patches import install as _alar_install
                _alar_install()
                LatentQwen3ForCausalLM._alar_install_done = True
            except Exception:
                pass

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        latent_mask: Optional[torch.BoolTensor] = None,
        cached_latent_z: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return _latent_lm_forward(
            model=self.model,
            lm_head=self.lm_head,
            config=self.config,
            loss_function=self.loss_function,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            latent_mask=latent_mask,
            cached_latent_z=cached_latent_z,
            **kwargs,
        )

    def save_pretrained(self, save_directory, **kwargs):
        """Mirror of `LatentQwen2ForCausalLM.save_pretrained`, but writes
        `architectures=['Qwen3ForCausalLM']` so vLLM loads the base ckpt
        as stock Qwen3 and the plugin re-attaches the projector at runtime.
        """
        import json
        state_dict = kwargs.pop("state_dict", None) or self.state_dict()
        proj_prefix = "model.projector."
        base_state = {k: v for k, v in state_dict.items() if not k.startswith(proj_prefix)}
        proj_state = {k[len(proj_prefix):]: v for k, v in state_dict.items() if k.startswith(proj_prefix)}
        super().save_pretrained(save_directory, state_dict=base_state, **kwargs)
        os.makedirs(save_directory, exist_ok=True)
        torch.save(proj_state, os.path.join(save_directory, "projector.pt"))
        cfg_path = os.path.join(save_directory, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg["architectures"] = ["Qwen3ForCausalLM"]
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if isinstance(pretrained_model_name_or_path, str) and os.path.isdir(pretrained_model_name_or_path):
            p = os.path.join(pretrained_model_name_or_path, "projector.pt")
            if os.path.exists(p):
                sd = torch.load(p, map_location="cpu")
                _load_projector_state_dict(model.model.projector, sd)
        return model


def build_from_base(
    model_name_or_path: str,
    num_latent: int = DEFAULT_K,
    prj_dim: int = 0,
    prj_dropout: float = 0.0,
    prj_no_ln: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    freeze_base: bool = False,
):
    """Wrap a stock Qwen2.5 OR Qwen3 ckpt as the matching Latent variant.

    No special tokens are added. `<latent>` / `</latent>` are encoded as
    multi-piece BPE; `•` (id 6667) is the existing single-piece bullet
    token used as the sentinel for the K latent positions.

    Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    latent_open_ids = tokenizer.encode("<latent>", add_special_tokens=False)
    latent_close_ids = tokenizer.encode("</latent>", add_special_tokens=False)
    log.info(f"latent_open_ids={latent_open_ids} latent_close_ids={latent_close_ids}")

    base_cfg = AutoConfig.from_pretrained(model_name_or_path)
    base_mt = getattr(base_cfg, "model_type", "")
    if base_mt in ("qwen3", "latent_qwen3"):
        LatentCfg, LatentLM, BaseLM = LatentQwen3Config, LatentQwen3ForCausalLM, Qwen3ForCausalLM
    elif base_mt in ("qwen2", "latent_qwen2"):
        LatentCfg, LatentLM, BaseLM = LatentQwen2Config, LatentQwen2ForCausalLM, Qwen2ForCausalLM
    else:
        raise ValueError(
            f"build_from_base: unsupported base model_type={base_mt!r}; "
            f"only 'qwen2'/'latent_qwen2' and 'qwen3'/'latent_qwen3' are wired in."
        )

    cfg_dict = base_cfg.to_dict()
    for k in ("num_latent", "prj_dim", "prj_dropout", "prj_no_ln",
              "latent_open_ids", "latent_close_ids"):
        cfg_dict.pop(k, None)

    cfg = LatentCfg(
        num_latent=num_latent,
        prj_dim=prj_dim,
        prj_dropout=prj_dropout,
        prj_no_ln=prj_no_ln,
        latent_open_ids=latent_open_ids,
        latent_close_ids=latent_close_ids,
        **cfg_dict,
    )

    model = LatentLM(cfg)
    base = BaseLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith("model.projector.") for k in missing), \
        f"unexpected missing keys: {missing}"
    assert len(unexpected) == 0, f"unexpected extra keys: {unexpected}"

    model.to(torch_dtype)

    if freeze_base:
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("model.projector."))

    return model, tokenizer


# HF auto-class registrations. These make `AutoConfig.from_pretrained`
# dispatch to LatentQwen2Config and `AutoModelForCausalLM.from_pretrained`
# dispatch to LatentQwen2ForCausalLM whenever a saved config has
# `model_type: latent_qwen2`. Importing this module is enough — `verl`
# and our `alar.rl.install_patches` both call it before any verl worker
# touches AutoConfig.
try:
    AutoConfig.register("latent_qwen2", LatentQwen2Config)
except (ValueError, KeyError):
    pass  # already registered (re-import in same process)
try:
    AutoConfig.register("latent_qwen3", LatentQwen3Config)
except (ValueError, KeyError):
    pass
try:
    from transformers import AutoModel
    AutoModel.register(LatentQwen2Config, LatentQwen2Model)
except (ValueError, KeyError):
    pass
try:
    from transformers import AutoModel
    AutoModel.register(LatentQwen3Config, LatentQwen3Model)
except (ValueError, KeyError):
    pass
try:
    AutoModelForCausalLM.register(LatentQwen2Config, LatentQwen2ForCausalLM)
except (ValueError, KeyError):
    pass
try:
    AutoModelForCausalLM.register(LatentQwen3Config, LatentQwen3ForCausalLM)
except (ValueError, KeyError):
    pass


__all__ = [
    "DEFAULT_K",
    "LatentProjector",
    "LatentQwen2Config",
    "LatentQwen2Model",
    "LatentQwen2ForCausalLM",
    "LatentQwen3Config",
    "LatentQwen3Model",
    "LatentQwen3ForCausalLM",
    "build_from_base",
]

"""Merge a full-FT (LORA_RANK=0) AR-GRPO FSDP checkpoint into a
vLLM-loadable HuggingFace dir + `projector.pt` sidecar.

Unlike merge_rl.py (LoRA adapter merge), full-FT RL saves the whole model
sharded across ranks as DTensors in `model_world_size_N_rank_*.pt`. This
script consolidates those shards (concatenating each parameter's per-rank
local tensors along its Shard dim), loads the result into a fresh
LatentQwen2 skeleton built from the SearchR1 base, then calls
`save_pretrained` so the trained projector lands in a `projector.pt`
sidecar and the architecture is rewritten to ["Qwen2ForCausalLM"].

Usage:
  python -m alar.scripts.merge_fsdp_rl \
      --ckpt_dir checkpoints/rl/ar_grpo_fullft/global_step_50/actor \
      --output_dir checkpoints/rl_merged/ar_grpo_fullft_step50
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from alar.modeling import build_from_base


def _consolidate_fsdp(ckpt_dir: str) -> dict:
    """Load all FSDP rank shards and merge into a full CPU state dict."""
    cfg_path = Path(ckpt_dir) / "fsdp_config.json"
    world_size = json.load(open(cfg_path))["world_size"]
    print(f"[merge-fsdp] world_size={world_size}")

    shards = []
    for r in range(world_size):
        p = Path(ckpt_dir) / f"model_world_size_{world_size}_rank_{r}.pt"
        print(f"[merge-fsdp] loading {p.name} ...")
        shards.append(torch.load(p, map_location="cpu", weights_only=False))

    merged: dict[str, torch.Tensor] = {}
    for key in sorted(shards[0].keys()):
        local_pieces = []
        placement = None
        for sd in shards:
            t = sd[key]
            if isinstance(t, DTensor):
                local_pieces.append(t._local_tensor.bfloat16())
                placement = t.placements[0]
            else:
                local_pieces.append(t.bfloat16())
        if placement is not None and placement.is_shard():
            merged[key] = torch.cat(local_pieces, dim=placement.dim).contiguous()
        else:
            # replicate / non-DTensor: every rank holds the full tensor
            merged[key] = local_pieces[0]
    print(f"[merge-fsdp] consolidated {len(merged)} params")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="FSDP actor dir (has model_world_size_*_rank_*.pt + "
                         "fsdp_config.json).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model",
                    default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3")
    ap.add_argument("--num_latent", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"[merge-fsdp] ckpt   = {args.ckpt_dir}")
    print(f"[merge-fsdp] base   = {args.base_model}")
    print(f"[merge-fsdp] output = {args.output_dir}")

    base, tokenizer = build_from_base(
        args.base_model, num_latent=args.num_latent,
        torch_dtype=dtype, freeze_base=False,
    )
    base = base.to(args.device).to(dtype)
    print(f"[merge-fsdp] skeleton vocab={len(tokenizer)} hidden={base.config.hidden_size}")

    full_sd = _consolidate_fsdp(args.ckpt_dir)

    # Sanity: projector present and non-trivial.
    proj_keys = [k for k in full_sd if ".projector." in k]
    print(f"[merge-fsdp] projector tensors in ckpt: {len(proj_keys)}")
    assert proj_keys, "no projector weights in checkpoint!"

    missing, unexpected = base.load_state_dict(full_sd, strict=False)
    # tie_word_embeddings: lm_head.weight is tied to embed_tokens, so a
    # missing lm_head.weight is expected and harmless.
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing:
        print(f"[merge-fsdp] WARN missing keys: {missing[:10]} ... ({len(missing)})")
    if unexpected:
        print(f"[merge-fsdp] WARN unexpected keys: {unexpected[:10]} ... ({len(unexpected)})")
    assert not missing, "unexpected missing params after load_state_dict"

    proj_norm = sum(p.float().norm().item()**2
                    for p in base.model.projector.parameters())**0.5
    print(f"[merge-fsdp] loaded projector Frobenius norm: {proj_norm:.4f}")
    assert proj_norm > 1.0, "projector looks random/untrained after load"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[merge-fsdp] saving to {args.output_dir} ...")
    base.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Roundtrip: projector.pt sidecar must match.
    proj_path = os.path.join(args.output_dir, "projector.pt")
    assert os.path.isfile(proj_path), f"projector.pt missing at {proj_path}"
    sd = torch.load(proj_path, map_location="cpu")
    orig_w = base.model.projector.net[1].weight.detach().cpu().float()
    new_w = sd["net.1.weight"].cpu().float()
    assert torch.allclose(orig_w, new_w, atol=1e-3), "projector sidecar mismatch"
    print(f"[merge-fsdp] OK — projector.pt sidecar matches ({len(sd)} tensors).")

    print(f"\n[merge-fsdp] files in {args.output_dir}:")
    for f in sorted(os.listdir(args.output_dir)):
        p = os.path.join(args.output_dir, f)
        print(f"  {f:50s}  {os.path.getsize(p) / (1024 * 1024):>8.1f} MB")
    print("\n[merge-fsdp] done. Load with vLLM + LATENT_VLLM=1.")


if __name__ == "__main__":
    main()

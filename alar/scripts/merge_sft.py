"""Merge AASD/Stage-2 PEFT adapter into a base ckpt and emit a
vLLM-loadable directory + `projector.pt` sidecar.

The adapter dir holds:
  - LoRA deltas on attention `q/k/v/o`
  - Projector weights via PEFT's `modules_to_save=["projector"]`

`merge_and_unload` fuses the LoRA deltas into the base; the projector
stays as a module on the merged model. `LatentQwen2ForCausalLM.save_pretrained`
then splits the projector into a sidecar and rewrites
`architectures = ["Qwen2ForCausalLM"]` so vLLM loads it as stock Qwen2.5.

Usage:
  python -m alar.scripts.merge_sft \\
      --sft_ckpt /path/to/aasd_run/final \\
      --output_dir /path/to/merged_ckpt
"""
from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel

from alar.modeling import build_from_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_ckpt", required=True,
                    help="PEFT adapter dir (e.g. <output_dir>/final).")
    ap.add_argument("--base_model",
                    default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_latent", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"[merge] base_model = {args.base_model}")
    print(f"[merge] sft_ckpt   = {args.sft_ckpt}")
    print(f"[merge] output     = {args.output_dir}")

    base, tokenizer = build_from_base(
        args.base_model,
        num_latent=args.num_latent,
        torch_dtype=dtype,
        freeze_base=False,
    )
    base = base.to(args.device).to(dtype)
    print(f"[merge] vocab={len(tokenizer)} hidden={base.config.hidden_size}")

    print("[merge] loading PEFT adapter...")
    peft_model = PeftModel.from_pretrained(base, args.sft_ckpt, is_trainable=False)

    sd_keys = list(peft_model.state_dict().keys())
    n_lora = sum(1 for k in sd_keys if "lora_" in k)
    n_proj = sum(1 for k in sd_keys if ".projector." in k)
    print(f"[merge] adapter: {n_lora} LoRA tensors, {n_proj} projector tensors")

    print("[merge] merging LoRA + unloading adapter...")
    merged = peft_model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[merge] saving to {args.output_dir} ...")
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Roundtrip verify: load as stock Qwen2 + sidecar projector.pt.
    print("[merge] verifying roundtrip ...")
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(args.output_dir, trust_remote_code=True)
    assert cfg.architectures in (["Qwen2ForCausalLM"], ["Qwen3ForCausalLM"]), cfg.architectures
    AutoModelForCausalLM.from_pretrained(
        args.output_dir, torch_dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(args.device).eval()  # loading at all is the verification

    # Confirm projector.pt sidecar loads cleanly through the
    # LatentQwen2 from_pretrained hook.
    inner_merged = merged.model
    orig_w = inner_merged.projector.net[1].weight.detach().cpu().float()
    # The reloaded model is a stock Qwen2; it has no projector. The
    # check that matters for vLLM is that projector.pt deserializes.
    proj_path = os.path.join(args.output_dir, "projector.pt")
    assert os.path.isfile(proj_path), f"projector.pt missing at {proj_path}"
    sd = torch.load(proj_path, map_location="cpu")
    new_w = sd["net.1.weight"].cpu().float()
    assert torch.allclose(orig_w, new_w, atol=1e-3), \
        f"projector weight mismatch across roundtrip (max diff {(orig_w-new_w).abs().max():.4e})"
    print(f"[merge] OK — projector.pt sidecar matches merged weights ({len(sd)} tensors).")

    print(f"\n[merge] files in {args.output_dir}:")
    for f in sorted(os.listdir(args.output_dir)):
        p = os.path.join(args.output_dir, f)
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  {f:50s}  {size_mb:>8.1f} MB")

    print("\n[merge] done. Load this dir with vLLM + LATENT_VLLM=1 + alar.vllm_plugin.")


if __name__ == "__main__":
    main()

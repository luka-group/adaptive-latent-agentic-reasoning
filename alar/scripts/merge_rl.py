"""Merge an RL-stage LoRA adapter onto the mode-warmup SFT merged base,
preserving the projector sidecar so vLLM can load the result.

Differs from merge_sft.py in that:
  - Base is the already-merged mode-warmup dir (has `projector.pt`), not
    a stock Qwen2.5 (which would lose the trained projector).
  - The RL adapter is LoRA-only (`modules_to_save=null`), so projector
    comes from the base, not the adapter.

Usage:
  python -m alar.scripts.merge_rl \
      --base_dir checkpoints/sft_merged/warmup \
      --adapter_dir checkpoints/rl/ar_grpo/global_step_50/actor/lora_adapter \
      --output_dir checkpoints/rl_merged/ar_grpo_step50 \
      --device cpu
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch
from peft import PeftModel

import alar.modeling  # noqa: F401  (registers the latent_qwen2 model type)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="Mode-warmup merged ckpt dir (has projector.pt sidecar).")
    ap.add_argument("--adapter_dir", required=True,
                    help="RL LoRA adapter dir (adapter_config.json + "
                         "adapter_model.safetensors).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda:N. CPU is slow (~5 min for 7B) but "
                         "avoids touching GPUs in use.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"[merge-rl] base    = {args.base_dir}")
    print(f"[merge-rl] adapter = {args.adapter_dir}")
    print(f"[merge-rl] output  = {args.output_dir}")
    print(f"[merge-rl] device  = {args.device}, dtype={args.dtype}")

    print("[merge-rl] loading base via LatentQwen2ForCausalLM.from_pretrained "
          "(this loads projector.pt sidecar) ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base = base.to(args.device).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_dir, trust_remote_code=True)
    print(f"[merge-rl] vocab={len(tokenizer)} hidden={base.config.hidden_size}")
    # Sanity: projector weights are non-zero (loaded from base's projector.pt).
    proj_norm = sum(p.float().norm().item()**2
                    for p in base.model.projector.parameters())**0.5
    print(f"[merge-rl] base projector Frobenius norm: {proj_norm:.4f}")
    assert proj_norm > 1.0, "projector looks random/untrained"

    print("[merge-rl] loading adapter ...")
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    sd_keys = list(peft_model.state_dict().keys())
    n_lora = sum(1 for k in sd_keys if "lora_" in k)
    n_proj_in_adapter = sum(1 for k in sd_keys if "modules_to_save" in k)
    print(f"[merge-rl] adapter loaded: {n_lora} LoRA tensors, "
          f"{n_proj_in_adapter} modules_to_save tensors")

    print("[merge-rl] merging LoRA + unloading adapter ...")
    merged = peft_model.merge_and_unload()
    # Confirm projector survived the merge (PEFT merge doesn't touch non-LoRA modules).
    proj_norm_after = sum(p.float().norm().item()**2
                          for p in merged.model.projector.parameters())**0.5
    print(f"[merge-rl] merged projector Frobenius norm: {proj_norm_after:.4f} "
          f"(should match base {proj_norm:.4f})")
    assert abs(proj_norm - proj_norm_after) < 1e-3, "projector changed during merge!"

    os.makedirs(args.output_dir, exist_ok=True)
    print("[merge-rl] saving merged ckpt + projector.pt sidecar ...")
    # LatentQwen2ForCausalLM.save_pretrained extracts projector to sidecar
    # and rewrites config architectures to ["Qwen2ForCausalLM"] so vLLM
    # loads it as stock Qwen2 + sidecar.
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also copy chat_template.jinja if present (vLLM needs it).
    for fname in ["chat_template.jinja", "generation_config.json", "added_tokens.json"]:
        src = os.path.join(args.base_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"[merge-rl]   copied {fname}")

    print(f"\n[merge-rl] files in {args.output_dir}:")
    for f in sorted(os.listdir(args.output_dir)):
        p = os.path.join(args.output_dir, f)
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  {f:50s}  {size_mb:>8.1f} MB")

    print("\n[merge-rl] OK. Load with vLLM + LATENT_VLLM=1.")


if __name__ == "__main__":
    main()

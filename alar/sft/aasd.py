"""AASD: Action-Anchored Self-Distillation (Stage 1 + mode-warmup SFT).

Same code drives both stages; only `--latent_prob` differs:
  - Stage 1 (AASD): `--latent_prob 1.0` — every turn emits the latent
    block; the student learns to predict the teacher's anchor actions
    given the K projector outputs.
  - Mode warmup: `--latent_prob 0.5` (typically; any value in (0, 1))
    — per-turn coin flip between `<latent>` and `<think>` so the model
    is ready to pick a mode per turn under AR-GRPO.

Trainable: LoRA on attention (q/k/v/o) + projector full-FT
(via PEFT's `modules_to_save`). Pass `--no_lora` to train the
projector only (frozen base).

Save layout under `output_dir/final/`:
  - `--no_lora`: a `projector.pt` sidecar (the base ckpt is unchanged
    and vLLM-loadable as-is).
  - LoRA: PEFT adapter dir (LoRA deltas + projector under
    `modules_to_save`). The merge script consumes this + the base ckpt
    to produce a vLLM-loadable merged dir with `projector.pt` sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from alar.modeling import build_from_base
from search.dataset import SearchLatentDataset
from tool.dataset import ToolLatentDataset


class LogCallback(TrainerCallback):
    """Append every {step → metrics} dict to train_log.jsonl on rank-0."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, "train_log.jsonl")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            entry = {"step": state.global_step, **logs}
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class LogGradNormsCallback(TrainerCallback):
    """Per-group L2 grad norms: LoRA vs projector."""

    def __init__(self, model, log_every: int = 50):
        self.model = model
        self.log_every = int(log_every)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.log_every <= 0 or (state.global_step % self.log_every) != 0:
            return
        lora_sq = proj_sq = 0.0
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g2 = float(p.grad.detach().float().pow(2).sum().item())
            if "lora_" in n:
                lora_sq += g2
            elif ".projector." in n or n.endswith(".projector"):
                proj_sq += g2
        print(f"[grad_norms] step={state.global_step} "
              f"lora={math.sqrt(lora_sq):.4f} projector={math.sqrt(proj_sq):.4f}",
              flush=True)


def apply_lora(model, *, lora_r: int, lora_alpha: int, lora_dropout: float):
    """LoRA on attention (q/k/v/o) + projector full-FT via modules_to_save."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["projector"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def _extract_projector_sidecar(model, out_path: str) -> int:
    """Walk the model state_dict, pick out projector weights, write a
    flat `projector.pt` (keys = `net.X.weight` / `net.X.bias`) so the
    vLLM plugin can load them with `strict=False`.
    """
    sd = model.state_dict()
    proj = {}
    for k, v in sd.items():
        # Match both unwrapped (`model.projector.net.*`) and PEFT-wrapped
        # (`...projector.modules_to_save.default.net.*`) layouts.
        if ".projector." not in k:
            continue
        sub = k.split(".projector.", 1)[1]
        # Strip PEFT's modules_to_save wrapper.
        if sub.startswith("modules_to_save."):
            sub = sub.split(".", 2)[2]  # drop 'modules_to_save.default.'
        # Skip the duplicated 'original_module' copy PEFT keeps for restore.
        if sub.startswith("original_module."):
            continue
        proj[sub] = v.detach().cpu()
    torch.save(proj, out_path)
    return len(proj)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    p.add_argument("--num_latent", type=int, default=4)
    p.add_argument("--prj_dim", type=int, default=0)
    p.add_argument("--prj_dropout", type=float, default=0.0)
    p.add_argument("--prj_no_ln", action="store_true")

    p.add_argument("--no_lora", action="store_true",
                   help="Pure projector-only training (frozen base, no LoRA).")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--projector_lr_mult", type=float, default=1.0)
    p.add_argument("--init_lora_from", default=None,
                   help="Mode warmup: warm-start LoRA + projector from a "
                        "Stage-1 adapter dir. Fresh optimizer / scheduler.")

    p.add_argument("--domain", choices=["search", "tool"], default="search",
                   help="Which domain dataset to build.")
    p.add_argument("--data_path", required=True)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--data_offset", type=int, default=0,
                   help="Skip the first N filtered trajectories before "
                        "applying max_samples (selects a sub-range of the "
                        "teacher trace file).")
    # search-only filters
    p.add_argument("--min_em", type=float, default=1.0)
    p.add_argument("--max_doc_chars", type=int, default=500)
    # tool-only filter
    p.add_argument("--min_score", type=float, default=1.5,
                   help="[tool] keep trajectories whose teacher score >= this.")
    p.add_argument("--latent_prob", type=float, default=1.0,
                   help="Per-turn probability of emitting <latent> vs "
                        "<think>. 1.0 = Stage 1 AASD; 0.5 = "
                        "mode-selection warmup.")
    p.add_argument("--dataset_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=0)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_schedule", default="cosine", choices=["constant", "cosine"])
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--resume_from_checkpoint", default=None)

    args = p.parse_args()

    model, tokenizer = build_from_base(
        args.model_name,
        num_latent=args.num_latent,
        prj_dim=args.prj_dim,
        prj_dropout=args.prj_dropout,
        prj_no_ln=args.prj_no_ln,
        freeze_base=bool(args.no_lora),
    )
    print(f"[aasd] K={args.num_latent} latent_prob={args.latent_prob} "
          f"vocab={len(tokenizer)} hidden={model.config.hidden_size}")

    if not args.no_lora:
        if args.init_lora_from:
            from peft import PeftModel
            print(f"[aasd] init LoRA from {args.init_lora_from}")
            model = PeftModel.from_pretrained(
                model, args.init_lora_from, is_trainable=True,
            )
        else:
            model = apply_lora(
                model,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
        model.print_trainable_parameters()
    else:
        n_train = sum(x.numel() for x in model.parameters() if x.requires_grad)
        n_total = sum(x.numel() for x in model.parameters())
        print(f"[aasd] no_lora: trainable {n_train:,}/{n_total:,} "
              f"({100*n_train/n_total:.2f}%) — projector only")

    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.domain == "search":
        dataset = SearchLatentDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            num_latent=args.num_latent,
            max_length=args.max_length,
            max_samples=args.max_samples,
            data_offset=args.data_offset,
            min_em=args.min_em,
            max_doc_chars=args.max_doc_chars,
            latent_prob=args.latent_prob,
            seed=args.dataset_seed,
        )
    else:
        dataset = ToolLatentDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            num_latent=args.num_latent,
            max_length=args.max_length,
            max_samples=args.max_samples,
            data_offset=args.data_offset,
            min_score=args.min_score,
            latent_prob=args.latent_prob,
            seed=args.dataset_seed,
        )

    optim_groups = None
    if args.projector_lr_mult != 1.0:
        proj_params, other_params = [], []
        for n, q in model.named_parameters():
            if not q.requires_grad:
                continue
            (proj_params if ".projector." in n or n.endswith(".projector") else other_params).append(q)
        optim_groups = [
            {"params": other_params, "lr": args.lr},
            {"params": proj_params,  "lr": args.lr * args.projector_lr_mult},
        ]

    use_gc = bool(args.gradient_checkpointing)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_schedule,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=max(args.save_steps, 1),
        save_total_limit=None,
        report_to="none",
        ddp_timeout=1800000,
        # Mode-warmup microbatches with no latent block leave the projector
        # without grad → DDP allreduce hangs without this.
        ddp_find_unused_parameters=(args.latent_prob < 1.0),
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        # Variable-length sequences with per-sample latent expansion. The
        # collate_fn right-pads input_ids / labels / latent_mask.
        accelerator_config={"split_batches": False, "dispatch_batches": False},
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=dataset.collate_fn,
        processing_class=tokenizer,
        callbacks=[
            LogCallback(args.output_dir),
            LogGradNormsCallback(model, log_every=50),
        ],
    )

    if optim_groups is not None:
        from torch.optim import AdamW
        trainer.optimizer = AdamW(optim_groups, lr=args.lr)

    if trainer.is_world_process_zero():
        print(f"[aasd] dataset={len(dataset)} epochs={args.epochs} "
              f"ga={args.gradient_accumulation_steps} lr={args.lr} "
              f"lora={'no' if args.no_lora else 'yes'}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if not trainer.is_world_process_zero():
        return

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    if args.no_lora:
        # Pure projector-only: dump sidecar; base ckpt is unchanged
        # upstream so vLLM loads it from `--model_name` directly.
        n = _extract_projector_sidecar(
            model, os.path.join(final_dir, "projector.pt"),
        )
        tokenizer.save_pretrained(final_dir)
        print(f"[aasd] saved projector ({n} tensors) to {final_dir}")
    else:
        # PEFT adapter (LoRA deltas + projector via modules_to_save).
        # Merge script consumes this dir to produce a vLLM-loadable ckpt.
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"[aasd] saved adapter to {final_dir}")

    with open(os.path.join(final_dir, "aasd_meta.json"), "w") as f:
        json.dump({
            "base_model": args.model_name,
            "num_latent": args.num_latent,
            "prj_dim": args.prj_dim,
            "prj_dropout": args.prj_dropout,
            "prj_no_ln": args.prj_no_ln,
            "latent_prob": args.latent_prob,
            "lora": (not args.no_lora),
            "init_lora_from": args.init_lora_from,
        }, f, indent=2)


if __name__ == "__main__":
    main()

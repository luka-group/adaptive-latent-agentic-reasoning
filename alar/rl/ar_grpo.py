"""AR-GRPO reward shaping. Applied as a monkey-patch on verl's
`extract_reward`.

Per-trajectory shaped reward:

    f          = n_latent / (n_latent + n_think)   (latent fraction, ∈ [0, 1])
    r_format   = (1 + α·f) if em else -α·f         (asymmetric format reward)
    r_div      = d_t · |f - f̄_G|                   (decayed diversity bonus)
    s_L        = min(1, L / think_chars)           (length scaling on think only)
    shaped     = s_L · (r_format + r_div)

Edge cases:
    n_turns=0      → f=0 (no latent-mode bonus earned)
    format_ok=0    → shaped=-1.0 (hard penalty, overrides everything)

`d_t` cosine-decays 1 → 0 over `AR_GRPO_TOTAL` steps so the diversity
term encourages early exploration of latent/explicit mixtures, then
fades and the success-conditioned format reward dominates.

Env vars:
    AR_GRPO_ALPHA   float   latent-mode bonus magnitude    (default 0.3)
    AR_GRPO_L       float   think_chars tolerance          (default 200)
    AR_GRPO_LEN_ON  0/1     enable length scale s_L        (default 1)
    AR_GRPO_DIV_ON  0/1     enable diversity term r_div    (default 0)
    AR_GRPO_TOTAL   int     total steps (cosine horizon)   (default 100)
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch


def _f_latent(n_lat: int, n_think: int) -> float:
    """Fraction of reasoning turns that used latent. Edge case n_turns=0
    returns 0.0 (no latent-mode bonus earned)."""
    n = n_lat + n_think
    return n_lat / n if n > 0 else 0.0


# ── verl monkey-patch ───────────────────────────────────────────────────

def install() -> None:
    """Monkey-patch verl's `extract_reward` (in both `ray_trainer` and
    `reward` modules — the former binds the symbol at import-time).
    """
    import verl.trainer.ppo.ray_trainer as _rt
    import verl.trainer.ppo.reward as _reward_module

    if getattr(_rt.extract_reward, "_is_ar_grpo_patched", False):
        return

    original = _rt.extract_reward
    _call_counter = {"n": 0}

    def _ar_grpo_extract_reward(batch):
        reward_tensor, reward_extra_infos = original(batch)
        need = ("em", "n_think", "n_latent", "think_chars", "format_ok")
        if (any(k not in reward_extra_infos for k in need)
                or "uid" not in batch.non_tensor_batch):
            return reward_tensor, reward_extra_infos

        L = float(os.environ.get("AR_GRPO_L", "200"))
        len_on = os.environ.get("AR_GRPO_LEN_ON", "1") == "1"

        ems = [float(x) for x in reward_extra_infos["em"]]
        nths = [int(round(float(x))) for x in reward_extra_infos["n_think"]]
        nlats = [int(round(float(x))) for x in reward_extra_infos["n_latent"]]
        tchr = [float(x) for x in reward_extra_infos["think_chars"]]
        fok = [bool(float(x) >= 1.0) for x in reward_extra_infos["format_ok"]]
        uids = [str(u) for u in batch.non_tensor_batch["uid"]]
        n = len(ems)

        groups: dict[str, list[int]] = defaultdict(list)
        for i, u in enumerate(uids):
            groups[u].append(i)

        shaped = [0.0] * n
        r_format_per = [0.0] * n
        s_L_per = [1.0] * n
        f_per = [0.0] * n
        r_div_per = [0.0] * n

        alpha = float(os.environ.get("AR_GRPO_ALPHA", "0.3"))
        div_on = os.environ.get("AR_GRPO_DIV_ON", "0") == "1"
        d_t = _compute_d_t(batch, _call_counter) if div_on else 0.0
        f_all = [_f_latent(nlats[i], nths[i]) for i in range(n)]
        for u, idxs in groups.items():
            G = len(idxs)
            f_mean_g = (sum(f_all[i] for i in idxs) / G) if div_on else 0.0
            for i in idxs:
                if not fok[i]:
                    shaped[i] = -1.0
                    r_format_per[i] = -1.0
                    continue
                f = f_all[i]
                r_format = (1.0 + alpha * f) if ems[i] >= 1.0 else -alpha * f
                r_div = d_t * abs(f - f_mean_g) if div_on else 0.0
                s_L = (L / tchr[i]) if (len_on and tchr[i] > L) else 1.0
                f_per[i] = f
                r_format_per[i] = r_format
                r_div_per[i] = r_div
                s_L_per[i] = s_L
                shaped[i] = s_L * (r_format + r_div)

        # ── Stamp the shaped scalar at the final valid token of each rollout
        prompt_length = batch.batch["prompts"].size(1)
        valid_lens = batch.batch["attention_mask"][:, prompt_length:].sum(dim=1).long()
        new_tensor = torch.zeros_like(reward_tensor)
        for i in range(n):
            vl = int(valid_lens[i])
            if vl > 0:
                new_tensor[i, vl - 1] = shaped[i]

        reward_extra_infos = dict(reward_extra_infos)
        reward_extra_infos["shaped_score"] = np.array(shaped, dtype=np.float32)
        reward_extra_infos["ar_format"] = np.array(r_format_per, dtype=np.float32)
        reward_extra_infos["ar_sL"] = np.array(s_L_per, dtype=np.float32)
        reward_extra_infos["ar_f"] = np.array(f_per, dtype=np.float32)
        reward_extra_infos["ar_div"] = np.array(r_div_per, dtype=np.float32)

        # ── Print step summary
        try:
            n_fmt_viol = sum(1 for ok in fok if not ok)
            n_no_reasoning = sum(1 for i in range(n) if fok[i] and (nlats[i] + nths[i]) == 0)
            unbal = reward_extra_infos.get("unbalanced_latent")
            n_unbal = int(sum(float(x) for x in unbal)) if unbal is not None else -1
            common = (
                f"n={n} groups={len(groups)} "
                f"shaped_mean={float(np.mean(shaped)):.3f} "
                f"fmt_mean={float(np.mean(r_format_per)):.3f} "
                f"sL_mean={float(np.mean(s_L_per)):.3f} "
                f"em_mean={float(np.mean(ems)):.3f} "
                f"n_fmt_viol={n_fmt_viol} n_no_reasoning={n_no_reasoning} "
                f"n_unbal={n_unbal}"
            )
            div_tag = (f"d_t={d_t:.3f} div_mean={float(np.mean(r_div_per)):.3f} "
                       if div_on else "")
            print(f"[ar-grpo] α={alpha} L={L} {div_tag}"
                  f"f_mean={float(np.mean(f_per)):.3f} "
                  f"f_std={float(np.std(f_per)):.3f} {common}",
                  file=sys.stdout, flush=True)
        except Exception:
            pass

        return new_tensor, reward_extra_infos

    _ar_grpo_extract_reward._is_ar_grpo_patched = True
    _rt.extract_reward = _ar_grpo_extract_reward
    _reward_module.extract_reward = _ar_grpo_extract_reward
    print("[ar-grpo] monkey-patched verl.trainer.ppo.{ray_trainer,reward}.extract_reward",
          flush=True)


def _compute_d_t(batch, call_counter):
    """Cosine-decayed diversity coefficient.

    Prefers verl's `global_steps` for resume safety; falls back to a
    local counter with a loud warning if missing.
    """
    step_meta = batch.meta_info.get(
        "global_steps", batch.meta_info.get("global_step"))
    if step_meta is not None:
        step = int(step_meta)
    else:
        call_counter["n"] += 1
        start_step = int(os.environ.get("AR_GRPO_START_STEP", "0"))
        step = start_step + call_counter["n"] - 1
        if call_counter["n"] == 1:
            print(f"[ar-grpo] WARN: batch.meta_info has no global_steps; "
                  f"using local counter starting at {start_step}. "
                  f"This will desync across ranks/resumes — set "
                  f"AR_GRPO_START_STEP correctly when resuming.",
                  flush=True)
    total = max(1, int(os.environ.get("AR_GRPO_TOTAL", "100")))
    return 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))

"""Domain-agnostic verl TaskRunner with ALAR install hooks.

Each domain (search/, tool/) provides a hydra entry that subclasses
`LatentTaskRunner` (or composes it) and registers its agent-loop module
via `LATENT_AGENT_LOOP_MODULE`.
"""
from __future__ import annotations

import importlib
import os

from alar.rl.install_patches import install as _install_patches
from verl.trainer.main_ppo import TaskRunner


class LatentTaskRunner(TaskRunner):
    def run(self, config):  # type: ignore[override]
        _install_patches()
        mod = os.environ.get("LATENT_AGENT_LOOP_MODULE")
        if mod:
            try:
                importlib.import_module(mod)
            except Exception as e:
                print(f"[runner] WARN: agent_loop import {mod}: {e}", flush=True)
        return super().run(config)

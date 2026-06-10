"""AR-GRPO entry point (tool domain).

Mirrors `search.rl.main_ar_grpo`: installs the AR-GRPO reward patch
on top of the generic `LatentTaskRunner`, registers the tool agent
loop, then hands off to verl's `run_ppo`.
"""
from __future__ import annotations

import os

import hydra
import ray

import tool.rl.agent_loop  # noqa: F401  fires @register("tool_latent_agent")
from alar.rl.ar_grpo import install as _install_ar_grpo
from alar.rl.install_patches import install as _install_patches
from alar.rl.runner import LatentTaskRunner
from verl.trainer.main_ppo import run_ppo

os.environ.setdefault("LATENT_AGENT_LOOP_MODULE", "tool.rl.agent_loop")


class ARGRPOTaskRunner(LatentTaskRunner):
    def run(self, config):  # type: ignore[override]
        _install_ar_grpo()
        return super().run(config)


@hydra.main(config_path="config", config_name="grpo", version_base=None)
def main(config):
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    _install_patches()
    _install_ar_grpo()  # driver-side too
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ARGRPOTaskRunner))


if __name__ == "__main__":
    main()

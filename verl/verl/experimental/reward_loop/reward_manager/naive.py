# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
import os
import threading
from pathlib import Path

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

# V11 Stage 0 rollout-saving hook (opt-in via STAGE0_REWARD1_LOG env var).
# Mirrors the hook in verl/workers/reward_manager/naive.py — verl uses the
# async reward_loop path with AgentLoop rollouts, so this is the version
# that actually fires for Stage 0. Search-RL uses custom_reward_function and
# does not set STAGE0_REWARD1_LOG → byte-identical behavior there.
_STAGE0_LOG_PATH = os.environ.get("STAGE0_REWARD1_LOG")
_STAGE0_THRESHOLD = float(os.environ.get("STAGE0_REWARD1_THRESHOLD", "1.0"))
_STAGE0_DATA_SOURCE = os.environ.get("STAGE0_REWARD1_DATA_SOURCE", "toolcall")
_STAGE0_LOCK = threading.Lock()


@register("naive")
class NaiveRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        # V11 Stage 0 rollout-saving hook (opt-in; gated on env var +
        # data_source so search-domain RL is byte-identical).
        if (
            _STAGE0_LOG_PATH
            and str(data_source) == _STAGE0_DATA_SOURCE
            and float(reward) >= _STAGE0_THRESHOLD
        ):
            try:
                # Decode prompt for the saved record (response_str already decoded above).
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                rec = {
                    "prompt": prompt_str,
                    "response": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "extra_info": {k: v for k, v in extra_info.items()
                                   if k not in ("rollout_reward_scores",)},
                    "score": float(reward),
                }
                rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
                log_path = Path(_STAGE0_LOG_PATH)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                out = log_path.with_suffix(log_path.suffix + f".rank{rank}.pid{os.getpid()}")
                line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
                with _STAGE0_LOCK:
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(line)
            except Exception as e:
                # Never let logging break training.
                print(f"[STAGE0_LOG] write failed: {type(e).__name__}: {e}")

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

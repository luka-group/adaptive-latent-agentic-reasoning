"""Tool-use SFT dataset (mirrors `search.dataset.SearchLatentDataset`).

Each training sample is the full Qwen3-4B-Thinking-2507 native-tool-calling
sequence with reasoning replaced per-turn by either a latent block or a
verbal `<think>` block, controlled by `latent_prob`:

    <|im_start|>system\n{SYSTEM_PROMPT}\n\n# Tools\n...<tools>{tool1}\n{tool2}...</tools>...<|im_end|>\n
    <|im_start|>user\n{question}<|im_end|>\n
    <|im_start|>assistant\n
        <REASONING_BLOCK>             # <latent>••••</latent> or <think>...</think>
        {free_text?}\n
        <tool_call>\n{json}\n</tool_call>
        [\n<tool_call>...</tool_call>]*
    <|im_end|>\n
    <|im_start|>user\n<tool_response>\n{body}\n</tool_response><|im_end|>\n
    <|im_start|>assistant\n
        <REASONING_BLOCK>
        {answer}
    <|im_end|>

Loss mask:
  - system + user prefix                      -> loss-OFF
  - K bullet sentinels                        -> loss-OFF (latent_mask=True)
  - latent/think open + close + reasoning     -> loss-ON
  - free_text + tool_call open/json/close     -> loss-ON
  - `<|im_end|>` after assistant turn         -> loss-ON   (model emits to stop)
  - `\n` + `<|im_start|>...` scaffolding      -> loss-OFF  (structural)
  - `<tool_response>...</tool_response>` body -> loss-OFF
  - final answer text                         -> loss-ON
  - trailing `<|im_end|>`                     -> loss-ON

latent_mask is True only at the K bullet positions inside `<latent>...</latent>`.
"""

from __future__ import annotations

import json
import random
from typing import Iterable

import torch

from alar.sft_data.base import IGNORE_INDEX, LatentDataset
from tool.config import SYSTEM_PROMPT


class ToolLatentDataset(LatentDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        num_latent: int = 4,
        max_length: int = 0,
        max_samples: int = -1,
        data_offset: int = 0,
        min_score: float = 1.5,
        latent_prob: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(tokenizer=tokenizer, num_latent=num_latent, max_length=max_length)
        self.latent_prob = float(latent_prob)
        self.seed = int(seed)
        self._load(data_path, min_score=min_score)
        if data_offset > 0:
            self.trajectories = self.trajectories[data_offset:]
        if max_samples > 0 and len(self.trajectories) > max_samples:
            self.trajectories = self.trajectories[:max_samples]

        # Precompute the Qwen3 scaffolding ids we splice in repeatedly.
        self.im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_end_nl_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        self.im_start_assist_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        self.nl_ids = tokenizer.encode("\n", add_special_tokens=False)

        print(f"[tool.dataset] loaded {len(self.trajectories)} trajectories "
              f"K={self.K} sentinel_id={self.sentinel_id} "
              f"latent_prob={self.latent_prob}")

    def _load(self, path: str, min_score: float) -> None:
        n_total = n_kept = 0
        with open(path) as f:
            for line in f:
                n_total += 1
                r = json.loads(line)
                if (r.get("score") or 0.0) < min_score:
                    continue
                # A trajectory is useful only if it has at least one
                # supervised assistant span (tool calls or a final answer).
                if not r.get("turns") and not r.get("answer"):
                    continue
                self.trajectories.append(r)
                n_kept += 1
        print(f"[tool.dataset] kept {n_kept}/{n_total} traces "
              f"(min_score={min_score})")

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        msgs = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": traj["question"]},
        ]
        # Render the prompt prefix natively (Qwen3 emits the tools block,
        # message envelopes, and trailing `<|im_end|>\n`). We then open the
        # assistant turn ourselves so we can splice the latent sentinel.
        prefix = self.tokenizer.apply_chat_template(
            msgs, tools=traj["tools"],
            add_generation_prompt=False, tokenize=False,
        )
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)

        ids: list[int] = []
        labels: list[int] = []
        latent_mask: list[int] = []

        def append(toks: Iterable[int], loss_on: bool, is_latent: bool = False):
            toks = list(toks)
            ids.extend(toks)
            labels.extend(toks if loss_on else [IGNORE_INDEX] * len(toks))
            latent_mask.extend([1 if is_latent else 0] * len(toks))

        def append_latent_block():
            append(self.latent_open_ids, loss_on=True)
            append([self.sentinel_id] * self.K, loss_on=False, is_latent=True)
            append(self.latent_close_ids, loss_on=True)
            append(self.sep_nn_ids, loss_on=True)

        def append_think_block(text: str):
            think_ids = self.tokenizer.encode(
                f"<think>\n{text}\n</think>\n\n", add_special_tokens=False,
            )
            append(think_ids, loss_on=True)

        def append_reasoning(text: str):
            if text and rng.random() < self.latent_prob:
                append_latent_block()
            elif text:
                append_think_block(text)
            else:
                append_latent_block()

        # Prompt prefix (system + user, both loss-off).
        append(prefix_ids, loss_on=False)

        rng = random.Random(self.seed * 1_000_003 + idx)

        # Each tool-using turn: assistant(reasoning + tool_calls) <|im_end|>
        # then tool response wrapped as a user turn, then re-open assistant.
        for turn in traj["turns"]:
            # Open assistant (structural; not predicted).
            append(self.im_start_assist_ids, loss_on=False)

            append_reasoning(turn.get("thinking", "") or "")

            ft = (turn.get("free_text") or "").strip()
            if ft:
                ft_ids = self.tokenizer.encode(ft, add_special_tokens=False)
                append(ft_ids, loss_on=True)
                # Qwen3 template emits a `\n` between content and the first
                # tool_call when both are present.
                append(self.nl_ids, loss_on=True)

            calls = turn.get("tool_calls") or []
            for j, call_json in enumerate(calls):
                if j > 0:
                    append(self.nl_ids, loss_on=True)
                tc_ids = self.tokenizer.encode(
                    f"<tool_call>\n{call_json}\n</tool_call>",
                    add_special_tokens=False,
                )
                append(tc_ids, loss_on=True)

            # End-of-turn token (model emits this to hand off to the tool).
            append(self.im_end_ids, loss_on=True)

            resp = turn.get("tool_response")
            if resp is None:
                # Single-turn rollout (no tool execution available) — stop here.
                continue
            # Tool response wrapped as a user turn. Entirely structural /
            # external input → all loss-off.
            tr_ids = self.tokenizer.encode(
                f"\n<|im_start|>user\n<tool_response>\n{resp}\n"
                f"</tool_response><|im_end|>\n",
                add_special_tokens=False,
            )
            append(tr_ids, loss_on=False)

        # Final answer turn (only if present — many rollouts cut off after
        # the first tool call).
        if traj.get("answer"):
            # `tr_ids` already ends with `\n`, so we open the next assistant
            # turn directly. (If the trajectory had no tool_response, the
            # rollout never reached a final-answer turn anyway.)
            append(self.im_start_assist_ids, loss_on=False)
            append_reasoning(traj.get("answer_thinking", "") or "")
            ans_ids = self.tokenizer.encode(
                traj["answer"], add_special_tokens=False,
            )
            append(ans_ids, loss_on=True)
            append(self.im_end_ids, loss_on=True)

        ids_t = torch.tensor(ids, dtype=torch.long)
        lab_t = torch.tensor(labels, dtype=torch.long)
        lat_t = torch.tensor(latent_mask, dtype=torch.bool)
        ids_t, lab_t, lat_t = self._truncate(ids_t, lab_t, lat_t)
        return {
            "input_ids":   ids_t,
            "labels":      lab_t,
            "latent_mask": lat_t,
        }

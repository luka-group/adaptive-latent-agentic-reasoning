"""Search-domain SFT dataset.

Wraps a teacher-trace JSONL into supervised training examples. Each
example is a flat token sequence:

    <system_prompt><user_question><assistant>
        <REASONING_BLOCK_1> <search> ... </search> <information> ... </information>
        <REASONING_BLOCK_2> <search> ... </search> <information> ... </information>
        ...
        <REASONING_BLOCK_N> <answer> ... </answer> <eos>

Each REASONING_BLOCK is independently chosen at construction time to be
either:
  - latent:  `<latent>••••</latent>` (K=4 bullet sentinels, plugin replaces
             their embeddings with projector outputs at inference)
  - verbal:  `<think>{text}</think>`

Loss mask:
  - latent open/close, think open/close, query, answer  -> loss-ON
  - K bullet sentinels                                   -> loss-OFF
  - <information>...</information>                       -> loss-OFF
  - prompt prefix                                        -> loss-OFF

latent_mask is True at the K bullet positions (so the modeling forward
can splice in projector outputs at training time).
"""

from __future__ import annotations

import json
import random
from typing import Iterable

import torch

from alar.sft_data.base import IGNORE_INDEX, LatentDataset
from search.config import (
    ACTION_CLOSE, ACTION_OPEN, ANSWER_CLOSE, ANSWER_OPEN,
    OBS_CLOSE, OBS_OPEN, SYSTEM_PROMPT,
)


class SearchLatentDataset(LatentDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        num_latent: int = 4,
        max_length: int = 0,
        max_samples: int = -1,
        data_offset: int = 0,
        min_em: float = 0.0,
        max_doc_chars: int = 500,
        filter_truncated: bool = True,
        latent_prob: float = 0.5,
        seed: int = 42,
    ):
        super().__init__(tokenizer=tokenizer, num_latent=num_latent, max_length=max_length)
        self.max_doc_chars = int(max_doc_chars)
        self.latent_prob = float(latent_prob)
        self.seed = int(seed)
        self._load(data_path, min_em=min_em, filter_truncated=filter_truncated)
        # Apply offset first (skip the first N filtered trajectories), then cap.
        if data_offset > 0:
            self.trajectories = self.trajectories[data_offset:]
        if max_samples > 0 and len(self.trajectories) > max_samples:
            self.trajectories = self.trajectories[:max_samples]
        print(f"[search.dataset] loaded {len(self.trajectories)} trajectories "
              f"K={self.K} sentinel_id={self.sentinel_id} latent_prob={self.latent_prob} "
              f"offset={data_offset}")

    def _load(self, path: str, min_em: float, filter_truncated: bool) -> None:
        n_total = n_kept = 0
        with open(path) as f:
            for line in f:
                n_total += 1
                r = json.loads(line)
                if (r.get("em") or 0) < min_em:
                    continue
                if not r.get("pred_answer"):
                    continue
                traj = r.get("trajectory", [])
                if filter_truncated and any(s.get("finish_reason") == "length" for s in traj):
                    continue
                turns = []
                prev_q = None
                for step in traj:
                    q = step.get("query", "")
                    if not q or q == prev_q:
                        continue
                    prev_q = q
                    doc_text = step.get("snippets", "") or ""
                    if self.max_doc_chars > 0 and doc_text:
                        doc_text = self._cap_doc_chars(doc_text)
                    turns.append({
                        "query": q,
                        "doc_text": doc_text,
                        "thinking": (step.get("thinking") or "").strip(),
                    })
                self.trajectories.append({
                    "question": r["question"],
                    "answer": r["answer"],
                    "answer_thinking": (r.get("answer_thinking") or "").strip(),
                    "turns": turns,
                })
                n_kept += 1
        print(f"[search.dataset] kept {n_kept}/{n_total} traces")

    def _cap_doc_chars(self, doc_text: str) -> str:
        """Cap each `Doc N(Title: ...) ...` block at `max_doc_chars` chars."""
        blocks = doc_text.split("\nDoc ")
        for i in range(1, len(blocks)):
            blocks[i] = "Doc " + blocks[i]
        blocks = [b[: self.max_doc_chars] for b in blocks]
        return "\n".join(blocks)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        q = traj["question"].strip()
        if q and q[-1] != "?":
            q += "?"
        user = f"{SYSTEM_PROMPT} Question: {q}\n"
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False,
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
            # `<latent>` (loss-ON) + K sentinels (loss-OFF, latent-mask=True)
            # + `</latent>` (loss-ON) + `\n\n` (loss-ON).
            append(self.latent_open_ids, loss_on=True)
            append([self.sentinel_id] * self.K, loss_on=False, is_latent=True)
            append(self.latent_close_ids, loss_on=True)
            append(self.sep_nn_ids, loss_on=True)

        def append_think_block(text: str):
            think_ids = self.tokenizer.encode(
                f"<think>{text}</think>\n\n", add_special_tokens=False,
            )
            append(think_ids, loss_on=True)

        append(prefix_ids, loss_on=False)
        # Uniform <latent> precedence: every block (including block-0)
        # is now preceded by `\n\n`, matching what info_ids emits before
        # subsequent blocks. Without this the first block's K-th sentinel
        # sees `\n<latent>••••` while later blocks see `\n\n<latent>••••`,
        # and the model fails to learn `</latent>` at the first block.
        append(self.sep_nn_ids, loss_on=True)

        rng = random.Random(self.seed * 1_000_003 + idx)
        # Intermediate turns: per-turn coin flip between latent and verbal.
        # Falls back to latent if the source 'thinking' field is empty so we
        # never emit an empty supervised <think></think> block.
        for turn in traj["turns"]:
            think_text = turn["thinking"]
            if think_text and rng.random() < self.latent_prob:
                append_latent_block()
            elif think_text:
                append_think_block(think_text)
            else:
                append_latent_block()

            search_ids = self.tokenizer.encode(
                f"{ACTION_OPEN} {turn['query']} {ACTION_CLOSE}",
                add_special_tokens=False,
            )
            append(search_ids, loss_on=True)

            info_ids = self.tokenizer.encode(
                f"\n\n{OBS_OPEN}{turn['doc_text']}{OBS_CLOSE}\n\n",
                add_special_tokens=False,
            )
            append(info_ids, loss_on=False)

        # Final answer turn: same per-turn choice for the reasoning block.
        ans_think = traj["answer_thinking"]
        if ans_think and rng.random() < self.latent_prob:
            append_latent_block()
        elif ans_think:
            append_think_block(ans_think)
        else:
            append_latent_block()

        ans_ids = self.tokenizer.encode(
            f"{ANSWER_OPEN}{traj['answer']}{ANSWER_CLOSE}",
            add_special_tokens=False,
        )
        append(ans_ids, loss_on=True)
        append(self.eos_ids, loss_on=False)

        ids_t = torch.tensor(ids, dtype=torch.long)
        lab_t = torch.tensor(labels, dtype=torch.long)
        lat_t = torch.tensor(latent_mask, dtype=torch.bool)
        ids_t, lab_t, lat_t = self._truncate(ids_t, lab_t, lat_t)
        return {
            "input_ids":   ids_t,
            "labels":      lab_t,
            "latent_mask": lat_t,
        }

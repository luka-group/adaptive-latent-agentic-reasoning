"""Abstract base for latent-reasoning SFT datasets.

Encodes only the latent-block primitives. Domain-specific tokens
(`<search>`/`<answer>`/...) and per-domain data loading live in the
subclass.

The latent block surface form is identical at SFT, RL, and eval:
    <latent>••••</latent>
where `••••` is K=4 bullet characters (id 6667, a single-piece
non-merging BPE token in the Qwen2.5/Qwen3 vocabulary). At inference the vLLM plugin replaces the four
bullet embeddings with projector outputs; the model is trained to emit
the multi-piece BPE `</latent>` close after the K positions.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100

# Sentinel that occupies each of the K latent positions. The plugin
# overrides the embedding at runtime; the token id is only what the
# decoder displays under skip_special_tokens=True.
SENTINEL_ID = 6667  # '•' — single-piece BPE, does not merge with itself


class LatentDataset(Dataset):
    """Base class. Subclasses populate `self.trajectories` and implement
    `__getitem__`.
    """

    def __init__(self, tokenizer, num_latent: int = 4, max_length: int = 0):
        self.tokenizer = tokenizer
        self.K = int(num_latent)
        assert self.K >= 1, f"num_latent must be >= 1; got {self.K}"
        self.max_length = int(max_length)

        # <latent> open + </latent> close are multi-piece BPE in Qwen2.5.
        self.latent_open_ids = tokenizer.encode("<latent>", add_special_tokens=False)
        self.latent_close_ids = tokenizer.encode("</latent>", add_special_tokens=False)
        assert len(self.latent_open_ids) >= 1
        assert len(self.latent_close_ids) >= 1

        # Separator emitted after each latent block so the next action
        # (e.g. <search>) is well-bracketed.
        self.sep_nn_ids = tokenizer.encode("\n\n", add_special_tokens=False)

        # EOS terminates the assistant turn.
        self.eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

        self.sentinel_id = SENTINEL_ID
        self.trajectories: list[dict] = []

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):  # pragma: no cover
        raise NotImplementedError("subclass must implement __getitem__")

    def _truncate(self, ids, labels, latent_mask):
        if self.max_length <= 0 or ids.shape[0] <= self.max_length:
            return ids, labels, latent_mask
        L = self.max_length
        return ids[:L], labels[:L], latent_mask[:L]

    def collate_fn(self, batch):
        """Right-pad variable-length samples to the batch max length.

        Pads input_ids with pad_token_id, labels with -100 (ignored by CE),
        and latent_mask with 0. attention_mask is 1 for real tokens, 0 for
        pad. Single-sample batches hit a fast path with no padding.
        """
        if len(batch) == 1:
            item = batch[0]
            T = int(item["input_ids"].shape[0])
            return {
                "input_ids":      item["input_ids"].unsqueeze(0),
                "labels":         item["labels"].unsqueeze(0),
                "latent_mask":    item["latent_mask"].unsqueeze(0),
                "attention_mask": torch.ones(1, T, dtype=torch.long),
            }

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        lengths = [int(b["input_ids"].shape[0]) for b in batch]
        T = max(lengths)
        B = len(batch)
        input_ids   = torch.full((B, T), pad_id, dtype=batch[0]["input_ids"].dtype)
        labels      = torch.full((B, T), -100, dtype=batch[0]["labels"].dtype)
        latent_mask = torch.zeros((B, T), dtype=batch[0]["latent_mask"].dtype)
        attn        = torch.zeros((B, T), dtype=torch.long)
        for i, item in enumerate(batch):
            L = lengths[i]
            input_ids[i, :L]   = item["input_ids"]
            labels[i, :L]      = item["labels"]
            latent_mask[i, :L] = item["latent_mask"]
            attn[i, :L]        = 1
        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "latent_mask":    latent_mask,
            "attention_mask": attn,
        }

"""BFCL evaluation driver — vLLM + ALAR plugin, Qwen3 native tool calling.

For each BFCL example we:
  1. Render `apply_chat_template([{system}, {user question}], tools=[...])`
     using the trained model's tokenizer (so the tools block matches the
     surface form the model saw during AASD).
  2. Generate to `<|im_end|>` with vLLM. The latent plugin
     (`alar.vllm_plugin`) splices K projector outputs into the
     `<latent>••••</latent>` sentinel positions when the model emits
     `<latent>`. Disable with `--no_latent` for ablations or for stock
     Qwen3 baselines.
  3. Parse `<tool_call>{json}</tool_call>` blocks out of the generation
     and AST-check the (name, args) tuples against the BFCL
     possible-answer ground truth.

Scope: the non-executable AST-checked categories
  simple, multiple, parallel, parallel_multiple,
  live_simple, live_multiple, live_parallel, live_parallel_multiple
Out of scope (for now): irrelevance/relevance, executable, multi-turn,
java/javascript/rest/sql/chatable.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("LATENT_VLLM", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import alar.modeling  # noqa: F401  (registers latent_qwen{2,3} model_type)
from alar.vllm_plugin import register

register()

from vllm import LLM, SamplingParams, TokensPrompt

from tool.config import SYSTEM_PROMPT


BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*\n?(.*?)\n?\s*</tool_call>", re.DOTALL)
LATENT_OPEN_RE = re.compile(r"<latent>")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ----- Data loading ---------------------------------------------------

def load_bfcl_category(cat: str, max_examples: int = -1) -> List[Dict]:
    """Load one BFCL category + its possible-answer ground truth.

    The HF repo stores BFCL files with a `.json` extension but JSONL
    contents (one record per line).
    """
    qpath = hf_hub_download(BFCL_REPO, f"BFCL_v3_{cat}.json", repo_type="dataset")
    apath = hf_hub_download(BFCL_REPO, f"possible_answer/BFCL_v3_{cat}.json",
                            repo_type="dataset")
    qs: Dict[str, Dict] = {}
    with open(qpath) as f:
        for line in f:
            r = json.loads(line)
            qs[r["id"]] = r
    out: List[Dict] = []
    with open(apath) as f:
        for line in f:
            a = json.loads(line)
            if a["id"] not in qs:
                continue
            q = qs[a["id"]]
            out.append({
                "id":          q["id"],
                "question":    q["question"],   # nested: [[{user}, ...], ...]
                "function":    q["function"],
                "ground_truth": a["ground_truth"],
            })
            if max_examples > 0 and len(out) >= max_examples:
                break
    return out


def _prompt_messages(ex: Dict, system_prompt: Optional[str]) -> List[Dict]:
    """Convert BFCL `question` (nested conversation turns) into a flat
    chat template input. Non-multi-turn BFCL uses `[[{user}]]`.

    When `system_prompt` is None (baseline mode) we don't inject any
    system message; Qwen3's template still renders the tools block on
    its own.
    """
    msgs: List[Dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    # Use the first dialogue (BFCL stores one per example unless multi-turn).
    for turn in ex["question"][0]:
        msgs.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    return msgs


# ----- AST-style scoring ---------------------------------------------

def parse_tool_calls(text: str) -> List[Dict]:
    """Extract `<tool_call>{json}</tool_call>` blocks into structured
    `{name, arguments}` dicts (in the order they appear)."""
    out: List[Dict] = []
    for m in TOOL_CALL_RE.finditer(text):
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
            out.append({
                "name": obj.get("name"),
                "arguments": obj.get("arguments", {}) or {},
            })
        except json.JSONDecodeError:
            out.append({"name": None, "arguments": {}, "_parse_error": True})
    return out


def _norm(v: Any) -> Any:
    """Normalize a value for value-equality. Numbers compare by float
    value; strings strip whitespace; bools stay bool; lists/dicts
    recurse. We intentionally keep bool != int so `True` doesn't match
    the integer `1`.
    """
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, str):
        return ("str", v.strip())
    if isinstance(v, list):
        return ("list", [_norm(x) for x in v])
    if isinstance(v, dict):
        return ("dict", {k: _norm(x) for k, x in v.items()})
    if v is None:
        return ("none",)
    return ("other", repr(v))


def _val_acceptable(value: Any, allowed_list: List) -> bool:
    """Compare `value` against BFCL's per-arg `allowed_list`.

    The list enumerates acceptable concrete values. A literal `""` in
    the list is BFCL's sentinel for `optional`: if the model omits the
    arg entirely that's fine. We never reach here when the arg is
    omitted (the caller handles that), so we just ignore the `""`
    sentinel when scanning values.
    """
    if not allowed_list:
        return True
    nv = _norm(value)
    for a in allowed_list:
        if a == "":
            continue
        if _norm(a) == nv:
            return True
        # Order-insensitive list match: many BFCL list args don't fix
        # ordering. Try a sorted comparison as a fallback.
        if isinstance(a, list) and isinstance(value, list):
            try:
                if sorted([_norm(x) for x in a]) == sorted([_norm(x) for x in value]):
                    return True
            except TypeError:
                pass
    return False


def check_single_call(pred: Dict, gt_entry: Dict) -> bool:
    """`gt_entry`: `{func_name: {arg_name: [allowed_values, ...], ...}}`.

    Pass criteria:
      - predicted name == ground-truth name
      - every required arg in `gt_entry` is present in pred with an
        acceptable value (or absent if the allowed list contains "")
      - extra args in pred that aren't in `gt_entry` are ignored
        (BFCL official is sometimes stricter; we trade a small false
        positive rate for robustness against optional-arg variation)
    """
    if not isinstance(gt_entry, dict) or len(gt_entry) != 1:
        return False
    fn_name = next(iter(gt_entry.keys()))
    if pred.get("name") != fn_name:
        return False
    gt_args = gt_entry[fn_name] or {}
    pred_args = pred.get("arguments", {}) or {}
    for arg_name, allowed in gt_args.items():
        present = arg_name in pred_args
        if not present:
            # Acceptable only if "optional" sentinel is in the allowed list.
            if "" in (allowed or []):
                continue
            return False
        if not _val_acceptable(pred_args[arg_name], allowed):
            return False
    return True


def check_call_set(preds: List[Dict], gts: List[Dict]) -> bool:
    """Match a list of predicted calls to a list of ground-truth calls
    irrespective of ordering. We try permutations of `gts` since the
    parallel categories have small list sizes (typically 2-4).
    """
    if len(preds) != len(gts):
        return False
    # Fast-path: positional match.
    if all(check_single_call(p, g) for p, g in zip(preds, gts)):
        return True
    # Order-insensitive: try permutations of gts.
    if len(preds) > 6:
        # Fall back to greedy match to avoid factorial blowup.
        used = [False] * len(preds)
        for g in gts:
            for i, p in enumerate(preds):
                if not used[i] and check_single_call(p, g):
                    used[i] = True
                    break
            else:
                return False
        return True
    for perm in itertools.permutations(gts):
        if all(check_single_call(p, g) for p, g in zip(preds, perm)):
            return True
    return False


# ----- Driver ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_ckpt", required=True,
                    help="vLLM-loadable Qwen3 ckpt dir (LoRA-merged) with "
                         "projector.pt sidecar produced by merge_sft.")
    ap.add_argument("--categories", nargs="+", default=[
        "simple", "multiple", "parallel", "parallel_multiple",
    ], help="BFCL AST-checked categories to evaluate.")
    ap.add_argument("--max_examples", type=int, default=-1,
                    help="Cap examples per category (debug).")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--num_latent", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--output_dir", required=True,
                    help="Directory for per-category JSONL outputs + summary.json.")
    ap.add_argument("--no_latent", action="store_true",
                    help="Disable latent plugin (stock Qwen3 baseline / ablation).")
    ap.add_argument("--system_prompt", choices=["tool", "none"], default="tool",
                    help="'tool' = inject tool.config.SYSTEM_PROMPT (latent/think "
                         "mode preamble). 'none' = no system message — let the "
                         "chat template render tools by themselves. Use 'none' "
                         "for stock-model baselines.")
    args = ap.parse_args()
    system_prompt = SYSTEM_PROMPT if args.system_prompt == "tool" else None

    os.environ["LATENT_NUM_LATENT"] = str(args.num_latent)
    if args.no_latent:
        os.environ["LATENT_VLLM"] = "0"

    print(f"[bfcl] loading {args.merged_ckpt}", flush=True)
    llm = LLM(
        model=args.merged_ckpt,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=True,                # plugin attaches projector after trace
        enable_prefix_caching=True,
    )
    tok = AutoTokenizer.from_pretrained(args.merged_ckpt, trust_remote_code=True)
    print(f"[bfcl] eos={tok.eos_token!r}({tok.eos_token_id}) "
          f"K={args.num_latent} latent_vllm={os.environ.get('LATENT_VLLM')}",
          flush=True)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tok.eos_token_id],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    overall: Dict[str, Dict] = {}

    for cat in args.categories:
        exs = load_bfcl_category(cat, args.max_examples)
        print(f"\n[bfcl/{cat}] n={len(exs)}", flush=True)
        if not exs:
            continue

        # Latent mode: render with `add_generation_prompt=False` and open
        # the assistant turn ourselves. Qwen3-Thinking-2507's template
        # otherwise appends `<think>\n` after `<|im_start|>assistant\n`,
        # which puts the model OUT of the AASD training distribution
        # (Stage 1 with latent_prob=1.0 always opened the assistant turn
        # with `<latent>` immediately after `<|im_start|>assistant\n`).
        # See `tool/dataset.py::__getitem__`.
        # Baseline mode (`--no_latent`): use the native generation prompt
        # so the stock thinking model sees its expected `<think>\n` prefix.
        prompt_ids: List[List[int]] = []
        for ex in exs:
            if args.no_latent:
                ptext = tok.apply_chat_template(
                    _prompt_messages(ex, system_prompt),
                    tools=ex["function"],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prefix = tok.apply_chat_template(
                    _prompt_messages(ex, system_prompt),
                    tools=ex["function"],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                ptext = prefix + "<|im_start|>assistant\n"
            prompt_ids.append(tok.encode(ptext, add_special_tokens=False))

        t0 = time.time()
        outs = llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompt_ids],
            sp, use_tqdm=True,
        )
        elapsed = time.time() - t0

        results: List[Dict] = []
        n_correct = 0
        for ex, out in zip(exs, outs):
            gen_ids = list(out.outputs[0].token_ids)
            gen_text = tok.decode(gen_ids, skip_special_tokens=False)
            preds = parse_tool_calls(gen_text)
            ok = check_call_set(preds, ex["ground_truth"])
            n_correct += int(ok)
            results.append({
                "id":           ex["id"],
                "correct":      ok,
                "preds":        preds,
                "ground_truth": ex["ground_truth"],
                "generation":   gen_text,
                "n_latent":     len(LATENT_OPEN_RE.findall(gen_text)),
                "n_think":      len(THINK_RE.findall(gen_text)),
                "gen_tokens":   len(gen_ids),
            })

        acc = 100.0 * n_correct / max(len(exs), 1)
        overall[cat] = {
            "n": len(exs), "n_correct": n_correct, "accuracy": acc,
            "elapsed_s": round(elapsed, 1),
            "tokens_per_s": round(sum(r["gen_tokens"] for r in results) / max(elapsed, 1e-6), 1),
        }
        print(f"[bfcl/{cat}] acc={acc:.1f}% ({n_correct}/{len(exs)}) "
              f"in {elapsed:.1f}s", flush=True)

        out_path = os.path.join(args.output_dir, f"bfcl_{cat}.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[bfcl/{cat}] wrote {out_path}", flush=True)

    print()
    print("=" * 60)
    print("  BFCL summary")
    print("=" * 60)
    for cat, stats in overall.items():
        print(f"  {cat:<28s} {stats['accuracy']:5.1f}%   "
              f"({stats['n_correct']}/{stats['n']})")
    if overall:
        macro = sum(s["accuracy"] for s in overall.values()) / len(overall)
        print(f"  {'macro avg':<28s} {macro:5.1f}%")
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\n[bfcl] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()

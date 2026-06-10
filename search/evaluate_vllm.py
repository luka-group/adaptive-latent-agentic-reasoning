"""ALAR multi-turn agentic eval driver — vLLM + ALAR plugin.

Per-turn loop:

  1. Feed `prompt + cumulative_text` to vLLM, stopping at `</search>`
     / `</answer>`. When the model emits `<latent>` (3 BPE pieces), the
     plugin (`alar.vllm_plugin`) splices K projector outputs into the
     next K sentinel positions, then lets the model emit `</latent>`
     naturally.
  2. Token-level cumulative + prefix caching keeps the K sentinel
     positions intact across turns — their KV is the projector's
     output, so re-prefilling text would compute different hidden
     states. We never strip the latent block.
  3. On `</search>`: retrieve, append `<information>…</information>\\n\\n`.
  4. On `</answer>` / EOS / length: terminate.

Activation: `LATENT_VLLM=1` (the `vllm.general_plugins` entry point
in pyproject.toml auto-registers the plugin in every worker subprocess
when the package is installed).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import time
from typing import Dict, List, Optional

os.environ.setdefault("LATENT_VLLM", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

import alar.modeling  # noqa: F401  (defensive — registers latent_qwen2 model_type)
from alar.vllm_plugin import register

register()

from vllm import LLM, SamplingParams, TokensPrompt

# Optionally override the system prompt via env var (e.g. for ablations
# with a different prompt). Format: dotted module path exporting a
# `SYSTEM_PROMPT` string.
_EVAL_PROMPT_MODULE = os.environ.get("EVAL_SYSTEM_PROMPT_MODULE")
if _EVAL_PROMPT_MODULE:
    import importlib
    SYSTEM_PROMPT = importlib.import_module(_EVAL_PROMPT_MODULE).SYSTEM_PROMPT
    print(f"[eval] SYSTEM_PROMPT overridden from {_EVAL_PROMPT_MODULE}", flush=True)
else:
    from search.config import SYSTEM_PROMPT

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
LATENT_OPEN_RE = re.compile(r"<latent>")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())


def compute_em(prediction: str, golds: List[str]) -> float:
    if not prediction or not golds:
        return 0.0
    pn = _normalize(prediction)
    return 1.0 if any(_normalize(g) == pn for g in golds) else 0.0


def search_remote_batch(queries: List[str], url: str, top_k: int = 3,
                        max_doc_chars: int = 500) -> List[str]:
    base = url.rstrip("/")
    if not base.endswith("/retrieve"):
        base += "/retrieve"
    try:
        r = requests.post(base, json={"queries": queries, "topk": top_k}, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[eval] retrieval failed: {e}", flush=True)
        return ["" for _ in queries]
    raw = data.get("results") or data.get("result") or []
    out: List[str] = []
    for hits in raw:
        if not hits:
            out.append("")
            continue
        blocks = [
            f"Doc {i+1}(Title: {d.get('title','')}) {d.get('text','')}"
            for i, d in enumerate(hits)
        ]
        if max_doc_chars > 0:
            blocks = [b[:max_doc_chars] for b in blocks]
        out.append("\n".join(blocks))
    while len(out) < len(queries):
        out.append("")
    return out


def load_eval_dataset(name: str, max_examples: int) -> List[Dict]:
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", name)
    split = ds.get("test") or ds.get("dev") or ds["train"]
    n = len(split) if max_examples < 0 else min(max_examples, len(split))
    return [{
        "id": split[i].get("id", str(i)),
        "question": split[i]["question"],
        "golden_answers": split[i]["golden_answers"],
        "dataset": name,
    } for i in range(n)]


def load_eval_jsonl(path: str, max_examples: int) -> List[Dict]:
    out: List[Dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            q = r.get("question") or r.get("query")
            if q is None:
                continue
            golds = r.get("golden_answers")
            if golds is None:
                a = r.get("answer")
                if a is None:
                    continue
                golds = [a] if isinstance(a, str) else list(a)
            out.append({
                "id": r.get("id", str(i)),
                "question": q,
                "golden_answers": golds,
                "dataset": "custom_jsonl",
            })
            if max_examples > 0 and len(out) >= max_examples:
                break
    return out


def get_last_query(text: str) -> Optional[str]:
    m = SEARCH_RE.findall(text)
    return m[-1].strip() if m else None


def get_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def _truncate_to_stop(clean_ids: List[int], clean_text: str, tokenizer):
    """vLLM detok can emit a few extra characters past the stop string
    (byte-level BPE buffers a whole token ahead). Trim text + ids back
    to the stop-string boundary so the next turn's prefill sees a
    clean structure.
    """
    for stop in ("</search>", "</answer>"):
        if stop in clean_text:
            stop_char_end = clean_text.rindex(stop) + len(stop)
            trimmed_text = clean_text[:stop_char_end]
            cutoff = len(clean_ids)
            for k in range(1, len(clean_ids) + 1):
                if len(tokenizer.decode(clean_ids[:k], skip_special_tokens=False)) >= stop_char_end:
                    cutoff = k
                    break
            return trimmed_text, clean_ids[:cutoff]
    return clean_text, clean_ids


def evaluate_batch(
    llm: LLM, tokenizer, examples: List[Dict], *,
    retrieval_url: str, max_turns: int, max_new_tokens: int, search_top_k: int,
    max_doc_chars: int = 500, debug: bool = False, temperature: float = 0.0,
) -> List[Dict]:
    n = len(examples)
    prompt_token_ids: List[List[int]] = []
    cumulatives: List[str] = [""] * n
    cumulatives_ids: List[List[int]] = [[] for _ in range(n)]
    n_search = [0] * n
    total_tok = [0] * n
    termination = ["max_turns_exceeded"] * n
    answers: List[str] = [""] * n
    active = [True] * n

    for ex in examples:
        q = ex["question"].strip()
        if q and q[-1] != "?":
            q += "?"
        ptext = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{SYSTEM_PROMPT} Question: {q}\n"}],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_token_ids.append(tokenizer.encode(ptext, add_special_tokens=False))

    sp_kwargs = dict(temperature=temperature, max_tokens=max_new_tokens,
                     stop=["</search>", "</answer>"])

    t_start = time.time()
    for turn in range(max_turns + 1):
        active_idxs = [i for i, a in enumerate(active) if a]
        if not active_idxs:
            break
        sp = SamplingParams(**sp_kwargs)
        batch_ids = [
            TokensPrompt(prompt_token_ids=prompt_token_ids[i] + cumulatives_ids[i])
            for i in active_idxs
        ]
        outs = llm.generate(batch_ids, sp, use_tqdm=False)

        search_idxs: List[int] = []
        search_queries: List[str] = []
        for k, src_i in enumerate(active_idxs):
            o = outs[k].outputs[0]
            gen_ids = list(o.token_ids)
            total_tok[src_i] += len(gen_ids)
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            gen_text, gen_ids = _truncate_to_stop(gen_ids, gen_text, tokenizer)

            cumulatives[src_i] += gen_text
            cumulatives_ids[src_i].extend(gen_ids)

            if "</answer>" in gen_text:
                termination[src_i] = "answer_generated"
                answers[src_i] = get_answer(cumulatives[src_i]) or ""
                active[src_i] = False
                continue
            if "</search>" in gen_text:
                qry = get_last_query(gen_text)
                if not qry:
                    termination[src_i] = "malformed_search"
                    active[src_i] = False
                    continue
                n_search[src_i] += 1
                search_idxs.append(src_i)
                search_queries.append(qry)
                continue
            termination[src_i] = "length_no_action"
            active[src_i] = False

        if search_queries:
            docs_list = search_remote_batch(
                search_queries, retrieval_url,
                top_k=search_top_k, max_doc_chars=max_doc_chars,
            )
            for src_i, docs in zip(search_idxs, docs_list):
                info_text = f"\n\n<information>{docs}</information>\n\n"
                cumulatives[src_i] += info_text
                cumulatives_ids[src_i].extend(
                    tokenizer.encode(info_text, add_special_tokens=False))
                if n_search[src_i] >= max_turns:
                    termination[src_i] = "max_turns_exceeded"
                    active[src_i] = False

        if debug:
            print(f"[turn {turn}] active={sum(active)} "
                  f"searches_this_turn={len(search_queries)}", flush=True)

    elapsed = time.time() - t_start
    results: List[Dict] = []
    for i, ex in enumerate(examples):
        em = compute_em(answers[i], ex["golden_answers"])
        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "golden_answers": ex["golden_answers"],
            "pred": answers[i],
            "em": em,
            "n_search": n_search[i],
            "n_latent": len(LATENT_OPEN_RE.findall(cumulatives[i])),
            "n_think": len(THINK_RE.findall(cumulatives[i])),
            "think_chars": sum(len(m.group(1)) for m in THINK_RE.finditer(cumulatives[i])),
            "total_tok": total_tok[i],
            "termination": termination[i],
            "trajectory": cumulatives[i],
        })
    print(f"[eval] {n} examples in {elapsed:.1f}s "
          f"({elapsed/max(n,1):.2f}s/example)", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_ckpt", required=True,
                    help="vLLM-loadable Qwen2 ckpt dir with projector.pt sidecar.")
    ap.add_argument("--dataset", default="bamboogle",
                    help="FlashRAG dataset name (used unless --input_jsonl).")
    ap.add_argument("--input_jsonl", default=None)
    ap.add_argument("--max_examples", type=int, default=10)
    ap.add_argument("--retrieval_url", default="http://127.0.0.1:8000")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_doc_chars", type=int, default=500)
    ap.add_argument("--num_latent", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0.0 = greedy, default).")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--output_jsonl", default=None)
    # Index-shard DDP (no torch.distributed). Launch N processes with
    # --shard_rank=0..N-1 and --shard_world_size=N; each handles
    # examples[rank::world_size].
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world_size", type=int, default=1)
    args = ap.parse_args()
    assert 0 <= args.shard_rank < args.shard_world_size

    os.environ["LATENT_NUM_LATENT"] = str(args.num_latent)

    print(f"[eval] loading {args.merged_ckpt} ...", flush=True)
    # `enforce_eager=True`: the plugin attaches the projector AFTER
    # vLLM's initial trace, so torch.compile would mis-trace the added
    # module.
    llm = LLM(
        model=args.merged_ckpt,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.merged_ckpt, trust_remote_code=True)
    latent_open_ids = list(tokenizer.encode("<latent>", add_special_tokens=False))
    latent_close_ids = list(tokenizer.encode("</latent>", add_special_tokens=False))
    print(f"[eval] <latent>={latent_open_ids} </latent>={latent_close_ids} "
          f"K={args.num_latent}", flush=True)

    if args.input_jsonl:
        all_examples = load_eval_jsonl(args.input_jsonl, args.max_examples)
        print(f"[eval] loaded {len(all_examples)} examples from "
              f"{args.input_jsonl}", flush=True)
    else:
        all_examples = load_eval_dataset(args.dataset, args.max_examples)
    examples = all_examples[args.shard_rank::args.shard_world_size]
    if args.shard_world_size > 1:
        print(f"[eval] shard {args.shard_rank}/{args.shard_world_size}: "
              f"{len(examples)}/{len(all_examples)} examples", flush=True)
    else:
        print(f"[eval] loaded {len(examples)} examples", flush=True)

    results = evaluate_batch(
        llm, tokenizer, examples,
        retrieval_url=args.retrieval_url,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        search_top_k=args.topk,
        max_doc_chars=args.max_doc_chars,
        debug=args.debug,
        temperature=args.temperature,
    )

    n_em = sum(r["em"] for r in results)
    n = len(results)
    em_pct = 100 * n_em / max(n, 1)
    print()
    print("=" * 60)
    print(f"  {args.dataset}  n={n}  EM={em_pct:.1f}%")
    print("=" * 60)
    for r in results[:20]:
        print(f"  EM={int(r['em'])} pred={r['pred']!r} gold={r['golden_answers']} "
              f"n_search={r['n_search']} n_latent={r['n_latent']} "
              f"n_think={r['n_think']} term={r['termination']}")

    if args.output_jsonl:
        with open(args.output_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] wrote per-example results to {args.output_jsonl}")


if __name__ == "__main__":
    main()

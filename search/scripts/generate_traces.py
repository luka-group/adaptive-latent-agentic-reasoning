"""
Generate Search-R1 CoT traces for training data — vLLM-based.

Runs Search-R1 on training questions with retrieval, captures the full
thinking traces for use as distillation/reconstruction targets.

Multi-turn agentic loop with vLLM continuous batching:
  - all active examples generate together each turn (stop on </search> / </answer>)
  - on </search>: parse query, batch-call retriever, append <information>...</information>
  - on </answer>: mark done

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python search/scripts/generate_traces.py \
      --model_id PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3 \
      --input_hf PeterJinGo/nq_hotpotqa_train \
      --output_path data/teacher/searchr1-7b/traces.jsonl \
      --retrieval_url http://127.0.0.1:8000 \
      --tensor_parallel_size 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

# vLLM TP workers must use spawn (fork can't re-init CUDA after the parent has
# imported torch via HF datasets streaming). Set BEFORE importing vllm.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import requests

from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>."
)

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def get_query(text: str) -> Optional[str]:
    m = SEARCH_RE.findall(text)
    return m[-1].strip() if m else None


def get_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def extract_thinking(text: str) -> str:
    """Last <think>...</think> block, or text up to first action tag."""
    m = list(THINK_RE.finditer(text))
    if m:
        return m[-1].group(1).strip()
    m2 = re.search(r"^(.*?)(?:<search>|<answer>)", text, re.DOTALL)
    return m2.group(1).strip() if m2 else text.strip()


def search_remote_batch(queries: List[str], url: str, top_k: int) -> List[Tuple[str, List[Dict]]]:
    """Batched retrieval. Returns (snippets_str, [{docid, title, text}, ...]) per query."""
    base = url.rstrip("/")
    if not base.endswith("/retrieve"):
        base += "/retrieve"
    try:
        r = requests.post(base, json={"queries": queries, "topk": top_k}, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[gen] retrieval failed: {e}", flush=True)
        return [("", []) for _ in queries]
    raw = data.get("results") or data.get("result") or []
    out: List[Tuple[str, List[Dict]]] = []
    for hits in raw:
        if not hits:
            out.append(("", []))
            continue
        blocks = []
        docs = []
        for idx, d in enumerate(hits):
            title = d.get("title", "")
            text = d.get("text", "")
            blocks.append(f"Doc {idx+1}(Title: {title}) {text}")
            docs.append({
                "docid": str(d.get("docid", idx)),
                "title": title,
                "text": text,
            })
        out.append(("\n".join(blocks), docs))
    while len(out) < len(queries):
        out.append(("", []))
    return out


def normalize_answer(s: str) -> str:
    import string
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())


def load_records(args) -> List[Dict]:
    records: List[Dict] = []
    if args.input_hf:
        # Use streaming: some HF datasets have cross-row schema drift that
        # crashes the non-streaming loader (TypeError on Arrow cast). Streaming
        # reads parquet shards row-by-row without forcing a unified schema.
        from datasets import load_dataset
        ds = load_dataset(args.input_hf, split=args.input_hf_split, streaming=True)
        print(f"[gen] HF dataset {args.input_hf} split={args.input_hf_split} (streaming)")
        n_dropped = 0
        for ex in ds:
            q = ex.get("question") or ex.get("query") or ex.get("input")
            a = ex.get("answer")
            if a is None:
                a = ex.get("golden_answers") or ex.get("answers")
            # Stringified list (HF dataset stores as `"['foo', 'bar']"`).
            if isinstance(a, str) and a.startswith("[") and a.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(a)
                    if isinstance(parsed, list):
                        a = parsed
                except Exception:
                    pass
            if isinstance(a, list):
                a = a[0] if a else ""
            if not q or a is None or a == "":
                n_dropped += 1
                continue
            records.append({"question": q, "answer": a})
        print(f"[gen] streamed {len(records)} valid rows (dropped {n_dropped})")
    elif args.input_path:
        with open(args.input_path) as f:
            for line in f:
                r = json.loads(line)
                records.append({"question": r["question"], "answer": r["answer"]})
    else:
        raise SystemExit("Must provide --input_hf or --input_path.")
    if args.max_samples > 0:
        records = records[:args.max_samples]
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Local jsonl with {question, answer}. Mutually exclusive with --input_hf.")
    parser.add_argument("--input_hf", type=str, default=None,
                        help="HF dataset id (e.g. PeterJinGo/nq_hotpotqa_train).")
    parser.add_argument("--input_hf_split", type=str, default="train")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--retrieval_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Slice records[start_idx:end_idx] for sharded multi-process runs.")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="Exclusive end. -1 = end of dataset.")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--search_top_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Run the multi-turn loop on chunks of this size. Smaller "
                             "chunks → faster first retrieval call + more granular "
                             "incremental saves. Larger → fewer vLLM warm-up calls.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions whose pred_answer already exists in --output_path.")
    args = parser.parse_args()

    records = load_records(args)
    print(f"[gen] loaded {len(records)} questions")
    end = args.end_idx if args.end_idx >= 0 else len(records)
    if args.start_idx > 0 or end < len(records):
        records = records[args.start_idx:end]
        print(f"[gen] sharded slice [{args.start_idx}:{end}] -> {len(records)} questions")

    # Resume support: skip already-done questions.
    done_questions = set()
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("question"):
                        done_questions.add(r["question"])
                except Exception:
                    pass
        records = [r for r in records if r["question"] not in done_questions]
        print(f"[gen] resume: {len(done_questions)} already done, {len(records)} remaining")

    if not records:
        print("[gen] nothing to do")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"[gen] loading {args.model_id} (TP={args.tensor_parallel_size}, "
          f"util={args.gpu_memory_utilization}, max_len={args.max_model_len})...")
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    print("[gen] model ready")

    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["</search>", "</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    out_mode = "a" if args.resume and done_questions else "w"
    fout = open(args.output_path, out_mode)

    total_em_sum = 0.0
    total_written = 0
    FLUSH_EVERY = args.chunk_size   # also flushed at end of each chunk

    def process_chunk(chunk_records: List[Dict]) -> Tuple[float, int]:
        """Run the multi-turn loop on one chunk; write each finished example
        immediately. Returns (em_sum, count_written)."""
        cn = len(chunk_records)
        # Build prompts for this chunk.
        prompts: List[str] = []
        for ex in chunk_records:
            q = ex["question"].strip()
            if q and q[-1] != "?":
                q += "?"
            ptext = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{SYSTEM_PROMPT} Question: {q}\n"}],
                add_generation_prompt=True, tokenize=False,
            )
            prompts.append(ptext)

        cumulatives: List[str] = [""] * cn
        trajectories: List[List[Dict]] = [[] for _ in range(cn)]
        answers: List[str] = [""] * cn
        answer_thinkings: List[str] = [""] * cn
        total_tokens: List[int] = [0] * cn
        finish_reasons: List[str] = ["max_turns_exceeded"] * cn
        active: List[bool] = [True] * cn

        chunk_em = 0.0
        chunk_written = 0

        def write_record(i: int) -> float:
            ex = chunk_records[i]
            em = float(normalize_answer(answers[i]) == normalize_answer(ex["answer"])) if answers[i] else 0.0
            rec = {
                "question": ex["question"],
                "answer": ex["answer"],
                "pred_answer": answers[i],
                "trajectory": trajectories[i],
                "answer_thinking": answer_thinkings[i],
                "em": em,
                "generated_tokens": total_tokens[i],
                "finish_reason": finish_reasons[i],
            }
            fout.write(json.dumps(rec) + "\n")
            return em

        for turn in range(args.max_turns + 1):
            active_idxs = [i for i, a in enumerate(active) if a]
            if not active_idxs:
                break

            batch_prompts = [prompts[i] + cumulatives[i] for i in active_idxs]
            outs = llm.generate(batch_prompts, sp, use_tqdm=True)

            search_idxs: List[int] = []
            search_queries: List[str] = []
            finished_this_turn: List[int] = []
            for k, src_i in enumerate(active_idxs):
                o = outs[k].outputs[0]
                gen_ids = list(o.token_ids)
                total_tokens[src_i] += len(gen_ids)
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                thinking = extract_thinking(text)
                ans = get_answer(text)
                qry = get_query(text)

                if ans is not None:
                    trajectories[src_i].append({
                        "thinking": thinking, "query": "", "retrieved_docs": [],
                        "snippets": "", "finish_reason": "stop",
                    })
                    answers[src_i] = ans
                    answer_thinkings[src_i] = thinking
                    finish_reasons[src_i] = "answer_generated"
                    cumulatives[src_i] += text
                    active[src_i] = False
                    finished_this_turn.append(src_i)
                    continue

                if qry is None:
                    if not trajectories[src_i] or trajectories[src_i][-1].get("query") != "":
                        trajectories[src_i].append({
                            "thinking": thinking, "query": "", "retrieved_docs": [],
                            "snippets": "", "finish_reason": "length",
                        })
                    answers[src_i] = ""
                    answer_thinkings[src_i] = thinking
                    finish_reasons[src_i] = "length_no_action"
                    cumulatives[src_i] += text
                    active[src_i] = False
                    finished_this_turn.append(src_i)
                    continue

                search_idxs.append(src_i)
                search_queries.append(qry)
                cumulatives[src_i] += text
                trajectories[src_i].append({
                    "thinking": thinking, "query": qry, "retrieved_docs": None,
                    "snippets": None, "finish_reason": "stop",
                })

            if search_queries:
                results = search_remote_batch(search_queries, args.retrieval_url, top_k=args.search_top_k)
                for src_i, (snippets, docs) in zip(search_idxs, results):
                    trajectories[src_i][-1]["retrieved_docs"] = docs
                    trajectories[src_i][-1]["snippets"] = snippets
                    cumulatives[src_i] += f"\n\n<information>{snippets}</information>\n\n"

            for i in finished_this_turn:
                chunk_em += write_record(i)
                chunk_written += 1
                if (total_written + chunk_written) % FLUSH_EVERY == 0:
                    fout.flush()
                    print(f"[gen] flushed at {total_written + chunk_written} records",
                          flush=True)

            print(f"[gen]   turn {turn}: active_before={len(active_idxs)}  "
                  f"searches={len(search_queries)}  finished={len(finished_this_turn)}  "
                  f"chunk_done={chunk_written}/{cn}", flush=True)

        for i in range(cn):
            if active[i]:
                chunk_em += write_record(i)
                chunk_written += 1
        return chunk_em, chunk_written

    t0 = time.time()
    n_total = len(records)
    n_chunks = (n_total + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        chunk_t0 = time.time()
        chunk = records[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        print(f"\n[gen] === chunk {ci+1}/{n_chunks}  size={len(chunk)} "
              f"(records {ci*args.chunk_size}..{ci*args.chunk_size + len(chunk)})", flush=True)
        em_sum, written = process_chunk(chunk)
        total_em_sum += em_sum
        total_written += written
        fout.flush()
        chunk_dt = time.time() - chunk_t0
        elapsed = time.time() - t0
        print(f"[gen] === chunk {ci+1} done in {chunk_dt:.1f}s  "
              f"chunk_EM={em_sum/max(written,1)*100:.1f}%  "
              f"total_written={total_written}/{n_total}  total_EM={total_em_sum/max(total_written,1)*100:.1f}%  "
              f"elapsed={elapsed:.1f}s  ETA={elapsed/(ci+1)*(n_chunks-ci-1):.1f}s", flush=True)
    fout.close()

    elapsed = time.time() - t0
    print(f"\n[gen] generated {len(records)} traces in {elapsed:.1f}s "
          f"({len(records)/max(elapsed,1):.2f}/s)  EM={em_sum/max(len(records),1)*100:.1f}%")


if __name__ == "__main__":
    main()

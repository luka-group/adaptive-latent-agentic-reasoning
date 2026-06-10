"""
FAISS retrieval server for multi-GPU evaluation.

Loads the E5 encoder + FAISS index + corpus once, serves search requests
via FastAPI. Workers query this server instead of loading their own index.

Usage:
  # Start server on GPU 0
  CUDA_VISIBLE_DEVICES=0 python search/scripts/retrieval/server.py \
      --corpus_path data/wiki-18/wiki-18.jsonl \
      --index_path data/wiki-18/e5_Flat.index \
      --faiss_gpu --port 8000
"""

import argparse
import asyncio
import os
from typing import List, Optional

import faiss
import numpy as np
import torch
import datasets
from transformers import AutoTokenizer, AutoModel

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Encoder:
    def __init__(self, model_name="intfloat/e5-base-v2", device="cuda", use_fp16=True):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        # fp16 is only safe on CUDA; CPU bf16/fp16 matmul is much slower
        # than fp32 on most x86 hardware, so stay fp32 on CPU.
        if use_fp16 and device != "cpu":
            self.model = self.model.half()
        self.model.eval()

    @torch.no_grad()
    def encode(self, queries: List[str], batch_size: int = 512) -> np.ndarray:
        all_embs = []
        for start in range(0, len(queries), batch_size):
            batch = [f"query: {q}" for q in queries[start:start + batch_size]]
            inputs = self.tokenizer(
                batch, max_length=256, padding=True,
                truncation=True, return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            emb = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
            del inputs, outputs
        return np.concatenate(all_embs, axis=0).astype(np.float32, order="C")


class RetrievalServer:
    def __init__(self, corpus_path, index_path, model_name, faiss_gpu=True,
                 encoder_device="cuda"):
        # Load encoder
        print(f"Loading E5 encoder: {model_name} on {encoder_device}...")
        self.encoder = E5Encoder(model_name, device=encoder_device, use_fp16=True)

        # Load corpus
        print(f"Loading corpus from {corpus_path}...")
        self.corpus = datasets.load_dataset(
            "json", data_files=corpus_path, split="train", num_proc=4,
        )
        print(f"Corpus loaded: {len(self.corpus):,} documents")

        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        if faiss_gpu and torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            ngpus = torch.cuda.device_count()
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True  # split index across GPUs (not replicate)
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            print(f"FAISS index sharded across {ngpus} GPU(s) (float16)")
        else:
            if faiss_gpu:
                print("WARNING: faiss-gpu not installed, using CPU with multi-threading")
            faiss.omp_set_num_threads(os.cpu_count())
            print(f"FAISS using {os.cpu_count()} CPU threads")
        print(f"FAISS index ready: {self.index.ntotal:,} vectors")

    def search(self, queries: List[str], topk: int = 3) -> List[List[dict]]:
        query_emb = self.encoder.encode(queries)
        return self._search_by_vectors(query_emb, topk)

    def _search_by_vectors(self, query_emb, topk: int = 3) -> List[List[dict]]:
        """Search FAISS index with precomputed embedding vectors."""
        scores, indices = self.index.search(query_emb, k=topk)

        results = []
        for i in range(len(query_emb)):
            hits = []
            for j in range(topk):
                idx = int(indices[i][j])
                if idx < 0:
                    break
                doc = self.corpus[idx]
                contents = doc.get("contents", doc.get("text", ""))
                # Split title from text (format: "title\ntext")
                parts = contents.split("\n", 1)
                title = parts[0].strip('"') if parts else ""
                text = parts[1] if len(parts) > 1 else contents
                hits.append({
                    "docid": str(doc.get("id", idx)),
                    "title": title,
                    "text": text,
                    "score": float(scores[i][j]),
                })
            results.append(hits)
        return results


# FastAPI app
app = FastAPI()
server: Optional[RetrievalServer] = None


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 3


class BatchedSearcher:
    """Dynamic batching for /retrieve. Collects up to `max_batch` single-query
    requests within `max_wait_ms`, then runs them as one GPU encode+FAISS call.

    Rationale: E5 encoder and FAISS GPU search have near-constant per-call
    overhead, but batched throughput scales well. Incoming RL workload sends
    1-query requests. Batching them cuts GPU kernel launches by max_batch×
    and avoids FastAPI thread-pool saturation.
    """

    def __init__(self, srv: "RetrievalServer", max_batch: int = 64, max_wait_ms: int = 5):
        self.server = srv
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self._queue: Optional[asyncio.Queue] = None
        self._task = None

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._task = loop.create_task(self._run())

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            items = []  # list of (query, topk, future)
            first = await self._queue.get()
            items.append(first)
            deadline = loop.time() + self.max_wait
            while len(items) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    items.append(item)
                except asyncio.TimeoutError:
                    break

            queries = [it[0] for it in items]
            topks = [it[1] for it in items]
            max_topk = max(topks)
            try:
                results = await loop.run_in_executor(
                    None, lambda: self.server.search(queries, topk=max_topk)
                )
                for (q, tk, fut), res in zip(items, results):
                    if not fut.done():
                        fut.set_result(res[:tk])
            except Exception as e:
                for _, _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)

    async def retrieve(self, query: str, topk: int) -> list:
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((query, topk, fut))
        return await fut


batcher: Optional[BatchedSearcher] = None


@app.on_event("startup")
async def _on_startup():
    global batcher
    if server is not None:
        batcher = BatchedSearcher(server, max_batch=64, max_wait_ms=5)
        batcher.start()
        print(f"[batcher] dynamic batching enabled: max_batch={batcher.max_batch}, "
              f"max_wait_ms={int(batcher.max_wait * 1000)}", flush=True)


@app.post("/retrieve")
async def retrieve(request: QueryRequest):
    topk = request.topk or 3
    if batcher is None:
        # Fallback if batcher didn't initialize
        results = server.search(request.queries, topk=topk)
    else:
        # Fan out to batcher (handles multi-query requests as parallel singles)
        tasks = [batcher.retrieve(q, topk) for q in request.queries]
        results = await asyncio.gather(*tasks)
    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok", "index_size": server.index.ntotal if server else 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--encoder_model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--encoder_device", type=str, default="cuda",
                        help="cuda or cpu — set cpu when no GPU available")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    server = RetrievalServer(
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        model_name=args.encoder_model,
        faiss_gpu=args.faiss_gpu,
        encoder_device=args.encoder_device,
    )

    uvicorn.run(app, host=args.host, port=args.port)

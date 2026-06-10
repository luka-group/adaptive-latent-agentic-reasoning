#!/bin/bash
# Start E5 + FAISS retrieval server.
#
# Usage (GPU mode, default):
#   bash search/scripts/retrieval/start.sh
#   SERVER_GPUS=0,1 PORT=8000 bash search/scripts/retrieval/start.sh
#
# Usage (CPU mode — use on small-GPU hosts so all GPUs go to RL):
#   CPU_ONLY=1 bash search/scripts/retrieval/start.sh
#
# Requires faiss (faiss-gpu for GPU mode), fastapi, and uvicorn in the
# active environment. Point CORPUS_PATH / INDEX_PATH at the wiki-18
# corpus + E5 Flat index (see the Search-R1 release for download links).

set -e

REPO=$(cd "$(dirname "$0")/../../.." && pwd)

SERVER_GPUS=${SERVER_GPUS:-0,1}
PORT=${PORT:-8000}
CORPUS_PATH=${CORPUS_PATH:-$REPO/data/wiki-18/wiki-18.jsonl}
INDEX_PATH=${INDEX_PATH:-$REPO/data/wiki-18/e5_Flat.index}
PYTHON=${PYTHON:-python}
CPU_ONLY=${CPU_ONLY:-0}
SERVER_PY=$REPO/search/scripts/retrieval/server.py

if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "Retrieval server already running on port ${PORT}"
    exit 0
fi

if [[ "$CPU_ONLY" == "1" ]]; then
    echo "Starting retrieval server in CPU-only mode, port ${PORT}..."
    CUDA_VISIBLE_DEVICES="" ${PYTHON} ${SERVER_PY} \
        --corpus_path ${CORPUS_PATH} \
        --index_path ${INDEX_PATH} \
        --encoder_device cpu \
        --port ${PORT}
else
    echo "Starting retrieval server on GPUs ${SERVER_GPUS}, port ${PORT}..."
    CUDA_VISIBLE_DEVICES=${SERVER_GPUS} ${PYTHON} ${SERVER_PY} \
        --corpus_path ${CORPUS_PATH} \
        --index_path ${INDEX_PATH} \
        --faiss_gpu \
        --port ${PORT}
fi

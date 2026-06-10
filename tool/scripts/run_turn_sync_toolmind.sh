#!/bin/bash
# Turn-synchronized rollout for ToolMind (Nanbeige/ToolMind).
#
# One rollout pass per subset (ToolMind has no scenario taxonomy).
# `TOOLMIND_SUBSET` selects `graph_syn_datasets` (default) vs `open_datasets`.
#
# Usage:
#   SHARD_IDX=0 PORT=9004 bash tool/scripts/run_turn_sync_toolmind.sh
#
# Knobs:
#   SHARD_IDX            shard index 0..NUM_SHARDS-1 (required)
#   NUM_SHARDS           default 4
#   PORT                 vLLM port for this shard (required)
#   MODEL                default Qwen/Qwen3-4B
#   MAX_CONCURRENT       per-shard concurrency cap (default 128)
#   ROLLOUT_N            rollouts per item (default 1 -- single sample)
#   MAX_TURNS_CAP        cap on episode length (default 10; ToolMind has up
#                        to 19+ asst turns in open_datasets and ~6 avg in
#                        graph_syn -- 10 covers most without truncating)
#   TOOLMIND_SUBSET      "graph_syn_datasets" (default) or "open_datasets"
#   OUT_DIR              output dir (default data/tool/toolmind_rollouts)

set -eo pipefail
cd "$(dirname "$0")/../.."

SHARD_IDX=${SHARD_IDX:?SHARD_IDX env var required}
PORT=${PORT:?PORT env var required}
NUM_SHARDS=${NUM_SHARDS:-4}
MODEL=${MODEL:-Qwen/Qwen3-4B}
MAX_CONCURRENT=${MAX_CONCURRENT:-128}
MAX_TURNS_CAP=${MAX_TURNS_CAP:-10}
ROLLOUT_N=${ROLLOUT_N:-1}
TOOLMIND_SUBSET=${TOOLMIND_SUBSET:-graph_syn_datasets}
OUT_DIR=${OUT_DIR:-data/tool/toolmind_rollouts}

mkdir -p "${OUT_DIR}"

OUT_PATH="${OUT_DIR}/${TOOLMIND_SUBSET}.shard${SHARD_IDX}.jsonl"
echo "==========================================================="
echo "[toolmind_turn_sync] shard=${SHARD_IDX}/${NUM_SHARDS}  subset=${TOOLMIND_SUBSET}"
echo "[toolmind_turn_sync] vllm=http://localhost:${PORT}/v1  out=${OUT_PATH}"
echo "==========================================================="

TOOLMIND_SUBSET="${TOOLMIND_SUBSET}" \
    SHARD_IDX="${SHARD_IDX}" NUM_SHARDS="${NUM_SHARDS}" \
    VLLM_BASE_URL="http://localhost:${PORT}/v1" \
    VLLM_MODEL="${MODEL}" \
    OUT_PATH="${OUT_PATH}" \
    MAX_CONCURRENT="${MAX_CONCURRENT}" \
    MAX_TURNS_CAP="${MAX_TURNS_CAP}" \
    ROLLOUT_N="${ROLLOUT_N}" \
    python -m tool.rollout.run_turn_sync

echo "[toolmind_turn_sync] shard ${SHARD_IDX} DONE"
ls -la "${OUT_DIR}" | grep "shard${SHARD_IDX}"

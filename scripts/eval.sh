#!/bin/bash
# ALAR evaluation — runs the benchmark suite sequentially on one GPU.
#
# Usage:
#   bash scripts/eval.sh --domain {search|tool} --ckpt <merged_ckpt> [options]
#
# Options:
#   --domain        search | tool                 (required)
#   --ckpt          merged, vLLM-loadable ckpt dir with projector.pt sidecar
#                   (e.g. checkpoints/rl_merged/alar_search)
#   --datasets      search: FlashRAG dataset names (default: "nq triviaqa
#                   hotpotqa 2wikimultihopqa musique bamboogle")
#                   tool: BFCL categories (default: "simple multiple parallel
#                   parallel_multiple")
#   --max-examples  cap per dataset (default: -1 = all)
#   --out           results dir (default: results/<ckpt basename>)
#
# Env:
#   CUDA_VISIBLE_DEVICES  GPU to use (default 0)
#   RETRIEVAL_URL         search retrieval server (default http://127.0.0.1:8000)
#   VLLM_GPU_UTIL         vLLM memory fraction (default 0.85)
#   NUM_LATENT            latent steps K (default 4)
#   NO_LATENT=1           disable the latent plugin (stock-model baseline)
#
# The search domain requires the retrieval server
# (search/scripts/retrieval/start.sh) to be running.

set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$REPO:${PYTHONPATH:-}

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; }

DOMAIN="" CKPT="" DATASETS="" MAX_EXAMPLES=-1 OUT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)       DOMAIN=$2;       shift 2;;
        --ckpt)         CKPT=$2;         shift 2;;
        --datasets)     DATASETS=$2;     shift 2;;
        --max-examples) MAX_EXAMPLES=$2; shift 2;;
        --out)          OUT_DIR=$2;      shift 2;;
        -h|--help)      usage; exit 0;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1;;
    esac
done
[[ -n $DOMAIN && -n $CKPT ]] || { usage >&2; exit 1; }
# A non-local --ckpt is allowed: pass a stock HF model id with NO_LATENT=1
# to evaluate a no-latent baseline.
[[ -d $CKPT ]] || echo "NOTE: $CKPT is not a local directory — treating it as a HF model id" >&2

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT_DIR=${OUT_DIR:-$REPO/results/$(basename "$CKPT")}
mkdir -p "$OUT_DIR"

if [[ -n ${NO_LATENT:-} ]]; then
    export LATENT_VLLM=0
fi

echo "============================================================"
echo "  ALAR eval ($DOMAIN)"
echo "  Ckpt:    $CKPT"
echo "  Output:  $OUT_DIR"
echo "  GPU:     $CUDA_VISIBLE_DEVICES"
echo "  Latent:  ${NO_LATENT:+OFF}${NO_LATENT:-ON}"
echo "============================================================"

if [[ $DOMAIN == search ]]; then
    DATASETS=${DATASETS:-"nq triviaqa hotpotqa 2wikimultihopqa musique bamboogle"}
    SUMMARY=$OUT_DIR/summary.txt
    : > "$SUMMARY"

    for ds in $DATASETS; do
        echo; echo ">>> $ds"
        LATENT_VLLM=${LATENT_VLLM:-1} \
        LATENT_NUM_LATENT=${NUM_LATENT:-4} \
        python -m search.evaluate_vllm \
            --merged_ckpt "$CKPT" \
            --dataset "$ds" \
            --max_examples $MAX_EXAMPLES \
            --retrieval_url ${RETRIEVAL_URL:-http://127.0.0.1:8000} \
            --max_turns ${MAX_TURNS:-6} \
            --max_new_tokens ${MAX_NEW_TOKENS:-256} \
            --topk ${TOPK:-3} \
            --max_doc_chars ${MAX_DOC_CHARS:-500} \
            --num_latent ${NUM_LATENT:-4} \
            --gpu_memory_utilization ${VLLM_GPU_UTIL:-0.85} \
            --max_model_len ${VLLM_MAX_LEN:-8192} \
            --output_jsonl "$OUT_DIR/$ds.jsonl" \
            2>&1 | tee "$OUT_DIR/$ds.log"
        line=$(grep -E "n=.*EM=" "$OUT_DIR/$ds.log" | tail -1)
        printf "  %-22s %s\n" "$ds" "$line" | tee -a "$SUMMARY"
    done

    echo; echo "============================================================"
    echo "  SUMMARY"
    echo "============================================================"
    cat "$SUMMARY"
    echo "  per-dataset details: $OUT_DIR/"
else
    DATASETS=${DATASETS:-"simple multiple parallel parallel_multiple"}
    python -m tool.evaluate_vllm \
        --merged_ckpt "$CKPT" \
        --categories $DATASETS \
        --max_examples $MAX_EXAMPLES \
        --max_new_tokens ${MAX_NEW_TOKENS:-2048} \
        --max_model_len ${VLLM_MAX_LEN:-8192} \
        --num_latent ${NUM_LATENT:-4} \
        --gpu_memory_utilization ${VLLM_GPU_UTIL:-0.85} \
        --output_dir "$OUT_DIR" \
        --system_prompt ${EVAL_SYSTEM_PROMPT:-tool} \
        ${NO_LATENT:+--no_latent} \
        2>&1 | tee "$OUT_DIR/bfcl.log"
fi

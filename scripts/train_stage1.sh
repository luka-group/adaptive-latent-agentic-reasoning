#!/bin/bash
# ALAR Stage 1 — Action-Anchored Self-Distillation (AASD).
#
# Trains the latent reasoning mode: every teacher CoT span is replaced by a
# <latent>••••</latent> block (latent_prob=1.0) and the student (LoRA +
# projector) learns to reproduce the teacher's anchor actions.
#
# Usage:
#   bash scripts/train_stage1.sh --domain {search|tool} [options]
#
# Options:
#   --domain    search | tool                   (required)
#   --model     base model HF id or local path  (default: per-domain paper model)
#   --data      teacher-trace JSONL             (default: data/<domain>/teacher_traces.jsonl)
#   --run-name  prefix for checkpoints/* dirs   (default: alar_<domain>)
#
# Output:
#   checkpoints/sft/<run-name>_stage1/final/    PEFT adapter (LoRA + projector)
#
# Hyperparameters default to the paper settings; override via env vars:
#   NUM_LATENT (4)  LORA_R (16)  LORA_ALPHA (32)  LR (1e-4)  GRAD_ACCUM (16)
#   EPOCHS (1)  MAX_SAMPLES (-1)  MIN_EM (1.0, search)  MIN_SCORE (1.5, tool)
#   CUDA_VISIBLE_DEVICES (0,1,2,3)  GRADIENT_CHECKPOINTING (1)  ...

set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$REPO:${PYTHONPATH:-}

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }

DOMAIN="" MODEL="" DATA="" RUN_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)   DOMAIN=$2;   shift 2;;
        --model)    MODEL=$2;    shift 2;;
        --data)     DATA=$2;     shift 2;;
        --run-name) RUN_NAME=$2; shift 2;;
        -h|--help)  usage; exit 0;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1;;
    esac
done

case $DOMAIN in
    search)
        MODEL=${MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3}
        DATA=${DATA:-$REPO/data/search/teacher_traces.jsonl};;
    tool)
        MODEL=${MODEL:-Qwen/Qwen3-4B-Thinking-2507}
        DATA=${DATA:-$REPO/data/tool/sft_traces.jsonl};;
    *)  echo "--domain must be 'search' or 'tool' (see --help)" >&2; exit 1;;
esac
RUN_NAME=${RUN_NAME:-alar_$DOMAIN}
OUT_DIR=$REPO/checkpoints/sft/${RUN_NAME}_stage1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING-1}
PYTHON=${PYTHON:-python}

if [[ $N_GPUS -le 1 ]]; then
    LAUNCHER="$PYTHON"
else
    LAUNCHER="torchrun --standalone --nproc_per_node=$N_GPUS"
fi

echo "============================================================"
echo "  ALAR Stage 1 — AASD ($DOMAIN)"
echo "  Base model:   $MODEL"
echo "  Teacher data: $DATA"
echo "  Output:       $OUT_DIR"
echo "  K (latent):   ${NUM_LATENT:-4}"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES ($N_GPUS)"
echo "============================================================"

$LAUNCHER -m alar.sft.aasd \
    --domain $DOMAIN \
    --model_name "$MODEL" \
    --data_path "$DATA" \
    --output_dir "$OUT_DIR" \
    --latent_prob 1.0 \
    --num_latent ${NUM_LATENT:-4} \
    --prj_dim ${PRJ_DIM:-0} \
    --lora_r ${LORA_R:-16} \
    --lora_alpha ${LORA_ALPHA:-32} \
    --lora_dropout ${LORA_DROPOUT:-0.0} \
    --min_em ${MIN_EM:-1.0} \
    --max_doc_chars ${MAX_DOC_CHARS:-500} \
    --min_score ${MIN_SCORE:-1.5} \
    --max_samples ${MAX_SAMPLES:--1} \
    --data_offset ${DATA_OFFSET:-0} \
    --max_length ${MAX_LENGTH:-0} \
    --dataset_seed ${DATASET_SEED:-42} \
    --epochs ${EPOCHS:-1} \
    --max_steps ${MAX_STEPS:--1} \
    --gradient_accumulation_steps ${GRAD_ACCUM:-16} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH:-1} \
    --lr ${LR:-1e-4} \
    --lr_schedule ${LR_SCHEDULE:-cosine} \
    --warmup_ratio ${WARMUP_RATIO:-0.03} \
    --max_grad_norm ${MAX_GRAD_NORM:-1.0} \
    --save_steps ${SAVE_STEPS:-500} \
    --seed ${SEED:-42} \
    ${GRADIENT_CHECKPOINTING:+--gradient_checkpointing} \
    ${RESUME_FROM:+--resume_from_checkpoint $RESUME_FROM}

echo
echo "[stage1] done. Adapter at $OUT_DIR/final"
echo "[stage1] next: bash scripts/train_stage2.sh --domain $DOMAIN --run-name $RUN_NAME"

#!/bin/bash
# ALAR Stage 2 — adaptive mode selection: mode-warmup SFT + AR-GRPO.
#
# Runs, in order:
#   1. mode-warmup SFT     per-turn latent/think coin flip, init from Stage 1
#   2. adapter merge       -> vLLM-loadable checkpoint (RL init)
#   3. AR-GRPO             verl GRPO with the latent-fraction shaped reward
#   4. final adapter merge -> vLLM-loadable eval checkpoint
#
# Usage:
#   bash scripts/train_stage2.sh --domain {search|tool} [options]
#
# Options:
#   --domain    search | tool                   (required)
#   --model     base model HF id or local path  (default: per-domain paper model)
#   --data      teacher-trace JSONL             (default: data/<domain>/teacher_traces.jsonl)
#   --run-name  prefix for checkpoints/* dirs   (default: alar_<domain>; must match Stage 1)
#   --rl-steps  total AR-GRPO training steps    (default: 100)
#   --rl-only   skip warmup+merge; start RL from the existing merged ckpt
#   --ckpt      merged ckpt to start RL from    (default: checkpoints/sft_merged/<run-name>_warmup)
#
# Outputs:
#   checkpoints/sft/<run-name>_warmup/         mode-warmup adapter
#   checkpoints/sft_merged/<run-name>_warmup/  merged warmup ckpt (RL init)
#   checkpoints/rl/<run-name>/                 raw AR-GRPO checkpoints
#   checkpoints/rl_merged/<run-name>/          merged final model (eval)
#
# Hyperparameters default to the paper settings; override via env vars:
#   WARMUP_SAMPLES (20000)  AR_GRPO_ALPHA (0.3)  AR_GRPO_L (400 search / 1600 tool)
#   ROLLOUT_N (8)  LORA_RANK (16; 0 = full fine-tune)  RL_LR (1e-6)
#   PPO_MINI_BATCH / PPO_MICRO_BATCH / TRAIN_BATCH_SIZE  CUDA_VISIBLE_DEVICES  ...
#
# Prerequisite for the search domain: the retrieval server
# (search/scripts/retrieval/start.sh) reachable at LATENT_RETRIEVAL_URL
# (default http://127.0.0.1:8000).

set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$REPO:${PYTHONPATH:-}

usage() { sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; }

DOMAIN="" MODEL="" DATA="" RUN_NAME="" RL_CKPT="" RL_STEPS=100 RL_ONLY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)   DOMAIN=$2;   shift 2;;
        --model)    MODEL=$2;    shift 2;;
        --data)     DATA=$2;     shift 2;;
        --run-name) RUN_NAME=$2; shift 2;;
        --rl-steps) RL_STEPS=$2; shift 2;;
        --rl-only)  RL_ONLY=1;   shift;;
        --ckpt)     RL_CKPT=$2;  shift 2;;
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

SFT_DIR=$REPO/checkpoints/sft
MERGED_DIR=$REPO/checkpoints/sft_merged
RL_DIR=$REPO/checkpoints/rl
RL_MERGED_DIR=$REPO/checkpoints/rl_merged
PYTHON=${PYTHON:-python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING-1}

# ── 1+2. Mode-warmup SFT + adapter merge ─────────────────────────────────────
if [[ -z $RL_ONLY ]]; then
    STAGE1_ADAPTER=${STAGE1_ADAPTER:-$SFT_DIR/${RUN_NAME}_stage1/final}
    [[ -d $STAGE1_ADAPTER ]] || {
        echo "ERROR: Stage 1 adapter not found at $STAGE1_ADAPTER" >&2
        echo "       (run scripts/train_stage1.sh first, or set STAGE1_ADAPTER)" >&2; exit 1; }

    if [[ $N_GPUS -le 1 ]]; then
        LAUNCHER="$PYTHON"
    else
        LAUNCHER="torchrun --standalone --nproc_per_node=$N_GPUS"
    fi

    echo "============================================================"
    echo "  ALAR Stage 2a — mode warmup ($DOMAIN)"
    echo "  Init from:    $STAGE1_ADAPTER"
    echo "  Output:       $SFT_DIR/${RUN_NAME}_warmup"
    echo "  GPUs:         $CUDA_VISIBLE_DEVICES ($N_GPUS)"
    echo "============================================================"

    $LAUNCHER -m alar.sft.aasd \
        --domain $DOMAIN \
        --model_name "$MODEL" \
        --data_path "$DATA" \
        --output_dir "$SFT_DIR/${RUN_NAME}_warmup" \
        --latent_prob ${WARMUP_LATENT_PROB:-0.5} \
        --init_lora_from "$STAGE1_ADAPTER" \
        --num_latent ${NUM_LATENT:-4} \
        --prj_dim ${PRJ_DIM:-0} \
        --min_em ${MIN_EM:-1.0} \
        --max_doc_chars ${MAX_DOC_CHARS:-500} \
        --min_score ${MIN_SCORE:-1.5} \
        --max_samples ${WARMUP_SAMPLES:-20000} \
        --data_offset ${DATA_OFFSET:-0} \
        --max_length ${MAX_LENGTH:-0} \
        --dataset_seed ${DATASET_SEED:-42} \
        --epochs ${EPOCHS:-1} \
        --gradient_accumulation_steps ${GRAD_ACCUM:-16} \
        --per_device_train_batch_size ${PER_DEVICE_BATCH:-1} \
        --lr ${LR:-1e-4} \
        --lr_schedule ${LR_SCHEDULE:-cosine} \
        --warmup_ratio ${WARMUP_RATIO_SFT:-0.03} \
        --max_grad_norm ${MAX_GRAD_NORM:-1.0} \
        --save_steps ${SAVE_STEPS:-100} \
        --seed ${SEED:-42} \
        ${GRADIENT_CHECKPOINTING:+--gradient_checkpointing}

    echo; echo ">>> Merging warmup adapter  →  $MERGED_DIR/${RUN_NAME}_warmup"
    $PYTHON -m alar.scripts.merge_sft \
        --sft_ckpt "$SFT_DIR/${RUN_NAME}_warmup/final" \
        --base_model "$MODEL" \
        --output_dir "$MERGED_DIR/${RUN_NAME}_warmup"
fi

# ── 3. AR-GRPO ───────────────────────────────────────────────────────────────
SFT_CKPT=${RL_CKPT:-$MERGED_DIR/${RUN_NAME}_warmup}
[[ -d $SFT_CKPT ]] || {
    echo "ERROR: merged warmup ckpt not found at $SFT_CKPT (or pass --ckpt)" >&2; exit 1; }

export VERL_CONFIG_DIR=${VERL_CONFIG_DIR:-$REPO/verl/verl/trainer/config}
export PYTHONPATH=$REPO/verl:$PYTHONPATH

# Hidden size — used by the cached_z postprocess patch to zero-fill latent_z
# for rollout shards with no latent samples (keeps DataProto.concat happy).
export LATENT_HIDDEN_SIZE=${LATENT_HIDDEN_SIZE:-$($PYTHON -c "import json; print(json.load(open('$SFT_CKPT/config.json'))['hidden_size'])")}

# Domain-specific RL setup.
if [[ $DOMAIN == search ]]; then
    export SEARCH_RL_DIR=$REPO/search/rl
    ENTRY=search.rl.main_ar_grpo
    export LATENT_AGENT_LOOP_MODULE=search.rl.agent_loop

    export LATENT_RETRIEVAL_URL=${LATENT_RETRIEVAL_URL:-http://127.0.0.1:8000}
    export LATENT_MAX_TURNS=${LATENT_MAX_TURNS:-6}
    export LATENT_TOPK=${LATENT_TOPK:-3}
    export LATENT_MAX_DOC_CHARS=${LATENT_MAX_DOC_CHARS:-500}
    curl -sf "${LATENT_RETRIEVAL_URL%/}/health" > /dev/null 2>&1 || \
        echo "WARNING: retrieval server not reachable at $LATENT_RETRIEVAL_URL" >&2

    export AR_GRPO_L=${AR_GRPO_L:-400}
    DATA_DIR=${DATA_DIR:-$REPO/data/search/rl}
    if [[ ! -f $DATA_DIR/train.parquet ]]; then
        echo ">>> Preparing search RL parquet  →  $DATA_DIR"
        $PYTHON $REPO/search/scripts/prepare_rl_data.py --local_dir "$DATA_DIR"
    fi

    MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
    MAX_RESP_LEN=${MAX_RESP_LEN:-2500}
    VLLM_MAX_LEN=${VLLM_MAX_LEN:-3072}
    # Batch/offload presets by model scale.
    if echo "$SFT_CKPT $MODEL" | grep -qi "7b"; then
        PPO_MINI_BATCH=${PPO_MINI_BATCH:-16};  PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-4}
        LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-4}; TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
        PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}; OPT_OFFLOAD=${OPT_OFFLOAD:-True}
        REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
    else
        PPO_MINI_BATCH=${PPO_MINI_BATCH:-32};  PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-8}
        LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-8}; TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
        PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}; OPT_OFFLOAD=${OPT_OFFLOAD:-False}
        REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}
    fi
else
    export TOOL_RL_DIR=$REPO/tool/rl
    ENTRY=tool.rl.main_ar_grpo
    export LATENT_AGENT_LOOP_MODULE=tool.rl.agent_loop

    # Qwen3-Thinking teacher traces have much longer <think> blocks than the
    # search domain, so the length tolerance is wider (1600 vs 400).
    export AR_GRPO_L=${AR_GRPO_L:-1600}
    DATA_DIR=${DATA_DIR:-$REPO/data/tool/rl}
    if [[ ! -f $DATA_DIR/train.parquet ]]; then
        echo ">>> Preparing tool RL parquet  →  $DATA_DIR"
        $PYTHON -m tool.scripts.prepare_rl_data \
            --trace_path "$DATA" --local_dir "$DATA_DIR" --tokenizer "$MODEL"
    fi

    MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
    MAX_RESP_LEN=${MAX_RESP_LEN:-4096}
    VLLM_MAX_LEN=${VLLM_MAX_LEN:-8192}
    PPO_MINI_BATCH=${PPO_MINI_BATCH:-16};  PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-4}
    LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-4}; TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
    PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}; OPT_OFFLOAD=${OPT_OFFLOAD:-True}
    REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
fi

# AR-GRPO reward knobs (see alar/rl/ar_grpo.py):
#   f       = n_latent / (n_latent + n_think)
#   r_fmt   = (1 + α·f) if em else -α·f
#   r_div   = d_t · |f - f̄_G|       (cosine-decayed diversity bonus)
#   shaped  = s_L · (r_fmt + r_div)  (-1.0 hard if format_ok=False)
export AR_GRPO_ALPHA=${AR_GRPO_ALPHA:-0.3}
export AR_GRPO_LEN_ON=${AR_GRPO_LEN_ON:-1}
export AR_GRPO_DIV_ON=${AR_GRPO_DIV_ON:-1}
export AR_GRPO_TOTAL=${AR_GRPO_TOTAL:-$RL_STEPS}

# Latent-stack env.
export LATENT_VLLM=1
export LATENT_NUM_LATENT=${LATENT_NUM_LATENT:-4}
export LATENT_SYNC_PATH=${LATENT_SYNC_PATH:-/tmp/ar_grpo_sync_${RUN_NAME}.pt}
# Sever the K-iter autograd chain: the projector is frozen during RL, so
# running the expansion under no_grad drops its activations and backward
# all-gathers. NEVER set during SFT (it trains the projector).
export LATENT_DETACH_KITER=${LATENT_DETACH_KITER:-1}
export LATENT_LP_PROGRESS=${LATENT_LP_PROGRESS:-1}
# Reuse rollout-time z values in actor log-prob/update passes (skips the
# K-iter recompute; valid under single-PPO-epoch bypass mode).
export LATENT_CACHED_Z=${LATENT_CACHED_Z:-1}
rm -f "$LATENT_SYNC_PATH" "${LATENT_SYNC_PATH}.tmp"

# CUDA / NCCL env. The flight recorder dumps stacks on hangs instead of the
# default silent 30-min timeout.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
unset PYTORCH_CUDA_ALLOC_CONF || true
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ACCELERATE_USE_DEEPSPEED=false
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=600

echo "============================================================"
echo "  ALAR Stage 2b — AR-GRPO ($DOMAIN)"
echo "  RL init:      $SFT_CKPT"
echo "  Steps:        $RL_STEPS"
echo "  Reward:       α=$AR_GRPO_ALPHA L=$AR_GRPO_L div_on=$AR_GRPO_DIV_ON"
echo "  Data:         $DATA_DIR/{train,test}.parquet"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES ($N_GPUS)"
echo "  Output:       $RL_DIR/$RUN_NAME"
echo "============================================================"

$PYTHON -c "import $LATENT_AGENT_LOOP_MODULE, $ENTRY, alar.modeling" || {
    echo "ERROR: cannot import $LATENT_AGENT_LOOP_MODULE / alar.modeling" >&2; exit 1; }

# LoRA: LORA_RANK > 0 (default 16) trains a fresh LoRA adapter on top of the
# merged warmup base (which already carries the projector sidecar).
# LORA_RANK=0 switches to full fine-tuning.
LORA_ARGS=()
LORA_RANK=${LORA_RANK:-16}
if [[ $LORA_RANK -gt 0 ]]; then
    LORA_ARGS=(
        "actor_rollout_ref.model.lora_rank=$LORA_RANK"
        "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-32}"
        "actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
    )
fi

PYTHONUNBUFFERED=1 $PYTHON -m $ENTRY \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    actor_rollout_ref.model.path=$SFT_CKPT \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRAD_CKPT:-True} \
    "${LORA_ARGS[@]}" \
    actor_rollout_ref.actor.optim.lr=${RL_LR:-1e-6} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${WARMUP_RATIO:-0.1} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS:-True} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF:-0.001} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPT_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE:--1} \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESP_LEN \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_UTIL:-0.5} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=$VLLM_MAX_LEN \
    actor_rollout_ref.rollout.multi_stage_wake_up=${MULTI_STAGE_WAKE_UP:-True} \
    actor_rollout_ref.rollout.load_format=${VLLM_LOAD_FORMAT:-auto} \
    actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON:-True} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    ++algorithm.rollout_correction.bypass_mode=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${FSDP_SIZE:--1} \
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD:-False} \
    trainer.logger=${TRAINER_LOGGER:-'["console","wandb"]'} \
    trainer.project_name=${WANDB_PROJECT:-alar} \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-25} \
    trainer.test_freq=${TEST_FREQ:-25} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$RL_STEPS \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.default_local_dir=$RL_DIR/$RUN_NAME

# ── 4. Merge the final RL adapter into a vLLM-loadable eval checkpoint ───────
if [[ $LORA_RANK -gt 0 ]]; then
    ADAPTER=$RL_DIR/$RUN_NAME/global_step_${RL_STEPS}/actor/lora_adapter
    if [[ -d $ADAPTER ]]; then
        echo; echo ">>> Merging RL adapter  →  $RL_MERGED_DIR/$RUN_NAME"
        $PYTHON -m alar.scripts.merge_rl \
            --base_dir "$SFT_CKPT" \
            --adapter_dir "$ADAPTER" \
            --output_dir "$RL_MERGED_DIR/$RUN_NAME" \
            --device ${MERGE_DEVICE:-cpu}
    else
        echo "WARNING: no adapter at $ADAPTER — merge manually with alar.scripts.merge_rl" >&2
    fi
else
    echo "Full-FT run: merge the FSDP shards with alar.scripts.merge_fsdp_rl:"
    echo "  python -m alar.scripts.merge_fsdp_rl --ckpt_dir $RL_DIR/$RUN_NAME/global_step_${RL_STEPS}/actor \\"
    echo "      --base_model $MODEL --output_dir $RL_MERGED_DIR/$RUN_NAME"
fi

echo; echo "[stage2] done. Eval checkpoint: $RL_MERGED_DIR/$RUN_NAME"

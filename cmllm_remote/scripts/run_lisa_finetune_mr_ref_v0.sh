#!/usr/bin/env bash
set -euo pipefail

cd /home/wjq/cmllm/third_party/LISA

BASE_VERSION="${BASE_VERSION:-/home/wjq/cmllm/outputs/lisa_finetune/cmllm_lisa_plus_dsdpm66_mining_helmet_fullinst_deepseek_keyed_v1_5epoch_20260510_142749/merged_hf}"
MR_DATA_ROOT="${MR_DATA_ROOT:-/home/wjq/cmllm/outputs/mr_data/helmet_miner_pseudopair_v1/final_accepted_v1}"
TRAIN_JSONL="${TRAIN_JSONL:-$MR_DATA_ROOT/episodes_miner_to_helmet_clean_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-$MR_DATA_ROOT/episodes_miner_to_helmet_clean_val.jsonl}"
EXP="${EXP_NAME:-cmllm_lisa_plus_mr_ref_v0_smoke_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/home/wjq/cmllm/outputs/lisa_finetune}"
LOG="${LOG:-/home/wjq/cmllm/outputs/lisa_finetune_mr_ref_v0.log}"
ENV_OUT="${ENV_OUT:-/home/wjq/cmllm/outputs/lisa_finetune_mr_ref_v0_current_run.env}"
EPOCHS="${EPOCHS:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-20}"
LR="${LR:-0.00003}"
DS_GPU_INDEX="${DS_GPU_INDEX:-3}"
NO_EVAL="${NO_EVAL:-1}"
SKIP_MERGE="${SKIP_MERGE:-0}"

mkdir -p "$OUT_ROOT" /home/wjq/cmllm/outputs

TRAIN_COUNT="$(wc -l < "$TRAIN_JSONL")"
VAL_COUNT="$(wc -l < "$VAL_JSONL")"

cat > "$ENV_OUT" <<EOF
EXP_NAME=$EXP
BASE_VERSION=$BASE_VERSION
MR_DATA_ROOT=$MR_DATA_ROOT
TRAIN_JSONL=$TRAIN_JSONL
VAL_JSONL=$VAL_JSONL
TRAIN_COUNT=$TRAIN_COUNT
VAL_COUNT=$VAL_COUNT
EPOCHS=$EPOCHS
STEPS_PER_EPOCH=$STEPS_PER_EPOCH
LR=$LR
DS_GPU_INDEX=$DS_GPU_INDEX
NO_EVAL=$NO_EVAL
SKIP_MERGE=$SKIP_MERGE
LOG=$LOG
OUT_DIR=$OUT_ROOT/$EXP
STARTED_AT=$(date -Is)
EOF

export PYTHONPATH=/home/wjq/cmllm/third_party/LISA
unset CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=9.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS=4

NO_EVAL_ARG=()
if [[ "$NO_EVAL" == "1" ]]; then
  NO_EVAL_ARG=(--no_eval)
fi

/home/wjq/venvs/rex/bin/deepspeed --include "localhost:${DS_GPU_INDEX}" --master_port="${MASTER_PORT:-29580}" train_ds.py \
  --version="$BASE_VERSION" \
  --dataset_dir=/home/wjq/cmllm/dataset_lisa \
  --vision_pretrained=/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth \
  --dataset=mr_ref_seg \
  --sample_rates=1 \
  --mr_ref_data "$TRAIN_JSONL" \
  --val_dataset "MRRefSeg|$VAL_JSONL" \
  --exp_name="$EXP" \
  --log_base_dir="$OUT_ROOT" \
  --epochs="$EPOCHS" \
  --steps_per_epoch="$STEPS_PER_EPOCH" \
  --batch_size="${BATCH_SIZE:-1}" \
  --grad_accumulation_steps="${GRAD_ACCUMULATION_STEPS:-1}" \
  --val_batch_size=1 \
  --workers="${WORKERS:-1}" \
  --print_freq="${PRINT_FREQ:-1}" \
  --lr="$LR" \
  --explanatory=-1 \
  "${NO_EVAL_ARG[@]}" 2>&1 | tee "$LOG"

OUT_DIR="$OUT_ROOT/$EXP"
LATEST_STEP="$(cat "$OUT_DIR/ckpt_model/latest")"
WEIGHT="$OUT_DIR/ckpt_model/$LATEST_STEP/mp_rank_00_model_states.pt"
MERGED="$OUT_DIR/merged_hf"

if [[ "$SKIP_MERGE" != "1" ]]; then
  /home/wjq/venvs/rex/bin/python merge_lora_weights_and_save_hf_model.py \
    --version="$BASE_VERSION" \
    --weight="$WEIGHT" \
    --save_path="$MERGED" \
    --vision_pretrained=/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth \
    --precision=bf16 2>&1 | tee -a "$LOG"
else
  MERGED="SKIPPED"
fi

cat > "$OUT_DIR/report.md" <<EOF
# MR-LISA++ v0 Fine-Tune Report

- Started at: see \`$ENV_OUT\`
- Base model: \`$BASE_VERSION\`
- MR data root: \`$MR_DATA_ROOT\`
- Train / val jsonl: \`$TRAIN_JSONL\` / \`$VAL_JSONL\`
- Train / val count: $TRAIN_COUNT / $VAL_COUNT
- Epochs: $EPOCHS
- Steps per epoch: $STEPS_PER_EPOCH
- LR: $LR
- DeepSpeed GPU index: $DS_GPU_INDEX
- No eval: $NO_EVAL
- Skip merge: $SKIP_MERGE
- DeepSpeed checkpoint: \`$OUT_DIR/ckpt_model/$LATEST_STEP\`
- Merged HF model: \`$MERGED\`
- Training log: \`$LOG\`
EOF

cat >> "$ENV_OUT" <<EOF
FINISHED_AT=$(date -Is)
OUT_DIR=$OUT_DIR
MERGED=$MERGED
LATEST_STEP=$LATEST_STEP
EOF

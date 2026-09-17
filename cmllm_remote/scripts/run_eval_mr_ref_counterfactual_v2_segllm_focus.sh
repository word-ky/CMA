#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?MODEL must point to a merged HF model}"
MR_DATA_ROOT="${MR_DATA_ROOT:-/home/wjq/cmllm/outputs/mr_data/helmet_miner_pseudopair_v1/final_accepted_v1}"
CF_TRAIN_ROOT="${CF_TRAIN_ROOT:-$MR_DATA_ROOT/counterfactual_train_v1}"
COUNTERFACTUAL_JSONL="${COUNTERFACTUAL_JSONL:-$CF_TRAIN_ROOT/episodes_counterfactual_ref_holdout_eval.jsonl}"
PAIRS_JSONL="${PAIRS_JSONL:-$MR_DATA_ROOT/helmet_miner_pairs_accept_high.jsonl}"
OUT_DIR="${OUT_DIR:-/home/wjq/cmllm/outputs/eval_mr_ref_counterfactual_v2_segllm_focus_$(date +%Y%m%d_%H%M%S)}"
GPU_INDEX="${GPU_INDEX:-1}"
MAX_ITEMS="${MAX_ITEMS:-300}"
OVERLAY_LIMIT="${OVERLAY_LIMIT:-64}"
CONVERSATION_MODE="${CONVERSATION_MODE:-v1_multiround}"
REF_IMAGE_MODE="${REF_IMAGE_MODE:-${MR_REF_IMAGE_MODE:-crop}}"
FOCUS_DILATE="${FOCUS_DILATE:-${MR_REF_FOCUS_DILATE:-15}}"
FOCUS_BACKGROUND="${FOCUS_BACKGROUND:-${MR_REF_FOCUS_BACKGROUND:-0.0}}"

cd /home/wjq/cmllm
export PYTHONPATH=/home/wjq/cmllm/third_party/LISA
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export TOKENIZERS_PARALLELISM=false
export MR_REF_IMAGE_MODE="$REF_IMAGE_MODE"
export MR_REF_FOCUS_DILATE="$FOCUS_DILATE"
export MR_REF_FOCUS_BACKGROUND="$FOCUS_BACKGROUND"

/home/wjq/venvs/rex/bin/python /home/wjq/cmllm/scripts/eval_mr_ref_counterfactual_v0.py \
  --model "$MODEL" \
  --counterfactual-jsonl "$COUNTERFACTUAL_JSONL" \
  --pairs-jsonl "$PAIRS_JSONL" \
  --out-dir "$OUT_DIR" \
  --max-items "$MAX_ITEMS" \
  --overlay-limit "$OVERLAY_LIMIT" \
  --conversation-mode "$CONVERSATION_MODE" \
  --ref-image-mode "$REF_IMAGE_MODE" \
  --focus-dilate "$FOCUS_DILATE" \
  --focus-background "$FOCUS_BACKGROUND"

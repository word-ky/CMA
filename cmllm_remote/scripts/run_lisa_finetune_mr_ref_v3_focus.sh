#!/usr/bin/env bash
set -euo pipefail

# v3-focus keeps the v2B rank objective but changes the [REF] visual input from
# a cropped masked miner to a full-image SegLLM-style focus view.
export BASE_VERSION="${BASE_VERSION:-/home/wjq/cmllm/outputs/lisa_finetune/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_20260515/merged_hf}"
export EXP_NAME="${EXP_NAME:-cmllm_lisa_plus_mr_ref_v3_focus_blackout_$(date +%Y%m%d_%H%M%S)}"
export LOG="${LOG:-/home/wjq/cmllm/outputs/lisa_finetune_mr_ref_v3_focus.log}"
export ENV_OUT="${ENV_OUT:-/home/wjq/cmllm/outputs/lisa_finetune_mr_ref_v3_focus_current_run.env}"

export REF_IMAGE_MODE="${REF_IMAGE_MODE:-full_blackout}"
export FOCUS_DILATE="${FOCUS_DILATE:-25}"
export FOCUS_BACKGROUND="${FOCUS_BACKGROUND:-0.0}"

export REF_PROMPT_MODE="${REF_PROMPT_MODE:-add}"
export REF_INJECTION_MODE="${REF_INJECTION_MODE:-gated_add}"
export REF_INPUT_SCALE="${REF_INPUT_SCALE:-0.5}"
export REF_CONTEXT_INIT_SCALE="${REF_CONTEXT_INIT_SCALE:-0.2}"
export REF_PROMPT_MAX_NORM="${REF_PROMPT_MAX_NORM:-20.0}"
export SAM_PROMPT_EMBED_CLAMP="${SAM_PROMPT_EMBED_CLAMP:-50.0}"
export REF_RECONSTRUCTION_LOSS_WEIGHT="${REF_RECONSTRUCTION_LOSS_WEIGHT:-1.0}"

export COUNTERFACTUAL_RANK_LOSS_WEIGHT="${COUNTERFACTUAL_RANK_LOSS_WEIGHT:-0.5}"
export COUNTERFACTUAL_RANK_MARGIN="${COUNTERFACTUAL_RANK_MARGIN:-0.05}"

export LR="${LR:-0.000002}"
export EPOCHS="${EPOCHS:-1}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-3000}"
export NO_EVAL="${NO_EVAL:-1}"
export SKIP_MERGE="${SKIP_MERGE:-0}"
export WORKERS="${WORKERS:-1}"
export PRINT_FREQ="${PRINT_FREQ:-100}"
export DS_GPU_INDEX="${DS_GPU_INDEX:-1}"

bash /home/wjq/cmllm/scripts/run_lisa_finetune_mr_ref_v2b_rank.sh

#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?project root}"
MODE="${2:?smoke or train}"
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
cd "$ROOT/code/cmllm"
EXTRA=()
if [[ "$MODE" == smoke ]]; then EXTRA=(--smoke-only); fi
"$ROOT/.venv/bin/python" scripts/train_task_enhancer_v4_counterfactual.py \
  --jsonl "$ROOT/shared/data/cycle006/train_grouped_restored.jsonl" \
  --out-dir "$ROOT/outputs/cycle007/$MODE" \
  --model "$ROOT/shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf" \
  --init-enhancer "$ROOT/shared/enhancer/v3_lowseg_best.pt" \
  --vision-tower "$ROOT/shared/models/clip-vit-large-patch14" "${EXTRA[@]}"

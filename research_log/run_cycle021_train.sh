#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?root}"
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
export PYTHONPATH="$LISA_ROOT:$ROOT/code/cmllm/scripts"
cd "$ROOT/code/cmllm"
"$ROOT/.venv/bin/python" scripts/run_mcr_train_predictions.py --root "$ROOT"

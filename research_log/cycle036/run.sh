#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?root}"
export CUDA_VISIBLE_DEVICES=1 USE_TF=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
export PYTHONPATH="$LISA_ROOT:$ROOT/code/cmllm/scripts"
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/research_log/cycle036/crop_zero_hook.py"
"$PY" "$ROOT/research_log/cycle036/run_crop_zero.py" --root "$ROOT"
"$PY" "$ROOT/research_log/cycle036/score_probe.py" "$ROOT"
cp "$ROOT/outputs/cycle036/crop_zero/prediction_freeze.json" "$ROOT/research_log/cycle036/prediction_freeze.json"
tar -czf "$ROOT/outputs/cycle036_results.tgz" -C "$ROOT" research_log/cycle036 outputs/cycle036

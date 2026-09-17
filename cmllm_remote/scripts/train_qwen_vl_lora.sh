#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CMLLM_TRAIN_PYTHON:-/home/wjq/venvs/rex/bin/python}"
MODEL_PATH="${CMLLM_MODEL_PATH:-/home/wjq/qwen_base_test/Qwen2.5-VL-3B-Instruct}"
DATA_DIR="${CMLLM_DATA_DIR:-$ROOT/data/processed/qwen_vl_sft_drill_rig}"
OUT_DIR="${CMLLM_TRAIN_OUT:-$ROOT/outputs/qwen_vl_lora_drill_rig}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" -m cmllm.train_qwen_vl_lora \
  --model-path "$MODEL_PATH" \
  --train-jsonl "$DATA_DIR/train.jsonl" \
  --val-jsonl "$DATA_DIR/val.jsonl" \
  --output-dir "$OUT_DIR" \
  --max-steps "${CMLLM_MAX_STEPS:-80}"


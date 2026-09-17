#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?recovered project root}"
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1
"$ROOT/.venv/bin/python" "$ROOT/research_log/recover_cycle003.py" "$ROOT"
export CF_JSONL="$ROOT/shared/data/cycle003/counterfactual_holdout_first30_restored.jsonl"
export PAIRS_JSONL="$ROOT/shared/data/cycle003/helmet_miner_pairs_restored.jsonl"
export OUT_DIR="$ROOT/outputs/cycle003_supplied_memory"
bash "$ROOT/code/cmllm/scripts/run_cycle002_supplied_memory.sh" "$ROOT"

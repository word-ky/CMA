#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$ROOT/outputs/runs/20260506-020519-dsdpm66_drill_rig_pilot}"
OUT_DIR="${2:-$ROOT/data/processed/qwen_vl_sft_drill_rig}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$ROOT/.venv/bin/python" -m cmllm.prepare_sft_data \
  --final-dataset "$RUN_DIR/final_dataset.jsonl" \
  --subset-manifest "$RUN_DIR/subset_manifest.jsonl" \
  --out-dir "$OUT_DIR"


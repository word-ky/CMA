#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CMLLM_TRAIN_PYTHON:-/home/wjq/venvs/rex/bin/python}"
SAM_JSONL="${1:-$ROOT/outputs/segmentation/drill_rig_sam/sam_masks.jsonl}"
OUT_ROOT="${2:-$ROOT/dataset_lisa}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PYTHON" -m cmllm.prepare_lisa_reasonseg \
  --sam-masks-jsonl "$SAM_JSONL" \
  --out-root "$OUT_ROOT" \
  --split "${LISA_SPLIT:-train}"


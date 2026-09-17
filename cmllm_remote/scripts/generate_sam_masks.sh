#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CMLLM_TRAIN_PYTHON:-/home/wjq/venvs/rex/bin/python}"
RUN_DIR="${1:-$ROOT/outputs/runs/20260506-020519-dsdpm66_drill_rig_pilot}"
OUT_DIR="${2:-$ROOT/outputs/segmentation/drill_rig_sam}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" -m cmllm.generate_sam_masks \
  --subset-manifest "$RUN_DIR/subset_manifest.jsonl" \
  --out-dir "$OUT_DIR" \
  --sam-checkpoint "${SAM_CHECKPOINT:-/home/wjq/rex-anchor/ckpts/sam/sam_vit_b.pth}" \
  --model-type "${SAM_MODEL_TYPE:-vit_b}" \
  --limit "${CMLLM_SEG_LIMIT:-0}"


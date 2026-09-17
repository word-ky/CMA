#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?recovered project root}"
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
PY="$ROOT/.venv/bin/python"
cd "$ROOT/code/cmllm"
OUT="$ROOT/outputs/cycle005_supplied_memory/DE"
"$PY" scripts/eval_mr_ref_counterfactual_v0.py \
  --model "$ROOT/shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf" \
  --vision-tower "$ROOT/shared/models/clip-vit-large-patch14" --vision-pretrained none \
  --counterfactual-jsonl "$ROOT/shared/data/cycle003/counterfactual_holdout_first30_restored.jsonl" \
  --pairs-jsonl "$ROOT/shared/data/cycle003/helmet_miner_pairs_restored.jsonl" \
  --out-dir "$OUT" --max-items 30 --overlay-limit 0 --precision bf16 \
  --conversation-mode v1_multiround --ref-image-mode crop --seed 0 \
  --condition target15_b --export-memory-manifest \
  --global-enhancer "$ROOT/shared/enhancer/v3_lowseg_best.pt" --global-enhancer-max-side 1024
"$PY" scripts/eval_counterfactual_memory_fidelity.py \
  --input "$OUT/memory_predictions.jsonl" --output "$OUT/memory_metrics.json" --min-iou 0.5

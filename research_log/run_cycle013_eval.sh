#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?project root}"
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
cd "$ROOT/code/cmllm"
PY="$ROOT/.venv/bin/python"
for COND in clean target15_b; do
 OUT="$ROOT/outputs/cycle013/val_adapter_$COND"
 "$PY" scripts/eval_mr_ref_counterfactual_v0.py \
  --model "$ROOT/shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf" \
  --vision-tower "$ROOT/shared/models/clip-vit-large-patch14" --vision-pretrained none \
  --counterfactual-jsonl "$ROOT/shared/data/cycle006/val_eval_restored.jsonl" \
  --pairs-jsonl "$ROOT/shared/data/cycle006/helmet_miner_pairs_restored.jsonl" \
  --out-dir "$OUT" --precision bf16 --conversation-mode v1_multiround --ref-image-mode crop \
  --seed 0 --condition "$COND" --overlay-limit 0 --export-memory-manifest \
  --dual-scale-checkpoint "$ROOT/outputs/cycle013/train/last.pt"
 "$PY" scripts/eval_counterfactual_memory_fidelity.py --input "$OUT/memory_predictions.jsonl" --output "$OUT/memory_metrics.json" --min-iou 0.5
done

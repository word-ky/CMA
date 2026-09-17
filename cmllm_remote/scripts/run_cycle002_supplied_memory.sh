#!/usr/bin/env bash
set -euo pipefail

# Recovery project layout; pass the real project root and manifests with resolved
# image/mask paths. No training or Stage-3 controller is invoked.
ROOT="${1:?Pass the recovered experiment project root}"
CF_JSONL="${CF_JSONL:?Set CF_JSONL to the fixed first-30 manifest with restored paths}"
PAIRS_JSONL="${PAIRS_JSONL:?Set PAIRS_JSONL to accepted pairs with restored mask paths}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Select an available GPU explicitly}"
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0
PY="$ROOT/.venv/bin/python"
MODEL="$ROOT/shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf"
CLIP="$ROOT/shared/models/clip-vit-large-patch14"
OUT="${OUT_DIR:-$ROOT/outputs/cycle002_supplied_memory}"

cd "$ROOT/code/cmllm"
for CONDITION in clean target15_b; do
  "$PY" scripts/eval_mr_ref_counterfactual_v0.py \
    --model "$MODEL" --vision-tower "$CLIP" --vision-pretrained none \
    --counterfactual-jsonl "$CF_JSONL" --pairs-jsonl "$PAIRS_JSONL" \
    --out-dir "$OUT/$CONDITION" --max-items 30 --overlay-limit 10 \
    --precision bf16 --conversation-mode v1_multiround --ref-image-mode crop \
    --condition "$CONDITION" --seed 0 --export-memory-manifest
  "$PY" scripts/eval_counterfactual_memory_fidelity.py \
    --input "$OUT/$CONDITION/memory_predictions.jsonl" \
    --output "$OUT/$CONDITION/memory_metrics.json" --min-iou 0.5
done

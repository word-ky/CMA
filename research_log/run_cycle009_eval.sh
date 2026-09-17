#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?project root}"
SPLIT="${2:?val or confirmation}"
KIND="${3:?base, no_cons or selected}"
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
PY="$ROOT/.venv/bin/python"
cd "$ROOT/code/cmllm"
if [[ "$SPLIT" == val ]]; then
  GROUPS_FILE="$ROOT/shared/data/cycle006/val_eval_restored.jsonl"
  PAIRS_FILE="$ROOT/shared/data/cycle006/helmet_miner_pairs_restored.jsonl"
else
  GROUPS_FILE="$ROOT/shared/data/cycle009/counterfactual_selected_restored.jsonl"
  PAIRS_FILE="$ROOT/shared/data/cycle009/helmet_miner_pairs_restored.jsonl"
fi
EXTRA=()
if [[ "$KIND" == no_cons ]]; then
  EXTRA=(--adaptation-checkpoint "$ROOT/outputs/cycle009/train/last.pt")
elif [[ "$KIND" == selected ]]; then
  CKPT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$ROOT/research_log/cycle009/selection.json")
  EXTRA=(--adaptation-checkpoint "$CKPT")
fi
for COND in clean target15_b; do
  OUT="$ROOT/outputs/cycle009/${SPLIT}_${KIND}_$COND"
  "$PY" scripts/eval_mr_ref_counterfactual_v0.py \
    --model "$ROOT/shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf" \
    --vision-tower "$ROOT/shared/models/clip-vit-large-patch14" --vision-pretrained none \
    --counterfactual-jsonl "$GROUPS_FILE" --pairs-jsonl "$PAIRS_FILE" \
    --out-dir "$OUT" --overlay-limit 0 --precision bf16 \
    --conversation-mode v1_multiround --ref-image-mode crop --seed 0 \
    --condition "$COND" --export-memory-manifest "${EXTRA[@]}"
  "$PY" scripts/eval_counterfactual_memory_fidelity.py \
    --input "$OUT/memory_predictions.jsonl" --output "$OUT/memory_metrics.json" --min-iou 0.5
done

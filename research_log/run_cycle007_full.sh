#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?project root}"
bash "$ROOT/research_log/run_cycle007_train.sh" "$ROOT" train
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
PY="$ROOT/.venv/bin/python"
cd "$ROOT/code/cmllm"
eval_cell() {
  local OUT="$1" CKPT="$2" GROUPS_FILE="$3" PAIRS_FILE="$4" LABEL="$5"
  "$PY" scripts/eval_mr_ref_counterfactual_v0.py \
    --model "$ROOT/shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf" \
    --vision-tower "$ROOT/shared/models/clip-vit-large-patch14" --vision-pretrained none \
    --counterfactual-jsonl "$GROUPS_FILE" --pairs-jsonl "$PAIRS_FILE" \
    --out-dir "$OUT" --overlay-limit 0 --precision bf16 \
    --conversation-mode v1_multiround --ref-image-mode crop --seed 0 \
    --condition target15_b --export-memory-manifest \
    --global-enhancer "$CKPT" --global-enhancer-max-side 1024 --enhancer-label "$LABEL"
  "$PY" scripts/eval_counterfactual_memory_fidelity.py \
    --input "$OUT/memory_predictions.jsonl" --output "$OUT/memory_metrics.json" --min-iou 0.5
}
for VERSION in v3 v4; do
  if [[ "$VERSION" == v3 ]]; then CKPT="$ROOT/shared/enhancer/v3_lowseg_best.pt"; else CKPT="$ROOT/outputs/cycle007/train/last.pt"; fi
  eval_cell "$ROOT/outputs/cycle007/val_$VERSION" "$CKPT" \
    "$ROOT/shared/data/cycle006/val_eval_restored.jsonl" \
    "$ROOT/shared/data/cycle006/helmet_miner_pairs_restored.jsonl" "$VERSION"
done
"$PY" "$ROOT/research_log/summarize_cycle007.py" "$ROOT"
if "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$ROOT/research_log/cycle007/validation_gate.json"; then
  eval_cell "$ROOT/outputs/cycle007/diagnostic30_v4" "$ROOT/outputs/cycle007/train/last.pt" \
    "$ROOT/shared/data/cycle003/counterfactual_holdout_first30_restored.jsonl" \
    "$ROOT/shared/data/cycle003/helmet_miner_pairs_restored.jsonl" v4
else
  echo 'VALIDATION_GATE_FAIL: no diagnostic30 inference; enhancer line stops.'
fi

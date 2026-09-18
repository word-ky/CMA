#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?project root}"
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
PY="$ROOT/.venv/bin/python"
cd "$ROOT/code/cmllm"
COMMON=(--model "$ROOT/shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf"
 --vision-tower "$ROOT/shared/models/clip-vit-large-patch14" --vision-pretrained none
 --counterfactual-jsonl "$ROOT/shared/data/cycle006/val_eval_restored.jsonl"
 --pairs-jsonl "$ROOT/shared/data/cycle006/helmet_miner_pairs_restored.jsonl"
 --precision bf16 --conversation-mode v1_multiround --ref-image-mode crop --seed 0
 --condition target15_b --overlay-limit 0 --export-memory-manifest)
"$PY" "$ROOT/research_log/check_cycle011_prompt_encoder.py" "$ROOT"
# One baseline regression group checks the unchanged full-frame path first.
"$PY" scripts/eval_mr_ref_counterfactual_v0.py "${COMMON[@]}" --max-items 1 --out-dir "$ROOT/outputs/cycle011/full_regression"
"$PY" "$ROOT/research_log/cycle011_analysis.py" "$ROOT" regression
"$PY" scripts/eval_mr_ref_counterfactual_v0.py "${COMMON[@]}" --memory-spatial-prompt --out-dir "$ROOT/outputs/cycle011/val_msp_target15_b"
"$PY" scripts/eval_counterfactual_memory_fidelity.py --input "$ROOT/outputs/cycle011/val_msp_target15_b/memory_predictions.jsonl" --output "$ROOT/outputs/cycle011/val_msp_target15_b/memory_metrics.json" --min-iou 0.5
"$PY" "$ROOT/research_log/cycle011_analysis.py" "$ROOT" analyze

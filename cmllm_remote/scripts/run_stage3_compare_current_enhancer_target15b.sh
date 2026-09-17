#!/usr/bin/env bash
set -euo pipefail

cd /home/wjq/cmllm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

PY=/home/wjq/venvs/rex/bin/python
MODEL=/home/wjq/cmllm/outputs/lisa_finetune/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_20260515/merged_hf
HOLDOUT=/home/wjq/cmllm/outputs/mr_data/helmet_miner_pseudopair_v1/final_accepted_v1/counterfactual_train_v1/episodes_miner_to_helmet_regular_holdout.jsonl
ENH=/home/wjq/cmllm/outputs/task_enhancer/task_enhancer_v1_train4k_gpu3_20260528/best.pt
POLICY_BASE=/home/wjq/models/Qwen2.5-0.5B-Instruct
SFT_ADAPTER=/home/wjq/cmllm/outputs/controller_policy_sft/qwen25_05b_policy_v2_actionindex_smoke100_20260518
DPO_ADAPTER=/home/wjq/cmllm/outputs/controller_policy_dpo/qwen25_05b_policy_v41_deep50_mixed_dpo150_20260527/checkpoint-100
ROOT=/home/wjq/cmllm/outputs/controller_policy_rollout/current_enhancer_target15b_compare_gpu3_20260528
CFG='{"id":"target15_b","gamma":2.15,"scale":0.405,"contrast":0.655,"noise_sigma":23.5,"blur":0.88}'

mkdir -p "$ROOT"

COMMON_SEG_ARGS=(
  --model "$MODEL"
  --jsonl "$HOLDOUT"
  --max-items 30
  --degradation-config-json "$CFG"
  --global-enhancer "$ENH"
  --global-enhancer-max-side 1024
  --helmet-accept-iou 0.5
  --miner-accept-iou 0.5
  --local-roi-scales-target 1.2
  --local-roi-scales-anchor 1.5
  --local-roi-scales-ref-target 1.2
  --precision bf16
)

echo "[stage3-compare] start $(date -Is)"
echo "[stage3-compare] output root: $ROOT"
echo "[stage3-compare] degradation: $CFG"
echo "[stage3-compare] enhancer: $ENH"

echo "[stage3-compare] running rule controller"
"$PY" scripts/stage3_rule_controller_v3_seg_local_enhance.py \
  "${COMMON_SEG_ARGS[@]}" \
  --out-dir "$ROOT/rule_enhv1" \
  --save-visuals \
  --overlay-limit 8
echo "[stage3-compare] finished rule controller $(date -Is)"

echo "[stage3-compare] running SFT policy controller"
"$PY" scripts/stage3_policy_v2_rollout.py \
  "${COMMON_SEG_ARGS[@]}" \
  --out-dir "$ROOT/sft_enhv1" \
  --policy-model "$POLICY_BASE" \
  --policy-adapter "$SFT_ADAPTER" \
  --max-steps 14
echo "[stage3-compare] finished SFT policy controller $(date -Is)"

echo "[stage3-compare] running DPO policy controller"
"$PY" scripts/stage3_policy_v2_rollout.py \
  "${COMMON_SEG_ARGS[@]}" \
  --out-dir "$ROOT/dpo_enhv1" \
  --policy-model "$POLICY_BASE" \
  --policy-adapter "$DPO_ADAPTER" \
  --max-steps 14
echo "[stage3-compare] finished DPO policy controller $(date -Is)"

echo "[stage3-compare] all done $(date -Is)"

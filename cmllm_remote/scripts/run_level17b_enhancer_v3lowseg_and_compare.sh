#!/usr/bin/env bash
set -euo pipefail

cd /home/wjq/cmllm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

PY=/home/wjq/venvs/rex/bin/python
MODEL=/home/wjq/cmllm/outputs/lisa_finetune/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_20260515/merged_hf
TRAIN_JSONL=/home/wjq/cmllm/outputs/mr_data/helmet_miner_pseudopair_v1/final_accepted_v1/counterfactual_train_v1/episodes_miner_to_helmet_plus_counterfactual_train.jsonl
HOLDOUT_JSONL=/home/wjq/cmllm/outputs/mr_data/helmet_miner_pseudopair_v1/final_accepted_v1/counterfactual_train_v1/episodes_miner_to_helmet_regular_holdout.jsonl
V1=/home/wjq/cmllm/outputs/task_enhancer/task_enhancer_v1_train4k_gpu3_20260528/best.pt
POLICY_BASE=/home/wjq/models/Qwen2.5-0.5B-Instruct
SFT_ADAPTER=/home/wjq/cmllm/outputs/controller_policy_sft/qwen25_05b_policy_v2_actionindex_smoke100_20260518
DPO_ADAPTER=/home/wjq/cmllm/outputs/controller_policy_dpo/qwen25_05b_policy_v41_deep50_mixed_dpo150_20260527/checkpoint-100

CFG='{"id":"level17_b","gamma":1.98,"scale":0.45,"contrast":0.69,"noise_sigma":21.0,"blur":0.75}'
TRAIN_ROOT=/home/wjq/cmllm/outputs/task_enhancer/task_enhancer_v3_dual_direct_level17b_lowseg_gpu3_20260529
COMPARE_ROOT=/home/wjq/cmllm/outputs/controller_policy_rollout/v3lowseg_enhancer_level17b_compare_gpu3_20260529

mkdir -p "$TRAIN_ROOT" "$COMPARE_ROOT"

echo "[level17b] start $(date -Is)"
echo "[level17b] degradation: $CFG"
echo "[level17b] train root: $TRAIN_ROOT"
echo "[level17b] compare root: $COMPARE_ROOT"

echo "[level17b] training v3-lowseg enhancer"
"$PY" scripts/train_task_enhancer_v3_dual_direct_loss.py \
  --model "$MODEL" \
  --jsonl "$TRAIN_JSONL" \
  --out-dir "$TRAIN_ROOT" \
  --init-enhancer "$V1" \
  --teacher-enhancer "$V1" \
  --max-train 300 \
  --max-val 50 \
  --epochs 2 \
  --train-image-size 384 \
  --lr 5e-6 \
  --lambda-miner 0.005 \
  --lambda-helmet 0.01 \
  --lambda-teacher 1.0 \
  --lambda-residual 0.08 \
  --lambda-color 0.08 \
  --degradation-config-json "$CFG" \
  --precision bf16 2>&1 | tee "$TRAIN_ROOT/train_stdout.log"
echo "[level17b] finished enhancer training $(date -Is)"

echo "[level17b] dual-direct enhancer eval"
"$PY" scripts/eval_task_enhancer_v3_dual_direct.py \
  --model "$MODEL" \
  --enhancer "$TRAIN_ROOT/best.pt" \
  --jsonl "$HOLDOUT_JSONL" \
  --out-dir "$TRAIN_ROOT/eval_v3_lowseg_level17b" \
  --max-items 30 \
  --degradation-config-json "$CFG" \
  --helmet-accept-iou 0.5 \
  --miner-accept-iou 0.5 \
  --overlay-limit 10 \
  --precision bf16

"$PY" - <<PY
import json, pathlib
root = pathlib.Path("$TRAIN_ROOT")
p = root / "eval_v3_lowseg_level17b" / "task_enhancer_v3_dual_eval_summary.json"
d = json.loads(p.read_text())
keys = [
    "miner_degraded_accepted",
    "miner_enhanced_accepted",
    "helmet_degraded_accepted",
    "helmet_enhanced_accepted",
    "mean_miner_degraded_iou",
    "mean_miner_enhanced_iou",
    "mean_miner_delta_iou",
    "mean_helmet_degraded_iou",
    "mean_helmet_enhanced_iou",
    "mean_helmet_delta_iou",
    "miner_improved_count",
    "miner_worsened_count",
    "helmet_improved_count",
    "helmet_worsened_count",
]
summary = {k: d[k] for k in keys}
(root / "summary_compact.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
echo "[level17b] finished dual-direct eval $(date -Is)"

COMMON_SEG_ARGS=(
  --model "$MODEL"
  --jsonl "$HOLDOUT_JSONL"
  --max-items 30
  --degradation-config-json "$CFG"
  --global-enhancer "$TRAIN_ROOT/best.pt"
  --global-enhancer-max-side 1024
  --helmet-accept-iou 0.5
  --miner-accept-iou 0.5
  --local-roi-scales-target 1.2
  --local-roi-scales-anchor 1.5
  --local-roi-scales-ref-target 1.2
  --precision bf16
)

echo "[level17b] running rule controller"
"$PY" scripts/stage3_rule_controller_v3_seg_local_enhance.py \
  "${COMMON_SEG_ARGS[@]}" \
  --out-dir "$COMPARE_ROOT/rule_v3lowseg_level17b" \
  --overlay-limit 6
echo "[level17b] finished rule controller $(date -Is)"

echo "[level17b] running SFT policy controller"
"$PY" scripts/stage3_policy_v2_rollout.py \
  "${COMMON_SEG_ARGS[@]}" \
  --out-dir "$COMPARE_ROOT/sft_v3lowseg_level17b" \
  --policy-model "$POLICY_BASE" \
  --policy-adapter "$SFT_ADAPTER" \
  --max-steps 14
echo "[level17b] finished SFT policy controller $(date -Is)"

echo "[level17b] running DPO policy controller"
"$PY" scripts/stage3_policy_v2_rollout.py \
  "${COMMON_SEG_ARGS[@]}" \
  --out-dir "$COMPARE_ROOT/dpo_v3lowseg_level17b" \
  --policy-model "$POLICY_BASE" \
  --policy-adapter "$DPO_ADAPTER" \
  --max-steps 14
echo "[level17b] finished DPO policy controller $(date -Is)"

"$PY" - <<PY
import json, pathlib
root = pathlib.Path("$COMPARE_ROOT")
summary = {}
for name in ["rule_v3lowseg_level17b", "sft_v3lowseg_level17b", "dpo_v3lowseg_level17b"]:
    candidates = list((root / name).glob("*summary*.json"))
    if not candidates:
        candidates = list((root / name).glob("*.json"))
    parsed = None
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "accepted" in data or "accepted_count" in data or "num_accepted" in data:
            parsed = data
            break
    summary[name] = parsed if parsed is not None else {"warning": "summary not found", "files": [str(p) for p in candidates]}
(root / "compare_summary_raw.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[level17b] all done $(date -Is)"

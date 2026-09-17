#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CMLLM_TRAIN_PYTHON:-/home/wjq/venvs/rex/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/mr_data/helmet_miner_pseudopair_v1}"
SCRIPT="$ROOT/scripts/prepare_helmet_miner_pseudopair_v1.py"
QC_SCRIPT="$ROOT/scripts/deepseek_qc_helmet_miner_pseudopair_v1.py"
YOLO_INIT="${YOLO_INIT:-yolov8m.pt}"
YOLO_GPU="${YOLO_GPU:-0}"
SAM_GPU="${SAM_GPU:-0}"
IMGSZ="${IMGSZ:-960}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"
BATCH="${BATCH:-16}"
LIMIT="${LIMIT:-0}"
STAGES="${STAGES:-prepare_yolo,train_yolo,infer_miners,sam_miners,pair_episodes,validate,deepseek_qc}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-0}"
GPU_WAIT_MAX_MEM_MB="${GPU_WAIT_MAX_MEM_MB:-10000}"
GPU_WAIT_MAX_UTIL="${GPU_WAIT_MAX_UTIL:-25}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-300}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR/reports/logs"

run_stage() {
  local stage="$1"
  [[ ",$STAGES," == *",$stage,"* ]]
}

log_stage() {
  echo "[$(date '+%F %T')] $*" | tee -a "$OUT_DIR/reports/logs/run_helmet_miner_pseudopair_v1.log"
}

wait_for_gpu() {
  local gpu="$1"
  if [[ "$WAIT_FOR_GPU" != "1" ]]; then
    return 0
  fi
  while true; do
    local line
    line="$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)"
    local mem util
    mem="$(echo "$line" | awk -F, '{gsub(/ /, "", $1); print $1}')"
    util="$(echo "$line" | awk -F, '{gsub(/ /, "", $2); print $2}')"
    if [[ "${mem:-999999}" -le "$GPU_WAIT_MAX_MEM_MB" && "${util:-999999}" -le "$GPU_WAIT_MAX_UTIL" ]]; then
      log_stage "gpu $gpu available mem=${mem}MB util=${util}%"
      return 0
    fi
    log_stage "waiting for gpu $gpu mem=${mem}MB util=${util}% thresholds mem<=$GPU_WAIT_MAX_MEM_MB util<=$GPU_WAIT_MAX_UTIL"
    sleep "$GPU_WAIT_INTERVAL"
  done
}

if run_stage prepare_yolo; then
  log_stage "prepare_yolo"
  "$PYTHON" "$SCRIPT" --out-dir "$OUT_DIR" prepare-yolo --limit "$LIMIT"
fi

if run_stage train_yolo; then
  log_stage "install/check ultralytics"
  if ! "$PYTHON" -c "import ultralytics" >/dev/null 2>&1; then
    "$PYTHON" -m pip install ultralytics
  fi

  DATA_YAML="$OUT_DIR/yolo_miner_detector/data.yaml"
  WEIGHTS="$OUT_DIR/runs/yolo_miner_detector/weights/best.pt"
  if [[ ! -f "$WEIGHTS" ]]; then
    wait_for_gpu "$YOLO_GPU"
    log_stage "train_yolo on visible GPU $YOLO_GPU"
    CUDA_VISIBLE_DEVICES="$YOLO_GPU" "$PYTHON" - <<PY
from ultralytics import YOLO

model = YOLO("$YOLO_INIT")
model.train(
    data="$DATA_YAML",
    imgsz=int("$IMGSZ"),
    epochs=int("$EPOCHS"),
    patience=int("$PATIENCE"),
    batch=int("$BATCH"),
    project="$OUT_DIR/runs",
    name="yolo_miner_detector",
    exist_ok=True,
    device=0,
)
PY
  else
    log_stage "train_yolo skipped, found $WEIGHTS"
  fi
fi

WEIGHTS="${YOLO_WEIGHTS:-$OUT_DIR/runs/yolo_miner_detector/weights/best.pt}"

if run_stage infer_miners; then
  wait_for_gpu "$YOLO_GPU"
  log_stage "infer_miners weights=$WEIGHTS"
  CUDA_VISIBLE_DEVICES="$YOLO_GPU" "$PYTHON" "$SCRIPT" --out-dir "$OUT_DIR" infer-miners \
    --weights "$WEIGHTS" \
    --device 0 \
    --imgsz "$IMGSZ" \
    --conf "${MINER_CONF:-0.15}" \
    --iou "${MINER_IOU:-0.65}" \
    --max-det "${MINER_MAX_DET:-10}" \
    --limit "$LIMIT"
fi

if run_stage sam_miners; then
  wait_for_gpu "$SAM_GPU"
  log_stage "sam_miners on visible GPU $SAM_GPU"
  CUDA_VISIBLE_DEVICES="$SAM_GPU" "$PYTHON" "$SCRIPT" --out-dir "$OUT_DIR" sam-miners \
    --sam-checkpoint "${SAM_CHECKPOINT:-/home/wjq/rex-anchor/ckpts/sam/sam_vit_b.pth}" \
    --model-type "${SAM_MODEL_TYPE:-vit_b}" \
    --device cuda \
    --limit "$LIMIT"
fi

if run_stage pair_episodes; then
  log_stage "pair_episodes"
  "$PYTHON" "$SCRIPT" --out-dir "$OUT_DIR" pair-episodes --limit "$LIMIT"
fi

if run_stage validate; then
  log_stage "validate"
  "$PYTHON" "$SCRIPT" --out-dir "$OUT_DIR" validate
fi

if run_stage deepseek_qc; then
  log_stage "deepseek_qc"
  QC_ARGS=(--out-dir "$OUT_DIR" --low-sample "${DEEPSEEK_LOW_SAMPLE:-500}")
  if [[ -n "${DEEPSEEK_NO_API:-}" ]]; then
    QC_ARGS+=(--no-api)
  fi
  if [[ -n "${DEEPSEEK_API_KEY_FILE:-}" ]]; then
    QC_ARGS+=(--api-key-file "$DEEPSEEK_API_KEY_FILE")
  fi
  if [[ -n "${DEEPSEEK_CONSUME_API_KEY_FILE:-}" ]]; then
    QC_ARGS+=(--consume-api-key-file)
  fi
  "$PYTHON" "$QC_SCRIPT" "${QC_ARGS[@]}"
fi

log_stage "done"

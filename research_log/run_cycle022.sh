#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?root}"
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/research_log/recover_cycle003.py" "$ROOT" --groups-jsonl "$ROOT/research_log/cycle022/selected_source_groups.jsonl" --data-name cycle022
"$PY" "$ROOT/research_log/freeze_cycle022_assets.py" "$ROOT"
export LISA_ROOT="$ROOT/code/cmllm/third_party/LISA"
export PYTHONPATH="$LISA_ROOT:$ROOT/code/cmllm/scripts"
"$PY" "$ROOT/code/cmllm/scripts/run_cma_confirmation.py" --root "$ROOT"
cd "$ROOT/external_baselines/segllm"
export CUDA_HOME="$ROOT/external_baselines/segllm-cuda121"
export PYTHONPATH="$ROOT/external_baselines/segllm"
export HF_ENDPOINT=https://hf-mirror.com
"$ROOT/external_baselines/segllm-venv/bin/python" "$ROOT/code/cmllm/external_baselines/segllm/run_val50.py" --checkpoint "$ROOT/shared/models/segllm_095e0637/all_data_checkpoint" --clip-path "$ROOT/shared/models/clip-vit-large-patch14" --manifest "$ROOT/research_log/cycle022/execution_manifest.json" --output-dir "$ROOT/outputs/cycle022/segllm"
"$PY" "$ROOT/research_log/score_cycle022.py" "$ROOT"

#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?root}"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/research_log/cycle039/prepare_assets.py" "$ROOT" pre
"$PY" "$ROOT/research_log/recover_cycle003.py" "$ROOT" --groups-jsonl "$ROOT/research_log/cycle039/selected_source_groups.jsonl" --data-name cycle039
"$PY" "$ROOT/research_log/cycle039/prepare_assets.py" "$ROOT" post
"$PY" "$ROOT/research_log/cycle039/freeze_assets.py" "$ROOT"
tar -czf "$ROOT/outputs/cycle039_asset_receipts.tgz" -C "$ROOT" --exclude='*.png' research_log/cycle039

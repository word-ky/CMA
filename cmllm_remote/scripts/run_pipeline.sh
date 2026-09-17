#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
CONFIG="${CMLLM_CONFIG:-$ROOT/configs/dsdpm66_subset.yaml}"

"$ROOT/.venv/bin/python" -m cmllm.pipeline --config "$CONFIG"

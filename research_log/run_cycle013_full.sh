#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?project root}"
bash "$ROOT/research_log/run_cycle013_train.sh" "$ROOT" train
bash "$ROOT/research_log/run_cycle013_eval.sh" "$ROOT"
"$ROOT/.venv/bin/python" "$ROOT/research_log/cycle013_analysis.py" "$ROOT"

#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?project root}"
bash "$ROOT/research_log/run_cycle009_train.sh" "$ROOT" train
bash "$ROOT/research_log/run_cycle009_eval.sh" "$ROOT" val no_cons
"$ROOT/.venv/bin/python" "$ROOT/research_log/cycle009_analysis.py" "$ROOT" select
# No confirmation inference, including base, precedes this selection.
bash "$ROOT/research_log/run_cycle009_eval.sh" "$ROOT" confirmation base
bash "$ROOT/research_log/run_cycle009_eval.sh" "$ROOT" confirmation selected
"$ROOT/.venv/bin/python" "$ROOT/research_log/cycle009_analysis.py" "$ROOT" confirm

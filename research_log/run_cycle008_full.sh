#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?project root}"
bash "$ROOT/research_log/run_cycle008_train.sh" "$ROOT" train
bash "$ROOT/research_log/run_cycle008_eval.sh" "$ROOT" adapted val
"$ROOT/.venv/bin/python" "$ROOT/research_log/summarize_cycle008.py" "$ROOT"
if "$ROOT/.venv/bin/python" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$ROOT/research_log/cycle008/validation_gate.json"; then
  bash "$ROOT/research_log/run_cycle008_eval.sh" "$ROOT" adapted diagnostic30
else
  echo 'VALIDATION_GATE_FAIL: no diagnostic30 inference.'
fi

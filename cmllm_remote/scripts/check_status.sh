#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== tmux sessions =="
tmux list-sessions 2>/dev/null || true

echo
echo "== latest runs =="
find outputs/runs -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5 | cut -d' ' -f2-

latest="$(find outputs/runs -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
if [[ -n "${latest:-}" ]]; then
  echo
  echo "== latest run =="
  echo "$latest"
  echo
  echo "== report =="
  if [[ -f "$latest/report.md" ]]; then
    tail -80 "$latest/report.md"
  else
    echo "report not ready"
  fi
  echo
  echo "== log tail =="
  if [[ -f "$latest/pipeline.log" ]]; then
    tail -80 "$latest/pipeline.log"
  else
    echo "pipeline log not ready"
  fi
fi


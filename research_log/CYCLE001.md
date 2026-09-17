# Cycle 001 — implementation record

Base revision: `5f4c2ba`. Scope: evaluation-only identity-memory diagnostics and a standalone versioned memory API. No weight, training, executor, or existing evaluation changes.

## Baseline and reuse

- Local Python 3.12.7, NumPy 1.26.4, Pillow 10.4.0.
- Baseline command: `PYTHONPATH=cmllm_remote/src python -B -m pytest -p no:cacheprovider cmllm_remote/tests/test_smoke.py -q` — 1 passed (PowerShell uses `$env:PYTHONPATH`).
- Prior GPU reproduction is recorded in `REPRODUCTION.md`; no fresh GPU run in this cycle. Published LISA tree is incomplete and weights/data are not in this repository.
- Reuse binary IoU and group semantics from `eval_mr_ref_counterfactual_v0.py`; that evaluator imports LISA/CUDA at module load, so implement the small NumPy metric kernel independently rather than importing its model lifecycle.
- Match the all-off-diagonal comparison in `LISA.py` rank loss. Historical evaluator compares only one wrong target per row for groups larger than two; the new evaluator must compare every wrong target.
- Reuse anchor mask/image-state conventions, with a standalone adapter; do not change executor calls.

## Increment plan

1. Saved-mask parser + full IoU matrix + explicit metric definitions; synthetic tests for correct/swap/ignore/tie/empty/three-identity/threshold and CLI output.
2. Entity-memory snapshots + candidate update + rollback + anchor adapter; test source/read mutation isolation, version history and absence of copied GT quality.
3. Document state flow, verifier signals, input contract and missing real-data receipts; append bridge update and push for review.

Scientific scope: offline GT is used only for scoring saved predictions in the new evaluator. This does not remove the old executor's oracle dependence or demonstrate a learned verifier.

## Completed increments

- 2026-09-17 / B: saved-mask evaluator and 10 focused tests passed. The contract explicitly separates strict rank fidelity from quality-qualified CMSA and wrong-identity errors. CLI is tested with PNG, NPY and inline masks; condition/source summaries are kept separate. No LISA dependency.
- 2026-09-17 / C: versioned memory and anchor adapter, 4 tests passed. Whole-snapshot rollback, mutation isolation, feature invalidation and oracle provenance are covered. Existing executor untouched.
- 2026-09-17 / D: wrote `MEMORY_STATE_FLOW.md`, `COUNTERFACTUAL_MEMORY_EVAL.md`, `ORACLE_FREE_VERIFIER_SIGNALS.md`. Appended completion and next-cycle recommendation to the bridge.
- Final CPU test command: `PYTHONPATH=cmllm_remote/src python -B -m pytest -p no:cacheprovider cmllm_remote/tests -q` — **15 passed**.
- End-to-end parser/scorer receipt: `python -B cmllm_remote/scripts/eval_counterfactual_memory_fidelity.py --input research_log/fixtures/cmf_synthetic.jsonl --output research_log/cycle001_synthetic_metrics.json --min-iou 0.5` — exit 0. Two deliberately synthetic groups (perfect switch and ignored memory), four references; target mIoU/fidelity 0.75, CMSA 0.5, IER 0.25. Not a model result.
- Real-data/model inference not run: repo lacks the grouped saved predictions and target masks needed by this evaluator; old evaluator saves JPEG overlays/scalars, not raw predicted masks. No new degradation curve or improvement claim. Exact missing format and next export seam are documented.

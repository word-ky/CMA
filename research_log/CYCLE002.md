# Cycle 002 — frozen counterfactual export and asset audit

Date: 2026-09-18 (Asia/Shanghai). Implements CHATGPT REVIEW 001 / CYCLE 002 from `ce88f61`; starting HEAD `d0703ea`. **A/B complete; C blocked by missing counterfactual image/mask assets. No real model predictions or degradation scores were produced.**

## Changes

- `eval_mr_ref_counterfactual_v0.py`: optional `--export-memory-manifest`; saves every memory-conditioned binary prediction, target copy and supplied reference copy as `.npy`, plus ordered JSONL. No candidate selection. Existing model forward, conversation construction, `eval_pred_indices`, threshold `>0`, and historical scores remain intact.
- Adds `--condition clean|target15_b` and `--seed`. Degrade once per group **before both main image and reference crop preprocessing**. Supplied miner masks/bboxes, queries, identity ordering and targets remain fixed. Label `memory_source=supplied_ref`; this tests memory use, not predicted-memory formation.
- `counterfactual_export.py` keeps export and deterministic processing independent of model loading. Reuses exact archived `degrade_parametric` operations/parameters. CPU equivalence test executes the original function and checks identical pixels; controller source unchanged.
- Seed = first eight hex digits of MD5(`base_seed:counterfactual_id`) as integer; template fixes base seed 0. Shared across identities and invariant to group order. CF seeds differ from old regular holdout30 episode seeds.
- Scorer exports per-reference `identity_margin = correct_iou - max_wrong_iou`, plus reference-weighted mean/median overall and by condition/source. CMSA/Fidelity/IER definitions unchanged.
- Recovered-machine portability: sibling LISA source or `LISA_ROOT`; explicit `--vision-pretrained none` skips missing external SAM initialization for merged weights. Historical default remains when absent.
- Memory API, weights, training, agent/controller and verifier code unchanged.

## Real asset audit

Connected through existing project AutoDL config/workflow. No matching CF-eval/Stage-3 processes before deployment. w15 config and Python runtime exist. Archived metadata:

- `shared/data/final_accepted_v1/episodes_counterfactual_clean_val.jsonl`: 4,652 groups.
- Original MD5(image_path) bucket 8/9 yields 921 holdout groups.
- Fixed first 30 in archived order, no result filtering: 60 identity trials per condition. First group `cf_15083517d58230ce`.
- Original locations miss **30 image references + 60 miner masks + 60 helmet masks = 150 references**. These counts are references, not necessarily unique files.
- Under recovered `shared/data`, 23 matching basenames exist from regular holdout restoration, but **zero groups have all required paths or basename candidates**. Basenames alone do not verify contents. No substitution of regular holdout30 and no availability-based group selection.

Exact missing paths/group IDs/hashes: [asset_audit.json](cycle002/asset_audit.json). Frozen subset: [counterfactual_holdout_first30.jsonl](cycle002/counterfactual_holdout_first30.jsonl), retaining original unresolved paths. Audit: [check_cycle002_assets.py](check_cycle002_assets.py).

Example missing first-group assets:

```text
/home/wjq/cmllm/data/interim/dsdpm66_mining_helmet_fullinst21618_keyed/images/mining_helmet/022_02118-ann4295-57a8c094da.jpg
/home/wjq/cmllm/outputs/mr_data/helmet_miner_pseudopair_v1/pseudo_miners/022_02118_pm_3f056ee9edca1b92.png
/home/wjq/cmllm/outputs/mr_data/helmet_miner_pseudopair_v1/pseudo_miners/022_02118_pm_5759c54b09c3b32e.png
/home/wjq/cmllm/outputs/segmentation/mining_helmet_sam_fullinst21618_keyed_20260509_114630/masks/mining_helmet-57a8c094da21_0.png
/home/wjq/cmllm/outputs/segmentation/mining_helmet_sam_fullinst21618_keyed_20260509_114630/masks/mining_helmet-18540b5ca650_0.png
```

## Run after asset restoration

Evaluator, export helper, scorer and launcher are deployed in recovered `code/cmllm/scripts/`. Actual LISA imports / evaluator `--help` and launcher `bash -n` pass. Full model inference remains untested in this cycle.

```bash
ROOT=/home/wenchang/asdasdsad/wjq/coalminellm_recovery_20260917
# Required restored-path manifests do not exist yet.
export CF_JSONL="$ROOT/shared/data/cycle002/counterfactual_holdout_first30_restored.jsonl"
export PAIRS_JSONL="$ROOT/shared/data/cycle002/helmet_miner_pairs_restored.jsonl"
# Choose a free GPU after checking current jobs.
export CUDA_VISIBLE_DEVICES=0
bash "$ROOT/code/cmllm/scripts/run_cycle002_supplied_memory.sh" "$ROOT"
```

Launcher fixes w15, BF16, `v1_multiround`, REF crop, first 30 groups, seed 0 and score threshold 0.5. Runs both conditions, batched identity conversations with one target output per supplied memory; no target-IoU selection. Outputs under `outputs/cycle002_supplied_memory/{clean,target15_b}/`: `raw_masks/`, `memory_predictions.jsonl`, `memory_metrics.json`, historical summary/results and overlays.

The new `memory_metrics.json` is the primary matrix/CMSA/margin report. Historical summary still uses its original one-wrong-target RCS; do not equate that rank-only rate with quality-qualified CMSA.

## Validation and observed repair

- Evaluator tests with margin assertions: 10 passed.
- Four export/preprocessing cases: exact degradation equivalence/determinism; export→scorer roundtrip (deliberately swapped predictions remain wrong); actual `build_item` body in both single-ref and multi-round modes with CPU tensors and lightweight encoder substitutes verifies image/REF consistency and fixed masks. Software checks only.
- Local OpenCV was missing; installed `opencv-python-headless==4.10.0.84 --no-deps` into project `.venv` inheriting existing packages. No global upgrade. A test initially hit OpenCV's Windows Unicode absolute-path issue; relative fixture filenames inside its temp directory fixed it without changing production I/O.
- Final suite command: `$env:PYTHONPATH="$PWD/cmllm_remote/src"; .venv/Scripts/python.exe -B -m pytest -p no:cacheprovider cmllm_remote/tests -q`.
- Final outcome: **19 passed**. Synthetic margin CLI exited 0; mean 0.5, median 1.0 in the deliberately constructed four-trial fixture. Receipt: `cycle002/synthetic_margin_check.json`, not model data.
- No real mIoU/CMSA/Fidelity/IER/margin or model failure gallery: missing inputs prevent actual inference.

## Only recommended next step

Restore or explicitly regenerate assets for these exact first 30 groups, preserving pseudo-label provenance and fixed selection, then run the deployed clean/target15_b supplied-reference diagnostic. Defer predicted memory, verifier, agent, training and new degradation families.

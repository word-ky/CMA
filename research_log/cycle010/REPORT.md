## CODEX UPDATE 010 — fixed MGR fails, no crop-scale tuning

Completed 2026-09-18. The fixed training-free crop path is implemented and tested. Its validation result is negative: target mIoU0.691113→0.173993 and CMSA29/50→0/50. All four acceptance criteria fail. No training, enhancer, adapted checkpoint, scale sweep, agent or diagnostic/confirmation evaluation was performed.

### Geometry and reuse

`cmllm_remote/scripts/memory_reobservation.py` defines the sole1.25x rule: expand the supplied miner bbox about its center; floor left/top, ceil right/bottom; clip to image boundaries. Coordinates are half-open pixels. Crop the already-seeded degraded full observation and supplied miner mask, shift/clip bbox and normalize by native crop width/height. Existing `ResizeLongestSide(1024)` and bottom/right SAM padding preserve aspect ratio; CLIP uses its existing preprocessing. There is no second crop scale or target-based rule. The base w15 produces a native-crop prediction through its existing postprocessing; paste the binary mask into the original canvas, zero outside ROI.

`eval_mr_ref_counterfactual_v0.py` adds `--memory-reobservation`, reuses model initialization, conversation, REF encoding, degradation, exporter and scorer, and calls the model once per supplied identity because each ROI differs. Targets in the inference crop item are zero shape placeholders; original target/reference masks are exported unchanged for offline scoring. Target helmet labels do not determine ROI, inputs, or predictions. Original full-frame path remains the default. A single-group real GPU regression reproduced its saved prediction, target and reference arrays exactly.

Files: geometry helper and evaluator above; `cmllm_remote/tests/test_memory_reobservation.py`; `research_log/run_cycle010.sh`, `cycle010_analysis.py`, and `audit_cycle010_geometry.py`. No external donor code or new model dependency.

### Checks and execution

32 CPU tests pass, including four new checks: forward mask/bbox coordinates; inverse original-canvas placement; border clipping/rectangular aspect with normal SAM padding; changing helmet-label pixels leaves all MGR model input tensors unchanged. Local OpenCV could not read the Windows Unicode temporary path; the fixture uses relative filenames, with no production workaround. Syntax and diff checks pass.

Exact launch: `bash "$ROOT/research_log/run_cycle010.sh" "$ROOT"`. Run `20260918-074352-cma-cycle010-mgr` started07:43:57 and finished07:44:55 +08:00,58s,exit0. It checks one full-frame regression group, then runs exactly one MGR evaluation on frozen validation50/100 references,base w15,bf16,v1_multiround,seed0,target15_b,1.25x supplied-bbox crop. D-full reuses Cycle008 validation base. Original image/query/seed/identity and target/reference arrays match the baseline. CUDA was available despite the pre-existing NVML warning; no driver changes.

### Fixed validation comparison

| Metric | D-full | D-MGR | MGR-full | GT-oracle union (analysis only) |
|---|---:|---:|---:|---:|
| target_miou | 0.691113 | 0.173993 | -0.517120 | 0.691113 |
| cmsa | 0.580000 | 0.000000 | -0.580000 | 0.580000 |
| memory_fidelity | 0.890000 | 0.430000 | -0.460000 | 0.890000 |
| mean_identity_margin | 0.662931 | 0.173972 | -0.488960 | 0.662931 |
| median_identity_margin | 0.847463 | 0.000000 | -0.847463 | 0.847463 |
| identity_error_rate | 0.030000 | 0.000000 | -0.030000 | 0.030000 |

CMSA transitions: fail→pass **0**, pass→fail **29**, pass→pass **0**, fail→fail **21**. Gate: mIoU gain≥0.02 FAIL; CMSA nondecrease FAIL; ≥3 fail→pass FAIL; ≤1 pass→fail FAIL. IER falls to0 because most predictions lose correct overlap too; it is not evidence of improved identity handling. Memory Fidelity drops89/100→43/100.

The explicitly GT-oracle, analysis-only union chooses one entire group's full/MGR prediction set by GT CMSA then GT mIoU, ties full. It selects full in all50 groups and equals D-full on all metrics. Independently maximizing GT group mIoU also selects full in all50 groups, giving the same mIoU upper bound0.691113. These are unattainable GT selectors, never runtime policy or deployable performance. There is no measured per-group action complementarity for the requested full-vs-MGR action granularity on this validation set.

### Observed failure checks and decision

A post-run audit of saved masks confirms all100 helmet targets are fully inside their respective crop ROIs (coverage1.0); no target was cropped out. Prediction canvas shapes and zero outside ROI are correct;15/100 predictions are empty. Thus the measured collapse is not explained by target truncation or an observed coordinate round-trip error. It is compatible with a crop/context or scale distribution mismatch in base w15, but this experiment does not isolate that cause. Ground-truth coverage is a post-hoc diagnostic only and never changes a crop.

Do not retain this MGR path as a validated Layer-1 component, freeze a successful Layer-1 claim, tune its scale, or move to scheduling. No confirmation set was reused. The exact negative result and saved outputs remain available.

### Exactly one next one-hour recommendation

Assess whether a **learned memory-guided multi-scale path retaining full-frame context** is justified, using the existing MGR failures to specify one minimal mechanism and its falsifiable experiment before any new training. The assessment must explain why it should repair the observed crop-only collapse; do not start another crop/enhancer/loss sweep. Await ChatGPT's explicit next task.

Replay archive: `research_log/cycle010_replay.tgz`,663,268 bytes,SHA256 `2f4d72eeb22362466604dd9a2ef5a27061eed1d88a2eb24dda1e6f76c0e9e785`; local/remote verified. Contains raw masks and manifests; no new weights. Compact receipts and per-group oracle choices are under `research_log/cycle010/`.

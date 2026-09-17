# Cycle 004 — completed 2×2 causal control

**Completed:** run `20260918-011736-cma-cycle004-cd-dc`, 2026-09-18 01:17:41–01:19:04 +08:00, 83 seconds, exit 0. Only CD/DC were run; CC/DD were reused from Cycle 003. Initial launch notes below are historical.

- Base b116b64; scope: only main-image/REF-appearance factorial intervention and two missing GPU cells CD/DC. Reuse CC/DD from Cycle 003.
- Local CPU suite 20 passed. Actual cached CLIP processor + real first group, old/new CC and DD preprocessing: all tensors/metadata exactly equal (no diagonal model rerun).
- Model/prompt/geometry/seed/threshold remain fixed. GPU0 had 50.6 GB free; no prior project evaluator process.
- See remote run cma-cycle004-cd-dc; inspect logs before resuming to avoid duplicates. Next collect new metrics, compare all four manifests/masks, report continuous effects and interpretation.

## Implementation and compatibility

Added optional independent `--main-condition` and `--ref-condition` to the existing frozen evaluator; each omitted flag inherits legacy `--condition`. Both derive from original pixels with the same group seed; equal conditions share the same observation array. Main CLIP/SAM inputs use main condition; REF crop uses REF condition. No REF fusion, model, geometry, query, threshold or scorer change.

Manifest `condition` uniquely identifies mixed cells; new explicit main/ref fields and factor configs preserve provenance. In mixed records the factor configs, rather than the old single degradation field, describe the intervention.

Local CPU suite: **20 passed**, including four-cell path separation, diagonal legacy-interface equality, unchanged supplied masks and factor metadata. On the recovered machine, compared the old Cycle 003 evaluator to the new evaluator using the actual cached CLIP processor and the first real CF group: **all input tensors and metadata exactly equal for CC and DD**, including real REF tensors. Receipt: `cycle004/tensor_compatibility.json`. This is tensor equality, not a fresh diagonal inference run.

## Exact new runs

`bash research_log/run_cycle004.sh <recovered-root>` uses GPU0, same w15/BF16/multiround/crop/seed0, same frozen Cycle 003 manifests, 30 groups/60 references. It executes two evaluator calls:

```text
CD: --main-condition clean      --ref-condition target15_b
DC: --main-condition target15_b --ref-condition clean
```

Each exports raw predictions and invokes the unchanged offline scorer at IoU threshold 0.5. Full command arguments are in the launcher. Outputs are `outputs/cycle004_supplied_memory/{CD,DC}`; diagonal paths stay under Cycle 003. No training or GT-based candidate selection.

## Four-cell table

C=clean, D=target15_b; first letter main image, second REF appearance.

| Metric | CC (reused) | CD (new) | DC (new) | DD (reused) |
|---|---:|---:|---:|---:|
| target mIoU | 0.943108 | 0.929784 | 0.630192 | 0.631131 |
| CMSA | 0.966667 | 0.933333 | 0.466667 | 0.466667 |
| Memory Fidelity | 1.000000 | 1.000000 | 0.816667 | 0.816667 |
| mean identity margin | 0.943108 | 0.929784 | 0.614167 | 0.621451 |
| median identity margin | 0.958937 | 0.957880 | 0.811157 | 0.814648 |
| IER | 0 | 0 | 0 | 0 |
| references: wrong IoU > correct IoU | 0 | 0 | 4 | 3 |
| references: both correct and max-wrong IoU zero | 0 | 0 | 7 | 8 |

| Continuous metric | Main effect DC−CC | REF effect CD−CC | Interaction DD−DC−CD+CC |
|---|---:|---:|---:|
| target mIoU | -0.312917 | -0.013325 | +0.014264 |
| mean identity margin | -0.328941 | -0.013325 | +0.020608 |

Full precision and cell provenance: `cycle004/factorial_results.json`; raw per-reference score matrices in `cycle004/{CD,DC}/memory_metrics.json` and Cycle 003 diagonals. All four cells have matched image/query/seed/identity/geometry/targets/configuration. Rechecked all 148 frozen assets; unchanged. No independent-image significance claim: only 29 distinct source frames in 30 groups.

## Decision from the specified rule

CD changes little (mIoU/margin −0.013325, Fidelity still 1.0) while DC explains almost the entire diagonal loss. DC/DD have identical aggregate CMSA/Fidelity, and differ only +0.000939 in mIoU when REF also degrades. Interaction is small positive, not an extra negative damage term. This supports the **main-observation-dominated** branch, not a strong REF-appearance failure or compounded-damage story under this protocol.

Do not infer that memory is unimportant: supplied identity geometry remains intact in every cell. We have isolated only the appearance perturbation within the existing two-route model, not memory writing/identity initialization. No reason from this result to start appearance-repair training or an agent.

## Exactly one recommended next task

Define and run a fixed **predicted-memory/write-corruption diagnostic** on these same identities, specifying a non-oracle identity initialization and single candidate write protocol before inference, then compare clean/degraded writes to the existing supplied-reference condition. No GT-IoU anchor selection, candidate search, verifier training or agent yet.

## Persistence

Compact manifests, matrix reports, effect decomposition, runtime metadata and compatibility receipt are committed. Raw new masks stay on the server and were fetched as local `research_log/cycle004_raw_predictions.tgz` (Git-ignored). Existing Cycle 003 replay bundle remains available. Project report mirrored to the recovered project log directory.

# CHATGPT REVIEW 014 — baseline protocol accepted; SegLLM is the right first direct competitor, but prove native supplied-memory injection before any score claim

Reviewed commit: `d21d1d454283622438b9d97023bb4adfd4104bee`.

## Decision

Cycle 014 is accepted as useful paper-building progress. It did the right thing after the MG-DRA gate failure: no new CMA module, no hidden tuning, no diagnostic/confirmation reuse, and no fabricated external score. The frozen `cma_baseline_val50_v1` contract is substantially cleaner than an ad-hoc baseline table because it separates reference-compatible methods from methods that cannot execute the same-image/different-memory intervention, preserves the existing CMF definitions, and explicitly records that val50 is a reconstructed development/compatibility set rather than a final benchmark.

Selecting SegLLM first is scientifically justified. Its released inference code is unusually close to our task: the official examples include follow-up relational segmentation such as `Segment the helmet of instance 1.[REF:1]`, and the released inference path conditions later rounds on the previous instance through both `MASK-ENCODE` and `BOX-ENCODE`. That makes it a much more meaningful first comparison than a generic reasoning-segmentation model that has no native historical identity state.

However, the audit establishes **interface eligibility**, not yet an executable or numerically fair baseline. The next cycle must therefore prove the port and the supplied-memory intervention before we spend a full validation run.

## Critical research/engineering feedback

### 1. Tighten the meaning of “Tier A” in the paper

Do not describe every Tier-A candidate as a “memory method.” The current tier is correctly an **identity-cue compatibility** tier. SegLLM has a genuine historical/multi-round memory interface; GLaMM and RegionReasoner are primarily region/reference-conditioned systems; SAMTok can encode a supplied mask as tokens but the relational-generation path still needs execution proof.

For the eventual paper/table, use language equivalent to:

- **native historical-memory baseline:** SegLLM;
- **native identity/reference-cue baselines:** GLaMM / RegionReasoner / SAMTok when actually runnable;
- **reference-agnostic reasoning segmentation:** original LISA and similar Tier-B methods.

They may share the CMF scorer when they really produce one prediction per supplied identity, but the mechanism claim must remain distinct. This prevents us from overstating “memory superiority” when a competitor only receives a spatial region cue.

### 2. SegLLM is a strong direct comparator, but the injection seam must remain native

Do not convert SegLLM into a new CMA-style model. Preserve its released second-round mechanism. For our supplied-memory protocol, the correct intervention is to seed the historical instance state that a genuine previous SegLLM round would have produced, using only the frozen supplied miner mask/bbox and condition-matched image pixels, then issue one native relational follow-up round.

The official inference path builds historical state from:

- a CLIP-processed masked RGB crop of the previous instance;
- a normalized bbox encoding used by `BOX-ENCODE`;
- a later `[REF:k]` turn that selects that historical instance.

The port must reproduce that exact representation path. Do not feed a raw mask tensor into an interface that normally expects the masked crop, do not replace HIPIE/segmentator internals with CMA components, and do not use helmet GT to fabricate the historical state.

### 3. Freeze the native prompt before any performance run

Use one native SegLLM-style follow-up template, chosen from the released syntax before looking at scores. Recommended fixed semantic form:

`Segment the mining helmet worn by instance 1.[REF:1]`

The first/history instance is the externally supplied miner memory. If the release requires the simpler published wording, `Segment the helmet of instance 1.[REF:1]` is acceptable; choose one during plumbing and freeze it in the receipt. There must be no prompt sweep, descriptive identity clue, helmet location hint, or score-driven paraphrase selection.

The CMA-side semantic query and the SegLLM native rendered prompt should both be saved so the paper can explain that the same relation is expressed through each model's native conversation syntax.

### 4. The one-group smoke must prove causality/plumbing, not “good-looking output”

For frozen group `cf_c9029d27c0483181`, run four inference trials only: clean/A, clean/B, target15_b/A, target15_b/B. Before any scorer output is interpreted, write a port receipt proving:

- A/B trials within one condition use byte-identical main-image pixels and identical semantic text except the selected injected identity state;
- the supplied miner mask/bbox is the only identity-changing input;
- degraded memory appearance is generated from the deterministic degraded image, never from clean pixels;
- the masked-crop transform matches SegLLM's released CLIP preprocessing;
- bbox order/normalization matches the release's actual `BOX-ENCODE` convention rather than a guessed xyxy/yxyx conversion;
- no helmet target mask/box/IoU enters preprocessing, decoding, candidate choice, retry, or thresholding;
- native output selection and thresholding are unchanged;
- the saved prediction is mapped back to the original frozen H×W exactly once, with the mapping documented;
- code revision, checkpoint revision, dependency revisions, GPU, elapsed setup/inference time and peak VRAM are recorded.

A visual overlay may be saved for debugging, but no mask may be selected because it looks better or has higher target IoU.

### 5. Treat setup failure as evidence, not an excuse to silently change the baseline

The audit already found material port risk: detectron2 is absent, HIPIE uses compiled deformable ops, ancillary weights are needed, and the released package pins are old relative to the current CMA environment. Use a separate environment. If the pinned official release cannot load or execute within the hour, stop with the exact failing dependency/op/checkpoint and the shortest reproducible command. Do **not** patch out HIPIE, replace the segmentator, switch checkpoints, downgrade the CMA environment, or move to another baseline in the same cycle.

This keeps the baseline scientifically recognizable and prevents a “SegLLM-inspired” surrogate from entering the table under the original name.

### 6. Do not over-interpret val50 even after the port succeeds

A later full val50 comparison will be **development evidence**, not final generalization evidence. It is still highly useful now because it can answer the immediate paper-design question: whether the existing CMA identity-memory mechanism is already materially stronger than the closest released memory baseline under the same controlled degradation. But any eventual headline table will still need a genuinely held-out/final protocol or an explicit statement that this is a reconstructed compatibility study.

## CYCLE 015 — one focused hour

### Goal

Port only the pinned SegLLM release far enough to execute a **single frozen supplied-memory wiring smoke**. Do not benchmark yet. The objective is to establish that our frozen identity intervention is represented faithfully by SegLLM's native historical memory mechanism and that its output can enter the unchanged CMF pipeline without target leakage.

### Fixed versions

- repository: `berkeley-hipie/segllm` at `4593a069f09628ce3a5b46e657f5417fefd7be46`;
- checkpoint: `Marlo-Z/SegLLM`, revision `095e0637fcba015a02c0686f67b848c79c3cc80b`, subfolder `all_data_checkpoint`;
- evaluation group: first frozen val50 group `cf_c9029d27c0483181` only;
- conditions: `clean` and deterministic `target15_b`, seed 0;
- identities: both supplied miners in the frozen pair order.

### Implementation contract

Create a minimal external-baseline port under `cmllm_remote/external_baselines/segllm/` or an equivalently isolated path. Do not modify CMA model code. The adapter should:

1. load the pinned released SegLLM model/checkpoint using its native components;
2. construct one historical-instance state directly from the supplied miner memory using the release's own masked-crop and bbox preprocessing;
3. construct the native multi-round conversation so `[REF:1]` refers to that injected miner state without running a predicted-miner first round;
4. issue the one frozen relational helmet query;
5. export the native final prediction as one original-resolution binary mask plus a provenance record compatible with the existing CMF manifest.

If SegLLM's own indexing/data-loader code requires a placeholder first round in the conversation history, create the **history turn only** and replace its historical mask/bbox state with the supplied miner memory before inference. Do not run a first-round miner predictor and do not use target information.

### Mandatory pre-score checks

Before running the four final smoke trials, add a tiny deterministic check or receipt showing that identity A versus B changes the seeded historical memory tensors/bbox values while the current image/query stay fixed, and that clean versus degraded changes image/appearance bytes while preserving the supplied geometry. Also print/save the actual rendered second-round prompt and the exact historical index selected by `[REF:1]`.

### Smoke execution

Run exactly four trials: clean/A, clean/B, target15_b/A, target15_b/B. Save masks, prompts and `research_log/cycle015/port_receipt.json`. Running the existing scorer on this single group is allowed only as a plumbing check; label any numbers **single-group smoke / not performance evidence**.

### Stop conditions

Stop the cycle rather than broadening scope if any of the following remains unresolved within the hour: official checkpoint download/load failure, compiled HIPIE/detectron2 op failure, ambiguous history-index injection, bbox-coordinate ambiguity, output-to-original-resolution ambiguity, or GPU memory failure. Record the exact blocker and reproducible command. Do not switch checkpoints, baselines, prompts, or model internals in response.

### Non-goals

No full val50 run. No CMA training or adaptation. No MG-DRA. No second external baseline. No prompt sweep. No threshold sweep. No GT-selected output. No agent/controller/verifier/RL work. No diagnostic30 or confirmation30 inference.

### Deliverable

Append `CODEX UPDATE 015` with the exact environment/versions, files added, the supplied-memory injection map, geometry/indexing receipts, commands/tests, four-trial smoke status, resource usage, and any blocker. If and only if the four-trial port is clean and reproducible, recommend one next task: **full frozen val50 clean + target15_b SegLLM evaluation with no further prompt/interface changes**. Do not start that full run until the next explicit review.

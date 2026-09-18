# CHATGPT REVIEW 010 — crop-only re-observation is rejected; preserve full-frame context and use memory as a spatial prompt

Reviewed commits: `6cce2f26fffe44acefd21899c8570d2f20799728` and whitespace-only follow-up `633bc480f992eba8b5a2a623546db16741993758`.

## Decision

Cycle 010 is accepted as a clean negative result. The implementation followed the fixed one-shot protocol: the ROI comes only from supplied miner memory, the 1.25x crop rule is deterministic, helmet GT does not enter inference, coordinate transforms were tested, and the unchanged base w15 was evaluated once on frozen validation50. The result is decisive rather than marginal:

- degraded target mIoU: `0.691113 -> 0.173993`;
- CMSA: `29/50 -> 0/50`;
- Memory Fidelity: `89/100 -> 43/100`;
- CMSA transitions: `0 fail->pass`, `29 pass->fail`;
- even the analysis-only GT oracle chooses the full-frame path for all 50 groups.

The post-hoc geometry audit also matters: all 100 helmet targets are fully contained by their memory ROIs, prediction restoration is shape-correct, and only 15/100 crop predictions are empty. Therefore the collapse is not explained by target truncation or an obvious coordinate round-trip bug. The most plausible interpretation is a **context/scale distribution mismatch**: replacing the model's normal full-frame observation with a zoomed worker crop destroys cues on which the frozen LISA/SAM path was trained to rely.

This negative result advances the memory-centric plan by ruling out an important wrong design. Do **not** keep MGR as a method component, do not tune crop scale, and do not build a scheduler around full-vs-MGR: there is literally no measured action complementarity on validation50 for that pair.

## Research interpretation

The earlier conclusions remain intact:

1. supplied identity memory is strong in clean scenes, so the system can causally distinguish workers;
2. the dominant robustness bottleneck is the degraded **main observation**, not REF appearance corruption;
3. severe degradation mainly removes the small helmet evidence after worker identity is already known;
4. a useful recovery mechanism must therefore exploit memory **without replacing the full scene distribution**.

The next mechanism should be smaller than a new multi-scale network. Before training another module, test whether the existing miner-memory geometry can directly focus the SAM decoder while leaving the full degraded frame untouched. This is the narrowest causal response to Cycle 010.

---

# CYCLE 011 — one-hour Codex task

## Goal

Test one training-free **Memory Spatial Prompt (MSP)**: keep the entire degraded observation unchanged, but pass the supplied miner-memory bbox as an explicit SAM spatial box prompt together with the existing text/REF prompt when decoding that identity's helmet.

This tests the hypothesis:

> identity memory should tell the pixel decoder **where to look**, while the full-frame visual context remains exactly the distribution w15 expects.

No training, no crop/zoom, no enhancer, no adaptation, no controller/RL.

## A. Minimal full-context spatial-prompt implementation

Inspect the local SAM `PromptEncoder` signature first. If the existing fork supports `boxes` together with `text_embeds`, add one optional evaluator/model path only.

For each supplied identity memory:

1. keep the degraded full-frame SAM image tensor and CLIP image tensor exactly unchanged from `D-full`;
2. take only that identity's supplied miner bbox; no helmet GT, predicted helmet, IoU, or result-dependent geometry;
3. transform the miner bbox into the coordinate system expected by SAM prompt encoding using the same `ResizeLongestSide` geometry already used for the full image;
4. pass that bbox to `visual_model.prompt_encoder(..., boxes=..., text_embeds=...)` for the mask decode while retaining the existing REF/text prompt;
5. keep all model weights frozen and keep the existing output/postprocessing path.

For `v1_multiround`, preserve the conversation and REF semantics. If the prompt encoder requires a box per decoded prompt row, replicate the same remembered-miner box across that identity's relevant segmentation rows rather than inventing a new target box. If simultaneous box+text prompting is not supported by the local fork without an architectural change, **stop and report that incompatibility**; do not redesign the prompt encoder in this cycle.

Prefer a small optional argument such as `spatial_memory_boxes_list` rather than changing default behavior. The disabled path must remain baseline-compatible.

## B. Required software/regression checks before the GPU run

Add focused tests/receipts for:

- original-pixel bbox -> SAM resized-input bbox transform, including border cases;
- `D-full` and MSP use identical full-frame image tensors / degradation seed;
- changing helmet target-mask bytes does not change the MSP box or any model input;
- with MSP disabled, one saved real group reproduces the existing full-frame prediction exactly;
- no MGR crop tensor enters the MSP path.

Do not modify `memory_reobservation.py` except documentation/default-off cleanup if needed.

## C. One fixed validation50 comparison

Use the frozen Cycle006 validation50, base w15, `target15_b`, seed 0, supplied identity memory, and the same CMF scorer. Run exactly one new MSP condition:

- `D-full`: reuse the existing base full-frame result;
- `D-MSP`: full degraded frame + remembered-miner SAM box prompt.

Report:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean/median identity margin;
- IER;
- CMSA `fail->pass / pass->fail / pass->pass / fail->fail` transitions.

Also compute an **analysis-only GT oracle union** between `D-full` and `D-MSP`, clearly labeled non-deployable, only to measure whether the new action has enough per-group complementarity to justify Layer-2 scheduling later.

## Predeclared decision rule

### Keep MSP as a Layer-1 recovery mechanism only if

- degraded target mIoU improves by at least `+0.02`;
- degraded CMSA does not decrease;
- at least `3` failed groups become CMSA successes;
- at most `1` success becomes a failure.

If this passes, freeze the mechanism and **stop Layer-1 architecture search**. The next cycle should immediately build the minimal oracle-free feedback/scheduler between ordinary full-frame decoding and MSP.

### If the direct gate fails

Do not tune box expansion, prompt weights, thresholds, or another spatial heuristic. Use the oracle union only as a research diagnostic:

- if the union improves full-frame mIoU by at least `+0.03` and creates at least `5` CMSA fail->pass opportunities, there is enough action complementarity to justify a later oracle-free scheduler despite average MSP weakness;
- otherwise retire training-free spatial-focus variants. The next Layer-1 attempt, if any, should be **one tiny learned full-context dual-scale residual feature adapter**, not another crop/prompt sweep.

Do not touch diagnostic30 or confirmation30 in this cycle.

## Non-goals

- no crop-scale sweep or MGR revival;
- no new enhancer;
- no w15 adaptation or loss-weight sweep;
- no new degradation family;
- no predicted-memory protocol;
- no agent/controller/verifier/RL training yet;
- no baseline ports;
- no GT-based runtime prediction selection.

## Deliverable

Before appending `CODEX UPDATE 011`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains the canonical handoff. Then append `CODEX UPDATE 011` with:

1. exact spatial-box plumbing and coordinate convention;
2. tests/regression receipts;
3. fixed validation50 `D-full` vs `D-MSP` table;
4. transition counts and analysis-only oracle union;
5. direct gate pass/fail and action-complementarity judgment;
6. exactly one recommended next one-hour task.

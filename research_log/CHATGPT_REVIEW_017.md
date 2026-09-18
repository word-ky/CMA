# CHATGPT REVIEW 017 — wiring is now trustworthy; run the frozen full SegLLM val50 comparison and stop touching the interface

Reviewed commit: `fbc4626d9cb25339ff0b2284ce121148bc1d368c`.

## Decision

Cycle 017 is accepted as a **successful weight-faithful wiring validation**. It materially advances the paper-evidence path because SegLLM is now a runnable native historical-memory competitor rather than a source-audit-only candidate.

The narrow loader repair is scientifically acceptable:

- native `from_pretrained` still exposes exactly the diagnosed `gamma_l/gamma_v -> weight_l/weight_v` compatibility collision;
- the helper restores only those two named tensors from the pinned safetensors checkpoint;
- both restored tensors exactly equal the released checkpoint after the intended BF16/device cast;
- `tgt_embed.weight[100,256]` independently equals the outer checkpoint without helper assignment, closing the earlier 900-vs-100 ancillary warning for the final executed tensor;
- no architecture, checkpoint, prompt, threshold, SegLLM revision, HIPIE implementation, or CMA model was changed.

The four frozen trials also provide the runtime proof we needed: `[REF:1]` resolves to history index 0; clean/degraded A/B share the same current image/text while the supplied miner state changes by identity; the actual MASK-ENCODE/BOX-ENCODE tensors match the intended supplied state; output index 1 is tied to the final relational MASK-DECODE; and four original-resolution binary masks were exported without helmet-target access before prediction freeze.

This is still **not a performance result**. The smoke has zero scores and only one group, so it supports interface validity, not superiority or degradation robustness.

## Critical research/engineering feedback

### 1. Freeze the SegLLM interface now

Do not make any more changes to:

- prompt wording;
- `[REF:1]` construction;
- supplied-memory representation;
- predicted-score selection;
- thresholding;
- output ordering;
- checkpoint/source/dependency versions;
- two-gamma compatibility shim.

The current interface has crossed the important threshold from plausible to auditable. Any further result-driven edit would weaken the baseline comparison.

### 2. Keep the scientific claim precise: this is supplied-memory use, not end-to-end memory writing

For the full comparison, SegLLM is seeded with the same externally supplied miner identity state used by the CMA supplied-memory protocol. That is exactly the fair comparison we want for **identity-conditioned memory use**.

It does **not** establish that SegLLM can write a stable miner identity memory from its own first-round prediction under degradation. Do not describe the full val50 result as an end-to-end memory-write comparison. The paper should keep the two questions separate:

1. given the correct entity memory, can the method use it to select the correct relational target?
2. can the system itself form/maintain that memory under degradation?

Cycle 018 answers only the first question against the strongest direct historical-memory baseline currently available.

### 3. Run the full val50 before any new agent work

This is now the highest-value next experiment. We already know internal ablations can produce modest gains, but the paper still lacks the external number that tells us whether the counterfactual identity-memory contribution is actually competitive.

The existing val50 is reconstructed development/compatibility evidence and has been used during CMA method selection, so it must not be sold as untouched final generalization. That limitation is acceptable here: the immediate purpose is to determine whether the Layer-1 story is strong enough to freeze and move to the agent layer.

### 4. Preserve strict target-free prediction freezing

For all 50 groups, generate and save every SegLLM prediction before the CMF scorer opens any helmet target. The full runner may read only:

- current clean/degraded images;
- supplied miner masks/bboxes / prepared memory state;
- frozen configuration/manifests/checkpoints.

Helmet targets must be scorer-only artifacts after prediction generation is complete. A crash may resume from deterministically predeclared missing trials, but never rerun or replace a trial based on prediction appearance or score.

### 5. Compare against CMA on the exact same manifest, not remembered headline numbers alone

The final table must prove identical group IDs, identity ordering, clean/`target15_b` condition, target definitions and scorer conventions. Prefer recomputing CMA metrics from already frozen saved predictions on this same val50 manifest. If those predictions are unavailable and previously recorded CMA metrics must be reused, include their artifact/manifest hashes and state explicitly that no CMA rerun or tuning occurred after seeing SegLLM.

Report at minimum, separately for clean and degraded:

- target mIoU;
- CMSA (`x/50` and rate);
- Memory Fidelity (`x/100` and rate);
- IER;
- mean and median identity margin;
- empty-mask count;
- paired clean->degraded deltas.

Also save per-group IoU matrices so any identity-swap versus generic-visibility failure can be audited later without rerunning the baseline.

### 6. Do not tune to the outcome

Whatever SegLLM scores, accept the first frozen full result.

- If CMA is clearly stronger on degraded mIoU **and** CMSA/identity margin, freeze this external baseline result and move next to the minimal oracle-free agent feedback layer.
- If SegLLM is comparable or better, do **not** paraphrase prompts, alter thresholds, or try another SegLLM configuration. The next task should be a bounded paired error analysis to identify whether CMA loses on category visibility, identity binding, or relation localization.

That decision rule prevents the baseline from becoming another hidden development loop.

---

# CYCLE 018 — one focused hour

## Goal

Run the **first full frozen external historical-memory comparison**: pinned SegLLM versus frozen CMA base w15 on the exact validation50 manifest, for clean and `target15_b`, with no interface/model tuning.

## Step 1 — freeze the execution manifest before scoring

Create one manifest/receipt containing exactly the existing 50 group IDs in frozen order and, for each group:

- two entity IDs in frozen order;
- clean image path/hash;
- `target15_b` image path/hash;
- supplied miner mask/bbox identity provenance/hash;
- prompt hash;
- checkpoint/source revision hashes;
- scorer version/hash.

Assert that this is the same val50 used by the existing CMA evaluation. Expected SegLLM workload is exactly `50 groups x 2 conditions x 2 identities = 200` relational forwards.

## Step 2 — run SegLLM with the Cycle017 interface unchanged

Reuse the exact Cycle017 loader and runtime:

- pinned SegLLM/HIPIE/checkpoint/dependency revisions;
- exact two-gamma compatibility restoration with fidelity assertion at process load;
- frozen prompt `Segment the mining helmet worn by instance 1.[REF:1]`;
- same condition-matched supplied miner appearance + supplied bbox;
- same native output selection/threshold/postprocessing;
- same original-resolution mapping.

Do not add a predicted-miner first round. Do not change prompt text or threshold for any group.

Save all 200 binary predictions and per-trial receipts before opening any helmet target.

## Step 3 — score only after prediction freeze

After all SegLLM predictions are frozen, run the unchanged model-agnostic CMF scorer on the full set.

Export:

- aggregate clean and degraded metrics;
- per-group IoU matrices and identity margins;
- CMSA/Fidelity/IER decisions;
- clean->degraded paired deltas;
- empty-mask count and any invalid-output count;
- runtime/VRAM summary labelled as engineering metadata, not a controlled efficiency benchmark.

No score-driven rerun is allowed.

## Step 4 — place CMA and SegLLM in one exact-protocol table

Use frozen CMA base-w15 predictions on the same manifest if available; otherwise use the already recorded frozen metrics only with explicit manifest/artifact provenance. Do not retrain or adapt CMA.

Produce one concise comparison table:

| Method | Clean mIoU | Clean CMSA | Clean Fidelity | Degraded mIoU | Degraded CMSA | Degraded Fidelity | Degraded mean margin | IER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Add a short paired interpretation that distinguishes generic segmentation collapse from identity-switch failure. Do not claim final generalization from val50.

## Stop conditions

Stop and report an exact blocker if:

- final weight-fidelity assertions fail;
- the 50-group manifest cannot be matched exactly to CMA val50;
- a trial cannot be tied unambiguously to its supplied identity state;
- output-to-original-image mapping changes from Cycle017;
- repeated OOM/runtime failure prevents completion.

Do not change prompt, threshold, checkpoint, model revision, memory representation, CMA weights, or scorer to get around a bad result.

## Non-goals

No CMA training/adaptation. No new degradation family. No second external baseline. No prompt/threshold sweep. No MG-DRA/enhancer work. No agent/controller/verifier/RL in this cycle. No diagnostic30/confirmation30 reuse.

## Deliverable

Append `CODEX UPDATE 018` with:

1. frozen val50 manifest receipt;
2. completed/failed SegLLM forward count (expected 200);
3. target-free prediction-freeze audit;
4. full clean/degraded CMF metrics and per-group artifacts;
5. exact-protocol CMA-vs-SegLLM table;
6. one bounded conclusion: either **Layer-1 external evidence is strong enough to freeze and move to agent feedback**, or **external comparison exposes a specific Layer-1 weakness requiring one targeted error-analysis cycle**.

Before appending `CODEX UPDATE 018`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical.

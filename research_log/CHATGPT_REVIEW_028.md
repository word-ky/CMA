# CHATGPT REVIEW 028 — implementation-grounded method draft accepted; next lock the paper around supplied identity-localized memory

Reviewed commit: `70c9c1cbc5ca71716fb388914bb5215f3b0b41f8`.

## Decision

Cycle 028 is accepted. It correctly remained inside the research stop: the commit adds manuscript/traceability artifacts only, with no scientific-code edits, model/scorer execution, metric recomputation, new sample selection, baseline change, or Layer-2 resurrection. The comparison from `73f07c7` to `70c9c1c` contains only the bridge plus new `research_log/cycle028/*` writing/receipt files.

The main advance is conceptual clarity. The paper can now describe a real implemented method rather than leaving the impression that the contribution is only a counterfactual evaluation protocol:

1. input-side `[REF]` conditioning from reference appearance plus bbox;
2. output-side REF hidden state plus supplied mask/bbox geometry injected into the SAM text prompt;
3. paired counterfactual identity ranking during training;
4. counterfactual switch evaluation with CMSA/Fidelity/IER/margin.

The source inspection supports these statements. `build_ref_input_embeddings(...)` produces the crop+bbox reference vector; `_replace_ref_input_embeddings(...)` implements `gated_add`; `build_ref_prompt_embeddings(...)` combines REF hidden state with downsampled mask+bbox geometry; the SAM branch adds this context to the SEG-derived prompt; and the rank term is the diagonal-vs-maximum-off-diagonal soft-IoU hinge. The w15 config values recorded in the draft (`ref_input_scale=0.5`, prompt mode `add`, mask pool 16, reconstruction weight 1.0, rank weight 1.5, rank margin 0.05) are tied to the checkpoint config hash used by the frozen primary evaluation.

No component is independently ablated, so the draft is right to treat these as the implemented CMA design and the Cycle025 result as whole-system evidence rather than assigning a measured gain to any one mechanism.

## What is now strong enough for the paper

### 1. The method story is implementation-grounded

The new traceability map is useful and should remain the controlling source for Method wording. In particular, it correctly distinguishes:

- configured scalar `gated_add` from a learned reliability gate;
- the learned REF prompt scale from its initialization value;
- binary offline identity metrics from the differentiable training soft-IoU objective;
- the frozen base-w15 inference path from later experimental branches present in the repository;
- method-native CMA/SegLLM memory encodings from the common underlying observation/query/identity information.

### 2. The metric definitions are faithful

The Method matches the frozen evaluator: Fidelity is strict diagonal dominance; CMSA quality-qualifies both members of a pair at IoU >= 0.5; IER requires a stronger wrong-identity IoU >= 0.5; and identity margin is correct-IoU minus strongest-wrong-IoU. Fidelity and IER are therefore not complements. The primary set has two identities per group, so the pairwise CMSA definition coincides with per-group two-identity success.

### 3. The scope boundary remains scientifically disciplined

The draft explicitly says the validated evidence is supplied-memory use, not autonomous memory writing/persistence/repair or a successful inference-time controller. That is essential. Cycles019–021 remain negative/partial evidence and must not be promoted into a second headline contribution.

## Critical paper-level corrections before full integration

### 1. Make the strength of the supplied memory explicit everywhere

This is now the largest review risk.

CMA does **not** receive only an identity name or compact embedding. At inference it receives a supplied miner mask and bbox, uses them to form the reference appearance crop, and uses the mask+bbox again in the output-side prompt. This is intentionally a strong identity-localized entity state, but it also supplies precise spatial information about the worker.

Therefore the paper/figure/abstract should use wording such as:

> **supplied identity-localized entity memory (appearance crop + miner mask + bbox)**

and should not imply autonomous person re-identification, tracking, memory acquisition, or recovery from an unknown worker location. The strongest claim is relational small-target segmentation **conditioned on a supplied localized entity state**.

A reviewer should be able to see this input contract from Figure 1 without reading the limitations.

### 2. Separate the broader mine motivation from the actual stressor

The Introduction currently mentions darkness, dust, blur and glare in one sentence. The frozen `target15_b` evaluation, however, is exactly a deterministic combination of gamma darkening, contrast/scale reduction, Gaussian noise and Gaussian blur. It does not synthesize dust or glare.

It is fine to mention dust/glare as broader application motivation once citations are added, but every empirical sentence must call the evaluated condition a **fixed compound photometric/noise/blur stressor** (or list its actual components). Do not write as if Cycle025 experimentally covers dust, glare, occlusion, or the full underground corruption distribution.

### 3. Do not accidentally turn same-frame reference appearance into “clean historical memory”

For Cycle025, the reference appearance crop is taken from the same condition image as the main observation, while the supplied miner mask/bbox remains fixed. Thus under `target15_b`, the reference appearance is degraded too. The draft already states this in Method; preserve it in the figure and abstract.

The paper may discuss entity/historical memory as a system concept, but the validated protocol is a **supplied same-condition identity reference with fixed geometry**, not an independently captured clean historical frame.

### 4. Keep the training provenance qualifier attached to the objective

The w15 checkpoint/config supports the reconstruction/rank settings, and historical logs identify the 1.5 rank branch, but the exact run-bound training manifest and some CE/BCE/Dice overrides remain unknown. Do not upgrade config traceability into a claim that every historical training detail is fully reconstructed. The current `UNKNOWN` fields should survive into reproducibility/limitations.

### 5. Avoid “agentic” as the paper's scientific headline

The validated result is memory-conditioned relational perception. The failed Layer-2 controller/verifier search is useful internal evidence that identity consistency is not equivalent to localization correctness, but it is not a successful agent contribution. A title/abstract centered on “agentic scheduling/controller” would now weaken the paper.

## Research stop remains binding

Do not run another model/scorer, add an ablation, significance test, corruption, baseline, sample, qualitative case, threshold, verifier, controller, Qwen/RL component, or data split. No result-driven rescue is authorized.

---

# CYCLE 029 — one focused hour

## Goal

Create a **submission-shell integration pass** that makes the frozen CMA contribution understandable in one page: lock terminology/input contract, draft a bounded title+abstract, render one method figure from the verified mechanism, and compile the existing Introduction/Method/Results/Limitations without changing scientific evidence.

No empirical work.

## A. Freeze paper terminology and input contract

Create `research_log/cycle029/PAPER_POSITIONING.md` with a small controlled vocabulary used by all following artifacts:

- primary validated object: **supplied identity-localized entity memory**;
- memory contents: reference appearance crop + supplied miner mask + bbox;
- target: the helmet relationally associated with that miner;
- intervention: same observation/query, switch only the supplied identity memory;
- evaluated degradation: exact `target15_b` photometric/noise/blur compound stressor (gamma/contrast/scale/noise/blur), not dust/glare/occlusion simulation;
- evidence status: `DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION`, reconstructed pseudo targets, exact historical exposure still unknown;
- comparator status: pinned SegLLM system-level comparator through its native memory interface;
- explicit non-claims: autonomous memory acquisition/writing/tracking, training-unseen proof, lower degradation sensitivity, architecture-only superiority, successful Layer-2 agent/controller, safety guarantee.

Audit all new title/abstract/figure labels against this vocabulary.

## B. Draft one working title and one abstract

Create `research_log/cycle029/TITLE_ABSTRACT_LATEX.tex`.

Requirements:

- one working title, not a list of variants;
- abstract <= 190 words;
- problem -> method -> counterfactual evaluation -> primary Cycle025 result -> limitation boundary;
- make the supplied mask/bbox memory contract visible rather than hiding it behind “identity memory”;
- use at most two headline numerical comparisons in the abstract; CMSA/identity fidelity is more distinctive than listing every metric;
- do not say dust/glare are experimentally evaluated;
- do not call the method a successful autonomous agent.

If a coal-mine application term is in the title, the title must still expose the general scientific problem (identity-conditioned relational segmentation), not read as a domain-only engineering system.

## C. Produce one main method-figure specification and vector rendering

Create `research_log/cycle029/METHOD_FIGURE_SPEC.md` and a source-controlled vector figure (`METHOD_OVERVIEW.svg` or TikZ/PDF) using only the verified mechanisms.

The figure must show, left-to-right:

1. the same scene and same relational query;
2. two counterfactual memory choices A/B, each explicitly containing **miner mask+bbox+appearance crop**;
3. input-side `[REF]` injection into the language model;
4. output-side REF hidden state + mask/bbox geometry fused into the SAM prompt;
5. helmet A vs helmet B output;
6. a small training inset for the diagonal-vs-off-diagonal counterfactual rank objective;
7. a small evaluation inset showing the IoU matrix intuition behind Fidelity/CMSA/IER.

Do not draw an autonomous memory writer, controller loop, Qwen/RL policy, enhancer, MGR/MSP/MG-DRA, or any component not in the frozen primary path.

The figure should be readable at two-column width and should make the supplied spatial memory assumption visually unavoidable.

## D. Compile a single manuscript shell

Create `research_log/cycle029/PAPER_SHELL.tex` that integrates, without scientifically rewriting:

- Cycle029 title/abstract;
- Cycle028 Introduction;
- Cycle028 Method;
- Cycle027 main Results table + qualitative figure;
- Cycle027 Results/Limitations prose;
- the new method figure.

Citation comments/placeholders may remain if no verified bibliography exists. Do not invent citations or novelty-priority claims in this cycle.

Perform only LaTeX/layout fixes. No result values may change.

## E. Final consistency audit

Create `research_log/cycle029/PAPER_SHELL_AUDIT.md` checking:

- every numerical value against Cycle025/Cycle027 frozen sources;
- every method label against Cycle028 traceability;
- the memory input contract appears in abstract, method figure and Method;
- `target15_b` wording matches its actual operations;
- the title/abstract do not resurrect Layer-2;
- no prohibited claim listed in `PAPER_POSITIONING.md` appears affirmatively;
- no new scientific selection or model/scorer execution occurred.

## Hard non-goals

- no inference/training/scoring;
- no new samples/qualitative selection;
- no new metrics/table values;
- no new baseline or component;
- no literature novelty search in this cycle;
- no title variants or result-driven marketing sweep;
- no modification of Cycle025 frozen evidence.

## Deliverable

Before `CODEX UPDATE 029`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 029` with:

1. `PAPER_POSITIONING.md`;
2. `TITLE_ABSTRACT_LATEX.tex`;
3. `METHOD_FIGURE_SPEC.md` + vector figure source/output;
4. `PAPER_SHELL.tex` (+ compiled PDF if local LaTeX is available);
5. `PAPER_SHELL_AUDIT.md`;
6. confirmation of zero model/scorer calls and zero new scientific selection;
7. exactly one next recommendation limited to verified bibliography/Related Work integration or final manuscript polish.
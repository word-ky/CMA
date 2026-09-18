# CHATGPT REVIEW 011 — MSP sharpens identity binding but harms target extent; switch to a zero-init residual local-evidence path

Reviewed commit: `ad34adcb84092498f25d9b7dacd35dd1c4cc4713`.

## Decision

Cycle 011 is accepted as a clean negative result and closes the training-free spatial-focus branch. The implementation is appropriately bounded: the full degraded observation is unchanged, the miner-memory bbox is the only added spatial signal, the same base w15 is frozen, target labels do not enter inference, disabled behavior reproduces the baseline, and the fixed validation50 result fails both the direct mechanism gate and the predeclared action-complementarity gate.

The decisive numbers are:

- target mIoU: `0.691113 -> 0.646997` (`-0.044116`);
- CMSA: `29/50 -> 26/50`;
- Memory Fidelity: `89/100 -> 94/100`;
- mean identity margin: `0.662931 -> 0.627106`;
- IER: `3/100 -> 2/100`;
- CMSA transitions: `2 fail->pass`, `5 pass->fail`;
- GT oracle union: mIoU `0.712135` (`+0.021022`) and CMSA `31/50`, below the predeclared complementarity threshold.

Therefore MSP is not a usable Layer-1 component and there is not enough full/MSP action complementarity to justify Layer-2 scheduling around it.

## Critical interpretation

The most informative part of Cycle 011 is the **metric split**: explicit miner geometry improves identity ranking/Fidelity while worsening target overlap and pairwise CMSA. This suggests the current identity-memory representation is not the main failure. The spatial intervention is instead changing how the SAM decoder interprets the target.

A likely mechanism is semantic mismatch in SAM prompting: a box prompt is normally an object-conditioning cue for the object to be segmented. Here the box describes the **miner**, while the requested segmentation target is the miner's **helmet**. Concatenating the miner box tokens with the existing REF/text tokens can therefore strengthen worker identity localization while biasing the decoder toward the wrong spatial extent for the relational target. The observed `Fidelity up / mIoU down` pattern is consistent with this interpretation, though it is not by itself causal proof.

This matters for the next design: memory geometry should act as a **support region for extra visual evidence**, not as a direct SAM object prompt, and local evidence must not replace the full-frame representation. Cycle 010 already showed that crop-only decoding destroys context/scale compatibility; Cycle 011 shows that direct miner-box prompting over-constrains the relational target. The remaining narrow hypothesis is:

> keep the exact full-frame w15 path as the anchor, and let a tiny learned residual inject high-resolution evidence from the remembered worker region without changing the base observation or treating the worker box as the target object.

This is the last justified Layer-1 architecture attempt before we stop local-focus design search.

---

# CYCLE 012 — one-hour Codex task

## Goal

Implement and evaluate exactly one **Memory-Guided Dual-Scale Residual Adapter (MG-DRA)**. It must preserve the full-frame base path exactly at initialization and use the remembered miner ROI only as an auxiliary high-resolution evidence source for the helmet row.

No crop-only prediction, no SAM box prompt, no enhancer, no broad w15 unfreezing, no agent/controller/RL, and no hyperparameter sweep.

## A. Minimal architecture

Use the existing base w15 and freeze **all existing model parameters**: LLM, CLIP, SAM image encoder, prompt encoder, mask decoder, REF projections and all previously adapted surfaces.

For each supplied identity memory in `v1_multiround`:

1. compute the normal full-frame SAM image embedding `E_global` exactly as base w15 does;
2. use the same deterministic `1.25x` miner-memory ROI geometry already validated in Cycle 010, but **do not decode from the crop**;
3. crop the same clean/degraded observation for that miner, resize through the normal SAM image preprocessing, and run the **frozen** SAM image encoder under `no_grad` to obtain a local embedding `E_local`;
4. map `E_local` back onto the corresponding ROI region of the global SAM feature grid by bilinear resize; outside the ROI it is zero. Also create a binary ROI gate `G` on the same feature grid;
5. build one tiny shared residual adapter on the feature grid:

   `R = Up(GELU(Down(concat(E_global, E_local_mapped, G))))`

   where `Down` and `Up` are `1x1` convolutions with bottleneck width `16`;
6. zero-initialize the final `Up` weights/bias and use one learnable scalar `alpha`, initialized to `0`; fused features are

   `E_fused = E_global + alpha * G * R`;
7. use `E_fused` **only for the helmet target row of that identity**. Miner rows remain on the unchanged `E_global` path. Existing REF/text prompts remain unchanged and `spatial_memory_boxes_list` stays disabled.

The adapter must be identity-specific at inference: miner A's local evidence may affect helmet A's decode only, and miner B's local evidence may affect helmet B's decode only.

Parameter budget: adapter + scalar only; target **< 100k trainable parameters**. If exact channel dimensions make the above exceed 100k, reduce bottleneck width deterministically to fit; do not redesign the adapter.

### Required architectural invariants

- with `alpha=0` / zero-init adapter, one real group must reproduce base full-frame logits or masks exactly within numerical tolerance;
- full-frame SAM/CLIP tensors and existing REF/text embeddings are unchanged;
- no helmet GT, predicted helmet, GT IoU or result-dependent geometry is used to build the ROI or residual;
- local SAM image-encoder features are frozen/no-grad; gradients update only adapter parameters and `alpha`;
- no MSP box tokens enter the prompt encoder.

If the current mask-decoder batching cannot accept identity-specific fused image embeddings cleanly, decode the two helmet identity rows separately while reusing the same full-frame prompt/text state. Do not modify the SAM architecture to force batching.

## B. One fixed training contract

Reuse the exact frozen Cycle006 `train300 / val50` split, assets, ordering and seed. Do not rebuild the split and do not use diagnostic30/confirmation30.

Train exactly one candidate from base w15 for one epoch / 300 groups. Use paired clean and `target15_b` observations and the existing supplied identity memories.

Only the adapter + `alpha` are trainable. Optimize:

`L = 0.5*(L_seg_clean + L_seg_deg) + 0.5*(L_rank_clean + L_rank_deg)`

where segmentation is the existing helmet BCE+Dice and ranking is the existing counterfactual soft-IoU rank loss with margin `0.05`.

Do **not** add the previous consistency loss, teacher loss, restoration loss, new regularizer, or auxiliary prompt loss in this cycle.

Predeclare:

- AdamW;
- learning rate `1e-4`;
- weight decay `1e-4`;
- gradient clip norm `1.0`;
- one epoch / exactly 300 groups;
- seed `20260528`;
- one final checkpoint only;
- no restart, LR change, bottleneck sweep, ROI-scale sweep, or loss-weight sweep.

Before the full run, execute one fixed-batch smoke backward and record: adapter/alpha finite non-zero gradients, all base-w15 gradients absent, and exact zero-init baseline reproduction before optimization.

## C. Fixed validation50 evaluation

Evaluate base w15 and the single MG-DRA checkpoint on the same val50 under both clean and `target15_b`, with no enhancer and no MSP.

Report:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean/median identity margin;
- IER;
- CMSA transitions for degraded base -> MG-DRA;
- trainable parameter count;
- inference cost: number of additional frozen SAM image-encoder calls per identity/group.

Also compute an **analysis-only** whole-group oracle union between base and MG-DRA on degraded val50 to quantify action complementarity; never use it as runtime performance or model selection.

## Predeclared gate

Keep MG-DRA as the Layer-1 recovery mechanism only if:

- degraded target mIoU improves by at least `+0.02` over base;
- degraded CMSA does not decrease;
- at least `3` degraded CMSA fail->pass groups;
- at most `1` pass->fail group;
- clean mIoU drops by no more than `0.01`;
- clean CMSA drops by at most `1/50` group.

If it passes, **freeze Layer-1 architecture work immediately**. The next cycle should lock a fresh confirmation set and then move to the minimal oracle-free agent feedback/scheduling contribution.

If it fails, do not tune this adapter, ROI scale, bottleneck, LR or losses. Report the oracle union only as a diagnostic. Retire local-focus architecture search and reassess the paper around the already-validated identity-memory mechanism plus comparative degradation robustness/baselines rather than inventing another module.

## Non-goals

- no crop-only MGR revival;
- no SAM spatial box prompt;
- no enhancer;
- no 73M-parameter w15 adaptation;
- no SAM image-encoder training;
- no new degradation family;
- no predicted-memory protocol;
- no agent/controller/verifier/RL yet;
- no baseline ports in this cycle;
- no GT-based runtime selection.

## Deliverable

Before appending `CODEX UPDATE 012`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 012` with: exact adapter insertion and parameter count; zero-init/gradient receipts; one training command/run; base-vs-MG-DRA clean/degraded val50 table; degraded transition/oracle-union analysis; gate pass/fail; and exactly one recommended next one-hour task.

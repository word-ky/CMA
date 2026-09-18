# CHATGPT REVIEW 012 — blocker is valid; correct the zero-residual initialization and finish the single MG-DRA test

Reviewed commit: `87ee1bf58cc0724316e7c2a74041203d462b7da2`.

## Decision

Cycle 012 is accepted as a **protocol-correctness stop**, not as a model failure. Codex was right not to silently change the requested architecture or launch a meaningless 300-step run. The original contract simultaneously required `Up=0`, `alpha=0`, exact baseline reproduction, and non-zero adapter gradients. Those requirements are mutually incompatible for

`E_fused = E_global + alpha * G * Up(GELU(Down(...)))`.

With both `Up` and `alpha` zero, `R=0`, `dL/dalpha=0`, `dL/dUp=0`, and `dL/dDown=0` for any downstream differentiable loss. The isolated implementation and regression test reproduce exactly that stationary point, and the 12,577-parameter count is within the intended budget. This is a specification error in the previous ChatGPT task, not evidence against the MG-DRA hypothesis.

The engineering behavior also advances the memory-centric plan in an important way: it preserves experimental validity. A silent initialization change or a dead 300-step run would have produced an uninterpretable result. No performance claim should be made from Cycle 012 because production integration, real-data gradient flow, training, and validation were correctly not run.

## Corrected initialization contract

Keep the architecture unchanged except for one initialization correction:

- zero-initialize the final `Up` weights and bias exactly as before;
- initialize the learnable scalar `alpha = 1.0`, not `0.0`;
- keep `Down` at the normal deterministic PyTorch initialization under seed `20260528`;
- retain `E_fused = E_global + alpha * G * R`.

This still gives **exact zero residual / baseline equality at initialization** because `Up(...) = 0`, while allowing the residual branch to begin learning.

The expected gradient sequence is now explicit and should be treated as a correctness property rather than a surprise:

1. **initial backward before any optimizer step:** `Up` must receive finite non-zero task gradients; `Down` and `alpha` are allowed (and expected) to have zero gradients because the zero `Up` still blocks them;
2. take exactly one smoke optimizer step on the isolated smoke state;
3. run a second forward/backward on the same fixed real batch: the residual is now non-zero, so `Down` must receive finite non-zero gradients, and `alpha` should receive a finite gradient (normally non-zero on this real task batch);
4. verify every frozen base-w15 parameter remains gradient-free in both backwards;
5. **discard the smoke optimizer/model state completely** and reinitialize the actual candidate from pristine base w15 + pristine adapter (`Up=0`, `alpha=1`) before the 300-group run.

Keep the existing double-zero regression test as evidence for the old blocker, and add a new corrected-initialization test instead of rewriting history.

## Research/engineering constraints that remain unchanged

Do not redesign the adapter because of this blocker. The scientific hypothesis remains the same and is still the last justified local-focus Layer-1 attempt:

> full-frame context remains the anchor; miner identity memory only supplies an auxiliary high-resolution evidence region; the learned residual may affect that identity's helmet decode, but must never replace the full observation or masquerade the miner box as the helmet target.

All original Cycle-012 constraints remain in force:

- base w15 fully frozen;
- frozen/no-grad SAM image encoder for local ROI features;
- same deterministic 1.25x miner-memory ROI;
- same `train300 / val50` split and seed;
- adapter + scalar only trainable, target `<100k` parameters;
- helmet rows use identity-specific fused features; miner rows remain on `E_global`;
- no MSP box tokens, enhancer, broad w15 adaptation, agent/controller/RL, new degradation, predicted-memory protocol, or hyperparameter sweep;
- one epoch / exactly 300 groups, AdamW `lr=1e-4`, weight decay `1e-4`, clip norm `1.0`;
- same paired clean/degraded segmentation + counterfactual rank objective, with no consistency/teacher/restoration auxiliary losses.

One additional implementation receipt is required because the adapter is not yet connected to production: record the exact tensor shapes and coordinate mapping for `E_local -> E_local_mapped -> G`, and verify on one real group that miner A's local feature can affect **helmet A only**, while miner B's feature can affect **helmet B only**. Changing A's local tensor must not change B's pre-threshold helmet logits when all shared/global inputs are held fixed. This is an identity-routing regression test, not a new metric.

---

# CYCLE 013 — one-hour Codex task

## Goal

Resume and complete the **single corrected MG-DRA experiment**. Do not start a new research branch. The only protocol change from Cycle 012 is `alpha: 0 -> 1` at initialization plus the corrected two-stage smoke expectation above.

## A. Correct the isolated adapter and smoke tests first

1. Change `MemoryDualScaleResidualAdapter.alpha` initialization to `1.0`.
2. Preserve zero-initialized `Up` so pre-optimization output exactly equals `E_global`.
3. Add/record:
   - exact baseline equality before optimization;
   - first backward: finite non-zero `Up` gradient, frozen-base gradients absent; `Down/alpha` may be zero;
   - one smoke optimizer step;
   - second backward: finite non-zero `Down` gradient and finite `alpha` gradient; record whether alpha is non-zero rather than forcing a fabricated claim if the task batch happens to make it zero;
   - discard the smoke state afterward.
4. Keep the old double-zero regression as a historical blocker test.

If `Up` has zero/non-finite gradient on the fixed **real** smoke batch after this correction, stop and report a production gradient-path blocker. Do not alter loss weights, initialization scale, or ROI geometry to force a pass.

## B. Integrate exactly the already-specified production path

Implement only what Cycle 012 had not yet reached:

- normal full-frame `E_global` from base w15;
- deterministic 1.25x memory ROI from supplied miner geometry;
- frozen/no-grad local SAM encoder call -> `E_local`;
- bilinear mapping onto the corresponding global feature-grid ROI plus binary gate `G`;
- shared 1x1 bottleneck-16 residual adapter;
- `E_fused` used only for the corresponding helmet identity row;
- miner rows and all other model paths unchanged;
- no SAM spatial box prompt.

Record trainable count, expected to remain 12,577 unless production channel dimensions prove different; if dimensions differ, apply the already-predeclared deterministic bottleneck reduction only if needed to stay below 100k.

Before training, run the identity-routing regression: perturb/zero only A's mapped local feature and verify B's helmet logits are unchanged (within numerical tolerance), and vice versa.

## C. Run the single prescribed candidate

Reuse exact Cycle006 `train300 / val50`; do not rebuild assets or inspect diagnostic30/confirmation30.

Train once from pristine base w15 + pristine corrected adapter:

`L = 0.5*(L_seg_clean + L_seg_deg) + 0.5*(L_rank_clean + L_rank_deg)`

with rank margin `0.05`, AdamW `lr=1e-4`, weight decay `1e-4`, clip norm `1.0`, seed `20260528`, one epoch / exactly 300 groups. No restart, no sweep, no intermediate model selection.

## D. Fixed validation50 decision

Compare base w15 vs the single MG-DRA checkpoint on clean and `target15_b` and report:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean/median identity margin;
- IER;
- degraded CMSA transition counts;
- analysis-only whole-group oracle union;
- trainable parameter count;
- inference cost as extra frozen SAM image-encoder calls per group/identity.

The original gate is unchanged. Keep MG-DRA only if:

- degraded target mIoU improves by at least `+0.02`;
- degraded CMSA does not decrease;
- at least `3` degraded fail->pass groups;
- at most `1` pass->fail group;
- clean mIoU drops by no more than `0.01`;
- clean CMSA drops by at most `1/50`.

If it passes, freeze Layer-1 architecture work immediately. The next cycle should lock a fresh confirmation set before any agent work, then proceed to minimal oracle-free agent scheduling only after confirmation.

If it fails, **retire local-focus architecture search permanently**. Do not tune alpha, bottleneck, ROI scale, LR, or losses. The next task should shift to paper-building evidence around the validated identity-memory mechanism and comparative degradation baselines, rather than inventing another recovery module.

## Deliverable

Before appending `CODEX UPDATE 013`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 013` with corrected-init receipts, production insertion/routing receipts, one training run, the full base-vs-MG-DRA validation table, gate decision, and exactly one next one-hour recommendation.
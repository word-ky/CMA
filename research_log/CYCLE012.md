# Cycle012 — initialization conflict blocks the requested training

2026-09-18 10:31 +08:00: synced9da34c9; mirrored CHATGPT_REVIEW_011.md verbatim into bridge. Read exact architecture before integration. The requested simultaneous Up=0 and alpha=0 creates a dead residual branch. Implemented only the exact isolated adapter and a focused backward reproduction; no production/evaluator/model changes, ROI feature extraction, training or inference launched.

## CODEX UPDATE 012 — BLOCKED: double-zero initialization makes the required nonzero gradients impossible

The Cycle012 architecture and smoke requirements conflict. With h=GELU(Down(concat(E_global,E_local_mapped,G))), R=Up(h), and E_fused=E_global+alpha*G*R, the requested initialization sets both R=0 and alpha=0.

For any differentiable task loss L, with Q=dL/dE_fused:

- dL/dalpha = sum(Q*G*R) = 0 because R=0;
- dL/dUp is multiplied by alpha = 0;
- dL/dDown is multiplied by alpha and the Up weights, both zero.

This holds for the requested BCE+Dice and counterfactual ranking losses, regardless of the frozen decoder, real data, ROI mapping or local image features. AdamW does not repair it: weight decay may shrink Down, but Up and alpha remain exactly zero. Consequently the branch remains inactive across all300 steps. The required finite **nonzero** adapter/alpha gradient smoke cannot pass under the specified initialization.

### Concrete evidence, not model performance

Added `cmllm_remote/scripts/memory_dual_scale_adapter.py`, an isolated exact implementation of the prescribed formula (not connected to the evaluator or training path), and `cmllm_remote/tests/test_dual_scale_initialization.py`. For C=256 and bottleneck16, the adapter has12,577 parameters: Down8,224,Up4,352,alpha1; this meets the parameter budget.

With deterministic synthetic nonzero global/local features and an all-one ROI gate, zero-init output is bitwise equal to the global features. Backward gives finite but exactly zero gradients for every parameter, including alpha. One synthetic AdamW step at the specified lr/weight decay leaves the residual output exactly zero. This is an algebra/software reproduction, **not a real-data smoke or segmentation result**. Receipt: `research_log/cycle012/initialization_receipt.json`. All35 CPU tests pass, including the test demonstrating this blocker.

### Execution status and unchanged scope

No A6000 job, model inference, candidate training, validation, diagnostic or confirmation run was launched. No checkpoint exists; performance tables, transition counts, oracle union and acceptance gate are **not evaluated**, rather than failed. The adapter has not been inserted into LISA, so identity-row routing and ROI-feature remapping have not yet been implemented or validated. Existing base and earlier negative results are unchanged. This is a protocol correction request, not evidence against MG-DRA's scientific hypothesis.

### Exactly one next one-hour recommendation

Revise the initialization contract to **zero Up with alpha initialized to1**, retaining exact baseline equality at initialization. Revise the smoke requirement accordingly: at the first backward Up should receive finite nonzero gradients while Down/alpha may correctly be zero; after one fixed smoke optimizer step, verify downstream adapter gradient flow and frozen-base gradients, then discard the smoke state and start the one candidate from the revised initialization. This is a proposed correction only; it was not implemented or trained. Ask ChatGPT to explicitly confirm the revised initialization/smoke contract in the bridge before resuming the remaining Cycle012 integration and single run. No other architectural, optimizer, data or gate change is proposed.

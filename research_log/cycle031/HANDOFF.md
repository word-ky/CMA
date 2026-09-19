# Cycle031 — method teaching and mechanism audit

REVIEW030 / 7763f3b implemented. Manuscript work stays paused. This cycle is source/config reading and Chinese technical teaching only.

1. METHOD_WALKTHROUGH.md connects need/motivation to the actual two-identity input, REF injection, LLM, SAM conditioning, training and offline paths, with code anchors and only source-supported dimensions.
2. INHERITED_VS_CMA.md separates reused behavior, repository REF/rank extensions and evaluation logic. Exact upstream diff is not established; first recovery import already includes extensions, so no novelty is inferred from location/history.
3. TOY_COUNTERFACTUAL_EXAMPLE.md uses one two-worker scene in three hypothetical states. At margin0.05, ignored-memory hinges are0/0.93, mean0.465 and weighted rank0.6975. Poor-localization and good-localization identity-correct examples both have zero rank, explaining why pixel losses remain necessary. These are teaching numbers, not empirical scores.
4. MEMORY_CONTRACT.md exposes same-condition crop+supplied worker mask/bbox and explains target-free inference placeholders versus legitimate reference inputs.
5. TEACHING_FLOW.md gives one compact Mermaid teaching diagram, with training branch and separate offline explanation.
6. METHOD_TEACHING_AUDIT.md and source_receipt.json record source support and limits. Zero manuscript, model, training, scorer, new sample/baseline/component or Layer2 work.

Two implementation subtleties are now explicit. Output-side REF-associated hidden extraction uses shifted indices, unlike actual REF-slot input injection and unshifted auxiliary REF reconstruction; under the stated single-image expansion layout it selects the preceding hidden position. In a causal model this preceding state cannot directly read the later REF injection. This is static reasoning, not a measured bug/effect, and no frozen source was changed. Separately, auxiliary reference-mask decoding executes before the inference return; its loss is training-only, but its computation is not currently skipped during inference.

Exactly one next recommendation: walk through the two-identity example with the user to clarify REF injection, shifted hidden extraction and the rank loss before any further development.

# Cycle023 — evidence freeze and training-overlap finding

Mode: claim-audit and numeric-audit. Review022 is the sole task; documentation/provenance only. No inference, training, tuning, new metric, bootstrap or subgroup rescoring.

## Audit outcome

**Exact w15 training exposure: UNKNOWN.** The exact run-specific combined manifest, recorded hash and environment overrides were not recovered. Frozen model-file hashes are in w15_training_exposure_audit.json and match the prior verified transfer. The archived builder and training shell match current code after newline normalization. Archived source inputs reconstruct 13,455 rows (9,724 regular +3,731 counterfactual), matching historical logs, with 8,123 unique image-byte SHA256 values.

This reconstructed documented protocol overlaps **37/50 Cycle022 images and 50/50 Cycle018 images**. It hashes per-instance image paths into buckets, so aliases of the same image bytes can fall into different buckets. Detailed matched training rows and image hashes are retained in protocol_reconstruction_audit.json and reconstructed_training_image_registry.json. This is positive overlap evidence under reconstruction, not merely a missing zero-overlap certificate. Exact historical counts/intersections remain null because run binding is missing. Earlier ancestor/pretraining exposure is not exhaustively covered by the reconstructed final stages.

The historical full 921-group counterfactual holdout was evaluated during checkpoint comparisons, documented in historical_log_excerpts.json lines877–882. Cycle022 freshness is limited to recorded Cycles001–021 iteration history. Neither evaluated split establishes training-unseen generalization or historically blind model selection. No overlapping group was removed or rescored. The original scores remain intact.

Search coverage: local recovery-file inventory, the 1,012-member emergency archive, archived scripts/logs and available A6000 project/shared paths. The exact run manifest/config remains missing; search stops within this documentation cycle. No replacement checkpoint or new evaluation split was created.

## Frozen outputs and checks

MAIN_LAYER1_TABLE.csv/.md retain eight method/condition rows with separate confirmation and development roles. All 64 numeric cells were compared directly to frozen Cycle018/022 summaries, and source hashes/pointers are retained. No pooled estimate or selective MCR metric was added. CLAIM_EVIDENCE_LEDGER.md bounds the causal intervention, system-level comparison, supplied memory, pseudo labels, sensitivity, Layer2 negative result and provenance claims.

Severity: material scientific limitation for training-unseen/generalization wording. Numeric inconsistency: none found in exported table. Citation metadata/context: not applicable; no literature claims added. No-invention status: all results preserved, unknown exact provenance explicitly null. The audit does not quantify how much overlap explains the performance gap.

Artifacts: research_log/cycle023 contains exact model hashes, reconstructed protocol/intersections, image registry, historical excerpts, source equivalence, claim ledger, frozen tables and verification receipt. The full reconstructed JSONL remains in the local outputs/cycle023_replay archive and A6000 outputs/cycle023; its path/hash are in the audit. verification.json identifies the replay archive. No images or weights enter Git.

Exactly one next recommendation: produce paper figures/text using the frozen table and claim ledger, explicitly disclosing reconstructed training overlaps and historical holdout selection. Method development remains frozen; Layer2 remains retired. Scientific interpretation belongs to ChatGPT review.

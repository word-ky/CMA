# USER DIRECTION OVERRIDE 039 — three contributions + multi-baseline large-scale evaluation

This directive supersedes only the scheduling/non-goal portion of CHATGPT REVIEW 038 / Cycle039 where it conflicts with the user's newer instruction. The scientific conclusions of REVIEW038 remain valid.

## User-approved project structure

Implement three connected contributions:

1. **Problem + benchmark/data**: degraded identity-grounded relational perception with same-scene/same-query counterfactual identity switching.
2. **CMA model**: supplied identity-localized memory, REF-conditioned relational semantics, explicit geometry prompting, and counterfactual identity ranking. Cycle038 further establishes strong frozen functional dependence on the crop-memory channel beyond retained mask/bbox localization.
3. **Degradation-aware Agentic Memory system**: the agent acts on memory state through ASSESS -> REPAIR/ENHANCE -> CMA -> CANDIDATE UPDATE -> COMMIT/ROLLBACK. Enhancement is a memory-repair action; rollback restores a prior stable memory version.

## Experiment requirement

The user requires comparison with **multiple baselines** and practical performance measurement on a **large test set**. If exhaustive evaluation is too slow, use a large deterministic test set rather than a tiny convenience subset.

Hard rules:
- one locked manifest shared by all methods;
- prefer the largest mechanically valid documented-disjoint pool;
- target >=200 counterfactual groups when available; if fewer exist, use all valid groups;
- freeze the manifest before new method outputs;
- no result-driven sample selection;
- clean + target15_b minimum;
- report mIoU, CMSA, Fidelity, IER, identity margin, and inference cost;
- small subsets are wiring/debug only, not headline evidence.

## Baseline priorities

Audit and execute at least four valid systems including CMA. Priority order:
- SegLLM (already integrated);
- RAS/ORES (ICCV 2025), high priority because it natively supports text + reference mask and complex relations;
- LISA (CVPR 2024) as a reasoning baseline, with reduced-input status stated if identical identity-memory input is impossible;
- VRP-SAM (CVPR 2024) and/or ProSAM (ICCV 2025) as visual-reference baselines, only if the adapter does not leak target information or falsely imply native relational-task equivalence;
- DINOv may be added if practical.

Task compatibility and input parity must be documented for each baseline.

## Revised CYCLE 039 — large-scale benchmark and baseline execution contract

The previous Cycle039 action-headroom work is postponed to Cycle040. Do not discard it.

### A. Build maximal documented-disjoint inventory
Reuse the existing source-image SHA256 exposure registry and Cycle024/025 provenance logic. Enumerate every mechanically valid counterfactual group outside the exclusion registry. Record deterministic rejection reasons. No model-output-based filtering.

### B. Freeze CMA_BENCH_LARGE_V1
If >=200 valid groups exist, freeze at least 200 using a predeclared deterministic hash order; prefer all valid groups if practical. If <200 exist, freeze all. Bind image hashes, identity/pair IDs, miner memory assets, target pseudo-mask receipts, condition recipes, and scorer version before new baseline inference.

### C. Build baseline compatibility matrix
For SegLLM, RAS, LISA, VRP-SAM, ProSAM, and optionally DINOv record:
- official code/checkpoint availability;
- accepted input modalities;
- native ability to express “helmet associated with this referenced miner”;
- required adapter;
- raw-information parity;
- target-leakage risk;
- estimated runtime/GPU;
- READY / ADAPTER_NEEDED / TASK_MISMATCH / BLOCKED.

Do not start heavy baseline training in this cycle.

### D. Audit agentic episode feasibility
Inspect metadata for persistent worker IDs / temporal adjacency. Report exactly one:
- REAL_SEQUENCE_EPISODES_AVAILABLE;
- PARTIAL_SEQUENCE_METADATA;
- NO_PERSISTENT_IDENTITY_EVIDENCE.

If real sequences are unavailable, define a controlled memory-corruption/transition benchmark using paired clean/degraded observations, explicitly not a real temporal-memory claim.

### E. Deliverable
Append CODEX UPDATE 039 with:
1. maximal candidate inventory;
2. frozen CMA_BENCH_LARGE_V1 and exact size;
3. baseline compatibility matrix;
4. persistent-identity/sequence audit;
5. runtime estimate;
6. one next recommendation.

Cycle040 should then return to the postponed Agentic Memory work: action-headroom audit + transactional memory state adapter, using the frozen benchmark as the evaluation substrate.

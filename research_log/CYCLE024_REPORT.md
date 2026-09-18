# Cycle024 — contamination-aware candidate inventory

Review023/3150e75 is implemented. This cycle performed file/metadata inventory only: zero CMA, SegLLM or SAM forwards, no scoring, no training, and no changes to frozen models, prompts, thresholds, degradation or historical manifests. The 13 post-hoc Cycle022 nonoverlap images were not reused or rescored.

## Decision

**ENOUGH_FOR_STRONGER_FROZEN_EVAL** at the requested metadata-candidate stage. There are **402 unique source-image candidates** outside the conservative documented exposure registry. The first **50** under `sha256("CYCLE024:" + image_sha256 + ":" + counterfactual_id)`, then ID, are frozen in `cycle024/documented_protocol_disjoint_candidates.jsonl` with status **DOCUMENTED_PROTOCOL_DISJOINT_CANDIDATES**.

This does not establish exact historical training disjointness. Exact w15 run-bound exposure and exhaustive ancestor/pretraining exposure remain UNKNOWN. The selection has no model outputs. Masks were not regenerated; validity here means existing accepted pairing metadata with distinct miner/helmet identities and reconstructible, in-bounds source boxes, not verified new mask quality.

## Exclusion registry

`cycle024/exclusion_registry.json` is keyed by source-image byte SHA256 and contains **9,549** unique hashes. Every entry retains provenance classes and source references. Counts below overlap and must not be summed:

| Provenance class | Unique source-image hashes |
|---|---:|
| Reconstructed documented final-stage w15 training | 8,123 |
| Historical full 921-group counterfactual holdout | 821 |
| Historical full 2,494-row ordinary holdout | 2,354 |
| Prior Cycle022 registry of Cycles001–021 use | 414 |
| Cycle022 selected images | 50 |
| Local historical/cycle references | 8,399 |
| Remote historical/cycle references | 8,171 |

The ordinary holdout is reconstructed using the unchanged historical builder on clean_train buckets8/9; its 2,494 rows match the logged full historical evaluation. Historical logs already identify the full921 and2494 evaluations. The local scan covered1,625 pre-existing text/manifests including archived history, and local+remote resolved-reference receipts total2,675 files. IDs, aliases and direct hashes are resolved to source bytes; conservative mentions are excluded. Source-pool manifests and the SAM label-generation inventory alone are not treated as evidence of task-model exposure. Hashes/sources for every scanned file are in history_scan_receipt.json; full local token records remain in the replay archive and local outputs/cycle024_support.

## Split-integrity finding

The reusable `research_log/check_image_split_integrity.py` reports **FAIL** on the historical roles: **1,239** source hashes occur in multiple roles. Training intersects the historical counterfactual holdout on595 images and the ordinary holdout on1,109; the two holdouts intersect on673. These are overlapping sets, not additive counts. The checker exits1 as expected on this observed failure. Two focused tests pass: annotation-path aliases crossing roles fail; duplicate identities within one role pass. Historical manifests remain unchanged.

## Candidate inventory

All ten existing final_accepted_v1 manifests were inventoried, including review/reject material, and their SHA256 values independently match the local emergency backup. The mining_helmet source ZIP has13,432 image members representing13,252 unique image-byte hashes. The two local recovery archives and accessible remote TGZ member inventories are recorded in candidate_inventory.json. The emergency archive contains the same accepted/review pairing pools; remote replay archives concern already studied assets. Raw helmet-only annotations without accepted miner associations do not supply an additional CMF group.

| Candidate source | Groups inspected | Excluded image bytes | Other metadata rejection | Valid before image deduplication |
|---|---:|---:|---:|---:|
| Existing clean counterfactual groups | 4,652 | 4,652 | 0 | 0 |
| Existing review counterfactual groups | 3,280 | 2,541 | 307 lack two accepted pairs | 432 |
| Combinations of existing accepted pairs on each image | 6,803 | 6,357 | 11 lack distinct miner/helmet identities | 435 |

The last two pools share images and pairs. Global image-byte deduplication leaves402 images;250 review-group alternatives and215 accepted-pair alternatives are duplicates under the fixed ordering. No new miner–helmet relation was inferred: each component pair is copied exactly from the existing accepted-high pair manifest. New combinations only join two such pairs already associated with the same source image.

The selected50 comprise23 existing review groups and27 accepted-pair combinations. **All23 selected review groups retain `all_pairs_have_clean_episode=false`, despite `all_pairs_deepseek_accept=true` and the recorded overlay check.** They are not relabelled as historical clean groups. Their original group QC records are preserved in selected_group_qc_provenance.json. The27 constructed combinations are also labelled as such, not represented as previously clean-approved groups. Thus the candidate pool has sufficient accepted pairing/reconstruction metadata under the Cycle024 criteria, but its episode-level QC history differs from the earlier clean pool and must remain visible to scientific review. No rejected/review component pair was promoted to accepted, and no quality flag was rewritten.

## Freeze and verification

- Candidate manifest SHA256: `de4335243a9188cf0e2b55046668eca338fc85ec0e8a5b3c609d490f07e8503f`.
- Exclusion registry SHA256: `837088966a5ef3fc6f24515318b4e4f0743a9c76e54a08fa1fcd8cbc17768aa7`.
- Inventory SHA256: `69bc3a724c49fd3707bfbc1b3529cd177b4839cee8fd29999258c74454b0cc8b`.
- Local independent checks confirm all402 hashes are absent from the registry, all are unique, selected50 exactly equal the mandated first50, and all selected pair records match the original accepted metadata. Source images are hashed directly from the original archive.
- `local_verification.json` records counts, pairwise intersections and the6,784,628-byte replay archive SHA256 `c2299dad54e90e20c85d6fe4cc603c789104556bb24b83b2635187276e66d710`. The archive is retained in local outputs and remote research_log; compact receipts and requested registry/manifest stay in research_log/cycle024.

The local history collector was restarted after observed slow scanning, with dependency-junction traversal removed and filename matching bounded at token boundaries. These were read-only collector retries; no experiment was restarted. A6000 CPU inventory completed successfully; no missing-source-image rejection occurred in the candidate pools. The user's unrelated local-backup cleanup assessment remains preserved, and no model backup was deleted.

Exactly one next recommendation: subject to ChatGPT's review of the retained episode-QC limitation, run one later frozen evaluation on this already selected50 using unchanged CMA base-w15 and pinned SegLLM, preserving the DOCUMENTED_PROTOCOL_DISJOINT_CANDIDATES qualification. Do not start that evaluation in Cycle024. Method development stays frozen and Layer2 stays retired.

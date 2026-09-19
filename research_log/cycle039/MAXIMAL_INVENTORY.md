# Maximal inventory — CMA_BENCH_LARGE_V1

All 352 remaining source-distinct, mechanically valid groups are frozen; none was selected using new model performance.

Cycle024 audited 14,735 raw group records: 4,652 clean, 3,280 review and 6,803 accepted-pair combinations. Its existing eligibility and source-image deduplication yielded 402 representatives. Cycle039 preserves those deterministic representatives, adds the 50 images used by Cycle025/038 to the documented exposure registry, and includes all remaining 352. Thus “maximal” means all available distinct-source-image representatives under this established policy, not every combinatorial identity pair or all original dataset images.

| Stage | Count |
|---|---:|
| Existing unique candidate pool | 402 |
| Additional previously evaluated images excluded | 50 |
| Candidate groups frozen before restoration | 352 |
| Mechanically valid after restoration | 352 |
| Rejected after restoration / replacements | 0 / 0 |
| Unique images / identities | 352 / 704 |
| Exclusion registry unique image hashes | 9,599 |
| Accepted-pair combinations / historical review groups | 193 / 159 |

Ordering is SHA256(`CMA_BENCH_LARGE_V1:` + image SHA256 + `:` + counterfactual ID), then ID. candidate_manifest.jsonl and additional_rejections.json preserve the selection and exclusions; Cycle024 candidate_group_audit.json preserves earlier rejection reasons. inventory_receipt.json binds the earlier inputs by hash.

The 159 review groups retain original QC, including missing clean-episode acceptance. Accepted-pair combinations retain constructed-group provenance. Passing mechanical checks does not make either source human-verified clean segmentation data. No new score threshold, hand correction or promotion was used.

Source: recovered DsDPM66 mining_helmet archive, 13,432 COCO image entries (13,252 distinct image byte hashes in the recovered inventory). This is a counterfactual subset, not a full original-dataset evaluation. Zero exact-byte overlap is verified against documented exposure; unknown historical w15/ancestor exposure remains unknown. No unseen-training guarantee is inferred.

# CODEX UPDATE 014 — Frozen external-baseline compatibility protocol

2026-09-18. Implements CHATGPT REVIEW 013 at `3d7c907`. **Complete: protocol and source audit; no external model performance result.**

`research_log/BASELINE_PROTOCOL.md` freezes reconstructed Cycle006 val50 (50 groups/100 ordered identities), clean + deterministic target15_b seed0, supplied miner geometry with condition-matched appearance, fixed semantic query, original-resolution targets, unchanged CMF definitions and no GT-dependent runtime choice. Manifest/scorer/degradation hashes and identity ordering are in `research_log/cycle014/protocol_receipt.json`; original asset hashes remain in Cycle006. This development set uses regenerated pseudo-labels and is not a final benchmark. No diagnostic30 or confirmation30 was used.

| Method | Proposed tier / native cue | Audit conclusion |
|---|---|---|
| LISA (original) | B / image + text | No native supplied identity cue; CMA's added REF is not original LISA. Memory scores N/A. |
| GLaMM | A / reference bbox | Same miner bbox can bind region identity; bbox-only information, relational output untested. |
| SegLLM | A / historical masked appearance + bbox | Direct multi-round competitor; native cue and released inference checkpoint confirmed. Selected. |
| RegionReasoner | A / reference bbox in native prompt | Native spatial reference support; worn-by relational task and coordinate wiring untested. |
| SAMTok | A / encoded mask tokens | Native supplied-mask encoding and generation supported; combined relational generation untested. |

Tier A here denotes source-supported interface eligibility, not successful end-to-end execution or identical cue information. `research_log/baseline_compatibility.json` records pinned official source links, HF checkpoint revisions, output format, local availability, estimated compute burden and dependency/license limitations. HF metadata confirms checkpoint shards, not numerical reproduction. No score-based baseline ranking occurred.

Tier B is excluded from the CMF trials schema. If evaluated on this query, its one unconditioned output may receive separately labelled reference-agnostic target mIoU, with the unavailable prior-memory referent disclosed. It cannot receive CMSA/Fidelity/margin/IER by duplicating that output. Native category results require corresponding complete labels; selected paired helmets are not automatically all helmets. Bbox-only Tier A rows are marked separately from mask/appearance cues, and different checkpoint training exposure prevents pure architectural claims.

**Scorer seam:** no added adapter or metric change. The existing model-free JSONL scorer already accepts generic saved predictions/targets and entity metadata, with no CMA model dependency. Existing tests cover two independent identity outputs through PNG/NPY/inline serialization and exact IoU matrix preservation. Focused command: `python -B -m pytest -p no:cacheprovider cmllm_remote/tests/test_counterfactual_memory_fidelity.py cmllm_remote/tests/test_counterfactual_export.py -q`; **15 passed in 7.58s**. Synthetic plumbing only, not model evidence. Tier membership is a protocol responsibility; the unchanged scorer cannot infer what a model consumed.

**Selected baseline: SegLLM**, under the mandated comparability → native cue → released code/checkpoint → churn rule. Repo `4593a069f09628ce3a5b46e657f5417fefd7be46`; HF `Marlo-Z/SegLLM` revision `095e0637fcba015a02c0686f67b848c79c3cc80b`, `all_data_checkpoint`. Its released interface/checkpoint meets the first three criteria. Local runnability is not claimed: scoped A6000 inventory found no SegLLM installation/weights and detectron2 is absent. Native HIPIE ops and ancillary weights require a separate port. Root license coverage/HF card are unclear despite an Apache package classifier and MIT HIPIE license; do not redistribute upstream assets under an invented blanket license. No model download, install, training, GPU smoke or full evaluation occurred.

**Exactly one Cycle015 recommendation (one hour, await review):** port only the pinned SegLLM all-data inference release in a separate environment and run first frozen val50 group `cf_c9029d27c0483181`, clean + target15_b, both supplied miner identities. Preserve native memory masked-crop and bbox encoding, verify yxyx/1024 and padding/original-resolution transforms, save four masks and prompts to the existing JSONL with provenance and measured resource/time receipt. Use no target-driven output choice. Stop with the concrete blocker if setup exceeds the hour. This is a wiring smoke, not a score claim; full val50 requires the next explicit task. No other baseline or model adaptation is recommended in this cycle.

Durable audit evidence is under `research_log/cycle014/`; source cache contains only small public docs/code, and published receipts contain their URLs/hashes. Latest Issue #1 comment was checked (2026-09-18 05:07:20 UTC), consistent with this task. Await new ChatGPT review.

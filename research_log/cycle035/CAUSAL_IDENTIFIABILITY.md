# Frozen evidence: causal identifiability

Scope: Cycle025 supplies a joint localized identity state, not an autonomous memory. Each row has exactly one status. Supported means observed on the frozen 50-group protocol, not universal control or perfect success.

| Claim | Status | Reason and evidence anchor |
|---|---|---|
| Joint supplied identity-localized memory controls helmet selection | SUPPORTED_BY_FROZEN_SYSTEM_EVIDENCE | Fixed image/query with A/B crop+mask+bbox changes shows identity-dependent selection; Cycle025 HANDOFF and cycle026/PAPER_LAYER1_TABLE.md report CMSA 92% clean / 58% degraded and Fidelity 97% / 84%. Failures remain. Cycle034 INPUT_CONTRACT_RECEIPT.md verifies corresponding source inputs. |
| Appearance/crop semantics add value beyond localization | NOT_IDENTIFIABLE_WITH_CURRENT_BUNDLE_INTERVENTION | Crop, mask and bbox change together. No localization-fixed contrast exists; cycle034/CMA_VS_SEGLLM_CAUSAL_MAP.md and LISA.py:764–768 show mixed crop+bbox input. Crop also contains mask shape. |
| Bbox alone is sufficient/insufficient | NOT_IDENTIFIABLE_WITH_CURRENT_BUNDLE_INTERVENTION | Neither frozen system is bbox-only. SegLLM also gets masked appearance, CMA also gets mask geometry; cycle034/MEMORY_INTERFACE_PARITY.md. Low or high complete-system scores cannot resolve bbox sufficiency. |
| Full mask geometry adds value beyond bbox | NOT_IDENTIFIABLE_WITH_CURRENT_BUNDLE_INTERVENTION | Direct 16x16 mask geometry exists, but its incremental benefit is unmeasured; LISA.py:828–924 and cycle034 causal map. Channel presence is not an effect estimate. |
| Counterfactual rank loss adds value | REQUIRES_NEW_MATCHED_TRAINING | Rank enters training at LISA.py:652–669,711–720; no matched objective contrast. Removing a training loss only at inference cannot test its benefit. No rank experiment is proposed in this cycle. |
| Corrected unshifted output REF semantics would add value | REQUIRES_NEW_MATCHED_TRAINING | Cycle032 REF_ALIGNMENT_AUDIT and Cycle033 runtime receipt locate current pre-REF read. Repointing a learned projection changes its input distribution; syntax/index correctness predicts neither improvement nor a comparable learned method. No hotfix or experiment is proposed. |
| CMA is robust because of semantic memory rather than supplied localization | NOT_IDENTIFIABLE_WITH_CURRENT_BUNDLE_INTERVENTION | Joint semantic/geometric paths and differently trained comparator confound attribution. Cycle025 supports higher absolute degraded performance under one compound stressor; CMA drops 20.75 mIoU points versus SegLLM 6.26. It does not establish smaller sensitivity or a semantic cause. |

No row requires the Layer2 status: all seven are Layer1 claims. Autonomous writing/retrieval/repair remains outside this audit. Targets are reconstructed pseudo labels; exact historical training exposure is UNKNOWN. Runtime inference is target-free per frozen read receipts, while asset preparation already accessed pseudo-targets.

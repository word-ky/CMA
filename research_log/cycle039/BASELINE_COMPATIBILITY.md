# Baseline compatibility — audit 2026-09-20

Only CMA and pinned SegLLM currently have completed project comparisons. No Cycle039 task-model result exists. Four valid systems including CMA have not yet been achieved. Official-source retrieval hashes are in official_source_receipts.json; cached texts are in outputs/cycle039_official_sources.

| System | Status for this benchmark | Native inputs and relation | Required work / information parity |
|---|---|---|---|
| CMA w15 | READY | RGB + language + miner crop/mask/bbox; miner→helmet | Existing frozen runner, new manifest binding; supplies same raw identity source as SegLLM |
| SegLLM | READY | RGB + language + reference history; masked appearance/bbox | Existing pinned native port, new manifest binding; raw source parity, different internal memory routes |
| RAS / ORES | ADAPTER_NEEDED | RGB + text + mask-pool references; relational queries | Register supplied miner mask in native reference pool and export selected predicted target masks; candidate proposals from current RGB only |
| LISA-7B-v1 | ADAPTER_NEEDED, reduced-input comparison only | RGB + reasoning text; no native supplied miner-mask memory | Separate reference-agnostic result; no equivalent identity grounding or valid CMF claim |
| VRP-SAM | TASK_MISMATCH | Query image + support image/mask; visual class reference | Miner support is not a native textual miner→helmet relation; learned checkpoint also unlocated in audited README/tree |
| ProSAM | BLOCKED | Published visual-reference segmentation method; native relational interface unverified | Official code/weights not located in bounded paper/author search; cannot claim a runnable adapter |

## Official sources and implementation boundaries

[SegLLM official repository](https://github.com/berkeley-hipie/SegLLM) is already pinned by protocol_receipt.json: source 4593a069f09628ce3a5b46e657f5417fefd7be46 and checkpoint 095e0637fcba015a02c0686f67b848c79c3cc80b. Existing native inference uses reference history. CMA additionally supplies separate mask/bbox geometry to SAM; this is a native-system comparison, not isolated architecture causality.

[RAS official repository](https://github.com/Shengcao-Cao/RAS), revision 7eacf350248067fc5f17261ea4b48382a9f6c11e, documents image/text/mask-reference use and SAM or Co-DETR mask proposals. [RAS-13B-General checkpoint](https://huggingface.co/Shengcao1006/RAS-13B-General) revision 13f85fc38645e0b452045e425f5f5c1707e10058 is public and ungated. eval_ores.py is offline scoring, not an inference adapter. Do not populate its candidate pool with helmet GT; gt_mask_pool belongs exclusively to offline evaluation. An adapter can preserve the common RGB/query/miner-mask information while exposing a different native representation. This remains proposed until implemented and tested.

[LISA official repository](https://github.com/JIA-Lab-research/LISA) and [released LISA-7B-v1](https://huggingface.co/xinlai/LISA-7B-v1), revision 43c754eef75871fcc11c84d3930a402f7b0a754f, provide image-plus-text segmentation. The audited chat entry point lacks the supplied reference-mask identity channel. The unchanged pronoun query would be under-specified. Never duplicate one identity-agnostic output into A/B and report it as a native memory-switch trial. Any reduced-input target-mIoU report must explain its query and task mismatch; no position-language or visual-mark adapter is silently assumed.

[VRP-SAM official repository](https://github.com/syp2ysy/VRP-SAM), revision 9c53a6a43373165d7cd185d96707b6a4d6665413, passes query_img/support_imgs/support_masks into its model. It does not expose the required text relation. A helmet support mask would introduce target information unavailable to the other systems. No learned VRP checkpoint link was found in the audited README/tree; that is a bounded availability finding, not proof of universal absence.

[ProSAM paper](https://arxiv.org/abs/2506.21835) and [author publication page](https://www.liu-ren.com/) identify probabilistic prompts for visual reference segmentation. Code/checkpoint and a native miner→helmet instruction path were not located. Do not confuse unrelated protein software named ProSAM with this method.

## Existing verified results, not new benchmark results

Cycle025 evaluated 50 groups / 100 identities under each of two conditions, using reconstructed pseudo-targets. Percentage units; IER lower is better.

| System / condition | mIoU | CMSA | Fidelity | IER |
|---|---:|---:|---:|---:|
| CMA clean | 88.28 | 92 | 97 | 1 |
| CMA target15_b | 67.53 | 58 | 84 | 3 |
| SegLLM clean | 42.71 | 4 | 49 | 42 |
| SegLLM target15_b | 36.44 | 2 | 46 | 38 |

Source: ../cycle025/HANDOFF.md and its frozen scoring reports. RAS, LISA, VRP-SAM and ProSAM have no comparable project result yet. Published scores on different tasks/datasets must not fill these missing cells. CMA's absolute degraded result is higher; its clean-to-degraded mIoU drop is also larger (20.75 versus 6.26 points).

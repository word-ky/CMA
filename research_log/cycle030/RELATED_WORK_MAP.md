# Nearest-neighbor task map

This is a positioning audit, not a benchmark. NE = **not established from the inspected source**, not proof of absence or inability. N/A means the specified input interface has no separate visual-reference variable. Each populated cell links to a primary paper and a locator. Descriptions concern reported formulations, not the limits of an architecture.

| Method | Text query | Visual/reference mask/box | Same/cross-image reference | Output relative to reference | Conversation/history | Same-observation counterfactual identity evaluation | Comparable identity-control metrics |
|---|---|---|---|---|---|---|---|
| SAM | Text proof of concept, distinct from principal geometric-prompt evaluations [S, Sec.6.2] | Point/box/mask prompts [S, Sec.3] | Same-image geometric prompting [S, Sec.2] | Prompted region [S, Sec.2] | Iterative mask prompting; entity-conversation memory NE [S, Sec.3] | NE [S, Sec.6] | NE; mask-quality/interactive evaluation [S, Sec.6] |
| LISA | Reasoning instruction [L, Abstract/Sec.3] | Supplied entity-memory mask/box NE [L, Secs.3--4] | N/A for image+instruction formulation [L, Secs.3--4] | Language-requested region; no separate reference required [L, Secs.3--4] | Explicit visual mask-history memory NE [L, Secs.3--4] | NE [L, Sec.5] | NE; overlap metrics [L, Sec.5] |
| Classical RIS, Ye2019 | Referring expression [R, Abstract] | Separate visual entity memory NE [R, Sec.3] | N/A for image+expression formulation [R, Abstract] | Language-referred object [R, Abstract] | NE [R, Sec.3] | NE [R, Sec.4] | NE; segmentation evaluation [R, Sec.4] |
| VRP-SAM | Not required by visual-reference formulation [V, Secs.3--4] | Annotated reference including mask/box/point/scribble [V, Fig.1] | Reference-to-target image transfer [V, Secs.3--4] | Reference-specified object concept in target [V, Fig.1] | Conversational memory NE [V, Sec.4] | NE [V, Sec.5] | NE; segmentation IoU [V, Sec.5] |
| DINOv | Not required by visual-prompt formulation [D, Sec.2] | Visual prompts and reference segments [D, Secs.2.2--2.3] | Both single- and cross-image [D, Fig.2/Sec.2.3] | Prompt-associated object(s)/concepts [D, Sec.2.1] | Conversation NE; video evaluation exists [D, Sec.3.2] | NE [D, Sec.3] | NE; generic/referring/video task metrics [D, Sec.3] |
| SegLLM | Multi-round queries [G, Sec.4.1] | Previous masks and spatial representation [G, Sec.4.2] | Same-image conversation [G, Figs.1--2] | Can select different relational target [G, Sec.4.1] | Yes, text and visual outputs [G, Sec.4.2] | Identical CMA intervention NE [G, Sec.5] | Identical paired identity metrics NE; per-round cIoU [G, Table1] |
| CMA (local implementation) | Fixed relational query [C] | Supplied miner mask+bbox+same-condition crop [C] | Same-image fixed geometry [C] | Associated helmet, not supplied miner [C] | Fixed forward with supplied history; no autonomous writer [C] | Yes, frozen two-identity intervention [C] | CMSA/Fidelity/IER plus target mIoU [C] |

## Sources and locators

- [S]: https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf
- [L]: https://openaccess.thecvf.com/content/CVPR2024/papers/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.pdf
- [R]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Ye_Cross-Modal_Self-Attention_Network_for_Referring_Image_Segmentation_CVPR_2019_paper.pdf
- [V]: https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_VRP-SAM_SAM_with_Visual_Reference_Prompt_CVPR_2024_paper.pdf
- [D]: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Visual_In-Context_Prompting_CVPR_2024_paper.pdf
- [G]: https://proceedings.iclr.cc/paper_files/paper/2025/file/8e125fb53198f2143716d969584d0a28-Paper-Conference.pdf
- [C]: [implementation trace](../cycle028/METHOD_TRACEABILITY.md), [frozen protocol](../cycle025/HANDOFF.md), [equations](../cycle028/METHOD_LATEX.tex).

## Positioning decision

SegLLM already handles relational outputs from earlier entities. Thus “reference entity -> different associated target” is not itself a defensible priority claim. Describe CMA's supplied worker-to-helmet contract and frozen identity-switch test, without asserting unique capability. Geometric prompt methods are not identical-memory experimental baselines. A NE cell must not be converted into a categorical absence claim in prose.

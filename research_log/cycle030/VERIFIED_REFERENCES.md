# Verified references — Cycle030

Verified 2026-09-19. Bounded six-paper bibliography requested by REVIEW029, not an exhaustive novelty search. Full papers remain in ignored outputs/cycle030_sources; compact download hashes are in source_download_receipt.json. No secondary source supports a manuscript claim. CVF BibTeX is retained with URL/capitalization normalization; the Ye page range is verified from the proceedings page. No DOI has been guessed.

## kirillov2023segment — SAM

- Exact title: Segment Anything
- Authors (official proceedings order): Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Dollar, Piotr and Girshick, Ross
- Venue/year: ICCV 2023
- Primary publication: https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html
- Status: official CVF proceedings HTML BibTeX and accepted PDF, downloaded and inspected.
- Supported claim: Promptable image segmentation supplies the inherited mask backbone.
- Evidence locator: Secs. 2--3; Fig. 1; Sec. 6.2 distinguishes the text-prompt proof of concept.
- PDF SHA256: `77992e7a518edcaf5200a750299b98f99de051066f7c1dac357e6baf8ce068bb`.

## lai2024lisa — LISA

- Exact title: LISA: Reasoning Segmentation via Large Language Model
- Authors (official proceedings order): Lai, Xin and Tian, Zhuotao and Chen, Yukang and Li, Yanwei and Yuan, Yuhui and Liu, Shu and Jia, Jiaya
- Venue/year: CVPR 2024
- Primary publication: https://openaccess.thecvf.com/content/CVPR2024/html/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.html
- Status: official CVF proceedings HTML BibTeX and accepted PDF, downloaded and inspected.
- Supported claim: Reasoning segmentation maps a SEG hidden representation to a mask.
- Evidence locator: Abstract; Sec. 4, embedding-as-mask.
- PDF SHA256: `730ab4ddde132309466e6d4a3f3315ef30ec3d1b4bda904ac92e53cb27846208`.

## sun2024vrp — VRP-SAM

- Exact title: VRP-SAM: SAM with Visual Reference Prompt
- Authors (official proceedings order): Sun, Yanpeng and Chen, Jiahui and Zhang, Shan and Zhang, Xinyu and Chen, Qiang and Zhang, Gang and Ding, Errui and Wang, Jingdong and Li, Zechao
- Venue/year: CVPR 2024
- Primary publication: https://openaccess.thecvf.com/content/CVPR2024/html/Sun_VRP-SAM_SAM_with_Visual_Reference_Prompt_CVPR_2024_paper.html
- Status: official CVF proceedings HTML BibTeX and accepted PDF, downloaded and inspected.
- Supported claim: An annotated reference image provides semantic prompts for target-image segmentation.
- Evidence locator: Abstract; Secs. 3--4; Fig. 1.
- PDF SHA256: `64b51913f72eef2824e6e238172323db7730fef6d7f6ddaa079ec89776349a94`.

## li2024visual — DINOv

- Exact title: Visual In-Context Prompting
- Authors (official proceedings order): Li, Feng and Jiang, Qing and Zhang, Hao and Ren, Tianhe and Liu, Shilong and Zou, Xueyan and Xu, Huaizhe and Li, Hongyang and Yang, Jianwei and Li, Chunyuan and Zhang, Lei and Gao, Jianfeng
- Venue/year: CVPR 2024
- Primary publication: https://openaccess.thecvf.com/content/CVPR2024/html/Li_Visual_In-Context_Prompting_CVPR_2024_paper.html
- Status: official CVF proceedings HTML BibTeX and accepted PDF, downloaded and inspected.
- Supported claim: Visual in-context prompts support referring and generic segmentation.
- Evidence locator: Secs. 2.1--2.4; Fig. 2; Sec. 3.2 video evaluation.
- PDF SHA256: `60cdce7659a900ce372092cb278714201b8bf30bdf3e90aae2ef03bd92ccfabb`.

## ye2019cross — RIS representative

- Exact title: Cross-Modal Self-Attention Network for Referring Image Segmentation
- Authors (official proceedings order): Ye, Linwei and Rochan, Mrigank and Liu, Zhi and Wang, Yang
- Venue/year: CVPR 2019
- Primary publication: https://openaccess.thecvf.com/content_CVPR_2019/html/Ye_Cross-Modal_Self-Attention_Network_for_Referring_Image_Segmentation_CVPR_2019_paper.html
- Status: official CVF proceedings HTML BibTeX and accepted PDF, downloaded and inspected.
- Supported claim: An image and referring language expression specify an object mask.
- Evidence locator: Abstract and Sec. 1.
- PDF SHA256: `63aa7ecdc95ce0a1d7b5565d48d614f552f322ef1472a15b6659fcd31ef17a59`.

## wang2025segllm — SegLLM

- Exact published title: SegLLM: Multi-round Reasoning Segmentation with Large Language Models
- Authors (official proceedings BibTeX order): Xudong Wang; Shaolun Zhang; Shufan Li; Kehan Li; Konstantinos Kallidromitis; Yusuke Kato; Kazuki Kozuka; Trevor Darrell.
- Venue/year: International Conference on Learning Representations, 2025; pp. 56526--56547 in proceedings metadata.
- Primary publication: https://proceedings.iclr.cc/paper_files/paper/2025/hash/8e125fb53198f2143716d969584d0a28-Abstract-Conference.html
- BibTeX: https://proceedings.iclr.cc/paper_files/paper/969-/bibtex
- Status: official proceedings metadata/BibTeX and published PDF inspected. OpenReview direct PDF returned HTTP403; the proceedings PDF was accessible. The arXiv version was also retrieved, but is not the cited venue version.
- Supported claim: reuses previous masks/text to segment relationally associated objects across conversational rounds.
- Evidence locator: published Abstract, Secs. 4.1--4.2, Fig. 2; Sec. 5/Table 1 for per-round overlap evaluation.
- Metadata discrepancy: published PDF and arXiv place Konstantinos Kallidromitis before Kehan Li, while official proceedings BibTeX reverses them. This bibliography follows the official BibTeX and records rather than hides the discrepancy. ArXiv uses a shorter title and year 2024; the citation uses the published 2025 title.

## Naming and interpretation

DINOv author order follows the CVF proceedings, including Jianwei Yang before Chunyuan Li. VRP-SAM's exact title includes the VRP-SAM prefix. The Ye2019 architecture's CMSA abbreviation means cross-modal self-attention and is unrelated to our CMSA evaluation measure; prose uses the full architecture name. Literature methods beyond SegLLM are not empirical comparison rows.

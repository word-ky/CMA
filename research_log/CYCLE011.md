# Cycle011 — full-context Memory Spatial Prompt

2026-09-18 08:52 +08:00: synced83538a7 and mirrored review010 verbatim. Confirmed actual remote SAM PromptEncoder accepts boxes+text_embeds jointly, concatenates two box-corner tokens with existing REF/text tokens; no architecture change required. Its _get_batch_size uses boxes rows. Original input image coordinates are resized by existing ResizeLongestSide.apply_boxes; SAM pads bottom/right, with no coordinate offset. One supplied miner box per identity repeated twice for v1_multiround segmentation rows. No expansion, crop, enhancer, adapted weights or training.

Reuse: existing full-frame build_item/model/degradation/export/scorer and Cycle008 D-full metrics. Optional spatial_memory_boxes_list only changes main SAM decode; auxiliary REF reconstruction decode remains unchanged. Zero helmet-target placeholders before inference retain original labels only for offline scoring. Baseline disabled behavior defaults None. MGR helper unchanged.

Checks: prior32 tests green; add coordinate/border/row-order and full-frame/target-independence tests; actual remote prompt-encoder joint-input receipt plus ResizeLongestSide numeric test; then one real disabled-path group regression before single validation50 MSP run. Direct gate: mIoU+0.02,CMSA nondecrease,>=3 recoveries,<=1 regression. Analysis-only GT union: per-group CMSA then mIoU,tie full; separate mIoU-max bound to avoid conflating objectives. If direct gate fails, complementarity needs union mIoU+0.03 and>=5 CMSA recoveries. No diagnostic/confirmation.

08:53 +08:00:34 CPU tests pass; syntax checks pass. Remote LISA file normalized bytes equal repository baseline before patch. MSP full-frame fixture has no crop functions loaded and still succeeds; changing target bytes leaves inputs unchanged.


08:54 +08:00: run20260918-085400-cma-cycle011-msp submitted; check logs before any retry.

08:54 +08:00: launcher exited2 before any Python/model execution because generated shell file had CRLF (pipefail carriage return). Converted run_cycle011.sh to LF only; retry is launch repair, zero previous MSP inference.


## CODEX UPDATE 011 — MSP fails direct and action-complementarity gates

Completed 2026-09-18. Exactly one full-context MSP validation condition was evaluated, with base w15 frozen. Target mIoU falls0.691113→0.646997; CMSA29/50→26/50. Neither the direct mechanism gate nor the predeclared oracle-complementarity rule passes. Training-free spatial-focus variants are retired; no scheduler work starts.

### Exact spatial-box plumbing

The installed SAM PromptEncoder supports boxes and text_embeds together: two box-corner embeddings concatenate with the existing REF/text tokens. No PromptEncoder architecture or weights changed. Actual installed-source SHA256 and a joint-input numeric receipt are in `research_log/cycle011/prompt_encoder_receipt.json`.

New `cmllm_remote/scripts/memory_spatial_prompt.py` takes only supplied miner pixel xyxy boxes and image dimensions, clips them to image bounds, then calls the existing `ResizeLongestSide.apply_boxes`. Coordinates refer to the resized unpadded image; bottom/right SAM padding adds no origin offset. No expansion or target-dependent adjustment. In v1_multiround each miner box repeats twice in order `[miner A,helmet A,miner B,helmet B]`.

The evaluator's optional `--memory-spatial-prompt` passes those boxes via `spatial_memory_boxes_list` to the existing main SAM decode in `third_party/LISA/model/LISA.py`. Existing REF/text context is retained, auxiliary REF reconstruction is unchanged, and disabled default remains `None`. Both SAM and CLIP full-frame inputs use the same pre-existing build_item/degradation path; no MGR transform is called or modified. Helmet targets are zero shape placeholders in the MSP inference item, while original targets remain in offline export/scoring only. No target box, IoU, predicted helmet, crop, enhancer, adapted weights or runtime selection enters MSP.

### Tests and actual execution

34 CPU tests pass. New checks cover clipped original-pixel boxes, identity/seg-row order, unchanged full-frame SAM/CLIP and REF tensors, fixed degradation seed, and helmet-label changes leaving model inputs unchanged. The full-frame test runs with no crop functions in its namespace. Actual remote `ResizeLongestSide` border/numeric checks and PromptEncoder box+text checks pass; existing text embeddings are preserved exactly. With MSP disabled, one real validation group reproduces saved full-frame prediction, target and reference arrays exactly.

Exact command: `bash "$ROOT/research_log/run_cycle011.sh" "$ROOT"`. First launcher `20260918-085400-cma-cycle011-msp` failed immediately on CRLF shell line endings before any Python/model execution. LF-only repair launched `20260918-085434-cma-cycle011-msp-lf`,08:54:38–08:55:27 +08:00,49s,exit0. There was one MSP model-evaluation run on50 groups/100 references, plus one disabled-path regression group. D-full metrics reused Cycle008. No training, diagnostic30 or confirmation30 calls.

### Frozen validation50 comparison

| Metric | D-full | D-MSP | MSP-full | GT oracle union (analysis only) |
|---|---:|---:|---:|---:|
| target_miou | 0.691113 | 0.646997 | -0.044116 | 0.712135 |
| cmsa | 0.580000 | 0.520000 | -0.060000 | 0.620000 |
| memory_fidelity | 0.890000 | 0.940000 | 0.050000 | 0.930000 |
| mean_identity_margin | 0.662931 | 0.627106 | -0.035825 | 0.694177 |
| median_identity_margin | 0.847463 | 0.802047 | -0.045416 | 0.847463 |
| identity_error_rate | 0.030000 | 0.020000 | -0.010000 | 0.020000 |

CMSA transitions: **2 fail→pass,5 pass→fail,24 pass→pass,19 fail→fail**. All direct criteria fail: mIoU delta−0.044116 is below+0.02; CMSA decreases; only2 recoveries instead of≥3;5 regressions instead of≤1. Supporting Fidelity improves89→94/100 and IER3→2/100, but this does not compensate for reduced correct-target overlap and pairwise task success. The result is consistent with geometry helping identity discrimination while harming helmet extent quality; it is not causal proof of that mechanism.

### Oracle union is analysis only, never runtime policy

For each whole group, a GT oracle chooses full/MSP by CMSA then group mIoU, ties full. Its result is mIoU0.712135 (+0.021022 over full),CMSA31/50 (2 new success opportunities). Separately maximizing group mIoU produces the same aggregate table here. The union fails both complementarity requirements: gain below+0.03 and only2 opportunities rather than≥5. Hence this evidence does not justify an oracle-free full/MSP scheduler. Individual selected group choices are saved in `comparison.json`; none is used for real inference selection.

### Decision and exactly one next one-hour recommendation

Retire fixed training-free spatial-focus variants, as predeclared. **Design one tiny learned full-context dual-scale residual feature adapter**, with an explicit frozen-base/zero-residual initialization and bounded training/evaluation contract, for review before implementation or training. This single design task should explain how local evidence can contribute while preserving full-frame context; do not launch another crop, prompt, enhancer or loss sweep. Await the next explicit ChatGPT task.

Replay archive: `research_log/cycle011_replay.tgz`,687,772 bytes,SHA256 `a11da1fb024537684bd8eef8e15c4da4a9818888d3775c9437b9a169d1ee24b4`; local/remote verified. Raw predictions and manifests retained; no new weights. Compact tables, transition/selection records, regression and source hashes reside in `research_log/cycle011/`.

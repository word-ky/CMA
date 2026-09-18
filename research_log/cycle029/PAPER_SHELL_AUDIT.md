# Cycle029 manuscript-shell audit

## Numerical and source consistency

| Surface | Check and source |
|---|---|
| Abstract | Exactly two comparisons: CMSA 58% vs 2% and Memory Fidelity 84% vs 46%. These are `/summary/cmsa` and `/summary/memory_fidelity` in Cycle025 `scoring/cma_target15_b_metrics.json` and `segllm_target15_b_metrics.json`, formatted as percentages. Abstract has 157 whitespace-delimited words, below 190. No other numerical performance claim is introduced. |
| Main and supplementary tables | Directly include unchanged Cycle027 LaTeX tables. All cell values retain the verified Cycle026 CSV provenance and Cycle027 table-format receipt. Main: Cycle025 only; supplementary: separate Cycle022/018 blocks, without pooling. |
| Results/Limitations numerical prose | Direct include of unchanged Cycle027 source: 50 groups, 100 references, CMSA/Fidelity/IER, mIoU and 31.08-point gap, 21/50 failures, 23/27 QC composition, and 20.75/6.26-point degradation declines remain as verified. No rounding or result value is edited. |
| Qualitative IoUs | Include the unchanged Cycle027 figure and caption; its input/output hash receipt and Cycle026 case pool remain controlling. No new raster case or mask is selected. |
| Method constants | Unchanged Cycle028 Method retains checkpoint-verified 0.5 input scale, 16x16 mask geometry, norm 20, clamp 50, initialization 0.2, reconstruction/rank weights 1.0/1.5, margin 0.05; code epsilon 1e-6. Unknown historical CE/BCE/Dice overrides remain unknown. |
| Metric threshold | New figure's IoU >=0.5 is the existing evaluator `min_iou=0.5`, not a new threshold. Symbolic J and S entries are definitions, not simulated results. Equation numbering and subsection numbers are layout only. |

`delivery_verification.json` checks the abstract's four displayed values against the frozen summaries and records unchanged hashes of all included Cycle027/028 artifacts. It does not run an evaluator or recompute metrics.

## Method and contract labels

- Figure input: common I/q and alternative memory A/B, each labelled miner mask + bbox + appearance crop. Explicit same-condition appearance / externally supplied fixed geometry banner. No weak-token-only or autonomous localization implication.
- Figure input-side branch: crop/bbox to REF and language model, mapped to Cycle028 `build_ref_input_embeddings` and `llava_llama.py` scalar addition.
- Figure output-side branch: REF hidden plus mask/bbox geometry, addition to SEG prompt and SAM decoding, mapped to `build_ref_prompt_embeddings` and the frozen add mode. I reaches both language and SAM paths. Weights are shared across A/B trials.
- Training inset: soft-IoU diagonal-versus-maximum-wrong hinge alongside existing text/mask/reference losses. It is separate from inference. Evaluation inset: binary J matrix, strict Fidelity, joint localization-qualified CMSA, stronger-wrong-qualified IER. No target-IoU input or selector appears in inference.
- A/B outputs are explicitly desired schematic outputs. They are not a new empirical success demonstration. The source-controlled SVG/PDF has editable vector shapes/text; PNG is a preview.

The abstract exposes crop + miner mask + bounding box. The method figure repeats all three; the unchanged Method defines R, b and C and explicitly describes fixed supplied geometry and same-condition crop appearance. The shell adds the exact supplied identity-localized entity memory contract after Introduction and native-interface comparator wording before Results.

## Stressor and claim boundaries

Inspected `cmllm_remote/scripts/counterfactual_export.py:9-37`: gamma 2.15, contrast 0.655, intensity scale 0.405, Gaussian noise sigma 23.5/255, then Gaussian blur sigma 0.88 with the existing odd kernel rule. These are existing operations/settings, not new tests. New empirical wording says fixed photometric/noise/blur stressor. Dust/glare in the unchanged Introduction are explicitly restricted to broader motivation by the immediately following integration paragraph; dust/glare/occlusion are not claimed to be simulated.

Title/abstract do not promote Layer-2. There is no affirmative claim of autonomous acquisition/writing/tracking/repair, training-unseen proof, lower degradation sensitivity, architecture-only superiority, successful agent/controller, safety guarantee, or isolated component gains. Primary evidence remains DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION, with pseudo targets and exact historical exposure UNKNOWN. A configuration hash does not recover a run-bound training manifest.

## Integration and limitations

The single shell directly includes existing Intro/Method/Results/Limitations, primary table and qualitative figure, then the separate supplementary table. Only new framing paragraphs, title/abstract, method figure and LaTeX layout are added. All six compiled pages were inspected: no clipping, broken references or overfull boxes; three inherited underfull spacing notices remain. The shell is a generic two-column article draft, not a venue-formatted, citation-complete submission. Verified bibliography and Related Work are still absent; no citation or novelty claim was invented.

Zero model/scorer executions, zero metric recomputations, zero new scientific selection. No training, new baseline/component, architecture change or literature novelty search occurred.

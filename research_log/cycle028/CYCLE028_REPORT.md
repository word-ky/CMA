# Cycle028 — implementation-grounded Introduction and Method

Review027 (73f07c7) implemented. Writing and source inspection only: zero model/scorer runs, metric recomputation, new scientific selection, baseline/component changes, literature-driven scope expansion, architecture diagram or Layer-2 work.

## Deliverables

- `research_log/cycle028/METHOD_TRACEABILITY.md`: maps every requested mechanism to source functions/lines/config fields and distinguishes implementation, whole-system support and absent matched ablations.
- `INTRODUCTION_LATEX.tex`: problem, identity-control gap, dual-stage insight and exactly three contributions. Bullet 2 now names the actual REF design plus paired identity objective/evaluation.
- `METHOD_LATEX.tex`: supplied-memory task; crop/bbox REF input and geometry/hidden SAM output conditioning; reference reconstruction and soft-IoU rank loss; separate binary evaluation definitions; frozen inference semantics and scope.
- `INTRO_METHOD_AUDIT.md`: paragraph/equation-level claim audit against the implementation map and frozen contribution, result and provenance limits.
- `W15_CONFIG_SNAPSHOT.json`, `source_receipt.json`, `delivery_verification.json`, and `ASSEMBLY_PREVIEW.tex/.pdf`: verified configuration, source hashes and a compiled two-page section preview.

## Exact implementation anchors and verified settings

The map uses `cmllm_remote/third_party/LISA/model/LISA.py:732` (`build_ref_input_embeddings`), `:828` (`build_ref_prompt_embeddings`), `:84` (`soft_iou_matrix`), `:652` (helmet rank selection), `:670` (reference reconstruction), and `:720` (total objective). Injection is anchored at `cmllm_remote/third_party/LISA/model/llava/model/language_model/llava_llama.py:91` (`_replace_ref_input_embeddings`). Group construction is at `cmllm_remote/scripts/build_mr_ref_counterfactual_train.py:54` and `cmllm_remote/third_party/LISA/utils/mr_ref_seg_dataset.py:209`. Frozen execution is `research_log/run_cma_cycle025.py:37`, with `cmllm_remote/scripts/eval_mr_ref_counterfactual_v0.py:84` (crop), `:202` (item) and `:342` (forward/threshold), and `run_mcr_train_predictions.py:21` (base loader). Evaluation definitions are `eval_counterfactual_memory_fidelity.py:15`, `:37` and `:103`. SegLLM native memory is `cmllm_remote/external_baselines/segllm/memory_state.py:19`.

The preserved w15 config SHA256 `2713d1a90022fa0e511063afbba73a269e0846e22a2d8f735931562aa36b28f3` matches Cycle025's frozen checkpoint receipt. It records `gated_add`/input scale 0.5, output prompt `add`, mask pool 16, reconstruction weight 1.0, rank weight 1.5 and margin 0.05. The generic launcher default 0.5 is not mistaken for w15's rank weight. Reference scale initialization 0.2 is not reported as its trained value. Exact historical run-bound training manifest, CE/BCE/Dice overrides and ancestor/pretraining exposure remain UNKNOWN.

The Method states that inference uses fixed conversation placeholders and a direct forward pass, not generated autonomous dialogue. Reference appearance is obtained from the same condition image, while supplied geometry remains fixed. CMA and SegLLM receive the same underlying observation/query and corresponding identity information through method-native interfaces; their internal memory encodings are not claimed identical. None of the components receives an unmeasured isolated causal-gain claim.

## Validation and remaining limits

All inspected source hashes remain unchanged. Config bytes match the primary inference receipt. LaTeX compiles in two final passes, two pages, with no unresolved reference or overfull box; two underfull spacing notices remain and do not clip content. The initial metric-equation overflow was repaired by splitting the equation and visually checking the resulting page. Existing Cycle027 tables, prose and figures are unchanged. The preview is not a complete paper or venue-ready layout.

No repository bibliography was found. Neutral citation comments remain for task context and inherited LISA/LLaVA/CLIP/SAM/SegLLM components, as Review027 permits. No citation or novelty-priority claim was invented. This is a citation-incomplete draft rather than a submission-ready manuscript.

Exactly one next recommendation: integrate the full paper with a bounded title/abstract and the existing Introduction, Method, frozen Results and limitations, preserving the documented mechanism and evidence boundaries.

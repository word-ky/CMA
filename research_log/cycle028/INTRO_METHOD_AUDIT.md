# Introduction/Method consistency audit

Scope: all substantive paragraphs, contribution bullets and equations in the two new LaTeX fragments. Sources: `METHOD_TRACEABILITY.md`, Cycle027 contributions and Results/Limitations, Cycle026 claim ledger, and Cycle025 frozen protocol/prediction receipts. This is a writing audit; no model, scorer or statistical analysis ran.

| Draft location | Statements checked | Source and disposition |
|---|---|---|
| Introduction paragraph 1 | Worker identity differs from equipment category; small PPE and degraded evidence motivate the task | Task motivation consistent with frozen formulation. No prevalence statistic, broad safety efficacy or new empirical claim. Citation slot remains a comment. |
| Introduction paragraph 2 | Aggregate IoU alone does not establish identity dependence; same-I/q intervention; localization requirement | Metric definitions (`score_masks`) and fixed-group construction support the distinction. Describes controlled input intervention, not a component causal estimate. |
| Introduction paragraph 3 | Crop/bbox REF input; REF/geometry output prompt; paired rank objective | Traceability rows 1–7. Explicitly disclaims independent component gains. |
| Contributions 1–3 | Same three roles as Cycle027; bullet 2 now exposes the implemented mechanism | Review027 explicitly requests this clarification. No fourth contribution. Empirical bullet preserves whole-system, absolute-degraded and exposure limitations. No numerical value is added or changed. |
| Method task subsection | I/q/R/b/Y definitions; masked crop; same-condition reference appearance | `build_item`, crop preprocessing, Cycle025 runner. Clarifies that supplied reference geometry is not independently corrupted or autonomously generated. |
| Method conditioning equations | Mean crop features + bbox MLP; REF addition; resized mask geometry; context and SAM prompt addition | `LISA.py`, `llava_llama.py`, hash-matched w15 config. Equations are explicitly conceptual abstractions. `gated_add` is scalar scaling, not a reliability estimator. 0.2 is initialization, not final learned scale. |
| Method loss subsection | Text/weighted mask losses, valid-reference reconstruction, soft IoU and max-wrong hinge | Training loss code and paired dataset. Config weight 1.5, margin 0.05, reconstruction weight 1.0 verified; epsilon 1e-6 is a code constant. Exact historical CE/BCE/Dice coefficients UNKNOWN. Group normalization described without claiming missing run provenance. |
| Method evaluation equations | Binary IoU, strict Fidelity, joint CMSA >=0.5, IER, margin and mIoU | Offline evaluator definitions, inspected only. Tie behavior and IER not being the complement of Fidelity are explicit. No alternative threshold/metric introduced. |
| Method inference paragraph | Fixed conversation/direct forward; helmet selection; zero placeholders/no target access; freeze before scoring | Cycle025 runner and prediction-freeze receipt. Supplied miner masks are inputs; no claim of wholly annotation-free inference. Native memory encoding distinction addresses Review027 and refines Cycle027's shorthand about common inputs. |
| Method scope paragraph | Supplied-memory evidence only; no successful agent lifecycle | Cycle027 limitations and Cycle025 status. Research-stop boundary unchanged. |

## Prohibited-claim check

No affirmative claim of exact training-unseen/generalization proof, lower degradation sensitivity, architecture-only superiority over SegLLM, autonomous memory writing/repair, successful Agent/Qwen/RL control, safety guarantee, or independently measured component gain appears. Such phrases occur only as explicit limits. No historical controller/enhancer/MGR/MSP/MG-DRA branch is promoted into the frozen main method.

Cycle025 remains DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION. Its 50 groups/100 references and 21/50 residual degraded CMSA failures remain in the unchanged Cycle027 Results/Limitations; this cycle does not reinterpret them or create new table values. Config verification does not resolve unknown exact historical training exposure.

## Citation and assembly status

No repository `.bib` file was found. Per Review027, neutral LaTeX comments mark citation slots for task context, LISA/LLaVA/CLIP/SAM and pinned SegLLM. No bibliography entry, literature-priority claim or related-work comparison is invented. These are implementation-grounded section drafts, not a citation-complete submission. No new literature search or architecture figure was performed. Generic build preview is used only to check syntax and readability, not venue page-budget compliance.

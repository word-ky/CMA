# Cycle029 — paper positioning and shell integration

Review028 (2bd4f33) implemented. Zero model/scorer calls, metric recomputation, new samples/qualitative selection, new baseline/component, training or Layer-2 work. No literature novelty search.

## Artifacts

`research_log/cycle029/PAPER_POSITIONING.md` freezes the supplied identity-localized entity memory contract: same-condition appearance crop plus supplied miner mask and bbox, with fixed geometry. It distinguishes relational helmet segmentation from autonomous localization/tracking and restricts evaluated degradation to the actual target15_b photometric/noise/blur operations.

`TITLE_ABSTRACT_LATEX.tex` contains one working title, **CMA: Identity-Conditioned Relational Segmentation with Supplied Localized Entity Memory**, and a 157-word abstract. It exposes the spatial memory inputs and same-condition appearance, names the implemented dual-stage REF/rank design, and uses only two headline comparisons: degraded CMSA 58% versus 2% and Memory Fidelity 84% versus 46%. Both are checked against the frozen Cycle025 summaries. Pseudo targets, exact-exposure uncertainty, system-level comparison and non-autonomous scope remain explicit.

`METHOD_FIGURE_SPEC.md` and `render_method_overview.py` define the verified method graph. `METHOD_OVERVIEW.svg/.pdf` provide a vector schematic, with `.png` preview and `method_figure_receipt.json`. It visibly labels both memory A/B choices as mask+bbox+appearance; shows the input REF language condition and output REF/geometry SAM prompt condition; and separates training soft-IoU rank and offline binary-IoU evaluation insets. Desired A/B outputs are explicitly schematic, not new predicted cases. No autonomous writer, controller, enhancement or later experimental branch is drawn.

`PAPER_SHELL.tex/.pdf` integrates the new title/abstract and method figure with the original Cycle028 Introduction/Method and Cycle027 Results/Limitations, main table and qualitative figure. The separate overlap-affected table remains supplementary. Original artifacts are directly included and hash-unchanged. Two integration paragraphs clarify that broader dust/glare motivation is not experimental coverage and that comparator memory encodings remain method-native.

`PAPER_SHELL_AUDIT.md`, `source_receipt.json` and `delivery_verification.json` cover numerical provenance, exact stressor operations, all method labels, input contract, non-claims and artifact hashes. The six-page generic two-column shell compiles in two passes with no unresolved references or overfull boxes. All six pages were visually inspected; three inherited underfull spacing notices remain without clipping. This is not a venue-formatted or citation-complete submission. No bibliography entries or novelty-priority claims were invented.

Exactly one next recommendation: integrate a verified bibliography and bounded Related Work for the existing task, inherited components and pinned comparator, without changing the frozen scientific scope or evidence.

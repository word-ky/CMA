# Frozen Layer-1 results

Rates in this Markdown table are percentages; CSV stores fractions. Margins are unscaled. Each split remains separate; no pooled result or selective MCR metric. Sources: frozen Cycle022 and Cycle018 scoring/comparison.json, with row pointers and SHA256 in table_provenance.json.

**Provenance limitation:** Cycle022 is untouched only by recorded Cycles001–021 iteration history. The historical full holdout was evaluated during checkpoint comparison. Exact w15 training exposure is UNKNOWN; reconstructed documented training overlaps 37/50 Cycle022 and 50/50 Cycle018 images by SHA256. Neither split establishes training-unseen generalization. Targets are reconstructed pseudo labels; identity memories are supplied. SegLLM is a system-level released-baseline comparison with different training exposure.

|method|split role|condition|mIoU|CMSA|Fidelity|IER|mean margin|median margin|N groups|N identity predictions|
|---|---|---|---|---|---|---|---|---|---|---|
|CMA base-w15|Cycle022 primary confirmation|clean|90.87%|94.00%|98.00%|2.00%|0.892993|0.956758|50|100|
|CMA base-w15|Cycle022 primary confirmation|target15_b|68.02%|60.00%|87.00%|2.00%|0.657803|0.860244|50|100|
|SegLLM pinned|Cycle022 primary confirmation|clean|37.29%|6.00%|43.00%|38.00%|0.041568|0.000000|50|100|
|SegLLM pinned|Cycle022 primary confirmation|target15_b|27.52%|2.00%|38.00%|30.00%|0.016394|0.000000|50|100|
|CMA base-w15|Cycle018 development/replication|clean|91.97%|94.00%|99.00%|0.00%|0.919718|0.963190|50|100|
|CMA base-w15|Cycle018 development/replication|target15_b|69.11%|58.00%|89.00%|3.00%|0.662931|0.847463|50|100|
|SegLLM pinned|Cycle018 development/replication|clean|41.57%|4.00%|47.00%|41.00%|0.047717|0.000000|50|100|
|SegLLM pinned|Cycle018 development/replication|target15_b|33.14%|0.00%|42.00%|39.00%|-0.027858|0.000000|50|100|

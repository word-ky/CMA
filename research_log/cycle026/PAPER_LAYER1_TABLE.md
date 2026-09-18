# Paper Layer-1 evidence

Cycle025 is primary. Cycles022/018 are separately labelled context; no pooling. CSV stores unrounded metric fractions; Markdown displays rates as percentages. Exact w15/ancestor exposure is UNKNOWN for all sets. Targets are reconstructed pseudo labels and memories are supplied. Comparison is system-level.

## PRIMARY: DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION — Cycle025

Selected before inference, disjoint by image SHA256 from the documented training/evaluation registry.23review groups retain incomplete clean-episode QC;27are accepted-pair combinations. Not exact training-unseen.

|Method|Condition|mIoU|CMSA|Fidelity|IER|Mean margin|Median margin|Groups / identities|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|CMA base-w15|clean|88.28%|46/50 (92%)|97/100 (97%)|1/100 (1%)|0.873944|0.952629|50 / 100|
|CMA base-w15|target15_b|67.53%|29/50 (58%)|84/100 (84%)|3/100 (3%)|0.646666|0.842875|50 / 100|
|SegLLM pinned|clean|42.71%|2/50 (4%)|49/100 (49%)|42/100 (42%)|0.049034|0.000000|50 / 100|
|SegLLM pinned|target15_b|36.44%|1/50 (2%)|46/100 (46%)|38/100 (38%)|0.031818|0.000000|50 / 100|

## Historical replication: documented overlap — Cycle022

Reconstructed training overlap: 37/50 images. Source holdout historically evaluated; fresh only relative to recorded Cycles001–021. Not a training-unseen test.

|Method|Condition|mIoU|CMSA|Fidelity|IER|Mean margin|Median margin|Groups / identities|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|CMA base-w15|clean|90.87%|47/50 (94%)|98/100 (98%)|2/100 (2%)|0.892993|0.956758|50 / 100|
|CMA base-w15|target15_b|68.02%|30/50 (60%)|87/100 (87%)|2/100 (2%)|0.657803|0.860244|50 / 100|
|SegLLM pinned|clean|37.29%|3/50 (6%)|43/100 (43%)|38/100 (38%)|0.041568|0.000000|50 / 100|
|SegLLM pinned|target15_b|27.52%|1/50 (2%)|38/100 (38%)|30/100 (30%)|0.016394|0.000000|50 / 100|

## Development replication: documented overlap — Cycle018

Reconstructed training overlap: 50/50 images. Source holdout historically evaluated; development-used. Not a training-unseen test.

|Method|Condition|mIoU|CMSA|Fidelity|IER|Mean margin|Median margin|Groups / identities|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|CMA base-w15|clean|91.97%|47/50 (94%)|99/100 (99%)|0/100 (0%)|0.919718|0.963190|50 / 100|
|CMA base-w15|target15_b|69.11%|29/50 (58%)|89/100 (89%)|3/100 (3%)|0.662931|0.847463|50 / 100|
|SegLLM pinned|clean|41.57%|2/50 (4%)|47/100 (47%)|41/100 (41%)|0.047717|0.000000|50 / 100|
|SegLLM pinned|target15_b|33.14%|0/50 (0%)|42/100 (42%)|39/100 (39%)|-0.027858|0.000000|50 / 100|

## Primary paired deltas only: CMA minus SegLLM

Rates in percentage points; identity margins unscaled.

|Condition|mIoU pp|CMSA pp|Fidelity pp|IER pp|Mean margin|Median margin|
|---|---:|---:|---:|---:|---:|---:|
|clean|45.57|88.00|48.00|-41.00|0.824910|0.952629|
|target15_b|31.08|56.00|38.00|-35.00|0.614847|0.842875|

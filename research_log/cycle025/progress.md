# Cycle025

2026-09-18T17:56:59.7866206Z — REVIEW024/598bd3c; final authorized evaluation, no performance gate. Reuse verified Cycle022 restoration/CMA/SegLLM/CMF kernels; four entrypoints differ only by cycle path.15 existing focused tests pass16.71s;3 asset checks pass; compile passes. Original50 manifest fixed, >=40 mechanical validity required. NVML mismatch observed, but CUDA allocation and A6000 device query pass; no driver changes. No extra baseline rerun: Cycle022 receipts supply prior reproduction evidence.

2026-09-18T18:09:18.783756+00:00 — Run015702exit0;50valid,0invalid.54CPUtests passed26.12s after src PYTHONPATH correction;3asset tests passed.400prediction hashes and timing/code/weight invariants verified locally. First degraded result:CMA67.53%/58%CMSA vs SegLLM36.44%/2%. Review024 mirrored verbatim before UPDATE025; all further model experimentation stops.

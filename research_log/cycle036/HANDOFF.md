# Cycle036 — frozen crop-feature probe implemented; execution blocked

REVIEW035/6b63b65 and Issue1 comment5740754378 addressed with the requested exact-blocker delivery. Scientific measurement remains pending.

Evaluation-only crop_zero_hook.py inserts zeros_like immediately after actual pooled crop assignment, before bbox addition. No production path/weight/REF-index change. run_crop_zero.py reuses Cycle025 inference and binds source, weights, input manifests and control predictions; score_probe.py reuses the unchanged CMF evaluator and prepares paired group mIoU/margin deltas and CMSA transition counts. Control predictions are unchanged. Intervention/source/checkpoint receipts, CROP_DEPENDENCE_RESULTS.md and Chinese TEACHING_UPDATE.md are under research_log/cycle036.

Tests: local/remote crop+bbox tensor self-check PASS;15focused scorer/export tests and54full project tests PASS; syntax PASS. These are software checks, not crop-dependence evidence. First shell launch failed on CRLF before Python, fixed to LF. Actual run20260919-174012-cma-cycle036-crop-zero passed pre-load provenance checks and loaded CPU shards, then CUDA allocation failed: competing vLLM process708005 occupied45.31GiB on physicalGPU1. Follow-up showed both A6000 cards nearly full. Zero model forwards, zero new predictions, zero scorer calls; no scientific score can be reported. Logs and exact blocker preserved. No other workloads stopped and no altered precision/checkpoint substituted.

The current authorized single arm can resume with the same command only after GPU capacity materially changes; do not redo implementation or rerun the frozen control. No manuscript, training or Layer2 work.

Exactly one recommendation: defer any matched-continuation training decision until this frozen diagnostic can finish on an available A6000; current software checks and OOM provide no evidence for or against crop-feature value.

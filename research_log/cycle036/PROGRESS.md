# Cycle036 progress

- 2026-09-19 17:35 +08:00: Fast-forwarded REVIEW035/6b63b65; Issue1 comment5740754378 agrees. Authorized one frozen crop-zero inference arm only. Preserved unrelated local audit files.
- Reuse map: Cycle025 runner/build_item/loader and unchanged CMF scorer; frozen Cycle025 control supplies baseline reproduction. Evaluation-only method hook is the sole scientific edit; default production source and weights unchanged.
- Remote status: no CMA run active. Other projects occupy ~24 GiB per GPU; recorded Cycle025 peak allocation is 17.32 GB. GPU1 has ~24 GiB free and no listed non-vLLM training process. Do not interrupt other workloads.
- Test sequence: tiny crop-zero tensor/AST checks, syntax and existing scorer/export tests, exact asset/checkpoint binding, then one full diagnostic run and frozen-prediction paired scoring. No separate model smoke or baseline rerun.
- Tensor/AST self-check PASS for float32 and BF16; syntax PASS; existing scorer/export tests: 15 passed in 16.04s. No production source edits.
- Run 20260919-173909-cma-cycle036-crop-zero exited2 before Python/model: Windows CRLF in shell script made pipefail invalid. Normalized run.sh to LF only; scientific settings unchanged, no predictions generated.
- Full cmllm_remote suite54 passed27.69s. Actual run20260919-174012-cma-cycle036-crop-zero started after LF shell correction; remote tensor self-check PASS. No completed model reruns or performance-based selection.
- Run174012 failed at model CUDA transfer after binding checks and CPU shard load: competing vLLM grew to45.31GiB; no predictions/forwards/scoring. Fetched run log/meta and confirmed output directory empty. Delivered exact resource blocker; unchanged diagnostic remains pending capacity.
- Delivery synced to A6000 with archive SHA256 0b162f9befab1011cb02b874af072ca1cab83425afd31b2753133ea76861eb0c matching locally. Added cycle-local shell LF attribute because actual CRLF launch failure and git autocrlf warning would otherwise recur on checkout.

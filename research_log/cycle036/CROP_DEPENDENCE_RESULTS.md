# Frozen inference dependence — BLOCKED_GPU_MEMORY

No crop-zero model prediction was generated, so no paired scientific result exists. This is not a trained ablation. The existing control is preserved unchanged.

| Condition | Arm | mIoU | CMSA | Fidelity | IER | Mean margin | Median margin |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | Frozen Cycle025 full | 88.28% | 92% | 97% | 1% | .873944 | .952629 |
| clean | crop-zero | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| target15_b | Frozen Cycle025 full | 67.53% | 58% | 84% | 3% | .646666 | .842875 |
| target15_b | crop-zero | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

Control values copied from cycle026/PAPER_LAYER1_TABLE.md, not recomputed. Paired per-group mIoU/margin deltas and CMSA change counts are unavailable, not zero. score_probe.py is ready to produce them after a complete prediction freeze; it was not executed.

Run20260919-174012-cma-cycle036-crop-zero passed the remote tensor self-check and runner's pre-load manifest/runtime-input/checkpoint/control-prediction/source checks. The four checkpoint shards loaded on CPU; model.bfloat16().cuda() then failed before hook installation/forward. Error: other process708005 used45.31GiB on visible CUDA0 (physicalGPU1); only13.31MiB free, allocation32MiB failed. Exit1 at17:40:45+08. Follow-up nvidia-smi reported GPU0 47,319/49,140MiB and GPU1 46,459/49,140MiB used. Remote outputs/cycle036 had no files. No inference, scoring, checkpoint change, precision change or competing-process termination occurred.

Initial status had ~24GiB free, above historical17.32GB allocated peak; another workload grew before model transfer. This is an observed capacity change, not an unavailable checkpoint/data substitution. run.log/run_meta.json preserve the failure. An earlier shell-only CRLF failure (run173909) was corrected before this attempt, producing no model output.

Next executable step when capacity materially changes: use the same run.sh once, preserving the unchanged protocol and control. No automatic rapid retries. Current task remains scientifically incomplete; a resource-blocker delivery is permitted by REVIEW035. Matched retraining is not supported or rejected by this failed launch.

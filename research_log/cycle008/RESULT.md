# Cycle008 — validation PASS, mixed diagnostic result

| Metric | Base C | Adapted C | Base D | Adapted D |
|---|---:|---:|---:|---:|
| target_miou | 0.919718 | 0.920641 | 0.691113 | 0.713416 |
| cmsa | 0.940000 | 0.960000 | 0.580000 | 0.620000 |
| memory_fidelity | 0.990000 | 1.000000 | 0.890000 | 0.930000 |
| mean_identity_margin | 0.919718 | 0.920641 | 0.662931 | 0.698181 |
| median_identity_margin | 0.963190 | 0.961073 | 0.847463 | 0.856061 |
| identity_error_rate | 0.000000 | 0.000000 | 0.030000 | 0.020000 |

Diagnostic30, base reused from Cycle003:

| Metric | Base C (reused) | Adapted C | Base DD (reused) | Adapted DD |
|---|---:|---:|---:|---:|
| target_miou | 0.943108 | 0.938923 | 0.631131 | 0.639207 |
| cmsa | 0.966667 | 0.966667 | 0.466667 | 0.433333 |
| memory_fidelity | 1.000000 | 1.000000 | 0.816667 | 0.816667 |
| mean_identity_margin | 0.943108 | 0.938923 | 0.621451 | 0.639008 |
| median_identity_margin | 0.958937 | 0.952980 | 0.814648 | 0.829782 |
| identity_error_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Validation degraded mIoU+0.022302/CMSA+0.04; diagnostic degraded mIoU+0.008076 but CMSA−0.033333. No repeated candidate selection.

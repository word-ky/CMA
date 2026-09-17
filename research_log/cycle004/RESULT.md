# Cycle 004: main-image quality × REF-appearance quality

C=clean, D=target15_b; first letter main image, second REF appearance. Same frozen 30 groups, 60 references/cell, supplied geometry. CC/DD reused from Cycle 003; only CD/DC newly run.

| Metric | CC | CD | DC | DD |
|---|---:|---:|---:|---:|
| target_miou | 0.943108 | 0.929784 | 0.630192 | 0.631131 |
| cmsa | 0.966667 | 0.933333 | 0.466667 | 0.466667 |
| memory_fidelity | 1.000000 | 1.000000 | 0.816667 | 0.816667 |
| mean_identity_margin | 0.943108 | 0.929784 | 0.614167 | 0.621451 |
| median_identity_margin | 0.958937 | 0.957880 | 0.811157 | 0.814648 |
| identity_error_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| wrong_gt_correct_count | 0.000000 | 0.000000 | 4.000000 | 3.000000 |
| both_iou_zero_count | 0.000000 | 0.000000 | 7.000000 | 8.000000 |

## Continuous decomposition

| Metric | Main: DC−CC | REF: CD−CC | Interaction: DD−DC−CD+CC |
|---|---:|---:|---:|
| target_miou | -0.312917 | -0.013325 | +0.014264 |
| mean_identity_margin | -0.328941 | -0.013325 | +0.020608 |

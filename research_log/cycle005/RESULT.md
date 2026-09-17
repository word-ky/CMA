# Cycle 005: frozen v3-lowseg recovery

Same 30 groups / 60 references; CC and DD reused, only DE newly run.

| Metric | CC | DD | DE |
|---|---:|---:|---:|
| target_miou | 0.943108 | 0.631131 | 0.615758 |
| cmsa | 0.966667 | 0.466667 | 0.433333 |
| memory_fidelity | 1.000000 | 0.816667 | 0.800000 |
| mean_identity_margin | 0.943108 | 0.621451 | 0.610108 |
| median_identity_margin | 0.958937 | 0.814648 | 0.816076 |
| identity_error_rate | 0.000000 | 0.000000 | 0.000000 |
| correct_iou_ge_0_5_count | 59.000000 | 43.000000 | 43.000000 |

DE minus DD: target_miou -0.015373; mean_identity_margin -0.011343; cmsa -0.033333; memory_fidelity -0.016667

| Group metric | Improved | Worsened | Tied |
|---|---:|---:|---:|
| target_miou | 18 | 12 | 0 |
| cmsa | 0 | 1 | 29 |

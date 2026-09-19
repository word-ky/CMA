# No-run preregistration: incremental crop feature under fixed localization

Status: **NON-EXECUTABLE DESIGN; budget/asset binding unresolved; experiment not authorized or run.** This freezes the scientific contrast and existing identifiers, not nonexistent execution facts. No new split or checkpoint search.

| Contract field | Proposed fixed specification |
|---|---|
| Estimand | Full-minus-zero difference after equal continuation from the same w15 starting state, conditional on supplied miner localization and prior w15 learning |
| Initialization | Both arms use identical w15 merged files in audit_receipt.json; both reset optimizer/scheduler and initialize any fresh adapters with the same RNG state |
| Training manifest | Existing Cycle023 reconstructed JSONL SHA256 9e9c6c7f288e7c7a50c9537d6394091f411724871b6caedaf8c8ed9541f001d1; identical row order/sampling and resolved assets; full asset binding UNKNOWN, not newly reconstructed here |
| Validation manifest | NONE used for selection or scheduling in this design. Historical launcher validation file hash UNKNOWN. No new validation split chosen |
| Evaluation manifest | Existing Cycle025 execution manifest SHA256 b1717faabc5d84a77556e7a09171b4323dc584ab318318f64ace79b198b930b7; same 50 groups/100 identities, clean and target15_b, protocol_receipt.json. Already inspected dataset; never call a new blind test |
| Full arm | c=mean(encode_images(ref_crop)); ref_input_fcs(c+bbox_features) |
| Zero arm | Compute same c then replace with zeros_like(c) before bbox addition, in training AND inference |
| Fixed inputs/routes | RGB, semantic query/tokenization, crop preparation, miner mask, bbox, targets, valid rows; full-image vision, bbox projection, 16x16 mask/bbox output route, auxiliary reconstruction and pre-REF output indexing remain available |
| Losses | Proposed explicit common CE=1, BCE=2, Dice=.5, REF reconstruction=1, rank=1.5, rank margin=.05. These are prospective choices from source/config, not proof of historical CE/BCE/Dice overrides |
| Optimizer | Proposed common fresh AdamW, lr=3e-5, betas=(.9,.95), weight_decay=0, clipping=1; same WarmupDecayLR. Total updates and warmup count TBD before any execution, not selected from Cycle025 scores |
| Budget | TBD equal total optimizer updates, no early stop/model selection; identical batch1, accumulation1 and proposed single-device BF16; final-step checkpoint only. No automatic use of launcher20-step default |
| Trainability | Same source recipe documented in Cycle033; effective list and fresh adapter initialization must be bound before execution; no additional modules |
| Seed policy | Proposed seed0 for both arms across Python/NumPy/Torch/CUDA/sampler/workers; identical initial sample order. Explicit setup currently missing in launcher. One seed would be descriptive, not variance/significance evidence |
| Outcomes | Existing evaluator only: target mIoU, CMSA, Memory Fidelity, IER and mean/median identity margin, separately clean/degraded; report all paired differences, never select one favorable metric |
| Prediction protocol | Same target-free runtime and offline frozen-prediction scoring as Cycle025; pseudo-target access during preparation is disclosed |

Positive means a consistent full-arm benefit on the predeclared outcome vector (higher mIoU/CMSA/Fidelity/margin and lower IER); mixed directions must be reported as mixed, without inventing a composite score. A positive effect supports incremental crop-feature utility in this continuation regime. It does not isolate pure identity semantics: the masked crop encodes shape and geometry, and full-image appearance remains in both arms.

Null means no observed difference, not proof that appearance is useless; one fixed budget/seed, prior crop-trained initialization and redundant geometry limit that inference. Negative means the zero arm performs better under the fixed settings; report it without claiming universal harm from appearance. No success threshold, post-result tuning, group deletion, rerun selection or Cycle025-based checkpoint selection. No prediction of improvement.

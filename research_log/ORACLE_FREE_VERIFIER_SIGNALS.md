# Oracle-free verifier signals — inventory, not a trained verifier

Scope: signals useful for identity-memory correctness, degradation or useful feedback. No new taxonomy or verifier training in Cycle 001. A computable signal is not automatically a calibrated probability of correct identity.

| Signal | Current implementation | Additional work before verifier use |
|---|---|---|
| Mask area, area ratio, components, largest-component ratio, edge contact, bbox | `stage3_rule_controller_v3_seg_local_enhance.py::mask_stats`; policy sees most scalar stats through sanitizer, but not bbox | Compute directly from the candidate; calibrate against identity errors rather than assuming large/connected means correct |
| Brightness, contrast, Laplacian std, dark/bright pixel ratio | `image_quality`; present in some trajectory entries, not guaranteed freshly recomputed at every action | Recompute for the current image and memory crop with correct provenance; distinguish degradation from real scene appearance |
| Helmet center inside miner/head region | `helmet_geometry`, used in REF prediction quality | Formula itself needs no GT, but current anchor was selected by GT; recompute with a genuinely predicted, non-oracle-selected reference |
| SAM predicted IoU | `LISA.py` receives `iou_predictions` from mask decoder but discards them in the inference return; runner returns binary masks | Expose score along with mask and empirically test calibration for LISA text prompts and degraded imagery; not currently available to controller |
| Mask/logit stability under small perturbations or thresholds | Not currently computed; runner thresholds logits at zero and discards them | Save logits or run specified perturbations; count extra calls. Consistently wrong masks can be stable |
| Direct-vs-REF agreement | Individual predictions occur, but no corresponding consistency signal is exposed and some state is overwritten | Retain predictions on the same image/version and compare them; agreement does not establish correct identity |
| Before/after enhancement agreement and reference identity continuity | Some old/new images and masks exist; current deltas use GT IoU, not observable consistency | Store matched snapshots and compute mask/appearance changes without GT; appearance changes from enhancement can be benign or harmful |
| Crop visual-language similarity | CLIP reference appearance features are encoded for REF; no verifier similarity score is currently exposed | Choose/cache appropriate image/text embeddings and compute similarity; current projected LLM/SAM features are not automatically calibrated CLIP similarity |
| Counterfactual sensitivity | Offline same-query/different-REF evaluation exists; new scorer measures labeled fidelity | At inference, candidate-memory interventions and prediction divergence require extra calls. Without labeled identity targets, divergence alone cannot tell correct switching from arbitrary changes |
| Version/provenance coherence | New standalone memory API stores source image/action/step and snapshots | Wire into a future executor to keep mask/image/features from the same evidence version; provenance is bookkeeping, not a learned reliability value |

Never reuse `quality['iou']`, local GT-IoU delta, oracle-chosen best target, oracle-chosen anchor, or offline CMSA as an inference verifier input. A consistency score computed from an oracle-selected candidate is still indirectly oracle-dependent.

Shortest next experiment: obtain matched per-identity predictions with a fixed GT-reference diagnostic and a separately labeled predicted-reference diagnostic; quantify where fidelity drops under one existing degradation preset. Decide which signal can address the observed failure before training or integrating a verifier.

# Matched continuation feasibility — metadata/source audit only

**A matched future pair is constructible in principle, but no fully executable, asset-bound training contract is recovered or frozen here.** This cycle reads JSON receipts and source text only. It does not verify present weight bytes, image availability or a training forward.

## One common initialization candidate

Use the same restored merged base-w15 checkpoint for BOTH newly trained arms, with fresh identical optimizer/scheduler states and identically seeded new adapters. Exact file digests are copied from cycle023/w15_training_exposure_audit.json into audit_receipt.json, and agree with the Cycle025 checkpoint provenance. This is a conditional continuation experiment from an already crop-trained model; it cannot estimate from-scratch necessity or erase historical exposure. Historic w15 itself is NOT the full-arm result. Both arms need equal new training.

The launcher names a pre-w15 pilot500 initialization, but its exact recoverable weight identity is not established by these receipts. Do not substitute that path as a verified common base. No new checkpoint is selected/downloaded.

## Data binding

One existing candidate is the Cycle023 reconstructed combined training JSONL: 13,455 rows (9,724 regular + 3,731 counterfactual), 8,123 unique image-byte hashes, SHA256 `9e9c6c7f288e7c7a50c9537d6394091f411724871b6caedaf8c8ed9541f001d1`. Full remote path is in audit_receipt.json. Its byte identity is documented; it is not the exact run-bound historical w15 manifest. A single shared resolved asset mapping and mask-byte manifest is still needed before this can be an executable pair: historical image/mask paths are not evidence that all training assets are currently accessible. No images or masks were opened to test that here.

The existing launcher names regular_holdout JSONL for validation; this audit has no bound validation manifest/hash and will not invent one or select a new split. Validation is unused in the proposed fixed-budget/no-selection comparison. Cycle025 execution manifest is already hash-bound as the descriptive evaluation set; it is known, not a fresh blind test.

## Documented versus unknown

| Field | Documented evidence | Historical / future limitation |
|---|---|---|
| Loss configuration | w15 receipts: rank 1.5, margin .05, REF reconstruction 1; train_ds.py defaults CE 1, BCE 2, Dice .5 | Exact historical overrides of CE/BCE/Dice UNKNOWN; proposed explicit values need not reconstruct history |
| Optimizer | train_ds.py:334–353 AdamW, weight decay 0, beta defaults .9/.95, WarmupDecayLR, 100 warmup steps, clipping 1 | Exact w15 optimizer state/environment overrides UNKNOWN; future pair would reset both identically |
| Budget/LR | Launcher defaults epochs 1, steps 20, LR 3e-5, batch 1, accumulation 1 | Defaults are not w15 facts or a selected future budget. In particular 20-step default is shorter than 100-step warmup. Future total updates/warmup require an explicit common choice |
| Trainability | Cycle033 TRAINABLE_PATH_AUDIT documents LoRA plus named REF/SAM-decoder/text modules, frozen vision/projector | Historical exact optimizer membership UNKNOWN; future recipe must bind effective parameter list and same newly initialized adapters |
| Seed | Current train_ds.py has no explicit seed-setting match from source search | Future explicit common RNG/sampler/worker setup needed; not patched here |
| Precision/interface | Frozen CMA BF16, crop mode, gated_add .5, prompt norm 20, clamp 50 | Future both arms preserve these settings; no shift correction |

## Local source change, not applied

At cmllm_remote/third_party/LISA/model/LISA.py:764, immediately after `crop_features = self.encode_images(ref_images).mean(dim=1)`, a single persisted experiment option would conditionally assign `torch.zeros_like(crop_features)`. The option must reach both training and evaluation construction and be recorded in the resulting checkpoint configuration. Keep line765 bbox projection and line768 summation unchanged. Do not zero `valid_embeddings` or all REF embeddings: those would also remove bbox information.

For a future execution, additionally bind explicit common seeds and the effective training settings. The smallest missing data provenance item is a shared executable training manifest-to-restored-assets mapping with image/mask hashes. The smallest configuration item is one explicit common launch specification (budget/warmup, seed setup, trainable parameters and runtime versions). Exact historical optimizer recovery is not required for fresh matched continuation, but is required to claim exact historical retraining. No such claim is made.

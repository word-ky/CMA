# CHATGPT REVIEW 016 — loader collision is real; authorize one exact weight-fidelity repair, then finish the same four-trial smoke

Reviewed commit: `c4a7f0ff93aae5f7fddcd9443b8f32a853b65634`.

## Decision

Cycle 016 is accepted as a **correct scientific stop**, not as a SegLLM performance result. Codex did the right thing by refusing to run the four-trial baseline smoke after native `from_pretrained` returned a model with two trained fusion parameters silently omitted.

The evidence supports a narrow compatibility diagnosis rather than an architecture/checkpoint mismatch:

- the pinned SegLLM checkpoint contains `...b_attn.gamma_l` with shape `[768]` and `...b_attn.gamma_v` with shape `[256]` in the raw safetensors shard;
- the pinned SegLLM/HIPIE source declares and uses those exact `gamma_*` parameters;
- Transformers 4.46.1 applies a legacy global `gamma -> weight` compatibility rename during loading;
- the outer load consequently reports `gamma_l/gamma_v` newly initialized and `weight_l/weight_v` unused;
- no forward, prediction, score, prompt change, checkpoint substitution, or CMA modification was made after detecting the mismatch.

This is therefore a **loader-name collision in the current runtime**, not evidence that the released SegLLM weights are absent. It is worth repairing because SegLLM is still the closest native historical-memory competitor, and obtaining one trustworthy external comparison is more valuable now than inventing another CMA module.

However, the earlier ancillary HIPIE warning (`tgt_embed.weight` 900x256 versus 100x256 plus unused MaskDINO keys) must not be hand-waved away. The outer SegLLM load appears to reduce the final warning set to the two gamma parameters, but before we call the model weight-faithful we need one explicit final-state audit proving that the outer checkpoint actually overwrites any ancillary mismatch that matters.

## Critical research/engineering feedback

### 1. Approve only a two-tensor compatibility repair; do not change the model or dependency stack

Do **not** downgrade/upgrade Transformers, switch SegLLM revisions, replace HIPIE, alter the model class, or introduce a broad key-renaming rule.

The safest repair is post-load and exact:

1. run the same pinned native `from_pretrained` load;
2. open the already verified checkpoint shard read-only;
3. read exactly the two raw tensors under their original names:
   - `model.segmentator.hipie.detr.detr.transformer.encoder.vl_layers.0.b_attn.gamma_l`;
   - `model.segmentator.hipie.detr.detr.transformer.encoder.vl_layers.0.b_attn.gamma_v`;
4. locate the identically named model parameters;
5. under `torch.no_grad()`, copy only those two tensors into the model, using only the model's intended dtype/device cast;
6. make no other tensor assignment.

This is a runtime compatibility shim, not a model modification: the final parameter values must be exactly the released checkpoint values after the deterministic dtype cast.

### 2. Prove equality after repair, not merely disappearance of warnings

Before any forward, save a `weight_fidelity_receipt.json` containing for both gamma parameters:

- raw checkpoint key, shard, shape, dtype, SHA256 of raw bytes;
- destination model key, shape, dtype/device;
- SHA256 of the expected casted tensor bytes;
- SHA256 of the actual loaded model tensor bytes;
- `torch.equal(actual.cpu(), raw.to(actual.dtype).cpu()) == True` (or an equivalent exact equality assertion after the intended cast).

If either name/shape differs or exact post-cast equality fails, stop. Do not use a tolerance to conceal a wrong assignment.

### 3. Close the ancillary-weight ambiguity before calling the load accepted

Audit the final in-memory model after the outer SegLLM checkpoint load for the previously warned `tgt_embed.weight` path.

- Determine whether the outer SegLLM checkpoint contains the corresponding final model key.
- If it does, prove the final in-memory tensor equals the outer checkpoint tensor after the intended dtype cast, using the same exact-equality/hash receipt.
- If it does not, or if the final model retains a random/truncated 100-query tensor because the ancillary 900-query load failed, stop and report this as a second weight-fidelity blocker.

Also record the final top-level missing/newly-initialized/unexpected key set. After the two-gamma compatibility repair, there must be no unexplained newly initialized parameter in the executed relational path.

Do not try to make all ancillary MaskDINO warnings disappear if those tensors are later overwritten or are not part of the final executed checkpoint; document them separately rather than broad-remapping keys.

### 4. If weight fidelity passes, resume exactly the already frozen four-trial smoke

Do not turn Cycle 017 into another setup cycle. Once the load is accepted, execute exactly:

- clean/A;
- clean/B;
- `target15_b`/A;
- `target15_b`/B;

for frozen group `cf_c9029d27c0483181` with the already fixed prompt:

`Segment the mining helmet worn by instance 1.[REF:1]`

No prompt, threshold, checkpoint, image preprocessing, supplied-memory representation, or output-selection change is allowed.

The runtime receipt must prove:

- `[REF:1]` resolves to historical index `0`;
- A/B within each condition have byte-identical current image and identical tokenized conversation;
- only the seeded supplied miner historical state changes between A/B;
- MASK-ENCODE / BOX-ENCODE inputs correspond to the intended supplied miner state;
- the exported mask is the final relational MASK-DECODE output, not the placeholder/history turn;
- no helmet target is opened before predictions are frozen;
- unpadding/resizing returns one binary mask at the original HxW exactly once.

Poor or empty predictions are valid smoke outputs. Do not retry or paraphrase based on appearance.

### 5. Keep interpretation bounded

Even if the four masks run successfully, this still establishes only **native weight-faithful wiring compatibility**. A single-group score is optional scorer plumbing and must be labelled `single-group smoke / not performance evidence`.

Do not run full val50 in the same cycle. The next review should decide whether the frozen full SegLLM val50 comparison is worth the compute based on a clean wiring receipt.

---

# CYCLE 017 — one focused hour

## Goal

Apply one exact two-gamma loader compatibility shim, prove final weight fidelity including the ancillary `tgt_embed.weight` ambiguity, and—if and only if the final model is weight-faithful—finish the unchanged four-trial supplied-memory SegLLM smoke.

## Step 1 — exact compatibility shim

Implement the smallest isolated loader helper under `cmllm_remote/external_baselines/segllm/` or the existing loader file. It may only restore the two exact `gamma_l/gamma_v` tensors from the already pinned raw safetensors checkpoint after native load. No dependency/version/model/checkpoint changes.

Before repair, capture the native loader warning set. After repair, assert exact equality to the checkpoint-after-cast values and save `weight_fidelity_receipt.json`.

## Step 2 — ancillary final-state audit

Resolve the `tgt_embed.weight` warning scientifically rather than assuming it is harmless. Prove whether the final outer checkpoint restores the executed model's target embedding. Save key/shape/dtype/hash/equality evidence. Stop if the final executed tensor cannot be tied exactly to a released checkpoint value or if any other unexplained newly initialized executed-path parameter remains.

## Step 3 — same four forwards only if Steps 1–2 pass

Reuse the already prepared environment/assets and execute the frozen four trials. Save:

- `pre_score_checks.json`;
- four original-resolution binary masks;
- `smoke_predictions.json`;
- `smoke_receipt.json`;
- per-trial inference time and peak VRAM;
- exact historical-index / injected-state / final-output-order proof.

Do not reinstall/redownload, do not edit prompts/thresholds, and do not run full val50.

## Step 4 — optional scorer plumbing

Only after all predictions are frozen, run the unchanged CMF scorer if useful. Mark any result strictly as `single-group smoke / not performance evidence`. No result-driven rerun.

## Stop conditions

Stop with an exact blocker if:

- the two gamma tensors cannot be restored with exact post-cast equality;
- the final `tgt_embed.weight` or another executed-path parameter remains untraceable/uninitialized;
- historical reference injection is ambiguous;
- output ordering/mapping is ambiguous;
- A6000 OOMs or native forward fails.

Do not switch model versions, baseline, checkpoint, prompt, threshold, segmentator, or CMA method.

## Non-goals

No full val50. No CMA adaptation. No new recovery module. No second baseline. No agent/controller/verifier/RL. No diagnostic30/confirmation30. No prompt/threshold sweep.

## Deliverable

Append `CODEX UPDATE 017` with the exact repair code/path, pre/post weight-fidelity evidence, ancillary final-state audit, four-trial execution status, runtime injection/export proof, resource use, and the exact blocker if incomplete.

If and only if all four trials are reproducible and weight-faithful, recommend exactly one next task: **run the frozen full val50 clean + target15_b SegLLM comparison with no further interface changes**.

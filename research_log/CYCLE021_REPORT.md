# Cycle021 — train-only MCR calibration fails the frozen validation gate; retire Layer 2

Implements REVIEW020 (`f057f09`). The final authorized verifier attempt is complete. Training calibration is feasible, but its single frozen threshold **tau=0.38560267857142855** fails on val50: degraded accepted CMSA28/40=70% and caught failures9/21, below75% and11. No second threshold, verifier feature, recovery action or confirmation evaluation was tried. Layer1 remains the primary research result.

## Prediction provenance and unchanged method

No complete protocol-matching train300 base-w15 prediction manifest existed in the project outputs inventory. Generated all300 original Cycle006 training groups under clean and deterministic target15_b, preserving both identities and their shared query. This required600 batched group forwards /1200 identity predictions. The original image-byte-disjoint split and holdout exclusion receipt was retained; exact group order, train manifest hash and all300 original image hashes were verified. Miner mask hashes and model file hashes are saved in inference_provenance.json. Train manifest SHA25682bd35c5137ec14ee68109cbf0736ca9c7875c54280567056354d90008bd3d05. No split rebuilding or subset selection.

Reused unadapted base-w15 from the existing merged checkpoint, same tokenizer/model initialization, BF16, max length512, v1_multiround, REF crop, seed0 and deterministic condition_image/group_seed as Cycle008. No adaptation checkpoint was loaded. Existing build_item gained only read_targets=False; default behavior is preserved. This mode does not read helmet masks. Its test proves all model inputs except target labels match the legacy builder. Before forward, all masks_list labels become zero placeholders; the actual model hook checks inference=True and zero labels every time. LISA's inference branch returns pred_masks before any target-dependent loss computation. cv2 raster reads were restricted to training images and supplied miner masks.

Unchanged Cycle020 MCR: H_j is the clipped upper60% of the mask-derived miner bbox using helmet_geometry coordinates and pixel-center rasterization. S_ij=area(P_i intersect H_j)/max(area(P_i),1); m_A=S_AA-S_AB, m_B=S_BB-S_BA, g=min(m_A,m_B). Original structural validity requires nonempty predictions, positive own support and both margins>0. Calibrated ACCEPT additionally requires g>=tau. One shared positive tau serves both conditions. No MCR feature changes.

All600 train prediction/score records froze before training labels were opened. The separate unchanged CMF scorer computed train labels after verifying prediction/miner hashes. Candidate thresholds were exactly the322 unique positive training g values. Each was checked against the six unchanged calibration constraints;116 were feasible. Chose the smallest feasible threshold, with no secondary objective. The complete table is in calibration_receipt.json and calibration_candidates.csv.

## Training calibration at the chosen threshold

| Condition | ACCEPT coverage | Accepted CMSA | Failure recall | Abstention precision | success->accept / success->abstain / failure->accept / failure->abstain |
|---|---:|---:|---:|---:|---|
| clean |290/300 (96.67%)|280/290 (96.55%)|7/17 (41.18%)|7/10 (70%)|280 /3 /10 /7|
| target15_b |233/300 (77.67%)|175/233 (75.11%)|67/125 (53.60%)|67/67 (100%)|175 /0 /58 /67|

All six training constraints pass. These are calibration-set measurements, not independent validation.

## One frozen validation evaluation

The calibration receipt was hashed before loading the frozen Cycle020 validation actions/scores. Those scores were reused unchanged, with zero validation model forwards. Before opening validation target-bearing manifests, froze100 new actions with clean48 ACCEPT/2 ABSTAIN and degraded40 ACCEPT/10 ABSTAIN. Verified the chronological ordering train-score freeze < tau freeze < val-action freeze < val scoring, plus all receipt hashes. The unchanged CMF scorer then ran once per condition.

| Condition | ACCEPT coverage | Accepted mIoU | Accepted CMSA | Accepted Fidelity | Mean margin | Median margin | Accepted IER |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean |48/50 (96%)|92.91%|46/48 (95.83%)|96/96 (100%)|0.929103|0.963190|0/96|
| target15_b |40/50 (80%)|75.76%|28/40 (70%)|78/80 (97.5%)|0.757415|0.859231|0/80|

| Condition | success->accept | success->abstain | failure->accept | failure->abstain | Failure recall | Abstention precision |
|---|---:|---:|---:|---:|---:|---:|
| clean |46|1|2|1|1/3 (33.33%)|1/2 (50%)|
| target15_b |28|1|12|9|9/21 (42.86%)|9/10 (90%)|

**Validation gate FAIL:** accepted degraded CMSA70%<75%;9 failures caught<11. Both actions, degraded coverage, abstention precision, clean coverage and clean accepted CMSA requirements pass. The rule still accepts12 CMSA failures despite accepted IER0, consistent with identity association being insufficient to certify localization. No conditional metric is a segmentation improvement: frozen full-set masks/metrics remain unchanged (clean mIoU91.97%, CMSA47/50; degraded mIoU69.11%, CMSA29/50).

## Tests, runtime and limitations

Six baseline tests passed;10 focused tests passed; full CPU suite54passed20.52s. Target-free builder equivalence, smallest-feasible selection, inclusive threshold ties, structural validity, infeasible clean preservation, nonpositive candidate exclusion and prior MCR target-independence checks passed. Initial full-suite collection missed cmllm_remote/src on PYTHONPATH; fixing the test environment resolved it without code change. First launcher205705 failed before Python due to CRLF shell line endings; normalized to LF and launched205906, with no successful predictions rerun. Existing NVML mismatch warning persisted but CUDA inference completed; no driver change.

Successful run20260918-205906-cma-cycle021-train-predictions ended21:04:22+08 exit0. Runtime from provenance start to prediction freeze: 306.03s; peak allocated CUDA memory17322769920bytes. Local replay verifies all1200 prediction file hashes. Result archive2411187bytes, SHA256 `f7444c0868f545592411179da587ee79523b1b5d4d7d999fabeb2e1985486d5a`. Raw masks remain in local outputs/cycle021_replay and remote outputs/cycle021; compact receipts, matrices and reports are under research_log/cycle021. Training-score data remain calibration evidence; val50 is a repeatedly studied development split with reconstructed pseudo labels and supplied memories, not an untouched generalization/safety test.

## Exactly one next recommendation

**Retire Layer-2 verifier/scheduler development and retain Layer1 as the paper core**, preserving Cycles019–021 as negative/partial evidence. No further threshold, learned calibrator, recovery action, Qwen/RL or confirmation evaluation is justified by this failed gate. Await a new research review; no next experimental cycle started.

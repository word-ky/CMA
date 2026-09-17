# CHATGPT REVIEW 007 — enhancer line closed; direct w15 adaptation is now the highest-value move

Cycle 007 is accepted as a clean negative result and the enhancer line is now closed. The experiment was sufficiently controlled to make that decision: the exact frozen 300/50 split was reused, the identity-aware REF loss had a verified non-zero gradient path through the main SAM image branch, w15 stayed frozen, one fixed training attempt was run, and the predeclared validation gate failed. New v4 worsened target mIoU from 0.685059 to 0.678440 and CMSA from 0.56 to 0.50, with Fidelity and mean identity margin also decreasing. Do not spend another cycle on teacher choice, enhancer resolution, another loss weight, or a second restoration architecture.

This negative result is scientifically useful because it narrows the Layer-1 bottleneck. Cycle 004 already showed that `target15_b` damage is dominated by the **main image / target-observation path**, not REF appearance. Cycle 005 showed that the frozen enhancer has almost no exploitable oracle upside, and Cycle 007 shows that even an identity-aware enhancer fine-tune does not reverse the loss. The next intervention should therefore act on the memory-grounded segmenter's ability to decode degraded visual evidence, while explicitly preserving the already-strong clean identity behavior.

The training-resolution mismatch in Cycle 007 (384 training versus native/max-side-1024 evaluation) is a plausible engineering contributor, but with the user's time constraint it is not worth reopening the enhancer branch. Treat it as a limitation of that bounded negative experiment, not a reason for another enhancer search.

## CYCLE 008 — one-hour Codex task

### Goal

Run **one bounded direct degradation adaptation of w15** using paired clean/degraded counterfactual groups. No enhancer and no agent. The target is exactly the primary paper claim: retain identity-memory switching on clean inputs while recovering performance under complex coal-mine degradation.

### Priority A — establish the correct no-enhancer validation baseline

On the exact frozen Cycle-006 validation 50 groups, evaluate the **unadapted w15 with no enhancer** under:

- `C`: clean main image + clean REF appearance;
- `D`: `target15_b` main image + `target15_b` REF appearance.

Use supplied miner identity geometry, the existing CMF scorer, the same queries/identity ordering/seed, and one prediction per memory identity. Report target mIoU, CMSA, Memory Fidelity, mean/median identity margin and IER. If these exact no-enhancer outputs are already cached, verify hashes/config and reuse them instead of rerunning.

Do not use old-v3/new-v4 enhancer numbers as the baseline for this cycle.

### Priority B — minimal trainable subset only

Initialize from the recovered w15 checkpoint. Freeze the LLM backbone, CLIP/vision tower, SAM image encoder and all unrelated parameters. Unfreeze only the existing segmentation/reference adaptation surfaces:

- `visual_model.mask_decoder`;
- `text_hidden_fcs`;
- `ref_hidden_fcs`;
- `ref_visual_fcs`;
- `ref_input_bbox_fcs`;
- `ref_input_fcs`;
- `ref_embedding_scale`.

Before optimization, print/record the exact trainable parameter names and count. Run one fixed-batch smoke backward and verify intended modules receive finite non-zero gradients while frozen modules remain gradient-free. Do not add LoRA to the backbone or unfreeze the SAM image encoder in this cycle.

### Priority C — paired clean/degraded counterfactual objective

Use the exact frozen Cycle-006 300 training groups and 50 validation groups; no split rebuilding and no diagnostic30 access during training/model selection.

For every training group, construct both observations with the same identity pair, masks, bboxes, targets and query:

- clean observation;
- deterministic `target15_b` observation with the existing group seed.

For each condition, run the normal w15 REF-conditioned target prediction for all identities. Optimize only:

1. `L_seg_clean`: helmet BCE + Dice on clean predictions;
2. `L_seg_deg`: helmet BCE + Dice on degraded predictions;
3. `L_rank_clean` and `L_rank_deg`: existing soft-IoU counterfactual ranking with margin `0.05`;
4. `L_cons`: degradation-consistency distillation from the same-memory clean prediction to the degraded prediction, implemented as binary cross-entropy with degraded logits against `sigmoid(clean_logits.detach())` for each identity.

Use the fixed total objective:

`L = 0.5*(L_seg_clean + L_seg_deg) + 0.5*(L_rank_clean + L_rank_deg) + 0.25*L_cons`.

This is not a new architecture. `L_cons` directly expresses the desired Layer-1 behavior: under the same entity memory, the degraded observation should preserve the clean target grounding.

Predeclare and do not sweep:

- AdamW;
- learning rate `5e-6`;
- weight decay `1e-4`;
- gradient clip norm `1.0`;
- one epoch / exactly 300 prescribed groups;
- seed `20260528`;
- one final checkpoint only, no intermediate checkpoint selection.

If memory requires sequential clean/degraded forwards, that is acceptable; do not reduce the dataset after seeing runtime or losses.

### Priority D — fixed validation gate

Evaluate both the unadapted base w15 and the single adapted checkpoint on the exact 50 validation groups under **both clean and target15_b, with no enhancer**.

Primary gate:

- degraded target mIoU improves by at least `+0.02` over unadapted w15;
- degraded CMSA does not decrease;
- clean target mIoU drops by no more than `0.01`;
- clean CMSA drops by at most one group (`0.02` absolute on 50 groups).

Also report Fidelity and identity margin for interpretation; do not change the gate after seeing them.

If the gate passes, run the untouched Cycle-003 diagnostic30 **once** with the adapted checkpoint, no enhancer, and report base-DD versus adapted-DD plus clean preservation. If the gate fails, do not touch diagnostic30 and do not start a hyperparameter sweep in the same cycle.

### Non-goals

- no enhancer work of any kind;
- no agent/controller/verifier/RL;
- no baseline ports;
- no new degradation family;
- no predicted-memory protocol;
- no SAM image-encoder unfreezing;
- no LLM/CLIP LoRA;
- no repeated learning-rate/loss-weight attempts;
- no GT-based candidate selection.

### Deliverable

Append `CODEX UPDATE 008` to `CHATGPT_CODEX_BRIDGE.md` with: no-enhancer base C/D validation metrics; exact trainable parameter list/count; gradient-smoke result; single training command/run; base-vs-adapted C/D validation table; pass/fail under the fixed gate; and, only if passed, the single diagnostic30 result. End with exactly one recommended next one-hour task.

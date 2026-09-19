# CHATGPT REVIEW 032 — REF alignment confirmed; now separate identity semantics from supplied geometry

Reviewed commits: `bc746143de17c51ec9ae007449fd3aaff7bc8d71` and delivery-only follow-up `6a0013c8f0010d807951820ab25af7bd0af8fc29`.

## Decision

Cycle032 is accepted as a useful **zero-model mechanism audit**. It followed the user's current instruction to pause manuscript work and focus on understanding the implemented method. No production model source, checkpoint, scientific result, baseline, sample set, scorer, or Layer-2 controller was changed. The second commit only records successful A6000 synchronization; it adds no new scientific result.

The cycle materially improves our understanding of the frozen w15 causal graph. The earlier static concern from Review031 is now supported by a deterministic source-extracted test rather than by prose inspection alone.

## 1. Confirmed finding: output `ref_hidden_fcs` does not read the injected REF identity under the intended single-front-image/256-patch layout

The audit directly executes the repository's actual token-mask helper AST and the three manual mask-assignment statements from `LISA.py`, replacing only `.cuda()` with `.cpu()`.

For the representative layout used by the frozen path:

- true injected REF slot: 260;
- current output-side `ref_hidden_fcs` selection: 259;
- auxiliary unshifted REF reconstruction: 260;
- current SEG-associated selections: 257 and 261, while the true SEG token positions are 258 and 262.

Therefore the current output `ref_hidden_fcs` branch reads the hidden state **one position before REF**, whereas appearance+bbox identity is injected at the actual REF position.

Because the decoder is causal, the pre-REF hidden state cannot directly read the later REF embedding. This validates the correction made in Review031: frozen w15 must not be taught as if the output branch performs a second semantic readout of the injected identity.

## 2. Stronger result: the pre-REF branch is identity-invariant for the frozen A/B counterfactual protocol

Cycle032 goes beyond token positions and audits the actual conversation construction for all 50 frozen Cycle025 groups. The two A/B rows have identical role/message content before the REF embedding; pair identity is not encoded into the text prefix. The same main image is also shared.

Therefore, for the frozen same-image/same-query intervention, the `ref_hidden_fcs` readout has no A/B-specific input source before REF. It can encode shared scene/query context, but it cannot be attributed to the selected miner identity.

This is an important conceptual cleanup. The current output context is better understood as:

`shared pre-REF context + identity-specific supplied geometry`

rather than:

`identity semantic hidden + identity geometry`.

## 3. Correct frozen-w15 mechanism

The most source-faithful current interpretation is:

**appearance+bbox REF injection -> later helmet SEG semantics that can depend on the chosen identity, plus explicit miner mask+bbox geometry prompting -> SAM, trained with ordinary mask losses + auxiliary reference reconstruction + counterfactual identity ranking.**

In short:

**REF-conditioned SEG semantics + localized worker-geometry prompting + counterfactual identity ranking.**

The true REF slot can causally influence the later second-round SEG state because REF occurs before that SEG position. The output geometry branch also receives the supplied miner mask and bbox directly. The unshifted REF auxiliary reconstruction uses the real REF position during training.

The current `ref_hidden_fcs` branch may still contribute useful shared context, but it is not a second identity-semantic path in the frozen architecture.

## 4. Critical scientific implication: Cycle025 identifies the supplied-memory bundle, not any individual memory channel

This is now the most important point for the memory-centric vision story.

The Cycle025 intervention changes a **bundle** of identity-localized state:

- same-condition miner appearance crop;
- miner bbox;
- miner mask.

The output-side geometry branch receives mask+bbox explicitly, while appearance+bbox also affect the true REF input and later SEG semantics. Consequently, the strong Cycle025 CMSA/Fidelity result proves that the whole supplied localized entity state controls the target output. It does **not** yet prove that:

- appearance semantics alone are necessary;
- the REF semantic route alone causes the gain;
- geometry is merely auxiliary;
- the model can identify the worker without supplied localization.

This does not weaken the measured system result, but it changes the scientifically precise claim. The validated object is **supplied identity-localized entity memory**, not a weak identity token and not autonomous semantic identity memory.

This distinction matters especially because the output geometry branch is identity-specific and directly reaches the SAM prompt. It is therefore plausible that a substantial part of the counterfactual switch behavior comes from supplied localization. We currently have no matched component ablation that quantifies the relative contribution of crop/REF semantics versus mask/bbox geometry, so do not assign isolated causal credit.

## 5. What Cycle032 advances in the memory-centric agentic vision plan

It advances Layer 1 by making the causal story more honest and technically sharper. We now know exactly which parts of the frozen system can carry identity-specific information and which cannot.

It does **not** advance Layer 2. The validated memory is still externally supplied and localized; there is no successful autonomous memory write/retrieve/verify/update controller in the frozen evidence. The earlier Layer-2 verifier/controller attempts remain negative/partial and should stay retired while the user is learning the method.

The right conceptual separation is now:

- **what is supplied:** miner crop + mask + bbox;
- **what is learned:** how that supplied state influences later relational helmet segmentation;
- **what is trained explicitly:** helmet localization, auxiliary reference reconstruction, and counterfactual identity preference;
- **what is not validated:** autonomous identity acquisition, persistent memory management, agentic repair, or geometry-free identity reasoning.

## 6. Remaining technical uncertainty from Cycle032

The audit deliberately did not load the actual tokenizer or vision-tower config at runtime. Its exact numeric REF/SEG expanded positions are conditional on the single-front-image, 256-patch layout hard-coded by the current manual `255`-zero offset.

The frozen w15 config names `openai/clip-vit-large-patch14`, but Cycle032 did not independently read the cached vision config/tokenizer to verify the actual patch count and serialized token positions used by the experiment runtime. The source-level conclusion is compelling, but the next teaching cycle should close this last runtime/config gap without loading model weights.

Do not use this caveat to hot-fix production code or restart performance experiments. It is simply the last step needed to make the mechanism explanation fully tied to the frozen runtime contract.

---

# CYCLE 033 — one focused hour: what identity information is supplied, what is learned, and where gradients go

## Goal

Finish the user's method-learning phase by turning the current causal graph into an exact **supplied-vs-learned identity-channel map**. No paper writing and no performance experiment.

## A. Close the exact runtime token/vision-layout uncertainty without model weights

Create `research_log/cycle033/RUNTIME_LAYOUT_VERIFICATION.md` and a small script that loads **configuration/tokenizer only**, not model weights and not images.

Verify from the actual recovered environment used by the frozen w15 path:

1. the CLIP vision config image size, patch size, and resulting `num_patches`;
2. the actual LLaVA tokenizer/template serialization of the frozen two-round conversation;
3. the actual raw REF and SEG token indices/positions for representative A/B rows;
4. the expanded REF/SEG positions implied by the verified patch count;
5. whether the current manual 255-offset mask matches that runtime layout exactly.

If tokenizer/config assets cannot be read without loading weights, report `UNKNOWN` rather than inferring values from model-name convention.

No hidden-state or prediction computation is needed.

## B. Build an identity-channel bundle map

Create `research_log/cycle033/IDENTITY_CHANNEL_BUNDLE.md`.

For the frozen Cycle025 A/B intervention, list every input that is:

- identical across A/B;
- changed across A/B;
- used only in training;
- used at inference.

At minimum distinguish:

- main RGB image;
- text/query prefix;
- appearance crop;
- bbox in REF input encoding;
- supplied miner mask;
- bbox in output geometry encoding;
- target helmet mask (training/scoring only).

For each changed channel, trace exactly where it first enters the computation and whether it can influence the final helmet mask.

End with one explicit identifiability statement:

> the current counterfactual experiment identifies the **joint supplied-memory bundle**, not the isolated causal contribution of any one channel.

## C. Audit what is actually trainable in frozen w15 training

Create `research_log/cycle033/TRAINABLE_PATH_AUDIT.md` from the historical training launcher/config/source.

List each relevant module and whether it was trainable/frozen in the documented w15 setup, with source/config evidence where available:

- LLM / LoRA or other adapted language parameters;
- multimodal projector / vision tower;
- `ref_input_*` projection/gate parameters;
- `text_hidden_fcs`;
- `ref_hidden_fcs`;
- `ref_visual_fcs`;
- SAM prompt/mask decoder parameters;
- auxiliary reference reconstruction branch parameters.

If exact historical `requires_grad` state is not recoverable, mark it UNKNOWN instead of assuming.

## D. Draw the training gradient graph

Create `research_log/cycle033/TRAINING_GRADIENT_GRAPH.md` showing which losses can, by graph connectivity, send gradients toward which trainable modules:

- CE;
- helmet BCE/Dice;
- reference reconstruction BCE/Dice;
- counterfactual rank loss.

Separate:

- graph connectivity;
- documented trainability;
- measured causal importance.

Do not convert a possible gradient path into an ablation claim.

## E. Update the teaching pack, not the manuscript

Add one concise teaching note answering exactly these four questions:

1. What does the user/model receive for Miner A?
2. Which A-specific signals reach SAM directly?
3. Which A-specific signals can reach the later SEG state through the causal decoder?
4. Which training losses pressure A and B to remain distinguishable?

This should be usable to teach the method in the user's requested structure: actual need -> motivation/contribution/solution -> concrete flow.

## Hard non-goals

- no model weights loaded;
- no inference/training/scoring;
- no ablation or performance test;
- no production-source patch;
- no paper/Related Work/venue work;
- no Layer-2 agent/controller resurrection;
- no claim that geometry or appearance is individually responsible for the Cycle025 gap.

## Deliverable

Before `CODEX UPDATE 033`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 033` with the runtime-layout result, identity-channel bundle map, trainable-parameter audit, gradient graph, corrected teaching summary, and exactly one next recommendation limited to continuing method teaching or, only if the user explicitly asks, resuming scientific development.

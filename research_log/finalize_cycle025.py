"""Publish the first immutable comparison; no model execution or rescoring."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1];work=root/'research_log/cycle025'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):p.write_bytes(s.encode('utf-8'))
r=load(work/'scoring/comparison.json');v=load(work/'local_verification.json');a=load(work/'asset_integrity_audit.json')
assets=load(work/'frozen_assets.json');assert len(assets['assets'])==250
log=root/'outputs/cycle025_replay/runs/20260919-015702-cma-cycle025-final-confirmation/train.log'
text=log.read_text(encoding='utf-8');assert '[autodl] exit_code=0' in text
write(work/'run_stdout.txt',text)
table='| Method | Condition | mIoU | CMSA | Fidelity | IER | Mean margin | Median margin | Empty masks | Groups / identities |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for key,s in r['summaries'].items():
 method='CMA base-w15' if key.startswith('cma_') else 'SegLLM pinned';condition='target15_b' if key.endswith('target15_b') else 'clean'
 table+=f"| {method} | {condition} | {s['target_miou']*100:.2f}% | {s['cmsa_count']}/50 ({s['cmsa']*100:.0f}%) | {s['fidelity_count']}/100 ({s['memory_fidelity']*100:.0f}%) | {s['ier_count']}/100 ({s['identity_error_rate']*100:.0f}%) | {s['mean_identity_margin']:.6f} | {s['median_identity_margin']:.6f} | {s['empty_masks']}/100 | 50 / 100 |\n"
def delta_table(block):
 t='| Comparison | mIoU pp | CMSA pp | Fidelity pp | IER pp | Mean-margin delta | Median-margin delta | Empty-mask delta |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for key,item in block.items():
  d=item['aggregate'];t+=f"| {item['direction']} | {100*d['target_miou']:+.2f} | {100*d['cmsa']:+.2f} | {100*d['memory_fidelity']:+.2f} | {100*d['identity_error_rate']:+.2f} | {d['mean_identity_margin']:+.6f} | {d['median_identity_margin']:+.6f} | {d['empty_masks']:+d} |\n"
 return t
report='''# Cycle025 — last frozen model evaluation

Status: **DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION**. Review024/598bd3c completed. The first result is retained with no performance gate, tuning branch, replacement or subgroup rerun. Model experimentation for this paper stops after this cycle; Layer2 remains retired.

## Pre-model asset integrity

Used exactly the Cycle024 manifest SHA256 `de4335243a9188cf0e2b55046668eca338fc85ec0e8a5b3c609d490f07e8503f` and registry SHA256 `837088966a5ef3fc6f24515318b4e4f0743a9c76e54a08fa1fcd8cbc17768aa7`. Original order preserved. **50/50 groups valid, 0 invalid, 0 replacements**; all100identities passed mechanical metadata and mask checks. Evaluated source counts remain23review groups and27accepted-pair combinations. The review groups retain their original `all_pairs_have_clean_episode=false`; no episode-QC status was promoted or rewritten.

Pre-restoration checks verified exact source bytes, absence from the exclusion registry, exact accepted-high pair records, distinct miner/helmet identities, source image/annotation dimensions and in-bounds boxes. Helmet annotation boxes were checked against stored metadata with0.002pixel serialization tolerance, not a quality threshold. The unchanged Cycle022-compatible SAM-B recipe restored50images and200nonempty image-aligned pseudo masks. Source annotation IDs, prompt boxes and original pair paths are retained in frozen_assets.json. The final complete valid/invalid audit was written and hashed before either CMA or SegLLM loaded; its hash is linked through selection_receipt.json into protocol_receipt.json. No quality-score sweep, morphology filter or visual ranking was used.

Asset preparation necessarily reads helmet pseudo-targets for mechanical validity and hashing. The guarantee is prediction-before-scoring isolation with target-free inference, not that targets were never accessed in preparation.

## Frozen methods and actual prediction isolation

Reused the already reproduced Cycle022 SAM recovery, CMA base-w15 loader/build_item/predict_item, SegLLM native interface and unchanged CMF scorer. Four cycle-specific entrypoints are exact copies after cycle022→cycle025 path/documentation substitution; adapter_equivalence.json records normalized source hashes and inverse-substitution equality. No scientific kernel changed; all frozen kernel hashes equal Cycle022. No extra baseline/model smoke was run.

CMA uses the same base-w15 model-file hashes, BF16,512tokens,v1_multiround,REF crop,seed0 and group-seed degradation convention. Actual CMA condition pixels were asserted identical to the prepared SegLLM pixels. Within each group the observation/query stay fixed while supplied miner identity changes. Every CMA forward asserts inference=True and zero target placeholders; raster reads are restricted to source/memory/prepared observations.

SegLLM retains source HEAD `4593a069f09628ce3a5b46e657f5417fefd7be46`, checkpoint revision `095e0637fcba015a02c0686f67b848c79c3cc80b`, the pinned three weight-shard hashes, two-gamma compatibility path, native `[REF:1]` prompt/history index0, and unchanged mask/box encoding, threshold and resolution mapping. Tracked source diff is empty. Source/weight audit was executed before inference in this cycle (its reused receipt scope text says post-run, but its timestamp and ordered run log establish actual pre-run execution).

CMA produced100group-condition forwards /200identity masks; SegLLM produced200native forwards. Both complete prediction manifests froze before scorer target access. The scorer verified prediction hashes, identity order and target-free runtime raster paths first, then opened target-bearing manifests and invoked the unchanged CMF scorer once per method/condition. Independent local replay verified all400raw prediction hashes and the full audit→protocol→both model starts→both prediction freezes→scoring time chain. No failed-looking group was rerun or removed.

## First frozen results

'''+table+'''
Paired CMA minus SegLLM on identical groups (rates in percentage points):

'''+delta_table(r['method_deltas'])+'''
Paired degraded minus clean, with no source-stratified performance selection:

'''+delta_table(r['degradation_deltas'])+'''
## Interpretation and fixed claim boundary

On this preselected documented-protocol-disjoint set, CMA retains a substantial system-level advantage under the fixed compound stressor: **+31.08points mIoU,+56points CMSA,+38points Fidelity and−35points IER** relative to pinned SegLLM. Its degraded CMSA is29/50, leaving21groups that fail the strict two-identity criterion. CMA's clean-to-degraded mIoU decline is20.75points, versus6.26for SegLLM; the evidence supports higher absolute degraded performance, not smaller degradation sensitivity.

Allowed wording: **On a counterfactual set selected before inference and disjoint by source-image SHA256 from the reconstructed documented final-stage training and historical evaluation registries, frozen CMA retains stronger supplied-identity-memory performance than the pinned released SegLLM system under the fixed target15_b stressor.**

Exact run-bound w15 and ancestor/pretraining exposure remain UNKNOWN. This is not exact training-unseen generalization, a controlled architecture-only superiority result, autonomous memory writing/repair, or broad corruption/safety robustness. Targets are reconstructed pseudo labels; identity memories are externally supplied; the review/constructed-group QC provenance remains a limitation. Old Cycle018/022 results and their overlap disclosures are unchanged and are not pooled with this set.

## Execution, tests and artifacts

Run `20260919-015702-cma-cycle025-final-confirmation` started2026-09-19 01:57:07+08 and completed02:06:47+08,exit0. No prediction retries. NVML reported a driver/library mismatch, but direct CUDA tensor allocation on A6000 succeeded and both inference jobs completed; no driver/environment modification was made. Initial full CPU test collection lacked the existing src package on PYTHONPATH; setting it resolved collection without a code change.15focused existing tests passed16.71s,3new mechanical asset tests passed, and the complete54-test cmllm_remote suite passed26.12s. Syntax checks and path-only adapter equivalence passed. No further training/checkpoint roundtrip applies to this frozen evaluation.

The archive outputs/cycle025_results.tgz contains raw predictions and compact run/evaluation receipts; source images and restored target/memory masks remain on A6000 shared/data/cycle025. Input images, model weights and prepared PNGs are not added to Git. The raw prediction replay is under local outputs/cycle025_replay. All per-group IoU matrices and paired deltas remain in scoring/comparison.json and the four CMF reports. Runtime durations below are metadata, not a controlled speed comparison.

'''
report+=f"CMA load-to-freeze {v['cma_load_to_freeze_seconds']:.2f}s; SegLLM {v['segllm_load_to_freeze_seconds']:.2f}s. Archive {v['archive_bytes']:,}bytes, SHA256 `{v['archive_sha256']}`. Asset audit SHA256 `{sha(work/'asset_integrity_audit.json')}`; protocol SHA256 `{sha(work/'protocol_receipt.json')}`.\n\n"
report+='Exactly one next recommendation: produce the paper tables, figures and text around this frozen documented-protocol-disjoint confirmation, retaining the exposure and pseudo-label/QC limitations. No further model experiment or in-repository subset search is authorized by this cycle.\n'
write(root/'research_log/CYCLE025_REPORT.md',report);write(work/'HANDOFF.md',report)
bridge=root/'CHATGPT_CODEX_BRIDGE.md';b=bridge.read_text(encoding='utf-8');assert '# CODEX UPDATE 025' not in b
review=(root/'research_log/CHATGPT_REVIEW_024.md').read_text(encoding='utf-8')
write(bridge,b+'\n\n'+review+'\n\n# CODEX UPDATE 025\n\n'+report)
with (work/'progress.md').open('a',encoding='utf-8') as f:f.write('\n'+datetime.now(timezone.utc).isoformat()+' — Run015702exit0;50valid,0invalid.54CPUtests passed26.12s after src PYTHONPATH correction;3asset tests passed.400prediction hashes and timing/code/weight invariants verified locally. First degraded result:CMA67.53%/58%CMSA vs SegLLM36.44%/2%. Review024 mirrored verbatim before UPDATE025; all further model experimentation stops.\n')
print(table)

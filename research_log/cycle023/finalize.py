"""Freeze documentation from existing receipts; never run model evaluation."""
import csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
root=Path(__file__).resolve().parents[2]
d=root/'research_log/cycle023'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s): p.write_bytes(s.encode('utf-8'))
a=json.loads((d/'protocol_reconstruction_audit.json').read_text(encoding='utf-8'))
a['historical_holdout_usage']={'source':'research_log/cycle023/historical_log_excerpts.json','lines':[877,878,879,881,882],'finding':'Full 921 counterfactual holdout evaluated during historical checkpoint comparisons; Cycle022 is not historically blind.'}
a['scope']='Reconstructed regular+counterfactual final-stage training only; ancestor/pretraining exposure not exhaustively audited.'
a['missing_artifacts']=['Exact historical w15 run-bound training manifest/hash','Run-specific environment/config proving no training-manifest overrides']
a['interpretation']='UNKNOWN exact exposure, with positive byte-level overlaps under the reconstructed documented protocol. Neither PROVEN_ZERO_OVERLAP nor PROTOCOL_INFERRED_ZERO_OVERLAP is supported.'
write(d/'w15_training_exposure_audit.json',json.dumps(a,indent=2)+'\n')
rows=list(csv.reader((d/'MAIN_LAYER1_TABLE.csv').open(encoding='utf-8',newline='')))
prov=json.loads((d/'table_provenance.json').read_text())
assert len(rows)==9
for row,src in zip(rows[1:],prov['sources']):
 p=root/src['path']; assert sha(p)==src['sha256']
 s=json.loads(p.read_text())['summaries'][src['pointer'].split('/')[-1]]
 assert [float(v) for v in row[3:]]==[s[k] for k in prov['metrics']]
assert a['reconstructed_manifest']['unique_image_bytes']==8123
assert a['reconstructed_protocol_intersections']['cycle022_confirmation50']['overlap_images']==37
assert a['reconstructed_protocol_intersections']['cycle018_val50']['overlap_images']==50
archive=root/'outputs/cycle023_reconstruction.tgz'
receipt={'time_utc':datetime.now(timezone.utc).isoformat(),'checks':'8 table rows / 64 numeric cells match frozen JSON exactly; input source hashes unchanged; reconstructed overlap counts checked','model_inference':False,'training':False,'archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)},'registry_sha256':sha(d/'reconstructed_training_image_registry.json')}
write(d/'verification.json',json.dumps(receipt,indent=2)+'\n')
report='''# Cycle023 — evidence freeze and training-overlap finding

Mode: claim-audit and numeric-audit. Review022 is the sole task; documentation/provenance only. No inference, training, tuning, new metric, bootstrap or subgroup rescoring.

## Audit outcome

**Exact w15 training exposure: UNKNOWN.** The exact run-specific combined manifest, recorded hash and environment overrides were not recovered. Frozen model-file hashes are in w15_training_exposure_audit.json and match the prior verified transfer. The archived builder and training shell match current code after newline normalization. Archived source inputs reconstruct 13,455 rows (9,724 regular +3,731 counterfactual), matching historical logs, with 8,123 unique image-byte SHA256 values.

This reconstructed documented protocol overlaps **37/50 Cycle022 images and 50/50 Cycle018 images**. It hashes per-instance image paths into buckets, so aliases of the same image bytes can fall into different buckets. Detailed matched training rows and image hashes are retained in protocol_reconstruction_audit.json and reconstructed_training_image_registry.json. This is positive overlap evidence under reconstruction, not merely a missing zero-overlap certificate. Exact historical counts/intersections remain null because run binding is missing. Earlier ancestor/pretraining exposure is not exhaustively covered by the reconstructed final stages.

The historical full 921-group counterfactual holdout was evaluated during checkpoint comparisons, documented in historical_log_excerpts.json lines877–882. Cycle022 freshness is limited to recorded Cycles001–021 iteration history. Neither evaluated split establishes training-unseen generalization or historically blind model selection. No overlapping group was removed or rescored. The original scores remain intact.

Search coverage: local recovery-file inventory, the 1,012-member emergency archive, archived scripts/logs and available A6000 project/shared paths. The exact run manifest/config remains missing; search stops within this documentation cycle. No replacement checkpoint or new evaluation split was created.

## Frozen outputs and checks

MAIN_LAYER1_TABLE.csv/.md retain eight method/condition rows with separate confirmation and development roles. All 64 numeric cells were compared directly to frozen Cycle018/022 summaries, and source hashes/pointers are retained. No pooled estimate or selective MCR metric was added. CLAIM_EVIDENCE_LEDGER.md bounds the causal intervention, system-level comparison, supplied memory, pseudo labels, sensitivity, Layer2 negative result and provenance claims.

Severity: material scientific limitation for training-unseen/generalization wording. Numeric inconsistency: none found in exported table. Citation metadata/context: not applicable; no literature claims added. No-invention status: all results preserved, unknown exact provenance explicitly null. The audit does not quantify how much overlap explains the performance gap.

Artifacts: research_log/cycle023 contains exact model hashes, reconstructed protocol/intersections, image registry, historical excerpts, source equivalence, claim ledger, frozen tables and verification receipt. The full reconstructed JSONL remains in the local outputs/cycle023_replay archive and A6000 outputs/cycle023; its path/hash are in the audit. verification.json identifies the replay archive. No images or weights enter Git.

Exactly one next recommendation: produce paper figures/text using the frozen table and claim ledger, explicitly disclosing reconstructed training overlaps and historical holdout selection. Method development remains frozen; Layer2 remains retired. Scientific interpretation belongs to ChatGPT review.
'''
write(root/'research_log/CYCLE023_REPORT.md',report)
write(d/'HANDOFF.md',report)
bridge=root/'CHATGPT_CODEX_BRIDGE.md'
b=bridge.read_text(encoding='utf-8')
review=(root/'research_log/CHATGPT_REVIEW_022.md').read_text(encoding='utf-8')
assert '# CODEX UPDATE 023' not in b
b+='\n\n'+review+'\n\n# CODEX UPDATE 023\n\n'+report.split('\n',1)[1]+'\n'+(d/'MAIN_LAYER1_TABLE.md').read_text(encoding='utf-8')+'\n'+(d/'CLAIM_EVIDENCE_LEDGER.md').read_text(encoding='utf-8').split('Primary allowed wording: ')[1]
write(bridge,b)
with (d/'progress.md').open('a',encoding='utf-8') as f:f.write('\n'+receipt['time_utc']+' — Closed provenance search; UNKNOWN exact run binding, reconstructed overlaps37/50 and50/50. Exported8frozen rows, checked64numeric cells; preserved all metrics and narrowed claims. Mirrored REVIEW022 verbatim before UPDATE023. No GPU work.\n')
print(json.dumps(receipt,indent=2))

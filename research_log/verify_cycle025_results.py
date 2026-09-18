"""Verify downloaded immutable predictions and timing/hash links, without rescoring."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1]; replay=root/'outputs/cycle025_replay';work=root/'research_log/cycle025'
remote='/home/wenchang/asdasdsad/wjq/coalminellm_recovery_20260917/'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def local(path):return replay/path.removeprefix(remote)
audit=read(work/'asset_integrity_audit.json');protocol=read(work/'protocol_receipt.json');scoring=read(work/'scoring/prediction_freeze_audit.json')
assert audit['valid_groups']==50 and audit['invalid_groups']==0
assert audit['frozen_unix']<protocol['frozen_unix']
selection=read(work/'selection_receipt.json')
assert selection['asset_integrity_audit_sha256']==sha(work/'asset_integrity_audit.json')
assert protocol['selection_sha256']==sha(work/'selection_receipt.json')
assert protocol['asset_receipt_sha256']==sha(work/'frozen_assets.json')
freezes={};count=0
for method in ['cma','segllm']:
 p=replay/'outputs/cycle025'/method/'prediction_freeze.json';f=read(p);freezes[method]=f
 assert protocol['frozen_unix']<f['started_unix']<f['frozen_unix']<scoring['scoring_started_unix']
 assert sha(p)==scoring['methods'][method]['sha256']
 assert len(f['trials'])==200
 for t in f['trials']:
  assert sha(local(t['prediction']))==t['prediction_sha256'];count+=1
 (work/(method+'_prediction_freeze.json')).write_bytes(p.read_bytes())
old=read(root/'research_log/cycle022/protocol_receipt.json')
assert old['frozen_code_hashes']==protocol['frozen_code_hashes']
assert freezes['cma']['checkpoint_provenance']==read(root/'research_log/cycle022/cma_prediction_freeze.json')['checkpoint_provenance']
record={'recorded_utc':datetime.now(timezone.utc).isoformat(),'run_id':'20260919-015702-cma-cycle025-final-confirmation','prediction_hashes_verified':count,'asset_audit_before_both_model_starts':True,'both_prediction_freezes_before_scoring':True,'frozen_kernel_hashes_equal_cycle022':True,'cma_weights_equal_cycle022':True,'groups':50,'identity_predictions_per_method_condition':100,'archive_bytes':(root/'outputs/cycle025_results.tgz').stat().st_size,'archive_sha256':sha(root/'outputs/cycle025_results.tgz'),'cma_load_to_freeze_seconds':freezes['cma']['elapsed_seconds'],'segllm_load_to_freeze_seconds':freezes['segllm']['frozen_unix']-freezes['segllm']['started_unix'],'new_tuning':False,'first_result_retained':True}
(work/'local_verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2));print(json.dumps(read(work/'scoring/comparison.json')['summaries'],indent=2))

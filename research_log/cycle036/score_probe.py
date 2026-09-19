"""Score only frozen crop-zero predictions; pair with unchanged Cycle025 reports."""
import json, hashlib, sys, time
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); work=root/'research_log/cycle036'; old=root/'research_log/cycle025'
sys.path.insert(0,str(root/'code/cmllm/scripts'))
from eval_counterfactual_memory_fidelity import evaluate_file
load=lambda p: json.loads(Path(p).read_text())
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=load(old/'protocol_receipt.json')
assert sha(root/'code/cmllm/scripts/eval_counterfactual_memory_fidelity.py')==protocol['frozen_code_hashes']['scripts/eval_counterfactual_memory_fidelity.py']
specs=load(old/'inference_specs.json'); execution=load(old/'execution_manifest.json')
assert sha(old/'execution_manifest.json')==protocol['execution_manifest_sha256']
control=load(old/'cma_prediction_freeze.json')
freeze_path=root/'outputs/cycle036/crop_zero/prediction_freeze.json'; frozen=load(freeze_path)
assert frozen['control_freeze_sha256']==sha(old/'cma_prediction_freeze.json')
assert frozen['checkpoint_provenance']==control['checkpoint_provenance']
assert frozen['protocol_sha256']==control['protocol_sha256']==sha(old/'protocol_receipt.json')
assert len(frozen['trials'])==len(control['trials'])==200
for f in [control,frozen]:
    for t in f['trials']: assert sha(t['prediction'])==t['prediction_sha256']
allowed={str(Path(p).resolve()) for s in specs for p in s['input_asset_hashes']}
allowed.update(o['image_path'] for g in execution for o in g['preparation']['observations'])
assert set(frozen['raster_reads'])<=allowed
assert time.time()>frozen['frozen_unix']
(work/'prediction_freeze_audit.json').write_text(json.dumps({'scoring_started_unix':time.time(),'crop_zero_freeze_sha256':sha(freeze_path),'control_freeze_sha256':sha(old/'cma_prediction_freeze.json'),'prediction_hashes_verified':True,'target_free_runtime_reads_verified':True},indent=2))
# Targets accessed only after both prediction freezes are checked.
for a in load(old/'frozen_assets.json')['assets']: assert sha(a['path'])==a['sha256']
pairs={r['pair_id']:r for r in map(json.loads,(root/'shared/data/cycle025/helmet_miner_pairs_restored.jsonl').read_text().splitlines())}
result={'label':'frozen inference dependence; zero minus full','conditions':{},'training':False,'tuning':False}
for condition in ['clean','target15_b']:
    grouped={}
    for t in frozen['trials']:
        if t['condition']==condition: grouped.setdefault(t['group_id'],[]).append(t)
    records=[]
    for s in specs:
        ts=grouped[s['group_id']]
        assert [t['entity_id'] for t in ts]==[i['entity_id'] for i in s['identities']]
        records.append(dict(group_id=s['group_id'],image_id=s['source_image'],query=s['semantic_query'],condition=condition,memory_source='supplied_ref',trials=[dict(entity_id=t['entity_id'],prediction=t['prediction'],target=pairs[t['entity_id']]['helmet_mask_path']) for t in ts]))
    path=work/(condition+'_predictions.jsonl'); path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    zero=evaluate_file(path,work/(condition+'_metrics.json'))
    full=load(old/'scoring'/('cma_'+condition+'_metrics.json'))
    rows=[]
    for z,f in zip(zero['groups'],full['groups']):
        assert z['group_id']==f['group_id'] and z['entity_ids']==f['entity_ids']
        rows.append(dict(group_id=z['group_id'],miou_delta=z['target_miou']-f['target_miou'],mean_identity_margin_delta=float(np.mean(z['identity_margin'])-np.mean(f['identity_margin'])),identity_margin_deltas=(np.array(z['identity_margin'])-f['identity_margin']).tolist(),cmsa_full=bool(f['pairs'][0]['success']),cmsa_zero=bool(z['pairs'][0]['success'])))
    keys=['target_miou','cmsa','memory_fidelity','identity_error_rate','mean_identity_margin','median_identity_margin']
    result['conditions'][condition]={'full':full['summary'],'zero':zero['summary'],'delta':{k:zero['summary'][k]-full['summary'][k] for k in keys},'cmsa_lost':sum(r['cmsa_full'] and not r['cmsa_zero'] for r in rows),'cmsa_gained':sum(r['cmsa_zero'] and not r['cmsa_full'] for r in rows),'cmsa_changed':sum(r['cmsa_zero']!=r['cmsa_full'] for r in rows),'paired_groups':rows,'control_metrics_sha256':sha(old/'scoring'/('cma_'+condition+'_metrics.json'))}
(work/'paired_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='paired_groups'} for k,v in result['conditions'].items()},indent=2))

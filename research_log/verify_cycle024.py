"""Independent local check of frozen candidate ordering, exposure and annotation provenance."""
import hashlib,json,itertools
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'research_log/cycle024'
src=Path('D:/work/fightccfa-agin/coalminellm/h100_backup/emergency-20260518-011026/outputs/mr_data/helmet_miner_pseudopair_v1/final_accepted_v1')
def read(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
inventory=json.loads((out/'candidate_inventory.json').read_text())
for n,r in inventory['sources'].items():assert sha(src/n)==r['sha256']
reg=json.loads((out/'exclusion_registry.json').read_text());unique=json.loads((out/'valid_unique_candidate_metadata.json').read_text())
selected=read(out/'documented_protocol_disjoint_candidates.jsonl')
assert len(reg)==9549 and len(unique)==402 and len(selected)==50
assert len({r['image_sha256'] for r in unique})==len(unique)
assert not ({r['image_sha256'] for r in unique}&set(reg))
for r in unique:
 assert r['order_key']==hashlib.sha256(('CYCLE024:'+r['image_sha256']+':'+r['counterfactual_id']).encode()).hexdigest()
assert selected==sorted(unique,key=lambda r:(r['order_key'],r['counterfactual_id']))[:50]
pairs={r['pair_id']:r for r in read(src/'helmet_miner_pairs_accept_high.jsonl')}
groups={r['counterfactual_id']:r for n in ['episodes_counterfactual_review.jsonl','episodes_counterfactual_clean_val.jsonl'] for r in read(src/n)}
qc=[]
for r in selected:
 assert r['pairs']==[pairs[p] for p in r['pair_ids']]
 assert len({p['pseudo_miner_id'] for p in r['pairs']})==2
 assert len({p['helmet_id'] for p in r['pairs']})==2
 assert {p['image_member'] for p in r['pairs']}=={r['image_member']}
 original=groups.get(r['counterfactual_id'])
 qc.append({'counterfactual_id':r['counterfactual_id'],'original_group_record':original,'generation_provenance':None if original else 'Two existing accepted pairs on the same source image; pair IDs and full accepted annotations preserved in frozen manifest.','all_pairs_accepted':True})
dump('selected_group_qc_provenance.json',qc)
roles=json.loads((out/'historical_split_roles.json').read_text());role_sets={}
for r in roles:role_sets.setdefault(r['role'],set()).add(r['image_sha256'])
cross={a+'__'+b:len(role_sets[a]&role_sets[b]) for a,b in itertools.combinations(sorted(role_sets),2)}
rec={'checks':'All10source manifests match archived local SHA256;402unique image hashes absent from registry;50selected equal exact first50 by mandated hash rule; all selected pair records identical to original accepted metadata with distinct miner/helmet identities.',
 'selection_sha256':sha(out/'documented_protocol_disjoint_candidates.jsonl'),'exclusion_registry_sha256':sha(out/'exclusion_registry.json'),
 'inventory_sha256':sha(out/'candidate_inventory.json'),'selected_sources':dict(Counter(r['source'] for r in selected)),
 'selected_original_group_qc':dict(Counter(str(r['original_group_record']['qc']) for r in qc if r['original_group_record'])),
 'pairwise_historical_role_intersections':cross,'history_files_scanned':len(json.loads((out/'history_scan_receipt.json').read_text())['files']),
 'new_model_forwards':0,'masks_regenerated':False,'exact_historical_exposure':'UNKNOWN',
 'archive':{'path':'outputs/cycle024_inventory.tgz','bytes':(root/'outputs/cycle024_inventory.tgz').stat().st_size,'sha256':sha(root/'outputs/cycle024_inventory.tgz')}}
dump('local_verification.json',rec);print(json.dumps(rec,indent=2))

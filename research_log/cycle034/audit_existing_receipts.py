"""Read frozen JSON/source receipts only; no images, weights, predictions or scorer."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2]; P=R/'research_log/cycle025'
def load(n):return json.loads((P/n).read_text())
specs=load('inference_specs.json');cma=load('cma_prediction_freeze.json');seg=load('segllm_prediction_freeze.json')
audit=load('scoring/prediction_freeze_audit.json');protocol=load('protocol_receipt.json')
assert len(specs)==50 and len(cma['trials'])==len(seg['trials'])==200
assert cma['zero_target_hook_passed'] and cma['same_condition_pixels_as_segllm']
assert not seg['helmet_targets_read'] and not seg['scorer_called']
assert audit['target_free_runtime_reads_verified'] and audit['source_identity_order_verified']
assert max(cma['frozen_unix'],seg['frozen_unix'])<audit['scoring_started_unix']
key=lambda t:(t['group_id'],t['condition'],t['entity_id'])
assert {key(t) for t in cma['trials']}=={key(t) for t in seg['trials']}
bykey={key(t):t for t in seg['trials']};checked=0
allowed=set()
for s in specs:
 gid=s['group_id'];prepared=load(f'prepared/{gid}/preparation_receipt.json')
 assert prepared['semantic_query']==s['semantic_query'] and prepared['native_prompt']==s['native_prompt']
 allowed.update(s['input_asset_hashes'])
 clean,degraded=prepared['observations']
 assert clean['identities']==degraded['identities']
 for obs in prepared['observations']:
  allowed.add(obs['image_path'])
  for identity in obs['identities']:
   original=next(x for x in s['identities'] if x['entity_id']==identity['entity_id'])
   assert identity['miner_mask_path']==original['miner_mask_path']
   assert identity['miner_mask_sha256']==s['input_asset_hashes'][identity['miner_mask_path']]
   assert identity['miner_bbox_xyxy']==original['miner_bbox_xyxy']
   trial=bykey[(gid,obs['condition'],identity['entity_id'])]
   assert trial['main_rgb_sha256']==obs['rgb_sha256']
   assert trial['supplied_bbox_xyxy']==identity['miner_bbox_xyxy']
   assert trial['injected_tensor_hashes_verified']
   assert trial['appearance_sha256']==trial['injected_appearance_sha256']
   assert trial['bbox_sha256']==trial['injected_bbox_sha256']
   checked+=1
 for identity in s['identities']:
  a=bykey[(gid,'clean',identity['entity_id'])];b=bykey[(gid,'target15_b',identity['entity_id'])]
  assert a['bbox_sha256']==b['bbox_sha256'] and a['appearance_sha256']!=b['appearance_sha256']
for method in ['cma','segllm']:
 assert set(audit['methods'][method]['raster_reads'])<=allowed
inputs=['protocol_receipt.json','inference_specs.json','cma_prediction_freeze.json','segllm_prediction_freeze.json','scoring/prediction_freeze_audit.json','local_verification.json']
result={'status':'PASS_METADATA_ONLY','groups':50,'paired_keys_per_method':200,'segllm_injected_trials_verified':checked,
        'same_raw_identity_sources':True,'CMA_pixel_equality':'recorded Cycle025 assertion; not re-read images',
        'prepared_geometry_fixed_across_conditions':True,'native_bbox_fixed_across_conditions':True,'native_appearance_changes_across_conditions':True,
        'runtime_raster_paths_within_target_free_allowlist':True,'both_freezes_before_scoring':True,
        'cma_freeze_unix':cma['frozen_unix'],'segllm_freeze_unix':seg['frozen_unix'],'scoring_start_unix':audit['scoring_started_unix'],
        'source_sha256':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in inputs},
        'model_calls':0,'scorer_calls':0,'images_or_predictions_opened':False}
(Path(__file__).parent/'input_contract_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

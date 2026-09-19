"""Mechanical pre-model asset audit for the frozen Cycle024 selection."""
import argparse,hashlib,io,json,re,time,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

SELECTION_SHA='e5635de999933389f69b170132d2fa82464f98044773973946eff327cb62672e'
QUERY='Based on the miner mask from the previous round, segment only the mining helmet worn by that miner.'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write_rows(p,r):p.write_text(''.join(json.dumps(x)+'\n' for x in r))
def mask_reason(mask,shape):
 if mask.shape!=shape:return 'mask_not_image_aligned'
 if not mask.any():return 'empty_mask'
 return None

def main(root,stage):
 root=Path(root);work=root/'research_log/cycle039';work.mkdir(exist_ok=True)
 frozen=root/'research_log/cycle039/candidate_manifest.jsonl'
 assert sha(frozen)==SELECTION_SHA
 selected=rows(frozen);assert len(selected)==352
 exclusion_path=root/'research_log/cycle039/exclusion_registry.json'
 assert sha(exclusion_path)=='17aaec67ea302dab77b489d1fb2bcf5d2ea8aca83e1df6a03eba5fa7fdfdc780'
 exclusion=json.loads(exclusion_path.read_text())
 pairs_path=root/'shared/data/final_accepted_v1/helmet_miner_pairs_accept_high.jsonl'
 pairs={p['pair_id']:p for p in rows(pairs_path)}
 if stage=='pre':
  old_reviews={g['counterfactual_id']:g for g in rows(root/'shared/data/final_accepted_v1/episodes_counterfactual_review.jsonl')}
  qc_records=[{'counterfactual_id':g['counterfactual_id'],'original_group_record':old_reviews.get(g['counterfactual_id']),'generation_provenance':g['source'],'all_pairs_accepted':True} for g in selected]
  (work/'selected_group_qc_provenance.json').write_text(json.dumps(qc_records,indent=2)+'\n')
 qc={r['counterfactual_id']:r for r in json.loads((root/'research_log/cycle039/selected_group_qc_provenance.json').read_text())}
 if stage=='pre':
  audit=[];valid=[]
  with zipfile.ZipFile(root/'shared/source/mining_helmet.zip') as z:
   annotations={}
   for name in z.namelist():
    if name.endswith('.json'):
     obj=json.loads(z.read(name))
     if isinstance(obj,dict) and 'annotations' in obj and 'images' in obj:
      images={i['id']:i for i in obj['images']}
      for a in obj['annotations']:annotations[(Path(images[a['image_id']]['file_name']).name,int(a['id']))]=(images[a['image_id']],a)
   for g in selected:
    reasons=[]
    try:
     raw=z.read(g['image_member']);assert hashlib.sha256(raw).hexdigest()==g['image_sha256'],'source_byte_mismatch'
     assert g['image_sha256'] not in exclusion,'registry_overlap'
     w,h=Image.open(io.BytesIO(raw)).size
     ps=[pairs[p] for p in g['pair_ids']];assert ps==g['pairs'],'accepted_pair_record_mismatch'
     assert len(set(g['pair_ids']))==len({p['pseudo_miner_id'] for p in ps})==len({p['helmet_id'] for p in ps})==2,'non_distinct_identity'
     for p in ps:
      assert p['image_member']==g['image_member'] and p['image_size']==[w,h],'source_member_or_size_mismatch'
      for k in ['miner_bbox_xyxy','helmet_bbox_xyxy']:
       b=p[k];assert len(b)==4 and np.isfinite(b).all() and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h,'box_out_of_bounds'
      ann_id=int(re.search(r'-ann(\d+)-',p['image_path']).group(1))
      meta,ann=annotations[(Path(g['image_member']).name,ann_id)]
      assert (meta['width'],meta['height'])==(w,h),'annotation_size_mismatch'
      x,y,bw,bh=map(float,ann['bbox']);x,y=max(0.,min(x,w-1.)),max(0.,min(y,h-1.));bw,bh=max(1.,min(bw,w-x)),max(1.,min(bh,h-y))
      assert np.allclose([x,y,x+bw,y+bh],p['helmet_bbox_xyxy'],rtol=0,atol=.002),'annotation_pair_box_mismatch'
    except (AssertionError,KeyError,ValueError,OSError,AttributeError) as e:reasons.append(str(e))
    audit.append({'group_id':g['counterfactual_id'],'source':g['source'],'original_qc':qc[g['counterfactual_id']], 'valid':not reasons,'reasons':reasons})
    if not reasons:
     valid.append({'counterfactual_id':g['counterfactual_id'],'image_member':g['image_member'],'image_path':g['pairs'][0]['image_path'],'pair_ids':g['pair_ids'],'same_round2_query':QUERY})
  write_rows(work/'selected_source_groups.jsonl',valid)
  (work/'pre_restoration_audit.json').write_text(json.dumps({'frozen_unix':time.time(),'selection_sha256':sha(frozen),'accepted_pair_manifest_sha256':sha(pairs_path),'valid_groups':len(valid),'groups':audit},indent=2)+'\n')
  print('PRE_RESTORATION_VALID',len(valid),flush=True)
  if len(valid)<1:raise SystemExit('STOP_NO_VALID_ASSETS')
 else:
  pre=json.loads((work/'pre_restoration_audit.json').read_text());data=root/'shared/data/cycle039'
  restored=rows(data/'counterfactual_selected_restored.jsonl');rp={p['pair_id']:p for p in rows(data/'helmet_miner_pairs_restored.jsonl')}
  assets=json.loads((work/'frozen_assets.json').read_text());by_pair={}
  for a in assets['assets']:
   assert sha(a['path'])==a['sha256']
   if 'pair_id' in a:by_pair[(a['pair_id'],a['type'])]=a
  records=[];valid=[];selected_by={g['counterfactual_id']:g for g in selected}
  for record in pre['groups']:
   record=dict(record);reasons=list(record['reasons']);gid=record['group_id'];g=selected_by[gid]
   if not reasons:
    try:
     restored_g=next(r for r in restored if r['counterfactual_id']==gid)
     assert restored_g['pair_ids']==g['pair_ids'],'restored_pair_order'
     assert sha(restored_g['image_path'])==g['image_sha256'],'restored_source_hash'
     w,h=Image.open(restored_g['image_path']).size;miner_hashes=[];target_hashes=[]
     for pid in g['pair_ids']:
      for kind in ['miner','helmet']:
       p=rp[pid][kind+'_mask_path'];reason=mask_reason(np.asarray(Image.open(p))>0,(h,w))
       assert reason is None,(pid,kind,reason)
       a=by_pair[(pid,kind+'_mask')];assert a['path']==p and a['source_path']==pairs[pid][kind+'_mask_path'],'mask_pair_provenance'
       if kind=='helmet':target_hashes.append(sha(p))
       if kind=='miner':
        assert a['prompt_box']==pairs[pid]['miner_bbox_xyxy'],'memory_box_provenance'
        miner_hashes.append(sha(p))
     assert len(set(miner_hashes))==2,'identical_miner_memories'
     assert len(set(target_hashes))==2,'identical_helmet_targets'
     valid.append(restored_g)
    except (AssertionError,KeyError,ValueError,OSError,StopIteration) as e:reasons.append(str(e))
   record.update(valid=not reasons,reasons=reasons);records.append(record)
  write_rows(data/'counterfactual_selected_restored.jsonl',valid)
  # Preserve the original restoration receipt; the final group manifest can exclude only mechanically invalid groups.
  receipt={'frozen_unix':time.time(),'status':'VALID_FOR_FROZEN_INFERENCE' if len(valid)>=1 else 'STOP_NO_VALID_ASSETS','valid_groups':len(valid),'invalid_groups':352-len(valid),'groups':records,'selection_sha256':sha(frozen),'exclusion_registry_sha256':sha(exclusion_path),'restoration_receipt_sha256':sha(work/'frozen_assets.json'),'pre_restoration_audit_sha256':sha(work/'pre_restoration_audit.json'),'valid_group_manifest_sha256':sha(data/'counterfactual_selected_restored.jsonl'),'source_counts':dict(Counter(r['source'] for r in records if r['valid'])),'thresholds_added':False,'replacement_groups':0,'cma_segllm_loads_before_audit':0}
  (work/'asset_integrity_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
  valid_ids={r['counterfactual_id'] for r in valid}
  (work/'selection_receipt.json').write_text(json.dumps({'groups':len(valid),'selected':[dict(g,group_id=g['counterfactual_id']) for g in selected if g['counterfactual_id'] in valid_ids],'zero_image_byte_overlap':True,'original_selection_sha256':sha(frozen),'asset_integrity_audit_sha256':sha(work/'asset_integrity_audit.json')},indent=2)+'\n')
  print('ASSET_AUDIT_FROZEN',len(valid),352-len(valid),flush=True)
  if len(valid)<1:raise SystemExit('STOP_NO_VALID_ASSETS')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('stage',choices=['pre','post']);a=p.parse_args();main(a.root,a.stage)

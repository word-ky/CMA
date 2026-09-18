"""Inventory documented-exposure-disjoint CMF candidates; CPU/file operations only."""
import hashlib,itertools,json,re,sys,tarfile,zipfile
from collections import defaultdict,Counter
from pathlib import Path
from check_image_split_integrity import audit_roles
root=Path(sys.argv[1]);out=root/'research_log/cycle024';out.mkdir(exist_ok=True)
data=root/'shared/data/final_accepted_v1'
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (out/name).write_text(json.dumps(obj,indent=2)+'\n')
def bucket(row):return int(hashlib.md5(row['image_path'].encode()).hexdigest()[:8],16)%10
sources={p.name:read(p) for p in sorted(data.glob('*.jsonl'))}
source_receipts={p.name:{'path':str(p),'sha256':sha(p),'rows':len(sources[p.name])} for p in sorted(data.glob('*.jsonl'))}
image_hash={}; aliases=defaultdict(set); ids=defaultdict(set)
with zipfile.ZipFile(root/'shared/source/mining_helmet.zip') as z:
 for m in z.namelist():
  if Path(m).suffix.lower() in {'.jpg','.jpeg','.png'}:
   h=hashlib.sha256(z.read(m)).hexdigest();image_hash[m]=h;aliases[Path(m).name].add(h)
 zip_inventory={'path':str(root/'shared/source/mining_helmet.zip'),'sha256':sha(root/'shared/source/mining_helmet.zip'),'members':len(z.namelist()),'image_members':len(image_hash),'unique_image_bytes':len(set(image_hash.values())),'annotation_members':[m for m in z.namelist() if Path(m).suffix.lower() in {'.json','.xml','.txt'}]}
for rows in sources.values():
 for r in rows:
  h=image_hash.get(r.get('image_member'))
  if not h:continue
  if r.get('image_path'):aliases[Path(r['image_path']).name].add(h)
  for k in ['counterfactual_id','pair_id','episode_id','helmet_id','pseudo_miner_id']:
   if r.get(k):ids[r[k]].add(h)
  for v in r.get('pair_ids',[]):ids[v].add(h)
registry=defaultdict(lambda:defaultdict(set));roles=[]
def add(h,kind,source):registry[h][kind].add(source)
training=json.loads((root/'research_log/cycle023/reconstructed_training_image_registry.json').read_text())
for row in training:
 h=row['image_sha256'];add(h,'reconstructed_final_stage_training','cycle023/reconstructed_training_image_registry.json')
 roles.append({'image_sha256':h,'role':'reconstructed_training'})
cf=sources['episodes_counterfactual_clean_val.jsonl']
holdout=[r for r in cf if bucket(r)>=8]
regular_holdout=[r for r in sources['episodes_miner_to_helmet_clean_train.jsonl'] if bucket(r)>=8]
assert len(holdout)==921 and len(regular_holdout)==2494
for name,rows in [('historical_cf_holdout',holdout),('historical_regular_holdout',regular_holdout)]:
 for r in rows:
  h=image_hash[r['image_member']];add(h,name,r.get('counterfactual_id',r.get('episode_id')))
  roles.append({'image_sha256':h,'role':name})
# Explicit prior-byte registry covers all earlier recorded cycle uses, even if an alias is unresolved below.
for r in json.loads((root/'research_log/cycle022/used_image_registry.json').read_text())['images']:
 for src in r['sources']:add(r['sha256'],'recorded_cycles001_021',src)
for r in json.loads((root/'research_log/cycle022/selection_receipt.json').read_text())['selected']:
 add(r['image_sha256'],'cycle022_selected',r['group_id'])
known=set(image_hash.values()); scanned=[]
def scan(source,digest,tokens,kind):
 hits=set()
 for t in tokens:
  if t in known:hits.add(t)
  hits.update(ids.get(t,set()));hits.update(aliases.get(t,set()))
 for h in hits:add(h,kind,source)
 scanned.append({'source':source,'sha256':digest,'resolved_source_images':len(hits)})
local=json.loads((out/'local_history_references.json').read_text())
for rec in local['records']:
 if rec['source'].replace('\\','/').endswith('/segmentation/sam_masks.jsonl'):continue # label-generation inventory, not task-model use
 scan(rec['source'],rec['sha256'],rec['tokens'],'local_historical_or_cycle_reference')
for base in [root/'research_log',root/'outputs']:
 for p in sorted(base.rglob('*')):
  if not p.is_file() or p.suffix not in {'.json','.jsonl','.md','.txt','.log'} or 'cycle024' in p.parts or 'cycle024' in p.name.lower():continue
  b=p.read_bytes();text=b.decode('utf-8',errors='replace')
  tokens=set(re.findall(r'(?:cf_|pair_|ep_)[0-9a-f]{16}|\b[0-9a-f]{64}\b|(?<![\w.-])[\w.-]{1,200}\.(?:jpg|jpeg|png)(?![\w.-])',text,re.I))
  scan(str(p.relative_to(root)),hashlib.sha256(b).hexdigest(),tokens,'remote_historical_or_cycle_reference')
dump('exclusion_registry.json',{h:{k:sorted(v) for k,v in sorted(classes.items())} for h,classes in sorted(registry.items())})
dump('history_scan_receipt.json',{'files':scanned,'matching':'Conservative exact source-byte hashes and source-resolved IDs/image basenames; source-pool manifests are inventoried separately, not treated as evaluation use.'})
dump('source_image_hashes.json',image_hash)
dump('historical_split_roles.json',roles)
split=audit_roles(roles);dump('split_integrity_audit.json',split)
class_counts=Counter(k for v in registry.values() for k in v)
pair_sources={}
for name in ['helmet_miner_pairs_accept_high.jsonl','helmet_miner_pairs_review.jsonl','helmet_miner_pairs_reject.jsonl']:
 for r in sources[name]:pair_sources[r['pair_id']]=(r,name)
accepted={r['pair_id']:r for r in sources['helmet_miner_pairs_accept_high.jsonl']}
groups=[]
for name in ['episodes_counterfactual_clean_val.jsonl','episodes_counterfactual_review.jsonl']:
 for r in sources[name]:groups.append((name,r))
# Existing accepted annotations permit new pair combinations; no detector or fabricated pair is introduced.
by_member=defaultdict(list)
for p in accepted.values():by_member[p['image_member']].append(p)
for member,pairs in sorted(by_member.items()):
 for a,b in itertools.combinations(sorted(pairs,key=lambda r:r['pair_id']),2):
  gid='acceptedpair_'+hashlib.sha256((a['pair_id']+':'+b['pair_id']).encode()).hexdigest()[:16]
  groups.append(('accepted_pair_combinations',{'counterfactual_id':gid,'image_member':member,'pair_ids':[a['pair_id'],b['pair_id']]}))
counts=defaultdict(Counter);valid=[];details=[]
for source,r in groups:
 counts[source]['total_groups']+=1;member=r['image_member'];h=image_hash.get(member)
 reason=None;ps=[]
 if h is None:reason='source_image_not_available'
 elif h in registry:reason='excluded_source_image_bytes'
 elif len(r['pair_ids'])!=2 or len(set(r['pair_ids']))!=2 or any(pid not in accepted for pid in r['pair_ids']):reason='not_two_accepted_pairs'
 else:
  ps=[accepted[pid] for pid in r['pair_ids']]
  if len({p['pseudo_miner_id'] for p in ps})!=2 or len({p['helmet_id'] for p in ps})!=2:reason='not_distinct_miner_and_helmet_identities'
  elif any(p['image_member']!=member for p in ps):reason='pair_source_mismatch'
  elif any(len(p.get(k,[]))!=4 or p[k][2]<=p[k][0] or p[k][3]<=p[k][1] for p in ps for k in ['miner_bbox_xyxy','helmet_bbox_xyxy']):reason='invalid_reconstruction_boxes'
  else:
   w,ht=ps[0]['image_size']
   if any(not(0<=p[k][0]<p[k][2]<=w and 0<=p[k][1]<p[k][3]<=ht) for p in ps for k in ['miner_bbox_xyxy','helmet_bbox_xyxy']):reason='reconstruction_box_outside_image'
 if reason:counts[source][reason]+=1
 else:
  counts[source]['valid_before_image_dedup']+=1
  gid=r['counterfactual_id'];valid.append({'counterfactual_id':gid,'image_sha256':h,'image_member':member,'source':source,'pair_ids':r['pair_ids'],'pairs':ps,'order_key':hashlib.sha256(('CYCLE024:'+h+':'+gid).encode()).hexdigest(),'target_provenance':'Existing accepted real helmet box + SAM-B reconstruction; miner memory from existing accepted pseudo-miner box. No masks regenerated/no model run.'})
 details.append({'source':source,'counterfactual_id':r['counterfactual_id'],'image_sha256':h,'reason':reason or 'valid_metadata'})
ordered=sorted(valid,key=lambda r:(r['order_key'],r['counterfactual_id']));unique=[];seen=set()
for r in ordered:
 if r['image_sha256'] in seen:counts[r['source']]['duplicate_valid_source_image']+=1
 else:seen.add(r['image_sha256']);unique.append(r)
enough=len(unique)>=30
selected=unique[:50] if enough else []
if enough:(out/'documented_protocol_disjoint_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
dump('candidate_group_audit.json',details)
dump('valid_unique_candidate_metadata.json',unique)
archives=[]
for p in sorted(root.rglob('*.tgz')):
 if 'cycle024' in p.name:continue
 with tarfile.open(p,'r:gz') as t:
  names=t.getnames();archives.append({'path':str(p.relative_to(root)),'members':len(names),'data_metadata_members':[n for n in names if n.endswith('.jsonl') and ('episode' in n or 'pair' in n)]})
dump('candidate_inventory.json',{'status':'DOCUMENTED_PROTOCOL_DISJOINT_CANDIDATES' if enough else 'INSUFFICIENT_CANDIDATES','conclusion':'ENOUGH_FOR_STRONGER_FROZEN_EVAL' if enough else 'NEED_NEW_INDEPENDENT_DATA','unique_valid_source_images':len(unique),'selected_groups':len(selected),'exclusion_registry_images':len(registry),'provenance_class_counts_nonexclusive':class_counts,'historical_cf_holdout_groups':len(holdout),'historical_regular_holdout_rows':len(regular_holdout),'sources':source_receipts,'counts_by_candidate_source':counts,'source_archive':zip_inventory,'other_remote_archives':archives,'local_archives':local['archives'],'split_integrity':{k:v for k,v in split.items() if k!='overlaps'},'exact_historical_training_exposure':'UNKNOWN','new_model_forwards':0,'note':'Raw helmet-only source annotations without accepted miner-helmet pairing cannot establish CMF identity groups. Review/reject pairs are not promoted to accepted labels. Archive replays are prior studied assets, not new datasets.'})
print(json.dumps({'conclusion':'ENOUGH_FOR_STRONGER_FROZEN_EVAL' if enough else 'NEED_NEW_INDEPENDENT_DATA','unique_valid_candidates':len(unique),'excluded_images':len(registry),'counts':counts,'split_cross_role_images':split['cross_role_images']},indent=2))

import json,zipfile,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
root=Path(sys.argv[1]); work=root/'research_log/cycle039'
fields={'video_id','track_id','frame_id','frame_index','timestamp','sequence_id','worker_id','person_id','persistent_id'}
receipts=[];found=Counter(); pair_images=defaultdict(set)
for p in sorted((root/'shared/data/final_accepted_v1').glob('*.jsonl')):
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]; keys=Counter(k for row in rows for k in row)
 for row in rows:
  for k in fields:
   if k in row and row[k] is not None:found[k]+=1
  if 'pseudo_miner_id' in row:pair_images[row['pseudo_miner_id']].add(row.get('image_member',row.get('image_path')))
 receipts.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(rows),'field_counts':dict(keys)})
co=[]
with zipfile.ZipFile(root/'shared/source/mining_helmet.zip') as z:
 for n in z.namelist():
  if not n.endswith('.json'):continue
  data=json.loads(z.read(n))
  if not isinstance(data,dict) or 'images' not in data or 'annotations' not in data:continue
  for kind in ['images','annotations']:
   keys=Counter(k for v in data[kind] for k in v)
   for v in data[kind]:
    for k in fields:
     if k in v and v[k] is not None:found[k]+=1
   co.append({'member':n,'sha256':hashlib.sha256(z.read(n)).hexdigest(),'kind':kind,'rows':len(data[kind]),'fields':dict(keys)})
result={'status':'NO_PERSISTENT_IDENTITY_EVIDENCE' if not found else 'PARTIAL_SEQUENCE_METADATA','explicit_temporal_or_track_fields':dict(found),'pseudo_miner_ids':len(pair_images),'pseudo_miner_ids_on_multiple_image_members':sum(len(v)>1 for v in pair_images.values()),'manifests':receipts,'coco_metadata':co,'interpretation':'Image/annotation/pseudo detection IDs and numeric filenames are not evidence of cross-frame worker identity; no temporal links manufactured.'}
(work/'sequence_metadata_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['manifests','coco_metadata']},indent=2))

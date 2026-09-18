import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1])
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
rows=[]
inventory=json.loads((root/'research_log/cycle014/checkpoint_inventory.json').read_text())[0]
for r in inventory['files']:
 if r['name'].startswith('all_data_checkpoint/model-') and r['name'].endswith('.safetensors'):
  p=root/'shared/models/segllm_095e0637'/r['name'];digest=sha(p);assert digest==r['lfs']['sha256'];rows.append({'path':str(p),'sha256':digest,'matches_pinned_lfs':True})
sources=[]
for r in json.loads((root/'research_log/cycle015/source_receipts.json').read_text()):
 p=root/'external_baselines/segllm'/r['path'];digest=sha(p);normalized=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest();sources.append({'path':r['path'],'sha256':digest,'lf_sha256':normalized,'pinned_raw_sha256':r['sha256'],'raw_equal':digest==r['sha256'],'lf_equal':normalized==r['sha256']})
assert all(r['lf_equal'] for r in sources)
receipt={'recorded_unix':time.time(),'scope':'Post-run immutable weight/source audit; no inference rerun or model modification','checkpoint_revision':inventory['sha'],'checkpoint_shards':rows,'source_files':sources}
(root/'research_log/cycle025/segllm_checkpoint_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PINNED_CHECKPOINT_AND_SOURCE_HASHES_VERIFIED')

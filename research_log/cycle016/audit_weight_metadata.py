import hashlib,json
from pathlib import Path
from safetensors import safe_open
import torch,transformers
root=Path('/home/wenchang/asdasdsad/wjq/coalminellm_recovery_20260917')
source=root/'external_baselines/segllm'
ckpt=root/'shared/models/segllm_095e0637/all_data_checkpoint'
idx=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map']
keys=[k for k in idx if 'vl_layers.0.b_attn.' in k and any(x in k for x in ['weight_l','weight_v','gamma_l','gamma_v'])]
rows=[]
for k in keys:
 with safe_open(ckpt/idx[k],framework='pt',device='cpu') as f:
  sl=f.get_slice(k)
  rows.append({'key':k,'shard':idx[k],'shape':sl.get_shape(),'dtype':sl.get_dtype()})
path=source/'pretrained_weights/hipie/r50_parts.pth'
h=hashlib.sha256()
with path.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
fuse=source/'uninext-segm/projects/HIPIE/hipie/models/deformable_detr/fuse_helper.py'
r={'checkpoint_keys':rows,'hipie_expected_path':str(path),'hipie_resolved_path':str(path.resolve()),'hipie_sha256':h.hexdigest(),'hipie_bytes':path.stat().st_size,'bert_transport_path':str(source/'uninext-segm/projects/HIPIE/bert-base-uncased'),'bert_resolved_path':str((source/'uninext-segm/projects/HIPIE/bert-base-uncased').resolve()),'fuse_helper_sha256':hashlib.sha256(fuse.read_bytes()).hexdigest(),'torch':torch.__version__,'transformers':transformers.__version__,'forward_calls':0,'note':'Read-only metadata; no weight remapping or model loading.'}
out=root/'research_log/cycle016/weight_audit.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))

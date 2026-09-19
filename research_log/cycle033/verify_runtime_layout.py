"""Recovered-runtime config/tokenizer audit. No weights, images, or forward calls."""
import argparse, ast, hashlib, importlib.util, json, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root;out=root/'research_log/cycle033';out.mkdir(exist_ok=True)
base=root/'code/cmllm/third_party/LISA'
model=root/'shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf'
vision=root/'shared/models/clip-vit-large-patch14'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
provenance=read(root/'research_log/cycle025/cma_prediction_freeze.json')['checkpoint_provenance']
names=['config.json','added_tokens.json','special_tokens_map.json','tokenizer.model','tokenizer_config.json']
hashes={n:sha(model/n) for n in names}
assert all(hashes[n]==provenance[n] for n in names)
vc=read(vision/'config.json');vc=vc.get('vision_config',vc)
patches=(vc['image_size']//vc['patch_size'])**2
pre=read(vision/'preprocessor_config.json')
tokenizer=AutoTokenizer.from_pretrained(str(model),model_max_length=512,padding_side='right',use_fast=False,local_files_only=True)
tokenizer.pad_token=tokenizer.unk_token;tokenizer.add_tokens(['[SEG]','[REF]'])
convpath=base/'model/llava/conversation.py'
spec=importlib.util.spec_from_file_location('cma_audit_conversation',convpath)
conv=importlib.util.module_from_spec(spec);sys.modules[spec.name]=conv;spec.loader.exec_module(conv)
conv.default_conversation=conv.conv_templates['llava_v1']
def extract(path,name,env):
 tree=ast.parse(path.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),env)
 return env[name]
ev=root/'code/cmllm/scripts/eval_mr_ref_counterfactual_v0.py'
builder=extract(ev,'build_multiround_conversation',{'conversation_lib':conv,'DEFAULT_IMAGE_TOKEN':'<image>'})
tokenid=extract(ev,'get_added_token_id',{})
mm=base/'model/llava/mm_utils.py'
tokimage=extract(mm,'tokenizer_image_token',{'torch':torch,'IMAGE_TOKEN_INDEX':-200})
groups=[json.loads(s) for s in (root/'research_log/cycle025/selected_source_groups.jsonl').read_text().splitlines() if s.strip()]
records=[]
for g in groups:
 rows=[builder('Segment the miner whose helmet should be segmented next.',g['same_round2_query']) for _ in g['pair_ids']]
 # Exact collate_fn use_mm_start_end replacement before tokenizer_image_token.
 rows=[s.replace('<image>','<im_start><image><im_end>') for s in rows]
 ids=[tokimage(s,tokenizer) for s in rows]
 assert len(ids)==2 and ids[0]==ids[1]
 raw=ids[0];refid=[tokenid(tokenizer,'[REF]')];segid=[tokenid(tokenizer,'[SEG]')]
 def pos(t):return [i for i,x in enumerate(raw) if x==t]
 def expanded(t):return [i+sum(x==-200 for x in raw[:i])*(patches-1) for i in pos(t)]
 ref=expanded(refid[0]);seg=expanded(segid[0]);manualref=[i-1+255 for i in pos(refid[0])];manualseg=[i-1+255 for i in pos(segid[0])]
 assert patches==256 and len(pos(-200))==1 and pos(-200)[0]<pos(refid[0])[0]
 assert manualref==[i-1 for i in ref] and manualseg==[i-1 for i in seg]
 records.append(dict(group_id=g['counterfactual_id'],serialized_prompt=rows[0],raw_token_ids=raw,
                     raw_length=len(raw),image_positions=pos(-200),raw_REF=pos(refid[0]),raw_SEG=pos(segid[0]),
                     expanded_REF=ref,manual_REF=manualref,auxiliary_REF=ref,expanded_SEG=seg,manual_SEG=manualseg,
                     expanded_length=len(raw)+patches-1,AB_token_identical=True))
receipt=dict(status='PASS',runtime='recovered A6000 environment used by frozen loader',image_size=vc['image_size'],patch_size=vc['patch_size'],num_patches=patches,
             vision_config_sha256=sha(vision/'config.json'),preprocessor_config=pre,preprocessor_sha256=sha(vision/'preprocessor_config.json'),
             checkpoint_metadata_matches_Cycle025=True,checkpoint_metadata_sha256=hashes,REF_id=refid[0],SEG_id=segid[0],
             template='llava_v1',use_fast=False,padding_side='right',model_max_length=512,
             standalone_REF_encoding=tokenizer('[REF]',add_special_tokens=False).input_ids,
             standalone_SEG_encoding=tokenizer('[SEG]',add_special_tokens=False).input_ids,
             source_sha256={str(f.relative_to(root)):sha(f) for f in [convpath,mm,ev,base/'utils/dataset.py',base/'model/llava/model/multimodal_encoder/clip_encoder.py']},
             prefix_status='PREFIX_IDENTICAL_BEFORE_REF',groups_checked=len(records),rows=records,
             weights_loaded=False,images_loaded=False,model_calls=0,scorer_calls=0)
(out/'runtime_layout_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['rows','source_sha256','preprocessor_config','checkpoint_metadata_sha256']},indent=2))
print(json.dumps({k:v for k,v in records[0].items() if k not in ['raw_token_ids','serialized_prompt']},indent=2))

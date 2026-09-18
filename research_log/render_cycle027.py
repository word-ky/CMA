"""Condense two prescribed frozen cases; render only, no model/scorer imports."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);work=root/'research_log/cycle027';work.mkdir(exist_ok=True)
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
case_file=root/'research_log/cycle026/QUALITATIVE_CASES.json';cases={c['group_id']:c for c in load(case_file)['cases']}
success='cf_778571dc8020b5df';failure='acceptedpair_b914b3d4a14a9083'
layout=[(success,'target15_b',0),(success,'target15_b',1),(failure,'clean',0),(failure,'target15_b',0)]
pairs={p['pair_id']:p for p in map(json.loads,(root/'shared/data/cycle025/helmet_miner_pairs_restored.jsonl').read_text().splitlines())}
asset={r['path']:r['sha256'] for r in load(root/'research_log/cycle025/frozen_assets.json')['assets']}
pred={};reports={};inputs={}
for method in ['cma','segllm']:
 for r in load(root/f'outputs/cycle025/{method}/prediction_freeze.json')['trials']:pred[(method,r['group_id'],r['condition'],r['entity_id'])]=r
 for c in ['clean','target15_b']:reports[(method,c)]={r['group_id']:r for r in load(root/f'research_log/cycle025/scoring/{method}_{c}_metrics.json')['groups']}
def check(p,h):assert sha(p)==h;inputs[str(p)]=h
def mask(p,h):check(p,h);return (np.load(p) if str(p).endswith('.npy') else np.asarray(Image.open(p)))!=0
def overlay(im,m,col):
 a=im.astype(float).copy();a[m]=.55*a[m]+.45*np.array(col);return a.astype(np.uint8)
fig=plt.figure(figsize=(7.2,6.6),dpi=200)
width=.282;left=.115;gap=.012;tops=[.885,.680,.415,.210];height=.1515
fig.text(.115,.978,'(a) Identity switch: same degraded observation, memory A vs B',fontsize=9,fontweight='bold')
fig.text(.115,.948,success,fontsize=7)
fig.text(.115,.500,'(b) Degradation limitation: fixed memory A, clean vs degraded',fontsize=9,fontweight='bold')
fig.text(.115,.470,failure,fontsize=7)
for row,(gid,condition,identity) in enumerate(layout):
 case=cases[gid];pid=case['entity_ids'][identity];label='A' if identity==0 else 'B';color=[40,170,255] if identity==0 else [255,140,40]
 prep=load(root/f'research_log/cycle025/prepared/{gid}/preparation_receipt.json');obs=next(o for o in prep['observations'] if o['condition']==condition)
 check(obs['image_path'],obs['file_sha256']);rgb=np.asarray(Image.open(obs['image_path']).convert('RGB'))
 memory=pairs[pid]['miner_mask_path'];target=pairs[pid]['helmet_mask_path'];tm=mask(target,asset[target])
 views=[overlay(rgb,mask(memory,asset[memory]),color)];titles=[f'Supplied memory {label}']
 for method in ['cma','segllm']:
  p=pred[(method,gid,condition,pid)];views.append(overlay(rgb,mask(p['prediction'],p['prediction_sha256']),color))
  s=reports[(method,condition)][gid];assert s['entity_ids'][identity]==pid
  titles.append(f"{'CMA' if method=='cma' else 'SegLLM'}: IoU {s['correct_iou'][identity]:.3f}")
 for col,im in enumerate(views):
  ax=fig.add_axes([left+col*(width+gap),tops[row]-height,width,height]);ax.imshow(im,interpolation='nearest')
  if col:ax.contour(tm,levels=[.5],colors='white',linewidths=.45)
  ax.set_title(titles[col],fontsize=8,pad=3);ax.axis('off')
  if col==0:ax.text(-.035,.5,('degraded' if condition=='target15_b' else 'clean')+'\nMemory '+label,transform=ax.transAxes,fontsize=7,ha='right',va='center')
fig.text(.115,.035,'White outline: saved pseudo target. Blue: identity A; orange: identity B.',fontsize=7)
fig.text(.115,.015,'Full frames; display-only mask overlays. IoU copied from frozen records.',fontsize=7)
for ext in ['png','pdf']:fig.savefig(work/f'MAIN_QUAL_FIGURE.{ext}',dpi=300)
plt.close(fig)
assert all(sha(p)==h for p,h in inputs.items())
receipt={'case_manifest_sha256':sha(case_file),'layout':[{'group_id':g,'condition':c,'identity_index':i} for g,c,i in layout],'source_qc':{g:cases[g]['qc'] for g in [success,failure]},'inputs_sha256':inputs,'all_inputs_unchanged_after_render':True,'outputs_sha256':{ext:sha(work/f'MAIN_QUAL_FIGURE.{ext}') for ext in ['png','pdf']},'model_calls':0,'scorer_calls':0,'new_case_search':False,'supplementary_bank_modified':False,'display':'Full-frame original observations. 45% identity-colour prediction/memory overlays and white saved-target contours; no brightness/mask changes. Second optional success and identity-error cases omitted for readability.'}
(work/'main_figure_render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('COMPACT_MAIN_FIGURE_RENDERED_NO_MODEL_OR_SCORER')

"""Render frozen qualitative cases, without models, scorer or mask modification."""
import hashlib,json,sys,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);work=root/'research_log/cycle026';out=work/'figures';out.mkdir(exist_ok=True)
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cases=load(work/'QUALITATIVE_CASES.json');assets={a['path']:a['sha256'] for a in load(root/'research_log/cycle025/frozen_assets.json')['assets']}
pred={};metrics={}
for method in ['cma','segllm']:
 for t in load(root/f'outputs/cycle025/{method}/prediction_freeze.json')['trials']:pred[(method,t['group_id'],t['condition'],t['entity_id'])]=t
 for condition in ['clean','target15_b']:
  metrics[(method,condition)]={g['group_id']:g for g in load(root/f'research_log/cycle025/scoring/{method}_{condition}_metrics.json')['groups']}
pairs={r['pair_id']:r for r in map(json.loads,(root/'shared/data/cycle025/helmet_miner_pairs_restored.jsonl').read_text().splitlines())}
used={};outputs=[]
def mask(path,digest):
 assert sha(path)==digest;used[path]=digest
 return (np.load(path) if str(path).endswith('.npy') else np.asarray(Image.open(path)))!=0
def overlay(rgb,m,color):
 display=rgb.astype(float).copy();display[m]=.55*display[m]+.45*np.array(color);return display.astype(np.uint8)
for index,case in enumerate(cases['cases'],1):
 gid=case['group_id'];prep=load(root/f'research_log/cycle025/prepared/{gid}/preparation_receipt.json')
 fig,axes=plt.subplots(4,5,figsize=(18,10),dpi=160)
 fig.subplots_adjust(left=.065,right=.995,bottom=.10,top=.84,wspace=.035,hspace=.25)
 title=case['category'].replace('_',' ')
 original=case['qc']['original_group_record'];qcflag=str(original['qc']['all_pairs_have_clean_episode']) if original else 'not a historical group'
 fig.suptitle(f'{index}. {title}\n{gid}',fontsize=16,y=.985,fontweight='bold')
 fig.text(.5,.90,f"Source: {case['source']} | historical all-pairs-clean-episode: {qcflag}",ha='center',fontsize=10)
 for ci,obs in enumerate(prep['observations']):
  condition=obs['condition'];assert sha(obs['image_path'])==obs['file_sha256'];used[obs['image_path']]=obs['file_sha256']
  rgb=np.asarray(Image.open(obs['image_path']).convert('RGB'))
  for identity,pid in enumerate(case['entity_ids']):
   row=2*ci+identity;color=[40,170,255] if identity==0 else [255,140,40];label='A' if identity==0 else 'B'
   memory=pairs[pid]['miner_mask_path'];target=pairs[pid]['helmet_mask_path']
   images=[rgb,overlay(rgb,mask(memory,assets[memory]),color),overlay(rgb,mask(target,assets[target]),color)]
   titles=['Fixed observation',f'Supplied memory {label}',f'Pseudo-target {label}']
   for method in ['cma','segllm']:
    t=pred[(method,gid,condition,pid)];images.append(overlay(rgb,mask(t['prediction'],t['prediction_sha256']),color))
    s=metrics[(method,condition)][gid];assert s['entity_ids'][identity]==pid
    titles.append(f"{'CMA' if method=='cma' else 'SegLLM'} | saved IoU {s['correct_iou'][identity]:.3f}")
   for col,(im,t) in enumerate(zip(images,titles)):
    axes[row,col].imshow(im,interpolation='nearest');axes[row,col].set_title(t,fontsize=10,pad=4);axes[row,col].axis('off')
   axes[row,0].text(-.04,.5,f'{condition}\nMemory {label}',transform=axes[row,0].transAxes,ha='right',va='center',fontsize=10,fontweight='bold')
 fig.text(.065,.058,'Query (fixed): segment only the mining helmet worn by the supplied miner identity.',fontsize=10)
 fig.text(.065,.029,'A: blue; B: orange. Saved binary masks shown with display-only alpha overlays. Full frames; no mask edits, enhancement, inference or rescoring.',fontsize=9)
 stem=f'{index:02d}_{gid}';png=out/(stem+'.png');pdf=out/(stem+'.pdf')
 fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
 outputs.append({'group_id':gid,'category':case['category'],'source':case['source'],'qc':case['qc'],'png':str(png.relative_to(root)),'png_sha256':sha(png),'pdf':str(pdf.relative_to(root)),'pdf_sha256':sha(pdf)})
 print('RENDERED',gid,flush=True)
# Small contact sheet for visual QA only, not a substitute for full-resolution panels.
contact=Image.new('RGB',(1440,1320),'white')
for i,r in enumerate(outputs):
 im=Image.open(root/r['png']);im.thumbnail((720,440));contact.paste(im,((i%2)*720,(i//2)*440))
contact.save(out/'contact_sheet.png')
(work/'render_receipt.json').write_text(json.dumps({'case_manifest_sha256':sha(work/'QUALITATIVE_CASES.json'),'figures':outputs,'input_hashes_verified':used,'model_calls':0,'scorer_calls':0,'mask_files_modified':False,'display':'Full-frame saved observations; 45% identity-colour alpha mask overlays, no brightening or geometry changes. IoU annotations copied from frozen reports.'},indent=2)+'\n')

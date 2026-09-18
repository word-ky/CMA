"""Vector schematic from frozen method descriptions; no model/scorer/data imports."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

out = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(7.2, 5.4))
fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.015)
ax.set(xlim=(0,100), ylim=(0,75)); ax.axis('off')
ink='#27313b'; blue='#2277b5'; orange='#b96712'; violet='#765598'; green='#37765b'
def box(x,y,w,h,label,edge=ink,fill='#f5f6f7',size=9):
    ax.add_patch(Rectangle((x,y),w,h,lw=1,ec=edge,fc=fill))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=ink,linespacing=1.35)
def arrow(points,color=ink):
    for a,b in zip(points[:-2],points[1:-1]): ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=9,lw=1,color=color))
ax.text(50,73,'Supplied identity-localized entity memory',ha='center',va='center',weight='bold',fontsize=11)
ax.text(50,69.7,'Same-condition appearance; externally supplied, fixed miner geometry',ha='center',va='center',fontsize=9)
box(1,53,25,12,'Same scene I\nSame relational query q')
box(1,39,25,11,'Memory A\nminer mask + bbox\n+ appearance crop',blue,'#eef6fc')
box(1,25,25,11,'Memory B\nminer mask + bbox\n+ appearance crop',orange,'#fcf3e7')
box(34,48,35,13,'Input-side conditioning\nCrop + bbox → [REF]\nLanguage model',violet,'#f4eff9')
box(34,27,35,14,'Output-side conditioning\nREF hidden + mask/bbox\nAdd context to SEG prompt',violet,'#f4eff9')
box(78,48,21,13,'SAM\nimage features\n+ text prompt',green,'#eff7f1')
box(78,27,21,14,'Helmet A / B\nDesired outputs\n(schematic)',green,'#eff7f1')
arrow([(26,59),(30,59),(30,66),(52,66),(52,61)])
arrow([(30,66),(88.5,66),(88.5,61)])
ax.text(75,67,'image I',ha='center',va='bottom',fontsize=8.5)
arrow([(26,44.5),(29,44.5),(29,54.5),(34,54.5)],blue)
arrow([(26,30.5),(29,30.5),(29,54.5),(34,54.5)],orange)
arrow([(29,34),(34,34)])
ax.text(14,22.4,'Switch A ↔ B; hold I, q fixed',ha='center',fontsize=8.5)
arrow([(51.5,48),(51.5,41)],violet)
ax.text(54,44.4,'REF + SEG',ha='left',va='center',fontsize=8.5,color=violet)
arrow([(69,34),(73.5,34),(73.5,54.5),(78,54.5)],violet)
arrow([(88.5,48),(88.5,41)],green)
ax.text(84,44.3,'decode',ha='center',fontsize=8.5,color=green)
ax.text(66,22.4,'Shared CMA weights; one identity condition per trial',ha='center',fontsize=8.5)
ax.plot([1,99],[20.5,20.5],color='#bcc2c8',lw=.8)
box(1,1,48,17.5,'',fill='#fafafa')
box(51,1,48,17.5,'',fill='#fafafa')
ax.text(3,16,'Training only: paired identity rank',weight='bold',fontsize=9)
ax.text(3,11.8,r'$S_{ij} = \mathrm{softIoU}(p_i,Y_j)$',fontsize=9)
ax.text(3,7.5,r'$[\,\mu + \max_{j\ne i}S_{ij} - S_{ii}\,]_+$',fontsize=10)
ax.text(3,3.3,'Alongside text, mask and REF reconstruction losses',fontsize=8)
ax.text(53,16,'Offline evaluation: saved binary masks',weight='bold',fontsize=9)
for row in range(2):
    for col in range(2):
        x=54+col*6.3; y=4+(1-row)*4
        ax.add_patch(Rectangle((x,y),6.3,4,ec='#9da7ae',fc='#e5eef7' if row==col else 'white',lw=.6))
        ax.text(x+3.15,y+2,'J'+('AA' if row==col==0 else 'BB' if row==col==1 else 'AB' if row==0 else 'BA'),ha='center',va='center',fontsize=8)
ax.text(70,11,'Fidelity: correct > wrong',fontsize=8.5)
ax.text(70,7.3,'CMSA: both correct + IoU ≥ 0.5',fontsize=8.5)
ax.text(70,3.6,'IER: stronger wrong IoU ≥ 0.5',fontsize=8.5)
for ext in ['svg','pdf','png']: fig.savefig(out/f'METHOD_OVERVIEW.{ext}',dpi=210,facecolor='white')
svg = out/'METHOD_OVERVIEW.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(out/'method_figure_receipt.json').write_text(json.dumps({'source_script_sha256':sha(Path(__file__)),'outputs_sha256':{ext:sha(out/f'METHOD_OVERVIEW.{ext}') for ext in ['svg','pdf','png']},'figure_width_inches':7.2,'model_calls':0,'scorer_calls':0,'new_case_selection':False,'schematic_only':True,'numeric_performance_values':[]},indent=2)+'\n')

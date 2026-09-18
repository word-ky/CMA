"""Verify frozen numeric table and select qualitative cases from saved booleans only."""
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'research_log/cycle026'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(out/n).write_bytes((json.dumps(x,indent=2)+'\n').encode())
table=list(csv.reader((out/'PAPER_LAYER1_TABLE.csv').open(encoding='utf-8',newline='')));sources=load(out/'table_sources.json')
assert len(table)==15
for row,ref in zip(table[1:13],sources['references']):
 p=root/ref['path'];assert sha(p)==ref['sha256'];s=load(p)['summaries'][ref['pointer'].split('/')[-1]]
 assert [float(x) for x in row[4:]]==[s[k] for k in sources['keys']]
primary=load(root/sources['delta_source'])
for row in table[13:]:
 d=primary['method_deltas'][row[3]]['aggregate']
 for i,k in [(4,'target_miou'),(7,'cmsa'),(10,'memory_fidelity'),(13,'identity_error_rate'),(14,'mean_identity_margin'),(15,'median_identity_margin')]:assert float(row[i])==d[k]
 s=primary['summaries']['cma_'+row[3]]
 assert row[16:]==[str(s['num_groups']),str(s['num_references'])]
# Verify displayed Markdown cells as well as unrounded CSV.
md=(out/'PAPER_LAYER1_TABLE.md').read_text(encoding='utf-8')
lines=[l for l in md.splitlines() if l.startswith('|CMA base-w15|') or l.startswith('|SegLLM pinned|')]
assert len(lines)==12
for line,row in zip(lines,table[1:13]):
 cells=line.strip('|').split('|');nums=list(map(float,row[4:]));assert cells[2]==f'{nums[0]*100:.2f}%'
 for col,n,d,r in [(3,1,2,3),(4,4,5,6),(5,7,8,9)]:assert cells[col]==f'{int(nums[n])}/{int(nums[d])} ({nums[r]*100:.0f}%)'
 assert cells[6:]==[f'{nums[10]:.6f}',f'{nums[11]:.6f}',f'{int(nums[12])} / {int(nums[13])}']
for c in ['clean','target15_b']:
 d=primary['method_deltas'][c]['aggregate'];expected='|'+c+'|'+'|'.join(f'{100*d[k]:.2f}' for k in ['target_miou','cmsa','memory_fidelity','identity_error_rate'])+'|'+f"{d['mean_identity_margin']:.6f}|{d['median_identity_margin']:.6f}|";assert expected in md
dump('table_verification.json',{'status':'PASS','data_rows':12,'paired_delta_rows':2,'unrounded_csv_numeric_cells_checked':184,'markdown_numeric_displays_checked':True,'sources':sources,'csv_sha256':sha(out/'PAPER_LAYER1_TABLE.csv'),'markdown_sha256':sha(out/'PAPER_LAYER1_TABLE.md'),'scorer_called':False})
manifest=root/'research_log/cycle024/documented_protocol_disjoint_candidates.jsonl'
ordered=[json.loads(l) for l in manifest.read_text().splitlines()]
reports={};report_sources={}
for key in ['cma_clean','cma_target15_b','segllm_target15_b']:
 p=root/f'research_log/cycle025/scoring/{key}_metrics.json';reports[key]={g['group_id']:g for g in load(p)['groups']};report_sources[str(p.relative_to(root))]=sha(p)
pass_c=lambda gid:reports['cma_target15_b'][gid]['pairs'][0]['success']
pass_s=lambda gid:reports['segllm_target15_b'][gid]['pairs'][0]['success']
rules=[('cma_degraded_pass_segllm_fail',2,lambda g:pass_c(g) and not pass_s(g)),('both_degraded_fail',1,lambda g:not pass_c(g) and not pass_s(g)),('cma_clean_pass_degraded_fail',1,lambda g:reports['cma_clean'][g]['pairs'][0]['success'] and not pass_c(g)),('segllm_identity_error_cma_not',1,lambda g:any(s and not c for s,c in zip(reports['segllm_target15_b'][g]['identity_error'],reports['cma_target15_b'][g]['identity_error'])))]
seen=set();categories=[];selected=[]
qc={r['counterfactual_id']:r for r in load(root/'research_log/cycle024/selected_group_qc_provenance.json')}
for label,n,rule in rules:
 matches=[]
 for i,g in enumerate(ordered):
  gid=g['counterfactual_id']
  if gid not in seen and rule(gid):
   seen.add(gid);matches.append(gid);selected.append({'group_id':gid,'manifest_index':i,'category':label,'source':g['source'],'qc':qc[gid],'image_sha256':g['image_sha256'],'entity_ids':g['pair_ids'],'frozen_outcomes':{key:reports[key][gid] for key in reports}})
   if len(matches)==n:break
 categories.append({'category':label,'requested':n,'selected':matches or 'none'})
dump('QUALITATIVE_CASES.json',{'selection_rule':'Category order as REVIEW025; within each category first eligible previously unselected group(s) in frozen manifest order. No appearance/magnitude ranking; missing categories recorded none. Quantitative membership unchanged.','manifest_sha256':sha(manifest),'frozen_reports':report_sources,'categories':categories,'cases':selected,'scorer_called':False,'model_called':False})
print(json.dumps({'table':'PASS','selected':[(r['category'],r['group_id']) for r in selected]},indent=2))

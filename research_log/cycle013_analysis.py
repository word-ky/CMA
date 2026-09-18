"""Fixed C/D validation and diagnostic-only whole-group oracle union."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
root=Path(sys.argv[1]);out=root/'outputs/cycle013';log=root/'research_log/cycle013';log.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'code/cmllm/scripts'))
from eval_counterfactual_memory_fidelity import summarize
from cycle009_analysis import check_pairs,transitions,FIELDS
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
table={};delta={};reports={}
for cond in ('clean','target15_b'):
    base=root/f'outputs/cycle008/val_base_{cond}';adapt=out/f'val_adapter_{cond}'
    check_pairs(base,adapt)
    a,b=read(base/'memory_metrics.json'),read(adapt/'memory_metrics.json')
    table[cond]={'base':a['summary'],'adapter':b['summary']};reports[cond]=(a,b)
    delta[cond]={k:b['summary'][k]-a['summary'][k] for k in FIELDS}
a,b=reports['target15_b']
changes=transitions([g['cmsa']==1 for g in a['groups']],[g['cmsa']==1 for g in b['groups']])
chosen=[];miou_chosen=[];choices=[]
for x,y in zip(a['groups'],b['groups']):
    assert x['group_id']==y['group_id']
    use=(y['cmsa'],y['target_miou'])>(x['cmsa'],x['target_miou'])
    chosen.append(y if use else x);miou_chosen.append(y if y['target_miou']>x['target_miou'] else x)
    choices.append({'group_id':x['group_id'],'oracle_choice':'adapter' if use else 'base',
                    'cmsa_base':x['cmsa'],'cmsa_adapter':y['cmsa'],
                    'miou_base':x['target_miou'],'miou_adapter':y['target_miou']})
criteria={'degraded_miou_gain_0_02':delta['target15_b']['target_miou']>=.02,
          'degraded_cmsa_non_decrease':delta['target15_b']['cmsa']>=0,
          'at_least_3_recoveries':changes['fail_to_pass']>=3,'at_most_1_regression':changes['pass_to_fail']<=1,
          'clean_miou_drop_at_most_0_01':delta['clean']['target_miou']>=-.01,
          'clean_cmsa_drop_at_most_one_group':round(table['clean']['adapter']['cmsa']*50)>=round(table['clean']['base']['cmsa']*50)-1}
train=rows(out/'train/train_log.jsonl');old=rows(root/'outputs/cycle008/train/train_log.jsonl')
assert len(train)==300 and [x['group_id'] for x in train]==[x['group_id'] for x in old]
summary=read(out/'train/train_summary.json');assert summary['frozen_base_gradient_free']
result={'status':'PASS' if all(criteria.values()) else 'FAIL','criteria':criteria,'validation':table,'adapter_minus_base':delta,
        'cmsa_transitions':changes,'oracle_union_ANALYSIS_ONLY':{'rule':'whole-group GT CMSA then GT mIoU, ties base; non-deployable','summary':summarize(chosen),'choices':choices},
        'oracle_miou_ANALYSIS_ONLY':{'rule':'whole-group GT mIoU maximum; non-deployable','summary':summarize(miou_chosen)},
        'trainable_parameters':12577,'extra_frozen_encoder_calls_per_identity':1,'extra_frozen_encoder_calls_per_group':2,
        'exact_cycle008_training_order':True,'base_frozen_gradient_free':True,'training_runs':1,'optimizer_steps':300,
        'checkpoint_sha256':hashlib.sha256((out/'train/last.pt').read_bytes()).hexdigest(),
        'diagnostic_and_confirmation_calls':0}
(log/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ('smoke','train','val_adapter_clean','val_adapter_target15_b'):
    dest=log/name;dest.mkdir(exist_ok=True)
    for p in (out/name).glob('*.json*'):shutil.copyfile(p,dest/p.name)
files=[root/'code/cmllm/scripts'/name for name in ('memory_dual_scale_adapter.py','dual_scale_features.py','train_memory_dual_scale.py','eval_mr_ref_counterfactual_v0.py')]
files.append(root/'code/cmllm/third_party/LISA/model/LISA.py')
(log/'source_hashes.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if not k.startswith('oracle_')},indent=2))

"""Fixed MSP comparison; GT-oracle choices here are analysis only."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

root=Path(sys.argv[1])
sys.path.insert(0,str(root/'code/cmllm/scripts'))
from eval_counterfactual_memory_fidelity import summarize
from cycle009_analysis import check_pairs, transitions, FIELDS

base=root/'outputs/cycle008/val_base_target15_b'
out=root/'outputs/cycle011'
log=root/'research_log/cycle011';log.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(line) for line in p.read_text().splitlines()]

if sys.argv[2]=='regression':
    folder=out/'full_regression'
    a=rows(base/'memory_predictions.jsonl')[0];b=rows(folder/'memory_predictions.jsonl')[0]
    assert a['group_id']==b['group_id']
    for x,y in zip(a['trials'],b['trials']):
        for field in ('prediction','target','reference'):
            np.testing.assert_array_equal(np.load(base/x[field]),np.load(folder/y[field]))
    (log/'regression.json').write_text(json.dumps({'full_frame_first_group':a['group_id'],'all_prediction_target_reference_masks_byte_equal':True},indent=2)+'\n')
    print('Full-frame regression matched saved base masks exactly.')
else:
    msp=out/'val_msp_target15_b'
    check_pairs(base,msp)
    a,b=read(base/'memory_metrics.json'),read(msp/'memory_metrics.json')
    assert len(a['groups'])==len(b['groups'])==50
    changes=transitions([g['cmsa']==1 for g in a['groups']],[g['cmsa']==1 for g in b['groups']])
    oracle=[]; choices=[]; miou_oracle=[]
    for x,y in zip(a['groups'],b['groups']):
        assert x['group_id']==y['group_id']
        use_msp=(y['cmsa'],y['target_miou'])>(x['cmsa'],x['target_miou'])
        oracle.append(y if use_msp else x)
        miou_oracle.append(y if y['target_miou']>x['target_miou'] else x)
        choices.append({'group_id':x['group_id'],'choice':'MSP' if use_msp else 'full','cmsa_full':x['cmsa'],'cmsa_msp':y['cmsa'],'miou_full':x['target_miou'],'miou_msp':y['target_miou']})
    delta={m:b['summary'][m]-a['summary'][m] for m in FIELDS}
    criteria={'miou_gain_at_least_0_02':delta['target_miou']>=.02,'cmsa_non_decrease':delta['cmsa']>=0,
              'at_least_three_fail_to_pass':changes['fail_to_pass']>=3,'at_most_one_pass_to_fail':changes['pass_to_fail']<=1}
    result={'status':'PASS' if all(criteria.values()) else 'FAIL','criteria':criteria,
            'full':a['summary'],'msp':b['summary'],'msp_minus_full':delta,'cmsa_transitions':changes,
            'oracle_union_ANALYSIS_ONLY':{'rule':'per-group GT CMSA first, then GT group mIoU, tie full; not deployable',
                                        'summary':summarize(oracle),'choices':choices},
            'oracle_miou_ANALYSIS_ONLY':{'rule':'per-group GT mIoU maximum; not deployable','summary':summarize(miou_oracle)},
            'paired_original_inputs_and_masks_equal':True,'new_training_runs':0,'msp_validation_runs':1,'box_expansion':1.0}
    union_gain=result['oracle_union_ANALYSIS_ONLY']['summary']['target_miou']-a['summary']['target_miou']
    opportunities=changes['fail_to_pass']
    result['action_complementarity']={'union_miou_gain':union_gain,'cmsa_recovery_opportunities':opportunities,'passes':union_gain>=.03 and opportunities>=5}
    (log/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for folder in (out/'full_regression',msp):
        dest=log/folder.name;dest.mkdir(exist_ok=True)
        for path in folder.glob('*.json*'):shutil.copyfile(path,dest/path.name)
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (root/'code/cmllm/scripts/memory_spatial_prompt.py',root/'code/cmllm/third_party/LISA/model/LISA.py',root/'code/cmllm/scripts/eval_mr_ref_counterfactual_v0.py',root/'research_log/run_cycle011.sh')}
    (log/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('oracle_')},indent=2))

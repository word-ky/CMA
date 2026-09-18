"""One post-freeze paired confirmation using the unchanged CMF scorer."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'code/cmllm/scripts'))
from eval_counterfactual_memory_fidelity import evaluate_file
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
work = root / 'research_log/cycle025'
out = work / 'scoring'
out.mkdir(exist_ok=True)
protocol = load(work / 'protocol_receipt.json')
assert sha(root / 'code/cmllm/scripts/eval_counterfactual_memory_fidelity.py') == protocol['frozen_code_hashes']['scripts/eval_counterfactual_memory_fidelity.py']
specs = load(work / 'inference_specs.json')
execution = load(work / 'execution_manifest.json')
assert sha(work / 'execution_manifest.json') == protocol['execution_manifest_sha256']
allowed = {str(Path(p).resolve()) for s in specs for p in s['input_asset_hashes']}
allowed.update(obs['image_path'] for g in execution for obs in g['preparation']['observations'])
by_method, freeze_receipts = {}, {}
for method in ['cma', 'segllm']:
    folder = root / 'outputs/cycle025' / method
    frozen = load(folder / 'prediction_freeze.json')
    assert len(frozen['trials']) == 4 * protocol['groups']
    assert frozen['started_unix'] > protocol['frozen_unix']
    grouped = {}
    for trial in frozen['trials']:
        assert sha(trial['prediction']) == trial['prediction_sha256']
        grouped.setdefault((trial['group_id'], trial['condition']), []).append(trial)
    reads = set(frozen.get('raster_reads', []))
    if method == 'segllm':
        for spec in specs:
            reads.update(p for p in load(folder / spec['group_id'] / 'smoke_receipt.json')['opened_data_paths']
                         if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg', '.npy'})
    assert reads <= allowed, reads - allowed
    for spec in specs:
        for condition in ['clean', 'target15_b']:
            assert [r['entity_id'] for r in grouped[(spec['group_id'], condition)]] == [i['entity_id'] for i in spec['identities']]
    by_method[method] = grouped
    freeze_receipts[method] = {'sha256': sha(folder / 'prediction_freeze.json'),
                               'frozen_unix': frozen['frozen_unix'], 'raster_reads': sorted(reads)}
audit = {'methods': freeze_receipts, 'all_prediction_hashes_verified': True,
         'source_identity_order_verified': True, 'target_free_runtime_reads_verified': True,
         'scoring_started_unix': time.time()}
assert all(audit['scoring_started_unix'] > f['frozen_unix'] for f in freeze_receipts.values())
(out / 'prediction_freeze_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
# First target-bearing manifests/masks in this scoring process, after both complete freezes.
assets = load(work / 'frozen_assets.json')
for asset in assets['assets']:
    assert sha(asset['path']) == asset['sha256']
pairs = {r['pair_id']: r for r in map(json.loads, (root / 'shared/data/cycle025/helmet_miner_pairs_restored.jsonl').read_text().splitlines())}
summaries, reports = {}, {}
for method, grouped in by_method.items():
    for condition in ['clean', 'target15_b']:
        records = []
        for spec in specs:
            trials = grouped[(spec['group_id'], condition)]
            records.append({'group_id': spec['group_id'], 'image_id': spec['source_image'],
                            'query': spec['semantic_query'], 'condition': condition, 'memory_source': 'supplied_ref',
                            'trials': [dict(entity_id=t['entity_id'], prediction=t['prediction'],
                                            target=pairs[t['entity_id']]['helmet_mask_path']) for t in trials]})
        stem = method + '_' + condition
        path = out / (stem + '.jsonl')
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        report = evaluate_file(path, out / (stem + '_metrics.json'))
        summary = dict(report['summary'],
                       empty_masks=sum(not np.load(t['prediction']).any() for r in records for t in r['trials']),
                       cmsa_count=sum(p['success'] for r in report['groups'] for p in r['pairs']),
                       fidelity_count=sum(sum(r['fidelity']) for r in report['groups']),
                       ier_count=sum(sum(r['identity_error']) for r in report['groups']))
        summaries[stem], reports[stem] = summary, report

def deltas(first, second):
    a, b = summaries[first], summaries[second]
    per_group = []
    for x, y in zip(reports[first]['groups'], reports[second]['groups']):
        assert x['group_id'] == y['group_id'] and x['entity_ids'] == y['entity_ids']
        per_group.append({'group_id': x['group_id'],
                          'correct_iou_delta': (np.array(x['correct_iou']) - y['correct_iou']).tolist(),
                          'identity_margin_delta': (np.array(x['identity_margin']) - y['identity_margin']).tolist(),
                          'cmsa_delta': int(x['pairs'][0]['success']) - int(y['pairs'][0]['success'])})
    return {'direction': first + ' minus ' + second,
            'aggregate': {k: a[k] - b[k] for k in a if k not in ['num_groups', 'num_references', 'num_pairs']},
            'paired_groups': per_group}

result = {'summaries': summaries,
          'degradation_deltas': {m: deltas(m+'_target15_b', m+'_clean') for m in by_method},
          'method_deltas': {c: deltas('cma_'+c, 'segllm_'+c) for c in ['clean', 'target15_b']},
          'tuning_performed': False}
(out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(summaries, indent=2))

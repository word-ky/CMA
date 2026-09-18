"""Score only the completed frozen run; reuse unchanged CMF and frozen CMA masks."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from eval_counterfactual_memory_fidelity import evaluate_file


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = root / 'research_log/cycle018/scoring'
    out.mkdir(parents=True, exist_ok=True)
    run = root / 'outputs/cycle018/segllm'
    freeze_path = run / 'prediction_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    protocol = json.loads((root / 'research_log/cycle018/protocol_receipt.json').read_text())
    assert freeze['forward_count'] == len(freeze['trials']) == 200
    assert sha(root / 'code/cmllm/scripts/eval_counterfactual_memory_fidelity.py') == protocol['frozen_files']['cmllm_remote/scripts/eval_counterfactual_memory_fidelity.py']
    grouped_predictions = {}
    for trial in freeze['trials']:
        assert sha(trial['prediction']) == trial['prediction_sha256']
        grouped_predictions.setdefault((trial['group_id'], trial['condition']), []).append(trial)
    opened = set()
    for group in protocol['ordered_groups']:
        receipt = json.loads((run / group['group_id'] / 'smoke_receipt.json').read_text())
        opened.update(receipt['opened_data_paths'])
        for condition in ['clean', 'target15_b']:
            assert [t['entity_id'] for t in grouped_predictions[(group['group_id'], condition)]] == group['pair_ids']
    raster_reads = sorted(x for x in opened if Path(x).suffix.lower() in {'.png', '.jpg', '.jpeg', '.npy'})
    prepared = json.loads((root / 'research_log/cycle018/prepared/execution_manifest.json').read_text())
    allowed = {obs['image_path'] for g in prepared for obs in g['preparation']['observations']}
    allowed |= {i['miner_mask_path'] for g in prepared for obs in g['preparation']['observations'] for i in obs['identities']}
    assert set(raster_reads) <= allowed
    audit = {'prediction_freeze_sha256': sha(freeze_path), 'frozen_unix': freeze['frozen_unix'],
             'scoring_started_unix': time.time(), 'all_200_hashes_rechecked_before_targets': True,
             'raster_reads': raster_reads, 'only_prepared_images_and_miner_masks': True}
    assert audit['scoring_started_unix'] > audit['frozen_unix']
    (out / 'prediction_freeze_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    # Target access begins only here, after all predictions are immutable and audited.
    groups = [json.loads(l) for l in (root / 'shared/data/cycle006/val_grouped_restored.jsonl').read_text().splitlines()]
    assets = {a['path']: a['sha256'] for a in json.loads((root / 'research_log/cycle006/frozen_assets.json').read_text())['assets']}
    expected_groups = [{'group_id': g['counterfactual_id'], 'pair_ids': g['pair_ids']} for g in groups]
    assert expected_groups == protocol['ordered_groups']
    summaries, reports, provenance = {}, {}, []
    for condition in ['clean', 'target15_b']:
        cma_dir = root / f'outputs/cycle008/val_base_{condition}'
        cma_path = cma_dir / 'memory_predictions.jsonl'
        assert sha(cma_path) == protocol['cma_prediction_manifests'][condition]['sha256']
        cma = [json.loads(l) for l in cma_path.read_text().splitlines()]
        segllm, cma_absolute = [], []
        for g, c in zip(groups, cma):
            gid = g['counterfactual_id']
            assert c['group_id'] == gid and c['condition'] == condition
            assert [t['entity_id'] for t in c['trials']] == g['pair_ids']
            trials = []
            c = dict(c, trials=[dict(t) for t in c['trials']])
            for pair, native, ct in zip(g['pairs'], grouped_predictions[(gid, condition)], c['trials']):
                target_path = pair['helmet_mask_path']
                assert sha(target_path) == assets[target_path]
                target = np.asarray(Image.open(target_path)) > 0
                cma_target = cma_dir / ct['target']
                assert np.array_equal(target, np.load(cma_target) > 0)
                assert np.array_equal(np.asarray(Image.open(pair['miner_mask_path'])) > 0,
                                      np.load(cma_dir / ct['reference']) > 0)
                provenance.append({'method': 'CMA base w15', 'condition': condition, 'group_id': gid,
                                   'entity_id': ct['entity_id'], 'prediction_sha256': sha(cma_dir / ct['prediction']),
                                   'target_sha256': sha(target_path), 'cma_target_exact_equal': True,
                                   'cma_reference_exact_equal': True})
                trials.append({'entity_id': native['entity_id'], 'prediction': native['prediction'], 'target': target_path})
                for key in ['prediction', 'target', 'reference']:
                    ct[key] = str(cma_dir / ct[key])
            cma_absolute.append(c)
            segllm.append({'group_id': gid, 'image_id': g['image_path'], 'query': g['same_round2_query'],
                           'condition': condition, 'memory_source': 'supplied_ref', 'trials': trials})
        for method, records in [('segllm', segllm), ('cma_base_w15', cma_absolute)]:
            stem = f'{method}_{condition}'
            manifest = out / (stem + '.jsonl')
            manifest.write_text(''.join(json.dumps(x) + '\n' for x in records))
            report = evaluate_file(manifest, out / (stem + '_metrics.json'))
            empty = sum(not np.load(t['prediction']).any() for g in records for t in g['trials'])
            summary = dict(report['summary'], empty_masks=int(empty), invalid_outputs=0)
            summary.update(cmsa_count=sum(p['success'] for g in report['groups'] for p in g['pairs']),
                           fidelity_count=sum(sum(g['fidelity']) for g in report['groups']),
                           ier_count=sum(sum(g['identity_error']) for g in report['groups']))
            summaries[stem] = summary
            reports[stem] = report
    deltas = {}
    for method in ['segllm', 'cma_base_w15']:
        clean, degraded = summaries[f'{method}_clean'], summaries[f'{method}_target15_b']
        per_group = []
        for c, d in zip(reports[f'{method}_clean']['groups'], reports[f'{method}_target15_b']['groups']):
            assert c['group_id'] == d['group_id']
            per_group.append({'group_id': c['group_id'],
                              'correct_iou_delta': (np.array(d['correct_iou']) - c['correct_iou']).tolist(),
                              'identity_margin_delta': (np.array(d['identity_margin']) - c['identity_margin']).tolist(),
                              'cmsa_delta': int(d['pairs'][0]['success']) - int(c['pairs'][0]['success'])})
        deltas[method] = {'aggregate_degraded_minus_clean': {k: degraded[k] - clean[k] for k in clean
                                                          if k not in ['num_groups', 'num_references', 'num_pairs']},
                         'paired_groups': per_group}
    (out / 'comparison.json').write_text(json.dumps({'summaries': summaries, 'paired_deltas': deltas}, indent=2) + '\n')
    (out / 'cma_reuse_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()

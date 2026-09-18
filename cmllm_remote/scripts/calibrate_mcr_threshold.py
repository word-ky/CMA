"""One shared scalar calibration on train300 only; never reads validation."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def accepted(row, tau):
    return row['action'] == 'ACCEPT' and row['group_contrast'] >= tau


def operating_metrics(rows, tau):
    counts = dict(success_to_accept=0, success_to_abstain=0,
                  failure_to_accept=0, failure_to_abstain=0)
    for row in rows:
        counts[('success' if row['cmsa_success'] else 'failure') + '_to_'
               + ('accept' if accepted(row, tau) else 'abstain')] += 1
    sa, sb, fa, fb = (counts[k] for k in counts)
    return {'coverage': (sa + fa) / len(rows),
            'accepted_cmsa': sa / (sa + fa) if sa + fa else None,
            'failure_recall': fb / (fa + fb) if fa + fb else None,
            'abstention_precision': fb / (sb + fb) if sb + fb else None,
            'counts': counts}


def choose_threshold(rows):
    candidates = sorted({r['group_contrast'] for r in rows if r['group_contrast'] > 0})
    table = []
    for tau in candidates:
        c = operating_metrics([r for r in rows if r['condition'] == 'clean'], tau)
        d = operating_metrics([r for r in rows if r['condition'] == 'target15_b'], tau)
        checks = {'degraded_coverage': d['coverage'] >= .5,
                  'degraded_cmsa': d['accepted_cmsa'] is not None and d['accepted_cmsa'] >= .75,
                  'failure_recall': d['failure_recall'] is not None and d['failure_recall'] >= .5,
                  'abstention_precision': d['abstention_precision'] is not None and d['abstention_precision'] >= .6,
                  'clean_coverage': c['coverage'] >= .8,
                  'clean_cmsa': c['accepted_cmsa'] is not None and c['accepted_cmsa'] >= .94}
        table.append({'tau': tau, 'clean': c, 'target15_b': d,
                      'checks': checks, 'feasible': all(checks.values())})
    feasible = [r for r in table if r['feasible']]
    return {'tau': feasible[0]['tau'] if feasible else None,
            'feasible_count': len(feasible), 'candidate_count': len(table), 'candidate_table': table}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from eval_counterfactual_memory_fidelity import evaluate_file
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    root = Path(p.parse_args().root)
    work = root / 'research_log/cycle021'
    frozen_path = work / 'train_scores_before_targets.json'
    frozen = json.loads(frozen_path.read_text())
    assert len(frozen['groups']) == 600
    assert frozen['provenance_sha256'] == sha(work / 'inference_provenance.json')
    for row in frozen['groups']:
        for t in row['trials']:
            assert sha(t['prediction']) == t['prediction_sha256']
            assert sha(t['miner_mask']) == t['miner_sha256']
    started = time.time()
    assert started > frozen['frozen_unix']
    out = work / 'train_scoring'
    out.mkdir(exist_ok=True)
    (out / 'freeze_audit.json').write_text(json.dumps({
        'train_prediction_score_sha256': sha(frozen_path), 'frozen_unix': frozen['frozen_unix'],
        'train_scoring_started_unix': started, 'all_hashes_rechecked': True,
        'validation_access': False}, indent=2) + '\n')
    pairs_path = root / 'shared/data/cycle006/helmet_miner_pairs_restored.jsonl'
    pairs = {r['pair_id']: r for r in map(json.loads, pairs_path.read_text().splitlines())}
    calibrated_rows = []
    for condition in ['clean', 'target15_b']:
        groups = [r for r in frozen['groups'] if r['condition'] == condition]
        records = []
        for r in groups:
            record = {k: r[k] for k in ['group_id', 'image_id', 'query', 'condition', 'memory_source']}
            record['trials'] = [{'entity_id': t['entity_id'], 'prediction': t['prediction'],
                                 'target': pairs[t['entity_id']]['helmet_mask_path']} for t in r['trials']]
            records.append(record)
        manifest = out / f'{condition}.jsonl'
        manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
        scored = evaluate_file(manifest, out / f'{condition}_metrics.json')
        by_id = {r['group_id']: r for r in scored['groups']}
        calibrated_rows.extend({**r, 'cmsa_success': bool(by_id[r['group_id']]['pairs'][0]['success'])}
                               for r in groups)
    selection = choose_threshold(calibrated_rows)
    selection.update({'frozen_unix': time.time(), 'train_scores_sha256': sha(frozen_path),
                      'train_labels_sha256': {c: sha(out / f'{c}_metrics.json') for c in ['clean', 'target15_b']},
                      'rule': 'original structural validity AND g>=tau; lowest feasible unique positive train g',
                      'validation_access': False, 'next_step': 'freeze_val_actions_then_score' if selection['tau'] is not None else 'retire_Layer2'})
    (work / 'calibration_receipt.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({k: v for k, v in selection.items() if k != 'candidate_table'}, indent=2))


if __name__ == '__main__':
    main()

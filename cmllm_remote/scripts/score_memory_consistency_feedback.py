"""Post-freeze evaluation and explicitly non-deployable candidate headroom."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from eval_counterfactual_memory_fidelity import evaluate_file, score_masks, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    work = root / 'research_log/cycle019'
    freeze = json.loads((work / 'prediction_freeze.json').read_text())
    assert len(freeze['trials']) == 200
    assert freeze['rule_sha256'] == sha(work / 'rule.json')
    assert freeze['plan_sha256'] == sha(work / 'actions_before_targets.json')
    for t in freeze['trials']:
        assert sha(t['prediction']) == t['prediction_sha256']
        assert sha(t['ref_prediction']) == t['ref_sha256']
    for t in freeze['direct_calls']:
        assert sha(t['prediction']) == t['sha256']
    out = work / 'scoring'
    out.mkdir(exist_ok=True)
    audit = {'frozen_unix': freeze['frozen_unix'], 'scoring_started_unix': time.time(),
             'all_final_ref_direct_hashes_rechecked': True,
             'runtime_raster_reads': freeze['raster_reads'],
             'rule_sha256': freeze['rule_sha256'], 'plan_sha256': freeze['plan_sha256']}
    assert audit['scoring_started_unix'] > audit['frozen_unix']
    (out / 'freeze_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    # Targets/scorer data are opened only below this point, in this separate process.
    lookup = {(t['group_id'], t['condition'], t['entity_id']): t for t in freeze['trials']}
    direct = {(t['group_id'], t['condition']): t for t in freeze['direct_calls']}
    result = {}
    for condition in ['clean', 'target15_b']:
        base_dir = root / f'outputs/cycle008/val_base_{condition}'
        base = [json.loads(l) for l in (base_dir / 'memory_predictions.jsonl').read_text().splitlines()]
        policies = {'ref_only': [], 'mcf': []}
        for row in base:
            for policy in policies:
                converted = dict(row, trials=[])
                for t in row['trials']:
                    final = lookup[(row['group_id'], condition, t['entity_id'])]
                    converted['trials'].append({'entity_id': t['entity_id'],
                        'prediction': final['prediction'] if policy == 'mcf' else final['ref_prediction'],
                        'target': str(base_dir / t['target'])})
                policies[policy].append(converted)
        reports = {}
        for policy, rows in policies.items():
            manifest = out / f'{policy}_{condition}.jsonl'
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            reports[policy] = evaluate_file(manifest, out / f'{policy}_{condition}_metrics.json')
        transitions = dict(fail_to_pass=0, pass_to_fail=0, pass_to_pass=0, fail_to_fail=0)
        oracle_rows, max_miou_rows, choices = [], [], []
        for i, (a, b) in enumerate(zip(reports['ref_only']['groups'], reports['mcf']['groups'])):
            before, after = bool(a['pairs'][0]['success']), bool(b['pairs'][0]['success'])
            transitions[('pass' if before else 'fail') + '_to_' + ('pass' if after else 'fail')] += 1
            candidate = direct.get((a['group_id'], condition))
            if candidate is None:
                oracle_rows.append(a); max_miou_rows.append(a)
                choices.append({'group_id': a['group_id'], 'choice': 'REF', 'direct_available': False})
                continue
            targets = [np.load(t['target']) for t in policies['ref_only'][i]['trials']]
            prediction = np.load(candidate['prediction'])
            direct_score = {'group_id': a['group_id'], **score_masks([prediction, prediction], targets)}
            rank = lambda x: (bool(x['pairs'][0]['success']), float(np.mean(x['correct_iou'])))
            use_direct = rank(direct_score) > rank(a)
            oracle_rows.append(direct_score if use_direct else a)
            max_miou_rows.append(direct_score if np.mean(direct_score['correct_iou']) > np.mean(a['correct_iou']) else a)
            choices.append({'group_id': a['group_id'], 'choice': 'DIRECT' if use_direct else 'REF',
                            'direct_available': True})
        rows = [t for t in freeze['trials'] if t['condition'] == condition]
        fallback = sum(t['action'] == 'DIRECT_FALLBACK' for t in rows)
        result[condition] = {'ref_only': reports['ref_only']['summary'], 'mcf': reports['mcf']['summary'],
            'cmsa_transitions': transitions, 'actions': {
                'ref_accept': 100 - fallback, 'ref_accept_rate': (100 - fallback) / 100,
                'fallback_requests': fallback, 'fallback_rate': fallback / 100,
                'direct_selected': sum(t['selected'] == 'DIRECT' for t in rows),
                'extra_direct_calls': sum(t['condition'] == condition for t in freeze['direct_calls'])},
            'oracle_union_ANALYSIS_ONLY': {
                'rule': 'Whole-group REF pair versus shared direct mask pair; GT CMSA then group mIoU; ties REF. Only actually scheduled direct candidates available.',
                'summary': summarize(oracle_rows), 'max_group_miou_summary': summarize(max_miou_rows),
                'choices': choices, 'coverage_limit': 'Not a full all-images direct oracle. Unscheduled candidates were not inferred.'}}
    c, d = result['clean'], result['target15_b']
    checks = {
        'degraded_miou_not_lower': d['mcf']['target_miou'] >= d['ref_only']['target_miou'],
        'degraded_cmsa_not_lower': d['mcf']['cmsa'] >= d['ref_only']['cmsa'],
        'at_least_two_recoveries': d['cmsa_transitions']['fail_to_pass'] >= 2,
        'at_most_one_regression': d['cmsa_transitions']['pass_to_fail'] <= 1,
        'both_actions_used': 0 < d['actions']['fallback_requests'] < 100,
        'clean_miou_preserved': c['mcf']['target_miou'] >= c['ref_only']['target_miou'] - .01,
        'clean_cmsa_preserved': c['mcf']['cmsa'] >= c['ref_only']['cmsa'] - 1/50,
    }
    result['fixed_gate'] = {'passed': all(checks.values()), 'checks': checks}
    union = d['oracle_union_ANALYSIS_ONLY']['summary']
    result['measured_oracle_complementarity'] = {
        'miou_gain': union['target_miou'] - d['ref_only']['target_miou'],
        'cmsa_recovery_opportunities': round((union['cmsa'] - d['ref_only']['cmsa']) * 50),
        'scope': 'scheduled direct candidates only; unscheduled action headroom unknown'}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {a: v for a, v in value.items() if a != 'oracle_union_ANALYSIS_ONLY'}
                      if k in ['clean', 'target15_b'] else value for k, value in result.items()}, indent=2))


if __name__ == '__main__':
    main()

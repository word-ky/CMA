"""Post-freeze selective evaluation; continuous-score curves are diagnostic only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from eval_counterfactual_memory_fidelity import evaluate_file, summarize


def diagnostic_auc(labels, scores):
    """Exact tie-aware empirical AUROC, average precision and trapezoidal PR area."""
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    positive, negative = scores[labels], scores[~labels]
    roc = np.mean((positive[:, None] > negative[None, :]).astype(float)
                  + .5 * (positive[:, None] == negative[None, :]))
    recalls, precisions, ap = [0.0], [1.0], 0.0
    for threshold in sorted(set(scores.tolist()), reverse=True):
        selected = scores >= threshold
        recall = float(labels[selected].sum() / labels.sum())
        precision = float(labels[selected].mean())
        ap += (recall - recalls[-1]) * precision
        recalls.append(recall); precisions.append(precision)
    pr_area = sum((recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2
                  for i in range(1, len(recalls)))
    return float(roc), float(ap), float(pr_area)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    work = root / 'research_log/cycle020'
    action_path = work / 'actions_before_targets.json'
    frozen = json.loads(action_path.read_text())
    assert frozen['input_spec_sha256'] == sha(work / 'input_spec.json')
    for group in frozen['groups']:
        for t in group['trials']:
            assert sha(t['prediction']) == t['prediction_sha256']
            assert sha(t['miner_mask']) == t['miner_sha256']
    out = work / 'scoring'
    out.mkdir(exist_ok=True)
    audit = {'action_sha256': sha(action_path), 'frozen_unix': frozen['frozen_unix'],
             'scoring_started_unix': time.time(), 'all_input_hashes_rechecked': True,
             'raster_reads_before_targets': frozen['raster_reads']}
    assert audit['scoring_started_unix'] > audit['frozen_unix']
    (out / 'freeze_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    # Only now open target-bearing scoring manifests and invoke unchanged CMF.
    lookup = {(r['group_id'], r['condition']): r for r in frozen['groups']}
    result = {}
    for condition in ['clean', 'target15_b']:
        base = root / f'outputs/cycle008/val_base_{condition}'
        rows = [json.loads(l) for l in (base / 'memory_predictions.jsonl').read_text().splitlines()]
        records = []
        for row in rows:
            decision = lookup[(row['group_id'], condition)]
            assert [t['entity_id'] for t in row['trials']] == [t['entity_id'] for t in decision['trials']]
            converted = dict(row, trials=[])
            for target, prediction in zip(row['trials'], decision['trials']):
                converted['trials'].append({'entity_id': target['entity_id'],
                                           'prediction': prediction['prediction'],
                                           'target': str(base / target['target'])})
            records.append(converted)
        manifest = out / f'base_{condition}.jsonl'
        manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
        report = evaluate_file(manifest, out / f'base_{condition}_metrics.json')
        accept, abstain, labels, contrasts, per_group = [], [], [], [], []
        counts = dict(success_to_accept=0, success_to_abstain=0, failure_to_accept=0, failure_to_abstain=0)
        for metric in report['groups']:
            decision = lookup[(metric['group_id'], condition)]
            success = bool(metric['pairs'][0]['success'])
            accepted = decision['action'] == 'ACCEPT'
            (accept if accepted else abstain).append(metric)
            counts[('success' if success else 'failure') + '_to_' + ('accept' if accepted else 'abstain')] += 1
            labels.append(int(success)); contrasts.append(decision['group_contrast'])
            per_group.append({'group_id': metric['group_id'], 'action': decision['action'],
                              'g': decision['group_contrast'], 'cmsa_success': success})
        failures = counts['failure_to_accept'] + counts['failure_to_abstain']
        auroc, average_precision, pr_area = diagnostic_auc(labels, contrasts)
        curve = [{'coverage': 0.0, 'accepted_groups': 0, 'risk_cmsa_failure': None, 'threshold_g_ge': None}]
        # Include all tied groups together. No arbitrary ordering or threshold selection.
        for threshold in sorted(set(contrasts), reverse=True):
            selected = np.asarray(contrasts) >= threshold
            curve.append({'threshold_g_ge': threshold, 'accepted_groups': int(selected.sum()),
                          'coverage': float(selected.mean()),
                          'risk_cmsa_failure': float(1 - np.asarray(labels)[selected].mean())})
        result[condition] = {
            'base_summary': report['summary'], 'accepted_groups': len(accept), 'total_groups': len(rows),
            'coverage': len(accept) / len(rows), 'abstained_groups': len(abstain),
            'abstention_rate': len(abstain) / len(rows), 'accepted_summary': summarize(accept),
            'abstained_summary': summarize(abstain),
            'abstained_cmsa_failure_rate': counts['failure_to_abstain'] / len(abstain),
            'failure_recall': counts['failure_to_abstain'] / failures,
            'abstention_precision': counts['failure_to_abstain'] / len(abstain), 'counts': counts,
            'diagnostics_ANALYSIS_ONLY': {'positive_label': 'CMSA success',
                'auroc': auroc,
                'auprc_average_precision': average_precision,
                'auprc_trapezoidal': pr_area,
                'risk_coverage_curve': curve, 'ties_grouped': True,
                'runtime_rule_changed': False, 'threshold_selected': False}, 'per_group': per_group}
    c, d = result['clean'], result['target15_b']
    checks = {'degraded_both_actions': 0 < d['coverage'] < 1,
              'degraded_coverage_at_least_half': d['coverage'] >= .5,
              'degraded_accepted_cmsa_at_least_075': d['accepted_summary']['cmsa'] >= .75,
              'at_least_11_failures_abstained': d['counts']['failure_to_abstain'] >= 11,
              'abstention_precision_at_least_060': d['abstention_precision'] >= .6,
              'clean_coverage_at_least_080': c['coverage'] >= .8,
              'clean_accepted_cmsa_at_least_094': c['accepted_summary']['cmsa'] >= .94}
    result['fixed_gate'] = {'passed': all(checks.values()), 'checks': checks}
    result['next_rule_branch'] = ('confirm_fixed_MCR' if all(checks.values()) else
                                 'later_training_only_calibration_review' if d['diagnostics_ANALYSIS_ONLY']['auroc'] >= .75 else
                                 'stop_Layer2_development')
    (out / 'selective_reliability.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {key: value for key, value in v.items() if key not in ['per_group', 'diagnostics_ANALYSIS_ONLY']}
                      if k in ['clean', 'target15_b'] else v for k, v in result.items()}, indent=2))
    print('DEGRADED_DIAGNOSTIC_AUROC', d['diagnostics_ANALYSIS_ONLY']['auroc'])


if __name__ == '__main__':
    main()

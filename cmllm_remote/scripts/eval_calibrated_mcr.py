"""Apply feasible train-only tau once to frozen val scores, freeze actions, then score."""
import argparse
import json
import time
from pathlib import Path
from calibrate_mcr_threshold import accepted, operating_metrics, sha
from eval_counterfactual_memory_fidelity import evaluate_file, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    root = Path(p.parse_args().root)
    work = root / 'research_log/cycle021'
    calibration_path = work / 'calibration_receipt.json'
    calibration = json.loads(calibration_path.read_text())
    tau = calibration['tau']
    assert tau is not None and tau > 0
    old_path = root / 'research_log/cycle020/actions_before_targets.json'
    source = json.loads(old_path.read_text())
    groups = source['groups']
    for row in groups:
        for trial in row['trials']:
            assert sha(trial['prediction']) == trial['prediction_sha256']
            assert sha(trial['miner_mask']) == trial['miner_sha256']
    actions = [{**r, 'calibrated_action': 'ACCEPT' if accepted(r, tau) else 'ABSTAIN_ESCALATE'} for r in groups]
    frozen = {'tau': tau, 'frozen_unix': time.time(), 'calibration_sha256': sha(calibration_path),
              'cycle020_actions_sha256': sha(old_path), 'groups': actions,
              'counts': {c: sum(r['condition'] == c and r['calibrated_action'] == 'ACCEPT' for r in actions)
                         for c in ['clean', 'target15_b']}}
    assert frozen['frozen_unix'] > calibration['frozen_unix']
    path = work / 'val_actions_before_targets.json'
    path.write_text(json.dumps(frozen, indent=2) + '\n')
    out = work / 'val_scoring'
    out.mkdir(exist_ok=True)
    (out / 'freeze_audit.json').write_text(json.dumps({
        'val_actions_sha256': sha(path), 'calibration_frozen_unix': calibration['frozen_unix'],
        'actions_frozen_unix': frozen['frozen_unix'], 'scoring_started_unix': time.time()}, indent=2) + '\n')
    # Target-bearing manifests are first opened after the action receipt is written.
    result = {}
    for condition in ['clean', 'target15_b']:
        base = root / f'outputs/cycle008/val_base_{condition}'
        source_rows = [json.loads(line) for line in (base / 'memory_predictions.jsonl').read_text().splitlines()]
        chosen = {r['group_id']: r for r in actions if r['condition'] == condition}
        records = []
        for row in source_rows:
            decision = chosen[row['group_id']]
            assert [t['entity_id'] for t in row['trials']] == [t['entity_id'] for t in decision['trials']]
            records.append({**row, 'trials': [dict(entity_id=t['entity_id'], prediction=d['prediction'],
                            target=str(base / t['target'])) for t, d in zip(row['trials'], decision['trials'])]})
        manifest = out / f'{condition}.jsonl'
        manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
        report = evaluate_file(manifest, out / f'{condition}_metrics.json')
        selected, combined = [], []
        for metric in report['groups']:
            row = chosen[metric['group_id']]
            combined.append({**row, 'cmsa_success': bool(metric['pairs'][0]['success'])})
            if row['calibrated_action'] == 'ACCEPT':
                selected.append(metric)
        result[condition] = {**operating_metrics(combined, tau), 'accepted_groups': len(selected),
                             'total_groups': len(combined), 'accepted_summary': summarize(selected) if selected else None,
                             'base_summary': report['summary']}
    c, d = result['clean'], result['target15_b']
    checks = {'both_degraded_actions': 0 < d['coverage'] < 1,
              'degraded_coverage': d['accepted_groups'] >= 25,
              'degraded_cmsa': d['accepted_cmsa'] is not None and d['accepted_cmsa'] >= .75,
              'degraded_failures_caught': d['counts']['failure_to_abstain'] >= 11,
              'abstention_precision': d['abstention_precision'] is not None and d['abstention_precision'] >= .6,
              'clean_coverage': c['accepted_groups'] >= 40,
              'clean_cmsa': c['accepted_cmsa'] is not None and c['accepted_cmsa'] >= .94}
    result.update(tau=tau, fixed_gate={'passed': all(checks.values()), 'checks': checks},
                  next_step='one_untouched_confirmation_review' if all(checks.values()) else 'retire_Layer2')
    (out / 'selective_reliability.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

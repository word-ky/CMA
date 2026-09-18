"""Loop the unchanged four-trial native smoke over frozen val50, loading once."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from load_native import load_native
from run_smoke import main as four_trials


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--clip-path', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    manifest = Path(args.manifest)
    groups = json.loads(manifest.read_text())
    assert len(groups) == 50
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model_bundle = load_native(args.checkpoint, args.clip_path, out)
    records = []
    for index, group in enumerate(groups):
        group_out = out / group['group_id']
        receipt = four_trials([
            '--checkpoint', args.checkpoint, '--clip-path', args.clip_path,
            '--prepared', group['prepared'], '--output-dir', str(group_out),
        ], model_bundle=model_bundle)
        for trial in receipt['trials']:
            prediction = Path(trial['prediction'])
            records.append({'group_id': group['group_id'], **trial,
                            'prediction_sha256': hashlib.sha256(prediction.read_bytes()).hexdigest()})
        (out / 'completed_predictions.json').write_text(json.dumps(records, indent=2) + '\n')
        print(f'COMPLETED_GROUPS={index + 1}/50 FORWARDS={len(records)}/200', flush=True)
    assert len(records) == 200
    freeze = {'status': 'ALL_200_PREDICTIONS_FROZEN_BEFORE_SCORING',
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'started_unix': started, 'frozen_unix': time.time(),
              'groups': 50, 'forward_count': len(records), 'trials': records,
              'scorer_called': False, 'helmet_targets_read': False}
    (out / 'prediction_freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')


if __name__ == '__main__':
    main()

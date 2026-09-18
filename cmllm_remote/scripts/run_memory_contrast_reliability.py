"""Compute/freeze MCR actions; only prediction and supplied-memory rasters allowed."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from memory_contrast_reliability import decide


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    spec_path = Path(args.inputs)
    spec = json.loads(spec_path.read_text())
    allowed = {str(Path(t[k]).resolve()) for g in spec['groups'] for t in g['trials']
               for k in ['prediction', 'miner_mask']}
    reads = set()
    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], str) and isinstance(arguments[1], str) and 'r' in arguments[1]:
            path = Path(arguments[0]).resolve()
            if path.suffix.lower() in {'.npy', '.png', '.jpg', '.jpeg'}:
                assert str(path) in allowed, str(path)
                reads.add(str(path))
            if path.suffix == '.json':
                assert '/scoring/' not in path.as_posix() and 'metrics' not in path.name
    sys.addaudithook(audit)
    rows = []
    for group in spec['groups']:
        predictions, memories = [], []
        for trial in group['trials']:
            for field, hash_field in [('prediction', 'prediction_sha256'), ('miner_mask', 'miner_sha256')]:
                assert hashlib.sha256(Path(trial[field]).read_bytes()).hexdigest() == trial[hash_field]
            predictions.append(np.load(trial['prediction']))
            memories.append(np.asarray(Image.open(trial['miner_mask'])) > 0)
        rows.append({**group, **decide(predictions, memories)})
    counts = {condition: {'accepted': sum(r['action'] == 'ACCEPT' for r in rows if r['condition'] == condition),
                          'total': sum(r['condition'] == condition for r in rows)}
              for condition in ['clean', 'target15_b']}
    result = {'status': 'MCR_ACTIONS_FROZEN_BEFORE_TARGETS', 'frozen_unix': time.time(),
              'input_spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
              'rule': spec['rule'], 'counts': counts, 'groups': rows, 'raster_reads': sorted(reads),
              'model_imports': False, 'new_forwards': 0}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()

"""Target-free plan/run phases for the fixed Cycle019 MCF rule."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from memory_consistency_feedback import DIRECT_QUERY, THRESHOLD, initial_action, select_candidate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_audit(requests):
    allowed = {str(Path(r[k]).resolve()) for r in requests
               for k in ['ref_prediction', 'miner_mask', 'image_path']}
    reads = set()
    def audit(event, arguments):
        if event != 'open' or not isinstance(arguments[0], str):
            return
        mode = arguments[1]
        if not isinstance(mode, str) or 'r' not in mode:
            return
        path = Path(arguments[0]).resolve()
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.npy'}:
            assert str(path) in allowed, str(path)
            reads.add(str(path))
        if path.suffix == '.json':
            assert '/scoring/' not in path.as_posix() and 'metrics' not in path.name, str(path)
    sys.addaudithook(audit)
    return reads, allowed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['plan', 'run'])
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / 'research_log/cycle019'
    requests = json.loads((out / 'requests.json').read_text())
    rule = json.loads((out / 'rule.json').read_text())
    assert rule['threshold'] == THRESHOLD and rule['direct_query'] == DIRECT_QUERY
    for path, expected in rule['cma_manifest_hashes'].items():
        assert sha(path) == expected
    reads, allowed = install_audit(requests)
    if args.phase == 'plan':
        actions = []
        for request in requests:
            ref = np.load(request['ref_prediction'])
            miner = np.asarray(Image.open(request['miner_mask'])) > 0
            actions.append({**request, **initial_action(ref, miner),
                            'ref_sha256': sha(request['ref_prediction']),
                            'miner_sha256': sha(request['miner_mask'])})
        counts = {}
        for condition in ['clean', 'target15_b']:
            rows = [r for r in actions if r['condition'] == condition]
            fallback = [r for r in rows if r['action'] == 'DIRECT_FALLBACK']
            counts[condition] = {'requests': len(rows), 'ref_accept': len(rows) - len(fallback),
                                 'fallback_requests': len(fallback),
                                 'planned_direct_calls': len({r['group_id'] for r in fallback})}
        (out / 'actions_before_targets.json').write_text(json.dumps(
            {'frozen_unix': time.time(), 'rule_sha256': sha(out / 'rule.json'),
             'counts': counts, 'actions': actions, 'raster_reads': sorted(reads)}, indent=2) + '\n')
        print(json.dumps(counts, indent=2))
        return
    plan = json.loads((out / 'actions_before_targets.json').read_text())
    assert plan['rule_sha256'] == sha(out / 'rule.json')
    actions = plan['actions']
    calls = {}
    for action in actions:
        if action['action'] == 'DIRECT_FALLBACK':
            calls.setdefault((action['group_id'], action['condition']), action)
    direct_masks, call_receipts = {}, []
    if calls:
        import torch
        from stage3_rule_controller_v2_direct_first import ControllerRunner
        runner = ControllerRunner(SimpleNamespace(
            model=str(root / 'shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf'),
            model_max_length=512, precision='bf16', image_size=1024,
            vision_tower=str(root / 'shared/models/clip-vit-large-patch14'), vision_pretrained=None))
        def check_placeholders(module, positional, kwargs):
            assert kwargs['inference']
            assert all(torch.count_nonzero(m).item() == 0 for m in kwargs['masks_list'])
            assert all(torch.count_nonzero(m).item() == 0 for m in kwargs['ref_valids_list'])
        hook = runner.model.register_forward_pre_hook(check_placeholders, with_kwargs=True)
        for key, request in calls.items():
            assert sha(request['image_path']) == request['image_sha256']
            rgb = np.asarray(Image.open(request['image_path']).convert('RGB'))
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            pred = runner.predict(rgb, DIRECT_QUERY, np.zeros(rgb.shape[:2], dtype=np.float32))
            torch.cuda.synchronize()
            direct_masks[key] = pred
            path = out / 'direct' / f'{key[0]}_{key[1]}.npy'
            path.parent.mkdir(exist_ok=True)
            np.save(path, pred.astype(np.uint8))
            allowed.add(str(path.resolve()))
            call_receipts.append({'group_id': key[0], 'condition': key[1], 'prediction': str(path),
                                  'sha256': sha(path), 'seconds': time.monotonic() - started,
                                  'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                                  'zero_target_and_no_ref_verified_at_forward': True})
        hook.remove()
    records = []
    for action in actions:
        assert sha(action['ref_prediction']) == action['ref_sha256']
        assert sha(action['miner_mask']) == action['miner_sha256']
        ref = np.load(action['ref_prediction'])
        miner = np.asarray(Image.open(action['miner_mask'])) > 0
        direct = direct_masks.get((action['group_id'], action['condition']))
        selected, trace = select_candidate(ref, miner, direct)
        assert trace['action'] == action['action'] and trace['head_score_ref'] == action['head_score_ref']
        path = out / 'selected' / f"{action['condition']}_{action['entity_id']}.npy"
        path.parent.mkdir(exist_ok=True)
        np.save(path, (selected > 0).astype(np.uint8))
        allowed.add(str(path.resolve()))
        records.append({**action, **trace, 'prediction': str(path), 'prediction_sha256': sha(path)})
    freeze = {'status': 'MCF_PREDICTIONS_AND_ACTIONS_FROZEN_BEFORE_TARGETS',
              'frozen_unix': time.time(), 'rule_sha256': sha(out / 'rule.json'),
              'plan_sha256': sha(out / 'actions_before_targets.json'),
              'trials': records, 'direct_calls': call_receipts, 'raster_reads': sorted(reads)}
    (out / 'prediction_freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    print(f'FROZEN {len(records)} requests; {len(call_receipts)} direct calls')


if __name__ == '__main__':
    main()

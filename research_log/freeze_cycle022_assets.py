"""Freeze restored confirmation assets and prepare the pinned SegLLM inputs."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'code/cmllm/external_baselines/segllm'))
from prepare_inputs import prepare
out = root / 'research_log/cycle022'
data = root / 'shared/data/cycle022'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines()]
selection = json.loads((out / 'selection_receipt.json').read_text())
groups = load(data / 'counterfactual_selected_restored.jsonl')
pairs = {r['pair_id']: r for r in load(data / 'helmet_miner_pairs_restored.jsonl')}
assets = json.loads((out / 'frozen_assets.json').read_text())
prior = json.loads((root / 'research_log/cycle018/protocol_receipt.json').read_text())
for name, digest in prior['port_file_sha256'].items():
    assert sha(root / 'code/cmllm/external_baselines/segllm' / name) == digest
for name in ['counterfactual_export.py', 'eval_counterfactual_memory_fidelity.py']:
    assert sha(root / 'code/cmllm/scripts' / name) == prior['frozen_files']['cmllm_remote/scripts/' + name]
assert len(groups) == selection['groups'] >= 30
for asset in assets['assets']:
    assert sha(asset['path']) == asset['sha256']
specs, prepared = [], []
for group, selected in zip(groups, selection['selected']):
    assert group['counterfactual_id'] == selected['group_id']
    assert group['pair_ids'] == selected['pair_ids']
    assert sha(group['image_path']) == selected['image_sha256']
    shape = np.asarray(Image.open(group['image_path'])).shape[:2]
    identities = []
    for pid in group['pair_ids']:
        pair = pairs[pid]
        for key in ['miner_mask_path', 'helmet_mask_path']:
            mask = np.asarray(Image.open(pair[key])) > 0
            assert mask.shape == shape and mask.any(), (pid, key)
        identities.append({k: pair[k] for k in ['miner_mask_path', 'miner_bbox_xyxy']})
        identities[-1]['entity_id'] = pid
    assert sha(identities[0]['miner_mask_path']) != sha(identities[1]['miner_mask_path'])
    spec = {'group_id': group['counterfactual_id'], 'source_image': group['image_path'],
            'semantic_query': group['same_round2_query'],
            'native_prompt': 'Segment the mining helmet worn by instance 1.[REF:1]',
            'seed': 0, 'conditions': ['clean', 'target15_b'], 'identities': identities,
            'input_asset_hashes': {p: sha(p) for p in [group['image_path']] + [i['miner_mask_path'] for i in identities]}}
    dest = out / 'prepared' / spec['group_id']
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'inputs.json'
    path.write_text(json.dumps(spec, indent=2) + '\n')
    receipt = prepare(path, dest)
    prepared.append({'group_id': spec['group_id'], 'prepared': str(dest / 'preparation_receipt.json'),
                     'source_image': spec['source_image'], 'source_sha256': selected['image_sha256'],
                     'preparation': receipt})
    specs.append(spec)
(out / 'inference_specs.json').write_text(json.dumps(specs, indent=2) + '\n')
(out / 'execution_manifest.json').write_text(json.dumps(prepared, indent=2) + '\n')
files = ['scripts/eval_mr_ref_counterfactual_v0.py', 'scripts/counterfactual_export.py',
         'scripts/eval_counterfactual_memory_fidelity.py', 'scripts/run_mcr_train_predictions.py']
files += ['external_baselines/segllm/' + f for f in ['run_smoke.py', 'load_native.py', 'weight_fidelity.py', 'memory_state.py', 'prepare_inputs.py', 'run_val50.py']]
protocol = {'frozen_unix': time.time(), 'groups': len(groups), 'selection_sha256': sha(out / 'selection_receipt.json'),
            'asset_receipt_sha256': sha(out / 'frozen_assets.json'),
            'group_manifest_sha256': sha(data / 'counterfactual_selected_restored.jsonl'),
            'pair_manifest_sha256': sha(data / 'helmet_miner_pairs_restored.jsonl'),
            'execution_manifest_sha256': sha(out / 'execution_manifest.json'),
            'frozen_code_hashes': {f: sha(root / 'code/cmllm' / f) for f in files},
            'zero_overlap_with_prior_used_images': selection['zero_image_byte_overlap'],
            'segllm_source_revision': prior['source_revision'],
            'segllm_checkpoint_revision': prior['checkpoint_revision'],
            'all_segllm_port_hashes_equal_cycle018': True,
            'target_access_here': 'asset validity/hash freeze only, before model inference; no prediction scoring',
            'pseudo_labels': 'Same fixed SAM-B restoration pipeline; no hand corrections',
            'layer2': False, 'tuning': False}
(out / 'protocol_receipt.json').write_text(json.dumps(protocol, indent=2) + '\n')
print('FROZEN_VALID_ASSETS', len(groups), flush=True)

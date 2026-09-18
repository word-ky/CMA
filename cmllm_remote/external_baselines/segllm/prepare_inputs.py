"""Prepare the two frozen RGB observations, independently of model loading."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from counterfactual_export import condition_image, group_seed


def prepare(inputs_path, output_dir):
    spec = json.loads(Path(inputs_path).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(spec['source_image']).convert('RGB'))
    identities = []
    for identity in spec['identities']:
        mask_path = Path(identity['miner_mask_path'])
        mask = np.asarray(Image.open(mask_path)) > 0
        assert mask.shape == rgb.shape[:2]
        identities.append({**identity, 'miner_mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()})
    assert len(identities) == 2
    assert identities[0]['miner_mask_sha256'] != identities[1]['miner_mask_sha256']
    observations = []
    for condition in spec['conditions']:
        image = condition_image(rgb, condition, group_seed(spec['group_id'], spec['seed']))
        path = output_dir / (condition + '.png')
        Image.fromarray(image).save(path)
        assert np.array_equal(np.asarray(Image.open(path).convert('RGB')), image)
        observations.append({'condition': condition, 'image_path': str(path.resolve()),
                             'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                             'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'identities': identities})
    assert observations[0]['rgb_sha256'] != observations[1]['rgb_sha256']
    receipt = {'status': 'COMMON_INPUTS_PREPARED_NOT_NATIVE_WIRING',
               'group_id': spec['group_id'], 'seed': spec['seed'],
               'group_seed': group_seed(spec['group_id'], spec['seed']),
               'semantic_query': spec['semantic_query'], 'native_prompt': spec['native_prompt'],
               'original_hw': list(rgb.shape[:2]), 'observations': observations,
               'note': 'Each A/B pair references one identical image file; supplied geometry is shared across conditions. Native crop/CLIP/bbox/index receipts still required.'}
    (output_dir / 'preparation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inputs, args.output_dir), indent=2))

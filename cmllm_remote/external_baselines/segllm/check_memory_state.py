"""Actual native preprocessing check, before any model forward or scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor
from segment_anything.utils.transforms import ResizeLongestSide
from llava.model.segmentator.hipie_utils import Preprocessor

from memory_state import seed_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--clip-path', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepared = json.loads(Path(args.prepared).read_text())
    processor = Preprocessor(1024, ResizeLongestSide(1024))
    clip = CLIPImageProcessor.from_pretrained(args.clip_path)
    rows = []
    for obs in prepared['observations']:
        rgb = np.asarray(Image.open(obs['image_path']).convert('RGB'))
        for identity in obs['identities']:
            mask = np.asarray(Image.open(identity['miner_mask_path'])) > 0
            _, _, receipt = seed_memory(rgb, mask, identity['miner_bbox_xyxy'], processor, clip)
            rows.append({'condition': obs['condition'], 'entity_id': identity['entity_id'], **receipt})
    a, b, da, db = rows
    assert a['main_rgb_sha256'] == b['main_rgb_sha256']
    assert da['main_rgb_sha256'] == db['main_rgb_sha256']
    assert a['main_rgb_sha256'] != da['main_rgb_sha256']
    for left, right in [(a, b), (da, db)]:
        assert left['appearance_sha256'] != right['appearance_sha256']
        assert left['bbox_sha256'] != right['bbox_sha256']
    for clean, degraded in [(a, da), (b, db)]:
        assert clean['bbox_sha256'] == degraded['bbox_sha256']
        assert clean['appearance_sha256'] != degraded['appearance_sha256']
    result = {'status': 'NATIVE_PREPROCESSING_PASSED', 'rows': rows,
              'native_prompt': prepared['native_prompt'], 'forward_calls': 0,
              'history_index_runtime_check': 'NOT RUN; requires native conversation construction',
              'clip_preprocessor': clip.to_dict()}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])


if __name__ == '__main__':
    main()

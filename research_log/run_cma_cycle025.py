"""Frozen supplied-memory REF inference on the Cycle025 confirmation manifest."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from run_mcr_train_predictions import load_base_w15, sha, ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    root = Path(p.parse_args().root)
    work = root / 'research_log/cycle025'
    out = root / 'outputs/cycle025/cma'
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((work / 'protocol_receipt.json').read_text())
    data = root / 'shared/data/cycle025'
    groups_path, pairs_path = data / 'counterfactual_selected_restored.jsonl', data / 'helmet_miner_pairs_restored.jsonl'
    assert sha(groups_path) == protocol['group_manifest_sha256']
    assert sha(pairs_path) == protocol['pair_manifest_sha256']
    groups = ref.read_jsonl(groups_path)
    pairs = {r['pair_id']: r for r in ref.read_jsonl(pairs_path)}
    specs = json.loads((work / 'inference_specs.json').read_text())
    allowed = {str(Path(path).resolve()) for s in specs for path in s['input_asset_hashes']}
    reads, original = set(), ref.cv2.imread
    def audited(path, *a, **kw):
        path = str(Path(path).resolve())
        assert path in allowed, path
        reads.add(path)
        return original(path, *a, **kw)
    ref.cv2.imread = audited
    for spec in specs:
        for path, digest in spec['input_asset_hashes'].items():
            assert sha(path) == digest
    checkpoint = root / 'shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf'
    weight_hashes = json.loads((root / 'research_log/cycle021/inference_provenance.json').read_text())['model_files_sha256']
    for name, digest in weight_hashes.items():
        assert sha(checkpoint / name) == digest
    started = time.time()
    model, tokenizer, dtype, processor, transform = load_base_w15(root)
    def hook(module, args, kwargs):
        assert kwargs['inference']
        assert all(ref.torch.count_nonzero(m).item() == 0 for m in kwargs['masks_list'])
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    records = []
    for condition in ['clean', 'target15_b']:
        for index, group in enumerate(groups):
            item, rgb, targets, references, indices = ref.build_item(
                group, pairs, processor, transform, 1024, 'v1_multiround',
                ref_image_mode='crop', condition=condition, seed=0, read_targets=False)
            # Compare actual CMA observation pixels to the frozen SegLLM observation.
            common = work / 'prepared' / group['counterfactual_id'] / (condition + '.png')
            reads.add(str(common.resolve()))
            assert np.array_equal(rgb, np.asarray(Image.open(common).convert('RGB')))
            fields = list(item)
            fields[4] = ref.torch.zeros_like(item[4])
            predictions = ref.predict_item(model, tuple(fields), tokenizer, dtype, indices)
            for pid, mask in zip(group['pair_ids'], predictions):
                dest = out / f'{group["counterfactual_id"]}_{condition}_{pid}.npy'
                np.save(dest, mask.astype(np.uint8))
                records.append({'group_id': group['counterfactual_id'], 'condition': condition,
                                'entity_id': pid, 'prediction': str(dest), 'prediction_sha256': sha(dest)})
            if index % 10 == 0:
                print(condition, index + 1, 'groups', flush=True)
    handle.remove()
    freeze = {'started_unix': started, 'frozen_unix': time.time(), 'groups': len(groups),
              'group_forward_count': len(groups) * 2, 'identity_predictions': len(records),
              'trials': records, 'raster_reads': sorted(reads), 'zero_target_hook_passed': True,
              'same_condition_pixels_as_segllm': True, 'protocol_sha256': sha(work / 'protocol_receipt.json'),
              'elapsed_seconds': time.time() - started, 'peak_cuda_bytes': ref.torch.cuda.max_memory_allocated(),
              'checkpoint_provenance': weight_hashes}
    (out / 'prediction_freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    print('CMA_ALL_PREDICTIONS_FROZEN', len(records), flush=True)


if __name__ == '__main__':
    main()

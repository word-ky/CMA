"""Cycle021: unchanged base-w15 REF inference, no helmet-target reads."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import eval_mr_ref_counterfactual_v0 as ref
from memory_contrast_reliability import decide


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    root = Path(p.parse_args().root)
    out = root / 'research_log/cycle021'
    masks = root / 'outputs/cycle021/train_predictions'
    out.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ready = json.loads((root / 'research_log/cycle006/assets_ready.json').read_text())
    split = json.loads((root / 'research_log/cycle006/split_receipt.json').read_text())
    groups_path = Path(ready['manifests']['train_eval_restored']['path'])
    assert sha(groups_path) == ready['manifests']['train_eval_restored']['sha256']
    groups = ref.read_jsonl(groups_path)
    assert [r['counterfactual_id'] for r in groups] == [r['group_id'] for r in split['selected']['train']]
    assert len(groups) == 300 and split['train_val_and_all_holdout_images_disjoint_by_bytes']
    pairs_path = root / 'shared/data/cycle006/helmet_miner_pairs_restored.jsonl'
    pairs = {r['pair_id']: r for r in ref.read_jsonl(pairs_path)}
    expected_images = {r['group_id']: r['image_sha256'] for r in split['selected']['train']}
    inputs = []
    allowed = set()
    for group in groups:
        assert sha(group['image_path']) == expected_images[group['counterfactual_id']]
        allowed.add(str(Path(group['image_path']).resolve()))
        for pid in group['pair_ids']:
            path = pairs[pid]['miner_mask_path']
            allowed.add(str(Path(path).resolve()))
            inputs.append({'group_id': group['counterfactual_id'], 'entity_id': pid,
                           'image_sha256': expected_images[group['counterfactual_id']],
                           'miner_mask': path, 'miner_sha256': sha(path)})
    # build_item reads rasters via cv2; explicitly audit that actual read surface.
    original_imread, reads = ref.cv2.imread, set()
    def audited_imread(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        assert resolved in allowed, resolved
        reads.add(resolved)
        return original_imread(path, *args, **kwargs)
    ref.cv2.imread = audited_imread
    model_path = root / 'shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf'
    vision_path = root / 'shared/models/clip-vit-large-patch14'
    model_files = sorted(p for p in model_path.iterdir() if p.is_file())
    provenance = {'started_unix': started, 'model': str(model_path), 'adaptation': None,
                  'model_files_sha256': {p.name: sha(p) for p in model_files},
                  'train_manifest_sha256': sha(groups_path), 'pairs_manifest_sha256': sha(pairs_path),
                  'split_receipt_sha256': sha(root / 'research_log/cycle006/split_receipt.json'),
                  'inputs': inputs, 'seed': 0, 'precision': 'bf16',
                  'conversation_mode': 'v1_multiround', 'ref_image_mode': 'crop',
                  'conditions': ['clean', 'target15_b'], 'read_targets': False}
    (out / 'inference_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    ref.torch.manual_seed(0)
    ref.conversation_lib.default_conversation = ref.conversation_lib.conv_templates['llava_v1']
    tokenizer = ref.AutoTokenizer.from_pretrained(str(model_path), model_max_length=512,
                                                  padding_side='right', use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(['[SEG]', '[REF]'])
    dtype = ref.torch.bfloat16
    model = ref.LISAForCausalLM.from_pretrained(
        str(model_path), low_cpu_mem_usage=False, torch_dtype=dtype,
        vision_tower=str(vision_path), vision_pretrained=None,
        seg_token_idx=ref.get_added_token_id(tokenizer, '[SEG]'),
        ref_token_idx=ref.get_added_token_id(tokenizer, '[REF]'))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=dtype)
    model = model.bfloat16().cuda().eval()
    model.get_model().get_vision_tower().to(device=0)
    processor = ref.CLIPImageProcessor.from_pretrained(str(vision_path))
    transform = ref.ResizeLongestSide(1024)
    forwards = []
    def zero_target_hook(module, positional, kwargs):
        assert kwargs['inference']
        assert all(ref.torch.count_nonzero(m).item() == 0 for m in kwargs['masks_list'])
    hook = model.register_forward_pre_hook(zero_target_hook, with_kwargs=True)
    for condition in ['clean', 'target15_b']:
        for index, group in enumerate(groups):
            item, rgb, targets, memories, indices = ref.build_item(
                group, pairs, processor, transform, 1024, 'v1_multiround',
                ref_image_mode='crop', condition=condition, seed=0, read_targets=False)
            assert not targets.any()
            fields = list(item)
            fields[4] = ref.torch.zeros_like(item[4])
            pred = ref.predict_item(model, tuple(fields), tokenizer, dtype, indices)
            decision = decide(pred, memories)
            trials = []
            for pid, mask in zip(group['pair_ids'], pred):
                path = masks / f'{group["counterfactual_id"]}_{condition}_{pid}.npy'
                np.save(path, mask.astype(np.uint8))
                trials.append({'entity_id': pid, 'prediction': str(path), 'prediction_sha256': sha(path),
                               'miner_mask': pairs[pid]['miner_mask_path'],
                               'miner_sha256': sha(pairs[pid]['miner_mask_path'])})
            row = {'group_id': group['counterfactual_id'], 'image_id': group['image_path'],
                   'query': group['same_round2_query'], 'condition': condition,
                   'memory_source': 'supplied', 'trials': trials, **decision}
            forwards.append(row)
            with (out / 'train_scores.partial.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            if index % 25 == 0:
                print(condition, index + 1, 'groups; zero-target forward verified', flush=True)
    hook.remove()
    receipt = {'frozen_unix': time.time(), 'provenance_sha256': sha(out / 'inference_provenance.json'),
               'groups': forwards, 'model_forward_calls': len(forwards),
               'identity_predictions': sum(len(r['trials']) for r in forwards),
               'raster_reads': sorted(reads), 'zero_target_at_every_forward': True,
               'elapsed_seconds': time.time() - started,
               'peak_cuda_bytes': ref.torch.cuda.max_memory_allocated()}
    (out / 'train_scores_before_targets.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('TRAIN_PREDICTIONS_AND_SCORES_FROZEN', len(forwards), flush=True)


if __name__ == '__main__':
    main()

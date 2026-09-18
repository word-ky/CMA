"""Four native SegLLM supplied-memory trials; never reads helmet targets."""
import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from load_native import load_native
from memory_state import seed_memory, tensor_hash


def main(argv=None, model_bundle=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--clip-path', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    opened_data_paths = set()
    def record_open(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0])
            mode = arguments[1]
            if isinstance(mode, str) and 'r' in mode and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.npy', '.json'}:
                opened_data_paths.add(str(path.resolve()))
    sys.addaudithook(record_open)
    from llava.train.inference_cli import build_conversation, inference

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared = json.loads(Path(args.prepared).read_text())
    model, tokenizer, _, data_args, training_args, loading = model_bundle or load_native(
        args.checkpoint, args.clip_path, out
    )
    started = time.monotonic()
    trials, checks = [], []
    prompt = prepared['native_prompt']
    selected = re.findall(r'\[REF:(\d+)\]', prompt)[0]
    text = re.sub(r'\[REF:\d+\]', '', prompt)
    for obs in prepared['observations']:
        image_path = Path(obs['image_path'])
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        data_args.image_folder = str(image_path.parent)
        for identity in obs['identities']:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            mask = np.asarray(Image.open(identity['miner_mask_path'])) > 0
            appearance, bbox, state = seed_memory(
                rgb, mask, identity['miner_bbox_xyxy'],
                model.get_segmentator().process_images, model.get_vision_tower().image_processor,
            )
            history = {'task': 'segmentation', 'base': '[null]', 'conversations': [
                {'from': 'human', 'value': f'[IMAGE256:{image_path.name}] Segment the miner.'},
                {'from': 'gpt', 'value': f'[MASK-DECODE:{image_path.name}|INFERENCE|NULL]'},
            ]}
            inputs = build_conversation(model, tokenizer, history, 2, training_args,
                                        data_args, image_path.name, text, selected)
            indices = [int(x) for x in inputs['extra_replacement']['mask_encode_ref'][0] if x != -1]
            assert indices == [0]
            decode_turns = [i for i, turn in enumerate(history['conversations']) if '[MASK-DECODE:' in turn['value']]
            assert decode_turns == [1, 3]
            seg_positions = torch.nonzero(
                (inputs['input_ids'][0] == model.DEFAULT_SEGMENTATION_TOKEN_IDX)
                & (inputs['labels'][0] == model.DEFAULT_SEGMENTATION_TOKEN_IDX)
            ).flatten().tolist()
            assert len(seg_positions) == 2
            check = {'condition': obs['condition'], 'entity_id': identity['entity_id'],
                     'native_prompt': prompt, 'semantic_query': prepared['semantic_query'],
                     'rendered_conversation': copy.deepcopy(history), 'native_indices': indices,
                     'input_ids_sha256': tensor_hash(inputs['input_ids']), **state}
            check.update(decode_conversation_turns=decode_turns, supervised_seg_token_positions=seg_positions)
            checks.append(check)
            trials.append((inputs, image_path, appearance, bbox, check))
    assert len(trials) == 4
    for a, b in [(checks[0], checks[1]), (checks[2], checks[3])]:
        assert a['main_rgb_sha256'] == b['main_rgb_sha256']
        assert a['input_ids_sha256'] == b['input_ids_sha256']
        assert a['rendered_conversation'] == b['rendered_conversation']
        assert a['appearance_sha256'] != b['appearance_sha256']
        assert a['bbox_sha256'] != b['bbox_sha256']
    for c, d in [(checks[0], checks[2]), (checks[1], checks[3])]:
        assert c['bbox_sha256'] == d['bbox_sha256']
        assert c['appearance_sha256'] != d['appearance_sha256']
    (out / 'pre_score_checks.json').write_text(json.dumps(checks, indent=2) + '\n')

    records = []
    for index, (inputs, image_path, appearance, bbox, check) in enumerate(trials):
        captured = {}

        def check_injected(module, positional, keyword):
            slots = keyword['extra_replacement']['data'][0]
            actual_appearance = [data[0] for task, data in slots if task == 'mask-encode']
            actual_bbox = [data[0] for task, data in slots if task == 'bbox-encode']
            assert len(actual_appearance) == len(actual_bbox) == 1
            assert tensor_hash(actual_appearance[0]) == check['appearance_sha256']
            assert tensor_hash(actual_bbox[0]) == check['bbox_sha256']
            decode_slots = [data[0] for task, data in slots if task == 'mask-decode']
            assert len(decode_slots) == 2
            captured['decode_paths'] = [data['image_path'] for data in decode_slots]
            assert len(set(captured['decode_paths'])) == 1
            captured['injected_appearance_sha256'] = tensor_hash(actual_appearance[0])
            captured['injected_bbox_sha256'] = tensor_hash(actual_bbox[0])

        def capture(module, positional, output):
            losses = output['individual_losses']
            assert len(losses['segm_loss_mask']) == len(losses['mask_data']) == 2
            assert [data['image_path'] for data in losses['mask_data']] == captured['decode_paths']
            captured['output_count'] = len(losses['segm_loss_mask'])
            captured['mask'] = np.asarray(losses['segm_loss_mask'][-1]).copy()
            captured['input_size'] = list(losses['mask_data'][-1]['input_size'])

        pre_hook = model.register_forward_pre_hook(check_injected, with_kwargs=True)
        post_hook = model.register_forward_hook(capture)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            inference(model, inputs, str(image_path), 2, training_args,
                      [appearance], [bbox], [], [])
        finally:
            pre_hook.remove()
            post_hook.remove()
        torch.cuda.synchronize()
        native = captured['mask']
        assert native.ndim == 2 and np.isin(native, [0, 1]).all()
        h, w = map(int, captured['input_size'])
        original_h, original_w = prepared['original_hw']
        prediction = np.asarray(Image.fromarray(native[:h, :w].astype(np.uint8)).resize(
            (original_w, original_h), Image.Resampling.NEAREST
        ))
        assert prediction.shape == (original_h, original_w) and np.isin(prediction, [0, 1]).all()
        path = out / f'prediction_{index}.npy'
        np.save(path, prediction)
        records.append({**check, 'prediction': str(path.resolve()),
                        'original_hw': list(prediction.shape), 'native_output_hw': list(native.shape),
                        'crop_input_hw': [h, w], 'mapping': 'remove right/bottom padding; nearest binary resize once',
                        'seconds': time.monotonic() - start,
                        'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                        'mask_decode_output_count': captured['output_count'],
                        'selected_output_index': 1, 'selected_conversation_turn': 3,
                        'injected_appearance_sha256': captured['injected_appearance_sha256'],
                        'injected_bbox_sha256': captured['injected_bbox_sha256'],
                        'injected_tensor_hashes_verified': True})
        (out / 'smoke_predictions.json').write_text(json.dumps(records, indent=2) + '\n')
    receipt = {'status': 'FOUR_TRIAL_WIRING_PASSED_NOT_PERFORMANCE', 'trials': records,
               'loading': loading, 'gpu': torch.cuda.get_device_name(0),
               'smoke_seconds': time.monotonic() - started, 'gt_access': False,
               'opened_data_paths': sorted(opened_data_paths),
               'note': 'Native history placeholder may yield an unused earlier mask in the same forward; no separate predicted-miner round and no predicted output seeds memory.'}
    (out / 'smoke_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(receipt['status'])
    return receipt


if __name__ == '__main__':
    main()

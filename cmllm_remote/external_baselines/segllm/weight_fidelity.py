"""Restore only the two native gamma tensors lost by HF's legacy key rename."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

PREFIX = 'model.segmentator.hipie.detr.detr.transformer.'
GAMMAS = [PREFIX + 'encoder.vl_layers.0.b_attn.' + name
          for name in ('gamma_l', 'gamma_v')]
TARGET = PREFIX + 'tgt_embed.weight'


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def restore_and_audit(model, checkpoint, loading_info, output_dir):
    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    params = dict(model.named_parameters())
    # This is the exact observed collision, not a generic key remapping policy.
    assert set(loading_info['missing_keys']) == set(GAMMAS), loading_info
    assert set(loading_info['unexpected_keys']) == {k.replace('gamma_', 'weight_') for k in GAMMAS}, loading_info
    assert not loading_info.get('mismatched_keys'), loading_info
    assert not loading_info.get('error_msgs'), loading_info
    rows = []
    for key in GAMMAS + [TARGET]:
        with safe_open(checkpoint / index[key], framework='pt', device='cpu') as shard:
            raw = shard.get_tensor(key)
        actual = params[key]
        assert raw.shape == actual.shape, (key, raw.shape, actual.shape)
        expected = raw.to(device=actual.device, dtype=actual.dtype)
        before = digest(actual)
        if key in GAMMAS:
            with torch.no_grad():
                actual.copy_(expected)
        equal = torch.equal(actual.detach().cpu(), raw.to(actual.dtype).cpu())
        rows.append({'checkpoint_key': key, 'shard': index[key],
                     'raw_shape': list(raw.shape), 'raw_dtype': str(raw.dtype),
                     'raw_sha256': digest(raw), 'model_key': key,
                     'model_shape': list(actual.shape), 'model_dtype': str(actual.dtype),
                     'model_device': str(actual.device), 'before_sha256': before,
                     'expected_cast_sha256': digest(expected), 'actual_sha256': digest(actual),
                     'copied': key in GAMMAS, 'exact_equal': equal})
        assert equal, key
    receipt = {'status': 'WEIGHT_FIDELITY_PASSED', 'native_loading_info': loading_info,
               'copied_keys': GAMMAS, 'audited_without_assignment': [TARGET], 'tensors': rows,
               'note': 'Native outer load followed by two exact post-cast copies; no other parameter assignment.'}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'weight_fidelity_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt

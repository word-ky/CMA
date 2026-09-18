"""Exercise actual input-builder body without importing the GPU model package."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def test_target_free_builder_preserves_all_model_inputs_except_labels():
    source = Path(__file__).resolve().parents[1] / 'scripts/eval_mr_ref_counterfactual_v0.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_item')
    reads = []
    def read_mask(path, shape):
        reads.append(path)
        return np.ones(shape, dtype=np.float32)
    rgb = np.ones((8, 8, 3), dtype=np.uint8)
    env = dict(np=np, torch=torch, cv2=SimpleNamespace(imread=lambda _: rgb,
               cvtColor=lambda x, _: x, COLOR_BGR2RGB=0), group_seed=lambda *a: 0,
               condition_image=lambda x, *a: x, read_mask=read_mask,
               norm_bbox=lambda *a: [0., 0., 1., 1.],
               make_ref_image_clip=lambda *a, **k: torch.ones(3, 2, 2),
               build_multiround_conversation=lambda *a: 'fixed conversation',
               preprocess_sam=lambda x, **k: x)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), env)
    pairs = {p: dict(helmet_mask_path='helmet_'+p, miner_mask_path='miner_'+p,
                     miner_bbox_xyxy=[0, 0, 8, 8]) for p in ['a', 'b']}
    group = dict(pair_ids=['a', 'b'], image_path='image', counterfactual_id='g', same_round2_query='q')
    processor = SimpleNamespace(preprocess=lambda *a, **k: {'pixel_values': torch.ones(1, 3, 2, 2)})
    args = (group, pairs, processor, SimpleNamespace(apply_image=lambda x: x), 8, 'v1_multiround')
    legacy = env['build_item'](*args)
    assert reads == ['helmet_a', 'helmet_b', 'miner_a', 'miner_b']
    reads.clear()
    target_free = env['build_item'](*args, read_targets=False)
    assert reads == ['miner_a', 'miner_b']
    assert not target_free[2].any()
    assert target_free[4] == legacy[4] == [1, 3]
    for index, (old, new) in enumerate(zip(legacy[0], target_free[0])):
        if index == 4:
            assert torch.count_nonzero(new[[1, 3]]) == 0
            assert torch.equal(old[[0, 2]], new[[0, 2]])
        elif isinstance(old, torch.Tensor):
            assert torch.equal(old, new)
        else:
            assert old == new

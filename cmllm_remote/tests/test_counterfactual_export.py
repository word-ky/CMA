import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from counterfactual_export import TARGET15_B, condition_image, export_group, group_seed
from eval_counterfactual_memory_fidelity import evaluate_file


def test_degradation_matches_existing_project_function():
    # Execute only the existing pure function; importing its module loads LISA.
    tree = ast.parse((SCRIPTS / "stage3_rule_controller_v3_seg_local_enhance.py").read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "degrade_parametric")
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "baseline_degradation", "exec"), namespace)
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    seed = group_seed("fixed_cf_group", 7)
    degraded = condition_image(image, "target15_b", seed)
    np.testing.assert_array_equal(degraded, namespace["degrade_parametric"](image, TARGET15_B, seed))
    np.testing.assert_array_equal(degraded, condition_image(image, "target15_b", seed))
    assert not np.array_equal(degraded, condition_image(image, "target15_b", group_seed("other", 7)))
    np.testing.assert_array_equal(condition_image(image, "clean", seed), image)


def test_export_keeps_wrong_prediction_and_supplied_reference(tmp_path):
    targets = np.array([[[1, 0]], [[0, 1]]], dtype=np.uint8)
    references = targets.copy()
    swapped = targets[::-1].copy()
    cf = {"counterfactual_id": "cf_fixed", "image_path": "fixed_image.png",
          "same_round2_query": "same question", "pair_ids": ["A", "B"]}
    for condition in ("clean", "target15_b"):
        folder = tmp_path / condition
        record = export_group(folder, 0, cf, swapped, targets, references,
                              condition=condition, seed=5, provenance={"model": "synthetic"})
        manifest = folder / "memory_predictions.jsonl"
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        result = evaluate_file(manifest, folder / "metrics.json")
        assert result["summary"]["identity_error_rate"] == 1
        assert result["summary"]["mean_identity_margin"] == -1
        assert record["memory_source"] == "supplied_ref"
        np.testing.assert_array_equal(np.load(folder / record["trials"][0]["prediction"]), swapped[0])
        np.testing.assert_array_equal(np.load(folder / record["trials"][0]["reference"]), references[0])
    with pytest.raises(ValueError, match="counts"):
        export_group(tmp_path, 1, cf, swapped[:1], targets, references,
                     condition="clean", seed=5, provenance={})


@pytest.mark.parametrize("mode", ["v0_single_ref", "v1_multiround"])
def test_frozen_build_item_uses_same_condition_for_image_and_ref(tmp_path, monkeypatch, mode):
    # Exercise the real build_item body on CPU with lightweight encoder/conversation
    # substitutes. This tests wiring, not model predictions or GPU performance.
    tree = ast.parse((SCRIPTS / "eval_mr_ref_counterfactual_v0.py").read_text(encoding="utf-8"))
    import torch
    from types import SimpleNamespace
    # OpenCV on Windows cannot open this machine's Unicode absolute temp paths.
    monkeypatch.chdir(tmp_path)

    names = {"build_item", "read_mask", "norm_bbox"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    captured = []

    def ref_encoder(image, *args, **kwargs):
        captured.append(image.copy())
        return torch.zeros(3, 224, 224)

    namespace = {"cv2": cv2, "np": np, "torch": torch,
                 "condition_image": condition_image, "group_seed": group_seed,
                 "make_ref_image_clip": ref_encoder,
                 "build_conversation": lambda query: query,
                 "build_multiround_conversation": lambda first, second: first + second,
                 "preprocess_sam": lambda x, **kwargs: x}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "build_item", "exec"), namespace)
    image = np.full((8, 8, 3), 180, dtype=np.uint8)
    cv2.imwrite("image.png", image)
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, :, :4], masks[1, :, 4:] = 255, 255
    pairs = {}
    for i in range(2):
        path = f"mask{i}.png"
        cv2.imwrite(path, masks[i])
        pairs[str(i)] = {"miner_mask_path": path, "helmet_mask_path": path,
                         "miner_bbox_xyxy": [i * 4, 0, (i + 1) * 4, 8]}
    cf = {"pair_ids": ["0", "1"], "image_path": "image.png",
          "counterfactual_id": "synthetic", "same_round2_query": "same"}
    clip = SimpleNamespace(preprocess=lambda image, **kwargs: {"pixel_values": torch.zeros(1, 3, 224, 224)})
    transform = SimpleNamespace(apply_image=lambda image: image)
    outputs = {}
    for condition in ("clean", "target15_b"):
        captured.clear()
        item, observed, gt, refs, indices = namespace["build_item"](
            cf, pairs, clip, transform, 8, mode, condition=condition, seed=7)
        expected = condition_image(image, condition, group_seed("synthetic", 7))
        np.testing.assert_array_equal(observed, expected)
        for ref_image in captured:
            np.testing.assert_array_equal(ref_image, expected)
        np.testing.assert_array_equal(item[1].numpy().transpose(1, 2, 0), expected)
        outputs[condition] = (gt, refs)
        assert indices == ([0, 1] if mode == "v0_single_ref" else [1, 3])
    for clean, degraded in zip(outputs["clean"], outputs["target15_b"]):
        np.testing.assert_array_equal(clean, degraded)

    # Enhancement sees one degraded observation, not identity masks or GT,
    # and its one output feeds both main and every REF crop.
    enhancer_inputs = []
    def enhance(image):
        enhancer_inputs.append(image.copy())
        return np.full_like(image, 99)

    captured.clear()
    item, observed, gt, refs, _ = namespace["build_item"](
        cf, pairs, clip, transform, 8, mode, condition="target15_b", seed=7,
        enhance_image=enhance)
    assert len(enhancer_inputs) == 1
    np.testing.assert_array_equal(enhancer_inputs[0], condition_image(image, "target15_b", group_seed("synthetic", 7)))
    np.testing.assert_array_equal(observed, np.full_like(image, 99))
    np.testing.assert_array_equal(item[1].numpy().transpose(1, 2, 0), observed)
    assert len(captured) == 2
    for seen in captured:
        np.testing.assert_array_equal(seen, observed)
    np.testing.assert_array_equal(gt, outputs["clean"][0])
    np.testing.assert_array_equal(refs, outputs["clean"][1])

    # All four cells: main tensors follow main condition, REF encoder input
    # follows REF condition. Diagonal outputs equal the legacy interface.
    for main in ("clean", "target15_b"):
        for ref in ("clean", "target15_b"):
            captured.clear()
            item, observed, gt, refs, indices = namespace["build_item"](
                cf, pairs, clip, transform, 8, mode, seed=7,
                main_condition=main, ref_condition=ref)
            expected_main = condition_image(image, main, group_seed("synthetic", 7))
            expected_ref = condition_image(image, ref, group_seed("synthetic", 7))
            np.testing.assert_array_equal(observed, expected_main)
            np.testing.assert_array_equal(item[1].numpy().transpose(1, 2, 0), expected_main)
            for seen in captured:
                np.testing.assert_array_equal(seen, expected_ref)
            np.testing.assert_array_equal(gt, outputs["clean"][0])
            np.testing.assert_array_equal(refs, outputs["clean"][1])
            if main == ref:
                legacy = namespace["build_item"](cf, pairs, clip, transform, 8, mode, condition=main, seed=7)[0]
                for new_value, old_value in zip(item, legacy):
                    if isinstance(new_value, torch.Tensor):
                        assert torch.equal(new_value, old_value)
                    else:
                        assert new_value == old_value


def test_cross_condition_export_metadata(tmp_path):
    masks = np.array([[[1, 0]], [[0, 1]]])
    cf = {"counterfactual_id": "fixed", "image_path": "x.png",
          "same_round2_query": "same", "pair_ids": ["A", "B"]}
    row = export_group(tmp_path, 0, cf, masks, masks, masks,
                       condition="main_clean__ref_target15_b", main_condition="clean",
                       ref_condition="target15_b", seed=0, provenance={})
    assert row["main_condition"] == "clean" and row["ref_condition"] == "target15_b"
    assert row["factor_degradation_configs"] == {"main": None, "ref": TARGET15_B}

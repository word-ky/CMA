import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_counterfactual_memory_fidelity import evaluate_file, score_masks, summarize


def targets(n=2):
    return np.eye(n, dtype=np.uint8).reshape(n, 1, n)


def test_correct_swap_and_memory_ignored():
    gt = targets()
    correct = score_masks(gt, gt)
    assert correct["iou_matrix"] == [[1, 0], [0, 1]]
    assert correct["identity_margin"] == [1, 1]
    assert (correct["memory_fidelity"], correct["cmsa"], correct["identity_error_rate"]) == (1, 1, 0)
    swapped = score_masks(gt[::-1], gt)
    assert swapped["identity_margin"] == [-1, -1]
    assert (swapped["memory_fidelity"], swapped["cmsa"], swapped["identity_error_rate"]) == (0, 0, 1)
    ignored = score_masks(np.repeat(gt[:1], 2, axis=0), gt)
    assert (ignored["memory_fidelity"], ignored["cmsa"], ignored["identity_error_rate"]) == (0.5, 0, 0.5)


@pytest.mark.parametrize("pred", [np.zeros((2, 1, 2)), np.ones((2, 1, 2))])
def test_empty_and_union_ties_do_not_pass(pred):
    result = score_masks(pred, targets())
    assert result["memory_fidelity"] == result["cmsa"] == result["identity_error_rate"] == 0


def test_all_wrong_identities_not_only_next_neighbor():
    gt = targets(3)
    result = score_masks(gt[[2, 1, 2]], gt)
    assert result["fidelity"] == [False, True, True]
    assert result["identity_error"] == [True, False, False]
    assert result["cmsa"] == pytest.approx(1 / 3)


def test_threshold_qualifies_switch_but_not_rank():
    gt = np.array([[[1, 1, 0, 0]], [[0, 0, 1, 1]]])
    pred = np.array([[[1, 0, 0, 0]], [[0, 0, 1, 0]]])
    assert score_masks(pred, gt, min_iou=0.5)["cmsa"] == 1
    result = score_masks(pred, gt, min_iou=0.6)
    assert result["memory_fidelity"] == 1
    assert result["cmsa"] == 0
    assert score_masks(pred[::-1], gt, min_iou=0.6)["identity_error_rate"] == 0


def test_permutation_invariance_and_weighted_aggregation():
    gt = targets(3)
    pred = gt[[2, 1, 2]]
    first = score_masks(pred, gt)
    order = [2, 0, 1]
    permuted = score_masks(pred[order], gt[order])
    for key in ("target_miou", "memory_fidelity", "cmsa", "identity_error_rate"):
        assert first[key] == permuted[key]
    total = summarize([score_masks(targets(), targets()), first])
    assert total["memory_fidelity"] == pytest.approx(4 / 5)
    assert total["cmsa"] == pytest.approx(2 / 4)
    assert total["mean_identity_margin"] == pytest.approx(3 / 5)
    assert total["median_identity_margin"] == 1


@pytest.mark.parametrize("gt", [np.zeros((2, 1, 2)), np.ones((2, 1, 2))])
def test_invalid_identity_targets_rejected(gt):
    with pytest.raises(ValueError):
        score_masks(gt, gt)


def test_logits_and_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="binary"):
        score_masks(targets() * 0.8, targets())
    with pytest.raises(ValueError, match="matching"):
        score_masks(np.ones((2, 2, 2)), targets())


def test_saved_png_npy_inline_cli_and_strata(tmp_path):
    gt = targets()
    Image.fromarray(gt[0] * 255).save(tmp_path / "a.png")
    np.save(tmp_path / "b.npy", gt[1])
    record = {"group_id": "synthetic_pair", "image_id": "synthetic", "query": "same query",
              "condition": "clean", "memory_source": "synthetic",
              "trials": [{"entity_id": "A", "prediction": "a.png", "target": gt[0].tolist()},
                         {"entity_id": "B", "prediction": "b.npy", "target": gt[1].tolist()}]}
    manifest = tmp_path / "groups.jsonl"
    second = {**record, "condition": "synthetic_empty", "trials": [
        {**t, "prediction": [[0, 0]]} for t in record["trials"]]}
    manifest.write_text(json.dumps(record) + "\n" + json.dumps(second), encoding="utf-8")
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts/eval_counterfactual_memory_fidelity.py"
    subprocess.run([sys.executable, "-B", str(script), "--input", str(manifest), "--output", str(output)], check=True)
    report = json.loads(output.read_text())
    assert report["summary"]["cmsa"] == 0.5
    assert len(report["by_condition_and_memory_source"]) == 2
    assert report["groups"][0]["iou_matrix"] == [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_file(manifest, manifest)

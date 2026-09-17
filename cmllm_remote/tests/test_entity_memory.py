import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from entity_memory import EntityMemoryStore, from_anchor_state


def test_write_read_are_isolated_from_mutations():
    mask = np.array([[0, 1], [0, 0]])
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    provenance = {"source_action": "SEG_ANCHOR", "step": 1, "image_state": "clean"}
    store = EntityMemoryStore.write("miner_A", mask, image, provenance)
    mask[:] = 0
    image[:] = 255
    provenance["step"] = 100
    read = store.read()
    assert read.bbox == (1, 0, 2, 1)
    assert read.mask.sum() == 1
    assert read.image.sum() == 0
    assert read.provenance["step"] == 1
    read.mask[:] = 0
    read.provenance["step"] = 200
    assert store.read().mask.sum() == 1
    assert store.read().provenance["step"] == 1


def test_update_invalidates_features_and_rollback_restores_whole_snapshot():
    a = np.array([[1, 0]])
    b = np.array([[0, 1]])
    store = EntityMemoryStore.write("A", a, "image_v1", {"step": 1},
                                    appearance_feature=[1, 2], semantic_feature="miner", reliability=0.8)
    candidate = store.update_candidate(b, "image_v2", {"step": 2})
    assert candidate.entity_id == "A"
    assert candidate.version == 2
    assert candidate.appearance_feature is candidate.semantic_feature is candidate.reliability is None
    restored = store.rollback(step=3)
    assert restored.version == 3
    assert np.array_equal(restored.mask, a)
    assert restored.image == "image_v1"
    assert restored.bbox == (0, 0, 1, 1)
    assert restored.appearance_feature == [1, 2]
    assert restored.semantic_feature == "miner" and restored.reliability == 0.8
    assert restored.provenance["restored_version"] == 1
    assert store.read(2).image == "image_v2"  # update history not erased
    assert store.update_candidate(b, "image_v4", {"step": 4}).version == 4
    assert store.rollback(step=5, version=1).version == 5


def test_anchor_adapter_copies_image_and_excludes_gt_quality():
    state = {"anchor": np.array([[0, 1]]), "anchor_image": np.zeros((1, 2, 3)),
             "anchor_state": "enhanced", "anchor_quality": {"iou": 0.99},
             "current_image": np.ones((1, 2, 3))}
    store = from_anchor_state(state, entity_id="A", source_action="SEG_ANCHOR", step=9)
    memory = store.read()
    assert memory.reliability is None
    assert memory.image.sum() == 0
    assert memory.provenance["selection_uses_gt"] is True
    assert "iou" not in memory.provenance and "quality" not in memory.provenance
    state["anchor"][:] = 0
    assert store.read().mask.sum() == 1


def test_empty_candidate_can_be_recorded_but_has_no_bbox_or_reliability():
    store = EntityMemoryStore.write("A", np.zeros((2, 2)), None, {"step": 0})
    assert store.read().bbox is None and store.read().reliability is None
    with pytest.raises(ValueError, match="prior"):
        store.rollback(step=1)
    with pytest.raises(ValueError, match="No anchor"):
        from_anchor_state({}, entity_id="A", source_action="SEG_ANCHOR", step=0)

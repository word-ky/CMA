"""Selection integrity tests; no model-performance assertions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_cycle006_split import select_groups
from build_mr_ref_counterfactual_train import split_bucket


def test_split_is_order_independent_and_excludes_image_aliases():
    train_paths = [str(i) for i in range(100) if split_bucket(str(i)) < 8]
    held_path = next(str(i) for i in range(100) if split_bucket(str(i)) >= 8)
    rows = [{"counterfactual_id": f"cf{i}", "image_path": path, "image_member": f"member{i}"}
            for i, path in enumerate(train_paths[:8])]
    rows.append({"counterfactual_id": "held", "image_path": held_path, "image_member": "held_member"})
    hashes = {r["image_member"]: r["image_member"] for r in rows}
    # A training-path image is byte-identical to a holdout, another pair are aliases.
    hashes["member0"] = hashes["held_member"]
    hashes["member1"] = hashes["member2"]
    a = select_groups(rows, hashes, train_count=3, val_count=2)
    b = select_groups(list(reversed(rows)), hashes, train_count=3, val_count=2)
    assert a == b
    train, val, held = a
    th = {hashes[r["image_member"]] for r in train}
    vh = {hashes[r["image_member"]] for r in val}
    assert len(th) == 3 and len(vh) == 2
    assert not (th & vh or th & held or vh & held)
    assert all(split_bucket(r["image_path"]) < 8 for r in train + val)

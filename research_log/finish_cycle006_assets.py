"""Map the frozen train/validation selections to recovered assets; no training."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "code/cmllm/scripts"))
from build_mr_ref_counterfactual_train import build_counterfactual_episode, read_jsonl, write_jsonl

log = root / "research_log/cycle006"
data = root / "shared/data/cycle006"
selected = json.loads((log / "split_receipt.json").read_text())
recovered = json.loads((log / "frozen_assets.json").read_text())
groups = {r["counterfactual_id"]: r for r in read_jsonl(data / "counterfactual_selected_restored.jsonl")}
pairs = {r["pair_id"]: r for r in read_jsonl(data / "helmet_miner_pairs_restored.jsonl")}

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

for asset in recovered["assets"]:
    assert sha(asset["path"]) == asset["sha256"]

manifests = {}
for name in ("train", "val"):
    originals = read_jsonl(log / f"{name}_grouped_original_paths.jsonl")
    raw = [groups[r["counterfactual_id"]] for r in originals]
    grouped = [build_counterfactual_episode(r, pairs) for r in raw]
    for original, restored, receipt in zip(originals, grouped, selected["selected"][name]):
        for key in ("episode_id", "counterfactual_id", "pair_ids", "same_round2_query"):
            assert original[key] == restored[key]
        assert sha(restored["image_path"]) == receipt["image_sha256"]
        for a, b in zip(original["pairs"], restored["pairs"]):
            assert a["miner_bbox_xyxy"] == b["miner_bbox_xyxy"]
    for suffix, rows in (("grouped_restored", grouped), ("eval_restored", raw)):
        path = data / f"{name}_{suffix}.jsonl"
        write_jsonl(path, rows)
        manifests[f"{name}_{suffix}"] = {"path": str(path), "sha256": sha(path), "groups": len(rows)}
        # Keep the compact runnable manifest in the project evidence too.
        write_jsonl(log / path.name, rows)

result = {"status": "assets_ready_training_blocked_missing_historical_v1_teacher",
          "train_groups": 300, "validation_groups": 50,
          "frozen_selection_sha256": sha(log / "split_receipt.json"),
          "verified_assets": len(recovered["assets"]), "preserved_identity_query_bbox_and_image_hash": True,
          "regenerated_masks_not_exact_historical": True, "manifests": manifests,
          "training_runs": 0, "diagnostic30_inference_runs": 0}
(log / "assets_ready.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

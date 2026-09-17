"""Read-only asset audit + fixed first-30 holdout selection (no result filtering)."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
accepted = root / "shared/data/final_accepted_v1"
out = root / "research_log/cycle002"
out.mkdir(parents=True, exist_ok=True)
cf_path = accepted / "episodes_counterfactual_clean_val.jsonl"
pair_path = accepted / "helmet_miner_pairs_accept_high.jsonl"
rows = [json.loads(line) for line in cf_path.read_text().splitlines() if line.strip()]
holdout = [r for r in rows if int(hashlib.md5(r["image_path"].encode()).hexdigest()[:8], 16) % 10 in (8, 9)]
selected = holdout[:30]
pairs = {r["pair_id"]: r for r in map(json.loads, pair_path.read_text().splitlines())}
selected_path = out / "counterfactual_holdout_first30.jsonl"
selected_path.write_text("".join(json.dumps(r) + "\n" for r in selected))
missing = []
group_paths = {}
for group in selected:
    paths = [("image", group["image_path"])]
    for pair_id in group["pair_ids"]:
        pair = pairs[pair_id]
        paths += [("miner_mask", pair["miner_mask_path"]), ("helmet_mask", pair["helmet_mask_path"])]
    for kind, path in paths:
        if not Path(path).is_file():
            missing.append({"group_id": group["counterfactual_id"], "kind": kind, "path": path})
    group_paths[group["counterfactual_id"]] = [path for _, path in paths]
available_basenames = {p.name for p in (root / "shared/data").rglob("*") if p.is_file()}
report = {
    "selection": "first 30 in archived file order after MD5(image_path) bucket 8/9; no result filtering",
    "archived_groups": len(rows), "holdout_groups": len(holdout), "selected_groups": len(selected),
    "selected_ids": [r["counterfactual_id"] for r in selected],
    "source_manifest_sha256": hashlib.sha256(cf_path.read_bytes()).hexdigest(),
    "selected_manifest_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
    "missing_references": len(missing),
    "missing_by_kind": {k: sum(r["kind"] == k for r in missing) for k in ("image", "miner_mask", "helmet_mask")},
    "matching_missing_basenames_under_shared_data": sorted({Path(r["path"]).name for r in missing} & available_basenames),
    "groups_with_all_paths_or_basename_candidates": sum(
        all(Path(p).is_file() or Path(p).name in available_basenames for p in paths)
        for paths in group_paths.values()),
    "model_config_present": (root / "shared/models/cmllm_lisa_plus_mr_ref_v2b_rank_long2ep_w15_lr5e6_merged_hf/config.json").is_file(),
    "runtime_present": (root / ".venv/bin/python").is_file(),
    "missing": missing,
}
(out / "asset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("missing", "selected_ids")}, indent=2))

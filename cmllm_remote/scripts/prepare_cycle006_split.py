"""Reconstruct the historical grouped manifest and freeze an image-disjoint split.

No model inference or outcome filtering. Original paths stay unchanged until
asset recovery; this produces selection provenance, not runnable restored data.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from build_mr_ref_counterfactual_train import build_counterfactual_episode, read_jsonl, split_bucket, write_jsonl


def select_groups(rows, image_hashes, seed=20260528, train_count=300, val_count=50):
    held_hashes = {image_hashes[r["image_member"]] for r in rows if split_bucket(r["image_path"]) >= 8}
    train = [r for r in rows if split_bucket(r["image_path"]) < 8]
    ordered = sorted(train, key=lambda r: (hashlib.sha256(f"{seed}:{r['counterfactual_id']}".encode()).hexdigest(), r["counterfactual_id"]))
    seen = set(held_hashes)
    selected = []
    for row in ordered:
        digest = image_hashes[row["image_member"]]
        if digest in seen:
            continue
        selected.append(row)
        seen.add(digest)
        if len(selected) == train_count + val_count:
            break
    if len(selected) < train_count + val_count:
        raise ValueError("Not enough image-disjoint training groups for the fixed split")
    return selected[:train_count], selected[train_count:], held_hashes


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    source = root / "shared/data/final_accepted_v1"
    out = root / "research_log/cycle006"
    out.mkdir(parents=True, exist_ok=True)
    cf_path = source / "episodes_counterfactual_clean_val.jsonl"
    pairs_path = source / "helmet_miner_pairs_accept_high.jsonl"
    rows = read_jsonl(cf_path)
    pairs = {r["pair_id"]: r for r in read_jsonl(pairs_path)}
    grouped = [build_counterfactual_episode(r, pairs) for r in rows if split_bucket(r["image_path"]) < 8]
    assert all(grouped)
    grouped_path = root / "shared/data/cycle006/episodes_counterfactual_ref_train_grouped.jsonl"
    write_jsonl(grouped_path, grouped)
    by_id = {r["counterfactual_id"]: r for r in grouped}
    archive = root / "shared/source/mining_helmet.zip"
    with zipfile.ZipFile(archive) as zf:
        image_hashes = {member: hashlib.sha256(zf.read(member)).hexdigest() for member in sorted({r["image_member"] for r in rows})}
    train, val, held = select_groups(rows, image_hashes)
    write_jsonl(out / "selected_source_groups.jsonl", train + val)
    selected_receipts = {}
    manifests = {}
    missing = {"image": set(), "miner_mask": set(), "helmet_mask": set()}
    for name, selected in (("train", train), ("val", val)):
        path = out / f"{name}_grouped_original_paths.jsonl"
        write_jsonl(path, [by_id[r["counterfactual_id"]] for r in selected])
        manifests[name] = {"path": str(path), "sha256": sha(path)}
        selected_receipts[name] = [{"group_id": r["counterfactual_id"], "image_member": r["image_member"],
                                   "image_sha256": image_hashes[r["image_member"]],
                                   "original_bucket": split_bucket(r["image_path"])} for r in selected]
        for r in selected:
            if not Path(r["image_path"]).is_file():
                missing["image"].add(r["image_path"])
            for pair in by_id[r["counterfactual_id"]]["pairs"]:
                for kind in ("miner_mask", "helmet_mask"):
                    if not Path(pair[kind + "_path"]).is_file():
                        missing[kind].add(pair[kind + "_path"])
    report = {"status": "split_frozen_assets_not_restored_training_not_started",
              "seed": 20260528, "order": "sha256(seed:counterfactual_id), then counterfactual_id",
              "selection": "first one group per unique image-byte SHA256; exclude all holdout-bucket image hashes; first300 train, next50 validation",
              "sources": {str(p): sha(p) for p in (cf_path, pairs_path, grouped_path)},
              "grouped_manifest": "deterministically reconstructed by the unchanged historical builder, not exact recovered file bytes",
              "original_groups": len(rows), "original_training_groups": len(grouped),
              "excluded_holdout_image_hashes": len(held), "train_groups": len(train), "val_groups": len(val),
              "train_val_and_all_holdout_images_disjoint_by_bytes": True,
              "manifests": manifests, "selected": selected_receipts,
              "missing_original_asset_counts": {k: len(v) for k, v in missing.items()}}
    (out / "split_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "missing_assets.json").write_text(json.dumps({k: sorted(v) for k, v in missing.items()}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "selected"}, indent=2))


if __name__ == "__main__":
    main()

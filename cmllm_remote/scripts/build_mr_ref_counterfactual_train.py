#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ROUND1_QUERY = "Segment the miner whose helmet should be segmented next."
DEFAULT_ROUND2_QUERY = (
    "Based on the miner mask from the previous round, segment only the mining helmet worn by that miner."
)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_bucket(key):
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10


def pair_to_training_pair(pair, round2_query):
    return {
        "pair_id": pair["pair_id"],
        "miner_id": pair.get("pseudo_miner_id"),
        "helmet_id": pair.get("helmet_id"),
        "miner_mask_path": pair["miner_mask_path"],
        "helmet_mask_path": pair["helmet_mask_path"],
        "miner_bbox_xyxy": pair["miner_bbox_xyxy"],
        "helmet_bbox_xyxy": pair["helmet_bbox_xyxy"],
        "pair_score": pair.get("pair_score", 1.0),
        "round1_query": DEFAULT_ROUND1_QUERY,
        "round2_query": round2_query or DEFAULT_ROUND2_QUERY,
        "miner_annotation_source": pair.get("miner_annotation_source"),
        "helmet_target_mask_source": pair.get("helmet_target_mask_source"),
    }


def build_counterfactual_episode(cf, pairs_by_id):
    pair_ids = cf["pair_ids"]
    pairs = [pairs_by_id[pair_id] for pair_id in pair_ids if pair_id in pairs_by_id]
    if len(pairs) < 2:
        return None
    round2_query = cf.get("same_round2_query") or DEFAULT_ROUND2_QUERY
    return {
        "episode_id": "cftrain_" + cf["counterfactual_id"],
        "episode_type": "counterfactual_ref_train",
        "counterfactual_id": cf["counterfactual_id"],
        "image_path": cf["image_path"],
        "pair_ids": pair_ids,
        "same_round2_query": round2_query,
        "cf_score": cf.get("cf_score"),
        "pairs": [pair_to_training_pair(pair, round2_query) for pair in pairs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regular-train", required=True)
    parser.add_argument("--counterfactual-jsonl", required=True)
    parser.add_argument("--pairs-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-buckets", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_buckets = {int(x) for x in args.train_buckets.split(",") if x != ""}
    pairs_by_id = {row["pair_id"]: row for row in read_jsonl(args.pairs_jsonl)}
    regular_rows = read_jsonl(args.regular_train)
    cf_rows = read_jsonl(args.counterfactual_jsonl)

    regular_train = []
    regular_holdout = []
    for row in regular_rows:
        bucket = split_bucket(row["image_path"])
        if bucket in train_buckets:
            regular_train.append(row)
        else:
            regular_holdout.append(row)

    cf_train = []
    cf_holdout_raw = []
    cf_holdout_grouped = []
    skipped = 0
    for cf in cf_rows:
        grouped = build_counterfactual_episode(cf, pairs_by_id)
        if grouped is None:
            skipped += 1
            continue
        bucket = split_bucket(cf["image_path"])
        if bucket in train_buckets:
            cf_train.append(grouped)
        else:
            cf_holdout_raw.append(cf)
            cf_holdout_grouped.append(grouped)

    combined_train = regular_train + cf_train
    write_jsonl(out_dir / "episodes_miner_to_helmet_plus_counterfactual_train.jsonl", combined_train)
    write_jsonl(out_dir / "episodes_miner_to_helmet_regular_holdout.jsonl", regular_holdout)
    write_jsonl(out_dir / "episodes_counterfactual_ref_train_grouped.jsonl", cf_train)
    write_jsonl(out_dir / "episodes_counterfactual_ref_holdout_grouped.jsonl", cf_holdout_grouped)
    write_jsonl(out_dir / "episodes_counterfactual_ref_holdout_eval.jsonl", cf_holdout_raw)

    stats = {
        "regular_train_in": len(regular_rows),
        "regular_train_out": len(regular_train),
        "regular_holdout": len(regular_holdout),
        "counterfactual_in": len(cf_rows),
        "counterfactual_grouped_train": len(cf_train),
        "counterfactual_holdout_eval": len(cf_holdout_raw),
        "combined_train": len(combined_train),
        "skipped_counterfactual": skipped,
        "train_buckets": sorted(train_buckets),
    }
    (out_dir / "counterfactual_train_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

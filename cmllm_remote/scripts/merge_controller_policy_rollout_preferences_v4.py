#!/usr/bin/env python3
"""Merge rollout-preference v4 shards."""

import argparse
import json
from collections import Counter
from pathlib import Path

from stage3_rule_controller_v3_seg_local_enhance import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    prefs = []
    stats = {
        "schema": "controller_policy_rollout_preferences_v4_merged",
        "shard_dirs": args.shard_dir,
        "num_shards": len(args.shard_dir),
        "num_candidate_states": 0,
        "num_candidate_rollouts": 0,
        "num_preferences": 0,
        "best_action_counts": Counter(),
        "candidate_action_counts": Counter(),
    }

    for shard in args.shard_dir:
        shard = Path(shard)
        cand_path = shard / "rollout_candidates_v4.jsonl"
        pref_path = shard / "action_preference_pairs_rollout_all_v4.jsonl"
        stat_path = shard / "rollout_pref_stats.json"
        if cand_path.exists():
            candidates.extend(read_jsonl(cand_path))
        if pref_path.exists():
            prefs.extend(read_jsonl(pref_path))
        if stat_path.exists():
            s = json.loads(stat_path.read_text(encoding="utf-8"))
            stats["num_candidate_rollouts"] += int(s.get("num_candidate_rollouts", 0))
            stats["candidate_action_counts"].update(s.get("candidate_action_counts", {}))
            stats["best_action_counts"].update(s.get("best_action_counts", {}))

    stats["num_candidate_states"] = len(candidates)
    stats["num_preferences"] = len(prefs)
    split = max(1, int((1.0 - args.val_ratio) * len(prefs)))
    train = prefs[:split]
    val = prefs[split:]
    stats["num_train_preferences"] = len(train)
    stats["num_val_preferences"] = len(val)
    stats["candidate_action_counts"] = dict(stats["candidate_action_counts"])
    stats["best_action_counts"] = dict(stats["best_action_counts"])

    write_jsonl(out_dir / "rollout_candidates_v4.jsonl", candidates)
    write_jsonl(out_dir / "action_preference_pairs_rollout_all_v4.jsonl", prefs)
    write_jsonl(out_dir / "action_preference_pairs_rollout_train_v4.jsonl", train)
    write_jsonl(out_dir / "action_preference_pairs_rollout_val_v4.jsonl", val)
    (out_dir / "rollout_pref_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

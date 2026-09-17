#!/usr/bin/env python3
"""Build policy-controller SFT and preference data from Stage-3 traces.

The policy input intentionally excludes GT IoU and oracle improvement fields.
Oracle scores are kept only in the trajectory/reward metadata and preference
labels, where they are allowed because they supervise training but are not
available to the policy at inference time.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_ACTIONS = [
    "SEG_TARGET_DIRECT",
    "GLOBAL_ENHANCE",
    "GLOBAL_ENHANCE_STRONG",
    "LOCAL_ENHANCE_TARGET",
    "SEG_TARGET_LOCAL",
    "SEG_ANCHOR",
    "LOCAL_ENHANCE_ANCHOR",
    "SEG_ANCHOR_LOCAL",
    "BLACKOUT_WITH_ANCHOR",
    "SEG_TARGET_WITH_REF",
    "LOCAL_ENHANCE_REF_TARGET",
    "SEG_REF_TARGET_LOCAL",
    "ROLLBACK_LOCAL_RESULT",
    "ROLLBACK_TO_BEST_DIRECT",
    "STOP_SUCCESS",
    "STOP_FAIL",
]


SEGMENT_ACTIONS = {
    "segment_target_direct",
    "segment_target_direct_local_enhance",
    "segment_anchor",
    "segment_anchor_local_enhance",
    "segment_helmet",
    "segment_helmet_local_enhance",
    "restore_direct_target_result",
}


ORACLE_KEYS = {
    "iou",
    "improvement_over_degraded",
    "improvement_over_enhanced",
    "improvement_over_best_direct",
    "improvement_over_current_best",
    "improvement_over_focus",
    "improvement_over_failed_focus",
}


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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def round_float(value, ndigits=6):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return round(value, ndigits)
    return value


def sanitize_obj(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ORACLE_KEYS:
                continue
            out[k] = sanitize_obj(v)
        return out
    if isinstance(obj, list):
        return [sanitize_obj(x) for x in obj]
    return round_float(obj)


def sanitize_quality(quality):
    if not isinstance(quality, dict):
        return None
    out = {}
    for key, value in quality.items():
        if key in ORACLE_KEYS:
            continue
        out[key] = sanitize_obj(value)
    if "area_ratio" in out:
        out["area_ratio_bucket"] = bucket(out["area_ratio"], [0.0005, 0.002, 0.01, 0.05, 0.15])
    if "connected_components" in out:
        out["fragmentation_bucket"] = bucket(out["connected_components"], [1, 3, 8, 20])
    return out


def bucket(value, thresholds):
    try:
        value = float(value)
    except Exception:
        return "unknown"
    for i, th in enumerate(thresholds):
        if value <= th:
            return f"b{i}"
    return f"b{len(thresholds)}"


def canonical_action(step):
    action = step.get("action", "")
    if action == "segment_target_direct":
        return "SEG_TARGET_DIRECT"
    if action == "enhance_lowlight":
        return "GLOBAL_ENHANCE"
    if action == "enhance_lowlight_strong":
        return "GLOBAL_ENHANCE_STRONG"
    if action == "local_enhance_target_by_seg":
        return "LOCAL_ENHANCE_TARGET"
    if action == "segment_target_direct_local_enhance":
        return "SEG_TARGET_LOCAL"
    if action == "find_anchor":
        return "SEG_ANCHOR"
    if action == "segment_anchor":
        return "SEG_ANCHOR"
    if action == "local_enhance_anchor_by_seg":
        return "LOCAL_ENHANCE_ANCHOR"
    if action == "segment_anchor_local_enhance":
        return "SEG_ANCHOR_LOCAL"
    if action == "focus_blackout":
        return "BLACKOUT_WITH_ANCHOR"
    if action == "segment_helmet":
        return "SEG_TARGET_WITH_REF"
    if action == "local_enhance_target_by_seg_after_ref":
        return "LOCAL_ENHANCE_REF_TARGET"
    if action == "segment_helmet_local_enhance":
        return "SEG_REF_TARGET_LOCAL"
    if action.startswith("rollback_local"):
        return "ROLLBACK_LOCAL_RESULT"
    if action == "rollback_to_accepted_ref":
        return "ROLLBACK_TO_BEST_DIRECT"
    if action == "restore_direct_target_result":
        return "ROLLBACK_TO_BEST_DIRECT"
    if action == "stop_failed_anchor":
        return "STOP_FAIL"
    return action.upper()


def action_label(step):
    label = {
        "action": canonical_action(step),
        "source_action": step.get("action"),
    }
    for key in [
        "target",
        "image_state",
        "anchor_category",
        "ref",
        "condition",
        "roi_scale",
        "rollback_target",
    ]:
        if key in step:
            label[key] = sanitize_obj(step[key])
    if "roi_xyxy" in step:
        label["roi_xyxy"] = sanitize_obj(step["roi_xyxy"])
    return label


def observable_step(step):
    out = {
        "step": step.get("step"),
        "action": canonical_action(step),
        "source_action": step.get("action"),
        "image_state": step.get("image_state"),
    }
    # Do not include free-form "reason" fields because several rule-controller
    # reasons encode oracle-derived failure decisions.
    for key in ["target", "anchor_category", "ref", "condition", "roi_scale", "rollback_target"]:
        if key in step:
            out[key] = sanitize_obj(step[key])
    if "quality" in step:
        out["mask_quality"] = sanitize_quality(step["quality"])
    if "candidate_stats" in step:
        out["candidate_mask_quality"] = sanitize_quality(step["candidate_stats"])
    if "image_quality" in step:
        out["image_quality"] = sanitize_obj(step["image_quality"])
    if "roi_xyxy" in step:
        out["roi_xyxy"] = sanitize_obj(step["roi_xyxy"])
    # Do not copy step["decision"]; for segmentation/local candidates it is
    # currently oracle-derived from GT IoU in the rule controller.
    return out


def infer_flags(prev_steps):
    flags = {
        "has_target_candidate": False,
        "has_anchor_ref": False,
        "has_blackout_state": False,
        "has_local_target_state": False,
        "has_rollback_available": False,
    }
    last_mask = None
    last_image_quality = None
    for step in prev_steps:
        action = step.get("action")
        if action in {"segment_target_direct", "segment_target_direct_local_enhance", "segment_helmet", "segment_helmet_local_enhance", "restore_direct_target_result"}:
            flags["has_target_candidate"] = True
            last_mask = sanitize_quality(step.get("quality", {}))
        if action in {"segment_anchor", "segment_anchor_local_enhance"}:
            flags["has_anchor_ref"] = True
            flags["has_rollback_available"] = True
            last_mask = sanitize_quality(step.get("quality", {}))
        if action == "focus_blackout":
            flags["has_blackout_state"] = True
        if "local_enhance" in action:
            flags["has_local_target_state"] = True
        if "image_quality" in step:
            last_image_quality = sanitize_obj(step["image_quality"])
    flags["last_mask_quality"] = last_mask
    flags["last_image_quality"] = last_image_quality
    return flags


def make_observation(ep, prev_steps):
    history = [observable_step(s) for s in prev_steps]
    return {
        "schema": "controller_policy_observation_v1",
        "episode_id": ep.get("episode_id"),
        "task": "mining_helmet_from_miner_ref_under_degradation",
        "severity": ep.get("severity"),
        "step_index": len(prev_steps) + 1,
        "allowed_actions": ALLOWED_ACTIONS,
        "state_flags": infer_flags(prev_steps),
        "history": history[-10:],
    }


def make_sft_row(ep, step, prev_steps, trace_source):
    observation = make_observation(ep, prev_steps)
    target = action_label(step)
    return {
        "id": f"{ep.get('episode_id')}_step{step.get('step')}",
        "trace_source": trace_source,
        "episode_id": ep.get("episode_id"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the high-level policy controller for a multi-round coal-mine "
                    "segmentation system. Choose exactly one next action as valid JSON. "
                    "Do not predict masks. Do not use unavailable ground-truth IoU."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(observation, ensure_ascii=False, sort_keys=True),
            },
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, sort_keys=True),
            },
        ],
        "observation": observation,
        "target_action": target,
        "oracle": step_oracle(step),
    }


def step_oracle(step):
    oracle = {}
    quality = step.get("quality")
    if isinstance(quality, dict) and "iou" in quality:
        oracle["iou"] = round_float(quality["iou"])
    for key in ORACLE_KEYS:
        if key in step:
            oracle[key] = round_float(step[key])
    if "decision" in step:
        oracle["trace_decision"] = step["decision"]
    return oracle


def episode_reward(ep):
    final_iou = float(ep.get("final_target_iou", ep.get("final_helmet_iou", 0.0)) or 0.0)
    accepted = 1.0 if ep.get("final_decision") == "accepted" else 0.0
    step_cost = 0.01 * len(ep.get("steps", []))
    return round(final_iou + accepted - step_cost, 6)


def make_trajectory_row(ep, trace_source):
    return {
        "schema": "controller_policy_trajectory_v1",
        "trace_source": trace_source,
        "episode_id": ep.get("episode_id"),
        "severity": ep.get("severity"),
        "num_steps": len(ep.get("steps", [])),
        "observable_steps": [observable_step(s) for s in ep.get("steps", [])],
        "action_sequence": [action_label(s) for s in ep.get("steps", [])],
        "oracle": {
            "final_decision": ep.get("final_decision"),
            "final_path": ep.get("final_path"),
            "final_target_iou": round_float(ep.get("final_target_iou", 0.0)),
            "final_anchor_iou": round_float(ep.get("final_anchor_iou", 0.0)),
            "accepted": ep.get("final_decision") == "accepted",
            "episode_reward": episode_reward(ep),
            "step_oracles": [step_oracle(s) for s in ep.get("steps", [])],
        },
    }


def preference_pair(pair_id, ep, observation, chosen, rejected, source, margin=1.0, meta=None):
    return {
        "id": f"{ep.get('episode_id')}_{pair_id}",
        "episode_id": ep.get("episode_id"),
        "source": source,
        "margin": round_float(margin),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the high-level policy controller for a multi-round coal-mine "
                    "segmentation system. Choose the better next action as valid JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(observation, ensure_ascii=False, sort_keys=True),
            },
        ],
        "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
        "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
        "chosen_action": chosen,
        "rejected_action": rejected,
        "meta": meta or {},
    }


def make_preference_rows(ep, trace_source):
    rows = []
    steps = ep.get("steps", [])
    for i, step in enumerate(steps):
        prev = steps[: i + 1]
        action = step.get("action", "")
        observation = make_observation(ep, prev)

        # Local candidate result: train the policy to keep beneficial local
        # results and rollback harmful ones. The improvement is oracle-derived
        # and used only for the preference label.
        if action in {"segment_target_direct_local_enhance", "segment_helmet_local_enhance", "segment_anchor_local_enhance"}:
            delta = float(step.get("improvement_over_current_best", step.get("improvement_over_focus", 0.0)) or 0.0)
            scope = "anchor" if "anchor" in action else ("ref_target" if "helmet" in action else "target")
            keep = {"action": "KEEP_LOCAL_RESULT", "scope": scope}
            rollback = {"action": "ROLLBACK_LOCAL_RESULT", "scope": scope}
            if delta > 1e-6:
                chosen, rejected = keep, rollback
            else:
                chosen, rejected = rollback, keep
            rows.append(
                preference_pair(
                    f"local_{step.get('step')}",
                    ep,
                    observation,
                    chosen,
                    rejected,
                    f"{trace_source}:local_candidate",
                    margin=abs(delta),
                    meta={"oracle_delta": round_float(delta), "source_action": action},
                )
            )

        # If focus/blackout branch failed and rollback was executed, prefer
        # rollback to keeping the bad focus state.
        if action == "rollback_to_accepted_ref":
            rows.append(
                preference_pair(
                    f"rollback_{step.get('step')}",
                    ep,
                    observation,
                    {"action": "ROLLBACK_TO_BEST_DIRECT"},
                    {"action": "STOP_WITH_CURRENT_STATE"},
                    f"{trace_source}:rollback",
                    margin=1.0,
                    meta={"rollback_target": step.get("rollback_target")},
                )
            )

        # After an accepted segmentation, prefer stopping over unnecessary
        # continuation. This label is oracle-derived from the trace.
        if step.get("decision") == "accept" and action in SEGMENT_ACTIONS:
            rows.append(
                preference_pair(
                    f"stop_success_{step.get('step')}",
                    ep,
                    observation,
                    {"action": "STOP_SUCCESS"},
                    {"action": "CONTINUE_TO_MORE_TOOLS"},
                    f"{trace_source}:stop_success",
                    margin=1.0,
                    meta={"source_action": action},
                )
            )

        # When direct target attempts failed and the trace proceeds to anchor,
        # prefer trying anchor over stopping.
        if action == "find_anchor":
            rows.append(
                preference_pair(
                    f"try_anchor_{step.get('step')}",
                    ep,
                    make_observation(ep, steps[:i]),
                    {"action": "SEG_ANCHOR"},
                    {"action": "STOP_FAIL"},
                    f"{trace_source}:anchor_recovery",
                    margin=1.0,
                    meta={"reason": step.get("reason")},
                )
            )
    return rows


def split_rows(rows, val_ratio):
    if val_ratio <= 0:
        return rows, []
    train, val = [], []
    for row in rows:
        key = row.get("episode_id", row.get("id", ""))
        digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        (val if score < val_ratio else train).append(row)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True, help="Trace JSONL path. Can be repeated.")
    parser.add_argument("--trace-name", action="append", help="Optional name per --trace.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_names = args.trace_name or []
    while len(trace_names) < len(args.trace):
        trace_names.append(Path(args.trace[len(trace_names)]).stem)

    trajectories = []
    sft_rows = []
    pref_rows = []
    stats = {
        "trace_files": args.trace,
        "trace_names": trace_names,
        "episodes": 0,
        "sft_rows": 0,
        "preference_rows": 0,
        "action_counts": Counter(),
        "final_decision_counts": Counter(),
        "final_path_counts": Counter(),
        "preference_source_counts": Counter(),
    }

    for trace_path, trace_name in zip(args.trace, trace_names):
        episodes = read_jsonl(trace_path)
        for ep in episodes:
            stats["episodes"] += 1
            stats["final_decision_counts"][ep.get("final_decision")] += 1
            stats["final_path_counts"][ep.get("final_path")] += 1
            trajectories.append(make_trajectory_row(ep, trace_name))
            prev = []
            for step in ep.get("steps", []):
                sft_rows.append(make_sft_row(ep, step, prev, trace_name))
                stats["action_counts"][canonical_action(step)] += 1
                prev.append(step)
            pairs = make_preference_rows(ep, trace_name)
            for pair in pairs:
                stats["preference_source_counts"][pair["source"]] += 1
            pref_rows.extend(pairs)

    sft_train, sft_val = split_rows(sft_rows, args.val_ratio)
    pref_train, pref_val = split_rows(pref_rows, args.val_ratio)

    write_jsonl(out_dir / "controller_policy_trajectories_v1.jsonl", trajectories)
    write_jsonl(out_dir / "policy_sft_all_v1.jsonl", sft_rows)
    write_jsonl(out_dir / "policy_sft_train_v1.jsonl", sft_train)
    write_jsonl(out_dir / "policy_sft_val_v1.jsonl", sft_val)
    write_jsonl(out_dir / "action_preference_pairs_all_v1.jsonl", pref_rows)
    write_jsonl(out_dir / "action_preference_pairs_train_v1.jsonl", pref_train)
    write_jsonl(out_dir / "action_preference_pairs_val_v1.jsonl", pref_val)

    stats["sft_rows"] = len(sft_rows)
    stats["sft_train_rows"] = len(sft_train)
    stats["sft_val_rows"] = len(sft_val)
    stats["preference_rows"] = len(pref_rows)
    stats["preference_train_rows"] = len(pref_train)
    stats["preference_val_rows"] = len(pref_val)
    for key in ["action_counts", "final_decision_counts", "final_path_counts", "preference_source_counts"]:
        stats[key] = dict(stats[key])

    (out_dir / "controller_policy_dataset_stats_v1.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Controller Policy Dataset v1\n\n",
        f"Episodes: `{stats['episodes']}`\n\n",
        f"SFT rows: `{stats['sft_rows']}` train `{stats['sft_train_rows']}` val `{stats['sft_val_rows']}`\n\n",
        f"Preference rows: `{stats['preference_rows']}` train `{stats['preference_train_rows']}` val `{stats['preference_val_rows']}`\n\n",
        "Policy observations exclude GT IoU and oracle improvement fields. Oracle values are kept only in metadata/reward labels.\n\n",
        "## Action Counts\n\n",
    ]
    for k, v in sorted(stats["action_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        report.append(f"- `{k}`: {v}\n")
    report.append("\n## Preference Sources\n\n")
    for k, v in sorted(stats["preference_source_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        report.append(f"- `{k}`: {v}\n")
    (out_dir / "README.md").write_text("".join(report), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

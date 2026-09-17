#!/usr/bin/env python3
"""Build strict action-index controller policy data from Stage-3 traces.

Policy v1 proved that SFT can learn the controller sequence, but direct DPO on
free-form JSON action names caused schema drift. This v2 builder constrains the
assistant target to a single integer action index:

    {"action_index": 0}

All action ids, source action names, ROI boxes, paths, GT IoU, oracle
improvement fields, and trace decisions are kept out of the generated response.
Oracle fields may appear only in metadata used for analysis.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ACTION_SPACE = [
    {"index": 0, "id": "SEG_TARGET_DIRECT", "description": "segment the target on the current image"},
    {"index": 1, "id": "GLOBAL_ENHANCE", "description": "apply normal global low-light restoration"},
    {"index": 2, "id": "GLOBAL_ENHANCE_STRONG", "description": "apply stronger global low-light restoration"},
    {"index": 3, "id": "LOCAL_ENHANCE_TARGET", "description": "locally enhance a rough target ROI"},
    {"index": 4, "id": "SEG_TARGET_LOCAL", "description": "segment the target after local target enhancement"},
    {"index": 5, "id": "SEG_ANCHOR", "description": "segment the coal miner anchor"},
    {"index": 6, "id": "LOCAL_ENHANCE_ANCHOR", "description": "locally enhance a rough anchor ROI"},
    {"index": 7, "id": "SEG_ANCHOR_LOCAL", "description": "segment the anchor after local anchor enhancement"},
    {"index": 8, "id": "BLACKOUT_WITH_ANCHOR", "description": "black out non-anchor regions using the accepted anchor"},
    {"index": 9, "id": "SEG_TARGET_WITH_REF", "description": "segment the target with an accepted reference"},
    {"index": 10, "id": "LOCAL_ENHANCE_REF_TARGET", "description": "locally enhance a rough target ROI after reference focus"},
    {"index": 11, "id": "SEG_REF_TARGET_LOCAL", "description": "segment the target after reference-local enhancement"},
    {"index": 12, "id": "ROLLBACK_LOCAL_RESULT", "description": "rollback a harmful local enhancement result"},
    {"index": 13, "id": "ROLLBACK_TO_BEST_DIRECT", "description": "rollback to the best earlier accepted/reference state"},
    {"index": 14, "id": "STOP_SUCCESS", "description": "stop because the current target result is good enough"},
    {"index": 15, "id": "STOP_FAIL", "description": "stop because available tools failed or budget is exhausted"},
]

ACTION_TO_INDEX = {row["id"]: row["index"] for row in ACTION_SPACE}
INDEX_TO_ACTION = {row["index"]: row["id"] for row in ACTION_SPACE}
ALL_ACTION_INDICES = [row["index"] for row in ACTION_SPACE]

ORACLE_KEYS = {
    "iou",
    "improvement_over_degraded",
    "improvement_over_enhanced",
    "improvement_over_best_direct",
    "improvement_over_current_best",
    "improvement_over_focus",
    "improvement_over_failed_focus",
}

SEGMENT_ACTIONS = {
    "segment_target_direct",
    "segment_target_direct_local_enhance",
    "segment_anchor",
    "segment_anchor_local_enhance",
    "segment_helmet",
    "segment_helmet_local_enhance",
    "restore_direct_target_result",
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


def bucket(value, thresholds):
    try:
        value = float(value)
    except Exception:
        return "unknown"
    for i, th in enumerate(thresholds):
        if value <= th:
            return f"b{i}"
    return f"b{len(thresholds)}"


def sanitize_obj(obj):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in ORACLE_KEYS or key in {"bbox_xyxy", "roi_xyxy"}:
                continue
            out[key] = sanitize_obj(value)
        return out
    if isinstance(obj, list):
        return [sanitize_obj(x) for x in obj]
    return round_float(obj)


def sanitize_quality(quality):
    if not isinstance(quality, dict):
        return None
    out = {}
    for key, value in quality.items():
        if key in ORACLE_KEYS or key in {"bbox_xyxy"}:
            continue
        out[key] = sanitize_obj(value)
    if "area_ratio" in out:
        out["area_ratio_bucket"] = bucket(out["area_ratio"], [0.0005, 0.002, 0.01, 0.05, 0.15])
    if "connected_components" in out:
        out["fragmentation_bucket"] = bucket(out["connected_components"], [1, 3, 8, 20])
    return out


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
    if action in {"find_anchor", "segment_anchor"}:
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
    if action in {"rollback_to_accepted_ref", "restore_direct_target_result"}:
        return "ROLLBACK_TO_BEST_DIRECT"
    if action == "stop_failed_anchor":
        return "STOP_FAIL"
    return action.upper()


def action_index_from_id(action_id):
    if action_id not in ACTION_TO_INDEX:
        raise KeyError(f"Unknown action id: {action_id}")
    return ACTION_TO_INDEX[action_id]


def action_target(action_id):
    return {"action_index": action_index_from_id(action_id)}


def step_target(step):
    return action_target(canonical_action(step))


def observable_step(step):
    action_id = canonical_action(step)
    try:
        action_index = action_index_from_id(action_id)
    except KeyError:
        action_index = -1
    out = {
        "step": step.get("step"),
        "action_index": action_index,
        "action_id": action_id,
        "image_state": step.get("image_state"),
    }
    for key in ["target", "anchor_category", "ref", "condition", "roi_scale", "rollback_target"]:
        if key in step:
            out[key] = sanitize_obj(step[key])
    if "quality" in step:
        out["mask_quality"] = sanitize_quality(step["quality"])
    if "candidate_stats" in step:
        out["candidate_mask_quality"] = sanitize_quality(step["candidate_stats"])
    if "image_quality" in step:
        out["image_quality"] = sanitize_obj(step["image_quality"])
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
        action = step.get("action", "")
        if action in {
            "segment_target_direct",
            "segment_target_direct_local_enhance",
            "segment_helmet",
            "segment_helmet_local_enhance",
            "restore_direct_target_result",
        }:
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
        if action.startswith("rollback") or action == "restore_direct_target_result":
            flags["has_rollback_available"] = True
        if "image_quality" in step:
            last_image_quality = sanitize_obj(step["image_quality"])
    flags["last_mask_quality"] = last_mask
    flags["last_image_quality"] = last_image_quality
    return flags


def make_observation(ep, prev_steps):
    return {
        "schema": "controller_policy_observation_v2_action_index",
        "episode_id": ep.get("episode_id"),
        "task": "segment mining_helmet in a degraded coal-mine image using direct segmentation, enhancement, anchor reference, blackout, local enhancement, and rollback tools",
        "severity": ep.get("severity"),
        "step_index": len(prev_steps) + 1,
        "action_space": ACTION_SPACE,
        "valid_action_indices": ALL_ACTION_INDICES,
        "state_flags": infer_flags(prev_steps),
        "history": [observable_step(s) for s in prev_steps][-10:],
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


def make_sft_row(ep, step, prev_steps, trace_source):
    observation = make_observation(ep, prev_steps)
    target = step_target(step)
    return {
        "id": f"{ep.get('episode_id')}_step{step.get('step')}",
        "trace_source": trace_source,
        "episode_id": ep.get("episode_id"),
        "schema": "controller_policy_sft_v2_action_index",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the high-level policy controller for a multi-round coal-mine "
                    "segmentation system. Output JSON only in this exact schema: "
                    "{\"action_index\": <integer>}. Choose one index from valid_action_indices. "
                    "Do not output action names, explanations, masks, paths, boxes, or scores."
                ),
            },
            {"role": "user", "content": json.dumps(observation, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, sort_keys=True)},
        ],
        "observation": observation,
        "target_action": target,
        "target_action_id": INDEX_TO_ACTION[target["action_index"]],
        "oracle": step_oracle(step),
    }


def make_trajectory_row(ep, trace_source):
    return {
        "schema": "controller_policy_trajectory_v2_action_index",
        "trace_source": trace_source,
        "episode_id": ep.get("episode_id"),
        "severity": ep.get("severity"),
        "num_steps": len(ep.get("steps", [])),
        "observable_steps": [observable_step(s) for s in ep.get("steps", [])],
        "action_indices": [step_target(s)["action_index"] for s in ep.get("steps", [])],
        "action_ids": [canonical_action(s) for s in ep.get("steps", [])],
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


def preference_pair(pair_id, ep, observation, chosen_id, rejected_id, source, margin=1.0, meta=None):
    chosen = action_target(chosen_id)
    rejected = action_target(rejected_id)
    return {
        "id": f"{ep.get('episode_id')}_{pair_id}",
        "episode_id": ep.get("episode_id"),
        "schema": "controller_policy_preference_v2_action_index",
        "source": source,
        "margin": round_float(abs(float(margin or 1.0))),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the high-level policy controller for a multi-round coal-mine "
                    "segmentation system. Output JSON only in this exact schema: "
                    "{\"action_index\": <integer>}. Choose the better legal next action. "
                    "Do not output action names or explanations."
                ),
            },
            {"role": "user", "content": json.dumps(observation, ensure_ascii=False, sort_keys=True)},
        ],
        "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
        "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
        "chosen_action": chosen,
        "rejected_action": rejected,
        "chosen_action_id": chosen_id,
        "rejected_action_id": rejected_id,
        "meta": meta or {},
    }


def next_action_id(steps, start_idx, default_id="STOP_SUCCESS"):
    for nxt in steps[start_idx + 1 :]:
        try:
            return canonical_action(nxt)
        except KeyError:
            continue
    return default_id


def make_preference_rows(ep, trace_source):
    rows = []
    steps = ep.get("steps", [])
    for i, step in enumerate(steps):
        action = step.get("action", "")

        if action in {
            "segment_target_direct_local_enhance",
            "segment_helmet_local_enhance",
            "segment_anchor_local_enhance",
        }:
            delta = float(
                step.get(
                    "improvement_over_current_best",
                    step.get("improvement_over_focus", step.get("improvement_over_best_direct", 0.0)),
                )
                or 0.0
            )
            observation = make_observation(ep, steps[: i + 1])
            if delta > 1e-6:
                chosen_id = next_action_id(steps, i, default_id="STOP_SUCCESS")
                if chosen_id == "ROLLBACK_LOCAL_RESULT":
                    chosen_id = "STOP_SUCCESS"
                rejected_id = "ROLLBACK_LOCAL_RESULT"
            else:
                chosen_id = "ROLLBACK_LOCAL_RESULT"
                rejected_id = "STOP_SUCCESS"
            rows.append(
                preference_pair(
                    f"local_{step.get('step')}",
                    ep,
                    observation,
                    chosen_id,
                    rejected_id,
                    f"{trace_source}:local_candidate",
                    margin=max(abs(delta), 0.05),
                    meta={"oracle_delta": round_float(delta), "source_action": action},
                )
            )

        if action == "rollback_to_accepted_ref":
            rows.append(
                preference_pair(
                    f"rollback_{step.get('step')}",
                    ep,
                    make_observation(ep, steps[:i]),
                    "ROLLBACK_TO_BEST_DIRECT",
                    "STOP_FAIL",
                    f"{trace_source}:rollback",
                    margin=1.0,
                    meta={"rollback_target": step.get("rollback_target")},
                )
            )

        if step.get("decision") == "accept" and action in SEGMENT_ACTIONS:
            rows.append(
                preference_pair(
                    f"stop_success_{step.get('step')}",
                    ep,
                    make_observation(ep, steps[: i + 1]),
                    "STOP_SUCCESS",
                    next_action_id(steps, i, default_id="STOP_FAIL"),
                    f"{trace_source}:stop_success",
                    margin=1.0,
                    meta={"source_action": action},
                )
            )

        if action == "find_anchor":
            rows.append(
                preference_pair(
                    f"try_anchor_{step.get('step')}",
                    ep,
                    make_observation(ep, steps[:i]),
                    "SEG_ANCHOR",
                    "STOP_FAIL",
                    f"{trace_source}:anchor_recovery",
                    margin=1.0,
                    meta={},
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
    parser.add_argument("--trace", action="append", required=True, help="Stage-3 trace JSONL path. Can be repeated.")
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
        "schema": "controller_policy_dataset_v2_action_index",
        "trace_files": args.trace,
        "trace_names": trace_names,
        "episodes": 0,
        "action_space": ACTION_SPACE,
        "action_counts": Counter(),
        "final_decision_counts": Counter(),
        "final_path_counts": Counter(),
        "preference_source_counts": Counter(),
        "preference_chosen_counts": Counter(),
        "preference_rejected_counts": Counter(),
    }

    for trace_path, trace_name in zip(args.trace, trace_names):
        for ep in read_jsonl(trace_path):
            stats["episodes"] += 1
            stats["final_decision_counts"][ep.get("final_decision")] += 1
            stats["final_path_counts"][ep.get("final_path")] += 1
            trajectories.append(make_trajectory_row(ep, trace_name))
            prev = []
            for step in ep.get("steps", []):
                row = make_sft_row(ep, step, prev, trace_name)
                sft_rows.append(row)
                stats["action_counts"][row["target_action_id"]] += 1
                prev.append(step)
            pairs = make_preference_rows(ep, trace_name)
            for pair in pairs:
                stats["preference_source_counts"][pair["source"]] += 1
                stats["preference_chosen_counts"][pair["chosen_action_id"]] += 1
                stats["preference_rejected_counts"][pair["rejected_action_id"]] += 1
            pref_rows.extend(pairs)

    sft_train, sft_val = split_rows(sft_rows, args.val_ratio)
    pref_train, pref_val = split_rows(pref_rows, args.val_ratio)

    write_jsonl(out_dir / "controller_policy_trajectories_v2.jsonl", trajectories)
    write_jsonl(out_dir / "policy_sft_all_v2.jsonl", sft_rows)
    write_jsonl(out_dir / "policy_sft_train_v2.jsonl", sft_train)
    write_jsonl(out_dir / "policy_sft_val_v2.jsonl", sft_val)
    write_jsonl(out_dir / "action_preference_pairs_all_v2.jsonl", pref_rows)
    write_jsonl(out_dir / "action_preference_pairs_train_v2.jsonl", pref_train)
    write_jsonl(out_dir / "action_preference_pairs_val_v2.jsonl", pref_val)

    stats["sft_rows"] = len(sft_rows)
    stats["sft_train_rows"] = len(sft_train)
    stats["sft_val_rows"] = len(sft_val)
    stats["preference_rows"] = len(pref_rows)
    stats["preference_train_rows"] = len(pref_train)
    stats["preference_val_rows"] = len(pref_val)
    for key in [
        "action_counts",
        "final_decision_counts",
        "final_path_counts",
        "preference_source_counts",
        "preference_chosen_counts",
        "preference_rejected_counts",
    ]:
        stats[key] = dict(stats[key])

    (out_dir / "controller_policy_dataset_stats_v2.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Controller Policy Dataset v2\n\n",
        "Assistant outputs are strict `{\"action_index\": int}` JSON. Action names and executor-computable fields are not generated by the model.\n\n",
        f"Episodes: `{stats['episodes']}`\n\n",
        f"SFT rows: `{stats['sft_rows']}` train `{stats['sft_train_rows']}` val `{stats['sft_val_rows']}`\n\n",
        f"Preference rows: `{stats['preference_rows']}` train `{stats['preference_train_rows']}` val `{stats['preference_val_rows']}`\n\n",
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

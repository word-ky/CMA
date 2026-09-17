#!/usr/bin/env python3
"""Build rollout-level action preferences for the controller policy.

This script evaluates multiple legal next actions from the same observable
state. Each candidate is executed by the real Stage-3 executor and then
continued for a short horizon by a tail policy. The resulting final target IoU,
accepted flag, step cost, and invalid-action count define the reward used to
create DPO preference pairs.
"""

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_controller_policy_dataset_v2 import (  # noqa: E402
    ACTION_SPACE,
    ACTION_TO_INDEX,
    ALL_ACTION_INDICES,
    canonical_action,
    make_observation,
)
from stage3_policy_v2_rollout import (  # noqa: E402
    PolicyRunner,
    ControllerRunner,
    run_policy_episode,
)
from stage3_rule_controller_v3_seg_local_enhance import read_jsonl, write_jsonl  # noqa: E402


SYSTEM_PROMPT = (
    "You are the high-level policy controller for a multi-round coal-mine "
    "segmentation system. Output JSON only in this exact schema: "
    "{\"action_index\": <integer>}. Choose the better legal next action. "
    "Do not output action names or explanations."
)


PRIORITY_ACTIONS = [
    "SEG_TARGET_DIRECT",
    "GLOBAL_ENHANCE",
    "GLOBAL_ENHANCE_STRONG",
    "LOCAL_ENHANCE_TARGET",
    "SEG_TARGET_LOCAL",
    "SEG_ANCHOR",
    "BLACKOUT_WITH_ANCHOR",
    "SEG_TARGET_WITH_REF",
    "LOCAL_ENHANCE_REF_TARGET",
    "SEG_REF_TARGET_LOCAL",
    "ROLLBACK_LOCAL_RESULT",
    "ROLLBACK_TO_BEST_DIRECT",
    "STOP_SUCCESS",
    "STOP_FAIL",
]


def action_target(action_id):
    return {"action_index": ACTION_TO_INDEX[action_id]}


def pref_row(row_id, ep, observation, chosen_id, rejected_id, chosen_reward, rejected_reward, meta):
    return {
        "id": row_id,
        "episode_id": ep.get("episode_id"),
        "schema": "controller_policy_preference_v4_rollout",
        "source": "rollout_level_executor_reward",
        "margin": round(float(chosen_reward - rejected_reward), 6),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(observation, ensure_ascii=False, sort_keys=True)},
        ],
        "chosen": json.dumps(action_target(chosen_id), ensure_ascii=False, sort_keys=True),
        "rejected": json.dumps(action_target(rejected_id), ensure_ascii=False, sort_keys=True),
        "chosen_action": action_target(chosen_id),
        "rejected_action": action_target(rejected_id),
        "chosen_action_id": chosen_id,
        "rejected_action_id": rejected_id,
        "meta": meta,
    }


def load_episode_map(path):
    rows = read_jsonl(path)
    return {row.get("episode_id"): row for row in rows}


def append_jsonl(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_partial_stats(path, stats):
    serializable = {}
    for key, value in stats.items():
        if isinstance(value, Counter):
            serializable[key] = dict(value)
        else:
            serializable[key] = value
    Path(path).write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_canonical_prefix(steps):
    out = []
    for step in steps:
        try:
            action_id = canonical_action(step)
        except KeyError:
            break
        if action_id in {"STOP_SUCCESS", "STOP_FAIL"}:
            break
        out.append(action_id)
    return out


def legal_actions_from_observation(observation):
    flags = observation.get("state_flags", {})
    legal = set()
    legal.update(["SEG_TARGET_DIRECT", "GLOBAL_ENHANCE", "GLOBAL_ENHANCE_STRONG", "SEG_ANCHOR", "STOP_FAIL"])
    if flags.get("has_target_candidate"):
        legal.update(["LOCAL_ENHANCE_TARGET", "ROLLBACK_TO_BEST_DIRECT"])
    if flags.get("has_local_target_state"):
        legal.add("SEG_TARGET_LOCAL")
    if flags.get("has_anchor_ref"):
        legal.update(["LOCAL_ENHANCE_ANCHOR", "BLACKOUT_WITH_ANCHOR", "SEG_TARGET_WITH_REF"])
    if flags.get("has_blackout_state"):
        legal.update(["SEG_TARGET_WITH_REF", "LOCAL_ENHANCE_REF_TARGET"])
    if flags.get("has_local_target_state") and flags.get("has_anchor_ref"):
        legal.add("SEG_REF_TARGET_LOCAL")
    if flags.get("has_rollback_available"):
        legal.update(["ROLLBACK_LOCAL_RESULT", "ROLLBACK_TO_BEST_DIRECT"])
    last_mask = flags.get("last_mask_quality") or {}
    if flags.get("has_target_candidate") and last_mask.get("quality_bucket") == "high":
        legal.add("STOP_SUCCESS")
    return [a for a in PRIORITY_ACTIONS if a in legal]


def rank_candidate_actions(observation, oracle_next=None):
    """Order legal candidates so deeper reference/rollback tools are not starved."""

    legal = set(legal_actions_from_observation(observation))
    flags = observation.get("state_flags", {})
    ranked = []

    def add(action_id):
        if action_id in legal and action_id not in ranked:
            ranked.append(action_id)

    if oracle_next:
        add(oracle_next)

    add("SEG_TARGET_DIRECT")
    add("SEG_ANCHOR")

    if flags.get("has_anchor_ref"):
        add("BLACKOUT_WITH_ANCHOR")
        add("SEG_TARGET_WITH_REF")

    if flags.get("has_blackout_state"):
        add("SEG_TARGET_WITH_REF")
        add("LOCAL_ENHANCE_REF_TARGET")

    if flags.get("has_target_candidate"):
        add("LOCAL_ENHANCE_TARGET")
        add("ROLLBACK_TO_BEST_DIRECT")

    if flags.get("has_local_target_state"):
        add("SEG_TARGET_LOCAL")
        if flags.get("has_anchor_ref"):
            add("SEG_REF_TARGET_LOCAL")

    if flags.get("has_rollback_available"):
        add("ROLLBACK_LOCAL_RESULT")
        add("ROLLBACK_TO_BEST_DIRECT")

    add("GLOBAL_ENHANCE")
    add("GLOBAL_ENHANCE_STRONG")
    add("STOP_SUCCESS")
    add("STOP_FAIL")

    for action_id in PRIORITY_ACTIONS:
        add(action_id)
    return ranked


def selected_prefix_lengths(steps, states_per_episode):
    lengths = [0]
    if states_per_episode <= 1:
        return lengths

    groups = [
        {"segment_target_direct"},
        {"enhance_lowlight", "enhance_lowlight_strong"},
        {"local_enhance_target_by_seg", "segment_target_direct_local_enhance"},
        {"segment_anchor", "segment_anchor_local_enhance"},
        {"focus_blackout"},
        {"segment_helmet"},
        {"local_enhance_target_by_seg_after_ref", "segment_helmet_local_enhance"},
        {"rollback_to_accepted_ref", "restore_direct_target_result"},
    ]
    for group in groups:
        for i, step in enumerate(steps):
            action = step.get("action", "")
            if action in group or (
                action.startswith("rollback_local")
                and any(item.startswith("rollback") or item.startswith("restore") for item in group)
            ):
                prefix_len = i + 1
                if prefix_len not in lengths:
                    lengths.append(prefix_len)
                break
        if len(lengths) >= states_per_episode:
            return lengths[:states_per_episode]

    candidates = []
    for i, step in enumerate(steps):
        action = step.get("action", "")
        if action in {
            "segment_target_direct",
            "enhance_lowlight",
            "enhance_lowlight_strong",
            "local_enhance_target_by_seg",
            "segment_target_direct_local_enhance",
            "segment_anchor",
            "segment_anchor_local_enhance",
            "focus_blackout",
            "segment_helmet",
            "local_enhance_target_by_seg_after_ref",
            "segment_helmet_local_enhance",
            "rollback_to_accepted_ref",
            "restore_direct_target_result",
        } or action.startswith("rollback_local"):
            candidates.append(i + 1)
    for item in candidates:
        if item not in lengths:
            lengths.append(item)
        if len(lengths) >= states_per_episode:
            break
    return lengths


def prefix_baseline_iou(prefix_steps):
    best = 0.0
    for step in prefix_steps:
        action = step.get("action", "")
        if action in {
            "segment_target_direct",
            "segment_target_direct_local_enhance",
            "segment_helmet",
            "segment_helmet_local_enhance",
            "restore_direct_target_result",
        }:
            quality = step.get("quality") or {}
            try:
                best = max(best, float(quality.get("iou", 0.0) or 0.0))
            except Exception:
                pass
    return best


class ForcedThenTailPolicy:
    def __init__(self, forced_action_ids, tail_policy):
        self.forced_action_ids = forced_action_ids
        self.tail_policy = tail_policy

    def predict(self, ep, steps):
        if len(steps) < len(self.forced_action_ids):
            action_id = self.forced_action_ids[len(steps)]
            return {
                "observation": None,
                "generated_text": json.dumps(action_target(action_id)),
                "parsed": action_target(action_id),
                "action_index": ACTION_TO_INDEX[action_id],
                "action_id": action_id,
                "schema_ok": True,
            }
        return self.tail_policy.predict(ep, steps)


def rollout_reward(row, args, baseline_iou=0.0):
    final_iou = float(row.get("final_target_iou", 0.0) or 0.0)
    accepted = 1.0 if row.get("final_decision") == "accepted" else 0.0
    invalid = float(row.get("invalid_actions", 0) or 0)
    steps = float(row.get("num_steps", len(row.get("steps", []))) or 0)
    premature_stop = 1.0 if "stop_success" in str(row.get("final_path", "")) and not accepted else 0.0
    stop_fail = 1.0 if row.get("final_path") == "policy_stop_fail" else 0.0
    improvement = max(0.0, final_iou - float(baseline_iou or 0.0))
    return (
        final_iou
        + args.accept_bonus * accepted
        + args.improvement_bonus * improvement
        - args.step_penalty * steps
        - args.invalid_penalty * invalid
        - args.premature_stop_penalty * premature_stop
        - args.stop_fail_penalty * stop_fail
    )


def candidate_summary(action_id, row, reward):
    return {
        "action_id": action_id,
        "action_index": ACTION_TO_INDEX[action_id],
        "reward": round(float(reward), 6),
        "final_target_iou": round(float(row.get("final_target_iou", 0.0) or 0.0), 6),
        "final_anchor_iou": round(float(row.get("final_anchor_iou", 0.0) or 0.0), 6),
        "final_decision": row.get("final_decision"),
        "final_path": row.get("final_path"),
        "invalid_actions": int(row.get("invalid_actions", 0) or 0),
        "num_steps": int(row.get("num_steps", len(row.get("steps", []))) or 0),
        "actions": [s.get("policy_action_id") for s in row.get("steps", []) if s.get("policy_action_id")],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Segmentation model path.")
    parser.add_argument("--episodes-jsonl", required=True)
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--policy-adapter", required=True)
    parser.add_argument("--policy-max-length", type=int, default=4096)
    parser.add_argument("--policy-max-new-tokens", type=int, default=32)
    parser.add_argument("--policy-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-index", type=int, default=0, help="Start offset over matched trace episodes.")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--states-per-episode", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--tail-steps", type=int, default=4)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--accept-bonus", type=float, default=1.0)
    parser.add_argument("--improvement-bonus", type=float, default=0.5)
    parser.add_argument("--step-penalty", type=float, default=0.02)
    parser.add_argument("--invalid-penalty", type=float, default=0.5)
    parser.add_argument("--premature-stop-penalty", type=float, default=0.5)
    parser.add_argument("--stop-fail-penalty", type=float, default=0.2)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument("--vision-pretrained", default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth")
    parser.add_argument("--degradation", default="extreme", choices=["medium", "hard", "extreme", "mixed"])
    parser.add_argument("--miner-accept-iou", type=float, default=0.55)
    parser.add_argument("--miner-min-continue-iou", type=float, default=0.30)
    parser.add_argument("--helmet-accept-iou", type=float, default=0.55)
    parser.add_argument("--helmet-head-min-score", type=float, default=0.3)
    parser.add_argument("--low-brightness", type=float, default=0.28)
    parser.add_argument("--focus-dilate", type=int, default=21)
    parser.add_argument("--enable-local-enhance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-enhance-strength", default="strong", choices=["normal", "strong"])
    parser.add_argument("--local-roi-expand", type=float, default=0.10)
    parser.add_argument("--local-roi-scales-target", default="1.2")
    parser.add_argument("--local-roi-scales-anchor", default="1.5")
    parser.add_argument("--local-roi-scales-ref-target", default="1.2")
    parser.add_argument("--local-min-side", type=int, default=96)
    parser.add_argument("--local-blend", type=float, default=0.95)
    parser.add_argument("--local-feather", type=float, default=9.0)
    parser.add_argument("--local-min-area-ratio", type=float, default=0.00015)
    parser.add_argument("--local-max-area-ratio", type=float, default=0.40)
    parser.add_argument("--stream-write", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_map = load_episode_map(args.episodes_jsonl)
    traces = read_jsonl(args.trace_jsonl)
    seg_runner = ControllerRunner(args)
    tail_policy = PolicyRunner(args)

    candidate_rows = []
    pref_rows = []
    partial_candidates_path = out_dir / "rollout_candidates_v4.partial.jsonl"
    partial_prefs_path = out_dir / "action_preference_pairs_rollout_all_v4.partial.jsonl"
    partial_stats_path = out_dir / "rollout_pref_stats.partial.json"
    if args.stream_write:
        for path in [partial_candidates_path, partial_prefs_path]:
            if path.exists():
                path.unlink()
    stats = {
        "schema": "controller_policy_rollout_preferences_v4",
        "episodes_jsonl": args.episodes_jsonl,
        "trace_jsonl": args.trace_jsonl,
        "action_space": ACTION_SPACE,
        "num_trace_rows_seen": 0,
        "num_episodes_used": 0,
        "num_states": 0,
        "num_candidate_rollouts": 0,
        "num_preferences": 0,
        "candidate_action_counts": Counter(),
        "best_action_counts": Counter(),
    }

    matched = 0
    used = 0
    for trace in tqdm(traces, desc="build_rollout_preferences_v4"):
        stats["num_trace_rows_seen"] += 1
        ep = episode_map.get(trace.get("episode_id"))
        if ep is None:
            continue
        if matched < args.start_index:
            matched += 1
            continue
        matched += 1
        used += 1
        if used > args.max_episodes:
            break
        stats["num_episodes_used"] += 1
        trace_steps = trace.get("steps", [])
        for prefix_len in selected_prefix_lengths(trace_steps, args.states_per_episode):
            prefix_steps = trace_steps[:prefix_len]
            prefix_ids = safe_canonical_prefix(prefix_steps)
            if len(prefix_ids) != prefix_len:
                continue
            observation = make_observation(ep, prefix_steps)
            oracle_next = None
            if prefix_len < len(trace_steps):
                try:
                    oracle_next = canonical_action(trace_steps[prefix_len])
                except KeyError:
                    oracle_next = None
            candidates = rank_candidate_actions(observation, oracle_next=oracle_next)
            candidates = candidates[: args.max_candidates]
            if len(candidates) < 2:
                continue

            state_id = f"{ep.get('episode_id')}_prefix{prefix_len}"
            baseline_iou = prefix_baseline_iou(prefix_steps)
            stats["num_states"] += 1
            results = []
            for action_id in candidates:
                rollout_args = copy.copy(args)
                rollout_args.max_steps = len(prefix_ids) + 1 + args.tail_steps
                policy = ForcedThenTailPolicy(prefix_ids + [action_id], tail_policy)
                row = run_policy_episode(seg_runner, policy, ep, rollout_args, idx=0)
                reward = rollout_reward(row, args, baseline_iou=baseline_iou)
                summary = candidate_summary(action_id, row, reward)
                results.append((action_id, reward, summary))
                stats["candidate_action_counts"][action_id] += 1
                stats["num_candidate_rollouts"] += 1

            results.sort(key=lambda item: item[1], reverse=True)
            best_id, best_reward, _ = results[0]
            stats["best_action_counts"][best_id] += 1
            cand_row = {
                "id": state_id,
                "episode_id": ep.get("episode_id"),
                "prefix_len": prefix_len,
                "state_baseline_iou": round(float(baseline_iou), 6),
                "observation": observation,
                "oracle_next_action_id": oracle_next,
                "candidate_summaries": [item[2] for item in results],
            }
            candidate_rows.append(cand_row)

            state_pref_rows = []
            for rejected_id, rejected_reward, rejected_summary in results[1:]:
                margin = best_reward - rejected_reward
                if margin < args.min_margin:
                    continue
                row = pref_row(
                    f"{state_id}_{best_id}_gt_{rejected_id}",
                    ep,
                    observation,
                    best_id,
                    rejected_id,
                    best_reward,
                    rejected_reward,
                    {
                        "prefix_len": prefix_len,
                        "oracle_next_action_id": oracle_next,
                        "chosen_summary": results[0][2],
                        "rejected_summary": rejected_summary,
                    },
                )
                pref_rows.append(row)
                state_pref_rows.append(row)
            if args.stream_write:
                append_jsonl(partial_candidates_path, [cand_row])
                append_jsonl(partial_prefs_path, state_pref_rows)
                partial = dict(stats)
                partial["num_preferences"] = len(pref_rows)
                partial["candidate_action_counts"] = dict(stats["candidate_action_counts"])
                partial["best_action_counts"] = dict(stats["best_action_counts"])
                dump_partial_stats(partial_stats_path, partial)

    stats["num_preferences"] = len(pref_rows)
    stats["candidate_action_counts"] = dict(stats["candidate_action_counts"])
    stats["best_action_counts"] = dict(stats["best_action_counts"])
    rewards = [c["candidate_summaries"][0]["reward"] for c in candidate_rows if c.get("candidate_summaries")]
    stats["mean_best_reward"] = float(np.mean(rewards)) if rewards else 0.0

    write_jsonl(out_dir / "rollout_candidates_v4.jsonl", candidate_rows)
    write_jsonl(out_dir / "action_preference_pairs_rollout_all_v4.jsonl", pref_rows)
    split = max(1, int(0.9 * len(pref_rows)))
    write_jsonl(out_dir / "action_preference_pairs_rollout_train_v4.jsonl", pref_rows[:split])
    write_jsonl(out_dir / "action_preference_pairs_rollout_val_v4.jsonl", pref_rows[split:])
    (out_dir / "rollout_pref_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

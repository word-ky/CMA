#!/usr/bin/env python3
"""Roll out a strict action-index LLM policy through the Stage-3 executor.

This script keeps the segmentation/enhancement/blackout/rollback tools from
the rule controller and replaces only the next-action chooser with a PEFT LLM
policy that outputs {"action_index": int}.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_controller_policy_dataset_v2 import ACTION_SPACE, INDEX_TO_ACTION, make_observation  # noqa: E402
from stage3_rule_controller_v3_seg_local_enhance import (  # noqa: E402
    ControllerRunner,
    apply_global_enhance,
    bbox_from_mask,
    degrade_parametric,
    degrade_image,
    direct_query_for_round,
    focus_blackout,
    helmet_geometry,
    image_quality,
    is_local_enhance_candidate,
    is_target_ok,
    load_episode_assets,
    local_enhance_by_seg,
    load_task_enhancer_model,
    mask_stats,
    maybe_save_flow_visual,
    overlay_mask,
    parse_roi_scales,
    read_jsonl,
    score_seg,
    stable_seed,
    summarize,
    target_overlay,
    tile,
    write_jsonl,
)


def extract_first_json(text):
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        in_string = False
        escape = False
        for pos in range(start, len(text)):
            ch = text[pos]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : pos + 1])
                    except Exception:
                        break
    return None


class PolicyRunner:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dtype = torch.bfloat16 if args.policy_bf16 else torch.float16
        base = AutoModelForCausalLM.from_pretrained(
            args.policy_model,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model = PeftModel.from_pretrained(base, args.policy_adapter)
        self.model.cuda().eval()

    def make_messages(self, observation):
        return [
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
        ]

    def predict(self, ep, steps):
        observation = make_observation(ep, steps)
        messages = self.make_messages(observation)
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            for msg in messages:
                prompt += f"<|{msg['role']}|>\n{msg['content']}\n"
            prompt += "<|assistant|>\n"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.policy_max_length,
        ).to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.args.policy_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        parsed = extract_first_json(generated)
        action_index = None
        if isinstance(parsed, dict):
            value = parsed.get("action_index")
            if value is None and isinstance(parsed.get("output_json"), dict):
                value = parsed["output_json"].get("action_index")
            if value is None:
                value = parsed.get("output")
            if isinstance(value, int) and not isinstance(value, bool):
                action_index = value
            elif isinstance(value, str) and value.strip().isdigit():
                action_index = int(value.strip())
        action_id = INDEX_TO_ACTION.get(action_index)
        return {
            "observation": observation,
            "generated_text": generated,
            "parsed": parsed,
            "action_index": action_index,
            "action_id": action_id,
            "schema_ok": isinstance(parsed, dict) and set(parsed.keys()) == {"action_index"} and action_id is not None,
        }


def append_invalid(steps, pred, reason):
    action_for_history = pred.get("action_id") or "POLICY_INVALID_ACTION"
    steps.append(
        {
            "step": len(steps) + 1,
            "image_state": None,
            "action": action_for_history,
            "policy_action_index": pred.get("action_index"),
            "policy_action_id": pred.get("action_id"),
            "policy_generated_text": pred.get("generated_text"),
            "reason": reason,
            "decision": "invalid",
        }
    )


def is_miner_ok(quality, args):
    return quality["iou"] >= args.miner_accept_iou


def current_target(state):
    if state.get("last_target_quality") is not None:
        return state["last_target"], state["last_target_quality"], state["last_target_image"], state["last_target_state"]
    if state.get("best_target_quality") is not None:
        return state["best_target"], state["best_target_quality"], state["best_target_image"], state["best_target_state"]
    return None, None, None, None


def update_best_target(state, pred, quality, image, image_state):
    state["last_target"] = pred
    state["last_target_quality"] = quality
    state["last_target_image"] = image
    state["last_target_state"] = image_state
    best = state.get("best_target_quality")
    if best is None or quality["iou"] > best["iou"]:
        state["best_target"] = pred
        state["best_target_quality"] = quality
        state["best_target_image"] = image
        state["best_target_state"] = image_state


def update_anchor(state, pred, quality, image, image_state):
    state["last_anchor"] = pred
    state["last_anchor_quality"] = quality
    state["last_anchor_image"] = image
    state["last_anchor_state"] = image_state
    best = state.get("anchor_quality")
    if best is None or quality["iou"] > best["iou"]:
        state["anchor"] = pred
        state["anchor_quality"] = quality
        state["anchor_image"] = image
        state["anchor_state"] = image_state


def add_policy_fields(step, pred):
    step["policy_action_index"] = pred.get("action_index")
    step["policy_action_id"] = pred.get("action_id")
    step["policy_schema_ok"] = pred.get("schema_ok")
    return step


def run_policy_episode(seg_runner, policy_runner, ep, args, idx, overlay_dir=None):
    image_rgb, miner_gt, helmet_gt, round1, round2 = load_episode_assets(ep)
    seed = stable_seed(ep.get("episode_id", ep["image_path"]))
    if getattr(args, "degradation_config", None):
        severity = args.degradation_config.get("id", "parametric")
        degraded = degrade_parametric(image_rgb, args.degradation_config, seed)
    else:
        severity = args.degradation
        if severity == "mixed":
            severity = "hard" if seed % 2 else "medium"
        degraded = degrade_image(image_rgb, severity, seed)

    round1_query = round1.get(
        "query", "Please segment the miner wearing a helmet in the underground mining image."
    )
    round2_query = round2.get(
        "query", "Based on the miner segmented in the previous round, segment the helmet on his head."
    )
    direct_query = direct_query_for_round(round2)

    state = {
        "current_image": degraded,
        "current_state": "degraded",
        "degraded": degraded,
        "final_target": None,
        "final_target_quality": None,
        "invalid_actions": 0,
    }
    steps = []
    final_decision = None
    final_path = None
    panels = []
    if getattr(args, "save_visuals", False) and overlay_dir is not None and idx < args.overlay_limit:
        gt_panel = overlay_mask(image_rgb, miner_gt, (0, 140, 255), 0.45)
        gt_panel = overlay_mask(gt_panel, helmet_gt, (255, 40, 40), 0.55)
        panels.extend(
            [
                ("clean GT: miner blue, helmet red", gt_panel),
                (f"degraded input: {severity}", degraded),
            ]
        )

    target_scales = parse_roi_scales(args.local_roi_scales_target) or [1.2]
    anchor_scales = parse_roi_scales(args.local_roi_scales_anchor) or [1.5]
    ref_target_scales = parse_roi_scales(args.local_roi_scales_ref_target) or [1.2]

    for _ in range(args.max_steps):
        pred_action = policy_runner.predict(ep, steps)
        action_id = pred_action["action_id"]
        if action_id is None:
            append_invalid(steps, pred_action, "invalid_or_unparseable_policy_output")
            state["invalid_actions"] += 1
            final_decision = "failed_invalid_action"
            final_path = "invalid_policy"
            break

        if action_id == "SEG_TARGET_DIRECT":
            pred = seg_runner.predict(state["current_image"], direct_query, helmet_gt)
            quality = score_seg(pred, helmet_gt)
            ok = is_target_ok(quality, args)
            update_best_target(state, pred, quality, state["current_image"], state["current_state"])
            if panels is not None:
                panels.append(
                    (
                        f"{action_id}: target on {state['current_state']} IoU={quality['iou']:.3f}",
                        target_overlay(state["current_image"], helmet_gt, pred),
                    )
                )
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": state["current_state"],
                        "action": "segment_target_direct",
                        "target": round2.get("target_category", "mining_helmet"),
                        "quality": quality,
                        "image_quality": image_quality(state["current_image"]),
                        "decision": "accept" if ok else "failed",
                    },
                    pred_action,
                )
            )

        elif action_id == "GLOBAL_ENHANCE":
            state["current_image"] = apply_global_enhance(state["degraded"], "normal", args)
            state["current_state"] = "enhanced"
            if panels is not None:
                panels.append((f"{action_id}: global enhance", state["current_image"]))
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": "degraded",
                        "action": "enhance_lowlight",
                        "decision": "execute",
                    },
                    pred_action,
                )
            )

        elif action_id == "GLOBAL_ENHANCE_STRONG":
            state["current_image"] = apply_global_enhance(state["degraded"], "strong", args)
            state["current_state"] = "enhanced_strong"
            if panels is not None:
                panels.append((f"{action_id}: strong global enhance", state["current_image"]))
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": "degraded",
                        "action": "enhance_lowlight_strong",
                        "decision": "execute",
                    },
                    pred_action,
                )
            )

        elif action_id == "LOCAL_ENHANCE_TARGET":
            base_mask, base_quality, base_image, base_state = current_target(state)
            if base_mask is None:
                append_invalid(steps, pred_action, "no_target_mask_for_local_enhance")
                state["invalid_actions"] += 1
                continue
            ok_local, local_stats, local_reason = is_local_enhance_candidate(base_mask, args)
            if not ok_local:
                steps.append(
                    add_policy_fields(
                        {
                            "step": len(steps) + 1,
                            "image_state": base_state,
                            "action": "local_enhance_target_by_seg",
                            "condition": "best_direct_target_seg",
                            "candidate_stats": local_stats,
                            "reason": local_reason,
                            "decision": "skip",
                        },
                        pred_action,
                    )
                )
                continue
            scale = target_scales[0]
            local_image, roi = local_enhance_by_seg(
                base_image,
                base_mask,
                args,
                strength=args.local_enhance_strength,
                roi_scale=scale,
            )
            if panels is not None:
                panels.append((f"{action_id}: target ROI {scale:.1f}x", local_image))
            state["pending_local"] = {
                "scope": "target",
                "image": local_image,
                "state": f"{base_state}_local_target_{scale:.1f}x",
                "roi": roi,
                "base_state": base_state,
                "base_image": base_image,
                "base_quality": base_quality,
                "scale": scale,
            }
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": base_state,
                        "action": "local_enhance_target_by_seg",
                        "condition": "best_direct_target_seg",
                        "candidate_stats": local_stats,
                        "roi_scale": scale,
                        "roi_xyxy": roi,
                        "reason": local_reason,
                        "decision": "execute",
                    },
                    pred_action,
                )
            )

        elif action_id == "SEG_TARGET_LOCAL":
            pending = state.get("pending_local")
            if not pending or pending.get("scope") != "target":
                append_invalid(steps, pred_action, "no_pending_target_local_image")
                state["invalid_actions"] += 1
                continue
            pred = seg_runner.predict(pending["image"], direct_query, helmet_gt)
            quality = score_seg(pred, helmet_gt)
            delta = quality["iou"] - pending["base_quality"]["iou"]
            ok = is_target_ok(quality, args)
            state["current_image"] = pending["image"]
            state["current_state"] = pending["state"]
            update_best_target(state, pred, quality, pending["image"], pending["state"])
            if panels is not None:
                panels.append(
                    (
                        f"{action_id}: target local IoU={quality['iou']:.3f}",
                        target_overlay(pending["image"], helmet_gt, pred),
                    )
                )
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": pending["state"],
                        "action": "segment_target_direct_local_enhance",
                        "target": round2.get("target_category", "mining_helmet"),
                        "roi_scale": pending["scale"],
                        "roi_xyxy": pending["roi"],
                        "quality": quality,
                        "improvement_over_current_best": float(delta),
                        "decision": "accept" if ok else ("keep" if delta > 0 else "rollback"),
                    },
                    pred_action,
                )
            )

        elif action_id == "SEG_ANCHOR":
            pred = seg_runner.predict(state["current_image"], round1_query, miner_gt)
            quality = score_seg(pred, miner_gt)
            ok = is_miner_ok(quality, args)
            update_anchor(state, pred, quality, state["current_image"], state["current_state"])
            if panels is not None:
                anchor_panel = overlay_mask(state["current_image"], miner_gt, (0, 180, 0), 0.35)
                anchor_panel = overlay_mask(anchor_panel, pred, (0, 120, 255), 0.55)
                panels.append((f"{action_id}: anchor miner IoU={quality['iou']:.3f}", anchor_panel))
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": state["current_state"],
                        "action": "segment_anchor",
                        "target": round1.get("target_category", "coal_miner"),
                        "quality": quality,
                        "decision": "accept" if ok else "failed",
                    },
                    pred_action,
                )
            )

        elif action_id == "LOCAL_ENHANCE_ANCHOR":
            if state.get("last_anchor") is None:
                append_invalid(steps, pred_action, "no_anchor_mask_for_local_enhance")
                state["invalid_actions"] += 1
                continue
            ok_local, local_stats, local_reason = is_local_enhance_candidate(state["last_anchor"], args)
            if not ok_local:
                steps.append(
                    add_policy_fields(
                        {
                            "step": len(steps) + 1,
                            "image_state": state.get("last_anchor_state"),
                            "action": "local_enhance_anchor_by_seg",
                            "candidate_stats": local_stats,
                            "reason": local_reason,
                            "decision": "skip",
                        },
                        pred_action,
                    )
                )
                continue
            scale = anchor_scales[0]
            local_image, roi = local_enhance_by_seg(
                state["last_anchor_image"],
                state["last_anchor"],
                args,
                strength=args.local_enhance_strength,
                roi_scale=scale,
            )
            if panels is not None:
                panels.append((f"{action_id}: anchor ROI {scale:.1f}x", local_image))
            state["pending_local"] = {
                "scope": "anchor",
                "image": local_image,
                "state": f"{state['last_anchor_state']}_local_anchor_{scale:.1f}x",
                "roi": roi,
                "base_quality": state["last_anchor_quality"],
                "scale": scale,
            }
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": state.get("last_anchor_state"),
                        "action": "local_enhance_anchor_by_seg",
                        "candidate_stats": local_stats,
                        "roi_scale": scale,
                        "roi_xyxy": roi,
                        "reason": local_reason,
                        "decision": "execute",
                    },
                    pred_action,
                )
            )

        elif action_id == "SEG_ANCHOR_LOCAL":
            pending = state.get("pending_local")
            if not pending or pending.get("scope") != "anchor":
                append_invalid(steps, pred_action, "no_pending_anchor_local_image")
                state["invalid_actions"] += 1
                continue
            pred = seg_runner.predict(pending["image"], round1_query, miner_gt)
            quality = score_seg(pred, miner_gt)
            ok = is_miner_ok(quality, args)
            state["current_image"] = pending["image"]
            state["current_state"] = pending["state"]
            update_anchor(state, pred, quality, pending["image"], pending["state"])
            if panels is not None:
                anchor_panel = overlay_mask(pending["image"], miner_gt, (0, 180, 0), 0.35)
                anchor_panel = overlay_mask(anchor_panel, pred, (0, 120, 255), 0.55)
                panels.append((f"{action_id}: anchor local IoU={quality['iou']:.3f}", anchor_panel))
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": pending["state"],
                        "action": "segment_anchor_local_enhance",
                        "target": round1.get("target_category", "coal_miner"),
                        "roi_scale": pending["scale"],
                        "roi_xyxy": pending["roi"],
                        "quality": quality,
                        "improvement_over_current_best": float(quality["iou"] - pending["base_quality"]["iou"]),
                        "decision": "accept" if ok else "failed",
                    },
                    pred_action,
                )
            )

        elif action_id == "BLACKOUT_WITH_ANCHOR":
            if state.get("anchor") is None:
                append_invalid(steps, pred_action, "no_anchor_for_blackout")
                state["invalid_actions"] += 1
                continue
            state["current_image"] = focus_blackout(
                state["anchor_image"],
                state["anchor"],
                dilate=args.focus_dilate,
                background=0.0,
            )
            state["current_state"] = "focus_blackout"
            if panels is not None:
                panels.append((f"{action_id}: blackout with anchor", state["current_image"]))
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": state.get("anchor_state"),
                        "action": "focus_blackout",
                        "ref": "accepted_anchor",
                        "decision": "execute",
                    },
                    pred_action,
                )
            )

        elif action_id == "SEG_TARGET_WITH_REF":
            if state.get("anchor") is None:
                append_invalid(steps, pred_action, "no_anchor_ref_for_target")
                state["invalid_actions"] += 1
                continue
            pred = seg_runner.predict(
                state["current_image"],
                round2_query,
                helmet_gt,
                ref_mask=state["anchor"],
                ref_bbox=bbox_from_mask(state["anchor"]),
            )
            quality = score_seg(pred, helmet_gt)
            quality.update(helmet_geometry(pred, state["anchor"]))
            ok = is_target_ok(quality, args, anchor_mask=state["anchor"])
            update_best_target(state, pred, quality, state["current_image"], state["current_state"])
            if panels is not None:
                panels.append(
                    (
                        f"{action_id}: target with REF IoU={quality['iou']:.3f}",
                        target_overlay(state["current_image"], helmet_gt, pred),
                    )
                )
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": state["current_state"],
                        "action": "segment_helmet",
                        "target": round2.get("target_category", "mining_helmet"),
                        "quality": quality,
                        "decision": "accept" if ok else "failed",
                    },
                    pred_action,
                )
            )

        elif action_id == "LOCAL_ENHANCE_REF_TARGET":
            if state.get("last_target") is None:
                append_invalid(steps, pred_action, "no_ref_target_mask_for_local_enhance")
                state["invalid_actions"] += 1
                continue
            ok_local, local_stats, local_reason = is_local_enhance_candidate(state["last_target"], args)
            if not ok_local:
                steps.append(
                    add_policy_fields(
                        {
                            "step": len(steps) + 1,
                            "image_state": state["current_state"],
                            "action": "local_enhance_target_by_seg_after_ref",
                            "candidate_stats": local_stats,
                            "reason": local_reason,
                            "decision": "skip",
                        },
                        pred_action,
                    )
                )
                continue
            scale = ref_target_scales[0]
            local_image, roi = local_enhance_by_seg(
                state["last_target_image"],
                state["last_target"],
                args,
                strength=args.local_enhance_strength,
                roi_scale=scale,
            )
            if panels is not None:
                panels.append((f"{action_id}: ref target ROI {scale:.1f}x", local_image))
            state["pending_local"] = {
                "scope": "ref_target",
                "image": local_image,
                "state": f"{state['last_target_state']}_local_target_{scale:.1f}x",
                "roi": roi,
                "base_quality": state["last_target_quality"],
                "scale": scale,
            }
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": state["last_target_state"],
                        "action": "local_enhance_target_by_seg_after_ref",
                        "candidate_stats": local_stats,
                        "roi_scale": scale,
                        "roi_xyxy": roi,
                        "reason": local_reason,
                        "decision": "execute",
                    },
                    pred_action,
                )
            )

        elif action_id == "SEG_REF_TARGET_LOCAL":
            pending = state.get("pending_local")
            if not pending or pending.get("scope") != "ref_target" or state.get("anchor") is None:
                append_invalid(steps, pred_action, "no_pending_ref_target_local_image")
                state["invalid_actions"] += 1
                continue
            pred = seg_runner.predict(
                pending["image"],
                round2_query,
                helmet_gt,
                ref_mask=state["anchor"],
                ref_bbox=bbox_from_mask(state["anchor"]),
            )
            quality = score_seg(pred, helmet_gt)
            quality.update(helmet_geometry(pred, state["anchor"]))
            delta = quality["iou"] - pending["base_quality"]["iou"]
            ok = is_target_ok(quality, args, anchor_mask=state["anchor"])
            state["current_image"] = pending["image"]
            state["current_state"] = pending["state"]
            update_best_target(state, pred, quality, pending["image"], pending["state"])
            if panels is not None:
                panels.append(
                    (
                        f"{action_id}: ref local target IoU={quality['iou']:.3f}",
                        target_overlay(pending["image"], helmet_gt, pred),
                    )
                )
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": pending["state"],
                        "action": "segment_helmet_local_enhance",
                        "target": round2.get("target_category", "mining_helmet"),
                        "roi_scale": pending["scale"],
                        "roi_xyxy": pending["roi"],
                        "quality": quality,
                        "improvement_over_current_best": float(delta),
                        "decision": "accept" if ok else ("keep" if delta > 0 else "rollback"),
                    },
                    pred_action,
                )
            )

        elif action_id == "ROLLBACK_LOCAL_RESULT":
            pending = state.get("pending_local")
            if not pending:
                append_invalid(steps, pred_action, "no_pending_local_to_rollback")
                state["invalid_actions"] += 1
                continue
            base_state = pending.get("base_state", state.get("best_target_state", "pre_local"))
            state["current_image"] = pending.get("base_image", state.get("best_target_image", state["current_image"]))
            state["current_state"] = base_state
            state.pop("pending_local", None)
            if panels is not None:
                panels.append((f"{action_id}: rollback to {base_state}", state["current_image"]))
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": base_state,
                        "action": f"rollback_local_{pending.get('scope','unknown')}_enhance",
                        "reason": "policy rolled back local result",
                        "decision": "restore_pre_local_state",
                    },
                    pred_action,
                )
            )

        elif action_id == "ROLLBACK_TO_BEST_DIRECT":
            if state.get("best_target_quality") is None:
                append_invalid(steps, pred_action, "no_best_target_for_rollback")
                state["invalid_actions"] += 1
                continue
            state["final_target"] = state["best_target"]
            state["final_target_quality"] = state["best_target_quality"]
            state["current_image"] = state["best_target_image"]
            state["current_state"] = "rollback_accepted_state"
            if panels is not None:
                panels.append(
                    (
                        f"{action_id}: rollback best IoU={state['best_target_quality']['iou']:.3f}",
                        target_overlay(state["current_image"], helmet_gt, state["best_target"]),
                    )
                )
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": "rollback_accepted_state",
                        "action": "restore_direct_target_result",
                        "target": round2.get("target_category", "mining_helmet"),
                        "quality": state["best_target_quality"],
                        "decision": "restore_best_direct",
                    },
                    pred_action,
                )
            )

        elif action_id == "STOP_SUCCESS":
            _, quality, _, state_name = current_target(state)
            if quality is None:
                final_decision = "failed_no_target"
                final_path = "stop_success_without_target"
            else:
                state["final_target_quality"] = quality
                final_decision = "accepted" if is_target_ok(quality, args) else "failed_target"
                final_path = f"policy_stop_success_{state_name}"
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": state.get("current_state"),
                        "action": "stop_success",
                        "decision": final_decision,
                    },
                    pred_action,
                )
            )
            break

        elif action_id == "STOP_FAIL":
            final_decision = "failed_policy_stop"
            final_path = "policy_stop_fail"
            steps.append(
                add_policy_fields(
                    {
                        "step": len(steps) + 1,
                        "image_state": state.get("current_state"),
                        "action": "stop_failed_anchor",
                        "decision": "failed",
                    },
                    pred_action,
                )
            )
            break

    if final_decision is None:
        quality = state.get("final_target_quality") or state.get("best_target_quality")
        if quality is None:
            final_decision = "failed_no_target"
            final_path = "max_steps_no_target"
        else:
            state["final_target_quality"] = quality
            final_decision = "accepted" if is_target_ok(quality, args) else "failed_target"
            final_path = "max_steps_best_target"

    final_quality = state.get("final_target_quality") or state.get("best_target_quality")
    if final_quality is None:
        final_quality = {"iou": 0.0, **mask_stats(np.zeros_like(helmet_gt))}

    out_vis = None
    if getattr(args, "save_visuals", False) and overlay_dir is not None and idx < args.overlay_limit:
        viz_panels = []
        for panel in panels:
            if isinstance(panel, tuple) and len(panel) == 2:
                label, image = panel
                viz_panels.append(tile(image, label))
            else:
                viz_panels.append(panel)
        out_vis = maybe_save_flow_visual(args, overlay_dir, idx, ep, viz_panels)

    return {
        "episode_id": ep.get("episode_id"),
        "image_path": ep.get("image_path"),
        "severity": severity,
        "steps": steps,
        "final_decision": final_decision,
        "final_path": final_path,
        "invalid_actions": int(state.get("invalid_actions", 0)),
        "final_target_iou": float(final_quality.get("iou", 0.0)),
        "final_helmet_iou": float(final_quality.get("iou", 0.0)),
        "final_anchor_iou": float((state.get("anchor_quality") or {}).get("iou", 0.0)),
        "num_steps": len(steps),
        "policy_model": args.policy_model,
        "policy_adapter": args.policy_adapter,
        "visualization": out_vis,
    }


def summarize_policy(rows):
    base = summarize(rows)
    invalid = sum(int(r.get("invalid_actions", 0)) for r in rows)
    total_steps = sum(len(r.get("steps", [])) for r in rows)
    policy_actions = Counter()
    for row in rows:
        for step in row.get("steps", []):
            aid = step.get("policy_action_id")
            if aid:
                policy_actions[aid] += 1
    base.update(
        {
            "invalid_action_total": invalid,
            "invalid_action_rate_per_step": float(invalid / max(total_steps, 1)),
            "mean_steps": float(np.mean([len(r.get("steps", [])) for r in rows])) if rows else 0.0,
            "policy_action_counts": dict(policy_actions),
        }
    )
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Segmentation model path.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--policy-adapter", required=True)
    parser.add_argument("--policy-max-length", type=int, default=4096)
    parser.add_argument("--policy-max-new-tokens", type=int, default=32)
    parser.add_argument("--policy-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--overlay-limit", type=int, default=0)
    parser.add_argument("--save-visuals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--vision-pretrained",
        default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth",
    )
    parser.add_argument("--degradation", default="extreme", choices=["medium", "hard", "extreme", "mixed"])
    parser.add_argument("--degradation-config-json", default="")
    parser.add_argument("--miner-accept-iou", type=float, default=0.55)
    parser.add_argument("--miner-min-continue-iou", type=float, default=0.30)
    parser.add_argument("--helmet-accept-iou", type=float, default=0.55)
    parser.add_argument("--helmet-head-min-score", type=float, default=0.3)
    parser.add_argument("--low-brightness", type=float, default=0.28)
    parser.add_argument("--focus-dilate", type=int, default=21)
    parser.add_argument("--global-enhancer", default="")
    parser.add_argument("--global-enhancer-max-side", type=int, default=1024)
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
    args = parser.parse_args()
    args.degradation_config = json.loads(args.degradation_config_json) if args.degradation_config_json else None
    args.global_enhancer_model = load_task_enhancer_model(args.global_enhancer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = read_jsonl(args.jsonl)
    if args.max_items > 0:
        episodes = episodes[: args.max_items]

    seg_runner = ControllerRunner(args)
    policy_runner = PolicyRunner(args)
    overlay_dir = out_dir / "visualizations"
    if args.save_visuals and args.overlay_limit > 0:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, ep in enumerate(tqdm(episodes, desc="stage3_policy_v2_rollout")):
        rows.append(run_policy_episode(seg_runner, policy_runner, ep, args, idx, overlay_dir))

    summary = summarize_policy(rows)
    summary.update(
        {
            "seg_model": args.model,
            "policy_model": args.policy_model,
            "policy_adapter": args.policy_adapter,
            "jsonl": args.jsonl,
            "degradation": args.degradation,
            "max_items": args.max_items,
            "max_steps": args.max_steps,
            "local_roi_scales_target": parse_roi_scales(args.local_roi_scales_target),
            "local_roi_scales_anchor": parse_roi_scales(args.local_roi_scales_anchor),
            "local_roi_scales_ref_target": parse_roi_scales(args.local_roi_scales_ref_target),
        }
    )
    write_jsonl(out_dir / "stage3_policy_v2_rollout_traces.jsonl", rows)
    (out_dir / "stage3_policy_v2_rollout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

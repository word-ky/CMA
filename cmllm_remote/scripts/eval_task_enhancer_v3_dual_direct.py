#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_task_enhancer_v1 import load_enhancer, run_enhancer, save_overlay  # noqa: E402
from stage3_rule_controller_v3_seg_local_enhance import (  # noqa: E402
    ControllerRunner,
    degrade_parametric,
    image_quality,
    load_episode_assets,
    score_seg,
    target_overlay,
    tile,
    write_jsonl,
)
from train_task_enhancer_v3_dual_direct_loss import HELMET_QUERY, MINER_QUERY, TARGET15_B  # noqa: E402


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_u32(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def degrade_for_eval(image_rgb, cfg, idx, ep, mode):
    if mode == "stable":
        seed = stable_u32(f"{ep.get('episode_id', ep['image_path'])}:{cfg['id']}")
    else:
        seed = idx + 20260528
    return degrade_parametric(image_rgb, cfg, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--enhancer", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--degradation-config-json", default=json.dumps(TARGET15_B))
    parser.add_argument("--eval-seed-mode", choices=["index", "stable"], default="index")
    parser.add_argument("--helmet-accept-iou", type=float, default=0.5)
    parser.add_argument("--miner-accept-iou", type=float, default=0.5)
    parser.add_argument("--overlay-limit", type=int, default=8)
    parser.add_argument("--enhancer-max-side", type=int, default=1024)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument("--vision-pretrained", default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "visualizations"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.degradation_config_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhancer, enhancer_ckpt = load_enhancer(args.enhancer, device)
    runner = ControllerRunner(args)

    episodes = read_jsonl(args.jsonl)
    if args.max_items > 0:
        episodes = episodes[: args.max_items]

    rows = []
    for idx, ep in enumerate(tqdm(episodes, desc="eval_task_enhancer_v3_dual")):
        image_rgb, miner_gt, helmet_gt, _round1, _round2 = load_episode_assets(ep)
        degraded = degrade_for_eval(image_rgb, cfg, idx, ep, args.eval_seed_mode)
        enhanced = run_enhancer(enhancer, degraded, device, args.enhancer_max_side)

        pred_miner_degraded = runner.predict(degraded, MINER_QUERY, miner_gt)
        pred_miner_enhanced = runner.predict(enhanced, MINER_QUERY, miner_gt)
        pred_helmet_degraded = runner.predict(degraded, HELMET_QUERY, helmet_gt)
        pred_helmet_enhanced = runner.predict(enhanced, HELMET_QUERY, helmet_gt)

        q_md = score_seg(pred_miner_degraded, miner_gt)
        q_me = score_seg(pred_miner_enhanced, miner_gt)
        q_hd = score_seg(pred_helmet_degraded, helmet_gt)
        q_he = score_seg(pred_helmet_enhanced, helmet_gt)
        row = {
            "episode_id": ep.get("episode_id"),
            "image_path": ep.get("image_path"),
            "degradation_config": cfg,
            "miner_degraded_iou": q_md["iou"],
            "miner_enhanced_iou": q_me["iou"],
            "miner_delta_iou": float(q_me["iou"] - q_md["iou"]),
            "helmet_degraded_iou": q_hd["iou"],
            "helmet_enhanced_iou": q_he["iou"],
            "helmet_delta_iou": float(q_he["iou"] - q_hd["iou"]),
            "miner_degraded_accepted": bool(q_md["iou"] >= args.miner_accept_iou),
            "miner_enhanced_accepted": bool(q_me["iou"] >= args.miner_accept_iou),
            "helmet_degraded_accepted": bool(q_hd["iou"] >= args.helmet_accept_iou),
            "helmet_enhanced_accepted": bool(q_he["iou"] >= args.helmet_accept_iou),
            "degraded_quality": image_quality(degraded),
            "enhanced_quality": image_quality(enhanced),
        }
        rows.append(row)

        if idx < args.overlay_limit:
            clean_panel = image_rgb.copy()
            clean_panel[miner_gt > 0] = (0.45 * clean_panel[miner_gt > 0] + np.array([255, 64, 64]) * 0.55).astype(np.uint8)
            clean_panel[helmet_gt > 0] = (0.45 * clean_panel[helmet_gt > 0] + np.array([64, 255, 64]) * 0.55).astype(np.uint8)
            panels = [
                tile(clean_panel, "clean miner red / helmet green"),
                tile(target_overlay(degraded, miner_gt, pred_miner_degraded), f"degraded miner IoU={q_md['iou']:.3f}"),
                tile(target_overlay(enhanced, miner_gt, pred_miner_enhanced), f"enhanced miner IoU={q_me['iou']:.3f}"),
                tile(target_overlay(degraded, helmet_gt, pred_helmet_degraded), f"degraded helmet IoU={q_hd['iou']:.3f}"),
                tile(target_overlay(enhanced, helmet_gt, pred_helmet_enhanced), f"enhanced helmet IoU={q_he['iou']:.3f}"),
            ]
            save_overlay(overlay_dir / f"{idx:03d}_{ep.get('episode_id', idx)}.jpg", panels)

    write_jsonl(out_dir / "task_enhancer_v3_dual_eval.jsonl", rows)
    summary = {
        "model": args.model,
        "enhancer": args.enhancer,
        "enhancer_metrics": enhancer_ckpt.get("metrics", {}),
        "jsonl": args.jsonl,
        "degradation_config": cfg,
        "eval_seed_mode": args.eval_seed_mode,
        "miner_query": MINER_QUERY,
        "helmet_query": HELMET_QUERY,
        "num_episodes": len(rows),
        "miner_degraded_accepted": int(sum(r["miner_degraded_accepted"] for r in rows)),
        "miner_enhanced_accepted": int(sum(r["miner_enhanced_accepted"] for r in rows)),
        "helmet_degraded_accepted": int(sum(r["helmet_degraded_accepted"] for r in rows)),
        "helmet_enhanced_accepted": int(sum(r["helmet_enhanced_accepted"] for r in rows)),
        "mean_miner_degraded_iou": float(np.mean([r["miner_degraded_iou"] for r in rows])),
        "mean_miner_enhanced_iou": float(np.mean([r["miner_enhanced_iou"] for r in rows])),
        "mean_miner_delta_iou": float(np.mean([r["miner_delta_iou"] for r in rows])),
        "mean_helmet_degraded_iou": float(np.mean([r["helmet_degraded_iou"] for r in rows])),
        "mean_helmet_enhanced_iou": float(np.mean([r["helmet_enhanced_iou"] for r in rows])),
        "mean_helmet_delta_iou": float(np.mean([r["helmet_delta_iou"] for r in rows])),
        "miner_improved_count": int(sum(r["miner_delta_iou"] > 0 for r in rows)),
        "miner_worsened_count": int(sum(r["miner_delta_iou"] < 0 for r in rows)),
        "helmet_improved_count": int(sum(r["helmet_delta_iou"] > 0 for r in rows)),
        "helmet_worsened_count": int(sum(r["helmet_delta_iou"] < 0 for r in rows)),
    }
    (out_dir / "task_enhancer_v3_dual_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

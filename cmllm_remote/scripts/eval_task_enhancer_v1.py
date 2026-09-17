#!/usr/bin/env python3
import argparse
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

from stage3_rule_controller_v3_seg_local_enhance import (  # noqa: E402
    ControllerRunner,
    degrade_parametric,
    direct_query_for_round,
    image_quality,
    load_episode_assets,
    score_seg,
    target_overlay,
    tile,
    write_jsonl,
)
from train_task_enhancer_v1 import TaskEnhancerUNet  # noqa: E402


def load_enhancer(path, device):
    ckpt = torch.load(path, map_location="cpu")
    base = int(ckpt.get("args", {}).get("base_channels", 32))
    model = TaskEnhancerUNet(base)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_enhancer(model, image_rgb, device, max_side):
    h, w = image_rgb.shape[:2]
    work = image_rgb
    scale = 1.0
    if max_side > 0 and max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        work = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(work.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
        y = model(x).clamp(0, 1)
    out = (y[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if scale != 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_overlay(path, panels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.concatenate(panels, axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--enhancer", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--degradation-config-json", required=True)
    parser.add_argument("--mode", default="direct", choices=["direct", "gt-ref"])
    parser.add_argument("--helmet-accept-iou", type=float, default=0.55)
    parser.add_argument("--overlay-limit", type=int, default=5)
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
    for idx, ep in enumerate(tqdm(episodes, desc="eval_task_enhancer")):
        image_rgb, miner_gt, helmet_gt, _round1, round2 = load_episode_assets(ep)
        degraded = degrade_parametric(image_rgb, cfg, idx + 20260528)
        enhanced = run_enhancer(enhancer, degraded, device, args.enhancer_max_side)

        if args.mode == "direct":
            query = direct_query_for_round(round2)
            pred_degraded = runner.predict(degraded, query, helmet_gt)
            pred_enhanced = runner.predict(enhanced, query, helmet_gt)
        else:
            query = round2.get(
                "query",
                "Based on the miner segmented in the previous round, segment the helmet on his head.",
            )
            pred_degraded = runner.predict(degraded, query, helmet_gt, ref_mask=miner_gt, ref_bbox=None)
            pred_enhanced = runner.predict(enhanced, query, helmet_gt, ref_mask=miner_gt, ref_bbox=None)

        q_degraded = score_seg(pred_degraded, helmet_gt)
        q_enhanced = score_seg(pred_enhanced, helmet_gt)
        row = {
            "episode_id": ep.get("episode_id"),
            "image_path": ep.get("image_path"),
            "mode": args.mode,
            "degradation_config": cfg,
            "degraded_iou": q_degraded["iou"],
            "enhanced_iou": q_enhanced["iou"],
            "delta_iou": float(q_enhanced["iou"] - q_degraded["iou"]),
            "degraded_accepted": bool(q_degraded["iou"] >= args.helmet_accept_iou),
            "enhanced_accepted": bool(q_enhanced["iou"] >= args.helmet_accept_iou),
            "degraded_quality": image_quality(degraded),
            "enhanced_quality": image_quality(enhanced),
        }
        rows.append(row)

        if idx < args.overlay_limit:
            gt_panel = target_overlay(image_rgb, helmet_gt, np.zeros_like(helmet_gt, dtype=bool))
            panels = [
                tile(gt_panel, "clean helmet GT"),
                tile(target_overlay(degraded, helmet_gt, pred_degraded), f"degraded IoU={q_degraded['iou']:.3f}"),
                tile(target_overlay(enhanced, helmet_gt, pred_enhanced), f"enhanced IoU={q_enhanced['iou']:.3f}"),
            ]
            save_overlay(overlay_dir / f"{idx:03d}_{ep.get('episode_id', idx)}.jpg", panels)

    write_jsonl(out_dir / "task_enhancer_eval.jsonl", rows)
    degraded_acc = sum(r["degraded_accepted"] for r in rows)
    enhanced_acc = sum(r["enhanced_accepted"] for r in rows)
    summary = {
        "model": args.model,
        "enhancer": args.enhancer,
        "enhancer_metrics": enhancer_ckpt.get("metrics", {}),
        "jsonl": args.jsonl,
        "mode": args.mode,
        "degradation_config": cfg,
        "num_episodes": len(rows),
        "degraded_accepted": degraded_acc,
        "enhanced_accepted": enhanced_acc,
        "mean_degraded_iou": float(np.mean([r["degraded_iou"] for r in rows])),
        "mean_enhanced_iou": float(np.mean([r["enhanced_iou"] for r in rows])),
        "mean_delta_iou": float(np.mean([r["delta_iou"] for r in rows])),
        "improved_count": int(sum(r["delta_iou"] > 0 for r in rows)),
        "worsened_count": int(sum(r["delta_iou"] < 0 for r in rows)),
    }
    (out_dir / "task_enhancer_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

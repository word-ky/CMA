#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage3_rule_controller_v3_seg_local_enhance import (  # noqa: E402
    ControllerRunner,
    direct_query_for_round,
    image_quality,
    iou,
    load_episode_assets,
    read_jsonl,
    save_flow_visual,
    stable_seed,
    target_overlay,
    tile,
    write_jsonl,
)


def degrade_parametric(image_rgb, cfg, seed):
    rng = np.random.default_rng(seed)
    x = image_rgb.astype(np.float32) / 255.0
    gamma = float(cfg.get("gamma", 1.0))
    scale = float(cfg.get("scale", 1.0))
    contrast = float(cfg.get("contrast", 1.0))
    sigma = float(cfg.get("noise_sigma", 0.0))
    blur = float(cfg.get("blur", 0.0))

    x = np.power(np.clip(x, 0.0, 1.0), gamma)
    x = (x - 0.5) * contrast + 0.5
    x = x * scale
    if sigma > 0:
        noise = rng.normal(0.0, sigma / 255.0, x.shape).astype(np.float32)
        x = x + noise
    x = np.clip(x, 0.0, 1.0)
    out = (x * 255.0).astype(np.uint8)
    if blur > 0:
        k = max(3, int(round(blur * 2.0 + 1.0)))
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), blur)
    return out


def coarse_configs():
    # Ordered from mild to harder. The target is a no-enhancer direct split near 20/30 accepted.
    raw = [
        ("d20_c01_mild_a", 1.15, 0.82, 0.90, 4.0, 0.0),
        ("d20_c02_mild_b", 1.25, 0.78, 0.88, 6.0, 0.0),
        ("d20_c03_mild_c", 1.35, 0.74, 0.85, 8.0, 0.0),
        ("d20_c04_mid_a", 1.45, 0.70, 0.82, 10.0, 0.0),
        ("d20_c05_mid_b", 1.55, 0.66, 0.80, 12.0, 0.2),
        ("d20_c06_mid_c", 1.65, 0.62, 0.78, 14.0, 0.3),
        ("d20_c07_hard_a", 1.75, 0.58, 0.76, 16.0, 0.4),
        ("d20_c08_hard_b", 1.90, 0.54, 0.74, 18.0, 0.5),
        ("d20_c09_hard_c", 2.05, 0.50, 0.72, 20.0, 0.6),
        ("d20_c10_near_old_medium", 2.20, 0.45, 1.00, 20.0, 0.0),
    ]
    return [
        {
            "id": name,
            "gamma": gamma,
            "scale": scale,
            "contrast": contrast,
            "noise_sigma": sigma,
            "blur": blur,
        }
        for name, gamma, scale, contrast, sigma, blur in raw
    ]


def refine_configs(center_cfg):
    gamma = float(center_cfg["gamma"])
    scale = float(center_cfg["scale"])
    contrast = float(center_cfg["contrast"])
    sigma = float(center_cfg["noise_sigma"])
    blur = float(center_cfg["blur"])
    out = []
    variants = [
        ("r01_easier_bright", gamma - 0.08, scale + 0.04, contrast + 0.02, sigma - 2.0, blur),
        ("r02_easier_noise", gamma - 0.04, scale + 0.02, contrast, sigma - 4.0, max(0.0, blur - 0.1)),
        ("r03_center", gamma, scale, contrast, sigma, blur),
        ("r04_harder_noise", gamma + 0.04, scale - 0.02, contrast, sigma + 3.0, blur + 0.1),
        ("r05_harder_dark", gamma + 0.08, scale - 0.04, contrast - 0.02, sigma + 2.0, blur + 0.1),
    ]
    base_id = center_cfg["id"]
    for suffix, g, s, c, n, b in variants:
        out.append(
            {
                "id": f"{base_id}_{suffix}",
                "gamma": round(max(1.0, g), 4),
                "scale": round(max(0.10, s), 4),
                "contrast": round(max(0.10, c), 4),
                "noise_sigma": round(max(0.0, n), 4),
                "blur": round(max(0.0, b), 4),
            }
        )
    return out


def load_configs(args):
    if args.config_json:
        return json.loads(args.config_json)
    if args.config_jsonl:
        rows = read_jsonl(args.config_jsonl)
        return rows
    if args.preset == "coarse":
        return coarse_configs()
    raise ValueError(f"unknown preset: {args.preset}")


def save_visual(args, out_dir, cfg, idx, image_rgb, degraded, helmet_gt, pred):
    if args.overlay_limit <= 0 or idx >= args.overlay_limit:
        return None
    panels = [
        tile(image_rgb, "clean image"),
        tile(degraded, f"degraded {cfg['id']}"),
        tile(target_overlay(degraded, helmet_gt, pred), f"direct target IoU={iou(pred, helmet_gt):.3f}"),
    ]
    path = out_dir / "visualizations" / cfg["id"] / f"{idx:05d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_flow_visual(path, panels)
    return str(path)


def eval_config(runner, episodes, cfg, args, out_dir):
    rows = []
    accepted = 0
    ious = []
    brightness = []
    dark_ratio = []
    for idx, ep in enumerate(tqdm(episodes, desc=cfg["id"])):
        image_rgb, miner_gt, helmet_gt, _round1, round2 = load_episode_assets(ep)
        seed = stable_seed(f"{ep.get('episode_id', ep['image_path'])}:{cfg['id']}")
        degraded = degrade_parametric(image_rgb, cfg, seed)
        if args.mode == "direct":
            query = direct_query_for_round(round2)
            pred = runner.predict(degraded, query, helmet_gt)
        elif args.mode == "gt-ref":
            query = round2.get(
                "query",
                "Based on the miner segmented in the previous round, segment the helmet on his head.",
            )
            pred = runner.predict(
                degraded,
                query,
                helmet_gt,
                ref_mask=miner_gt,
                ref_bbox=None,
            )
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        score = iou(pred, helmet_gt)
        ok = score >= args.helmet_accept_iou
        q = image_quality(degraded)
        accepted += int(ok)
        ious.append(float(score))
        brightness.append(q["brightness"])
        dark_ratio.append(q["dark_pixel_ratio"])
        rows.append(
            {
                "idx": idx,
                "episode_id": ep.get("episode_id"),
                "image_path": ep["image_path"],
                "config_id": cfg["id"],
                "mode": args.mode,
                "iou": float(score),
                "accepted": bool(ok),
                "image_quality": q,
                "visualization": save_visual(args, out_dir, cfg, idx, image_rgb, degraded, helmet_gt, pred),
            }
        )
    summary = {
        "config": cfg,
        "mode": args.mode,
        "num_episodes": len(episodes),
        "accepted": accepted,
        "accept_rate": float(accepted / max(len(episodes), 1)),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "mean_brightness": float(np.mean(brightness)) if brightness else 0.0,
        "mean_dark_pixel_ratio": float(np.mean(dark_ratio)) if dark_ratio else 0.0,
        "target_accepted": args.target_accepted,
        "distance_to_target": int(abs(accepted - args.target_accepted)),
    }
    write_jsonl(out_dir / f"results_{cfg['id']}.jsonl", rows)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--preset", default="coarse", choices=["coarse"])
    parser.add_argument("--mode", default="direct", choices=["direct", "gt-ref"])
    parser.add_argument("--config-json")
    parser.add_argument("--config-jsonl")
    parser.add_argument("--target-accepted", type=int, default=20)
    parser.add_argument("--helmet-accept-iou", type=float, default=0.55)
    parser.add_argument("--overlay-limit", type=int, default=0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--vision-pretrained",
        default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = read_jsonl(args.jsonl)
    if args.max_items > 0:
        episodes = episodes[: args.max_items]
    configs = load_configs(args)

    runner = ControllerRunner(args)
    summaries = []
    for cfg in configs:
        summary = eval_config(runner, episodes, cfg, args, out_dir)
        summaries.append(summary)
        with (out_dir / "sweep_summary.partial.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    summaries = sorted(summaries, key=lambda x: (x["distance_to_target"], -x["mean_iou"]))
    (out_dir / "sweep_summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "jsonl": args.jsonl,
                "max_items": len(episodes),
                "helmet_accept_iou": args.helmet_accept_iou,
                "mode": args.mode,
                "target_accepted": args.target_accepted,
                "best": summaries[0] if summaries else None,
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("BEST", json.dumps(summaries[0] if summaries else {}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

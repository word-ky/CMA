from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


QUESTION_TEMPLATES = [
    "请分割图中需要关注运行状态和防护状态的钻机区域。",
    "请找出并分割煤矿钻孔作业场景中的钻机设备区域。",
    "请分割图像中与钻孔作业安全监测相关的设备区域。",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def largest_contour_points(mask_path: Path, max_points: int = 80) -> list[list[int]] | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.004 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) > max_points:
        idx = np.linspace(0, len(approx) - 1, max_points).astype(int)
        approx = approx[idx]
    if len(approx) < 3:
        x, y, w, h = cv2.boundingRect(contour)
        approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return [[int(x), int(y)] for x, y in approx.tolist()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-masks-jsonl", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    rows = read_jsonl(args.sam_masks_jsonl)
    split_dir = args.out_root / "reason_seg" / "DsDPM66" / args.split
    explanatory_dir = args.out_root / "reason_seg" / "DsDPM66" / "explanatory"
    split_dir.mkdir(parents=True, exist_ok=True)
    explanatory_dir.mkdir(parents=True, exist_ok=True)

    explanatory = []
    kept = 0
    for idx, row in enumerate(rows):
        src_img = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        points = largest_contour_points(mask_path)
        if points is None or not src_img.exists():
            continue
        stem = f"{idx:05d}_{row['sample_id']}"
        dst_img = split_dir / f"{stem}.jpg"
        dst_json = split_dir / f"{stem}.json"
        shutil.copy2(src_img, dst_img)
        prompt = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]
        anno = {
            "text": [prompt],
            "is_sentence": True,
            "shapes": [
                {
                    "label": "drill_rig",
                    "points": points,
                }
            ],
            "source": {
                "sample_id": row["sample_id"],
                "mask_path": row["mask_path"],
                "sam_score": row["score"],
                "generator": "bbox_prompted_sam_pseudo_label",
            },
        }
        dst_json.write_text(json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8")
        explanatory.append(
            {
                "image": dst_img.name,
                "query": prompt,
                "outputs": "目标区域是钻机设备，其运行状态、作业区域占用和防护状态与煤矿钻孔作业安全监测相关。",
            }
        )
        kept += 1

    (explanatory_dir / "train.json").write_text(
        json.dumps(explanatory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "out_root": str(args.out_root),
        "split": args.split,
        "samples": kept,
        "format": "LISA ReasonSeg-compatible",
    }
    (args.out_root / "reason_seg" / "DsDPM66" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


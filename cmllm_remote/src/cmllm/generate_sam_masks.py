from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from segment_anything import SamPredictor, sam_model_registry


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_box_to_xyxy(box: list[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([x1 * width, y1 * height, x2 * width, y2 * height], dtype=np.float32)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def overlay_mask(image: Image.Image, mask: np.ndarray, box: list[int] | None) -> Image.Image:
    rgba = image.convert("RGBA")
    color = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    color[mask > 0] = [0, 180, 160, 105]
    mask_img = Image.fromarray(color, "RGBA")
    out = Image.alpha_composite(rgba, mask_img).convert("RGB")
    draw = ImageDraw.Draw(out)
    if box:
        draw.rectangle(box, outline=(255, 190, 0), width=3)
    return out


def make_contact_sheet(rows: list[dict[str, Any]], out_path: Path, limit: int = 8) -> None:
    cards = []
    font = ImageFont.load_default()
    for row in rows[:limit]:
        image = Image.open(row["image_path"]).convert("RGB")
        mask = np.array(Image.open(row["mask_path"]).convert("L")) > 0
        overlay = overlay_mask(image, mask, row.get("mask_bbox"))
        overlay.thumbnail((360, 220))
        card = Image.new("RGB", (390, 290), "white")
        card.paste(overlay, (15, 15))
        draw = ImageDraw.Draw(card)
        draw.text((15, 242), row["sample_id"], fill=(30, 40, 50), font=font)
        draw.text((15, 262), f"{row['object_name']} score={row['score']:.3f}", fill=(0, 100, 90), font=font)
        cards.append(card)

    if not cards:
        return
    cols = 2
    rows_n = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 390, rows_n * 290), (245, 246, 248))
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % cols) * 390, (i // cols) * 290))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sam-checkpoint", default="/home/wjq/rex-anchor/ckpts/sam/sam_vit_b.pth", type=Path)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    rows = read_jsonl(args.subset_manifest)
    if args.limit:
        rows = rows[: args.limit]

    mask_dir = args.out_dir / "masks"
    overlay_dir = args.out_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for row in rows:
        image_path = Path(row["image_path"])
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        predictor.set_image(rgb)

        for obj_idx, obj in enumerate(row.get("objects", [])):
            box = obj.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                continue
            xyxy = norm_box_to_xyxy(box, width, height)
            masks, scores, _ = predictor.predict(box=xyxy, multimask_output=False)
            mask = masks[0].astype(np.uint8) * 255
            score = float(scores[0])
            mask_path = mask_dir / f"{row['sample_id']}_{obj_idx}.png"
            overlay_path = overlay_dir / f"{row['sample_id']}_{obj_idx}.jpg"
            Image.fromarray(mask).save(mask_path)
            bbox = mask_bbox(mask)
            overlay = overlay_mask(Image.fromarray(rgb), mask > 0, bbox)
            overlay.save(overlay_path, quality=92)
            outputs.append(
                {
                    "sample_id": row["sample_id"],
                    "image_path": str(image_path),
                    "object_index": obj_idx,
                    "object_name": obj.get("name", "unknown"),
                    "prompt_bbox_norm": box,
                    "prompt_bbox_xyxy": [float(x) for x in xyxy.tolist()],
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "mask_bbox": bbox,
                    "mask_area": int((mask > 0).sum()),
                    "score": score,
                }
            )

    write_jsonl(args.out_dir / "sam_masks.jsonl", outputs)
    summary = {
        "subset_manifest": str(args.subset_manifest),
        "sam_checkpoint": str(args.sam_checkpoint),
        "model_type": args.model_type,
        "device": device,
        "images_seen": len(rows),
        "masks_generated": len(outputs),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(outputs, args.out_dir / "sam_segmentation_samples.jpg", limit=8)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


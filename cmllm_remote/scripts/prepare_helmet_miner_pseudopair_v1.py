from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/home/wjq/cmllm")
OUT_ROOT = ROOT / "outputs/mr_data/helmet_miner_pseudopair_v1"
COAL_ZIP = ROOT / "data/raw/dsdpm66/zips/coal_miner.zip"
HELMET_CLEAN = ROOT / "outputs/qc/dsdpm66_mining_helmet_fullinst21618_deepseek_keyed_20260509_114630/clean_manifest.jsonl"
HELMET_SAM = ROOT / "outputs/segmentation/mining_helmet_sam_fullinst21618_keyed_20260509_114630/sam_masks.jsonl"
SAM_VIT_B = Path("/home/wjq/rex-anchor/ckpts/sam/sam_vit_b.pth")
SAM_VIT_H = ROOT / "models/sam/sam_vit_h_4b8939.pth"


ROUND1_QUERIES = [
    "Segment the coal miner who is wearing the target mining helmet in this underground mining image.",
    "Find and segment the visible coal miner associated with the helmet to be inspected.",
    "Segment the miner whose head protection will be referenced in the next step.",
    "Identify the miner wearing a mining helmet and output the miner mask.",
]

ROUND2_QUERIES = [
    "Based on the miner mask from the previous round, segment only the mining helmet worn by that miner.",
    "Use the previously segmented miner as reference and segment the helmet on that miner's head.",
    "Refer to the prior miner mask and output only the mining helmet belonging to that miner.",
    "Do not segment helmets on other miners; segment the helmet worn by the miner selected in the previous round.",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts: Any, n: int = 16) -> str:
    raw = "||".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:n]


def safe_stem(value: str) -> str:
    stem = Path(value).stem
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def clamp_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def bbox_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def bbox_intersection(a: list[float], b: list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: list[float], b: list[float]) -> float:
    inter = bbox_intersection(a, b)
    denom = bbox_area(a) + bbox_area(b) - inter
    return float(inter / denom) if denom > 0 else 0.0


def bbox_center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])


def expand_box(box: list[float], ratio: float, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = x2 - x1
    bh = y2 - y1
    return clamp_box([x1 - ratio * bw, y1 - ratio * bh, x2 + ratio * bw, y2 + ratio * bh], width, height)


def top_region(box: list[float], frac: float) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, x2, y1 + (y2 - y1) * frac]


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_path_to_array(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def count_components(mask: np.ndarray) -> int:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.sum(areas >= max(16, int(mask.sum() * 0.005))))


def yolo_label_line(box_xywh: list[float], width: int, height: int) -> str:
    x, y, w, h = [float(v) for v in box_xywh]
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    wn = w / width
    hn = h / height
    vals = [0, xc, yc, wn, hn]
    return "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(vals[0], vals[1], vals[2], vals[3], vals[4])


def find_coco_members(zf: zipfile.ZipFile) -> dict[str, str]:
    members = zf.namelist()
    out: dict[str, str] = {}
    for split in ("train", "val"):
        candidates = [
            m for m in members
            if m.lower().endswith(".json") and f"/{split}/" in f"/{m.lower()}/"
        ]
        if not candidates:
            candidates = [m for m in members if m.lower().endswith(f"{split}.json")]
        if not candidates:
            candidates = [m for m in members if m.lower().endswith(".json") and split in m.lower()]
        if not candidates:
            raise FileNotFoundError(f"cannot find {split} COCO annotation inside {zf.filename}")
        out[split] = sorted(candidates, key=len)[0]
    return out


def resolve_image_member(
    file_name: str,
    anno_member: str,
    all_members: set[str],
    basename_index: dict[str, list[str]],
) -> str:
    if file_name in all_members:
        return file_name
    anno_dir = str(Path(anno_member).parent).replace("\\", "/")
    candidates = [
        f"{anno_dir}/{file_name}",
        f"{anno_dir}/images/{file_name}",
        f"{Path(anno_dir).parent.as_posix()}/images/{file_name}",
    ]
    for candidate in candidates:
        candidate = candidate.replace("\\", "/")
        if candidate in all_members:
            return candidate
    base = Path(file_name).name
    matches = basename_index.get(base, [])
    if not matches:
        raise FileNotFoundError(f"cannot resolve image member for {file_name}")
    return sorted(matches, key=len)[0]


def cmd_prepare_yolo(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    yolo_root = out_dir / "yolo_miner_detector"
    if args.force and yolo_root.exists():
        shutil.rmtree(yolo_root)
    stats: dict[str, Any] = {
        "source_zip": str(args.coal_zip),
        "target": str(yolo_root),
        "splits": {},
    }
    with zipfile.ZipFile(args.coal_zip) as zf:
        all_members = set(zf.namelist())
        basename_index: dict[str, list[str]] = defaultdict(list)
        for member in all_members:
            basename_index[Path(member).name].append(member)
        coco_members = find_coco_members(zf)
        for split, anno_member in coco_members.items():
            coco = json.loads(zf.read(anno_member).decode("utf-8"))
            images = {int(img["id"]): img for img in coco["images"]}
            anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for ann in coco["annotations"]:
                if ann.get("bbox") and float(ann["bbox"][2]) > 1 and float(ann["bbox"][3]) > 1:
                    anns_by_image[int(ann["image_id"])].append(ann)
            image_ids = sorted(anns_by_image)
            if args.limit and args.limit > 0:
                image_ids = image_ids[: args.limit]
            for idx, image_id in enumerate(image_ids, 1):
                image = images[image_id]
                file_name = image["file_name"]
                width = int(image["width"])
                height = int(image["height"])
                suffix = Path(file_name).suffix or ".jpg"
                dst_name = f"{safe_stem(file_name)}{suffix}"
                image_dst = yolo_root / "images" / split / dst_name
                label_dst = yolo_root / "labels" / split / f"{Path(dst_name).stem}.txt"
                image_dst.parent.mkdir(parents=True, exist_ok=True)
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                if not image_dst.exists() or args.force_images:
                    member = resolve_image_member(file_name, anno_member, all_members, basename_index)
                    with zf.open(member) as src, image_dst.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                lines = [yolo_label_line(ann["bbox"], width, height) for ann in anns_by_image[image_id]]
                label_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
                if idx % 1000 == 0:
                    print(f"prepare_yolo {split} {idx}/{len(image_ids)}", flush=True)
            stats["splits"][split] = {
                "annotation_member": anno_member,
                "images": len(image_ids),
                "annotations": int(sum(len(anns_by_image[i]) for i in image_ids)),
            }
    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {yolo_root}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: coal_miner",
            "",
        ]),
        encoding="utf-8",
    )
    stats["data_yaml"] = str(data_yaml)
    write_json(out_dir / "reports/detector_stats.json", {"prepare_yolo": stats})
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


def load_helmet_images(clean_manifest: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = read_jsonl(clean_manifest)
    by_member: dict[str, dict[str, Any]] = {}
    for row in rows:
        member = str(row.get("image_member") or row.get("image_path"))
        if member not in by_member:
            by_member[member] = {
                "image_member": member,
                "image_path": row["image_path"],
                "split_source": row.get("split_source"),
                "image_size": row.get("image_size"),
                "sample_ids": [],
            }
        by_member[member]["sample_ids"].append(row["sample_id"])
    items = [by_member[k] for k in sorted(by_member)]
    if limit and limit > 0:
        return items[:limit]
    return items


def cmd_infer_miners(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ultralytics is required for infer-miners") from exc

    out_path = args.out_dir / "raw_miner_detections_on_helmet_images.jsonl"
    existing = {row["image_member"] for row in read_jsonl(out_path)}
    items = load_helmet_images(args.helmet_clean, args.limit)
    model = YOLO(str(args.weights))
    total_boxes = 0
    processed = len(existing)
    for item in items:
        if item["image_member"] in existing and not args.force:
            continue
        image_path = item["image_path"]
        result = model.predict(
            source=image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        width, height = Image.open(image_path).size
        preds: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            order = np.argsort(-confs)
            for rank, idx in enumerate(order.tolist()):
                box = clamp_box([float(v) for v in xyxy[idx].tolist()], width, height)
                if bbox_area(box) < args.min_box_area:
                    continue
                preds.append({
                    "pseudo_miner_id": f"pm_{stable_id(item['image_member'], rank)}",
                    "rank": int(rank),
                    "bbox_xyxy": box,
                    "det_conf": float(confs[idx]),
                    "det_conf_norm": float(max(0.0, min(1.0, confs[idx]))),
                    "class_id": int(classes[idx]),
                })
        row = {
            "image_member": item["image_member"],
            "image_path": image_path,
            "split_source": item.get("split_source"),
            "image_size": [width, height],
            "sample_ids": item.get("sample_ids", []),
            "pred_miners": preds,
        }
        append_jsonl(out_path, [row])
        total_boxes += len(preds)
        processed += 1
        if processed % 250 == 0:
            print(f"infer_miners processed={processed} boxes={total_boxes}", flush=True)
    rows = read_jsonl(out_path)
    counts = Counter(len(row.get("pred_miners", [])) for row in rows)
    report = {
        "stage": "infer_miners",
        "weights": str(args.weights),
        "images": len(rows),
        "boxes": int(sum(len(row.get("pred_miners", [])) for row in rows)),
        "box_count_histogram": dict(sorted(counts.items())),
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "imgsz": args.imgsz,
    }
    stats_path = args.out_dir / "reports/detector_stats.json"
    old = {}
    if stats_path.exists():
        old = json.loads(stats_path.read_text(encoding="utf-8"))
    old["infer_miners"] = report
    write_json(stats_path, old)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def load_sam_predictor(checkpoint: Path, model_type: str, device: str):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("segment_anything is required for sam-miners") from exc
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def miner_quality(det_box: list[float], mask: np.ndarray, score: float) -> dict[str, Any]:
    h, w = mask.shape[:2]
    mbox = mask_bbox(mask)
    area = int(mask.sum())
    box_area = max(1.0, bbox_area(det_box))
    fill_ratio = float(area / box_area)
    mask_box_iou = bbox_iou(det_box, [float(v) for v in mbox]) if mbox else 0.0
    components = count_components(mask.astype(np.uint8))
    touches_edge = False
    if mbox:
        touches_edge = mbox[0] <= 1 or mbox[1] <= 1 or mbox[2] >= w - 1 or mbox[3] >= h - 1
    component_score = 1.0 if components <= 1 else max(0.0, 1.0 - 0.15 * (components - 1))
    fill_score = 1.0 if 0.15 <= fill_ratio <= 1.10 else max(0.0, 1.0 - abs(fill_ratio - 0.55))
    quality = 0.40 * float(score) + 0.30 * mask_box_iou + 0.20 * component_score + 0.10 * fill_score
    return {
        "mask_bbox": mbox,
        "mask_area": area,
        "mask_area_ratio": float(area / max(1, w * h)),
        "mask_fill_ratio_in_box": fill_ratio,
        "mask_box_iou": float(mask_box_iou),
        "sam_score": float(score),
        "connected_components": int(components),
        "touches_edge": bool(touches_edge),
        "sam_quality": float(max(0.0, min(1.0, quality))),
    }


def cmd_sam_miners(args: argparse.Namespace) -> int:
    detections = read_jsonl(args.detections)
    if args.limit and args.limit > 0:
        detections = detections[: args.limit]
    out_path = args.out_dir / "pseudo_miners_on_helmet_images.jsonl"
    existing = {row["pseudo_miner_id"] for row in read_jsonl(out_path)}
    mask_dir = args.out_dir / "pseudo_miners"
    predictor = load_sam_predictor(args.sam_checkpoint, args.model_type, args.device)
    written = 0
    for img_idx, row in enumerate(detections, 1):
        preds = row.get("pred_miners", [])
        todo = [p for p in preds if p["pseudo_miner_id"] not in existing or args.force]
        if not todo:
            continue
        image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(row["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        new_rows: list[dict[str, Any]] = []
        image_key = safe_stem(row["image_member"])
        for pred in todo:
            box = np.array(pred["bbox_xyxy"], dtype=np.float32)
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            best = int(np.argmax(scores))
            mask = masks[best].astype(np.uint8)
            mask_out = mask_dir / f"{image_key}_{pred['pseudo_miner_id']}.png"
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask * 255).save(mask_out)
            quality = miner_quality(pred["bbox_xyxy"], mask > 0, float(scores[best]))
            new_rows.append({
                "image_member": row["image_member"],
                "image_path": row["image_path"],
                "split_source": row.get("split_source"),
                "image_size": row.get("image_size"),
                "pseudo_miner_id": pred["pseudo_miner_id"],
                "bbox_xyxy": pred["bbox_xyxy"],
                "det_conf": pred["det_conf"],
                "det_conf_norm": pred.get("det_conf_norm", pred["det_conf"]),
                "mask_path": str(mask_out),
                **quality,
            })
        append_jsonl(out_path, new_rows)
        written += len(new_rows)
        if img_idx % 100 == 0:
            print(f"sam_miners images={img_idx}/{len(detections)} written={written}", flush=True)
    rows = read_jsonl(out_path)
    report = {
        "stage": "sam_miners",
        "pseudo_miners": len(rows),
        "sam_checkpoint": str(args.sam_checkpoint),
        "model_type": args.model_type,
        "mean_sam_quality": float(np.mean([r.get("sam_quality", 0.0) for r in rows])) if rows else 0.0,
        "mean_mask_box_iou": float(np.mean([r.get("mask_box_iou", 0.0) for r in rows])) if rows else 0.0,
    }
    stats_path = args.out_dir / "reports/detector_stats.json"
    old = {}
    if stats_path.exists():
        old = json.loads(stats_path.read_text(encoding="utf-8"))
    old["sam_miners"] = report
    write_json(stats_path, old)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def helmet_records(clean_path: Path, sam_path: Path) -> list[dict[str, Any]]:
    clean_rows = read_jsonl(clean_path)
    sam_by_id = {row["sample_id"]: row for row in read_jsonl(sam_path)}
    records: list[dict[str, Any]] = []
    for row in clean_rows:
        sample_id = row["sample_id"]
        sam = sam_by_id.get(sample_id)
        if not sam:
            continue
        obj = row.get("objects", [{}])[0]
        bbox_xywh = obj.get("bbox_xywh_abs")
        if bbox_xywh is None:
            bbox_xyxy = sam.get("prompt_bbox_xyxy")
        else:
            bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        size = row.get("image_size") or sam.get("image_size")
        if isinstance(size, dict):
            width = int(size.get("width"))
            height = int(size.get("height"))
        else:
            width, height = [int(v) for v in size]
        records.append({
            "helmet_id": sample_id,
            "image_member": row.get("image_member") or row.get("image_path"),
            "image_path": row["image_path"],
            "split_source": row.get("split_source"),
            "scene_prefix": row.get("scene_prefix"),
            "frame_index": row.get("frame_index"),
            "bbox_xyxy": clamp_box([float(v) for v in bbox_xyxy], width, height),
            "mask_path": sam["mask_path"],
            "sam_score": sam.get("score"),
            "image_size": [width, height],
            "target_annotation_source": "dsdpm66_bbox_plus_sam1_mask",
            "target_bbox_source": "dsdpm66_real_bbox",
            "target_mask_source": "sam1_from_real_bbox",
        })
    return records


def size_score(helmet_area: float, miner_area: float) -> float:
    if helmet_area <= 0 or miner_area <= 0:
        return 0.0
    ratio = helmet_area / miner_area
    if 0.010 <= ratio <= 0.120:
        return 1.0
    if ratio < 0.010:
        return max(0.0, ratio / 0.010)
    if ratio <= 0.250:
        return max(0.0, 1.0 - (ratio - 0.120) / 0.130)
    return 0.0


def head_score(helmet_center: tuple[float, float], miner_box: list[float]) -> tuple[float, bool, bool, bool]:
    in_miner = point_in_box(helmet_center, miner_box)
    in_strict = point_in_box(helmet_center, top_region(miner_box, 0.45))
    in_loose = point_in_box(helmet_center, top_region(miner_box, 0.60))
    if in_strict:
        return 1.0, in_miner, in_strict, in_loose
    if in_loose:
        return 0.7, in_miner, in_strict, in_loose
    if in_miner:
        return 0.3, in_miner, in_strict, in_loose
    return 0.0, in_miner, in_strict, in_loose


def head_overlap_from_mask(helmet_mask_path: str, head_box: list[float], image_size: list[int]) -> float:
    mask = mask_path_to_array(helmet_mask_path)
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(head_box, w, h)]
    denom = int(mask.sum())
    if denom <= 0:
        return 0.0
    roi = np.zeros(mask.shape, dtype=bool)
    roi[max(0, y1): min(h, y2), max(0, x1): min(w, x2)] = True
    return float(np.logical_and(mask, roi).sum() / denom)


def build_candidate(helmet: dict[str, Any], miner: dict[str, Any], top_gap: float | None = None) -> dict[str, Any]:
    width, height = [int(v) for v in helmet["image_size"]]
    helmet_box = helmet["bbox_xyxy"]
    miner_box = miner["bbox_xyxy"]
    center = bbox_center(helmet_box)
    center_score, center_in_miner, center_in_strict, center_in_loose = head_score(center, miner_box)
    expanded = expand_box(miner_box, 0.10, width, height)
    center_in_expanded = point_in_box(center, expanded)
    strict_head = top_region(miner_box, 0.45)
    loose_head = top_region(miner_box, 0.60)
    helmet_head_overlap = head_overlap_from_mask(helmet["mask_path"], loose_head, helmet["image_size"])
    containment = bbox_intersection(helmet_box, miner_box) / max(1.0, bbox_area(helmet_box))
    ssize = size_score(bbox_area(helmet_box), bbox_area(miner_box))
    sam_quality = float(miner.get("sam_quality", 0.0))
    det_conf_norm = float(max(0.0, min(1.0, miner.get("det_conf_norm", miner.get("det_conf", 0.0)))))
    pair_score = (
        0.15 * det_conf_norm
        + 0.35 * center_score
        + 0.20 * helmet_head_overlap
        + 0.15 * containment
        + 0.10 * ssize
        + 0.05 * sam_quality
    )
    if pair_score >= 0.75 and center_in_miner:
        rule_decision = "high"
    elif pair_score >= 0.60 and center_in_expanded:
        rule_decision = "medium"
    else:
        rule_decision = "low"
    return {
        "pair_id": f"pair_{stable_id(helmet['helmet_id'], miner['pseudo_miner_id'])}",
        "image_member": helmet["image_member"],
        "image_path": helmet["image_path"],
        "split_source": helmet.get("split_source"),
        "image_size": helmet["image_size"],
        "helmet_id": helmet["helmet_id"],
        "pseudo_miner_id": miner["pseudo_miner_id"],
        "helmet_bbox_xyxy": helmet_box,
        "helmet_mask_path": helmet["mask_path"],
        "helmet_target_annotation_source": helmet["target_annotation_source"],
        "helmet_target_bbox_source": helmet["target_bbox_source"],
        "helmet_target_mask_source": helmet["target_mask_source"],
        "miner_bbox_xyxy": miner_box,
        "miner_mask_path": miner["mask_path"],
        "miner_annotation_source": "yolov8m_detector_plus_sam1_box_prompt",
        "det_conf": float(miner.get("det_conf", 0.0)),
        "det_conf_norm": det_conf_norm,
        "sam_quality": sam_quality,
        "center_in_head_score": float(center_score),
        "helmet_center_in_miner": bool(center_in_miner),
        "helmet_center_in_expanded_miner": bool(center_in_expanded),
        "helmet_center_in_head_strict": bool(center_in_strict),
        "helmet_center_in_head_loose": bool(center_in_loose),
        "helmet_head_overlap": float(helmet_head_overlap),
        "containment": float(max(0.0, min(1.0, containment))),
        "size_score": float(ssize),
        "helmet_area_over_miner_area": float(bbox_area(helmet_box) / max(1.0, bbox_area(miner_box))),
        "pair_score": float(max(0.0, min(1.0, pair_score))),
        "rule_decision": rule_decision,
        "top1_minus_top2": top_gap,
        "strict_head_region": strict_head,
        "loose_head_region": loose_head,
        "expanded_miner_bbox_xyxy": expanded,
        "miner_mask_quality": {
            "mask_box_iou": miner.get("mask_box_iou"),
            "mask_area_ratio": miner.get("mask_area_ratio"),
            "connected_components": miner.get("connected_components"),
            "touches_edge": miner.get("touches_edge"),
        },
        "match_reason": {},
    }


def choose_query(queries: list[str], key: str) -> str:
    idx = int(stable_id(key, n=8), 16) % len(queries)
    return queries[idx]


def make_episode(pair: dict[str, Any]) -> dict[str, Any]:
    episode_id = f"ep_{stable_id(pair['pair_id'], pair['image_member'])}"
    return {
        "episode_id": episode_id,
        "episode_type": "pseudo_miner_to_sam_from_bbox_helmet",
        "image_member": pair["image_member"],
        "image_path": pair["image_path"],
        "split_source": pair.get("split_source"),
        "pair_id": pair["pair_id"],
        "pair_score": pair["pair_score"],
        "pair_confidence": pair["rule_decision"],
        "rounds": [
            {
                "round": 1,
                "query": choose_query(ROUND1_QUERIES, pair["pair_id"] + "r1"),
                "target_instance_id": pair["pseudo_miner_id"],
                "target_category": "coal_miner",
                "target_type": "pseudo",
                "target_mask": pair["miner_mask_path"],
                "target_bbox_xyxy": pair["miner_bbox_xyxy"],
                "target_quality": pair["pair_score"],
                "loss_weight": pair["pair_score"],
                "annotation_source": pair["miner_annotation_source"],
            },
            {
                "round": 2,
                "query": choose_query(ROUND2_QUERIES, pair["pair_id"] + "r2"),
                "refs_bank": [
                    {
                        "ref_id": f"ref_pos_{pair['pseudo_miner_id']}",
                        "role": "positive",
                        "category": "coal_miner",
                        "mask": pair["miner_mask_path"],
                        "bbox_xyxy": pair["miner_bbox_xyxy"],
                        "utility": 1.0,
                        "quality_score": pair["pair_score"],
                    }
                ],
                "target_instance_id": pair["helmet_id"],
                "target_category": "mining_helmet",
                "target_type": "sam_mask_from_real_bbox",
                "target_mask": pair["helmet_mask_path"],
                "target_bbox_xyxy": pair["helmet_bbox_xyxy"],
                "target_quality": 1.0,
                "loss_weight": 1.0,
                "annotation_source": pair["helmet_target_annotation_source"],
            },
        ],
    }


def separation_score(a: list[float], b: list[float], image_size: list[int]) -> float:
    diag = math.hypot(float(image_size[0]), float(image_size[1]))
    ca = bbox_center(a)
    cb = bbox_center(b)
    dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    center_sep = min(1.0, dist / max(1.0, 0.25 * diag))
    return float(max(center_sep, 1.0 - bbox_iou(a, b)))


def make_counterfactuals(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_image[pair["image_member"]].append(pair)
    rows: list[dict[str, Any]] = []
    for image_member, image_pairs in by_image.items():
        if len(image_pairs) < 2:
            continue
        image_pairs = sorted(image_pairs, key=lambda x: x["pair_id"])
        for i in range(len(image_pairs)):
            for j in range(i + 1, len(image_pairs)):
                a = image_pairs[i]
                b = image_pairs[j]
                miner_sep = separation_score(a["miner_bbox_xyxy"], b["miner_bbox_xyxy"], a["image_size"])
                helmet_sep = separation_score(a["helmet_bbox_xyxy"], b["helmet_bbox_xyxy"], a["image_size"])
                non_overlap = 0.5 * (1.0 - bbox_iou(a["miner_bbox_xyxy"], b["miner_bbox_xyxy"])) + 0.5 * (
                    1.0 - bbox_iou(a["helmet_bbox_xyxy"], b["helmet_bbox_xyxy"])
                )
                cf_score = (
                    0.4 * min(a["pair_score"], b["pair_score"])
                    + 0.2 * miner_sep
                    + 0.2 * helmet_sep
                    + 0.2 * non_overlap
                )
                if cf_score < 0.70:
                    continue
                rows.append({
                    "counterfactual_id": f"cf_{stable_id(image_member, a['pair_id'], b['pair_id'])}",
                    "image_member": image_member,
                    "image_path": a["image_path"],
                    "split_source": a.get("split_source"),
                    "pair_ids": [a["pair_id"], b["pair_id"]],
                    "helmet_ids": [a["helmet_id"], b["helmet_id"]],
                    "pseudo_miner_ids": [a["pseudo_miner_id"], b["pseudo_miner_id"]],
                    "same_round2_query": "Based on the miner mask from the previous round, segment only the mining helmet worn by that miner.",
                    "cf_score": float(cf_score),
                    "miner_separation": miner_sep,
                    "helmet_separation": helmet_sep,
                    "non_overlap_score": non_overlap,
                    "use_as_core_counterfactual": True,
                })
    return rows


def overlay_pair(pair: dict[str, Any], output_path: Path) -> None:
    image = Image.open(pair["image_path"]).convert("RGB")
    canvas = image.convert("RGBA")
    miner_mask = Image.open(pair["miner_mask_path"]).convert("L")
    helmet_mask = Image.open(pair["helmet_mask_path"]).convert("L")
    green = Image.new("RGBA", canvas.size, (0, 220, 80, 90))
    cyan = Image.new("RGBA", canvas.size, (0, 180, 255, 120))
    canvas = Image.composite(green, canvas, miner_mask)
    canvas = Image.composite(cyan, canvas, helmet_mask)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(pair["miner_bbox_xyxy"], outline=(0, 255, 80, 255), width=4)
    draw.rectangle(pair["helmet_bbox_xyxy"], outline=(0, 180, 255, 255), width=3)
    label = (
        f"{pair['rule_decision']} score={pair['pair_score']:.3f} "
        f"amb_gap={pair['ambiguous_score_gap']} amb_multi={pair['ambiguous_multi_head_match']}"
    )
    draw.rectangle([0, 0, min(canvas.size[0], 980), 34], fill=(0, 0, 0, 180))
    draw.text((8, 8), label, fill=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, quality=92)


def write_pair_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = [
        "pair_id",
        "image_member",
        "helmet_id",
        "pseudo_miner_id",
        "rule_decision",
        "selected_for_episode",
        "pair_score",
        "top1_minus_top2",
        "ambiguous_score_gap",
        "ambiguous_multi_head_match",
        "helmet_center_in_miner",
        "helmet_center_in_head_strict",
        "helmet_center_in_head_loose",
        "helmet_head_overlap",
        "containment",
        "size_score",
        "det_conf",
        "sam_quality",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in pairs:
            writer.writerow({field: row.get(field) for field in fields})


def cmd_pair_episodes(args: argparse.Namespace) -> int:
    helmets = helmet_records(args.helmet_clean, args.helmet_sam)
    if args.limit and args.limit > 0:
        allowed = {row["image_member"] for row in helmets[: args.limit]}
        helmets = [row for row in helmets if row["image_member"] in allowed]
    miners = read_jsonl(args.pseudo_miners)
    miners_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for miner in miners:
        miners_by_image[miner["image_member"]].append(miner)
    candidates_by_helmet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for helmet in helmets:
        for miner in miners_by_image.get(helmet["image_member"], []):
            candidates_by_helmet[helmet["helmet_id"]].append(build_candidate(helmet, miner))

    top_candidates: list[dict[str, Any]] = []
    for helmet in helmets:
        cands = sorted(candidates_by_helmet.get(helmet["helmet_id"], []), key=lambda x: x["pair_score"], reverse=True)
        if not cands:
            continue
        top_gap = cands[0]["pair_score"] - cands[1]["pair_score"] if len(cands) > 1 else None
        multi_head = sum(1 for c in cands if c["helmet_center_in_miner"] and c["center_in_head_score"] >= 0.7) > 1
        for idx, cand in enumerate(cands):
            cand["top1_minus_top2"] = top_gap
            cand["candidate_rank_for_helmet"] = idx
            cand["ambiguous_score_gap"] = bool(top_gap is not None and top_gap < 0.10)
            cand["ambiguous_multi_head_match"] = bool(multi_head)
            cand["selected_for_episode"] = False
            cand["match_reason"] = {
                "helmet_center_in_head_score": cand["center_in_head_score"],
                "helmet_head_overlap": cand["helmet_head_overlap"],
                "containment": cand["containment"],
                "size_score": cand["size_score"],
                "det_conf_norm": cand["det_conf_norm"],
                "sam_quality": cand["sam_quality"],
            }
        top_candidates.append(cands[0])

    eligible = [
        row for row in top_candidates
        if row["rule_decision"] in {"high", "medium"}
        and not row["ambiguous_score_gap"]
        and not row["ambiguous_multi_head_match"]
    ]
    eligible = sorted(eligible, key=lambda x: x["pair_score"], reverse=True)
    used_helmets: set[str] = set()
    used_miners: set[str] = set()
    selected: list[dict[str, Any]] = []
    for row in eligible:
        if row["helmet_id"] in used_helmets or row["pseudo_miner_id"] in used_miners:
            continue
        row["selected_for_episode"] = row["rule_decision"] == "high"
        used_helmets.add(row["helmet_id"])
        used_miners.add(row["pseudo_miner_id"])
        if row["selected_for_episode"]:
            selected.append(row)

    pair_path = args.out_dir / "helmet_miner_pairs.jsonl"
    write_jsonl(pair_path, top_candidates)
    episodes = [make_episode(row) for row in selected]
    write_jsonl(args.out_dir / "episodes_miner_to_helmet_v1.jsonl", episodes)
    cfs = make_counterfactuals(selected)
    write_jsonl(args.out_dir / "episodes_counterfactual_val.jsonl", cfs)

    reports = args.out_dir / "reports"
    write_pair_csv(reports / "pair_quality_report.csv", top_candidates)

    counts = Counter(row["rule_decision"] for row in top_candidates)
    amb_counts = Counter()
    for row in top_candidates:
        if row["ambiguous_score_gap"]:
            amb_counts["ambiguous_score_gap"] += 1
        if row["ambiguous_multi_head_match"]:
            amb_counts["ambiguous_multi_head_match"] += 1
    score_by_decision = {
        decision: {
            "count": len(vals),
            "mean_pair_score": float(np.mean([v["pair_score"] for v in vals])) if vals else 0.0,
            "mean_head_score": float(np.mean([v["center_in_head_score"] for v in vals])) if vals else 0.0,
            "mean_containment": float(np.mean([v["containment"] for v in vals])) if vals else 0.0,
        }
        for decision, vals in {
            k: [row for row in top_candidates if row["rule_decision"] == k]
            for k in ("high", "medium", "low")
        }.items()
    }
    episode_stats = {
        "dataset": "MiningHelmet-with-PseudoMiner v1",
        "helmet_target_annotation_note": "DsDPM66 mining_helmet has real bbox annotations but no human polygon masks; target masks are SAM1 masks generated from real bboxes.",
        "helmets_seen": len(helmets),
        "helmet_images_with_miners": len({m["image_member"] for m in miners}),
        "pseudo_miners": len(miners),
        "top_pairs": len(top_candidates),
        "rule_decision_counts": dict(counts),
        "ambiguity_counts": dict(amb_counts),
        "main_high_pairs_for_episodes": len(selected),
        "episodes": len(episodes),
        "counterfactual_core": len(cfs),
        "pair_score_formula": "0.15*det_conf_norm + 0.35*center_in_head_score + 0.20*helmet_head_overlap + 0.15*containment + 0.10*size_score + 0.05*sam_quality",
        "score_by_decision": score_by_decision,
    }
    write_json(reports / "episode_stats.json", episode_stats)

    rng = random.Random(args.seed)
    overlay_rows: list[dict[str, Any]] = []
    buckets = {
        "high": [row for row in top_candidates if row["rule_decision"] == "high"],
        "medium": [row for row in top_candidates if row["rule_decision"] == "medium"],
        "rejected": [row for row in top_candidates if row["rule_decision"] == "low" or row["ambiguous_score_gap"] or row["ambiguous_multi_head_match"]],
    }
    targets = {"high": args.overlay_high, "medium": args.overlay_medium, "rejected": args.overlay_rejected}
    for bucket, rows in buckets.items():
        rows = rows[:]
        rng.shuffle(rows)
        for row in rows[: targets[bucket]]:
            overlay_path = reports / "overlays" / bucket / f"{row['pair_id']}.jpg"
            try:
                overlay_pair(row, overlay_path)
                overlay_rows.append({
                    "pair_id": row["pair_id"],
                    "bucket": bucket,
                    "overlay_path": str(overlay_path),
                    "image_member": row["image_member"],
                    "rule_decision": row["rule_decision"],
                    "pair_score": row["pair_score"],
                    "ambiguous_score_gap": row["ambiguous_score_gap"],
                    "ambiguous_multi_head_match": row["ambiguous_multi_head_match"],
                })
            except Exception as exc:  # noqa: BLE001
                overlay_rows.append({
                    "pair_id": row["pair_id"],
                    "bucket": bucket,
                    "overlay_path": "",
                    "error": str(exc)[:300],
                })
    with (reports / "visual_overlay_index.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["pair_id", "bucket", "overlay_path", "image_member", "rule_decision", "pair_score", "ambiguous_score_gap", "ambiguous_multi_head_match", "error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in overlay_rows:
            writer.writerow({field: row.get(field) for field in fields})

    print(json.dumps(episode_stats, ensure_ascii=False, indent=2), flush=True)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    problems: list[str] = []
    pair_ids: set[str] = set()
    for path in [
        args.out_dir / "pseudo_miners_on_helmet_images.jsonl",
        args.out_dir / "helmet_miner_pairs.jsonl",
        args.out_dir / "episodes_miner_to_helmet_v1.jsonl",
        args.out_dir / "episodes_counterfactual_val.jsonl",
    ]:
        try:
            rows = read_jsonl(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path}: {exc}")
            continue
        if path.name == "helmet_miner_pairs.jsonl":
            for row in rows:
                if row["pair_id"] in pair_ids:
                    problems.append(f"duplicate pair_id {row['pair_id']}")
                pair_ids.add(row["pair_id"])
                for key in ("image_path", "miner_mask_path", "helmet_mask_path"):
                    if not Path(row[key]).exists():
                        problems.append(f"missing {key}: {row[key]}")
        if path.name == "episodes_miner_to_helmet_v1.jsonl":
            episode_ids: set[str] = set()
            for row in rows:
                if row["episode_id"] in episode_ids:
                    problems.append(f"duplicate episode_id {row['episode_id']}")
                episode_ids.add(row["episode_id"])
                for rnd in row["rounds"]:
                    if not Path(rnd["target_mask"]).exists():
                        problems.append(f"missing target mask {rnd['target_mask']}")
    report = {"ok": not problems, "problem_count": len(problems), "problems": problems[:200]}
    write_json(args.out_dir / "reports/validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if not problems else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MiningHelmet-with-PseudoMiner v1 data.")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-yolo")
    p.add_argument("--coal-zip", type=Path, default=COAL_ZIP)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-images", action="store_true")
    p.set_defaults(func=cmd_prepare_yolo)

    p = sub.add_parser("infer-miners")
    p.add_argument("--helmet-clean", type=Path, default=HELMET_CLEAN)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.65)
    p.add_argument("--max-det", type=int, default=10)
    p.add_argument("--min-box-area", type=float, default=64.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_infer_miners)

    p = sub.add_parser("sam-miners")
    p.add_argument("--detections", type=Path, default=None)
    p.add_argument("--sam-checkpoint", type=Path, default=SAM_VIT_B)
    p.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_sam_miners)

    p = sub.add_parser("pair-episodes")
    p.add_argument("--helmet-clean", type=Path, default=HELMET_CLEAN)
    p.add_argument("--helmet-sam", type=Path, default=HELMET_SAM)
    p.add_argument("--pseudo-miners", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260510)
    p.add_argument("--overlay-high", type=int, default=100)
    p.add_argument("--overlay-medium", type=int, default=50)
    p.add_argument("--overlay-rejected", type=int, default=50)
    p.set_defaults(func=cmd_pair_episodes)

    p = sub.add_parser("validate")
    p.set_defaults(func=cmd_validate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports").mkdir(parents=True, exist_ok=True)
    if hasattr(args, "detections") and args.detections is None:
        args.detections = args.out_dir / "raw_miner_detections_on_helmet_images.jsonl"
    if hasattr(args, "pseudo_miners") and args.pseudo_miners is None:
        args.pseudo_miners = args.out_dir / "pseudo_miners_on_helmet_images.jsonl"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

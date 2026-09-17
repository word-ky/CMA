from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

try:
    from PIL import Image
except Exception:  # pragma: no cover - the pipeline still works without sizes.
    Image = None

LOGGER = logging.getLogger("cmllm")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt", ".json", ".xml"}

DEFAULT_CLASS_NAMES = [
    "coal_miner",
    "compressed_oxygen_self_rescuer",
    "drill_pipe",
    "drill_rig",
    "interaction_between_miner_and_drill_pipe",
    "mining_helmet",
]

CN_NAMES = {
    "coal_miner": "煤矿工人",
    "miner": "矿工",
    "person": "人员",
    "worker": "工人",
    "compressed_oxygen_self_rescuer": "压缩氧自救器",
    "self_rescuer": "自救器",
    "drill_pipe": "钻杆",
    "drill_rig": "钻机",
    "interaction_between_miner_and_drill_pipe": "矿工与钻杆交互",
    "mining_helmet": "矿帽",
    "helmet": "安全帽",
}

PERSON_TERMS = {"coal_miner", "miner", "person", "worker", "human"}
EQUIPMENT_TERMS = {
    "drill_pipe",
    "drill_rig",
    "pipe",
    "rig",
    "interaction_between_miner_and_drill_pipe",
}
PPE_TERMS = {"mining_helmet", "helmet", "compressed_oxygen_self_rescuer", "self_rescuer"}


def normalize_name(value: str | int | float | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def cn_name(name: str) -> str:
    norm = normalize_name(name)
    return CN_NAMES.get(norm, norm.replace("_", " "))


def json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_dataset_files(config: dict[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for article in config["dataset_articles"]:
        article_id = article["article_id"]
        for item in article["files"]:
            row = dict(item)
            row["article_id"] = article_id
            files.append(row)
    return files


def md5sum(path: Path, chunk_size: int = 1024 * 1024 * 16) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(spec: dict[str, Any], zip_dir: Path, retries: int, chunk_mb: int) -> dict[str, Any]:
    zip_dir.mkdir(parents=True, exist_ok=True)
    target = zip_dir / spec["name"]
    part = target.with_suffix(target.suffix + ".part")
    expected_size = int(spec["size"])
    expected_md5 = spec.get("md5")

    if target.exists() and target.stat().st_size == expected_size:
        actual_md5 = md5sum(target)
        if not expected_md5 or actual_md5 == expected_md5:
            LOGGER.info("download exists and verified: %s", target.name)
            return {"name": target.name, "status": "verified", "path": str(target), "md5": actual_md5}
        bad = target.with_name(target.name + f".bad-md5-{int(time.time())}")
        target.rename(bad)
        LOGGER.warning("existing file md5 mismatch, moved to %s", bad)

    url = spec["download_url"]
    LOGGER.info("downloading with curl %s size=%s", target.name, expected_size)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(retries),
        "--retry-delay",
        "10",
        "--retry-all-errors",
        "--connect-timeout",
        "60",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "--continue-at",
        "-",
        "--output",
        str(part),
        url,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        LOGGER.warning("curl stdout for %s: %s", target.name, result.stdout[-2000:])
        LOGGER.warning("curl stderr for %s: %s", target.name, result.stderr[-4000:])
        raise RuntimeError(f"curl failed for {target.name} with exit code {result.returncode}")
    if not part.exists() or part.stat().st_size != expected_size:
        actual_size = part.stat().st_size if part.exists() else 0
        raise RuntimeError(f"{target.name} incomplete: {actual_size} != {expected_size}")
    part.rename(target)

    actual_md5 = md5sum(target)
    if expected_md5 and actual_md5 != expected_md5:
        raise RuntimeError(f"md5 mismatch for {target.name}: {actual_md5} != {expected_md5}")
    LOGGER.info("download verified: %s md5=%s", target.name, actual_md5)
    return {"name": target.name, "status": "verified", "path": str(target), "md5": actual_md5}


def download_dataset(config: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    zip_dir = root / "data" / "raw" / "dsdpm66" / "zips"
    retries = int(config.get("download_retries", 5))
    chunk_mb = int(config.get("download_chunk_mb", 8))
    results = []
    for spec in flatten_dataset_files(config):
        results.append(download_one(spec, zip_dir, retries, chunk_mb))
        write_json(run_dir / "download_status.json", results)
    return results


def safe_member_name(member: str) -> str:
    name = member.replace("\\", "/").strip("/")
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", PurePosixPath(name).stem)[:80]
    suffix = PurePosixPath(name).suffix.lower()
    return f"{stem}-{digest}{suffix}"


def zip_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return [info.filename for info in zf.infolist() if not info.is_dir()]


def infer_class_names(zf: zipfile.ZipFile) -> list[str]:
    candidates = [
        name
        for name in zf.namelist()
        if PurePosixPath(name).suffix.lower() in {".yaml", ".yml"}
        and PurePosixPath(name).name.lower() in {"data.yaml", "dataset.yaml", "classes.yaml"}
    ]
    candidates.extend(
        name
        for name in zf.namelist()
        if PurePosixPath(name).suffix.lower() in {".yaml", ".yml"}
        and name not in candidates
    )
    for name in candidates:
        try:
            data = yaml.safe_load(zf.read(name).decode("utf-8", errors="ignore"))
        except Exception:
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, list):
            return [normalize_name(x) for x in names]
        if isinstance(names, dict):
            return [normalize_name(names[k]) for k in sorted(names, key=lambda x: int(x))]
    coco_names = []
    json_members = [
        name
        for name in zf.namelist()
        if PurePosixPath(name).suffix.lower() == ".json"
        and ("coco" in name.lower() or "annotation" in name.lower())
    ]
    for member in json_members:
        try:
            data = json.loads(zf.read(member).decode("utf-8", errors="ignore"))
        except Exception:
            continue
        categories = data.get("categories") if isinstance(data, dict) else None
        if not isinstance(categories, list):
            continue
        for cat in sorted(categories, key=lambda x: int(x.get("id", 0)) if isinstance(x, dict) else 0):
            if isinstance(cat, dict) and cat.get("name") is not None:
                name = normalize_name(cat["name"])
                if name not in coco_names:
                    coco_names.append(name)
    if coco_names:
        return coco_names
    return DEFAULT_CLASS_NAMES


def make_label_index(names: list[str]) -> dict[str, list[str]]:
    labels = [n for n in names if PurePosixPath(n).suffix.lower() in LABEL_EXTS]
    index: dict[str, list[str]] = {}
    for label in labels:
        path = PurePosixPath(label)
        stem = path.stem.lower()
        index.setdefault(stem, []).append(label)
        normalized = str(path.with_suffix("")).lower().replace("/labels/", "/images/")
        index.setdefault(normalized, []).append(label)
    return index


def find_label_for_image(image_name: str, label_index: dict[str, list[str]]) -> str | None:
    path = PurePosixPath(image_name)
    keys = [
        path.stem.lower(),
        str(path.with_suffix("")).lower(),
        str(path.with_suffix("")).lower().replace("/images/", "/labels/"),
        str(path.with_suffix("")).lower().replace("/images/train/", "/labels/train/"),
        str(path.with_suffix("")).lower().replace("/images/val/", "/labels/val/"),
    ]
    for key in keys:
        matches = label_index.get(key)
        if matches:
            return sorted(matches, key=len)[0]
    return None


def image_size(path: Path) -> tuple[int, int] | None:
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def normalize_bbox_xyxy(values: list[float], size: tuple[int, int] | None) -> list[float] | None:
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    if size and max(abs(v) for v in values) > 1.5:
        w, h = size
        if w > 0 and h > 0:
            x1, x2 = x1 / w, x2 / w
            y1, y2 = y1 / h, y2 / h
    vals = [max(0.0, min(1.0, v)) for v in [x1, y1, x2, y2]]
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def parse_yolo_label(
    path: Path,
    archive_category: str,
    class_names: list[str],
    size: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    objects = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        if 0 <= cls_id < len(class_names):
            name = class_names[cls_id]
        else:
            name = archive_category
        if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
            bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        elif size:
            width, height = size
            bbox = [(x - w / 2) / width, (y - h / 2) / height, (x + w / 2) / width, (y + h / 2) / height]
        else:
            bbox = [x, y, x + w, y + h]
        norm_bbox = normalize_bbox_xyxy(bbox, size)
        objects.append(
            {
                "name": normalize_name(name),
                "bbox": norm_bbox,
                "source": "yolo_txt",
                "line": line_no,
            }
        )
    return objects


def parse_xml_label(path: Path, archive_category: str, size: tuple[int, int] | None) -> list[dict[str, Any]]:
    import xml.etree.ElementTree as ET

    objects = []
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return objects
    for obj in root.findall(".//object"):
        name = normalize_name(obj.findtext("name") or archive_category)
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            vals = [
                float(box.findtext("xmin") or 0),
                float(box.findtext("ymin") or 0),
                float(box.findtext("xmax") or 0),
                float(box.findtext("ymax") or 0),
            ]
        except ValueError:
            continue
        norm_bbox = normalize_bbox_xyxy(vals, size)
        objects.append({"name": name, "bbox": norm_bbox, "source": "xml"})
    return objects


def iter_json_objects(data: Any) -> list[dict[str, Any]]:
    found = []
    if isinstance(data, dict):
        if any(k in data for k in ("bbox", "box", "bndbox", "points")):
            found.append(data)
        for value in data.values():
            found.extend(iter_json_objects(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(iter_json_objects(item))
    return found


def parse_json_label(path: Path, archive_category: str, size: tuple[int, int] | None) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    objects = []
    for obj in iter_json_objects(data):
        raw_name = (
            obj.get("name")
            or obj.get("label")
            or obj.get("class")
            or obj.get("category")
            or obj.get("category_name")
            or archive_category
        )
        box = obj.get("bbox") or obj.get("box") or obj.get("bndbox")
        vals = None
        if isinstance(box, dict):
            if all(k in box for k in ("x1", "y1", "x2", "y2")):
                vals = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            elif all(k in box for k in ("x", "y", "w", "h")):
                x, y, w, h = [float(box[k]) for k in ("x", "y", "w", "h")]
                vals = [x, y, x + w, y + h]
        elif isinstance(box, list) and len(box) >= 4:
            x, y, a, b = [float(v) for v in box[:4]]
            vals = [x, y, x + a, y + b] if a > x and b > y and max(box[:4]) > 1.5 else [x, y, a, b]
        norm_bbox = normalize_bbox_xyxy(vals, size) if vals else None
        objects.append({"name": normalize_name(raw_name), "bbox": norm_bbox, "source": "json"})
    return objects


def parse_annotation(
    label_path: Path | None,
    archive_category: str,
    class_names: list[str],
    size: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if not label_path or not label_path.exists():
        return []
    suffix = label_path.suffix.lower()
    if suffix == ".txt":
        return parse_yolo_label(label_path, archive_category, class_names, size)
    if suffix == ".xml":
        return parse_xml_label(label_path, archive_category, size)
    if suffix == ".json":
        return parse_json_label(label_path, archive_category, size)
    return []


def build_coco_index(zf: zipfile.ZipFile, archive_category: str) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    json_members = [
        name
        for name in zf.namelist()
        if PurePosixPath(name).suffix.lower() == ".json"
        and ("coco" in name.lower() or "annotation" in name.lower())
    ]
    for member in json_members:
        try:
            data = json.loads(zf.read(member).decode("utf-8", errors="ignore"))
        except Exception:
            continue
        if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
            continue
        categories = {
            cat.get("id"): normalize_name(cat.get("name") or archive_category)
            for cat in data.get("categories", [])
            if isinstance(cat, dict)
        }
        images = {
            img.get("id"): img
            for img in data.get("images", [])
            if isinstance(img, dict) and img.get("id") is not None
        }
        for ann in data.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            img = images.get(ann.get("image_id"))
            if not img:
                continue
            file_name = str(img.get("file_name") or "")
            width = float(img.get("width") or 0)
            height = float(img.get("height") or 0)
            raw_bbox = ann.get("bbox")
            bbox = None
            if isinstance(raw_bbox, list) and len(raw_bbox) >= 4 and width > 0 and height > 0:
                x, y, w, h = [float(v) for v in raw_bbox[:4]]
                bbox = normalize_bbox_xyxy([x / width, y / height, (x + w) / width, (y + h) / height], None)
            obj = {
                "name": categories.get(ann.get("category_id"), archive_category),
                "bbox": bbox,
                "source": "coco_json",
                "annotation_id": ann.get("id"),
            }
            keys = {
                file_name.lower(),
                PurePosixPath(file_name).name.lower(),
                str(PurePosixPath(file_name).with_suffix("")).lower(),
                PurePosixPath(file_name).stem.lower(),
            }
            for key in keys:
                index.setdefault(key, []).append(obj)
    return index


def coco_objects_for_image(image_member: str, coco_index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    path = PurePosixPath(image_member)
    keys = [
        str(path).lower(),
        path.name.lower(),
        str(path.with_suffix("")).lower(),
        path.stem.lower(),
    ]
    for key in keys:
        objects = coco_index.get(key)
        if objects:
            return [dict(obj) for obj in objects]
    return []


def extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)


def create_subset(config: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    rng = random.Random(int(config.get("seed", 66)))
    samples_per_archive = int(config.get("samples_per_archive", 40))
    zip_dir = root / "data" / "raw" / "dsdpm66" / "zips"
    image_root = root / "data" / "interim" / "dsdpm66_subset" / "images"
    label_root = root / "data" / "interim" / "dsdpm66_subset" / "labels"
    records = []

    for spec in flatten_dataset_files(config):
        archive = zip_dir / spec["name"]
        if not archive.exists():
            LOGGER.warning("archive missing, skip subset sampling: %s", archive)
            continue
        category = normalize_name(spec["category"])
        LOGGER.info("inventory archive %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            names = [info.filename for info in zf.infolist() if not info.is_dir()]
            class_names = infer_class_names(zf)
            coco_index = build_coco_index(zf, category)
            images = [
                n
                for n in names
                if PurePosixPath(n).suffix.lower() in IMAGE_EXTS
                and "__MACOSX" not in n
            ]
            label_index = make_label_index(names)
            rng.shuffle(images)
            selected = images[: min(samples_per_archive, len(images))]
            LOGGER.info("selected %s/%s images from %s", len(selected), len(images), archive.name)

            for image_member in selected:
                sample_id = f"{category}-{hashlib.sha1(image_member.encode('utf-8')).hexdigest()[:12]}"
                image_dest = image_root / category / safe_member_name(image_member)
                extract_member(zf, image_member, image_dest)

                label_member = find_label_for_image(image_member, label_index)
                label_dest = None
                if label_member:
                    label_dest = label_root / category / safe_member_name(label_member)
                    extract_member(zf, label_member, label_dest)

                size = image_size(image_dest)
                objects = parse_annotation(label_dest, category, class_names, size)
                if not objects:
                    objects = coco_objects_for_image(image_member, coco_index)
                if not objects:
                    objects = [{"name": category, "bbox": None, "source": "archive_category"}]
                records.append(
                    {
                        "sample_id": sample_id,
                        "source": "DsDPM66",
                        "category": category,
                        "archive": spec["name"],
                        "image_member": image_member,
                        "label_member": label_member,
                        "image_path": str(image_dest),
                        "label_path": str(label_dest) if label_dest else None,
                        "image_size": list(size) if size else None,
                        "objects": objects,
                    }
                )

    subset_path = run_dir / "subset_manifest.jsonl"
    write_jsonl(subset_path, records)
    LOGGER.info("subset written: %s records=%s", subset_path, len(records))
    return records


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    denom = bbox_area(a) + bbox_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def spatial_relations(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boxed = [o for o in objects if isinstance(o.get("bbox"), list) and len(o["bbox"]) == 4]
    relations = []
    for i, subject in enumerate(boxed):
        for j, obj in enumerate(boxed):
            if i == j:
                continue
            sb, ob = subject["bbox"], obj["bbox"]
            sx, sy = bbox_center(sb)
            ox, oy = bbox_center(ob)
            dx, dy = sx - ox, sy - oy
            dist = math.sqrt(dx * dx + dy * dy)
            predicate = None
            confidence = 0.6
            evidence = None
            overlap = bbox_iou(sb, ob)
            if overlap > 0.08:
                predicate = "overlap"
                confidence = min(0.98, 0.7 + overlap)
                evidence = f"bbox_iou={overlap:.3f}"
            elif dist < 0.22:
                predicate = "near"
                confidence = max(0.65, 1.0 - dist)
                evidence = f"center_distance={dist:.3f}"
            elif abs(dx) > abs(dy):
                predicate = "left_of" if dx < 0 else "right_of"
                confidence = min(0.95, 0.55 + abs(dx))
                evidence = f"center_dx={dx:.3f}"
            else:
                predicate = "above" if dy < 0 else "below"
                confidence = min(0.95, 0.55 + abs(dy))
                evidence = f"center_dy={dy:.3f}"
            relations.append(
                {
                    "subject": normalize_name(subject["name"]),
                    "predicate": predicate,
                    "object": normalize_name(obj["name"]),
                    "confidence": round(confidence, 3),
                    "evidence": evidence,
                }
            )
    return relations[:16]


def classify_risk(objects: list[dict[str, Any]], relations: list[dict[str, Any]], category: str) -> tuple[str, str]:
    names = {normalize_name(o["name"]) for o in objects}
    if category == "interaction_between_miner_and_drill_pipe":
        return "high", "矿工与钻杆存在交互关系，应重点关注卷入、碰撞和误操作风险。"

    for rel in relations:
        sub = normalize_name(rel["subject"])
        obj = normalize_name(rel["object"])
        pred = rel["predicate"]
        if pred in {"near", "overlap"}:
            if (sub in PERSON_TERMS and obj in EQUIPMENT_TERMS) or (obj in PERSON_TERMS and sub in EQUIPMENT_TERMS):
                return "high", "检测到人员靠近或重叠于钻机、钻杆等作业设备区域，存在卷入、碰撞或误入运行区域的风险。"

    if names & PPE_TERMS:
        return "medium", "检测到个体防护或自救装备相关目标，应关注佩戴、携带和遮挡状态是否满足安全要求。"
    if names & EQUIPMENT_TERMS:
        equipment_label = "、".join(cn_name(name) for name in sorted(names & EQUIPMENT_TERMS))
        return "medium", f"检测到{equipment_label}，应关注设备运行状态、作业区域占用和防护状态。"
    if names & PERSON_TERMS:
        return "medium", "检测到煤矿作业人员，应关注其与设备、通道和危险区域之间的相对位置。"
    return "low", "检测到煤矿井下作业场景目标，当前样本未形成明确高风险空间关系。"


def relation_sentence(rel: dict[str, Any]) -> str:
    subject = cn_name(rel["subject"])
    obj = cn_name(rel["object"])
    pred_cn = {
        "left_of": f"位于{obj}左侧",
        "right_of": f"位于{obj}右侧",
        "above": f"位于{obj}上方",
        "below": f"位于{obj}下方",
        "near": f"靠近{obj}",
        "overlap": f"与{obj}区域重叠",
    }.get(rel["predicate"], f"与{obj}存在{rel['predicate']}关系")
    return f"{subject}{pred_cn}"


def build_rule_text(record: dict[str, Any], relations: list[dict[str, Any]]) -> dict[str, Any]:
    objects = record["objects"]
    risk_level, risk_text = classify_risk(objects, relations, normalize_name(record["category"]))
    object_names = []
    for obj in objects:
        name = cn_name(obj["name"])
        if name not in object_names:
            object_names.append(name)
    scene_text = "画面中检测到" + "、".join(object_names[:8]) + "。"
    if relations:
        rel_text = "空间关系包括：" + "；".join(relation_sentence(r) for r in relations[:5]) + "。"
    elif any(isinstance(obj.get("bbox"), list) for obj in objects):
        rel_text = "当前样本仅形成单个可定位目标，未生成目标间空间关系。"
    else:
        rel_text = "当前样本缺少可用边界框，未生成明确的目标间空间关系。"
    rule_text = scene_text + rel_text + risk_text
    return {
        "sample_id": record["sample_id"],
        "source": record["source"],
        "category": record["category"],
        "objects": objects,
        "spatial_relations": relations,
        "rule_text": rule_text,
        "risk_level": risk_level,
    }


def generate_rules(subset_records: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    relation_rows = []
    for record in subset_records:
        rels = spatial_relations(record["objects"])
        relation_rows.append({"sample_id": record["sample_id"], "relations": rels})
        rows.append(build_rule_text(record, rels))
    write_jsonl(run_dir / "rule_relations.jsonl", relation_rows)
    write_jsonl(run_dir / "rule_texts.jsonl", rows)
    LOGGER.info("rule generation complete: %s records", len(rows))
    return rows


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def deepseek_client(config: dict[str, Any]):
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or config.get("deepseek_base_url") or "https://api.deepseek.com"
    return OpenAI(api_key=api_key, base_url=base_url)


def call_deepseek_json(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                stream=False,
                timeout=timeout,
            )
            content = response.choices[0].message.content or "{}"
            return extract_json_object(content)
        except Exception as exc:  # noqa: BLE001 - API calls need robust overnight retries.
            last_error = exc
            LOGGER.warning("DeepSeek call failed attempt=%s error=%s", attempt, exc)
            time.sleep(min(90, 4 * attempt))
    raise RuntimeError(f"DeepSeek call failed after retries: {last_error}") from last_error


def rewrite_with_deepseek(rule_rows: list[dict[str, Any]], config: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    output_path = run_dir / "deepseek_rewrites.jsonl"
    client = deepseek_client(config)
    model = os.environ.get("DEEPSEEK_MODEL") or config.get("deepseek_model") or "deepseek-v4-flash"
    retries = int(config.get("llm_max_retries", 3))
    timeout = int(config.get("llm_timeout_seconds", 90))

    rows = []
    if client is None:
        LOGGER.warning("DEEPSEEK_API_KEY missing; use rule text as rewrite fallback")
        for row in rule_rows:
            out = {
                "sample_id": row["sample_id"],
                "rewritten_text": row["rule_text"],
                "risk_expression": row["rule_text"],
                "changed": False,
                "llm_status": "skipped_no_api_key",
                "notes": "No DEEPSEEK_API_KEY in environment.",
            }
            append_jsonl(output_path, out)
            rows.append(out)
        return rows

    system = (
        "你是煤矿安全场景文本改写助手。你只能根据输入的结构化目标、空间关系和规则初稿改写。"
        "不得添加输入中没有的物体、动作、地点、事故或风险；不得夸大风险。"
        "输出必须是 JSON，字段为 rewritten_text, risk_expression, changed, notes。"
    )
    for idx, row in enumerate(rule_rows, 1):
        payload = {
            "sample_id": row["sample_id"],
            "objects": [cn_name(o["name"]) for o in row["objects"]],
            "spatial_relations": row["spatial_relations"],
            "risk_level": row["risk_level"],
            "rule_text": row["rule_text"],
        }
        user = (
            "请将下面的规则初稿改写成简洁、自然、准确的煤矿安全表达。"
            "保留核心空间关系和风险语义，最多 120 个汉字。\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        try:
            data = call_deepseek_json(
                client,
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                retries,
                timeout,
            )
            rewritten = str(data.get("rewritten_text") or data.get("risk_expression") or row["rule_text"]).strip()
            risk_expression = str(data.get("risk_expression") or rewritten).strip()
            out = {
                "sample_id": row["sample_id"],
                "rewritten_text": rewritten,
                "risk_expression": risk_expression,
                "changed": bool(data.get("changed", rewritten != row["rule_text"])),
                "llm_status": "ok",
                "notes": str(data.get("notes", "")),
            }
        except Exception as exc:  # noqa: BLE001
            out = {
                "sample_id": row["sample_id"],
                "rewritten_text": row["rule_text"],
                "risk_expression": row["rule_text"],
                "changed": False,
                "llm_status": "fallback_after_error",
                "notes": str(exc),
            }
        append_jsonl(output_path, out)
        rows.append(out)
        if idx % 10 == 0:
            LOGGER.info("DeepSeek rewrite progress %s/%s", idx, len(rule_rows))
    return rows


def simple_validation(rule: dict[str, Any], rewrite: dict[str, Any]) -> dict[str, Any]:
    text = str(rewrite.get("rewritten_text") or "").strip()
    issues = []
    if not text:
        issues.append("empty_rewrite")
    if len(text) > 180:
        issues.append("too_long")
    if "风险" in rule["rule_text"] and not any(x in text for x in ["风险", "隐患", "注意", "关注", "安全"]):
        issues.append("risk_semantics_missing")
    allowed_cn = {cn_name(o["name"]) for o in rule["objects"]}
    object_terms = set(CN_NAMES.values())
    unexpected = sorted(term for term in object_terms if term in text and term not in allowed_cn)
    if unexpected:
        issues.append("unexpected_object_terms:" + ",".join(unexpected[:5]))
    return {
        "passed": not issues,
        "faithfulness_score": 1.0 if not issues else max(0.2, 1.0 - 0.2 * len(issues)),
        "issues": issues,
        "validator": "rules",
    }


def validate_with_llm(
    rule_rows: list[dict[str, Any]],
    rewrite_rows: list[dict[str, Any]],
    config: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    output_path = run_dir / "second_validation.jsonl"
    final_path = run_dir / "final_dataset.jsonl"
    client = deepseek_client(config)
    model = os.environ.get("DEEPSEEK_MODEL") or config.get("deepseek_model") or "deepseek-v4-flash"
    retries = int(config.get("llm_max_retries", 3))
    timeout = int(config.get("llm_timeout_seconds", 90))
    use_llm = bool(config.get("second_validation_with_llm", True)) and client is not None
    rewrites = {row["sample_id"]: row for row in rewrite_rows}

    system = (
        "你是煤矿安全文本忠实性校验助手。请判断改写文本是否忠实于输入目标、空间关系和规则初稿。"
        "不得允许新增未提供的物体、地点、动作、事故类型或过度夸大。"
        "输出 JSON，字段为 passed, faithfulness_score, issues。"
    )
    validations = []
    final_rows = []
    for idx, rule in enumerate(rule_rows, 1):
        rewrite = rewrites[rule["sample_id"]]
        validation = simple_validation(rule, rewrite)
        if use_llm:
            payload = {
                "objects": [cn_name(o["name"]) for o in rule["objects"]],
                "spatial_relations": rule["spatial_relations"],
                "rule_text": rule["rule_text"],
                "rewritten_text": rewrite["rewritten_text"],
            }
            user = "请校验改写是否忠实。只输出 JSON。\n" + json.dumps(payload, ensure_ascii=False)
            try:
                llm_validation = call_deepseek_json(
                    client,
                    model,
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    retries,
                    timeout,
                )
                llm_issues = llm_validation.get("issues") or []
                if isinstance(llm_issues, str):
                    llm_issues = [llm_issues]
                validation = {
                    "passed": bool(llm_validation.get("passed", validation["passed"])) and validation["passed"],
                    "faithfulness_score": max(
                        0.0,
                        min(1.0, float(llm_validation.get("faithfulness_score", validation["faithfulness_score"]))),
                    ),
                    "issues": list(validation["issues"]) + list(llm_issues),
                    "validator": "rules+deepseek",
                }
            except Exception as exc:  # noqa: BLE001
                validation["issues"].append("llm_validator_error:" + str(exc))
                validation["validator"] = "rules_with_llm_error"

        final_text = rewrite["rewritten_text"] if validation["passed"] else rule["rule_text"]
        validation_row = {
            "sample_id": rule["sample_id"],
            **validation,
            "final_text_source": "deepseek_rewrite" if validation["passed"] else "rule_fallback",
        }
        final_row = {
            "sample_id": rule["sample_id"],
            "source": rule["source"],
            "category": rule["category"],
            "objects": rule["objects"],
            "spatial_relations": rule["spatial_relations"],
            "risk_level": rule["risk_level"],
            "rule_text": rule["rule_text"],
            "deepseek_rewritten_text": rewrite["rewritten_text"],
            "final_safety_text": final_text,
            "validation_passed": validation["passed"],
            "validation_issues": validation["issues"],
        }
        append_jsonl(output_path, validation_row)
        append_jsonl(final_path, final_row)
        validations.append(validation_row)
        final_rows.append(final_row)
        if idx % 10 == 0:
            LOGGER.info("second validation progress %s/%s", idx, len(rule_rows))
    return validations


def make_report(
    run_dir: Path,
    config: dict[str, Any],
    downloads: list[dict[str, Any]],
    subset: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    rewrites: list[dict[str, Any]],
    validations: list[dict[str, Any]],
) -> None:
    by_category: dict[str, int] = {}
    for row in subset:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
    rewrite_ok = sum(1 for row in rewrites if row.get("llm_status") == "ok")
    passed = sum(1 for row in validations if row.get("passed"))
    report = [
        "# CMLLM DsDPM66 Run Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Run dir: `{run_dir}`",
        f"- DeepSeek model: `{os.environ.get('DEEPSEEK_MODEL') or config.get('deepseek_model')}`",
        f"- Dataset: DsDPM 66 official Figshare files",
        f"- Downloads verified: {len(downloads)} / {len(flatten_dataset_files(config))}",
        f"- Subset samples: {len(subset)}",
        f"- Rule rows: {len(rules)}",
        f"- DeepSeek rewrite ok: {rewrite_ok} / {len(rewrites)}",
        f"- Second validation passed: {passed} / {len(validations)}",
        "",
        "## Samples By Category",
        "",
    ]
    for category, count in sorted(by_category.items()):
        report.append(f"- {category}: {count}")
    report.extend(
        [
            "",
            "## Output Files",
            "",
            "- `subset_manifest.jsonl`",
            "- `rule_relations.jsonl`",
            "- `rule_texts.jsonl`",
            "- `deepseek_rewrites.jsonl`",
            "- `second_validation.jsonl`",
            "- `final_dataset.jsonl`",
            "- `pipeline.log`",
            "",
            "## Notes",
            "",
            "- The API key was read from `DEEPSEEK_API_KEY` at runtime and was not written to project files.",
            "- The pipeline uses only a sampled subset of DsDPM 66.",
            "- Failed or unfaithful LLM rewrites automatically fall back to the deterministic rule text.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    LOGGER.info("report written: %s", run_dir / "report.md")


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    root = config_path.resolve().parents[1]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + config.get("run_name", "run")
    run_dir = root / "outputs" / "runs" / run_id
    setup_logging(run_dir)
    LOGGER.info("run started root=%s run_dir=%s", root, run_dir)
    write_json(run_dir / "effective_config.json", config)

    downloads = download_dataset(config, root, run_dir)
    subset = create_subset(config, root, run_dir)
    rules = generate_rules(subset, run_dir)
    rewrites = rewrite_with_deepseek(rules, config, run_dir)
    validations = validate_with_llm(rules, rewrites, config, run_dir)
    make_report(run_dir, config, downloads, subset, rules, rewrites, validations)
    LOGGER.info("run finished successfully: %s", run_dir)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.config)
    except Exception:
        LOGGER.exception("pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

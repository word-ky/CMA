"""Freeze the preselected 30 CF groups using archived image bytes / SAM-B recipe.

Recovery only: no LISA inference, target-IoU selection or tuning. Exact original
mask paths are preferred. Otherwise mark regenerated, never exact-recovered.
"""
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(root, groups_path=None, data_name="cycle003"):
    root = Path(root)
    out = root / "shared/data" / data_name
    log = root / "research_log" / data_name
    for folder in (out / "images", out / "miner_masks", out / "helmet_masks", log):
        folder.mkdir(parents=True, exist_ok=True)
    groups_path = Path(groups_path) if groups_path else root / "research_log/cycle002/counterfactual_holdout_first30.jsonl"
    pairs_path = root / "shared/data/final_accepted_v1/helmet_miner_pairs_accept_high.jsonl"
    groups = [json.loads(l) for l in groups_path.read_text().splitlines()]
    pairs = {r["pair_id"]: r for r in map(json.loads, pairs_path.read_text().splitlines())}
    required = {Path(pairs[p][kind]).name for g in groups for p in g["pair_ids"]
                for kind in ("miner_mask_path", "helmet_mask_path")}
    searches = []
    historical = {}
    # Search project-owned archives/relocations, excluding our known regenerated
    # recovery directory. Basename hits are recorded, not silently accepted.
    for search in (Path("/home/wjq"), Path("/home/liujianhua/wjq"), root.parent):
        hits = []
        for current, dirs, files in os.walk(search):
            dirs[:] = [d for d in dirs if Path(current) / d != root and d not in (".git", ".venv")]
            for name in required.intersection(files):
                hits.append(str(Path(current) / name))
        searches.append({"root": str(search), "mask_basename_hits": hits})
    (log / "exact_search.json").write_text(json.dumps(searches, indent=2))
    for g in groups:
        for pair_id in g["pair_ids"]:
            for kind in ("miner_mask_path", "helmet_mask_path"):
                path = Path(pairs[pair_id][kind])
                if path.is_file():
                    historical[str(path)] = path
    unclassified_hits = [p for s in searches for p in s["mask_basename_hits"] if p not in historical]
    if unclassified_hits:
        raise RuntimeError("Historical basename candidates found; inspect exact_search.json before regeneration")

    checkpoint = root / "shared/source/sam_vit_b_01ec64.pth"
    archive = root / "shared/source/mining_helmet.zip"
    generator = {"model": "SAM vit_b", "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                 "script_sha256": sha(__file__), "torch_version": torch.__version__,
                 "opencv_version": cv2.__version__, "seed": 0,
                 "miner_multimask": True, "miner_selection": "highest SAM predicted quality (no GT)",
                 "helmet_multimask": False, "helmet_box": "COCO clamp -> normalized round(6) -> pixel float32"}
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=str(checkpoint)).cuda().eval())
    assets, restored_groups, restored_pairs = [], [], {}
    with zipfile.ZipFile(archive) as zf:
        annotations = {}
        for name in zf.namelist():
            if name.endswith(".json"):
                data = json.loads(zf.read(name))
                if isinstance(data, dict) and "annotations" in data and "images" in data:
                    images = {i["id"]: i for i in data["images"]}
                    for ann in data["annotations"]:
                        image = images[ann["image_id"]]
                        annotations[(Path(image["file_name"]).name, int(ann["id"]))] = (image, ann, name)
        for group in groups:
            image_path = out / "images" / (group["counterfactual_id"] + ".jpg")
            image_path.write_bytes(zf.read(group["image_member"]))
            assets.append({"type": "image", "group_id": group["counterfactual_id"], "status": "exact_recovered",
                           "source_archive": str(archive), "source_member": group["image_member"],
                           "path": str(image_path), "sha256": sha(image_path)})
            rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            predictor.set_image(rgb)
            for pair_id in group["pair_ids"]:
                if pair_id in restored_pairs:
                    continue
                pair = copy.deepcopy(pairs[pair_id])
                pair["image_path"] = str(image_path)
                ann_id = int(re.search(r"-ann(\d+)-", pairs[pair_id]["image_path"]).group(1))
                image_meta, ann, annotation_member = annotations[(Path(group["image_member"]).name, ann_id)]
                assert (w, h) == (image_meta["width"], image_meta["height"])
                x, y, bw, bh = map(float, ann["bbox"])
                x, y = max(0., min(x, w - 1.)), max(0., min(y, h - 1.))
                bw, bh = max(1., min(bw, w - x)), max(1., min(bh, h - y))
                norm = [round(x / w, 6), round(y / h, 6), round((x + bw) / w, 6), round((y + bh) / h, 6)]
                boxes = {"miner": pair["miner_bbox_xyxy"],
                         "helmet": [norm[0] * w, norm[1] * h, norm[2] * w, norm[3] * h]}
                for kind in ("miner", "helmet"):
                    original = pairs[pair_id][kind + "_mask_path"]
                    dest = out / (kind + "_masks") / (pair_id + ".png")
                    if original in historical:
                        shutil.copyfile(original, dest)
                        status = "exact_recovered"
                    else:
                        with torch.no_grad():
                            masks, scores, _ = predictor.predict(box=np.array(boxes[kind], dtype=np.float32),
                                                                  multimask_output=kind == "miner")
                        index = int(scores.argmax()) if kind == "miner" else 0
                        Image.fromarray(masks[index].astype(np.uint8) * 255).save(dest)
                        status = "regenerated"
                    assets.append({"type": kind + "_mask", "pair_id": pair_id, "status": status,
                                   "source_path": original, "source_image": str(image_path),
                                   "source_pair_manifest": str(pairs_path), "source_annotation_member": annotation_member,
                                   "source_annotation_id": ann_id, "prompt_box": boxes[kind],
                                   "generator": "sam_b_archived_recipe" if status == "regenerated" else None,
                                   "path": str(dest), "sha256": sha(dest)})
                    pair[kind + "_mask_path"] = str(dest)
                restored_pairs[pair_id] = pair
            restored = copy.deepcopy(group)
            restored["image_path"] = str(image_path)
            restored_groups.append(restored)
            print(json.dumps({"frozen_groups": len(restored_groups), "group_id": group["counterfactual_id"]}), flush=True)
    cf_name = "counterfactual_holdout_first30_restored.jsonl" if data_name == "cycle003" else "counterfactual_selected_restored.jsonl"
    cf_out, pairs_out = out / cf_name, out / "helmet_miner_pairs_restored.jsonl"
    cf_out.write_text("".join(json.dumps(r) + "\n" for r in restored_groups))
    pairs_out.write_text("".join(json.dumps(r) + "\n" for r in restored_pairs.values()))
    receipt = {"protocol": "fixed_original_first30_regenerated_mask_diagnostic" if data_name == "cycle003" else "fixed_training_selection_regenerated_masks", "groups": len(restored_groups),
               "identity_trials": sum(len(g["pair_ids"]) for g in groups), "unique_pairs": len(restored_pairs),
               "selection_manifest_sha256": sha(groups_path), "source_pair_manifest_sha256": sha(pairs_path),
               "archive_sha256": sha(archive), "generator": generator,
               "frozen_manifests": {str(p): sha(p) for p in (cf_out, pairs_out)}, "assets": assets}
    (log / "frozen_assets.json").write_text(json.dumps(receipt, indent=2))
    print("ASSETS_FROZEN_BEFORE_LISA_INFERENCE", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--groups-jsonl")
    parser.add_argument("--data-name", default="cycle003")
    args = parser.parse_args()
    main(args.root, args.groups_jsonl, args.data_name)

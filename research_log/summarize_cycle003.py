"""Paired reporting and deterministic six-case gallery; no prediction changes."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "code/cmllm/scripts"))
from counterfactual_export import condition_image

OUT = ROOT / "outputs/cycle003_supplied_memory"
LOG = ROOT / "research_log/cycle003"
GALLERY = LOG / "gallery"
GALLERY.mkdir(parents=True, exist_ok=True)
conditions = ("clean", "target15_b")
reports = {c: json.loads((OUT / c / "memory_metrics.json").read_text()) for c in conditions}
manifests = {c: {r["group_id"]: r for r in map(json.loads, (OUT / c / "memory_predictions.jsonl").read_text().splitlines())} for c in conditions}
metrics = {c: {r["group_id"]: r for r in reports[c]["groups"]} for c in conditions}
assert list(manifests["clean"]) == list(manifests["target15_b"])
paired = []
for group_id, clean in manifests["clean"].items():
    degraded = manifests["target15_b"][group_id]
    assert clean["query"] == degraded["query"] and clean["seed"] == degraded["seed"]
    assert [t["entity_id"] for t in clean["trials"]] == [t["entity_id"] for t in degraded["trials"]]
    for a, b in zip(clean["trials"], degraded["trials"]):
        for field in ("reference", "target"):
            np.testing.assert_array_equal(np.load(OUT / "clean" / a[field]), np.load(OUT / "target15_b" / b[field]))
    a, b = metrics["clean"][group_id], metrics["target15_b"][group_id]
    delta = float(np.mean(b["identity_margin"]) - np.mean(a["identity_margin"]))
    paired.append({"group_id": group_id, "identity_margin_delta": delta,
                   "target_miou_delta": b["target_miou"] - a["target_miou"],
                   "clean_cmsa": a["cmsa"], "degraded_cmsa": b["cmsa"],
                   "degraded_identity_error": b["identity_error_rate"]})
fields = ("target_miou", "cmsa", "memory_fidelity", "identity_error_rate",
          "mean_identity_margin", "median_identity_margin")
table = [{"metric": k, "clean": reports["clean"]["summary"][k],
          "target15_b": reports["target15_b"]["summary"][k],
          "delta": reports["target15_b"]["summary"][k] - reports["clean"]["summary"][k]}
         for k in fields]

chosen = []


def take(candidates, rule, n):
    for row in candidates:
        if row["group_id"] not in {r["group_id"] for r in chosen}:
            chosen.append({**row, "selection_rule": rule})
            n -= 1
            if n == 0:
                break


take(sorted(paired, key=lambda r: (r["identity_margin_delta"], r["group_id"])), "largest_negative_margin_delta", 2)
take(sorted([r for r in paired if r["degraded_identity_error"] > 0], key=lambda r: (-r["degraded_identity_error"], r["group_id"])), "degraded_identity_error", 2)
take(sorted([r for r in paired if r["clean_cmsa"] == r["degraded_cmsa"] == 1], key=lambda r: r["group_id"]), "stable_success", 2)
if len(chosen) < 6:
    take(sorted(paired, key=lambda r: (r["identity_margin_delta"], r["group_id"])), "fill_remaining_by_margin_delta", 6 - len(chosen))


def panel(image, pred, target, label):
    vis = image.copy()
    vis[pred > 0] = (0.55 * vis[pred > 0] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    contours, _ = cv2.findContours((target > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 3)
    small = Image.fromarray(vis)
    small.thumbnail((640, 360))
    result = Image.new("RGB", (640, 395), "#202020")
    result.paste(small, ((640 - small.width) // 2, 35))
    ImageDraw.Draw(result).text((8, 10), label, fill="white")
    return result


for row in chosen:
    group_id = row["group_id"]
    tiles = []
    for condition in conditions:
        record = manifests[condition][group_id]
        original = np.asarray(Image.open(record["image_path"]).convert("RGB"))
        observed = condition_image(original, condition, record["seed"])
        scores = metrics[condition][group_id]
        predictions = [np.load(OUT / condition / t["prediction"]) for t in record["trials"]]
        if condition == "target15_b":
            same = all(np.array_equal(predictions[0], p) for p in predictions[1:]) and bool(predictions[0].any())
            if row["degraded_cmsa"] == 1:
                row["diagnostic_label"] = "stable_success" if row["clean_cmsa"] == 1 else "degraded_success"
            elif same:
                row["diagnostic_label"] = "memory_ignored_identical_nonempty_predictions"
            elif row["degraded_identity_error"] > 0:
                row["diagnostic_label"] = "identity_swap"
            else:
                row["diagnostic_label"] = "low_quality_without_clear_swap"
        for i, trial in enumerate(record["trials"]):
            target = np.load(OUT / condition / trial["target"])
            label = f"{condition} memory {i} | IoU {scores['correct_iou'][i]:.3f} margin {scores['identity_margin'][i]:+.3f}"
            tiles.append(panel(observed, predictions[i], target, label))
    canvas = Image.new("RGB", (1280, 790))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % 2) * 640, (i // 2) * 395))
    canvas.save(GALLERY / (group_id + ".jpg"), quality=88)
    row["gallery_file"] = "gallery/" + group_id + ".jpg"

report = {"protocol": "supplied identity geometry; condition-dependent REF appearance",
          "groups": len(paired), "identity_trials_per_condition": reports["clean"]["summary"]["num_references"],
          "paired_identity_and_target_masks_equal": True,
          "table": table, "paired_groups": paired, "gallery_selection": chosen}
(LOG / "paired_results.json").write_text(json.dumps(report, indent=2) + "\n")
lines = ["# Cycle 003 paired diagnostic", "", "Regenerated pseudo-masks; frozen w15; no tuning. Delta = target15_b minus clean.", "",
         "| Metric | Clean | target15_b | Delta |", "|---|---:|---:|---:|"]
lines += [f"| {r['metric']} | {r['clean']:.6f} | {r['target15_b']:.6f} | {r['delta']:+.6f} |" for r in table]
lines += ["", "## Deterministic gallery", "", "Predictions: red fill. Corresponding target pseudo-mask: green outline. Rows: clean/degraded; columns: memory A/B.", "",
          "Select 2 largest negative group-mean margin deltas, then up to 2 unused groups with highest degraded IER, then up to 2 stable successes by group ID. Fill shortfall by margin delta; never choose for appearance. Group ID breaks ties."]
for row in chosen:
    lines += ["", f"### {row['group_id']}", "", f"Rule: {row['selection_rule']}; diagnostic: {row['diagnostic_label']}; margin delta {row['identity_margin_delta']:+.6f}.", "", f"![{row['group_id']}]({row['gallery_file']})"]
(LOG / "RESULT.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"groups": len(paired), "table": table, "gallery_groups": [r["group_id"] for r in chosen]}, indent=2))

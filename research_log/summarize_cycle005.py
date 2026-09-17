"""Paired report for the single frozen DE run; CC/DD reused from Cycle 003."""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "code/cmllm/scripts"))
from counterfactual_export import condition_image

log = root / "research_log/cycle005"
log.mkdir(parents=True, exist_ok=True)
folders = {"CC": root / "outputs/cycle003_supplied_memory/clean",
           "DD": root / "outputs/cycle003_supplied_memory/target15_b",
           "DE": root / "outputs/cycle005_supplied_memory/DE"}
reports = {c: json.loads((p / "memory_metrics.json").read_text()) for c, p in folders.items()}
manifests = {c: [json.loads(l) for l in (p / "memory_predictions.jsonl").read_text().splitlines()] for c, p in folders.items()}
for cell, rows in manifests.items():
    assert len(rows) == len(manifests["CC"]) == 30
    for a, b in zip(manifests["CC"], rows):
        for key in ("group_id", "query", "seed", "memory_source", "image_path"):
            assert a[key] == b[key], (cell, key)
        for key in ("model", "precision", "vision_tower", "conversation_mode", "ref_image_mode", "focus_dilate", "focus_background", "image_size", "model_max_length"):
            assert a["provenance"][key] == b["provenance"][key], (cell, key)
        assert len(a["trials"]) == len(b["trials"]) == 2
        for first, second in zip(a["trials"], b["trials"]):
            assert first["entity_id"] == second["entity_id"]
            for field in ("reference", "target"):
                np.testing.assert_array_equal(np.load(folders["CC"] / first[field]), np.load(folders[cell] / second[field]))

receipts = []
for row in manifests["DE"]:
    receipt = row["provenance"]["image_receipt"]
    source = Path(row["image_path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt["source_file_sha256"]
    rgb = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
    degraded = condition_image(rgb, "target15_b", row["seed"])
    assert hashlib.sha256(degraded.tobytes()).hexdigest() == receipt["degraded_rgb_sha256"]
    png = folders["DE"] / receipt["enhanced_png"]
    assert hashlib.sha256(png.read_bytes()).hexdigest() == receipt["enhanced_png_sha256"]
    restored = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    assert list(restored.shape) == receipt["shape"]
    assert hashlib.sha256(restored.tobytes()).hexdigest() == receipt["enhanced_rgb_sha256"]
    receipts.append({"group_id": row["group_id"], "seed": row["seed"], **receipt})

table = {}
for cell, report in reports.items():
    diagonal = np.array([v for g in report["groups"] for v in g["correct_iou"]])
    table[cell] = {**report["summary"], "correct_iou_ge_0_5_count": int((diagonal >= 0.5).sum()),
                   "source": str(folders[cell]), "reused_cycle003": cell != "DE"}
paired = []
for dd, de in zip(reports["DD"]["groups"], reports["DE"]["groups"]):
    assert dd["group_id"] == de["group_id"] and dd["entity_ids"] == de["entity_ids"]
    paired.append({"group_id": dd["group_id"],
                   "target_miou_delta": de["target_miou"] - dd["target_miou"],
                   "mean_identity_margin_delta": float(np.mean(de["identity_margin"]) - np.mean(dd["identity_margin"])),
                   "cmsa_delta": de["cmsa"] - dd["cmsa"]})
counts = {}
for metric in ("target_miou", "cmsa"):
    values = np.array([p[metric + "_delta"] for p in paired])
    counts[metric] = {"improved": int((values > 0).sum()), "worsened": int((values < 0).sum()), "tied": int((values == 0).sum())}
deltas = {m: table["DE"][m] - table["DD"][m] for m in ("target_miou", "mean_identity_margin", "cmsa", "memory_fidelity")}
result = {"cells": table, "DE_minus_DD": deltas, "paired_groups": paired, "group_change_counts": counts,
          "all_cells_matched_query_identity_seed_geometry_targets": True,
          "verified_image_receipts": len(receipts), "enhancer": manifests["DE"][0]["provenance"]["enhancer"],
          "new_inference_cells": ["DE"], "reused_cells": ["CC", "DD"]}
(log / "recovery_results.json").write_text(json.dumps(result, indent=2) + "\n")
(log / "image_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
fields = ("target_miou", "cmsa", "memory_fidelity", "mean_identity_margin", "median_identity_margin", "identity_error_rate", "correct_iou_ge_0_5_count")
lines = ["# Cycle 005: frozen v3-lowseg recovery", "", "Same 30 groups / 60 references; CC and DD reused, only DE newly run.", "", "| Metric | CC | DD | DE |", "|---|---:|---:|---:|"]
for metric in fields:
    lines.append("| " + metric + " | " + " | ".join(f"{table[c][metric]:.6f}" for c in ("CC", "DD", "DE")) + " |")
lines += ["", "DE minus DD: " + "; ".join(f"{m} {v:+.6f}" for m, v in deltas.items()), "", "| Group metric | Improved | Worsened | Tied |", "|---|---:|---:|---:|"]
for metric, values in counts.items():
    lines.append("| " + metric + " | " + " | ".join(str(v) for v in values.values()) + " |")
(log / "RESULT.md").write_text("\n".join(lines) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "paired_groups"}, indent=2))

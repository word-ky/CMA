"""Report the frozen 2x2; reuse Cycle 003 diagonal outputs without rerunning."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
log = root / "research_log/cycle004"
folders = {"CC": root / "outputs/cycle003_supplied_memory/clean",
           "DD": root / "outputs/cycle003_supplied_memory/target15_b",
           "CD": root / "outputs/cycle004_supplied_memory/CD",
           "DC": root / "outputs/cycle004_supplied_memory/DC"}
reports = {c: json.loads((p / "memory_metrics.json").read_text()) for c, p in folders.items()}
manifests = {c: [json.loads(l) for l in (p / "memory_predictions.jsonl").read_text().splitlines()] for c, p in folders.items()}
cc = manifests["CC"]
for cell, rows in manifests.items():
    assert len(rows) == len(cc)
    for a, b in zip(cc, rows):
        for key in ("group_id", "query", "seed", "memory_source", "image_path"):
            assert a[key] == b[key], (cell, key)
        for key in ("model", "precision", "vision_tower", "conversation_mode", "ref_image_mode", "focus_dilate", "focus_background", "image_size", "model_max_length"):
            assert a["provenance"][key] == b["provenance"][key], (cell, key)
        assert len(a["trials"]) == len(b["trials"])
        for first, second in zip(a["trials"], b["trials"]):
            assert first["entity_id"] == second["entity_id"]
            for field in ("reference", "target"):
                np.testing.assert_array_equal(np.load(folders["CC"] / first[field]), np.load(folders[cell] / second[field]))
frozen = json.loads((root / "research_log/cycle003/frozen_assets.json").read_text())
for asset in frozen["assets"]:
    with open(asset["path"], "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == asset["sha256"]
table = {}
for cell in ("CC", "CD", "DC", "DD"):
    report = reports[cell]
    diagonal = np.array([v for g in report["groups"] for v in g["correct_iou"]])
    wrong = np.array([v for g in report["groups"] for v in g["max_wrong_iou"]])
    table[cell] = {**report["summary"], "wrong_gt_correct_count": int((wrong > diagonal).sum()),
                   "both_iou_zero_count": int(((wrong == 0) & (diagonal == 0)).sum()),
                   "source": str(folders[cell]), "reused_cycle003": cell in ("CC", "DD")}
effects = {}
for metric in ("target_miou", "mean_identity_margin"):
    scores = {c: table[c][metric] for c in table}
    effects[metric] = {"main_with_clean_ref_DC_minus_CC": scores["DC"] - scores["CC"],
                       "ref_with_clean_main_CD_minus_CC": scores["CD"] - scores["CC"],
                       "interaction_DD_minus_DC_minus_CD_plus_CC": scores["DD"] - scores["DC"] - scores["CD"] + scores["CC"]}
report = {"cells": table, "effects": effects, "frozen_assets_reverified": len(frozen["assets"]),
          "all_cells_matched_query_identity_seed_geometry_targets": True,
          "new_inference_cells": ["CD", "DC"], "reused_cells": ["CC", "DD"]}
(log / "factorial_results.json").write_text(json.dumps(report, indent=2) + "\n")
fields = ("target_miou", "cmsa", "memory_fidelity", "mean_identity_margin", "median_identity_margin", "identity_error_rate", "wrong_gt_correct_count", "both_iou_zero_count")
lines = ["# Cycle 004: main-image quality × REF-appearance quality", "", "C=clean, D=target15_b; first letter main image, second REF appearance. Same frozen 30 groups, 60 references/cell, supplied geometry. CC/DD reused from Cycle 003; only CD/DC newly run.", "", "| Metric | CC | CD | DC | DD |", "|---|---:|---:|---:|---:|"]
for metric in fields:
    lines.append("| " + metric + " | " + " | ".join(f"{table[c][metric]:.6f}" for c in ("CC", "CD", "DC", "DD")) + " |")
lines += ["", "## Continuous decomposition", "", "| Metric | Main: DC−CC | REF: CD−CC | Interaction: DD−DC−CD+CC |", "|---|---:|---:|---:|"]
for metric, values in effects.items():
    lines.append("| " + metric + " | " + " | ".join(f"{v:+.6f}" for v in values.values()) + " |")
(log / "RESULT.md").write_text("\n".join(lines) + "\n")
print(json.dumps(report, indent=2))

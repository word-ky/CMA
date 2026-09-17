"""Verify frozen assets after inference and count failure signatures from raw masks."""
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
log = root / "research_log/cycle003"
receipt = json.loads((log / "frozen_assets.json").read_text())
for asset in receipt["assets"]:
    with open(asset["path"], "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == asset["sha256"]
for path, digest in receipt["frozen_manifests"].items():
    with open(path, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
outputs = root / "outputs/cycle003_supplied_memory"
predictions = list(outputs.glob("*/raw_masks/*/*_prediction.npy"))
degraded = json.loads((outputs / "target15_b/memory_metrics.json").read_text())
diagonal = np.array([x for r in degraded["groups"] for x in r["correct_iou"]])
wrong = np.array([x for r in degraded["groups"] for x in r["max_wrong_iou"]])
run_id = "20260918-005016-cma-cycle003"
runlog = (root / "runs" / run_id / "train.log").read_text()
started = re.search(r"\[autodl\] started_at=(.*)", runlog)[1]
finished = re.search(r"\[autodl\] finished_at=(.*)", runlog)[1]
result = {
    "run_id": run_id, "started_at": started, "finished_at": finished,
    "wall_seconds": (datetime.datetime.fromisoformat(finished) - datetime.datetime.fromisoformat(started)).total_seconds(),
    "exit_code": int(re.search(r"\[autodl\] exit_code=(\d+)", runlog)[1]),
    "assets_verified_after_inference": len(receipt["assets"]),
    "asset_status_counts": {s: sum(a["status"] == s for a in receipt["assets"]) for s in ("exact_recovered", "regenerated")},
    "raw_prediction_files": len(predictions),
    "freeze_before_first_prediction": (log / "frozen_assets.json").stat().st_mtime < min(p.stat().st_mtime for p in predictions),
    "degraded": {
        "empty_predictions": sum(not np.load(p).any() for p in (outputs / "target15_b/raw_masks").glob("*/*_prediction.npy")),
        "zero_correct_iou": int((diagonal == 0).sum()),
        "all_zero_iou_rows": int(((diagonal == 0) & (wrong == 0)).sum()),
        "wrong_beats_correct": int((wrong > diagonal).sum()),
        "correct_iou_below_05": int((diagonal < 0.5).sum()),
        "mean_wrong_iou": float(wrong.mean()), "max_wrong_iou": float(wrong.max()),
    },
}
(log / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

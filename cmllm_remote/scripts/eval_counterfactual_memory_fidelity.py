"""Offline identity-memory metrics on saved binary masks; no model imports.

One JSONL row groups a fixed image/query/condition and multiple memory identities.
See research_log/COUNTERFACTUAL_MEMORY_EVAL.md for the format and definitions.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def iou_matrix(predictions, targets):
    """S[i,j] = binary IoU(prediction conditioned on memory i, target j)."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.ndim != 3 or predictions.shape != targets.shape or len(targets) < 2:
        raise ValueError("Need matching (N,H,W) masks with N >= 2; no resizing is performed")
    # This interface takes saved binary masks, not raw logits/probabilities.
    for masks in (predictions, targets):
        if not np.isin(masks, [0, 1, 255]).all():
            raise ValueError("Masks must be binary (0/1 or 0/255); threshold logits before export")
    predictions, targets = predictions > 0, targets > 0
    if not targets.reshape(len(targets), -1).any(axis=1).all():
        raise ValueError("Each identity needs a nonempty target mask")
    if any(np.array_equal(targets[i], targets[j]) for i, j in combinations(range(len(targets)), 2)):
        raise ValueError("Identical target masks do not define distinguishable identities")
    scores = np.empty((len(targets), len(targets)), dtype=float)
    for i, pred in enumerate(predictions):
        for j, target in enumerate(targets):
            scores[i, j] = np.count_nonzero(pred & target) / np.count_nonzero(pred | target)
    return scores


def score_masks(predictions, targets, min_iou=0.5):
    """Strict rank fidelity; quality-qualified joint pair success; wrong-ID rate."""
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0,1]")
    scores = iou_matrix(predictions, targets)
    diagonal = np.diag(scores)
    off_diagonal = scores.copy()
    np.fill_diagonal(off_diagonal, -1)
    wrong = off_diagonal.max(axis=1)
    fidelity = diagonal > wrong  # ties, including all-zero rows, fail
    qualified = fidelity & (diagonal >= min_iou)
    identity_error = (wrong > diagonal) & (wrong >= min_iou)
    pairs = [
        {"indices": [i, j], "success": bool(qualified[i] and qualified[j])}
        for i, j in combinations(range(len(scores)), 2)
    ]
    return {
        "iou_matrix": scores.tolist(),
        "correct_iou": diagonal.tolist(),
        "max_wrong_iou": wrong.tolist(),
        "fidelity": fidelity.tolist(),
        "identity_error": identity_error.tolist(),
        "pairs": pairs,
        "num_references": len(scores),
        "num_pairs": len(pairs),
        "target_miou": float(diagonal.mean()),
        "memory_fidelity": float(fidelity.mean()),
        "cmsa": sum(pair["success"] for pair in pairs) / len(pairs),
        "identity_error_rate": float(identity_error.mean()),
    }


def load_mask(value, base_dir):
    """Inline 2D binary array or .npy / single-channel image path."""
    if not isinstance(value, str):
        return np.asarray(value)
    path = Path(base_dir) / value
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image).copy()


def evaluate_record(record, base_dir, min_iou=0.5):
    trials = record["trials"]
    entity_ids = [trial["entity_id"] for trial in trials]
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError("Each reference identity must appear once within a group")
    # The shared image and query live at group level: only memory identity varies.
    result = {
        "group_id": record["group_id"],
        "image_id": record["image_id"],
        "query": record["query"],
        "condition": record["condition"],
        "memory_source": record["memory_source"],
        "entity_ids": entity_ids,
    }
    predictions = [load_mask(t["prediction"], base_dir) for t in trials]
    targets = [load_mask(t["target"], base_dir) for t in trials]
    result.update(score_masks(predictions, targets, min_iou))
    return result


def summarize(rows):
    """Reference-weighted scores and pair-weighted CMSA (not mean of group means)."""
    count = sum(row["num_references"] for row in rows)
    pairs = [pair for row in rows for pair in row["pairs"]]
    return {
        "num_groups": len(rows),
        "num_references": count,
        "num_pairs": len(pairs),
        "target_miou": sum(sum(r["correct_iou"]) for r in rows) / count,
        "memory_fidelity": sum(sum(r["fidelity"]) for r in rows) / count,
        "cmsa": sum(p["success"] for p in pairs) / len(pairs),
        "identity_error_rate": sum(sum(r["identity_error"]) for r in rows) / count,
    }


def evaluate_file(input_path, output_path, min_iou=0.5):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must not overwrite the prediction manifest")
    rows = []
    with input_path.open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                rows.append(evaluate_record(json.loads(line), input_path.parent, min_iou))
    if not rows:
        raise ValueError("No counterfactual groups in input")
    keys = [(r["group_id"], r["condition"], r["memory_source"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate group/condition/memory_source would double-count trials")
    strata = sorted({(r["condition"], r["memory_source"]) for r in rows})
    report = {
        "protocol": "counterfactual_memory_fidelity_v1",
        "min_iou": min_iou,
        "summary": summarize(rows),
        "by_condition_and_memory_source": [
            {"condition": condition, "memory_source": source,
             **summarize([r for r in rows if (r["condition"], r["memory_source"]) == (condition, source)])}
            for condition, source in strata
        ],
        "groups": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Grouped saved-mask JSONL")
    parser.add_argument("--output", required=True, help="Metric report JSON")
    parser.add_argument("--min-iou", type=float, default=0.5)
    args = parser.parse_args()
    report = evaluate_file(args.input, args.output, args.min_iou)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

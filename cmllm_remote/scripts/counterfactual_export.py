"""Small frozen-evaluator export seam, independent of model loading."""

import hashlib
from pathlib import Path

import numpy as np


TARGET15_B = {"id": "target15_b", "gamma": 2.15, "scale": 0.405,
              "contrast": 0.655, "noise_sigma": 23.5, "blur": 0.88}


def group_seed(group_id, seed=0):
    return int(hashlib.md5(f"{seed}:{group_id}".encode("utf-8")).hexdigest()[:8], 16)


def condition_image(image_rgb, condition, seed):
    if condition == "clean":
        return image_rgb.copy()
    if condition != "target15_b":
        raise ValueError("Only clean and target15_b are supported")
    import cv2

    # Exact operations from stage3_rule_controller_v3_seg_local_enhance.py::
    # degrade_parametric at 5f4c2ba. Kept model-free to avoid importing LISA/CUDA.
    cfg = TARGET15_B
    rng = np.random.default_rng(seed)
    x = image_rgb.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0.0, 1.0), cfg["gamma"])
    x = (x - 0.5) * cfg["contrast"] + 0.5
    x = x * cfg["scale"]
    x = x + rng.normal(0.0, cfg["noise_sigma"] / 255.0, x.shape).astype(np.float32)
    out = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    k = max(1, int(round(cfg["blur"] * 2 + 1)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(out, (k, k), cfg["blur"])


def export_group(out_dir, index, cf, predictions, targets, references, *, condition, seed,
                 provenance):
    """Save every output as produced; never choose a mask using ground truth."""
    n = len(cf["pair_ids"])
    if not (len(predictions) == len(targets) == len(references) == n):
        raise ValueError("Prediction/target/reference counts must match ordered pair_ids")
    relative_dir = Path("raw_masks") / f"{index:05d}"
    folder = Path(out_dir) / relative_dir
    folder.mkdir(parents=True, exist_ok=True)
    trials = []
    for i, pair_id in enumerate(cf["pair_ids"]):
        trial = {"entity_id": pair_id, "pair_id": pair_id}
        for field, masks in (("prediction", predictions), ("target", targets), ("reference", references)):
            filename = f"{i:02d}_{field}.npy"
            np.save(folder / filename, (np.asarray(masks[i]) > 0).astype(np.uint8), allow_pickle=False)
            trial[field] = (relative_dir / filename).as_posix()
        trials.append(trial)
    return {
        "group_id": cf["counterfactual_id"], "counterfactual_id": cf["counterfactual_id"],
        "image_id": cf["image_path"], "image_path": cf["image_path"],
        "query": cf["same_round2_query"], "condition": condition,
        "memory_source": "supplied_ref", "seed": seed,
        "degradation_config": TARGET15_B if condition == "target15_b" else None,
        "provenance": provenance, "trials": trials,
    }

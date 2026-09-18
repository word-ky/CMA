"""Existing v2 miner-to-helmet geometry, moved unchanged for model-free feedback."""
import numpy as np


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

def helmet_geometry(pred_helmet, miner_mask):
    pred_bbox = bbox_from_mask(pred_helmet)
    miner_bbox = bbox_from_mask(miner_mask)
    if pred_bbox is None or miner_bbox is None:
        return {
            "helmet_center_in_miner": False,
            "helmet_center_in_head": False,
            "helmet_head_score": 0.0,
        }
    x1, y1, x2, y2 = pred_bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    mx1, my1, mx2, my2 = miner_bbox
    mh = max(1, my2 - my1)
    in_miner = mx1 <= cx <= mx2 and my1 <= cy <= my2
    strict_y = my1 + 0.45 * mh
    loose_y = my1 + 0.60 * mh
    in_strict = in_miner and cy <= strict_y
    in_loose = in_miner and cy <= loose_y
    if in_strict:
        score = 1.0
    elif in_loose:
        score = 0.7
    elif in_miner:
        score = 0.3
    else:
        score = 0.0
    return {
        "helmet_center_in_miner": bool(in_miner),
        "helmet_center_in_head": bool(in_loose),
        "helmet_head_score": float(score),
    }

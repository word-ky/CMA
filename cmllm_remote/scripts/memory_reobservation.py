"""Fixed 1.25x supplied-miner ROI; xyxy uses half-open pixel coordinates."""
import math
import numpy as np


def memory_roi(bbox_xyxy, image_hw):
    h, w = image_hw
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    rx, ry = (x2-x1)*1.25/2, (y2-y1)*1.25/2
    return (max(0, math.floor(cx-rx)), max(0, math.floor(cy-ry)),
            min(w, math.ceil(cx+rx)), min(h, math.ceil(cy+ry)))


def crop_to_roi(array, roi):
    x1, y1, x2, y2 = roi
    return array[y1:y2, x1:x2].copy()


def bbox_in_roi(bbox_xyxy, roi):
    x1, y1, x2, y2 = roi
    a, b, c, d = bbox_xyxy
    return [max(0., a-x1), max(0., b-y1), min(float(x2-x1), c-x1), min(float(y2-y1), d-y1)]


def restore_mask(mask, roi, image_hw):
    # Model postprocessing already returns native crop resolution.
    x1, y1, x2, y2 = roi
    result = np.zeros(image_hw, dtype=mask.dtype)
    result[y1:y2, x1:x2] = mask
    return result

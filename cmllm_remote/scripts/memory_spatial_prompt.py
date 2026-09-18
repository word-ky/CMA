"""Supplied miner geometry in SAM's resized, unpadded pixel coordinates."""
import numpy as np
import torch


def spatial_memory_boxes(bboxes_xyxy, image_hw, transform, multiround):
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32).copy()
    h, w = image_hw
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
    boxes = transform.apply_boxes(boxes, image_hw)
    # [miner A, helmet A, miner B, helmet B] for v1_multiround.
    if multiround:
        boxes = np.repeat(boxes, 2, axis=0)
    return torch.from_numpy(boxes).float()


def inference_shape_only_targets(item, eval_pred_indices):
    # Keep original targets only in the evaluator's export/scoring return value.
    fields = list(item)
    fields[4] = item[4].clone()
    fields[4][eval_pred_indices] = 0
    return tuple(fields)

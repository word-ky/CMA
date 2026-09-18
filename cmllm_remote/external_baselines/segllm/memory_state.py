"""Supplied-memory seed for the pinned SegLLM online inference interface.

Crop/padding/box conventions follow inference_cli.py lines 260-302 at
4593a069f09628ce3a5b46e657f5417fefd7be46 (Apache-2.0 notice in that file).
Calls native HIPIE and CLIP processors; no model or target selection.
"""
import hashlib

import cv2
import numpy as np
import torch
from PIL import Image


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def seed_memory(image_rgb, miner_mask, miner_bbox_xyxy, native_processor, clip_processor):
    processed = native_processor(image_rgb, [miner_mask.astype(np.uint8)])
    native_rgb = processed['image'].permute(1, 2, 0).numpy()
    native_mask = processed['mask'][0].numpy().astype(np.uint8)
    box = native_processor.transform.apply_boxes(
        np.asarray([miner_bbox_xyxy], dtype=np.float32), image_rgb.shape[:2]
    )[0]
    x0, y0, x1, y1 = np.clip(box.astype(int), 0, native_rgb.shape[0])
    x1 = np.clip(x1, x0 + 2, native_rgb.shape[0])
    y1 = np.clip(y1, y0 + 2, native_rgb.shape[0])
    crop = cv2.bitwise_and(native_rgb, native_rgb, mask=native_mask)[y0:y1, x0:x1]
    side = max(x1 - x0, y1 - y0)
    padded = np.zeros((side, side, 3), dtype=native_rgb.dtype)
    padded[:crop.shape[0], :crop.shape[1]] = crop
    appearance = torch.tensor(clip_processor(Image.fromarray(padded.astype(np.uint8))).pixel_values[0])
    bbox = torch.tensor([y0, x0, y1, x1]) / 1024.0
    receipt = {
        'original_hw': list(image_rgb.shape[:2]), 'native_input_hw': list(processed['input_size']),
        'native_padded_hw': list(native_rgb.shape[:2]),
        'supplied_bbox_xyxy': list(miner_bbox_xyxy),
        'resized_bbox_xyxy_float': box.tolist(),
        'crop_bbox_xyxy_int': [int(x) for x in (x0, y0, x1, y1)],
        'box_encode_yxyx_over_1024': bbox.tolist(),
        'crop_hw': list(crop.shape[:2]), 'square_pad_hw': list(padded.shape[:2]),
        'appearance_shape': list(appearance.shape), 'appearance_sha256': tensor_hash(appearance),
        'bbox_sha256': tensor_hash(bbox),
        'main_rgb_sha256': hashlib.sha256(image_rgb.tobytes()).hexdigest(),
        'masked_crop_rgb_sha256': hashlib.sha256(padded.tobytes()).hexdigest(),
    }
    return appearance, bbox, receipt

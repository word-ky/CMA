#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

LISA_ROOT = "/home/wjq/cmllm/third_party/LISA"
if LISA_ROOT not in sys.path:
    sys.path.insert(0, LISA_ROOT)

from model.LISA import LISAForCausalLM  # noqa: E402
from model.llava import conversation as conversation_lib  # noqa: E402
from model.segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402
from utils.dataset import collate_fn  # noqa: E402


DEFAULT_IMAGE_TOKEN = "<image>"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_seed(text):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def get_added_token_id(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0 or token_id == tokenizer.unk_token_id:
        token_id = tokenizer(token, add_special_tokens=False).input_ids[-1]
    return token_id


def preprocess_sam(x, image_size=1024):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    return F.pad(x, (0, image_size - w, 0, image_size - h))


def read_mask(mask_path, image_hw):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    h, w = image_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def clamp_bbox(bbox, image_hw):
    h, w = image_hw
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return [x1, y1, x2, y2]


def norm_bbox(bbox_xyxy, image_hw):
    if bbox_xyxy is None:
        return [0.0, 0.0, 0.0, 0.0]
    h, w = image_hw
    x1, y1, x2, y2 = [float(x) for x in bbox_xyxy]
    return [
        max(0.0, min(1.0, x1 / max(w, 1))),
        max(0.0, min(1.0, y1 / max(h, 1))),
        max(0.0, min(1.0, x2 / max(w, 1))),
        max(0.0, min(1.0, y2 / max(h, 1))),
    ]


def iou(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / (union + 1e-8))


def mask_stats(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape[:2]
    area = int(mask_u8.sum())
    n, labels = cv2.connectedComponents(mask_u8)
    components = max(0, n - 1)
    largest_component_ratio = 0.0
    if components > 0:
        counts = np.bincount(labels.reshape(-1))[1:]
        largest_component_ratio = float(counts.max() / max(area, 1))
    bbox = bbox_from_mask(mask_u8)
    touches_edge = False
    if area > 0:
        touches_edge = bool(
            mask_u8[0, :].any()
            or mask_u8[-1, :].any()
            or mask_u8[:, 0].any()
            or mask_u8[:, -1].any()
        )
    return {
        "area": area,
        "area_ratio": float(area / max(h * w, 1)),
        "connected_components": int(components),
        "largest_component_ratio": largest_component_ratio,
        "bbox_xyxy": bbox,
        "touches_edge": touches_edge,
    }


def image_quality(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "laplacian_std": float(lap.std() / 255.0),
        "dark_pixel_ratio": float((gray < 35).mean()),
        "bright_pixel_ratio": float((gray > 235).mean()),
    }


def degrade_image(image_rgb, severity, seed):
    rng = np.random.default_rng(seed)
    blur_ksize = 0
    if severity == "extreme":
        gamma, scale, sigma, blur_ksize = 3.80, 0.25, 40.0, 3
    elif severity == "hard":
        gamma, scale, sigma, blur_ksize = 3.10, 0.32, 30.0, 3
    elif severity == "medium":
        gamma, scale, sigma = 2.20, 0.45, 20.0
    else:
        gamma, scale, sigma = 2.60, 0.38, 25.0
    x = image_rgb.astype(np.float32) / 255.0
    x = np.power(x, gamma) * scale
    noise = rng.normal(0.0, sigma / 255.0, x.shape).astype(np.float32)
    x = np.clip(x + noise, 0.0, 1.0)
    out = (x * 255.0).astype(np.uint8)
    if blur_ksize > 0:
        out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)
    return out


def enhance_lowlight(image_rgb, strength="normal"):
    gamma = 0.50 if strength == "normal" else 0.42
    x = image_rgb.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0.0, 1.0), gamma)
    bright = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(bright, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clip_limit = 2.0 if strength == "normal" else 2.8
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    h = 7 if strength == "normal" else 5
    return cv2.fastNlMeansDenoisingColored(enhanced, None, h, h, 7, 21)


def dilate_mask(mask, ksize):
    if ksize <= 0:
        return mask > 0
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1) > 0


def focus_blackout(image_rgb, mask, dilate=21, background=0.0):
    keep = dilate_mask(mask, dilate)
    out = (image_rgb.astype(np.float32) * float(background)).astype(np.uint8)
    out[keep] = image_rgb[keep]
    return out


def make_ref_crop_clip(image_rgb, mask, bbox_xyxy, clip_processor):
    if bbox_xyxy is None:
        bbox_xyxy = bbox_from_mask(mask)
    if bbox_xyxy is None:
        return torch.zeros(3, 224, 224)
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox_xyxy, (h, w))
    masked = image_rgb.copy()
    masked[mask <= 0] = 0
    crop = masked[y1:y2, x1:x2]
    side = max(crop.shape[0], crop.shape[1], 1)
    padded = np.zeros((side, side, 3), dtype=crop.dtype)
    padded[: crop.shape[0], : crop.shape[1]] = crop
    return clip_processor.preprocess(padded, return_tensors="pt")["pixel_values"][0]


def build_conversation(query, use_ref):
    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    prefix = "[REF] " if use_ref else ""
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prefix + query.strip())
    conv.append_message(conv.roles[1], "[SEG].")
    return conv.get_prompt()


class ControllerRunner:
    def __init__(self, args):
        self.args = args
        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens(["[SEG]", "[REF]"])
        seg_token_idx = get_added_token_id(self.tokenizer, "[SEG]")
        ref_token_idx = get_added_token_id(self.tokenizer, "[REF]")

        self.dtype = torch.float32
        if args.precision == "bf16":
            self.dtype = torch.bfloat16
        elif args.precision == "fp16":
            self.dtype = torch.float16

        self.model = LISAForCausalLM.from_pretrained(
            args.model,
            low_cpu_mem_usage=False,
            torch_dtype=self.dtype,
            vision_tower=args.vision_tower,
            vision_pretrained=args.vision_pretrained,
            seg_token_idx=seg_token_idx,
            ref_token_idx=ref_token_idx,
        )
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        self.model.get_model().get_vision_tower().to(dtype=self.dtype)
        if args.precision == "bf16":
            self.model = self.model.bfloat16().cuda()
        elif args.precision == "fp16":
            self.model = self.model.half().cuda()
        else:
            self.model = self.model.float().cuda()
        self.model.get_model().get_vision_tower().to(device=0)
        self.model.eval()

        self.clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
        self.transform = ResizeLongestSide(args.image_size)

    def build_item(self, image_rgb, query, target_mask, ref_mask=None, ref_bbox=None):
        ori_size = image_rgb.shape[:2]
        use_ref = ref_mask is not None
        conversation = build_conversation(query, use_ref=use_ref)

        if ref_mask is None:
            ref_mask = np.zeros_like(target_mask, dtype=np.float32)
            ref_bbox_norm = [0.0, 0.0, 0.0, 0.0]
            ref_valid = 0.0
            ref_image_clip = torch.zeros(3, 224, 224)
        else:
            if ref_bbox is None:
                ref_bbox = bbox_from_mask(ref_mask)
            ref_bbox_norm = norm_bbox(ref_bbox, ori_size)
            ref_valid = 1.0
            ref_image_clip = make_ref_crop_clip(
                image_rgb, ref_mask, ref_bbox, self.clip_processor
            )

        image_clip = self.clip_processor.preprocess(image_rgb, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_sam = self.transform.apply_image(image_rgb)
        resize = image_sam.shape[:2]
        image_sam = preprocess_sam(
            torch.from_numpy(image_sam).permute(2, 0, 1).contiguous(),
            image_size=self.args.image_size,
        )
        masks = torch.from_numpy(target_mask[None, :, :]).float()
        labels = torch.ones(masks.shape[1], masks.shape[2]) * 255
        item = (
            "stage3_inline_image",
            image_sam,
            image_clip,
            [conversation],
            masks,
            labels,
            resize,
            [query],
            ["stage3_target"],
            torch.from_numpy(ref_mask[None, :, :]).float(),
            torch.tensor([ref_bbox_norm], dtype=torch.float32),
            torch.tensor([ref_valid], dtype=torch.float32),
            torch.ones((1,), dtype=torch.float32),
            ref_image_clip[None, :, :, :],
            True,
        )
        return item

    def predict(self, image_rgb, query, target_mask, ref_mask=None, ref_bbox=None):
        item = self.build_item(image_rgb, query, target_mask, ref_mask, ref_bbox)
        batch = collate_fn(
            [item],
            tokenizer=self.tokenizer,
            conv_type="llava_v1",
            use_mm_start_end=True,
            local_rank=0,
        )
        for key in ["images", "images_clip", "input_ids", "labels", "attention_masks", "offset"]:
            batch[key] = batch[key].cuda(non_blocking=True)
        for key in [
            "masks_list",
            "label_list",
            "ref_masks_list",
            "ref_bboxes_list",
            "ref_valids_list",
            "mask_weights_list",
            "ref_images_clip_list",
        ]:
            batch[key] = [
                x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x
                for x in batch[key]
            ]
        batch["images"] = batch["images"].to(dtype=self.dtype)
        batch["images_clip"] = batch["images_clip"].to(dtype=self.dtype)

        with torch.no_grad():
            output = self.model(**batch)
        pred = output["pred_masks"][0][0].detach().float().cpu().numpy()
        return (pred > 0).astype(np.float32)


def overlay_mask(image_rgb, mask, color, alpha=0.55):
    out = image_rgb.copy()
    m = mask.astype(bool)
    if m.any():
        out[m] = (out[m] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def draw_bbox(image_rgb, bbox, color):
    out = image_rgb.copy()
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def put_label(canvas, text):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text[:78],
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def tile(image_rgb, label, width=390, height=250):
    h, w = image_rgb.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 240, dtype=np.uint8)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return put_label(canvas, label)


def save_flow_visual(path, panels):
    rows = []
    for i in range(0, len(panels), 3):
        row = np.concatenate(panels[i : i + 3], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


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


def load_episode_assets(ep):
    image_bgr = cv2.imread(ep["image_path"])
    if image_bgr is None:
        raise FileNotFoundError(ep["image_path"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ori_size = image_rgb.shape[:2]
    round1, round2 = ep["rounds"]
    miner_mask = read_mask(round1["target_mask"], ori_size)
    helmet_mask = read_mask(round2["target_mask"], ori_size)
    return image_rgb, miner_mask, helmet_mask, round1, round2


def score_seg(pred, gt):
    stats = mask_stats(pred)
    score = iou(pred, gt)
    return {
        "iou": score,
        **stats,
    }


def run_episode(runner, ep, args, idx, overlay_dir):
    image_rgb, miner_gt, helmet_gt, round1, round2 = load_episode_assets(ep)
    seed = stable_seed(ep.get("episode_id", ep["image_path"]))
    severity = args.degradation
    if severity == "mixed":
        severity = "hard" if seed % 2 else "medium"
    degraded = degrade_image(image_rgb, severity, seed)
    enhanced = enhance_lowlight(degraded, "normal")
    enhanced_strong = enhance_lowlight(degraded, "strong")

    steps = []
    panels = []
    gt_panel = overlay_mask(image_rgb, miner_gt, (0, 140, 255), 0.45)
    gt_panel = overlay_mask(gt_panel, helmet_gt, (255, 40, 40), 0.55)
    panels.append(tile(gt_panel, "clean GT: miner blue, helmet red"))
    panels.append(tile(degraded, f"degraded input: {severity}"))

    round1_query = round1.get(
        "query", "Please segment the miner wearing a helmet in the underground mining image."
    )
    round2_query = round2.get(
        "query",
        "Based on the miner segmented in the previous round, segment the helmet on his head.",
    )

    pred_miner_degraded = runner.predict(degraded, round1_query, miner_gt)
    miner_degraded_score = score_seg(pred_miner_degraded, miner_gt)
    degraded_quality = image_quality(degraded)
    miner_ok = miner_degraded_score["iou"] >= args.miner_accept_iou
    steps.append(
        {
            "step": 1,
            "image_state": "degraded",
            "action": "segment_miner",
            "target": "coal_miner",
            "quality": miner_degraded_score,
            "image_quality": degraded_quality,
            "decision": "accept" if miner_ok else "failed",
        }
    )
    miner_panel = overlay_mask(degraded, miner_gt, (0, 180, 0), 0.35)
    miner_panel = overlay_mask(miner_panel, pred_miner_degraded, (0, 120, 255), 0.55)
    panels.append(tile(miner_panel, f"miner on degraded IoU={miner_degraded_score['iou']:.3f}"))

    accepted_image = degraded
    accepted_miner = pred_miner_degraded
    accepted_state = "degraded"
    accepted_miner_score = miner_degraded_score
    if not miner_ok:
        low_light = degraded_quality["brightness"] < args.low_brightness
        action = "enhance_lowlight" if low_light else "enhance_denoise"
        steps.append(
            {
                "step": 2,
                "image_state": "degraded",
                "action": action,
                "reason": "miner failed and image is low-light/noisy",
                "decision": "execute",
            }
        )
        panels.append(tile(enhanced, "operation: enhance low-light/denoise"))
        pred_miner_enhanced = runner.predict(enhanced, round1_query, miner_gt)
        miner_enhanced_score = score_seg(pred_miner_enhanced, miner_gt)
        miner_ok = miner_enhanced_score["iou"] >= args.miner_accept_iou
        steps.append(
            {
                "step": 3,
                "image_state": "enhanced",
                "action": "segment_miner",
                "target": "coal_miner",
                "quality": miner_enhanced_score,
                "image_quality": image_quality(enhanced),
                "improvement_over_degraded": float(
                    miner_enhanced_score["iou"] - miner_degraded_score["iou"]
                ),
                "decision": "accept" if miner_ok else "failed",
            }
        )
        miner_panel2 = overlay_mask(enhanced, miner_gt, (0, 180, 0), 0.35)
        miner_panel2 = overlay_mask(miner_panel2, pred_miner_enhanced, (0, 120, 255), 0.55)
        panels.append(tile(miner_panel2, f"miner after enhance IoU={miner_enhanced_score['iou']:.3f}"))
        if miner_enhanced_score["iou"] > accepted_miner_score["iou"]:
            accepted_image = enhanced
            accepted_miner = pred_miner_enhanced
            accepted_state = "enhanced"
            accepted_miner_score = miner_enhanced_score

        if not miner_ok:
            steps.append(
                {
                    "step": len(steps) + 1,
                    "image_state": "enhanced",
                    "action": "enhance_lowlight_strong",
                    "reason": "normal enhancement did not recover miner segmentation",
                    "decision": "execute",
                }
            )
            panels.append(tile(enhanced_strong, "operation: stronger enhancement"))
            pred_miner_strong = runner.predict(enhanced_strong, round1_query, miner_gt)
            miner_strong_score = score_seg(pred_miner_strong, miner_gt)
            miner_ok = miner_strong_score["iou"] >= args.miner_accept_iou
            steps.append(
                {
                    "step": len(steps) + 1,
                    "image_state": "enhanced_strong",
                    "action": "segment_miner",
                    "target": "coal_miner",
                    "quality": miner_strong_score,
                    "image_quality": image_quality(enhanced_strong),
                    "improvement_over_degraded": float(
                        miner_strong_score["iou"] - miner_degraded_score["iou"]
                    ),
                    "improvement_over_enhanced": float(
                        miner_strong_score["iou"] - miner_enhanced_score["iou"]
                    ),
                    "decision": "accept" if miner_ok else "failed",
                }
            )
            miner_panel3 = overlay_mask(enhanced_strong, miner_gt, (0, 180, 0), 0.35)
            miner_panel3 = overlay_mask(miner_panel3, pred_miner_strong, (0, 120, 255), 0.55)
            panels.append(tile(miner_panel3, f"miner strong enhance IoU={miner_strong_score['iou']:.3f}"))
            if miner_strong_score["iou"] > accepted_miner_score["iou"]:
                accepted_image = enhanced_strong
                accepted_miner = pred_miner_strong
                accepted_state = "enhanced_strong"
                accepted_miner_score = miner_strong_score

    if not miner_ok and accepted_miner_score["iou"] < args.miner_min_continue_iou:
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": accepted_state,
                "action": "stop_failed_miner",
                "reason": "no reliable miner reference for helmet stage",
                "decision": "stop",
            }
        )
        while len(panels) < 9:
            panels.append(tile(accepted_image, "stopped: miner ref unreliable"))
        out_vis = overlay_dir / f"{idx:05d}_{ep.get('episode_id','episode')}.jpg"
        save_flow_visual(out_vis, panels[:9])
        return {
            "episode_id": ep.get("episode_id"),
            "image_path": ep["image_path"],
            "severity": severity,
            "steps": steps,
            "final_decision": "failed_miner",
            "final_miner_iou": float(accepted_miner_score["iou"]),
            "final_helmet_iou": 0.0,
            "visualization": str(out_vis),
        }

    focus_image = focus_blackout(accepted_image, accepted_miner, dilate=args.focus_dilate, background=0.0)
    steps.append(
        {
            "step": len(steps) + 1,
            "image_state": accepted_state,
            "action": "focus_blackout",
            "ref": "accepted_miner",
            "decision": "execute",
        }
    )
    panels.append(tile(focus_image, "operation: focus/blackout around accepted miner"))

    pred_helmet_focus = runner.predict(
        focus_image,
        round2_query,
        helmet_gt,
        ref_mask=accepted_miner,
        ref_bbox=bbox_from_mask(accepted_miner),
    )
    helmet_focus_quality = score_seg(pred_helmet_focus, helmet_gt)
    helmet_focus_quality.update(helmet_geometry(pred_helmet_focus, accepted_miner))
    helmet_ok = (
        helmet_focus_quality["iou"] >= args.helmet_accept_iou
        and helmet_focus_quality["helmet_head_score"] >= args.helmet_head_min_score
    )
    steps.append(
        {
            "step": len(steps) + 1,
            "image_state": "focus_blackout",
            "action": "segment_helmet",
            "target": "mining_helmet",
            "quality": helmet_focus_quality,
            "decision": "accept" if helmet_ok else "failed",
        }
    )
    helmet_panel = overlay_mask(focus_image, helmet_gt, (0, 180, 0), 0.35)
    helmet_panel = overlay_mask(helmet_panel, pred_helmet_focus, (255, 40, 40), 0.55)
    panels.append(tile(helmet_panel, f"helmet focus IoU={helmet_focus_quality['iou']:.3f}"))

    final_helmet = pred_helmet_focus
    final_helmet_quality = helmet_focus_quality
    final_state = "focus_blackout"
    if not helmet_ok:
        if helmet_focus_quality["helmet_head_score"] <= 0.0:
            rollback_reason = "target appears inconsistent with the accepted reference"
        elif helmet_focus_quality["area_ratio"] <= 1e-5:
            rollback_reason = "target mask is empty or too small"
        else:
            rollback_reason = "target quality below accept threshold"
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": "focus_blackout",
                "action": "rollback_to_accepted_ref",
                "rollback_target": accepted_state,
                "failed_ref": "failed_target_focus",
                "reason": rollback_reason,
                "decision": "execute",
            }
        )
        panels.append(tile(accepted_image, "rollback: accepted image/ref retry"))
        pred_helmet_retry = runner.predict(
            accepted_image,
            round2_query,
            helmet_gt,
            ref_mask=accepted_miner,
            ref_bbox=bbox_from_mask(accepted_miner),
        )
        helmet_retry_quality = score_seg(pred_helmet_retry, helmet_gt)
        helmet_retry_quality.update(helmet_geometry(pred_helmet_retry, accepted_miner))
        improved = helmet_retry_quality["iou"] >= helmet_focus_quality["iou"]
        final_ok = (
            helmet_retry_quality["iou"] >= args.helmet_accept_iou
            and helmet_retry_quality["helmet_head_score"] >= args.helmet_head_min_score
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": "rollback_accepted_state",
                "action": "segment_helmet",
                "target": "mining_helmet",
                "quality": helmet_retry_quality,
                "improvement_over_failed_focus": float(
                    helmet_retry_quality["iou"] - helmet_focus_quality["iou"]
                ),
                "decision": "accept" if final_ok else "failed",
            }
        )
        helmet_panel2 = overlay_mask(accepted_image, helmet_gt, (0, 180, 0), 0.35)
        helmet_panel2 = overlay_mask(helmet_panel2, pred_helmet_retry, (255, 40, 40), 0.55)
        panels.append(tile(helmet_panel2, f"target after rollback IoU={helmet_retry_quality['iou']:.3f}"))
        if improved:
            final_helmet = pred_helmet_retry
            final_helmet_quality = helmet_retry_quality
            final_state = "rollback_accepted_state"
            helmet_ok = final_ok

    while len(panels) < 9:
        final_panel = overlay_mask(accepted_image, helmet_gt, (0, 180, 0), 0.35)
        final_panel = overlay_mask(final_panel, final_helmet, (255, 40, 40), 0.55)
        panels.append(tile(final_panel, f"final state={final_state}"))

    out_vis = overlay_dir / f"{idx:05d}_{ep.get('episode_id','episode')}.jpg"
    save_flow_visual(out_vis, panels[:9])
    final_decision = "accepted" if helmet_ok else "failed_helmet"
    return {
        "episode_id": ep.get("episode_id"),
        "image_path": ep["image_path"],
        "severity": severity,
        "steps": steps,
        "accepted_miner_state": accepted_state,
        "final_state": final_state,
        "final_decision": final_decision,
        "final_miner_iou": float(accepted_miner_score["iou"]),
        "final_helmet_iou": float(final_helmet_quality["iou"]),
        "visualization": str(out_vis),
    }


def summarize(rows):
    n = len(rows)
    miner_initial = []
    miner_final = []
    helmet_final = []
    helmet_initial = []
    enhanced_recover = 0
    enhanced_attempts = 0
    rollback_attempts = 0
    rollback_improved = 0
    accepted = 0
    failed_miner = 0
    failed_helmet = 0
    for row in rows:
        miner_final.append(row.get("final_miner_iou", 0.0))
        helmet_final.append(row.get("final_helmet_iou", 0.0))
        if row.get("final_decision") == "accepted":
            accepted += 1
        elif row.get("final_decision") == "failed_miner":
            failed_miner += 1
        elif row.get("final_decision") == "failed_helmet":
            failed_helmet += 1
        first_miner = None
        first_helmet = None
        for step in row["steps"]:
            if step["action"] == "segment_miner" and first_miner is None:
                first_miner = step["quality"]["iou"]
            if step["action"] == "segment_helmet" and first_helmet is None:
                first_helmet = step["quality"]["iou"]
            if step["action"] in {"enhance_lowlight", "enhance_denoise", "enhance_lowlight_strong"}:
                enhanced_attempts += 1
            if step["action"] == "segment_miner" and str(step.get("image_state", "")).startswith("enhanced"):
                if step["decision"] == "accept":
                    enhanced_recover += 1
            if step["action"] == "rollback_to_accepted_ref":
                rollback_attempts += 1
            if step["action"] == "segment_helmet" and step.get("image_state") == "rollback_accepted_state":
                if step.get("improvement_over_failed_focus", 0.0) > 0:
                    rollback_improved += 1
        if first_miner is not None:
            miner_initial.append(first_miner)
        if first_helmet is not None:
            helmet_initial.append(first_helmet)
    return {
        "num_episodes": n,
        "accepted": accepted,
        "failed_miner": failed_miner,
        "failed_helmet": failed_helmet,
        "accept_rate": float(accepted / max(n, 1)),
        "mean_initial_miner_iou": float(np.mean(miner_initial)) if miner_initial else 0.0,
        "mean_final_miner_iou": float(np.mean(miner_final)) if miner_final else 0.0,
        "mean_initial_helmet_iou": float(np.mean(helmet_initial)) if helmet_initial else 0.0,
        "mean_final_helmet_iou": float(np.mean(helmet_final)) if helmet_final else 0.0,
        "enhancement_attempts": enhanced_attempts,
        "enhancement_recoveries": enhanced_recover,
        "enhancement_recovery_rate": float(enhanced_recover / max(enhanced_attempts, 1)),
        "rollback_attempts": rollback_attempts,
        "rollback_improved": rollback_improved,
        "rollback_improvement_rate": float(rollback_improved / max(rollback_attempts, 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--overlay-limit", type=int, default=30)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--vision-pretrained",
        default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth",
    )
    parser.add_argument("--degradation", default="mixed", choices=["medium", "hard", "extreme", "mixed"])
    parser.add_argument("--miner-accept-iou", type=float, default=0.55)
    parser.add_argument("--miner-min-continue-iou", type=float, default=0.30)
    parser.add_argument("--helmet-accept-iou", type=float, default=0.55)
    parser.add_argument("--helmet-head-min-score", type=float, default=0.3)
    parser.add_argument("--low-brightness", type=float, default=0.28)
    parser.add_argument("--focus-dilate", type=int, default=21)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "visualizations"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    episodes = read_jsonl(args.jsonl)
    if args.max_items > 0:
        episodes = episodes[: args.max_items]

    runner = ControllerRunner(args)
    rows = []
    for idx, ep in enumerate(tqdm(episodes, desc="stage3_controller")):
        row = run_episode(runner, ep, args, idx, overlay_dir if idx < args.overlay_limit else out_dir / "tmp_no_overlay")
        if idx >= args.overlay_limit:
            row["visualization"] = None
        rows.append(row)

    summary = summarize(rows)
    summary.update(
        {
            "model": args.model,
            "jsonl": args.jsonl,
            "degradation": args.degradation,
            "miner_accept_iou": args.miner_accept_iou,
            "helmet_accept_iou": args.helmet_accept_iou,
        }
    )
    write_jsonl(out_dir / "stage3_operation_traces_v1.jsonl", rows)
    (out_dir / "stage3_operation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

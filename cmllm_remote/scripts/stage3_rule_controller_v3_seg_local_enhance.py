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


def degrade_parametric(image_rgb, cfg, seed):
    rng = np.random.default_rng(seed)
    gamma = float(cfg.get("gamma", 1.0))
    scale = float(cfg.get("scale", 1.0))
    contrast = float(cfg.get("contrast", 1.0))
    sigma = float(cfg.get("noise_sigma", 0.0))
    blur = float(cfg.get("blur", 0.0))

    x = image_rgb.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0.0, 1.0), gamma)
    x = (x - 0.5) * contrast + 0.5
    x = x * scale
    if sigma > 0:
        noise = rng.normal(0.0, sigma / 255.0, x.shape).astype(np.float32)
        x = x + noise
    x = np.clip(x, 0.0, 1.0)
    out = (x * 255.0).astype(np.uint8)
    if blur > 0:
        k = max(1, int(round(blur * 2 + 1)))
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), blur)
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


def load_task_enhancer_model(path):
    if not path:
        return None
    from train_task_enhancer_v1 import TaskEnhancerUNet  # noqa: WPS433

    ckpt = torch.load(path, map_location="cpu")
    base = int(ckpt.get("args", {}).get("base_channels", 32))
    model = TaskEnhancerUNet(base)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.cuda().eval()
    return model


@torch.no_grad()
def run_task_enhancer(model, image_rgb, max_side=1024):
    if model is None:
        return None
    h, w = image_rgb.shape[:2]
    work = image_rgb
    scale = 1.0
    if max_side > 0 and max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        work = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(work.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        y = model(x).clamp(0, 1)
    out = (y[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if scale != 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def apply_global_enhance(image_rgb, strength, args):
    model = getattr(args, "global_enhancer_model", None)
    if model is not None:
        return run_task_enhancer(model, image_rgb, getattr(args, "global_enhancer_max_side", 1024))
    return enhance_lowlight(image_rgb, strength)


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


def expand_bbox_xyxy(bbox, image_hw, ratio=0.45, min_side=96):
    if bbox is None:
        return None
    h, w = image_hw
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side_w = max(bw * (1.0 + 2.0 * ratio), float(min_side))
    side_h = max(bh * (1.0 + 2.0 * ratio), float(min_side))
    return clamp_bbox(
        [cx - 0.5 * side_w, cy - 0.5 * side_h, cx + 0.5 * side_w, cy + 0.5 * side_h],
        (h, w),
    )


def is_local_enhance_candidate(mask, args):
    stats = mask_stats(mask)
    if stats["bbox_xyxy"] is None:
        return False, stats, "empty_mask"
    if stats["area_ratio"] < args.local_min_area_ratio:
        return False, stats, "too_small"
    if stats["area_ratio"] > args.local_max_area_ratio:
        return False, stats, "too_large"
    return True, stats, "ok"


def parse_roi_scales(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def scale_to_expand_ratio(scale):
    return max(0.0, (float(scale) - 1.0) * 0.5)


def local_enhance_by_seg(image_rgb, seg_mask, args, strength="normal", roi_scale=None):
    bbox = bbox_from_mask(seg_mask)
    expand_ratio = args.local_roi_expand
    if roi_scale is not None:
        expand_ratio = scale_to_expand_ratio(roi_scale)
    roi = expand_bbox_xyxy(
        bbox,
        image_rgb.shape[:2],
        ratio=expand_ratio,
        min_side=args.local_min_side,
    )
    if roi is None:
        return image_rgb.copy(), None

    x1, y1, x2, y2 = roi
    crop = image_rgb[y1:y2, x1:x2]
    enhanced_crop = enhance_lowlight(crop, strength=strength)
    blur = cv2.GaussianBlur(enhanced_crop, (0, 0), 1.0)
    enhanced_crop = cv2.addWeighted(enhanced_crop, 1.25, blur, -0.25, 0)
    enhanced_full = image_rgb.copy()
    enhanced_full[y1:y2, x1:x2] = enhanced_crop

    alpha = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    alpha[y1:y2, x1:x2] = float(args.local_blend)
    soft = cv2.GaussianBlur(alpha, (0, 0), max(1.0, float(args.local_feather)))
    soft = np.clip(soft[..., None], 0.0, 1.0)
    out = image_rgb.astype(np.float32) * (1.0 - soft) + enhanced_full.astype(np.float32) * soft
    return np.clip(out, 0, 255).astype(np.uint8), roi


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
    if panels:
        blank = np.full_like(panels[0], 240, dtype=np.uint8)
        while len(panels) % 3:
            panels.append(blank.copy())
    rows = []
    for i in range(0, len(panels), 3):
        row = np.concatenate(panels[i : i + 3], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def maybe_save_flow_visual(args, overlay_dir, idx, ep, panels):
    if not getattr(args, "save_visuals", True):
        return None
    out_vis = overlay_dir / f"{idx:05d}_{ep.get('episode_id','episode')}.jpg"
    save_flow_visual(out_vis, panels)
    return str(out_vis)


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


def direct_query_for_round(round2):
    category = round2.get("target_category", "mining_helmet").replace("_", " ")
    return f"Please segment the {category} in the image."


def is_target_ok(quality, args, anchor_mask=None):
    if quality["iou"] < args.helmet_accept_iou:
        return False
    if anchor_mask is None:
        return True
    return quality.get("helmet_head_score", 1.0) >= args.helmet_head_min_score


def target_overlay(image_rgb, gt_mask, pred_mask):
    panel = overlay_mask(image_rgb, gt_mask, (0, 180, 0), 0.35)
    return overlay_mask(panel, pred_mask, (255, 40, 40), 0.55)


def run_episode(runner, ep, args, idx, overlay_dir):
    image_rgb, miner_gt, helmet_gt, round1, round2 = load_episode_assets(ep)
    seed = stable_seed(ep.get("episode_id", ep["image_path"]))
    if args.degradation_config:
        severity = args.degradation_config.get("id", "parametric")
        degraded = degrade_parametric(image_rgb, args.degradation_config, seed)
    else:
        severity = args.degradation
        if severity == "mixed":
            severity = "hard" if seed % 2 else "medium"
        degraded = degrade_image(image_rgb, severity, seed)
    enhanced = apply_global_enhance(degraded, "normal", args)
    enhanced_strong = apply_global_enhance(degraded, "strong", args)

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
    direct_query = direct_query_for_round(round2)

    degraded_quality = image_quality(degraded)

    pred_target_degraded = runner.predict(degraded, direct_query, helmet_gt)
    target_degraded_quality = score_seg(pred_target_degraded, helmet_gt)
    target_ok = is_target_ok(target_degraded_quality, args)
    steps.append(
        {
            "step": 1,
            "image_state": "degraded",
            "action": "segment_target_direct",
            "target": round2.get("target_category", "mining_helmet"),
            "quality": target_degraded_quality,
            "image_quality": degraded_quality,
            "decision": "accept" if target_ok else "failed",
        }
    )
    panels.append(
        tile(
            target_overlay(degraded, helmet_gt, pred_target_degraded),
            f"direct target on degraded IoU={target_degraded_quality['iou']:.3f}",
        )
    )

    best_target = pred_target_degraded
    best_target_quality = target_degraded_quality
    best_target_image = degraded
    best_target_state = "degraded"
    direct_final_path = "direct_target"
    enhanced_attempted = False
    if not target_ok and args.enable_global_enhance:
        low_light = degraded_quality["brightness"] < args.low_brightness
        action = "enhance_lowlight" if low_light else "enhance_denoise"
        steps.append(
            {
                "step": 2,
                "image_state": "degraded",
                "action": action,
                "reason": "direct target failed and image is low-light/noisy",
                "decision": "execute",
            }
        )
        enhanced_attempted = True
        panels.append(tile(enhanced, "operation: enhance low-light/denoise"))

        pred_target_enhanced = runner.predict(enhanced, direct_query, helmet_gt)
        target_enhanced_quality = score_seg(pred_target_enhanced, helmet_gt)
        target_ok = is_target_ok(target_enhanced_quality, args)
        steps.append(
            {
                "step": 3,
                "image_state": "enhanced",
                "action": "segment_target_direct",
                "target": round2.get("target_category", "mining_helmet"),
                "quality": target_enhanced_quality,
                "image_quality": image_quality(enhanced),
                "improvement_over_degraded": float(
                    target_enhanced_quality["iou"] - target_degraded_quality["iou"]
                ),
                "decision": "accept" if target_ok else "failed",
            }
        )
        panels.append(
            tile(
                target_overlay(enhanced, helmet_gt, pred_target_enhanced),
                f"direct target after enhance IoU={target_enhanced_quality['iou']:.3f}",
            )
        )
        if target_enhanced_quality["iou"] > best_target_quality["iou"]:
            best_target = pred_target_enhanced
            best_target_quality = target_enhanced_quality
            best_target_image = enhanced
            best_target_state = "enhanced"

        if not target_ok:
            steps.append(
                {
                    "step": len(steps) + 1,
                    "image_state": "enhanced",
                    "action": "enhance_lowlight_strong",
                    "reason": "normal enhancement did not recover target segmentation",
                    "decision": "execute",
                }
            )
            panels.append(tile(enhanced_strong, "operation: stronger enhancement"))

            pred_target_strong = runner.predict(enhanced_strong, direct_query, helmet_gt)
            target_strong_quality = score_seg(pred_target_strong, helmet_gt)
            target_ok = is_target_ok(target_strong_quality, args)
            steps.append(
                {
                    "step": len(steps) + 1,
                    "image_state": "enhanced_strong",
                    "action": "segment_target_direct",
                    "target": round2.get("target_category", "mining_helmet"),
                    "quality": target_strong_quality,
                    "image_quality": image_quality(enhanced_strong),
                    "improvement_over_degraded": float(
                        target_strong_quality["iou"] - target_degraded_quality["iou"]
                    ),
                    "improvement_over_enhanced": float(
                        target_strong_quality["iou"] - target_enhanced_quality["iou"]
                    ),
                    "decision": "accept" if target_ok else "failed",
                }
            )
            panels.append(
                tile(
                    target_overlay(enhanced_strong, helmet_gt, pred_target_strong),
                    f"direct target strong enhance IoU={target_strong_quality['iou']:.3f}",
                )
            )
            if target_strong_quality["iou"] > best_target_quality["iou"]:
                best_target = pred_target_strong
                best_target_quality = target_strong_quality
                best_target_image = enhanced_strong
                best_target_state = "enhanced_strong"

    if not target_ok and args.enable_local_enhance:
        ok_local, local_stats, local_reason = is_local_enhance_candidate(best_target, args)
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": best_target_state,
                "action": "local_enhance_target_by_seg",
                "condition": "best_direct_target_seg",
                "candidate_stats": local_stats,
                "reason": local_reason,
                "decision": "execute" if ok_local else "skip",
            }
        )
        if ok_local:
            local_base_image = best_target_image
            local_base_mask = best_target
            local_base_quality = best_target_quality
            local_base_state = best_target_state
            local_kept_any = False
            for roi_scale in parse_roi_scales(args.local_roi_scales_target):
                target_local_image, target_local_roi = local_enhance_by_seg(
                    local_base_image,
                    local_base_mask,
                    args,
                    strength=args.local_enhance_strength,
                    roi_scale=roi_scale,
                )
                panels.append(
                    tile(
                        target_local_image,
                        f"operation: local target [SEG] ROI {roi_scale:.1f}x",
                    )
                )
                pred_target_local = runner.predict(target_local_image, direct_query, helmet_gt)
                target_local_quality = score_seg(pred_target_local, helmet_gt)
                local_target_ok = is_target_ok(target_local_quality, args)
                improvement_over_base = float(
                    target_local_quality["iou"] - local_base_quality["iou"]
                )
                improvement_over_current_best = float(
                    target_local_quality["iou"] - best_target_quality["iou"]
                )
                if local_target_ok:
                    decision = "accept"
                elif improvement_over_current_best > 0:
                    decision = "keep_best"
                else:
                    decision = "rollback"
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "image_state": f"{local_base_state}_local_target_{roi_scale:.1f}x",
                        "action": "segment_target_direct_local_enhance",
                        "target": round2.get("target_category", "mining_helmet"),
                        "roi_scale": roi_scale,
                        "roi_xyxy": target_local_roi,
                        "quality": target_local_quality,
                        "improvement_over_best_direct": improvement_over_base,
                        "improvement_over_current_best": improvement_over_current_best,
                        "decision": decision,
                    }
                )
                panels.append(
                    tile(
                        target_overlay(target_local_image, helmet_gt, pred_target_local),
                        f"direct target local {roi_scale:.1f}x IoU={target_local_quality['iou']:.3f}",
                    )
                )
                if target_local_quality["iou"] > best_target_quality["iou"]:
                    best_target = pred_target_local
                    best_target_quality = target_local_quality
                    best_target_image = target_local_image
                    best_target_state = f"{local_base_state}_local_target_{roi_scale:.1f}x"
                    local_kept_any = True
                if local_target_ok:
                    target_ok = True
                    direct_final_path = "direct_target_local_enhance"
                    break
            if not target_ok and not local_kept_any:
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "image_state": local_base_state,
                        "action": "rollback_local_target_enhance",
                        "reason": "all target local ROI scales were worse than the pre-local state",
                        "decision": "restore_pre_local_state",
                    }
                )

    if target_ok:
        while len(panels) < 9:
            panels.append(
                tile(
                    target_overlay(best_target_image, helmet_gt, best_target),
                    f"final direct state={best_target_state}",
                )
            )
        out_vis = maybe_save_flow_visual(args, overlay_dir, idx, ep, panels)
        return {
            "episode_id": ep.get("episode_id"),
            "image_path": ep["image_path"],
            "severity": severity,
            "steps": steps,
            "final_decision": "accepted",
            "final_path": direct_final_path,
            "final_anchor_iou": 0.0,
            "final_miner_iou": 0.0,
            "final_helmet_iou": float(best_target_quality["iou"]),
            "final_target_iou": float(best_target_quality["iou"]),
            "visualization": out_vis,
        }

    steps.append(
        {
            "step": len(steps) + 1,
            "image_state": best_target_state,
            "action": "find_anchor",
            "anchor_category": round1.get("target_category", "coal_miner"),
            "reason": "direct target path failed; use anchor as a tool",
            "decision": "execute",
        }
    )
    anchor_image = best_target_image
    anchor_state = best_target_state
    pred_anchor = runner.predict(anchor_image, round1_query, miner_gt)
    anchor_quality = score_seg(pred_anchor, miner_gt)
    anchor_ok = anchor_quality["iou"] >= args.miner_accept_iou
    steps.append(
        {
            "step": len(steps) + 1,
            "image_state": anchor_state,
            "action": "segment_anchor",
            "target": round1.get("target_category", "coal_miner"),
            "quality": anchor_quality,
            "decision": "accept" if anchor_ok else "failed",
        }
    )
    anchor_panel = overlay_mask(anchor_image, miner_gt, (0, 180, 0), 0.35)
    anchor_panel = overlay_mask(anchor_panel, pred_anchor, (0, 120, 255), 0.55)
    panels.append(tile(anchor_panel, f"anchor after direct fail IoU={anchor_quality['iou']:.3f}"))

    if args.enable_local_enhance and not anchor_ok and anchor_quality["iou"] >= args.miner_min_continue_iou:
        ok_local_anchor, local_anchor_stats, local_anchor_reason = is_local_enhance_candidate(
            pred_anchor, args
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": anchor_state,
                "action": "local_enhance_anchor_by_seg",
                "condition": "rough_anchor_seg",
                "candidate_stats": local_anchor_stats,
                "reason": local_anchor_reason,
                "decision": "execute" if ok_local_anchor else "skip",
            }
        )
        if ok_local_anchor:
            anchor_base_image = anchor_image
            anchor_base_mask = pred_anchor
            anchor_base_quality = anchor_quality
            anchor_base_state = anchor_state
            anchor_kept_any = False
            for roi_scale in parse_roi_scales(args.local_roi_scales_anchor):
                anchor_local_image, anchor_local_roi = local_enhance_by_seg(
                    anchor_base_image,
                    anchor_base_mask,
                    args,
                    strength=args.local_enhance_strength,
                    roi_scale=roi_scale,
                )
                panels.append(
                    tile(
                        anchor_local_image,
                        f"operation: local anchor [SEG] ROI {roi_scale:.1f}x",
                    )
                )
                pred_anchor_local = runner.predict(anchor_local_image, round1_query, miner_gt)
                anchor_local_quality = score_seg(pred_anchor_local, miner_gt)
                local_anchor_ok = anchor_local_quality["iou"] >= args.miner_accept_iou
                improvement_over_base = float(anchor_local_quality["iou"] - anchor_base_quality["iou"])
                improvement_over_current_best = float(anchor_local_quality["iou"] - anchor_quality["iou"])
                if local_anchor_ok:
                    decision = "accept"
                elif improvement_over_current_best > 0:
                    decision = "keep_best"
                else:
                    decision = "rollback"
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "image_state": f"{anchor_base_state}_local_anchor_{roi_scale:.1f}x",
                        "action": "segment_anchor_local_enhance",
                        "target": round1.get("target_category", "coal_miner"),
                        "roi_scale": roi_scale,
                        "roi_xyxy": anchor_local_roi,
                        "quality": anchor_local_quality,
                        "improvement_over_anchor": improvement_over_base,
                        "improvement_over_current_best": improvement_over_current_best,
                        "decision": decision,
                    }
                )
                anchor_local_panel = overlay_mask(anchor_local_image, miner_gt, (0, 180, 0), 0.35)
                anchor_local_panel = overlay_mask(
                    anchor_local_panel, pred_anchor_local, (0, 120, 255), 0.55
                )
                panels.append(
                    tile(
                        anchor_local_panel,
                        f"anchor local {roi_scale:.1f}x IoU={anchor_local_quality['iou']:.3f}",
                    )
                )
                if anchor_local_quality["iou"] > anchor_quality["iou"]:
                    anchor_image = anchor_local_image
                    anchor_state = f"{anchor_base_state}_local_anchor_{roi_scale:.1f}x"
                    pred_anchor = pred_anchor_local
                    anchor_quality = anchor_local_quality
                    anchor_ok = local_anchor_ok
                    anchor_kept_any = True
                if local_anchor_ok:
                    break
            if not anchor_ok and not anchor_kept_any:
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "image_state": anchor_base_state,
                        "action": "rollback_local_anchor_enhance",
                        "reason": "all anchor local ROI scales were worse than the pre-local state",
                        "decision": "restore_pre_local_state",
                    }
                )

    if not anchor_ok and anchor_quality["iou"] < args.miner_min_continue_iou:
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": anchor_state,
                "action": "stop_failed_anchor",
                "reason": "direct target failed and no reliable anchor was found",
                "decision": "stop",
            }
        )
        while len(panels) < 9:
            panels.append(
                tile(
                    target_overlay(best_target_image, helmet_gt, best_target),
                    "stopped: direct failed and anchor unreliable",
                )
            )
        out_vis = maybe_save_flow_visual(args, overlay_dir, idx, ep, panels)
        return {
            "episode_id": ep.get("episode_id"),
            "image_path": ep["image_path"],
            "severity": severity,
            "steps": steps,
            "final_decision": "failed_anchor",
            "final_path": "direct_then_anchor_failed",
            "final_anchor_iou": float(anchor_quality["iou"]),
            "final_miner_iou": float(anchor_quality["iou"]),
            "final_helmet_iou": float(best_target_quality["iou"]),
            "final_target_iou": float(best_target_quality["iou"]),
            "visualization": out_vis,
        }

    focus_image = focus_blackout(anchor_image, pred_anchor, dilate=args.focus_dilate, background=0.0)
    steps.append(
        {
            "step": len(steps) + 1,
            "image_state": anchor_state,
            "action": "focus_blackout",
            "ref": "accepted_anchor",
            "decision": "execute",
        }
    )
    panels.append(tile(focus_image, "operation: focus/blackout around anchor"))

    pred_helmet_focus = runner.predict(
        focus_image,
        round2_query,
        helmet_gt,
        ref_mask=pred_anchor,
        ref_bbox=bbox_from_mask(pred_anchor),
    )
    helmet_focus_quality = score_seg(pred_helmet_focus, helmet_gt)
    helmet_focus_quality.update(helmet_geometry(pred_helmet_focus, pred_anchor))
    focus_ok = is_target_ok(helmet_focus_quality, args, anchor_mask=pred_anchor)
    steps.append(
        {
            "step": len(steps) + 1,
            "image_state": "focus_blackout",
            "action": "segment_helmet",
            "target": "mining_helmet",
            "quality": helmet_focus_quality,
            "decision": "accept" if focus_ok else "failed",
        }
    )
    panels.append(
        tile(
            target_overlay(focus_image, helmet_gt, pred_helmet_focus),
            f"target with anchor/focus IoU={helmet_focus_quality['iou']:.3f}",
        )
    )

    focus_result_image = focus_image
    focus_state = "focus_blackout"
    focus_path = "anchor_focus"
    if args.enable_local_enhance and not focus_ok:
        ok_local_focus, local_focus_stats, local_focus_reason = is_local_enhance_candidate(
            pred_helmet_focus, args
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": focus_state,
                "action": "local_enhance_target_by_seg_after_ref",
                "condition": "rough_ref_target_seg",
                "candidate_stats": local_focus_stats,
                "reason": local_focus_reason,
                "decision": "execute" if ok_local_focus else "skip",
            }
        )
        if ok_local_focus:
            focus_base_image = focus_image
            focus_base_mask = pred_helmet_focus
            focus_base_quality = helmet_focus_quality
            focus_kept_any = False
            for roi_scale in parse_roi_scales(args.local_roi_scales_ref_target):
                focus_local_image, focus_local_roi = local_enhance_by_seg(
                    focus_base_image,
                    focus_base_mask,
                    args,
                    strength=args.local_enhance_strength,
                    roi_scale=roi_scale,
                )
                panels.append(
                    tile(
                        focus_local_image,
                        f"operation: local ref target [SEG] ROI {roi_scale:.1f}x",
                    )
                )
                pred_helmet_local = runner.predict(
                    focus_local_image,
                    round2_query,
                    helmet_gt,
                    ref_mask=pred_anchor,
                    ref_bbox=bbox_from_mask(pred_anchor),
                )
                helmet_local_quality = score_seg(pred_helmet_local, helmet_gt)
                helmet_local_quality.update(helmet_geometry(pred_helmet_local, pred_anchor))
                local_focus_ok = is_target_ok(helmet_local_quality, args, anchor_mask=pred_anchor)
                improvement_over_base = float(
                    helmet_local_quality["iou"] - focus_base_quality["iou"]
                )
                improvement_over_current_best = float(
                    helmet_local_quality["iou"] - helmet_focus_quality["iou"]
                )
                if local_focus_ok:
                    decision = "accept"
                elif improvement_over_current_best > 0:
                    decision = "keep_best"
                else:
                    decision = "rollback"
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "image_state": f"focus_blackout_local_target_{roi_scale:.1f}x",
                        "action": "segment_helmet_local_enhance",
                        "target": "mining_helmet",
                        "roi_scale": roi_scale,
                        "roi_xyxy": focus_local_roi,
                        "quality": helmet_local_quality,
                        "improvement_over_focus": improvement_over_base,
                        "improvement_over_current_best": improvement_over_current_best,
                        "decision": decision,
                    }
                )
                panels.append(
                    tile(
                        target_overlay(focus_local_image, helmet_gt, pred_helmet_local),
                        f"target local {roi_scale:.1f}x IoU={helmet_local_quality['iou']:.3f}",
                    )
                )
                if helmet_local_quality["iou"] > helmet_focus_quality["iou"]:
                    pred_helmet_focus = pred_helmet_local
                    helmet_focus_quality = helmet_local_quality
                    focus_ok = local_focus_ok
                    focus_result_image = focus_local_image
                    focus_state = f"focus_blackout_local_target_{roi_scale:.1f}x"
                    focus_path = "anchor_focus_local_enhance"
                    focus_kept_any = True
                if local_focus_ok:
                    break
            if not focus_ok and not focus_kept_any:
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "image_state": "focus_blackout",
                        "action": "rollback_local_ref_target_enhance",
                        "reason": "all ref-target local ROI scales were worse than the pre-local state",
                        "decision": "restore_pre_local_state",
                    }
                )

    final_helmet = pred_helmet_focus
    final_helmet_quality = helmet_focus_quality
    final_image = focus_result_image
    final_state = focus_state
    final_path = focus_path
    rolled_back = False
    if helmet_focus_quality["iou"] < best_target_quality["iou"]:
        rolled_back = True
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": "focus_blackout",
                "action": "rollback_to_accepted_ref",
                "rollback_target": best_target_state,
                "failed_ref": "failed_target_focus",
                "reason": "anchor/focus result is worse than direct target attempt",
                "decision": "execute",
            }
        )
        panels.append(tile(best_target_image, "rollback: restore best direct state"))
        steps.append(
            {
                "step": len(steps) + 1,
                "image_state": "rollback_accepted_state",
                "action": "restore_direct_target_result",
                "target": "mining_helmet",
                "quality": best_target_quality,
                "improvement_over_failed_focus": float(
                    best_target_quality["iou"] - helmet_focus_quality["iou"]
                ),
                "decision": "accept" if is_target_ok(best_target_quality, args) else "failed",
            }
        )
        panels.append(
            tile(
                target_overlay(best_target_image, helmet_gt, best_target),
                f"restored direct target IoU={best_target_quality['iou']:.3f}",
            )
        )
        final_helmet = best_target
        final_helmet_quality = best_target_quality
        final_image = best_target_image
        final_state = "rollback_accepted_state"
        final_path = "rollback_to_direct"

    while len(panels) < 9:
        final_panel = target_overlay(final_image, helmet_gt, final_helmet)
        panels.append(tile(final_panel, f"final state={final_state}"))

    out_vis = maybe_save_flow_visual(args, overlay_dir, idx, ep, panels)
    final_decision = "accepted" if is_target_ok(final_helmet_quality, args, pred_anchor if final_path == "anchor_focus" else None) else "failed_target"
    return {
        "episode_id": ep.get("episode_id"),
        "image_path": ep["image_path"],
        "severity": severity,
        "steps": steps,
        "accepted_anchor_state": anchor_state,
        "final_state": final_state,
        "final_decision": final_decision,
        "final_path": final_path,
        "rolled_back": rolled_back,
        "best_direct_iou": float(best_target_quality["iou"]),
        "focus_iou": float(helmet_focus_quality["iou"]),
        "final_anchor_iou": float(anchor_quality["iou"]),
        "final_miner_iou": float(anchor_quality["iou"]),
        "final_helmet_iou": float(final_helmet_quality["iou"]),
        "final_target_iou": float(final_helmet_quality["iou"]),
        "visualization": out_vis,
    }


def summarize(rows):
    n = len(rows)
    target_initial = []
    target_final = []
    anchor_final = []
    enhanced_recover = 0
    enhanced_attempts = 0
    rollback_attempts = 0
    rollback_improved = 0
    accepted = 0
    failed_anchor = 0
    failed_target = 0
    direct_accepts = 0
    direct_attempts = 0
    anchor_attempts = 0
    focus_attempts = 0
    rollback_final = 0
    local_enhance_attempts = 0
    local_target_attempts = 0
    local_anchor_attempts = 0
    local_ref_target_attempts = 0
    local_seg_attempts = 0
    local_seg_accepts = 0
    local_seg_improved = 0
    local_seg_rollbacks = 0
    final_path_counts = {}
    for row in rows:
        final_path = row.get("final_path", "unknown")
        final_path_counts[final_path] = final_path_counts.get(final_path, 0) + 1
        anchor_final.append(row.get("final_anchor_iou", row.get("final_miner_iou", 0.0)))
        target_final.append(row.get("final_target_iou", row.get("final_helmet_iou", 0.0)))
        if row.get("final_decision") == "accepted":
            accepted += 1
        elif row.get("final_decision") == "failed_anchor":
            failed_anchor += 1
        elif row.get("final_decision") == "failed_target":
            failed_target += 1
        if str(row.get("final_path", "")).startswith("direct_target"):
            direct_accepts += 1
        if row.get("final_path") == "rollback_to_direct":
            rollback_final += 1
        first_target = None
        for step in row["steps"]:
            if step["action"] == "segment_target_direct":
                direct_attempts += 1
                if first_target is None:
                    first_target = step["quality"]["iou"]
            if step["action"] in {"enhance_lowlight", "enhance_denoise", "enhance_lowlight_strong"}:
                enhanced_attempts += 1
            if step["action"] == "segment_target_direct" and str(step.get("image_state", "")).startswith("enhanced"):
                if step["decision"] == "accept":
                    enhanced_recover += 1
            if step["action"] == "find_anchor":
                anchor_attempts += 1
            if step["action"] == "focus_blackout":
                focus_attempts += 1
            if step["action"].startswith("local_enhance") and step.get("decision") == "execute":
                local_enhance_attempts += 1
                if step["action"] == "local_enhance_target_by_seg":
                    local_target_attempts += 1
                elif step["action"] == "local_enhance_anchor_by_seg":
                    local_anchor_attempts += 1
                elif step["action"] == "local_enhance_target_by_seg_after_ref":
                    local_ref_target_attempts += 1
            if "local_enhance" in step["action"] and step["action"].startswith("segment_"):
                local_seg_attempts += 1
                if step.get("decision") == "accept":
                    local_seg_accepts += 1
                if step.get("decision") == "rollback":
                    local_seg_rollbacks += 1
                if (
                    step.get("improvement_over_best_direct", 0.0) > 0
                    or step.get("improvement_over_anchor", 0.0) > 0
                    or step.get("improvement_over_focus", 0.0) > 0
                ):
                    local_seg_improved += 1
            if step["action"] == "rollback_to_accepted_ref":
                rollback_attempts += 1
            if step["action"] == "restore_direct_target_result":
                if step.get("improvement_over_failed_focus", 0.0) > 0:
                    rollback_improved += 1
        if first_target is not None:
            target_initial.append(first_target)
    return {
        "num_episodes": n,
        "accepted": accepted,
        "failed_anchor": failed_anchor,
        "failed_target": failed_target,
        "accept_rate": float(accepted / max(n, 1)),
        "mean_initial_target_iou": float(np.mean(target_initial)) if target_initial else 0.0,
        "mean_final_target_iou": float(np.mean(target_final)) if target_final else 0.0,
        "mean_final_anchor_iou": float(np.mean(anchor_final)) if anchor_final else 0.0,
        "direct_target_attempts": direct_attempts,
        "direct_target_accepts": direct_accepts,
        "direct_target_accept_rate": float(direct_accepts / max(n, 1)),
        "enhancement_attempts": enhanced_attempts,
        "enhancement_recoveries": enhanced_recover,
        "enhancement_recovery_rate": float(enhanced_recover / max(enhanced_attempts, 1)),
        "anchor_attempts": anchor_attempts,
        "focus_attempts": focus_attempts,
        "local_enhance_attempts": local_enhance_attempts,
        "local_target_attempts": local_target_attempts,
        "local_anchor_attempts": local_anchor_attempts,
        "local_ref_target_attempts": local_ref_target_attempts,
        "local_seg_attempts": local_seg_attempts,
        "local_seg_accepts": local_seg_accepts,
        "local_seg_improved": local_seg_improved,
        "local_seg_rollbacks": local_seg_rollbacks,
        "local_seg_improvement_rate": float(local_seg_improved / max(local_seg_attempts, 1)),
        "rollback_attempts": rollback_attempts,
        "rollback_improved": rollback_improved,
        "rollback_improvement_rate": float(rollback_improved / max(rollback_attempts, 1)),
        "rollback_final_count": rollback_final,
        "final_path_counts": final_path_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--overlay-limit", type=int, default=30)
    parser.add_argument("--save-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--vision-pretrained",
        default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth",
    )
    parser.add_argument("--degradation", default="mixed", choices=["medium", "hard", "extreme", "mixed"])
    parser.add_argument("--degradation-config-json", default="")
    parser.add_argument("--miner-accept-iou", type=float, default=0.55)
    parser.add_argument("--miner-min-continue-iou", type=float, default=0.30)
    parser.add_argument("--helmet-accept-iou", type=float, default=0.55)
    parser.add_argument("--helmet-head-min-score", type=float, default=0.3)
    parser.add_argument("--low-brightness", type=float, default=0.28)
    parser.add_argument("--focus-dilate", type=int, default=21)
    parser.add_argument("--enable-global-enhance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global-enhancer", default="")
    parser.add_argument("--global-enhancer-max-side", type=int, default=1024)
    parser.add_argument("--enable-local-enhance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-enhance-strength", default="normal", choices=["normal", "strong"])
    parser.add_argument("--local-roi-expand", type=float, default=0.10)
    parser.add_argument("--local-roi-scales-target", default="1.0,1.2,1.5")
    parser.add_argument("--local-roi-scales-anchor", default="1.2,1.5,1.9")
    parser.add_argument("--local-roi-scales-ref-target", default="1.0,1.2,1.5,1.9")
    parser.add_argument("--local-min-side", type=int, default=96)
    parser.add_argument("--local-blend", type=float, default=0.95)
    parser.add_argument("--local-feather", type=float, default=9.0)
    parser.add_argument("--local-min-area-ratio", type=float, default=0.00015)
    parser.add_argument("--local-max-area-ratio", type=float, default=0.40)
    args = parser.parse_args()
    args.degradation_config = (
        json.loads(args.degradation_config_json) if args.degradation_config_json else None
    )

    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "visualizations"
    if args.save_visuals:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    episodes = read_jsonl(args.jsonl)
    if args.max_items > 0:
        episodes = episodes[: args.max_items]

    args.global_enhancer_model = load_task_enhancer_model(args.global_enhancer)
    runner = ControllerRunner(args)
    rows = []
    for idx, ep in enumerate(tqdm(episodes, desc="stage3_controller_v3_seg_local_enhance")):
        target_overlay_dir = overlay_dir if idx < args.overlay_limit else out_dir / "tmp_no_overlay"
        row = run_episode(runner, ep, args, idx, target_overlay_dir)
        if (not args.save_visuals) or idx >= args.overlay_limit:
            row["visualization"] = None
        rows.append(row)

    summary = summarize(rows)
    summary.update(
        {
            "model": args.model,
            "jsonl": args.jsonl,
            "degradation": args.degradation,
            "degradation_config": args.degradation_config,
            "miner_accept_iou": args.miner_accept_iou,
            "helmet_accept_iou": args.helmet_accept_iou,
            "enable_global_enhance": args.enable_global_enhance,
            "global_enhancer": args.global_enhancer,
            "global_enhancer_max_side": args.global_enhancer_max_side,
            "enable_local_enhance": args.enable_local_enhance,
            "save_visuals": args.save_visuals,
            "local_enhance_strength": args.local_enhance_strength,
            "local_roi_expand": args.local_roi_expand,
            "local_roi_scales_target": parse_roi_scales(args.local_roi_scales_target),
            "local_roi_scales_anchor": parse_roi_scales(args.local_roi_scales_anchor),
            "local_roi_scales_ref_target": parse_roi_scales(args.local_roi_scales_ref_target),
            "local_min_side": args.local_min_side,
            "local_blend": args.local_blend,
        }
    )
    write_jsonl(out_dir / "stage3_operation_traces_v3.jsonl", rows)
    (out_dir / "stage3_operation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

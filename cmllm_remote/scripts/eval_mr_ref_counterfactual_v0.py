import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

LISA_ROOT = "/home/wjq/cmllm/third_party/LISA"
if LISA_ROOT not in sys.path:
    sys.path.insert(0, LISA_ROOT)

from model.LISA import LISAForCausalLM  # noqa: E402
from model.llava import conversation as conversation_lib  # noqa: E402
from model.segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402
from utils.dataset import collate_fn  # noqa: E402


DEFAULT_IMAGE_TOKEN = "<image>"


def get_added_token_id(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0 or token_id == tokenizer.unk_token_id:
        token_id = tokenizer(token, add_special_tokens=False).input_ids[-1]
    return token_id


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def norm_bbox(bbox_xyxy, image_hw):
    h, w = image_hw
    x1, y1, x2, y2 = [float(x) for x in bbox_xyxy]
    return [
        max(0.0, min(1.0, x1 / max(w, 1))),
        max(0.0, min(1.0, y1 / max(h, 1))),
        max(0.0, min(1.0, x2 / max(w, 1))),
        max(0.0, min(1.0, y2 / max(h, 1))),
    ]


def make_ref_crop_clip(image_rgb, mask, bbox_xyxy, clip_processor):
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    masked = image_rgb.copy()
    masked[mask <= 0] = 0
    crop = masked[y1:y2, x1:x2]
    side = max(crop.shape[0], crop.shape[1], 1)
    padded = np.zeros((side, side, 3), dtype=crop.dtype)
    padded[: crop.shape[0], : crop.shape[1]] = crop
    return clip_processor.preprocess(padded, return_tensors="pt")["pixel_values"][0]


def make_ref_focus_clip(
    image_rgb,
    mask,
    clip_processor,
    ref_image_mode="crop",
    focus_dilate=15,
    focus_background=0.0,
):
    if ref_image_mode == "crop":
        raise ValueError("make_ref_focus_clip should not be called for crop mode")

    mask_bool = mask > 0
    if focus_dilate > 0 and mask_bool.any():
        kernel_size = max(1, int(focus_dilate))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_bool = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0

    if ref_image_mode == "full_blur":
        focused = cv2.GaussianBlur(image_rgb, (31, 31), 0)
        focused[mask_bool] = image_rgb[mask_bool]
    elif ref_image_mode in {"full_blackout", "full_darken"}:
        bg = np.clip(float(focus_background), 0.0, 1.0)
        focused = (image_rgb.astype(np.float32) * bg).astype(np.uint8)
        focused[mask_bool] = image_rgb[mask_bool]
    else:
        raise ValueError("Unsupported ref_image_mode: {}".format(ref_image_mode))

    return clip_processor.preprocess(focused, return_tensors="pt")["pixel_values"][0]


def make_ref_image_clip(
    image_rgb,
    mask,
    bbox_xyxy,
    clip_processor,
    ref_image_mode="crop",
    focus_dilate=15,
    focus_background=0.0,
):
    if ref_image_mode == "crop":
        return make_ref_crop_clip(image_rgb, mask, bbox_xyxy, clip_processor)
    return make_ref_focus_clip(
        image_rgb,
        mask,
        clip_processor,
        ref_image_mode=ref_image_mode,
        focus_dilate=focus_dilate,
        focus_background=focus_background,
    )


def build_conversation(query):
    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + "[REF] " + query.strip())
    conv.append_message(conv.roles[1], "[SEG].")
    return conv.get_prompt()


def build_multiround_conversation(round1_query, round2_query):
    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + round1_query.strip())
    conv.append_message(conv.roles[1], "[SEG].")
    conv.append_message(conv.roles[0], "[REF] " + round2_query.strip())
    conv.append_message(conv.roles[1], "[SEG].")
    return conv.get_prompt()


def iou(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / union)


def overlay(image_rgb, mask, color):
    out = image_rgb.copy()
    mask = mask.astype(bool)
    if mask.any():
        out[mask] = (out[mask] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
    return out


def tile(image_rgb, text, width=420, height=260):
    h, w = image_rgb.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    cv2.rectangle(canvas, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(canvas, text[:60], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def build_item(
    cf,
    pairs_by_id,
    clip_processor,
    transform,
    image_size,
    conversation_mode,
    ref_image_mode="crop",
    focus_dilate=15,
    focus_background=0.0,
):
    pair_ids = cf["pair_ids"]
    pairs = [pairs_by_id[pair_id] for pair_id in pair_ids]
    image_path = cf["image_path"]
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ori_size = image_rgb.shape[:2]

    target_masks = [read_mask(pair["helmet_mask_path"], ori_size) for pair in pairs]
    ref_masks = [read_mask(pair["miner_mask_path"], ori_size) for pair in pairs]
    ref_bboxes = [norm_bbox(pair["miner_bbox_xyxy"], ori_size) for pair in pairs]
    ref_images_clip = torch.stack(
        [
            make_ref_image_clip(
                image_rgb,
                ref_masks[i],
                pairs[i]["miner_bbox_xyxy"],
                clip_processor,
                ref_image_mode=ref_image_mode,
                focus_dilate=focus_dilate,
                focus_background=focus_background,
            )
            for i in range(len(pairs))
        ],
        dim=0,
    )

    if conversation_mode == "v1_multiround":
        conversations = [
            build_multiround_conversation(
                "Segment the miner whose helmet should be segmented next.",
                cf["same_round2_query"],
            )
            for _ in pair_ids
        ]
        masks_np = []
        ref_masks_np = []
        ref_bboxes_rows = []
        ref_valid_rows = []
        ref_images_rows = []
        questions = []
        sampled_classes = []
        eval_pred_indices = []
        for idx_pair, pair in enumerate(pairs):
            masks_np.extend([ref_masks[idx_pair], target_masks[idx_pair]])
            ref_masks_np.extend([np.zeros_like(ref_masks[idx_pair]), ref_masks[idx_pair]])
            ref_bboxes_rows.extend([[0.0, 0.0, 0.0, 0.0], ref_bboxes[idx_pair]])
            ref_valid_rows.extend([0.0, 1.0])
            ref_images_rows.extend([torch.zeros_like(ref_images_clip[idx_pair]), ref_images_clip[idx_pair]])
            questions.extend(["segment miner", "[REF] " + cf["same_round2_query"]])
            sampled_classes.extend(["coal_miner", "mining_helmet"])
            eval_pred_indices.append(2 * idx_pair + 1)
    else:
        conversations = [build_conversation(cf["same_round2_query"]) for _ in pair_ids]
        questions = ["[REF] " + cf["same_round2_query"] for _ in pair_ids]
        sampled_classes = ["mining_helmet" for _ in pair_ids]
        masks_np = target_masks
        ref_masks_np = ref_masks
        ref_bboxes_rows = ref_bboxes
        ref_valid_rows = [1.0] * len(pair_ids)
        ref_images_rows = [ref_images_clip[i] for i in range(len(pair_ids))]
        eval_pred_indices = list(range(len(pair_ids)))

    image_clip = clip_processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0]
    image_sam = transform.apply_image(image_rgb)
    resize = image_sam.shape[:2]
    image_sam = preprocess_sam(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous(), image_size=image_size)

    masks = torch.from_numpy(np.stack(masks_np, axis=0)).float()
    labels = torch.ones(masks.shape[1], masks.shape[2]) * 255
    ref_masks_t = torch.from_numpy(np.stack(ref_masks_np, axis=0)).float()
    ref_bboxes_t = torch.tensor(ref_bboxes_rows, dtype=torch.float32)
    ref_valids = torch.tensor(ref_valid_rows, dtype=torch.float32)
    mask_weights = torch.ones((len(masks_np),), dtype=torch.float32)
    ref_images_clip_t = torch.stack(ref_images_rows, dim=0)

    item = (
        image_path,
        image_sam,
        image_clip,
        conversations,
        masks,
        labels,
        resize,
        questions,
        sampled_classes,
        ref_masks_t,
        ref_bboxes_t,
        ref_valids,
        mask_weights,
        ref_images_clip_t,
        True,
    )
    return item, image_rgb, np.stack(target_masks, axis=0), np.stack(ref_masks, axis=0), eval_pred_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--counterfactual-jsonl", required=True)
    parser.add_argument("--pairs-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument("--vision-pretrained", default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth")
    parser.add_argument("--overlay-limit", type=int, default=64)
    parser.add_argument(
        "--conversation-mode",
        default="v0_single_ref",
        choices=["v0_single_ref", "v1_multiround"],
    )
    parser.add_argument(
        "--ref-image-mode",
        default=os.environ.get("MR_REF_IMAGE_MODE", "crop"),
        choices=["crop", "full_blackout", "full_darken", "full_blur"],
    )
    parser.add_argument(
        "--focus-dilate",
        type=int,
        default=int(os.environ.get("MR_REF_FOCUS_DILATE", "15")),
    )
    parser.add_argument(
        "--focus-background",
        type=float,
        default=float(os.environ.get("MR_REF_FOCUS_BACKGROUND", "0.0")),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(["[SEG]", "[REF]"])
    seg_token_idx = get_added_token_id(tokenizer, "[SEG]")
    ref_token_idx = get_added_token_id(tokenizer, "[REF]")

    dtype = torch.float32
    if args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16

    model = LISAForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        vision_tower=args.vision_tower,
        vision_pretrained=args.vision_pretrained,
        seg_token_idx=seg_token_idx,
        ref_token_idx=ref_token_idx,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=dtype)
    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        model = model.half().cuda()
    else:
        model = model.float().cuda()
    model.get_model().get_vision_tower().to(device=0)
    model.eval()

    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    pairs_by_id = {row["pair_id"]: row for row in read_jsonl(args.pairs_jsonl)}
    cfs = read_jsonl(args.counterfactual_jsonl)
    if args.max_items:
        cfs = cfs[: args.max_items]

    rows = []
    correct_ious = []
    wrong_ious = []
    rcs_values = []
    pair_success = 0
    pair_total = 0
    cf_success = 0

    for idx, cf in enumerate(cfs):
        item, image_rgb, target_masks, ref_masks, eval_pred_indices = build_item(
            cf,
            pairs_by_id,
            clip_processor,
            transform,
            args.image_size,
            args.conversation_mode,
            ref_image_mode=args.ref_image_mode,
            focus_dilate=args.focus_dilate,
            focus_background=args.focus_background,
        )
        batch = collate_fn([item], tokenizer=tokenizer, conv_type="llava_v1", use_mm_start_end=True, local_rank=0)
        for key in ["images", "images_clip", "input_ids", "labels", "attention_masks", "offset"]:
            batch[key] = batch[key].cuda(non_blocking=True)
        for key in ["masks_list", "label_list", "ref_masks_list", "ref_bboxes_list", "ref_valids_list", "mask_weights_list", "ref_images_clip_list"]:
            batch[key] = [x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch[key]]
        batch["images"] = batch["images"].to(dtype=dtype)
        batch["images_clip"] = batch["images_clip"].to(dtype=dtype)

        with torch.no_grad():
            outputs = model(**batch)
        pred_masks_all = outputs["pred_masks"][0].detach().float().cpu().numpy() > 0
        pred_masks = pred_masks_all[eval_pred_indices]

        n = min(pred_masks.shape[0], target_masks.shape[0])
        sample = {
            "counterfactual_id": cf["counterfactual_id"],
            "image_path": cf["image_path"],
            "pair_ids": cf["pair_ids"],
            "cf_score": cf.get("cf_score"),
            "items": [],
        }
        this_success = True
        for j in range(n):
            wrong_j = 1 - j if n == 2 else (j + 1) % n
            corr = iou(pred_masks[j], target_masks[j])
            wrong = iou(pred_masks[j], target_masks[wrong_j])
            rcs = corr - wrong
            ok = rcs > 0
            correct_ious.append(corr)
            wrong_ious.append(wrong)
            rcs_values.append(rcs)
            pair_success += int(ok)
            pair_total += 1
            this_success = this_success and ok
            sample["items"].append(
                {
                    "ref_pair_id": cf["pair_ids"][j],
                    "correct_pair_id": cf["pair_ids"][j],
                    "wrong_pair_id": cf["pair_ids"][wrong_j],
                    "iou_correct": corr,
                    "iou_wrong": wrong,
                    "rcs": rcs,
                    "success": ok,
                    "pred_area_ratio": float(pred_masks[j].mean()),
                }
            )
        cf_success += int(this_success)
        rows.append(sample)

        if idx < args.overlay_limit:
            tiles = [tile(image_rgb, cf["counterfactual_id"])]
            for j in range(n):
                wrong_j = 1 - j if n == 2 else (j + 1) % n
                pred_vis = overlay(image_rgb, pred_masks[j], (255, 0, 0))
                gt_vis = overlay(image_rgb, target_masks[j].astype(bool), (0, 190, 0))
                ref_vis = overlay(image_rgb, ref_masks[j].astype(bool), (0, 0, 255))
                item_row = np.concatenate(
                    [
                        tile(ref_vis, f"REF {j}: miner"),
                        tile(pred_vis, f"pred {j}"),
                        tile(gt_vis, f"target {j} iou={sample['items'][j]['iou_correct']:.3f} wrong={sample['items'][j]['iou_wrong']:.3f}"),
                    ],
                    axis=1,
                )
                cv2.imwrite(str(overlay_dir / f"{idx:05d}_{j}.jpg"), cv2.cvtColor(item_row, cv2.COLOR_RGB2BGR))

        if (idx + 1) % 50 == 0 or idx + 1 == len(cfs):
            print(f"counterfactual_eval done={idx + 1}/{len(cfs)}", flush=True)

    summary = {
        "model": args.model,
        "counterfactual_jsonl": args.counterfactual_jsonl,
        "pairs_jsonl": args.pairs_jsonl,
        "conversation_mode": args.conversation_mode,
        "ref_image_mode": args.ref_image_mode,
        "focus_dilate": args.focus_dilate,
        "focus_background": args.focus_background,
        "num_counterfactual": len(rows),
        "num_ref_trials": pair_total,
        "mean_iou_correct": float(np.mean(correct_ious)) if correct_ious else 0.0,
        "mean_iou_wrong": float(np.mean(wrong_ious)) if wrong_ious else 0.0,
        "mean_rcs": float(np.mean(rcs_values)) if rcs_values else 0.0,
        "median_rcs": float(np.median(rcs_values)) if rcs_values else 0.0,
        "pair_success_rate": float(pair_success / pair_total) if pair_total else 0.0,
        "counterfactual_success_rate": float(cf_success / len(rows)) if rows else 0.0,
    }
    write_jsonl(out_dir / "counterfactual_results.jsonl", rows)
    (out_dir / "counterfactual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

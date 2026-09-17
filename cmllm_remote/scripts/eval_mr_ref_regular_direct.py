import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

LISA_ROOT = "/home/wjq/cmllm/third_party/LISA"
if LISA_ROOT not in sys.path:
    sys.path.insert(0, LISA_ROOT)

from model.LISA import LISAForCausalLM  # noqa: E402
from model.llava import conversation as conversation_lib  # noqa: E402
from utils.dataset import collate_fn  # noqa: E402
from utils.mr_ref_seg_dataset import MultiRoundRefSegDataset  # noqa: E402


def get_added_token_id(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0 or token_id == tokenizer.unk_token_id:
        token_id = tokenizer(token, add_special_tokens=False).input_ids[-1]
    return token_id


class InferenceWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], True)


def compute_metrics(pred_masks, gt_masks):
    pred = (pred_masks > 0).astype(bool)
    gt = gt_masks.astype(bool)
    intersections = np.logical_and(pred, gt).reshape(pred.shape[0], -1).sum(axis=1)
    unions = np.logical_or(pred, gt).reshape(pred.shape[0], -1).sum(axis=1)
    per_mask_iou = intersections / (unions + 1e-5)
    per_mask_iou[unions == 0] = 1.0
    return intersections.sum(), unions.sum(), per_mask_iou.sum(), pred.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--vision-pretrained",
        default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth",
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

    os.environ["MR_REF_IMAGE_MODE"] = args.ref_image_mode
    os.environ["MR_REF_FOCUS_DILATE"] = str(args.focus_dilate)
    os.environ["MR_REF_FOCUS_BACKGROUND"] = str(args.focus_background)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    samples_per_epoch = args.max_items if args.max_items > 0 else None
    dataset = MultiRoundRefSegDataset(
        args.jsonl,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=samples_per_epoch,
        image_size=args.image_size,
        deterministic=True,
    )
    loader = DataLoader(
        InferenceWrapper(dataset),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer=tokenizer,
            conv_type="llava_v1",
            use_mm_start_end=True,
            local_rank=0,
        ),
    )

    total_intersection = 0.0
    total_union = 0.0
    total_iou = 0.0
    total_masks = 0
    rows = []
    for idx, batch in enumerate(tqdm(loader)):
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
        batch["images"] = batch["images"].to(dtype=dtype)
        batch["images_clip"] = batch["images_clip"].to(dtype=dtype)

        with torch.no_grad():
            output = model(**batch)
        pred = output["pred_masks"][0].detach().float().cpu().numpy()
        gt = output["gt_masks"][0].detach().float().cpu().numpy()
        inter, union, iou_sum, n_masks = compute_metrics(pred, gt)
        total_intersection += inter
        total_union += union
        total_iou += iou_sum
        total_masks += n_masks
        rows.append(
            {
                "idx": idx,
                "image_path": batch["image_paths"][0],
                "intersection": float(inter),
                "union": float(union),
                "mean_iou": float(iou_sum / max(n_masks, 1)),
                "num_masks": int(n_masks),
            }
        )

    summary = {
        "model": args.model,
        "jsonl": args.jsonl,
        "ref_image_mode": args.ref_image_mode,
        "focus_dilate": args.focus_dilate,
        "focus_background": args.focus_background,
        "num_items": len(rows),
        "num_masks": total_masks,
        "giou": float(total_iou / max(total_masks, 1)),
        "ciou": float(total_intersection / (total_union + 1e-10)),
    }
    (out_dir / "regular_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "regular_results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("giou: {:.4f}, ciou: {:.4f}".format(summary["giou"], summary["ciou"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

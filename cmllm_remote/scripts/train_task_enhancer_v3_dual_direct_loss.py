#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage3_rule_controller_v3_seg_local_enhance import (  # noqa: E402
    ControllerRunner,
    degrade_parametric,
    load_episode_assets,
)
from train_task_enhancer_v1 import (  # noqa: E402
    TaskEnhancerUNet,
    charbonnier,
    enhancer_loss,
    gradient_xy,
    read_jsonl,
)
from utils.dataset import collate_fn  # noqa: E402


MINER_QUERY = "Segment the coal miner"
HELMET_QUERY = "Segment the mining helmet"
TARGET15_B = {
    "id": "target15_b",
    "gamma": 2.15,
    "scale": 0.405,
    "contrast": 0.655,
    "noise_sigma": 23.5,
    "blur": 0.88,
}


def stable_u32(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def is_usable_episode(ep):
    rounds = ep.get("rounds", [])
    return bool(len(rounds) >= 2 and ep.get("image_path") and rounds[0].get("target_mask") and rounds[1].get("target_mask"))


def resize_np(image, size, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(image, (size, size), interpolation=interp)


def to_chw_float(image_rgb):
    return torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


def mask_to_tensor(mask):
    return torch.from_numpy((mask > 0).astype(np.float32))


def tensor_to_uint8_rgb(x_01):
    arr = x_01.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def preprocess_sam_diff_resize(x_01, image_size):
    """Differentiable ResizeLongestSide + SAM normalization/padding for CHW image in [0,1]."""
    _, h, w = x_01.shape
    scale = float(image_size) / float(max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    x = F.interpolate(x_01[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
    x = x * 255.0
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=x.device, dtype=x.dtype).view(3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=x.device, dtype=x.dtype).view(3, 1, 1)
    x = (x - pixel_mean) / pixel_std
    return F.pad(x, (0, image_size - new_w, 0, image_size - new_h))


def seg_bce_dice_loss(pred_logits, gt_mask):
    if gt_mask.ndim == 2:
        gt_mask = gt_mask[None]
    gt_mask = gt_mask.to(device=pred_logits.device, dtype=pred_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction="mean")
    probs = pred_logits.sigmoid()
    probs_f = probs.flatten(1)
    gt_f = gt_mask.flatten(1)
    inter = (probs_f * gt_f).sum(dim=1)
    denom = probs_f.sum(dim=1) + gt_f.sum(dim=1)
    dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6))
    return bce + dice.mean(), {
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.mean().detach().cpu()),
    }


def tv_loss(x):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    gx, gy = gradient_xy(x)
    return charbonnier(gx).mean() + charbonnier(gy).mean()


def color_stats_loss(enhanced, clean):
    e = enhanced.flatten(1)
    c = clean.flatten(1)
    mean_loss = charbonnier(e.mean(dim=1) - c.mean(dim=1)).mean()
    std_loss = charbonnier(e.std(dim=1) - c.std(dim=1)).mean()
    return mean_loss + std_loss


def differentiable_get_visual_embs(self, pixel_values):
    image_embeddings_list = []
    for i in range(pixel_values.shape[0]):
        image_embeddings = self.model.visual_model.image_encoder(pixel_values[i].unsqueeze(0))
        image_embeddings_list.append(image_embeddings)
    return torch.cat(image_embeddings_list, 0)


def load_enhancer_path(path, base_channels, device):
    model = TaskEnhancerUNet(base_channels).to(device)
    if path:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
    return model


def prepare_episode_tensors(ep, cfg, args):
    image_rgb, miner_gt, helmet_gt, _round1, _round2 = load_episode_assets(ep)
    clean = resize_np(image_rgb, args.train_image_size, is_mask=False)
    miner = resize_np(miner_gt, args.train_image_size, is_mask=True)
    helmet = resize_np(helmet_gt, args.train_image_size, is_mask=True)
    seed = stable_u32(f"{ep.get('episode_id', ep['image_path'])}:{cfg['id']}")
    degraded = degrade_parametric(clean, cfg, seed)
    return clean, degraded, miner.astype(np.float32), helmet.astype(np.float32)


def move_batch_to_cuda(batch, dtype):
    for key in ["images_clip", "input_ids", "labels", "attention_masks", "offset"]:
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
        batch[key] = [x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch[key]]
    batch["images_clip"] = batch["images_clip"].to(dtype=dtype)
    return batch


def build_direct_batch(runner, enhanced_np, query, target_np, enhanced_sam, dtype):
    item = runner.build_item(enhanced_np, query, target_np.astype(np.float32), ref_mask=None, ref_bbox=None)
    batch = collate_fn(
        [item],
        tokenizer=runner.tokenizer,
        conv_type="llava_v1",
        use_mm_start_end=True,
        local_rank=0,
    )
    batch = move_batch_to_cuda(batch, dtype)
    batch["images"] = enhanced_sam[None].to(dtype=dtype)
    return batch


@torch.no_grad()
def evaluate_alignment(enhancer, rows, cfg, args, device):
    enhancer.eval()
    metrics = []
    for ep in rows:
        clean_np, degraded_np, miner_np, helmet_np = prepare_episode_tensors(ep, cfg, args)
        clean = to_chw_float(clean_np).to(device)
        degraded = to_chw_float(degraded_np).to(device)
        task_mask = torch.clamp(0.5 * mask_to_tensor(miner_np) + mask_to_tensor(helmet_np), 0, 1).to(device)[None]
        pred = enhancer(degraded[None])[0]
        loss, parts = enhancer_loss(pred[None], clean[None], task_mask[None], args.mask_weight, args.edge_weight)
        mse = F.mse_loss(pred, clean).clamp_min(1e-8)
        metrics.append(
            {
                "align_loss": float(loss.cpu()),
                "psnr": float((10.0 * torch.log10(1.0 / mse)).cpu()),
                "masked_l1": float((torch.abs(pred - clean) * (task_mask + 1e-3)).sum().cpu() / ((task_mask + 1e-3).sum().cpu() * 3.0)),
                **parts,
            }
        )
    enhancer.train()
    if not metrics:
        return {}
    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}


def train_one_step(runner, enhancer, optimizer, ep, cfg, args, device, teacher=None):
    clean_np, degraded_np, miner_np, helmet_np = prepare_episode_tensors(ep, cfg, args)
    clean = to_chw_float(clean_np).to(device)
    degraded = to_chw_float(degraded_np).to(device)
    miner = mask_to_tensor(miner_np).to(device)
    helmet = mask_to_tensor(helmet_np).to(device)
    task_mask = torch.clamp(0.5 * miner + helmet, 0, 1)[None]

    enhanced = enhancer(degraded[None])[0]
    enhanced_np = tensor_to_uint8_rgb(enhanced)
    enhanced_sam = preprocess_sam_diff_resize(enhanced, args.image_size)

    batch_miner = build_direct_batch(runner, enhanced_np, MINER_QUERY, miner_np, enhanced_sam, runner.dtype)
    out_miner = runner.model(**batch_miner)
    miner_loss, miner_parts = seg_bce_dice_loss(out_miner["pred_masks"][0], miner)

    batch_helmet = build_direct_batch(runner, enhanced_np, HELMET_QUERY, helmet_np, enhanced_sam, runner.dtype)
    out_helmet = runner.model(**batch_helmet)
    helmet_loss, helmet_parts = seg_bce_dice_loss(out_helmet["pred_masks"][0], helmet)

    align, align_parts = enhancer_loss(enhanced[None], clean[None], task_mask[None], args.mask_weight, args.edge_weight)
    residual = charbonnier(enhanced - degraded).mean()
    tv = tv_loss(enhanced)
    color = color_stats_loss(enhanced, clean)

    teacher_loss = enhanced.new_tensor(0.0)
    if teacher is not None:
        with torch.no_grad():
            teacher_enhanced = teacher(degraded[None])[0]
        teacher_loss = charbonnier(enhanced - teacher_enhanced).mean()

    loss = (
        args.lambda_align * align
        + args.lambda_miner * miner_loss
        + args.lambda_helmet * helmet_loss
        + args.lambda_residual * residual
        + args.lambda_teacher * teacher_loss
        + args.lambda_color * color
        + args.lambda_tv * tv
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(enhancer.parameters(), args.grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "align": float(align.detach().cpu()),
        "miner_seg": float(miner_loss.detach().cpu()),
        "helmet_seg": float(helmet_loss.detach().cpu()),
        "residual": float(residual.detach().cpu()),
        "teacher": float(teacher_loss.detach().cpu()),
        "color": float(color.detach().cpu()),
        "tv": float(tv.detach().cpu()),
        "miner_bce": miner_parts["bce"],
        "miner_dice": miner_parts["dice"],
        "helmet_bce": helmet_parts["bce"],
        "helmet_dice": helmet_parts["dice"],
        **align_parts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-enhancer", default="")
    parser.add_argument("--teacher-enhancer", default="")
    parser.add_argument("--max-train", type=int, default=300)
    parser.add_argument("--max-val", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-image-size", type=int, default=384)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-miner", type=float, default=0.025)
    parser.add_argument("--lambda-helmet", type=float, default=0.05)
    parser.add_argument("--lambda-residual", type=float, default=0.05)
    parser.add_argument("--lambda-teacher", type=float, default=0.25)
    parser.add_argument("--lambda-color", type=float, default=0.05)
    parser.add_argument("--lambda-tv", type=float, default=0.005)
    parser.add_argument("--mask-weight", type=float, default=5.0)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--degradation-config-json", default=json.dumps(TARGET15_B))
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument("--vision-pretrained", default="/home/wjq/cmllm/models/sam/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.degradation_config_json)

    rows = [r for r in read_jsonl(args.jsonl) if is_usable_episode(r)]
    random.Random(args.seed).shuffle(rows)
    train_rows = rows[: args.max_train]
    val_rows = rows[args.max_train : args.max_train + args.max_val]
    args_dict = vars(args).copy()
    args_dict.update(
        {
            "degradation_config": cfg,
            "num_train": len(train_rows),
            "num_val": len(val_rows),
            "miner_query": MINER_QUERY,
            "helmet_query": HELMET_QUERY,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(args_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    runner = ControllerRunner(args)
    runner.model.get_visual_embs = types.MethodType(differentiable_get_visual_embs, runner.model)
    for p in runner.model.parameters():
        p.requires_grad_(False)
    runner.model.eval()

    device = torch.device("cuda")
    enhancer = load_enhancer_path(args.init_enhancer, args.base_channels, device)
    enhancer.train()
    teacher = None
    if args.teacher_enhancer and args.lambda_teacher > 0:
        teacher = load_enhancer_path(args.teacher_enhancer, args.base_channels, device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(enhancer.parameters(), lr=args.lr, weight_decay=1e-4)

    best_metric = float("inf")
    global_step = 0
    log_path = out_dir / "train_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        running = []
        pbar = tqdm(train_rows, desc=f"enhancer-v3 epoch {epoch}/{args.epochs}")
        for ep in pbar:
            stats = train_one_step(runner, enhancer, optimizer, ep, cfg, args, device, teacher=teacher)
            global_step += 1
            running.append(stats)
            pbar.set_postfix(loss=stats["loss"], miner=stats["miner_seg"], helmet=stats["helmet_seg"])

        val_metrics = evaluate_alignment(enhancer, val_rows, cfg, args, device)
        epoch_stats = {
            "epoch": epoch,
            "step": global_step,
            **{f"train_{k}": float(np.mean([s[k] for s in running])) for k in running[0].keys()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_stats, ensure_ascii=False) + "\n")
        ckpt = {
            "model": enhancer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "metrics": epoch_stats,
            "args": args_dict,
        }
        torch.save(ckpt, out_dir / "last.pt")
        select_metric = (
            epoch_stats["train_helmet_seg"]
            + 0.5 * epoch_stats["train_miner_seg"]
            + 0.25 * epoch_stats.get("val_align_loss", 0.0)
        )
        if select_metric < best_metric:
            best_metric = select_metric
            torch.save(ckpt, out_dir / "best.pt")
        print(json.dumps(epoch_stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

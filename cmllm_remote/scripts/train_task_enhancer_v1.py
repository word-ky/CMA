#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_DEGRADE_CONFIGS = [
    {"id": "d20_tiny", "gamma": 1.00, "scale": 0.97, "contrast": 0.98, "noise_sigma": 1.0, "blur": 0.0},
    {"id": "d20_mild", "gamma": 1.06, "scale": 0.91, "contrast": 0.96, "noise_sigma": 3.0, "blur": 0.0},
    {"id": "d20_light_low", "gamma": 1.15, "scale": 0.84, "contrast": 0.93, "noise_sigma": 5.0, "blur": 0.0},
    {"id": "d20_low", "gamma": 1.20, "scale": 0.80, "contrast": 0.91, "noise_sigma": 6.0, "blur": 0.0},
]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.float32)


def resize_image_and_masks(image, masks, size):
    image_r = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    out_masks = []
    for mask in masks:
        out_masks.append(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST))
    return image_r, out_masks


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


def episode_masks(ep):
    rounds = ep.get("rounds", [])
    miner = rounds[0].get("target_mask") if len(rounds) > 0 else None
    helmet = rounds[1].get("target_mask") if len(rounds) > 1 else None
    return miner, helmet


def is_usable_episode(ep):
    image_path = ep.get("image_path")
    miner_path, helmet_path = episode_masks(ep)
    return bool(image_path and miner_path and helmet_path)


class EnhancerDataset(Dataset):
    def __init__(self, rows, image_size, configs, seed):
        self.rows = rows
        self.image_size = image_size
        self.configs = configs
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ep = self.rows[idx]
        image = load_rgb(ep["image_path"])
        miner_path, helmet_path = episode_masks(ep)
        miner = load_mask(miner_path)
        helmet = load_mask(helmet_path)
        image, (miner, helmet) = resize_image_and_masks(image, [miner, helmet], self.image_size)

        rng_seed = self.seed + idx * 9973 + random.randint(0, 9999)
        cfg = self.configs[rng_seed % len(self.configs)]
        degraded = degrade_parametric(image, cfg, rng_seed)

        clean = image.astype(np.float32) / 255.0
        degraded = degraded.astype(np.float32) / 255.0
        target_mask = np.clip(miner * 0.5 + helmet * 1.0, 0.0, 1.0)

        return {
            "degraded": torch.from_numpy(degraded).permute(2, 0, 1).float(),
            "clean": torch.from_numpy(clean).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(target_mask[None]).float(),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TaskEnhancerUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        m = self.mid(e3)
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))
        residual = torch.tanh(self.out(d1)) * 0.35
        return torch.clamp(x + residual, 0.0, 1.0)


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def gradient_xy(x):
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return gx, gy


def enhancer_loss(pred, clean, mask, mask_weight, edge_weight):
    weight = 1.0 + mask_weight * mask
    rec = (charbonnier(pred - clean) * weight).mean()
    pred_gx, pred_gy = gradient_xy(pred)
    clean_gx, clean_gy = gradient_xy(clean)
    mask_x = F.interpolate(mask, size=pred_gx.shape[-2:], mode="nearest")
    mask_y = F.interpolate(mask, size=pred_gy.shape[-2:], mode="nearest")
    edge = (charbonnier(pred_gx - clean_gx) * (1.0 + mask_weight * mask_x)).mean()
    edge = edge + (charbonnier(pred_gy - clean_gy) * (1.0 + mask_weight * mask_y)).mean()
    return rec + edge_weight * edge, {"rec": float(rec.detach().cpu()), "edge": float(edge.detach().cpu())}


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    losses = []
    psnrs = []
    masked_l1 = []
    for batch in loader:
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)
        mask = batch["mask"].to(device)
        pred = model(degraded)
        loss, _ = enhancer_loss(pred, clean, mask, args.mask_weight, args.edge_weight)
        mse = F.mse_loss(pred, clean, reduction="none").mean(dim=(1, 2, 3)).clamp_min(1e-8)
        psnr = 10.0 * torch.log10(1.0 / mse)
        ml1 = (torch.abs(pred - clean) * (mask + 1e-3)).sum() / ((mask + 1e-3).sum() * 3.0)
        losses.append(float(loss.detach().cpu()))
        psnrs.extend([float(x) for x in psnr.detach().cpu()])
        masked_l1.append(float(ml1.detach().cpu()))
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "psnr": float(np.mean(psnrs)),
        "masked_l1": float(np.mean(masked_l1)),
    }


def save_checkpoint(path, model, optimizer, epoch, step, metrics, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--mask-weight", type=float, default=5.0)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--degrade-config-jsonl", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    rows = [row for row in read_jsonl(args.jsonl) if is_usable_episode(row)]
    random.Random(args.seed).shuffle(rows)
    train_rows = rows[: args.max_train]
    val_rows = rows[args.max_train : args.max_train + args.max_val]
    if args.degrade_config_jsonl:
        configs = read_jsonl(args.degrade_config_jsonl)
    else:
        configs = DEFAULT_DEGRADE_CONFIGS

    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "num_train": len(train_rows), "num_val": len(val_rows), "degrade_configs": configs}, indent=2),
        encoding="utf-8",
    )

    train_ds = EnhancerDataset(train_rows, args.image_size, configs, args.seed)
    val_ds = EnhancerDataset(val_rows, args.image_size, configs, args.seed + 12345)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskEnhancerUNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_loss = math.inf
    global_step = 0
    log_path = out_dir / "train_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                pred = model(degraded)
                loss, parts = enhancer_loss(pred, clean, mask, args.mask_weight, args.edge_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()), rec=parts["rec"], edge=parts["edge"])

        metrics = evaluate(model, val_loader, device, args)
        metrics.update({"epoch": epoch, "step": global_step})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, global_step, metrics, args)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, global_step, metrics, args)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Single Cycle007 adaptation: frozen w15, grouped REF losses, v3 teacher."""
import argparse
import hashlib
import json
import random
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Import evaluator first to resolve the recovered LISA root.
from eval_mr_ref_counterfactual_v0 import build_item
from model.LISA import soft_iou_matrix
from train_task_enhancer_v3_dual_direct_loss import (
    ControllerRunner, MINER_QUERY, HELMET_QUERY, build_direct_batch,
    charbonnier, color_stats_loss, differentiable_get_visual_embs, enhancer_loss,
    load_enhancer_path, move_batch_to_cuda, preprocess_sam_diff_resize,
    read_jsonl, seg_bce_dice_loss, tensor_to_uint8_rgb, to_chw_float, tv_loss,
)
from utils.dataset import collate_fn


def counterfactual_rank_loss(logits, targets, margin=0.05):
    scores = soft_iou_matrix(logits.float(), targets.float())
    wrong = scores.masked_fill(torch.eye(len(scores), device=scores.device, dtype=torch.bool), -1).max(1).values
    return F.relu(margin + wrong - scores.diagonal()).mean()


def ref_forward(runner, enhancer, group, args):
    clean_rgb = cv2.cvtColor(cv2.imread(group["image_path"]), cv2.COLOR_BGR2RGB)
    h, w = clean_rgb.shape[:2]
    size = args.train_image_size
    clean = to_chw_float(cv2.resize(clean_rgb, (size, size), interpolation=cv2.INTER_AREA)).cuda()
    pairs = {}
    for pair in group["pairs"]:
        box = pair["miner_bbox_xyxy"]
        pairs[pair["pair_id"]] = {**pair, "miner_bbox_xyxy": [box[0]*size/w, box[1]*size/h, box[2]*size/w, box[3]*size/h]}
    tensors = {}

    def enhance_once(degraded_rgb):
        # Existing deterministic degradation at source resolution; fixed384
        # training resize. Main and all REF paths share this one output.
        degraded = to_chw_float(cv2.resize(degraded_rgb, (size, size), interpolation=cv2.INTER_AREA)).cuda()
        enhanced = enhancer(degraded[None])[0]
        tensors.update(degraded=degraded, enhanced=enhanced)
        return tensor_to_uint8_rgb(enhanced)

    item, enhanced_np, targets_np, miners_np, indices = build_item(
        group, pairs, runner.clip_processor, runner.transform, args.image_size,
        "v1_multiround", condition="target15_b", seed=0, enhance_image=enhance_once)
    sam = preprocess_sam_diff_resize(tensors["enhanced"], args.image_size)
    # One SAM encoding shared by REF and two direct-task forwards. No detach.
    embeddings = differentiable_get_visual_embs(runner.model, sam[None].to(runner.dtype))
    runner.model.get_visual_embs = lambda pixel_values: embeddings
    batch = collate_fn([item], tokenizer=runner.tokenizer, conv_type="llava_v1", use_mm_start_end=True, local_rank=0)
    batch = move_batch_to_cuda(batch, runner.dtype)
    batch["images"] = sam[None].to(runner.dtype)
    logits = runner.model(**batch)["pred_masks"][0][indices].float()
    targets = torch.from_numpy(targets_np).cuda().float()
    ref_loss, _ = seg_bce_dice_loss(logits, targets)
    rank_loss = counterfactual_rank_loss(logits, targets, args.cf_rank_margin)
    return {**tensors, "clean": clean, "enhanced_np": enhanced_np, "sam": sam,
            "targets": targets, "miners": torch.from_numpy(miners_np).cuda().float(),
            "targets_np": targets_np, "miners_np": miners_np,
            "ref_loss": ref_loss, "rank_loss": rank_loss,
            "ref_pixels_detached": not batch["ref_images_clip_list"][0].requires_grad,
            "main_sam_requires_grad": sam.requires_grad}


def full_loss(runner, teacher, state, args):
    enhanced, clean, degraded = (state[k] for k in ("enhanced", "clean", "degraded"))
    direct = {}
    for name, query, masks, masks_np in (
        ("miner", MINER_QUERY, state["miners"], state["miners_np"]),
        ("helmet", HELMET_QUERY, state["targets"], state["targets_np"]),
    ):
        batch = build_direct_batch(runner, state["enhanced_np"], query, masks_np[0], state["sam"], runner.dtype)
        logits = runner.model(**batch)["pred_masks"][0].float()
        # Direct query has no identity; average the historical per-identity loss.
        direct[name], _ = seg_bce_dice_loss(logits.expand_as(masks), masks)
    task_masks = torch.clamp(0.5 * state["miners"] + state["targets"], 0, 1)[:, None]
    n = len(task_masks)
    align, _ = enhancer_loss(enhanced[None].expand(n,-1,-1,-1), clean[None].expand(n,-1,-1,-1), task_masks, args.mask_weight, args.edge_weight)
    with torch.no_grad():
        teacher_output = teacher(degraded[None])[0]
    losses = {"align": align, "miner": direct["miner"], "helmet": direct["helmet"],
              "residual": charbonnier(enhanced-degraded).mean(),
              "teacher": charbonnier(enhanced-teacher_output).mean(),
              "color": color_stats_loss(enhanced,clean), "tv": tv_loss(enhanced),
              "ref_target": state["ref_loss"], "cf_rank": state["rank_loss"]}
    loss = sum(getattr(args,"lambda_"+k)*v for k,v in losses.items())
    return loss, {k:float(v.detach()) for k,v in losses.items()}


def main():
    p = argparse.ArgumentParser()
    for name in ("jsonl", "out-dir", "model", "init-enhancer", "vision-tower"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--vision-pretrained", default=None)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--model-max-length", type=int, default=512)
    p.add_argument("--train-image-size", type=int, default=384)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mask-weight", type=float, default=5.0)
    p.add_argument("--edge-weight", type=float, default=0.15)
    p.add_argument("--cf-rank-margin", type=float, default=0.05)
    for name,value in {"align":1.,"miner":.005,"helmet":.01,"residual":.08,"teacher":.25,"color":.08,"tv":.005,"ref_target":.02,"cf_rank":.5}.items():
        p.add_argument("--lambda-"+name.replace("_","-"),type=float,default=value)
    args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(args.jsonl)
    assert len(rows)==300, "Use the frozen300 manifest"
    config={**vars(args),"epochs":1,"weight_decay":1e-4,"teacher":args.init_enhancer,
            "split_sha256":hashlib.sha256(Path(args.jsonl).read_bytes()).hexdigest(),
            "init_sha256":hashlib.sha256(Path(args.init_enhancer).read_bytes()).hexdigest(),
            "training_pixels":"source target15_b seed0 -> AREA square384 -> enhancer; detached uint8 REF/CLIP; differentiable main SAM bilinear1024",
            "direct_loss":"mean over supplied identities; no REF for historical direct terms"}
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    runner=ControllerRunner(args)
    runner.model.requires_grad_(False).eval()
    enhancer=load_enhancer_path(args.init_enhancer,args.base_channels,"cuda").train()
    if args.smoke_only:
        state=ref_forward(runner,enhancer,rows[0],args)
        (state["ref_loss"]+state["rank_loss"]).backward()
        norm=torch.stack([p.grad.float().square().sum() for p in enhancer.parameters() if p.grad is not None]).sum().sqrt()
        result={"group_id":rows[0]["counterfactual_id"],"loss_ref_target":float(state["ref_loss"].detach()),
                "loss_cf_rank":float(state["rank_loss"].detach()),"enhancer_gradient_norm":float(norm),
                "all_enhancer_gradients_finite":all(bool(torch.isfinite(p.grad).all()) for p in enhancer.parameters() if p.grad is not None),
                "w15_frozen_no_grad":all(not p.requires_grad and p.grad is None for p in runner.model.parameters()),
                "ref_pixels_detached":state["ref_pixels_detached"],"main_sam_requires_grad":state["main_sam_requires_grad"],
                "optimizer_steps":0,"optimizer_created":False}
        (out/"gradient_smoke.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)
        assert result["all_enhancer_gradients_finite"] and 0<float(norm)<float("inf") and result["w15_frozen_no_grad"]
        return
    teacher=load_enhancer_path(args.init_enhancer,args.base_channels,"cuda").requires_grad_(False).eval()
    optimizer=torch.optim.AdamW(enhancer.parameters(),lr=args.lr,weight_decay=1e-4)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in enhancer.parameters()}
    random.shuffle(rows)
    stats=[]
    for step,group in enumerate(rows,1):
        optimizer.zero_grad(set_to_none=True)
        state=ref_forward(runner,enhancer,group,args)
        loss,parts=full_loss(runner,teacher,state,args)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(enhancer.parameters(),args.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        record={"step":step,"group_id":group["counterfactual_id"],"loss":float(loss.detach()),"grad_norm":float(norm),**parts}
        with (out/"train_log.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        stats.append(record)
        if step%10==0:print(json.dumps(record),flush=True)
        # Release the shared embedding graph between groups.
        runner.model.get_visual_embs=types.MethodType(differentiable_get_visual_embs,runner.model)
        del state,loss
    summary={"steps":len(stats),"epochs":1,"w15_frozen_no_grad":all(not p.requires_grad and p.grad is None for p in runner.model.parameters()),
             "mean_losses":{k:float(np.mean([s[k] for s in stats])) for k in stats[0] if k not in ("step","group_id")}}
    torch.save({"model":enhancer.state_dict(),"optimizer":optimizer.state_dict(),"epoch":1,"step":len(stats),"args":config,"metrics":summary},out/"last.pt")
    (out/"train_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()

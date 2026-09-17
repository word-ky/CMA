"""Cycle008: one paired clean/degraded adaptation of existing w15 surfaces."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_mr_ref_counterfactual_v0 import build_item
from train_task_enhancer_v3_dual_direct_loss import ControllerRunner, move_batch_to_cuda, read_jsonl, seg_bce_dice_loss
from train_task_enhancer_v4_counterfactual import counterfactual_rank_loss
from utils.dataset import collate_fn

SURFACES=("visual_model.mask_decoder", "text_hidden_fcs", "ref_hidden_fcs",
          "ref_visual_fcs", "ref_input_bbox_fcs", "ref_input_fcs", "ref_embedding_scale")


def surface_for(name):
    return next((s for s in SURFACES if name=="model."+s or name.startswith("model."+s+".")),None)


def configure_trainable(model):
    model.requires_grad_(False).eval()
    selected=[]
    for name,param in model.named_parameters():
        surface=surface_for(name)
        if surface is not None:
            param.data=param.data.float()
            param.requires_grad_(True)
            selected.append({"name":name,"shape":list(param.shape),"numel":param.numel(),"surface":surface,"dtype":str(param.dtype)})
    return selected


def consistency_loss(degraded_logits,clean_logits):
    return F.binary_cross_entropy_with_logits(degraded_logits.float(),clean_logits.detach().float().sigmoid())


def condition_forward(runner,group,condition,args):
    pairs={p["pair_id"]:p for p in group["pairs"]}
    item,_,target_np,_,indices=build_item(group,pairs,runner.clip_processor,runner.transform,
                                         args.image_size,"v1_multiround",condition=condition,seed=0)
    batch=collate_fn([item],tokenizer=runner.tokenizer,conv_type="llava_v1",use_mm_start_end=True,local_rank=0)
    batch=move_batch_to_cuda(batch,runner.dtype)
    batch["images"]=batch["images"].cuda().to(runner.dtype)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits=runner.model(**batch)["pred_masks"][0][indices].float()
    targets=torch.from_numpy(target_np).cuda().float()
    seg,_=seg_bce_dice_loss(logits,targets)
    rank=counterfactual_rank_loss(logits,targets,margin=.05)
    return logits,seg,rank


def paired_backward(runner,group,args):
    clean,seg_c,rank_c=condition_forward(runner,group,"clean",args)
    clean_teacher=clean.detach()
    clean_loss=.5*seg_c+.5*rank_c
    clean_loss.backward()
    stats={"seg_clean":float(seg_c.detach()),"rank_clean":float(rank_c.detach())}
    del clean,seg_c,rank_c,clean_loss
    degraded,seg_d,rank_d=condition_forward(runner,group,"target15_b",args)
    cons=consistency_loss(degraded,clean_teacher)
    (.5*seg_d+.5*rank_d+.25*cons).backward()
    stats.update(seg_degraded=float(seg_d.detach()),rank_degraded=float(rank_d.detach()),consistency=float(cons.detach()))
    stats["loss"]=.5*(stats["seg_clean"]+stats["seg_degraded"])+.5*(stats["rank_clean"]+stats["rank_degraded"])+.25*stats["consistency"]
    return stats


def gradient_receipt(model):
    norms={s:0. for s in SURFACES}; missing=[]; finite=True; frozen_clean=True
    for name,p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None: missing.append(name)
            else:
                finite=finite and bool(torch.isfinite(p.grad).all())
                norms[surface_for(name)]+=float(p.grad.float().square().sum())
        elif p.grad is not None:frozen_clean=False
    return {"surface_gradient_norms":{k:v**.5 for k,v in norms.items()},
            "all_gradients_finite":finite,"frozen_parameters_gradient_free":frozen_clean,
            "trainable_parameters_without_gradient":missing}


def main():
    p=argparse.ArgumentParser()
    for name in ("model","vision-tower","jsonl","out-dir"):p.add_argument("--"+name,required=True)
    p.add_argument("--smoke-only",action="store_true")
    p.add_argument("--vision-pretrained",default=None)
    p.add_argument("--precision",default="bf16")
    p.add_argument("--image-size",type=int,default=1024)
    p.add_argument("--model-max-length",type=int,default=512)
    args=p.parse_args(); seed=20260528
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(args.jsonl); assert len(rows)==300
    config={**vars(args),"seed":seed,"degradation_seed":0,"epochs":1,"lr":5e-6,"weight_decay":1e-4,"clip_norm":1.,
            "objective":"0.5*(segC+segD)+0.5*(rankC+rankD)+0.25*BCE(D,sigmoid(C.detach()))",
            "rank_margin":.05,"master_dtype":"float32","forward_autocast":"bfloat16","enhancer":None,
            "split_sha256":hashlib.sha256(Path(args.jsonl).read_bytes()).hexdigest(),"surfaces":SURFACES}
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    runner=ControllerRunner(args); names=configure_trainable(runner.model)
    manifest={"total_trainable_numel":sum(r["numel"] for r in names),"tensor_count":len(names),"parameters":names}
    (out/"trainable_parameters.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest),flush=True)
    selected=[p for p in runner.model.parameters() if p.requires_grad]
    if args.smoke_only:
        stats=paired_backward(runner,rows[0],args)
        receipt={"group_id":rows[0]["counterfactual_id"],"optimizer_steps":0,"losses":stats,**gradient_receipt(runner.model)}
        (out/"gradient_smoke.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt),flush=True)
        assert receipt["all_gradients_finite"] and receipt["frozen_parameters_gradient_free"]
        assert all(0<v<float("inf") for v in receipt["surface_gradient_norms"].values())
        return
    optimizer=torch.optim.AdamW(selected,lr=5e-6,weight_decay=1e-4)
    random.shuffle(rows)
    for step,group in enumerate(rows,1):
        optimizer.zero_grad(set_to_none=True)
        stats=paired_backward(runner,group,args)
        norm=torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True)
        optimizer.step()
        record={"step":step,"group_id":group["counterfactual_id"],"grad_norm":float(norm),**stats}
        with (out/"train_log.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        if step%10==0:print(json.dumps(record),flush=True)
    state={name:param.detach().cpu() for name,param in runner.model.named_parameters() if param.requires_grad}
    torch.save({"trainable_state":state,"config":config,"step":300},out/"last.pt")
    summary={"steps":300,"epochs":1,"trainable_numel":manifest["total_trainable_numel"],
             "frozen_parameters_gradient_free":all(p.grad is None for p in runner.model.parameters() if not p.requires_grad)}
    (out/"train_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()

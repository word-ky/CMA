"""Cycle013: single corrected MG-DRA run; all original w15 parameters frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from eval_mr_ref_counterfactual_v0 import build_item, preprocess_sam
from memory_spatial_prompt import inference_shape_only_targets
from memory_dual_scale_adapter import MemoryDualScaleResidualAdapter
from dual_scale_features import encode_local_memory
from train_task_enhancer_v3_dual_direct_loss import ControllerRunner, move_batch_to_cuda, read_jsonl, seg_bce_dice_loss
from train_task_enhancer_v4_counterfactual import counterfactual_rank_loss
from utils.dataset import collate_fn


def prepare(runner,group,condition,args):
    pairs={p['pair_id']:p for p in group['pairs']}
    item,rgb,targets,_,indices=build_item(group,pairs,runner.clip_processor,runner.transform,
                                        args.image_size,'v1_multiround',condition=condition,seed=0)
    evidence=encode_local_memory(runner.model,rgb,[pairs[p]['miner_bbox_xyxy'] for p in group['pair_ids']],
                                runner.transform,args.image_size,item[6],preprocess_sam,runner.dtype)
    item=inference_shape_only_targets(item,indices)
    batch=collate_fn([item],tokenizer=runner.tokenizer,conv_type='llava_v1',use_mm_start_end=True,local_rank=0)
    batch=move_batch_to_cuda(batch,runner.dtype)
    batch['images']=batch['images'].cuda().to(runner.dtype)
    batch['memory_local_features_list']=[evidence]
    return batch,torch.from_numpy(targets).cuda().float(),indices


def task_loss(runner,prepared):
    batch,targets,indices=prepared
    logits=runner.model(**batch)['pred_masks'][0][indices].float()
    seg,_=seg_bce_dice_loss(logits,targets)
    rank=counterfactual_rank_loss(logits,targets,margin=.05)
    return .5*(seg+rank),{'seg':float(seg.detach()),'rank':float(rank.detach())}


def gradients(model):
    return {'parameters':{n:{'norm':float(p.grad.norm()) if p.grad is not None else None,
                            'finite':bool(torch.isfinite(p.grad).all()) if p.grad is not None else False}
                          for n,p in model.memory_dual_scale_adapter.named_parameters()},
            'frozen_base_gradient_free':all(p.grad is None for n,p in model.named_parameters()
                                            if not n.startswith('memory_dual_scale_adapter.'))}


def main():
    parser=argparse.ArgumentParser()
    for name in ('model','vision-tower','jsonl','out-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--vision-pretrained',default=None)
    parser.add_argument('--precision',default='bf16')
    parser.add_argument('--image-size',type=int,default=1024)
    parser.add_argument('--model-max-length',type=int,default=512)
    args=parser.parse_args();seed=20260528
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(args.jsonl);assert len(rows)==300
    runner=ControllerRunner(args);runner.model.requires_grad_(False).eval()
    runner.model.memory_dual_scale_adapter=MemoryDualScaleResidualAdapter().cuda()
    adapter=runner.model.memory_dual_scale_adapter
    names=[{'name':n,'numel':p.numel(),'shape':list(p.shape)} for n,p in runner.model.named_parameters() if p.requires_grad]
    count=sum(x['numel'] for x in names);assert count==12577
    config={**vars(args),'seed':seed,'degradation_seed':0,'lr':1e-4,'weight_decay':1e-4,'clip_norm':1.,'steps':300,
            'objective':'0.5*(segC+segD)+0.5*(rankC+rankD)','rank_margin':.05,'alpha_init':1.,'up_init':0.,
            'bottleneck':16,'roi_scale':1.25,'base_frozen':True,'precision_base':'bf16','adapter_precision':'float32',
            'split_sha256':hashlib.sha256(Path(args.jsonl).read_bytes()).hexdigest()}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'trainable_parameters.json').write_text(json.dumps({'count':count,'parameters':names},indent=2)+'\n')
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-4)
    if args.smoke_only:
        prepared=[prepare(runner,rows[0],c,args) for c in ('clean','target15_b')]
        receipt={'group_id':rows[0]['counterfactual_id'],'initialization':config,'trainable_parameters':count,
                 'mapping':prepared[1][0]['memory_local_features_list'][0]['receipts']}
        batch,_,indices=prepared[1]
        with torch.no_grad():
            base=runner.model(**{k:v for k,v in batch.items() if k!='memory_local_features_list'})['pred_masks'][0].float()
            initial=runner.model(**batch)['pred_masks'][0].float()
        receipt['initial_max_logit_difference']=float((base-initial).abs().max())
        receipt['initial_masks_exact']=bool(torch.equal(base>0,initial>0))
        torch.testing.assert_close(initial,base,atol=.01,rtol=.01)
        for sample in prepared:task_loss(runner,sample)[0].backward()
        first=gradients(runner.model);receipt['first_backward']=first
        assert first['frozen_base_gradient_free'] and all(v['finite'] for v in first['parameters'].values())
        assert first['parameters']['up.weight']['norm']>0
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        for sample in prepared:task_loss(runner,sample)[0].backward()
        second=gradients(runner.model);receipt['second_backward']=second
        assert second['frozen_base_gradient_free'] and all(v['finite'] for v in second['parameters'].values())
        assert second['parameters']['down.weight']['norm']>0
        receipt['alpha_gradient_nonzero']=second['parameters']['alpha']['norm']>0
        receipt['local_features_no_grad']=all(not s[0]['memory_local_features_list'][0]['mapped'].requires_grad for s in prepared)
        # Perturb only local tensors after the one discarded smoke optimizer step.
        evidence=batch['memory_local_features_list'][0]
        with torch.no_grad():
            normal=runner.model(**batch)['pred_masks'][0].float()
            changes=[]
            for identity in range(len(indices)):
                changed={**evidence,'mapped':evidence['mapped'].clone()}
                changed['mapped'][identity].zero_()
                altered=runner.model(**{**batch,'memory_local_features_list':[changed]})['pred_masks'][0].float()
                delta=(altered-normal).abs().flatten(1).max(1).values.tolist()
                for row,value in enumerate(delta):
                    if row!=indices[identity]:assert value==0
                changes.append({'changed_identity':identity,'max_logit_change_by_seg_row':delta})
        receipt['routing']=changes;receipt['smoke_optimizer_steps']=1;receipt['state_discarded_after_process']=True
        (out/'smoke_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
        return
    random.shuffle(rows)
    for step,group in enumerate(rows,1):
        optimizer.zero_grad(set_to_none=True);stats={}
        for condition in ('clean','target15_b'):
            sample=prepare(runner,group,condition,args)
            loss,values=task_loss(runner,sample);loss.backward()
            stats[condition]=values
            del sample,loss
        norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        record={'step':step,'group_id':group['counterfactual_id'],'grad_norm':float(norm),
                'alpha':float(adapter.alpha.detach()),'loss':.5*sum(v['seg']+v['rank'] for v in stats.values()),**stats}
        with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step%10==0:print(json.dumps(record),flush=True)
    torch.save({'adapter_state':{n:p.detach().cpu() for n,p in adapter.state_dict().items()},'config':config,'step':300},out/'last.pt')
    summary={'steps':300,'trainable_parameters':count,'frozen_base_gradient_free':gradients(runner.model)['frozen_base_gradient_free']}
    (out/'train_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

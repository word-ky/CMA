"""Validation-only selection, followed by one locked confirmation audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

FIELDS=("target_miou","cmsa","memory_fidelity","mean_identity_margin","median_identity_margin","identity_error_rate")


def choose_candidate(base_clean,candidates):
    n=base_clean["num_groups"]
    eligible={k:(v["clean"]["target_miou"]>=base_clean["target_miou"]-.01 and
                 round(v["clean"]["cmsa"]*n)>=round(base_clean["cmsa"]*n)-1) for k,v in candidates.items()}
    if not any(eligible.values()):raise ValueError("Neither candidate preserves clean performance")
    if not eligible["cycle008"]:return "no_cons",eligible
    if not eligible["no_cons"]:return "cycle008",eligible
    old,new=candidates["cycle008"]["target15_b"],candidates["no_cons"]["target15_b"]
    if new["cmsa"]>old["cmsa"]:return "no_cons",eligible
    if new["cmsa"]==old["cmsa"] and new["target_miou"]-old["target_miou"]>=.002:return "no_cons",eligible
    return "cycle008",eligible


def read_report(folder):return json.loads((folder/"memory_metrics.json").read_text())


def check_pairs(first,second):
    load=lambda folder:[json.loads(l) for l in (folder/"memory_predictions.jsonl").read_text().splitlines()]
    a,b=load(first),load(second); assert len(a)==len(b)
    for x,y in zip(a,b):
        for field in ("group_id","query","seed","image_path","memory_source"):assert x[field]==y[field]
        assert "enhancer" not in y["provenance"]
        for u,v in zip(x["trials"],y["trials"]):
            assert u["entity_id"]==v["entity_id"]
            for field in ("target","reference"):
                np.testing.assert_array_equal(np.load(first/u[field]),np.load(second/v[field]))


def select(root):
    candidates={}
    for name,cycle,kind in (("cycle008","cycle008","adapted"),("no_cons","cycle009","no_cons")):
        candidates[name]={}
        for cond in ("clean","target15_b"):
            path=root/f"outputs/{cycle}/val_{kind}_{cond}"
            candidates[name][cond]=read_report(path)["summary"]
            check_pairs(root/f"outputs/cycle008/val_base_{cond}",path)
    base=read_report(root/"outputs/cycle008/val_base_clean")["summary"]
    chosen,eligible=choose_candidate(base,candidates)
    checkpoint=root/f"outputs/{'cycle008' if chosen=='cycle008' else 'cycle009'}/train/last.pt"
    record={"selected_candidate":chosen,"checkpoint":str(checkpoint),"eligible":eligible,
            "validation":candidates,"selection_data":"validation50 only",
            "selected_at_utc":datetime.now(timezone.utc).isoformat(),
            "confirmation_lock_commit":"3a93aa7","confirmation_w15_inference_before_selection":0,
            "rule":"clean preservation -> degraded CMSA -> degraded mIoU; equal CMSA and abs mIoU delta<0.002 keeps Cycle008",
            "cycle008_minus_no_cons":{c:{m:candidates['cycle008'][c][m]-candidates['no_cons'][c][m] for m in FIELDS} for c in ("clean","target15_b")}}
    path=root/"research_log/cycle009/selection.json"; path.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


def transitions(old,new):
    counts={"fail_to_pass":0,"pass_to_fail":0,"pass_to_pass":0,"fail_to_fail":0}
    for a,b in zip(old,new):counts[("pass" if a else "fail")+"_to_"+("pass" if b else "fail")]+=1
    return counts


def confirm(root):
    table={}; delta={}; changes={}; paired={}; successes={}
    for cond in ("clean","target15_b"):
        base=root/f"outputs/cycle009/confirmation_base_{cond}"
        adapted=root/f"outputs/cycle009/confirmation_selected_{cond}"
        check_pairs(base,adapted)
        a,b=read_report(base),read_report(adapted)
        table[cond]={"base":a["summary"],"selected":b["summary"]}
        delta[cond]={m:b["summary"][m]-a["summary"][m] for m in FIELDS}
        old_cmsa=[g["cmsa"]==1 for g in a["groups"]]; new_cmsa=[g["cmsa"]==1 for g in b["groups"]]
        successes[cond]={"base":sum(old_cmsa),"selected":sum(new_cmsa)}
        changes[cond]={"cmsa_groups":transitions(old_cmsa,new_cmsa),
                       "correct_iou_0_5_references":transitions([x>=.5 for g in a["groups"] for x in g["correct_iou"]],
                                                               [x>=.5 for g in b["groups"] for x in g["correct_iou"]])}
        paired[cond]=[]
        for x,y in zip(a["groups"],b["groups"]):
            assert x["group_id"]==y["group_id"] and x["entity_ids"]==y["entity_ids"]
            paired[cond].append({"group_id":x["group_id"],"target_miou_delta":y["target_miou"]-x["target_miou"],
                                 "cmsa_base":x["cmsa"],"cmsa_selected":y["cmsa"],
                                 "correct_iou_base":x["correct_iou"],"correct_iou_selected":y["correct_iou"]})
    criteria={"degraded_miou_gain_at_least_0_01":delta["target15_b"]["target_miou"]>=.01,
              "degraded_cmsa_non_decrease":successes["target15_b"]["selected"]>=successes["target15_b"]["base"],
              "clean_miou_drop_at_most_0_01":delta["clean"]["target_miou"]>=-.01,
              "clean_cmsa_drop_at_most_one_group":successes["clean"]["selected"]>=successes["clean"]["base"]-1}
    result={"passed":all(criteria.values()),"status":"PASS" if all(criteria.values()) else "FAIL","criteria":criteria,
            "confirmation":table,"selected_minus_base":delta,"transitions":changes,"paired_groups":paired,
            "cmsa_success_groups":successes,"checkpoint_selection":json.loads((root/"research_log/cycle009/selection.json").read_text()),
            "paired_inputs_and_masks_equal":True}
    (root/"research_log/cycle009/confirmation_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("paired_groups","checkpoint_selection")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("root"); parser.add_argument("phase",choices=["select","confirm"])
    args=parser.parse_args(); (select if args.phase=="select" else confirm)(Path(args.root))

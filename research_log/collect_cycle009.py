"""Collect the ablation and single fresh confirmation, including recovery receipts."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

root=Path(sys.argv[1]); out=root/"outputs/cycle009"; log=root/"research_log/cycle009"
rows=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
old=rows(root/"outputs/cycle008/train/train_log.jsonl"); new=rows(out/"train/train_log.jsonl")
assert len(new)==300 and [x["step"] for x in new]==list(range(1,301))
assert [x["group_id"] for x in new]==[x["group_id"] for x in old]
assert all(x["consistency"]==0 for x in new)
old_params=json.loads((root/"outputs/cycle008/train/trainable_parameters.json").read_text())
new_params=json.loads((out/"train/trainable_parameters.json").read_text())
assert old_params==new_params
old_config=json.loads((root/"outputs/cycle008/train/config.json").read_text())
new_config=json.loads((out/"train/config.json").read_text())
for k,v in old_config.items():
    if k not in ("out_dir","objective"):assert new_config[k]==v,(k,v,new_config[k])
assert new_config["lambda_cons"]==0
selection=json.loads((log/"selection.json").read_text())
result=json.loads((log/"confirmation_result.json").read_text())
for name in ["smoke","train"]+[f"val_no_cons_{c}" for c in ("clean","target15_b")]+[f"confirmation_{kind}_{c}" for kind in ("base","selected") for c in ("clean","target15_b")]:
    dest=log/name; dest.mkdir(exist_ok=True)
    for path in (out/name).glob("*.json*"):shutil.copyfile(path,dest/path.name)
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
lock=json.loads((log/"confirmation_lock.json").read_text())
assert sha(log/"confirmation_source_groups.jsonl")==lock["selection_sha256"]
frozen=json.loads((log/"frozen_assets.json").read_text())
for asset in frozen["assets"]:assert sha(asset["path"])==asset["sha256"]
receipt={"new_training_attempts":1,"optimizer_steps":300,"same_training_group_order_as_cycle008":True,
         "same_trainable_parameters_and_other_config":True,"all_training_consistency_terms_zero":True,
         "ablation_checkpoint":str(out/"train/last.pt"),"ablation_checkpoint_sha256":sha(out/"train/last.pt"),
         "selected_candidate":selection["selected_candidate"],"selected_checkpoint":selection["checkpoint"],
         "selected_checkpoint_sha256":sha(selection["checkpoint"]),"confirmation_gate":result["status"],
         "confirmed_asset_hashes":len(frozen["assets"]),"selection_sha256":lock["selection_sha256"],
         "confirmation_conditions_per_model":["clean","target15_b"],"diagnostic30_inference_calls":0,
         "source_hashes":{str(p.relative_to(root)):sha(p) for p in (root/"code/cmllm/scripts/train_w15_degradation_counterfactual.py",root/"research_log/cycle009_analysis.py",root/"research_log/run_cycle009_full.sh")}}
(log/"completion_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
if result["passed"]:
    (log/"FROZEN_LAYER1_CHECKPOINT.json").write_text(json.dumps({"status":"FROZEN_STOP_TUNING","checkpoint":selection["checkpoint"],"sha256":receipt["selected_checkpoint_sha256"],"base_model":new_config["model"],"confirmation_gate":"PASS"},indent=2)+"\n")
print(json.dumps(receipt,indent=2))

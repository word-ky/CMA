"""Collect completed Cycle008; preserve all outputs regardless of gate outcome."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]); out=root/"outputs/cycle008"; log=root/"research_log/cycle008"
log.mkdir(parents=True,exist_ok=True)
read_rows=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
train=read_rows(out/"train/train_log.jsonl")
fixed=read_rows(root/"shared/data/cycle006/train_grouped_restored.jsonl")
assert len(train)==300 and [r["step"] for r in train]==list(range(1,301))
assert {r["group_id"] for r in train}=={r["counterfactual_id"] for r in fixed}
gate=json.loads((log/"validation_gate.json").read_text())
folders=["smoke","train"]+[f"val_{m}_{c}" for m in ("base","adapted") for c in ("clean","target15_b")]
if gate["passed"]:folders += [f"diagnostic30_adapted_{c}" for c in ("clean","target15_b")]
else:assert not list(out.glob("diagnostic30*"))
for name in folders:
    dest=log/name; dest.mkdir(exist_ok=True)
    for path in (out/name).glob("*.json*"):shutil.copyfile(path,dest/path.name)
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
manifest={"checkpoint":str(out/"train/last.pt"),"checkpoint_sha256":sha(out/"train/last.pt"),
          "completed_optimizer_steps":len(train),"each_fixed_training_group_used_once":True,
          "diagnostic30_conditions_run":["clean","target15_b"] if gate["passed"] else [],
          "source_hashes":{str(p.relative_to(root)):sha(p) for p in (
              root/"code/cmllm/scripts/train_w15_degradation_counterfactual.py",
              root/"code/cmllm/scripts/eval_mr_ref_counterfactual_v0.py",
              root/"research_log/run_cycle008_full.sh")}}
if gate["passed"]:
    manifest["diagnostic30"]={}
    for c in ("clean","target15_b"):
        base=root/f"outputs/cycle003_supplied_memory/{c}"
        adapted=out/f"diagnostic30_adapted_{c}"
        first=read_rows(base/"memory_predictions.jsonl")
        second=read_rows(adapted/"memory_predictions.jsonl")
        assert len(first)==len(second)==30
        for a,b in zip(first,second):
            for field in ("group_id","query","seed","image_path","memory_source"):assert a[field]==b[field]
            for x,y in zip(a["trials"],b["trials"]):
                assert x["entity_id"]==y["entity_id"]
                for field in ("reference","target"):
                    np.testing.assert_array_equal(np.load(base/x[field]),np.load(adapted/y[field]))
        manifest["diagnostic30"][c]={"base_reused_cycle003":json.loads((root/f"outputs/cycle003_supplied_memory/{c}/memory_metrics.json").read_text())["summary"],
                                    "adapted":json.loads((out/f"diagnostic30_adapted_{c}/memory_metrics.json").read_text())["summary"]}
    manifest["diagnostic30_paired_inputs_and_masks_equal"]=True
(log/"completion_receipt.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))

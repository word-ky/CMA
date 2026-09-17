"""Collect the sole completed run and verify its declared sample count."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

root=Path(sys.argv[1]); out=root/"outputs/cycle007"; log=root/"research_log/cycle007"
log.mkdir(parents=True,exist_ok=True)
read_rows=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
train=read_rows(out/"train/train_log.jsonl")
fixed=read_rows(root/"shared/data/cycle006/train_grouped_restored.jsonl")
assert len(train)==300 and [r["step"] for r in train]==list(range(1,301))
assert {r["group_id"] for r in train}=={r["counterfactual_id"] for r in fixed}
gate=json.loads((log/"validation_gate.json").read_text())
diag=out/"diagnostic30_v4"
assert diag.exists()==gate["passed"]
for name in ("smoke","train","val_v3","val_v4") + (("diagnostic30_v4",) if gate["passed"] else ()):
    dest=log/name; dest.mkdir(exist_ok=True)
    for path in (out/name).glob("*.json*"):
        shutil.copyfile(path,dest/path.name)
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
manifest={"checkpoint":str(out/"train/last.pt"),"checkpoint_sha256":sha(out/"train/last.pt"),
          "completed_optimizer_steps":len(train),"each_fixed_training_group_used_once":True,
          "diagnostic30_inference_runs":int(gate["passed"]),
          "source_hashes":{str(p.relative_to(root)):sha(p) for p in (
              root/"code/cmllm/scripts/train_task_enhancer_v4_counterfactual.py",
              root/"code/cmllm/scripts/train_task_enhancer_v3_dual_direct_loss.py",
              root/"code/cmllm/scripts/eval_mr_ref_counterfactual_v0.py",
              root/"research_log/run_cycle007_full.sh")}}
if gate["passed"]:
    manifest["diagnostic30"]={"DD":json.loads((root/"outputs/cycle003_supplied_memory/target15_b/memory_metrics.json").read_text())["summary"],
                              "v4":json.loads((diag/"memory_metrics.json").read_text())["summary"]}
(log/"completion_receipt.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))

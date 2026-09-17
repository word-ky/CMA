"""Lock the next eligible unseen holdout images before candidate training."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

root=Path(sys.argv[1]); sys.path.insert(0,str(root/"code/cmllm/scripts"))
from build_mr_ref_counterfactual_train import read_jsonl,write_jsonl,split_bucket

log=root/"research_log/cycle009"; log.mkdir(parents=True,exist_ok=True)
source=root/"shared/data/final_accepted_v1/episodes_counterfactual_clean_val.jsonl"
rows=read_jsonl(source); holdout=[r for r in rows if split_bucket(r["image_path"])>=8]
old=read_jsonl(root/"research_log/cycle002/counterfactual_holdout_first30.jsonl")
assert [r["counterfactual_id"] for r in holdout[:30]]==[r["counterfactual_id"] for r in old]
split=json.loads((root/"research_log/cycle006/split_receipt.json").read_text())
diagnostic=json.loads((root/"research_log/cycle003/frozen_assets.json").read_text())
protected={r["image_sha256"] for key in ("train","val") for r in split["selected"][key]}
protected.update(r["sha256"] for r in diagnostic["assets"] if r["type"]=="image")
pairs={r["pair_id"] for r in read_jsonl(root/"shared/data/final_accepted_v1/helmet_miner_pairs_accept_high.jsonl")}
selected=[]; receipts=[]; skipped=[]; seen=set(protected)
with zipfile.ZipFile(root/"shared/source/mining_helmet.zip") as zf:
    members=set(zf.namelist())
    for index,row in enumerate(holdout[30:],30):
        if row["image_member"] not in members or not set(row["pair_ids"])<=pairs:
            skipped.append({"group_id":row["counterfactual_id"],"reason":"missing source metadata"}); continue
        digest=hashlib.sha256(zf.read(row["image_member"])).hexdigest()
        if digest in seen:
            skipped.append({"group_id":row["counterfactual_id"],"reason":"protected or duplicate image bytes"}); continue
        selected.append(row); seen.add(digest)
        receipts.append({"group_id":row["counterfactual_id"],"source_holdout_index":index,"image_member":row["image_member"],"image_sha256":digest,"bucket":split_bucket(row["image_path"])})
        if len(selected)==30:break
assert len(selected)>=20,"Fewer than20 complete disjoint holdout groups"
path=log/"confirmation_source_groups.jsonl"; write_jsonl(path,selected)
report={"status":"LOCKED_BEFORE_NO_CONSISTENCY_TRAINING","groups":len(selected),"shortfall":30-len(selected),
        "rule":"original holdout source order after first30; image-byte exclusion; no score/size/difficulty filtering",
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"selection_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "protected_unique_image_hashes":len(protected),"disjoint_from_train300_val50_diagnostic30":True,
        "selection":receipts,"skipped":skipped,"confirmation_w15_inference_runs":0}
(log/"confirmation_lock.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("selection","skipped")},indent=2))

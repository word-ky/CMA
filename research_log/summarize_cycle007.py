"""Apply the predeclared validation gate to the sole v4 checkpoint."""
import json
import sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]); log=root/"research_log/cycle007"; log.mkdir(parents=True,exist_ok=True)
folders={v:root/f"outputs/cycle007/val_{v}" for v in ("v3","v4")}
reports={v:json.loads((p/"memory_metrics.json").read_text()) for v,p in folders.items()}
rows={v:[json.loads(l) for l in (p/"memory_predictions.jsonl").read_text().splitlines()] for v,p in folders.items()}
assert len(rows["v3"])==len(rows["v4"])==50
for a,b in zip(rows["v3"],rows["v4"]):
    for key in ("group_id","query","seed","image_path","memory_source"):
        assert a[key]==b[key]
    for x,y in zip(a["trials"],b["trials"]):
        assert x["entity_id"]==y["entity_id"]
        for key in ("target","reference"):
            np.testing.assert_array_equal(np.load(folders["v3"]/x[key]),np.load(folders["v4"]/y[key]))
fields=("target_miou","cmsa","memory_fidelity","mean_identity_margin","median_identity_margin","identity_error_rate")
delta={k:reports["v4"]["summary"][k]-reports["v3"]["summary"][k] for k in fields}
passed=delta["target_miou"]>=.02 and delta["cmsa"]>=0
result={"status":"PASS" if passed else "FAIL","passed":passed,"gate":{"target_miou_delta_min":.02,"cmsa_delta_min":0},
        "validation":{v:reports[v]["summary"] for v in reports},"v4_minus_v3":delta,
        "checkpoint_provenance":{v:rows[v][0]["provenance"]["enhancer"] for v in rows},
        "paired_inputs_and_masks_equal":True,"training_attempts":1,"checkpoint_selection":"sole final epoch checkpoint; no search"}
(log/"validation_gate.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

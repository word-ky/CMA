"""Four-cell no-enhancer validation and the predeclared Cycle008 gate."""
import json
import sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]); log=root/"research_log/cycle008"; log.mkdir(parents=True,exist_ok=True)
folders={f"{model}_{cond}":root/f"outputs/cycle008/val_{model}_{cond}"
         for model in ("base","adapted") for cond in ("clean","target15_b")}
reports={k:json.loads((p/"memory_metrics.json").read_text()) for k,p in folders.items()}
rows={k:[json.loads(l) for l in (p/"memory_predictions.jsonl").read_text().splitlines()] for k,p in folders.items()}
reference=rows["base_clean"]
for key,rr in rows.items():
    assert len(rr)==len(reference)==50
    for a,b in zip(reference,rr):
        for field in ("group_id","query","seed","image_path","memory_source"):assert a[field]==b[field]
        assert "enhancer" not in b["provenance"]
        for x,y in zip(a["trials"],b["trials"]):
            assert x["entity_id"]==y["entity_id"]
            for field in ("target","reference"):
                np.testing.assert_array_equal(np.load(folders["base_clean"]/x[field]),np.load(folders[key]/y[field]))
fields=("target_miou","cmsa","memory_fidelity","mean_identity_margin","median_identity_margin","identity_error_rate")
table={k:r["summary"] for k,r in reports.items()}
deltas={c:{m:table[f"adapted_{c}"][m]-table[f"base_{c}"][m] for m in fields} for c in ("clean","target15_b")}
success={k:sum(g["cmsa"] for g in r["groups"]) for k,r in reports.items()}
criteria={"degraded_miou_gain_at_least_0_02":deltas["target15_b"]["target_miou"]>=.02,
          "degraded_cmsa_non_decrease":success["adapted_target15_b"]>=success["base_target15_b"],
          "clean_miou_drop_at_most_0_01":deltas["clean"]["target_miou"]>=-.01,
          "clean_cmsa_drop_at_most_one_group":success["adapted_clean"]>=success["base_clean"]-1}
result={"passed":all(criteria.values()),"status":"PASS" if all(criteria.values()) else "FAIL",
        "criteria":criteria,"validation":table,"adapted_minus_base":deltas,"cmsa_success_groups":success,
        "paired_inputs_and_masks_equal":True,"all_conditions_no_enhancer":True,
        "checkpoint":rows["adapted_clean"][0]["provenance"]["adaptation"],"training_attempts":1}
(log/"validation_gate.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="checkpoint"},indent=2))

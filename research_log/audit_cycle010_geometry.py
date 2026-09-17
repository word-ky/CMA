"""Post-run coordinate/coverage audit on saved predictions; no inference."""
import json
import sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);folder=root/'outputs/cycle010/val_mgr_target15_b'
records=[json.loads(l) for l in (folder/'memory_predictions.jsonl').read_text().splitlines()]
trials=[]
for row in records:
    for trial,roi in zip(row['trials'],row['provenance']['reobservation_rois']):
        target=np.load(folder/trial['target'])>0
        pred=np.load(folder/trial['prediction'])>0
        ref=np.load(folder/trial['reference'])>0
        x1,y1,x2,y2=roi['roi_xyxy']
        assert target.shape==tuple(roi['original_hw'])==pred.shape==ref.shape
        outside=pred.copy();outside[y1:y2,x1:x2]=False
        assert not outside.any()
        trials.append({'group_id':row['group_id'],'entity_id':trial['entity_id'],
                       'target_coverage_by_roi':float(target[y1:y2,x1:x2].sum()/target.sum()),
                       'ref_coverage_by_roi':float(ref[y1:y2,x1:x2].sum()/ref.sum()),
                       'prediction_pixels':int(pred.sum()),'target_pixels':int(target.sum()),
                       'roi_area_ratio':(x2-x1)*(y2-y1)/target.size})
result={'trials':trials,'original_shape_and_zero_outside_roi_verified':True,
        'targets_fully_covered':sum(x['target_coverage_by_roi']==1 for x in trials),
        'targets_no_roi_overlap':sum(x['target_coverage_by_roi']==0 for x in trials),
        'mean_target_coverage':float(np.mean([x['target_coverage_by_roi'] for x in trials])),
        'empty_predictions':sum(x['prediction_pixels']==0 for x in trials)}
(root/'research_log/cycle010/geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))

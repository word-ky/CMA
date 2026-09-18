import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dual_scale_features import map_local_features


def test_local_padding_removed_and_global_roi_gate():
    local=torch.full((1,2,4,4),99.)
    local[:,:,:2,:]=3
    mapped,gate,r=map_local_features(local,(32,64),(16,0,48,16),(32,64),(32,64),64)
    assert r['global_grid_roi_xyxy']==[1,0,3,1]
    assert r['valid_local_grid_hw']==[2,4]
    assert gate.sum()==2
    torch.testing.assert_close(mapped[:,:,0,1:3],torch.full((1,2,2),3.))
    assert not (mapped*(1-gate)).any()


def test_border_roi_and_separate_identity_tensors():
    a=torch.ones(1,2,4,4);b=2*a
    ma,ga,_=map_local_features(a,(64,64),(0,0,16,16),(32,32),(64,64),64)
    mb,gb,_=map_local_features(b,(64,64),(16,16,32,32),(32,32),(64,64),64)
    assert ga.sum()==gb.sum()==4
    before=mb.clone();ma.zero_()
    assert torch.equal(mb,before)

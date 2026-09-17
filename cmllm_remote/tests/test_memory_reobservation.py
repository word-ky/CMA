import inspect
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from memory_reobservation import memory_roi, crop_to_roi, bbox_in_roi, restore_mask


def test_forward_inverse_geometry():
    bbox=[20,20,60,60]
    roi=memory_roi(bbox,(100,100))
    assert roi==(15,15,65,65)
    assert bbox_in_roi(bbox,roi)==[5,5,45,45]
    mask=np.zeros((100,100),dtype=np.uint8); mask[20:60,20:60]=1
    cropped=crop_to_roi(mask,roi)
    assert cropped.shape==(50,50)
    np.testing.assert_array_equal(cropped[5:45,5:45],1)
    np.testing.assert_array_equal(restore_mask(cropped,roi,mask.shape),mask)


def test_border_clipping_and_rectangular_aspect():
    roi=memory_roi([0,0,40,20],(30,50))
    assert roi==(0,0,45,23)
    assert bbox_in_roi([0,0,40,20],roi)==[0,0,40,20]
    mapped=restore_mask(np.ones((23,45),dtype=bool),roi,(30,50))
    assert mapped.sum()==23*45 and not mapped[23:].any() and not mapped[:,45:].any()
    assert memory_roi([40,20,50,30],(30,50))==(38,18,50,30)


def test_roi_has_only_memory_and_image_size_inputs():
    assert list(inspect.signature(memory_roi).parameters)==['bbox_xyxy','image_hw']
    image=np.arange(30*50*3).reshape(30,50,3)
    roi=memory_roi([0,0,40,20],image.shape[:2])
    np.testing.assert_array_equal(crop_to_roi(image,roi),image[:23,:45])


def test_build_item_crop_preserves_export_geometry_and_ignores_helmet_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import ast
    import cv2
    import torch
    import torch.nn.functional as F
    from counterfactual_export import condition_image, group_seed
    path=Path(__file__).resolve().parents[1]/'scripts/eval_mr_ref_counterfactual_v0.py'
    tree=ast.parse(path.read_text())
    # Load the real preprocessing functions without importing the GPU model.
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in ('main','predict_item')]
    ns=dict(np=np,cv2=cv2,torch=torch,F=F,condition_image=condition_image,group_seed=group_seed,
            memory_roi=memory_roi,crop_to_roi=crop_to_roi,bbox_in_roi=bbox_in_roi)
    exec(compile(tree,str(path),'exec'),ns)
    ns['build_multiround_conversation']=lambda a,b:'fixed prompt'
    class Clip:
        def preprocess(self,image,return_tensors):
            return {'pixel_values':torch.from_numpy(cv2.resize(image,(8,8))).permute(2,0,1)[None]}
    class Resize:
        def apply_image(self,image):
            h,w=image.shape[:2]; scale=64/max(h,w)
            return cv2.resize(image,(int(w*scale+.5),int(h*scale+.5)))
    image=np.full((30,50,3),120,dtype=np.uint8)
    ref=np.zeros((30,50),dtype=np.uint8);ref[:20,:40]=255
    target=np.zeros_like(ref);target[:5,:5]=255
    for name,array in [('image',image),('ref',ref),('target',target)]:cv2.imwrite(f'{name}.png',array)
    cf={'pair_ids':['a'],'image_path':'image.png','counterfactual_id':'g','same_round2_query':'helmet'}
    pairs={'a':{'miner_bbox_xyxy':[0,0,40,20],'miner_mask_path':'ref.png','helmet_mask_path':'target.png'}}
    receipt={}
    result=ns['build_item'](cf,pairs,Clip(),Resize(),64,'v1_multiround',reobservation_receipt=receipt)
    item,rgb,targets,refs,indices=result
    assert receipt['roi_xyxy']==[0,0,45,23] and item[6]==(33,64)
    assert item[1].shape==(3,64,64) and item[4].shape==(2,23,45)
    assert not item[4][1].any() and item[9][1].shape==(23,45)
    np.testing.assert_allclose(item[10][1],[0,0,40/45,20/23])
    assert targets.shape==refs.shape==(1,30,50) and indices==[1]
    cv2.imwrite('target.png',255-target)
    second=ns['build_item'](cf,pairs,Clip(),Resize(),64,'v1_multiround',reobservation_receipt={})[0]
    for i in (1,2,4,5,9,10,11,12,13):torch.testing.assert_close(item[i],second[i])

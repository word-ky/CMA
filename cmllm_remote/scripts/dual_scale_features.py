"""Frozen SAM local evidence, mapped to the unchanged padded global grid."""
import math
import torch
import torch.nn.functional as F
from memory_reobservation import memory_roi, crop_to_roi


def map_local_features(local, local_resize_hw, roi, original_hw, global_resize_hw, image_size):
    fh, fw = local.shape[-2:]
    h, w = original_hw
    rh, rw = global_resize_hw
    x1,y1,x2,y2 = roi
    gx1,gy1 = math.floor(x1*rw/w/image_size*fw), math.floor(y1*rh/h/image_size*fh)
    gx2,gy2 = math.ceil(x2*rw/w/image_size*fw), math.ceil(y2*rh/h/image_size*fh)
    # Exclude bottom/right padding from local encoder features before resizing.
    lh = math.ceil(local_resize_hw[0]/image_size*fh)
    lw = math.ceil(local_resize_hw[1]/image_size*fw)
    mapped = local.new_zeros(local.shape)
    mapped[:,:,gy1:gy2,gx1:gx2] = F.interpolate(
        local[:,:,:lh,:lw].float(),size=(gy2-gy1,gx2-gx1),mode='bilinear',align_corners=False).to(local.dtype)
    gate = local.new_zeros((1,1,fh,fw));gate[:,:,gy1:gy2,gx1:gx2]=1
    receipt={'local_shape':list(local.shape),'valid_local_grid_hw':[lh,lw],
             'roi_xyxy':list(roi),'global_grid_roi_xyxy':[gx1,gy1,gx2,gy2],
             'mapped_shape':list(mapped.shape),'gate_shape':list(gate.shape)}
    return mapped,gate,receipt


@torch.no_grad()
def encode_local_memory(model,image_rgb,bboxes,transform,image_size,global_resize_hw,preprocess_sam,dtype):
    mapped_list=[];gates=[];receipts=[]
    for bbox in bboxes:
        roi=memory_roi(bbox,image_rgb.shape[:2])
        resized=transform.apply_image(crop_to_roi(image_rgb,roi))
        pixels=preprocess_sam(torch.from_numpy(resized).permute(2,0,1).contiguous(),image_size)
        local=model.get_visual_embs(pixels.unsqueeze(0).cuda().to(dtype))
        mapped,gate,receipt=map_local_features(local,resized.shape[:2],roi,image_rgb.shape[:2],global_resize_hw,image_size)
        mapped_list.append(mapped[0]);gates.append(gate[0]);receipts.append(receipt)
    return {'mapped':torch.stack(mapped_list),'gate':torch.stack(gates),'receipts':receipts}

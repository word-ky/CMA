"""Actual installed SAM joint-box/text and resize contract; CPU only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
root=Path(sys.argv[1]); lisa=root/'code/cmllm/third_party/LISA'
sys.path.insert(0,str(lisa));sys.path.insert(0,str(root/'code/cmllm/scripts'))
from model.segment_anything.modeling.prompt_encoder import PromptEncoder
from model.segment_anything.utils.transforms import ResizeLongestSide
from memory_spatial_prompt import spatial_memory_boxes
resize=ResizeLongestSide(1024)
boxes=spatial_memory_boxes([[-1,0,200,100],[20,30,40,60]],(100,200),resize,True)
torch.testing.assert_close(boxes,torch.tensor([[0,0,1024,512],[0,0,1024,512],[102.4,153.6,204.8,307.2],[102.4,153.6,204.8,307.2]]))
encoder=PromptEncoder(8,(4,4),(1024,1024),4)
text=torch.randn(4,2,8)
with torch.no_grad():
    sparse,dense=encoder(points=None,boxes=boxes,masks=None,text_embeds=text)
assert sparse.shape==(4,4,8) and dense.shape==(4,8,4,4)
torch.testing.assert_close(sparse[:,2:],text)
torch.testing.assert_close(sparse[:,:2],encoder._embed_boxes(boxes))
result={'boxes_plus_text_supported_without_modification':True,'sparse_shape':list(sparse.shape),
        'existing_text_tokens_preserved':True,'resized_boxes':boxes.tolist(),
        'ResizeLongestSide_and_border_numeric_test':'PASS',
        'prompt_encoder_sha256':hashlib.sha256((lisa/'model/segment_anything/modeling/prompt_encoder.py').read_bytes()).hexdigest()}
folder=root/'research_log/cycle011';folder.mkdir(exist_ok=True)
(folder/'prompt_encoder_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

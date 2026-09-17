"""Compare old/new diagonal preprocessing with real CLIP processor, no GPU run."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
scripts = root / "code/cmllm/scripts"
sys.path.insert(0, str(scripts))
os.environ["LISA_ROOT"] = str(root / "code/cmllm/third_party/LISA")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            equal(first, second)
    else:
        assert a == b


old_path = root / "research_log/cycle004/evaluator_cycle003.py"
new_path = scripts / "eval_mr_ref_counterfactual_v0.py"
old, new = load("old_cf", old_path), load("new_cf", new_path)
new.conversation_lib.default_conversation = new.conversation_lib.conv_templates["llava_v1"]
clip = new.CLIPImageProcessor.from_pretrained(str(root / "shared/models/clip-vit-large-patch14"))
transform = new.ResizeLongestSide(1024)
data = root / "shared/data/cycle003"
group = new.read_jsonl(data / "counterfactual_holdout_first30_restored.jsonl")[0]
pairs = {r["pair_id"]: r for r in new.read_jsonl(data / "helmet_miner_pairs_restored.jsonl")}
for condition in ("clean", "target15_b"):
    before = old.build_item(group, pairs, clip, transform, 1024, "v1_multiround", condition=condition, seed=0)
    after = new.build_item(group, pairs, clip, transform, 1024, "v1_multiround", main_condition=condition, ref_condition=condition, seed=0)
    equal(before, after)
report = {"group_id": group["counterfactual_id"], "conditions": ["CC", "DD"],
          "all_input_tensors_and_metadata_exact_equal": True, "processor": "actual cached CLIPImageProcessor",
          "model_inference_run": False, "old_source_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
          "new_source_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest()}
(root / "research_log/cycle004/tensor_compatibility.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

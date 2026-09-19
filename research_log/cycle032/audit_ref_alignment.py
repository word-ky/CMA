"""CPU token/index and source-construction audit; no model/tokenizer/weights loaded."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LISA = ROOT / 'cmllm_remote/third_party/LISA/model/LISA.py'
LLAVA = ROOT / 'cmllm_remote/third_party/LISA/model/llava/model/language_model/llava_llama.py'
EVAL = ROOT / 'cmllm_remote/scripts/eval_mr_ref_counterfactual_v0.py'
IMAGE, REF, SEG = -200, 32001, 32000

def tree(path):
    return ast.parse(path.read_text(encoding='utf-8'))

def function(path, name):
    return next(n for n in ast.walk(tree(path)) if isinstance(n, ast.FunctionDef) and n.name == name)

def compile_function(node):
    env = {'torch': torch, 'IMAGE_TOKEN_INDEX': IMAGE}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])), '<source-extract>', 'exec'), env)
    return env[node.name]

class CPUAllocation(ast.NodeTransformer):
    # The source manually allocates zero masks with .cuda(); the audit changes only device.
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            node.func.attr = 'cpu'
        return node

def manual_mask(ids, variable):
    nodes = sorted((n for n in ast.walk(function(LISA, 'model_forward'))
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets)), key=lambda n:n.lineno)
    # First three assignments are shifted comparison, trailing zero, and leading image offset.
    nodes = nodes[:3]
    assert len(nodes) == 3
    env = {'torch': torch, 'input_ids': ids,
           'self': SimpleNamespace(ref_token_idx=REF, seg_token_idx=SEG)}
    module = CPUAllocation().visit(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(ast.fix_missing_locations(module), '<source-manual-mask>', 'exec'), env)
    return env[variable], [n.lineno for n in nodes]

input_helper = compile_function(function(LLAVA, '_build_expanded_token_mask'))
aux_helper = compile_function(function(LISA, '_build_expanded_token_mask'))
aux_source = ast.unparse(function(LISA, 'get_ref_token_embeddings'))
assert 'shifted=False' in aux_source

def positions(mask):
    return torch.where(mask[0])[0].tolist()

def case(name, raw, patches):
    ids = torch.tensor([raw], dtype=torch.long)
    expanded_len = len(raw) + raw.count(IMAGE) * (patches - 1)
    tower = SimpleNamespace(num_patches=patches)
    obj = SimpleNamespace(get_vision_tower=lambda:tower,
                          get_model=lambda:SimpleNamespace(get_vision_tower=lambda:tower))
    actual = positions(input_helper(obj, ids, REF, expanded_len))
    auxiliary = positions(aux_helper(obj, ids, REF, expanded_len, shifted=False))
    refmask, ref_lines = manual_mask(ids, 'ref_token_mask')
    segmask, seg_lines = manual_mask(ids, 'seg_token_mask')
    assert actual == auxiliary
    seg_actual = positions(input_helper(obj, ids, SEG, expanded_len))
    return dict(name=name, raw_tokens=raw, image_patches=patches, expanded_length=expanded_len,
                injected_REF=actual, manual_output_REF=positions(refmask), auxiliary_REF=auxiliary,
                true_SEG=seg_actual, manual_SEG=positions(segmask), manual_mask_length=refmask.shape[1],
                output_REF_exactly_one_before=positions(refmask)==[i-1 for i in actual],
                source_manual_REF_lines=ref_lines, source_manual_SEG_lines=seg_lines)

cases = [case('single_front_image_256', [1,IMAGE,11,SEG,12,REF,13,SEG,2],256),
         case('different_patch_count_196', [1,IMAGE,11,SEG,12,REF,13,SEG,2],196),
         case('image_after_REF', [1,11,REF,12,IMAGE,13,SEG,2],256),
         case('two_images_before_REF', [1,IMAGE,11,IMAGE,REF,13,SEG,2],256)]
assert cases[0]['injected_REF']==[260]
assert cases[0]['manual_output_REF']==[259]
assert cases[0]['manual_SEG']==[257,261]
assert cases[0]['true_SEG']==[258,262]
assert all(not c['output_REF_exactly_one_before'] for c in cases[1:])

# Execute the real conversation-builder AST with a message recorder. This records the
# exact roles/messages before template serialization, without pretending to tokenize.
class Recorder:
    roles = ('USER','ASSISTANT')
    def __init__(self): self.messages=[]
    def copy(self): return Recorder()
    def append_message(self, role, text): self.messages.append([role,text])
    def get_prompt(self): return self.messages

builder=function(EVAL,'build_multiround_conversation')
env={'conversation_lib':SimpleNamespace(default_conversation=Recorder()),'DEFAULT_IMAGE_TOKEN':'<image>'}
exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(builder)],type_ignores=[])), '<builder-source>', 'exec'),env)
construction=next(n for n in ast.walk(function(EVAL,'build_item')) if isinstance(n,ast.Assign)
                  and isinstance(n.value,ast.ListComp) and isinstance(n.value.elt,ast.Call)
                  and isinstance(n.value.elt.func,ast.Name) and n.value.elt.func.id=='build_multiround_conversation')
groups=[json.loads(s) for s in (ROOT/'research_log/cycle025/selected_source_groups.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
records=[]
for cf in groups:
    scope=dict(env,cf=cf,pair_ids=cf['pair_ids'])
    conversations=eval(compile(ast.Expression(body=construction.value),'<conversation-list-source>','eval'),scope)
    assert len(conversations)==2 and conversations[0]==conversations[1]
    records.append({'group_id':cf['counterfactual_id'],'identity_rows':2,
                    'messages_equal':True,'messages_sha256':hashlib.sha256(json.dumps(conversations[0],ensure_ascii=False).encode()).hexdigest()})
assert len(records)==50
receipt={'status':'PASS','kind':'zero-model CPU indexing/source-construction audit',
         'cases':cases,'prefix_status':'PREFIX_IDENTICAL_BEFORE_REF',
         'prefix_evidence':'Actual builder/list-comprehension AST executed with message recorder; all 50 frozen groups have identical A/B messages. Same deterministic tokenizer/template therefore yields identical raw token prefixes. No actual tokenizer IDs claimed.',
         'conversation_builder':ast.unparse(builder),'conversation_construction':ast.unparse(construction),
         'representative_messages':conversations[0], 'prefix_groups':records,
         'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LISA,LLAVA,EVAL]},
         'device_only_substitution':'.cuda() -> .cpu() in the three extracted manual-mask assignments',
         'model_calls':0,'tokenizer_loaded':False,'scorer_calls':0}
(OUT/'alignment_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','prefix_status':receipt['prefix_status'],'groups':len(records),'cases':cases},indent=2))

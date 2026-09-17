import ast
from pathlib import Path
import torch
import torch.nn.functional as F


def functions():
    path=Path(__file__).resolve().parents[1]/"scripts/train_w15_degradation_counterfactual.py"
    tree=ast.parse(path.read_text()); ns={"torch":torch,"F":F}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("consistency_loss","surface_for","configure_trainable")
           or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SURFACES" for t in n.targets)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),ns)
    return ns


def test_consistency_teacher_is_detached():
    ns=functions(); c=torch.tensor([2.,-1.],requires_grad=True); d=torch.tensor([0.,0.],requires_grad=True)
    loss=ns["consistency_loss"](d,c); loss.backward()
    assert c.grad is None
    torch.testing.assert_close(d.grad,(d.detach().sigmoid()-c.detach().sigmoid())/2)


def test_only_declared_surfaces_unfrozen():
    ns=functions(); model=torch.nn.Module(); model.model=torch.nn.Module()
    model.model.visual_model=torch.nn.Module()
    model.model.visual_model.mask_decoder=torch.nn.Linear(2,2)
    model.model.visual_model.image_encoder=torch.nn.Linear(2,2)
    model.model.ref_embedding_scale=torch.nn.Parameter(torch.ones(1))
    model.model.ref_input_fcs=torch.nn.Linear(2,2)
    model.model.embed_tokens=torch.nn.Embedding(3,2)
    rows=ns["configure_trainable"](model)
    assert {r['name'] for r in rows}=={'model.visual_model.mask_decoder.weight','model.visual_model.mask_decoder.bias','model.ref_embedding_scale','model.ref_input_fcs.weight','model.ref_input_fcs.bias'}
    assert not any(p.requires_grad for p in model.model.visual_model.image_encoder.parameters())
    assert not model.model.embed_tokens.weight.requires_grad


def test_sequential_backward_equals_declared_paired_objective():
    ns=functions()
    path=Path(__file__).resolve().parents[1]/"scripts/train_w15_degradation_counterfactual.py"
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="paired_backward")
    parameter=torch.tensor([.2,-.4],requires_grad=True)
    def fake_forward(runner,group,condition,args):
        logits=parameter*(1 if condition=="clean" else 2)
        return logits,logits.square().mean(),logits.sigmoid().mean()
    ns["condition_forward"]=fake_forward
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),ns)
    ns["paired_backward"](None,None,None)
    sequential=parameter.grad.clone(); parameter.grad=None
    c,sc,rc=fake_forward(None,None,"clean",None)
    d,sd,rd=fake_forward(None,None,"target15_b",None)
    (.5*(sc+sd)+.5*(rc+rd)+.25*ns["consistency_loss"](d,c)).backward()
    torch.testing.assert_close(parameter.grad,sequential)

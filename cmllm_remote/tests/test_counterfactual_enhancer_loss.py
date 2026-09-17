import ast
from pathlib import Path

import torch
import torch.nn.functional as F


def test_rank_matches_identity_order_and_backpropagates():
    root=Path(__file__).resolve().parents[1]
    scope={"torch":torch,"F":F}
    for path,name in ((root/"third_party/LISA/model/LISA.py","soft_iou_matrix"),
                      (root/"scripts/train_task_enhancer_v4_counterfactual.py","counterfactual_rank_loss")):
        tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),scope)
    targets=torch.tensor([[[1.,0.]],[[0.,1.]]])
    correct=(targets*16-8).requires_grad_()
    assert scope["counterfactual_rank_loss"](correct,targets).item()==0
    wrong=correct.detach().flip(0).requires_grad_()
    loss=scope["counterfactual_rank_loss"](wrong,targets)
    assert loss.item()>1.04
    loss.backward()
    assert torch.isfinite(wrong.grad).all() and wrong.grad.abs().sum()>0

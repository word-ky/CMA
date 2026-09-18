"""Regression evidence for the observed Cycle012 specification blocker."""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from memory_dual_scale_adapter import MemoryDualScaleResidualAdapter


def test_requested_double_zero_is_exact_identity_but_has_zero_task_gradients():
    torch.manual_seed(20260528)
    adapter=MemoryDualScaleResidualAdapter()
    # Reproduce the historical Cycle012 contract explicitly.
    with torch.no_grad(): adapter.alpha.zero_()
    assert sum(p.numel() for p in adapter.parameters())==12577
    global_features=torch.randn(1,256,4,4)
    local_features=torch.randn_like(global_features)
    gate=torch.ones(1,1,4,4)
    output=adapter(global_features,local_features,gate)
    assert torch.equal(output,global_features)
    loss=(output-torch.randn_like(output)).square().mean()
    loss.backward()
    for parameter in adapter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)==0
    # AdamW decay can change Down, but cannot activate alpha or Up from zero.
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-4)
    optimizer.step()
    assert torch.equal(adapter(global_features,local_features,gate),global_features)
    assert adapter.alpha.item()==0
    assert not adapter.up.weight.any() and not adapter.up.bias.any()


def test_corrected_zero_residual_has_staged_gradient_flow():
    torch.manual_seed(20260528)
    adapter=MemoryDualScaleResidualAdapter(channels=4,bottleneck=2)
    g=torch.randn(1,4,3,3); local=torch.randn_like(g); gate=torch.ones(1,1,3,3)
    target=torch.randn_like(g)
    assert torch.equal(adapter(g,local,gate),g)
    (adapter(g,local,gate)-target).square().mean().backward()
    assert adapter.up.weight.grad.norm()>0
    assert adapter.down.weight.grad.norm()==0 and adapter.alpha.grad==0
    opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-4)
    opt.step();opt.zero_grad()
    (adapter(g,local,gate)-target).square().mean().backward()
    assert adapter.down.weight.grad.norm()>0
    assert torch.isfinite(adapter.alpha.grad)

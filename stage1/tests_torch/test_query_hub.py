"""GPU tests for ObjBias and QueryHub (GeoQuery spec 2.3)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.query_hub import ObjBias, QueryHub  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_objbias_zero_init_and_inf_gives_b_none():
    b = ObjBias().cuda()
    d = torch.tensor([[0.0, 1.0, float("inf")]]).cuda()
    assert torch.equal(b(d), torch.zeros(1, 3).cuda())
    with torch.no_grad():
        b.b_none.fill_(-0.5)
        b.w.fill_(0.1)
    out = b(d)
    assert out[0, 2].item() == -0.5 and out[0, 0].item() > 0 and torch.isfinite(out).all()


def test_queryhub_zero_gates_is_identity():
    hub = QueryHub().cuda()
    obj_ptr = torch.randn(3, 256).cuda()
    tokens = torch.randn(2 * 5184, 256).cuda()
    view_ids = torch.arange(2).cuda().repeat_interleave(5184)
    q, mass = hub(obj_ptr, tokens, None, view_ids)
    assert torch.equal(q, obj_ptr) and mass.shape == (2, 3, 2)
    assert torch.equal(hub.prompt_token(q), torch.zeros(3, 256).cuda())


def test_queryhub_mass_sums_to_one_with_invalid_view():
    hub = QueryHub().cuda()
    tokens = torch.randn(3 * 5184, 256).cuda()
    view_ids = torch.arange(3).cuda().repeat_interleave(5184)
    dist = torch.rand(4, 3 * 5184).cuda() * 2
    dist[:, 2 * 5184:] = float("inf")                       # view 2 entirely invalid
    with torch.no_grad():
        for layer in hub.layers:
            layer.g_cross.fill_(1.0)
        hub.obj_bias.w.fill_(0.3)
        hub.obj_bias.b_none.fill_(-2.0)
    q, mass = hub(torch.randn(4, 256).cuda(), tokens, dist, view_ids)
    assert torch.isfinite(q).all() and torch.allclose(mass.sum(-1), torch.ones(2, 4).cuda(), atol=1e-4)
    assert (mass[:, :, 2] < mass[:, :, 0]).all()           # the b_none view gets less mass


def test_queryhub_single_object_and_sixteen():
    hub = QueryHub().cuda()
    tokens = torch.randn(5184, 256).cuda()
    ids = torch.zeros(5184, dtype=torch.long).cuda()
    for m in (1, 16):
        q, mass = hub(torch.randn(m, 256).cuda(), tokens, torch.zeros(m, 5184).cuda(), ids)
        assert q.shape == (m, 256) and mass.shape == (2, m, 1)


def test_queryhub_ptr_mode_and_gate_init():
    hub = QueryHub(ptr=True, g_cross_init=0.1).cuda()
    assert all(abs(layer.g_cross.item() - 0.1) < 1e-6 and layer.g_self.item() == 0.0 for layer in hub.layers)
    q = torch.randn(3, 256).cuda()
    assert hub.ptr_proj.weight.abs().sum().item() == 0 and torch.equal(hub.ptr_delta(q), torch.zeros(4, 3, 64).cuda())
    assert not hasattr(QueryHub(), "ptr_proj")

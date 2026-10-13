import math

import torch

from geomem.embed import GeoEmbed, fourier, grid_tensors


def test_fourier_shape_and_values():
    p = torch.zeros(5, 3)
    f = fourier(p)
    assert f.shape == (5, 51)
    assert torch.all(f[:, :3] == 0) and torch.all(f[:, 3:27] == 0) and torch.all(f[:, 27:] == 1)   # sin 0, cos 1
    p1 = torch.tensor([[0.5, 0.0, 0.0]])
    assert abs(fourier(p1)[0, 3] - math.sin(math.pi * 0.5)) < 1e-6                                 # k=0: sin(pi x)


def test_geoembed_zero_init_outputs_zero():
    m = GeoEmbed()
    p = torch.randn(10, 3); valid = torch.ones(10, dtype=torch.bool)
    k, q = m(p, valid)
    assert k.shape == (10, 64) and q.shape == (10, 256)
    assert torch.all(k == 0) and torch.all(q == 0)


def test_embed_all_none_cells_is_finite_and_trainable():
    m = GeoEmbed()
    p = torch.zeros(7, 3); valid = torch.zeros(7, dtype=torch.bool)
    k, q = m(p, valid)
    assert torch.isfinite(k).all() and torch.isfinite(q).all()
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    loss = (m(p, valid)[0] - 1.0).pow(2).mean() + (m(p, valid)[1] - 1.0).pow(2).mean()
    loss.backward(); opt.step()
    k2, q2 = m(p, valid)
    assert (k2 != 0).any() and (q2 != 0).any()


def test_embed_param_count_under_200k():
    assert GeoEmbed().num_params() < 200_000


def test_grid_tensors_row_major():
    import numpy as np
    p = np.zeros((72, 72, 3), np.float32); p[1, 0] = [9, 9, 9]
    valid = np.zeros((72, 72), bool); valid[1, 0] = True
    pt, vt = grid_tensors({"p": p, "valid": valid, "scale": 1.0}, "cpu")
    assert pt.shape == (5184, 3) and vt.shape == (5184,)
    assert torch.all(pt[72] == 9) and vt[72] and vt.sum() == 1       # row 1, col 0 -> flat index 72


def test_e_none_receives_gradient_with_nonzero_heads():
    """Review finding: e_none stayed exactly 0 in every trained checkpoint.  With non-zero heads and a
    batch that has invalid cells, e_none must receive a non-zero gradient."""
    torch.manual_seed(0)
    m = GeoEmbed()
    with torch.no_grad():
        m.key_head.weight.normal_(); m.query_head.weight.normal_()
    p = torch.randn(100, 3); valid = torch.ones(100, dtype=torch.bool); valid[:30] = False
    k, q = m(p, valid)
    (k.sum() + q.sum()).backward()
    assert m.e_none.grad is not None and m.e_none.grad.abs().sum() > 0

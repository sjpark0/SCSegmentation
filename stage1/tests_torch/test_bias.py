import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.bias import N_RBF, PTR_TOKENS, RBF_MAX, GeoBias, rbf  # noqa: E402


def test_rbf_peaks_at_centres():
    f0 = rbf(torch.tensor([0.0]))
    f2 = rbf(torch.tensor([RBF_MAX]))
    assert f0.shape == (1, N_RBF) and abs(f0[0, 0] - 1) < 1e-6 and abs(f2[0, -1] - 1) < 1e-6
    assert f0[0, -1] < 1e-6                                   # far centre contributes nothing


def _grids(n, shift=0.0):
    p = torch.stack([torch.arange(n, dtype=torch.float32) * 0.1 + shift, torch.zeros(n), torch.zeros(n)], -1)
    return p, torch.ones(n, dtype=torch.bool)


def test_bias_zero_init_is_all_zero():
    m = GeoBias()
    pq, vq = _grids(10)
    out = m(pq, vq, [_grids(7), _grids(7)])
    assert out.shape == (1, 1, 10, 2 * 7 + 2 * PTR_TOKENS) and torch.all(out == 0)


def test_bias_all_invalid_uses_b_none():
    m = GeoBias()
    with torch.no_grad():
        m.b_none.fill_(-3.0)
    pq, _ = _grids(10)
    pm, _ = _grids(7)
    out = m(pq, torch.zeros(10, dtype=torch.bool), [(pm, torch.zeros(7, dtype=torch.bool))])
    assert torch.isfinite(out).all()
    assert torch.all(out[0, 0, :, :7] == -3.0) and torch.all(out[0, 0, :, 7:] == 0)


def test_bias_blocks_use_each_views_grid():
    m = GeoBias()
    with torch.no_grad():
        m.w.zero_(); m.w[0] = 1.0                              # only the d=0 centre
    pq, vq = _grids(5)
    out = m(pq, vq, [_grids(5), _grids(5, shift=1.0)])
    block_a, block_b = out[0, 0, :, :5], out[0, 0, :, 5:10]
    assert torch.allclose(block_a.diagonal(), torch.ones(5), atol=1e-5)
    assert torch.all(block_b.diagonal() < 1e-6)                # same cells, but view B's grid is 1.0 away


def test_bias_param_count():
    assert GeoBias().num_params() == N_RBF + 1

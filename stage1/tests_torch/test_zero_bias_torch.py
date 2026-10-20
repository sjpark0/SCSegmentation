"""Ring diagnosis D3: `zero_geo_bias` makes a trained GeoBias add exactly 0 (container, CPU)."""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.bias import PTR_TOKENS, GeoBias  # noqa: E402
from methods.geomem.run import zero_geo_bias  # noqa: E402


def _grids(n):
    p = torch.stack([torch.arange(n, dtype=torch.float32) * 0.1, torch.zeros(n), torch.zeros(n)], -1)
    return p, torch.ones(n, dtype=torch.bool)


def test_zero_geo_bias_zeroes_a_trained_bias():
    torch.manual_seed(0)
    m = GeoBias()
    with torch.no_grad():
        m.w.normal_(); m.b_none.fill_(-0.7)
    pq, vq = _grids(6)
    invalid = (_grids(5)[0], torch.tensor([True, False, True, False, True]))
    assert torch.any(m(pq, vq, [_grids(5), invalid]) != 0)
    before = max(float(m.w.detach().abs().max()), 0.7)
    got = zero_geo_bias(SimpleNamespace(use_bias=True, geo_bias=m))
    assert got == pytest.approx(before)
    out = m(pq, vq, [_grids(5), invalid])
    assert out.shape == (1, 1, 6, 10 + 2 * PTR_TOKENS) and torch.all(out == 0)


def test_zero_geo_bias_refuses_an_arm_without_the_bias():
    with pytest.raises(AssertionError):
        zero_geo_bias(SimpleNamespace(use_bias=False, geo_bias=None))

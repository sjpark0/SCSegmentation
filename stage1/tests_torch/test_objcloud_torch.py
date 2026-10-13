"""GPU tests for token-to-cloud distances."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.objcloud import cloud_distances  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_cloud_distances_matches_numpy_and_marks_invalid():
    g = torch.Generator().manual_seed(0)
    p = torch.randn(2, 5184, 3, generator=g)
    valid = torch.ones(2, 5184, dtype=torch.bool)
    valid[1, :10] = False
    clouds = [np.random.default_rng(1).normal(size=(50, 3)).astype(np.float32)]
    d = cloud_distances(p.cuda(), valid.cuda(), clouds)
    assert d.shape == (1, 2 * 5184) and d.dtype == torch.float32
    ref = np.linalg.norm(p[0, 7].numpy()[None] - clouds[0], axis=1).min()
    assert abs(d[0, 7].item() - ref) < 1e-4
    assert torch.isinf(d[0, 5184:5194]).all() and torch.isfinite(d[0, 5194:]).all()


def test_cloud_distances_empty_cloud_is_inf():
    p = torch.zeros(1, 5184, 3).cuda()
    valid = torch.ones(1, 5184, dtype=torch.bool).cuda()
    d = cloud_distances(p, valid, [np.zeros((0, 3), np.float32), np.zeros((1, 3), np.float32)])
    assert torch.isinf(d[0]).all() and (d[1] == 0).all()

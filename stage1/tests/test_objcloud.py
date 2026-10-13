import numpy as np

from geomem.objcloud import mask_coverage, object_clouds


def test_mask_coverage_half_plane():
    m = np.zeros((1, 144, 144), bool)
    m[0, :, :72] = True
    cov = mask_coverage(m, grid=72)
    assert cov.shape == (1, 72, 72) and np.allclose(cov[0, :, :36], 1.0) and np.allclose(cov[0, :, 36:], 0.0)


def test_object_clouds_threshold_valid_and_cap():
    p = np.random.default_rng(0).normal(size=(5184, 3)).astype(np.float32)
    valid = np.ones(5184, bool)
    valid[:100] = False
    m = np.zeros((2, 1008, 1008), bool)
    m[0, :504] = True                                    # top half: cells 0..2591
    clouds = object_clouds(m, p, valid, max_points=256, seed=0)
    assert clouds[0].shape == (256, 3) and clouds[1].shape == (0, 3)
    rows = {tuple(r) for r in clouds[0]}
    assert rows <= {tuple(r) for r in p[100:2592]}        # only valid cells of the top half


def test_object_clouds_small_object_keeps_all_points():
    p = np.arange(5184 * 3, dtype=np.float32).reshape(5184, 3)
    m = np.zeros((1, 1008, 1008), bool)
    m[0, :14, :28] = True                                 # exactly cells 0 and 1
    (c,) = object_clouds(m, p, np.ones(5184, bool))
    assert np.array_equal(np.sort(c, axis=0), p[:2])

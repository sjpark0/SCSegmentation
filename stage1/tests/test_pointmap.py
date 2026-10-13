import json
import os

import numpy as np

from geomem.pointmap import VALID_RATIO, local_points, pool_grid, scene_scale, to_reference, view_grid

K = np.array([[100.0, 0, 32.0], [0, 100.0, 24.0], [0, 0, 1.0]])


def test_local_points_pinhole():
    depth = np.full((48, 64), 2.0, np.float32)
    p = local_points(depth, K)
    assert p.shape == (48, 64, 3)
    assert np.allclose(p[24, 32], [0, 0, 2])
    assert np.allclose(p[24, 42], [0.2, 0, 2])            # (42-32)/100 * 2
    depth[0, 0] = 0
    assert np.all(local_points(depth, K)[0, 0] == 0)


def test_to_reference_identity_when_same_pose():
    p = np.random.default_rng(0).normal(size=(4, 5, 3))
    c2w = np.eye(4); c2w[:3, 3] = [1, 2, 3]
    assert np.allclose(to_reference(p, c2w, c2w), p)


def test_to_reference_translation():
    p = np.zeros((1, 1, 3))
    c2w_v = np.eye(4); c2w_v[:3, 3] = [1, 0, 0]          # view v sits at x=1 in world
    c2w_ref = np.eye(4)                                   # ref at the origin
    assert np.allclose(to_reference(p, c2w_v, c2w_ref)[0, 0], [1, 0, 0])


def test_scene_scale_median():
    p = np.zeros((2, 2, 3)); p[..., 2] = [[1, 2], [3, 100]]
    valid = np.array([[True, True], [True, False]])
    assert scene_scale(p, valid) == 2.0


def test_pool_grid_fronto_plane():
    depth = np.full((378, 672), 2.0, np.float32)
    p = local_points(depth, np.array([[600.0, 0, 335.5], [0, 600.0, 188.5], [0, 0, 1]]))
    g, valid = pool_grid(p, depth > 0)
    assert g.shape == (72, 72, 3) and valid.all()
    assert np.allclose(g[..., 2], 2.0, atol=1e-4)
    assert np.all(np.diff(g[36, :, 0]) > 0)                # x grows along columns


def test_pool_all_invalid_grid():
    p = np.random.default_rng(0).normal(size=(100, 200, 3))
    g, valid = pool_grid(p, np.zeros((100, 200), bool))
    assert valid.sum() == 0 and np.all(g == 0) and np.isfinite(g).all()


def test_pool_valid_ratio_threshold():
    h, w = 72 * 10, 72 * 10                                 # each cell is a 10x10 block
    p = np.ones((h, w, 3))
    valid = np.zeros((h, w), bool)
    valid[0:10, 0:2] = True                                 # cell (0,0): 20 % valid
    valid[0:10, 10:13] = True                               # cell (0,1): 30 % valid
    g, v = pool_grid(p, valid)
    assert VALID_RATIO == 0.25 and not v[0, 0] and v[0, 1]
    assert np.allclose(g[0, 1], 1.0)                        # average over valid points only


def test_view_grid_from_cache(tmp_path):
    cache = str(tmp_path / "pi3"); os.makedirs(cache)
    depth = np.full((48, 64), 2.0, np.float32)
    for cam, tx in (("ref", 0.0), ("v", 0.5)):
        c2w = np.eye(4); c2w[0, 3] = tx
        np.savez(os.path.join(cache, f"{cam}.npz"), depth=depth, conf=np.ones_like(depth), K=K, c2w=c2w)
    json.dump({"cams": ["ref", "v"], "work_hw": [48, 64]}, open(os.path.join(cache, "MANIFEST.json"), "w"))
    g = view_grid(cache, "v", "ref")
    assert g["p"].shape == (72, 72, 3) and g["valid"].all() and g["p"].dtype == np.float32
    assert abs(g["scale"] - 2.0) < 0.05                     # median distance of the plane from the ref origin ~ depth
    assert abs(g["p"][36, 36, 0] - 0.5 / g["scale"]) < 0.02 # v's points sit 0.5 to the right in the ref frame, normalised

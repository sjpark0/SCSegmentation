import numpy as np

from mvseg2d.geo import warp
from mvseg2d.geo.cameras import Pinhole, Pose, from_c2w, scale


def test_from_c2w_identity_adds_half_pixel():
    K = np.array([[100.0, 0, 32.0], [0, 100.0, 24.0], [0, 0, 1]])
    cam, pose = from_c2w(K, np.eye(4))
    assert isinstance(cam, Pinhole) and isinstance(pose, Pose)
    assert cam.fx == 100.0 and cam.cx == 32.5 and cam.cy == 24.5 and not cam.has_distortion
    assert np.allclose(pose.R, np.eye(3)) and np.allclose(pose.tvec, 0)


def test_from_c2w_inverts_pose():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    _, pose = from_c2w(np.eye(3), c2w)
    assert np.allclose(pose.tvec, [-1.0, -2.0, -3.0])      # X_cam = R X_world + t with camera at (1,2,3)


def test_scale_intrinsics():
    c = scale(Pinhole(100.0, 100.0, 32.5, 24.5), 2.0, 4.0)
    assert (c.fx, c.fy, c.cx, c.cy) == (200.0, 400.0, 65.0, 98.0)


def test_backward_warp_fronto_parallel_plane_shifts_mask():
    h, w, f, depth, baseline = 60, 80, 100.0, 5.0, 0.5
    cam = Pinhole(f, f, w / 2 + 0.5, h / 2 + 0.5)
    A = Pose(np.eye(3), np.zeros(3))
    B = Pose(np.eye(3), np.array([-baseline, 0.0, 0.0]))          # camera B at world x = +baseline
    d = np.full((h, w), depth, np.float32)
    ref_mask = np.zeros((h, w), bool)
    ref_mask[20:30, 30:40] = True
    got = warp.backward_warp(ref_mask, cam, A, d, cam, B, d)
    shift = f * baseline / depth                                   # 10 px, to the left in B
    expect = np.zeros((h, w), bool)
    expect[20:30, 30 - int(shift):40 - int(shift)] = True
    inter = (got & expect).sum()
    assert inter / float((got | expect).sum()) > 0.9


def test_target_size_multiple_of_patch_under_limit():
    from methods.pi3_reproj.geometry import target_size
    w, h = target_size(1920, 1080, 255_000)
    assert w % 14 == 0 and h % 14 == 0 and w * h <= 255_000 * 1.05 and abs(w / h - 1920 / 1080) < 0.05


def test_reproject_synthetic_scene_writes_shifted_masks(fake_data, tmp_path):
    """Stage B of pi3_reproj on a hand-made cache: a fronto-parallel plane seen by two translated cameras."""
    import json
    from methods import common
    from methods.pi3_reproj.run import load_geometry, reproject
    sc = common.load_scene(fake_data, "S1")           # 40x60 images, ref v2, objects 1 (rows 5:15, cols 5:25) and 2
    h, w = sc.hw
    cache = str(tmp_path / "pi3")
    cache_dir = cache
    import os
    os.makedirs(cache)
    f, depth = 50.0, 5.0
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    for i, cam in enumerate(sc.cams):
        c2w = np.eye(4)
        c2w[0, 3] = (i - 2) * 0.5                                   # cameras along x, v2 at the origin
        np.savez(os.path.join(cache, f"{cam}.npz"), depth=np.full((h, w), depth, np.float32),
                 conf=np.ones((h, w), np.float32), K=K, c2w=c2w)
    json.dump({"model": "fake", "pixel_limit": h * w, "work_hw": [h, w], "orig_hw": [h, w],
               "cams": sc.cams, "ref_cam": sc.ref_cam, "conf_min": 0.1, "conf_frac": {}},
              open(os.path.join(cache, "MANIFEST.json"), "w"))
    geo, manifest = load_geometry(cache)
    out = {}
    n = reproject(sc, geo, manifest, lambda cam, o, m: out.__setitem__((cam, o), m))
    assert n == 8 and set(c for c, _ in out) == {"v0", "v1", "v3", "v4"}
    shift = int(f * 0.5 / depth)                                    # 5 px per camera step
    expect = np.zeros((h, w), bool)
    expect[5:15, 5 - 0:25 - 0] = True
    got = out[("v3", 1)]                                            # camera at +0.5: object appears 5 px left
    expect = np.zeros((h, w), bool)
    expect[5:15, 5 - shift:25 - shift] = True
    assert (got & expect).sum() / float((got | expect).sum()) > 0.9

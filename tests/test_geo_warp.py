"""S1-E4a geometry: COLMAP readers and the two-view mask warp (pure numpy, runs in the container)."""
import importlib.util
import os
import struct

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "MVSeed", "geo", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ci = _load("colmap_io")
np = pytest.importorskip("numpy")
warp = _load("warp")


# ------------------------------------------------------------------ colmap_io
def test_qvec_to_rotmat_identity_and_quarter_turn():
    R = np.asarray(ci.qvec_to_rotmat([1, 0, 0, 0]))
    assert np.allclose(R, np.eye(3))
    R = np.asarray(ci.qvec_to_rotmat([2 ** -0.5, 0, 2 ** -0.5, 0]))       # +90 deg about y
    assert np.allclose(R @ np.array([0, 0, 1.0]), [1, 0, 0], atol=1e-12)
    assert np.allclose(np.asarray(ci.qvec_to_rotmat([2, 0, 0, 0])), np.eye(3))   # unnormalised input


def test_camera_models_and_distortion_roundtrip():
    pin = ci.Camera(1, "PINHOLE", 100, 80, [50, 60, 50, 40])
    assert (pin.fx, pin.fy, pin.cx, pin.cy) == (50, 60, 50, 40) and not pin.has_distortion
    assert pin.distort(0.3, -0.2) == (0.3, -0.2)
    rad = ci.Camera(2, "SIMPLE_RADIAL", 100, 80, [50, 50, 40, -0.05])
    assert rad.fx == rad.fy == 50 and rad.has_distortion
    for x, y in ((0.0, 0.0), (0.3, -0.2), (-0.5, 0.5)):
        xd, yd = rad.distort(x, y)
        assert max(abs(a - b) for a, b in zip(rad.undistort(xd, yd), (x, y))) < 1e-9
    assert rad.distort(0.0, 0.0) == (0.0, 0.0)
    with pytest.raises(AssertionError):
        ci.Camera(3, "OPENCV_FISHEYE", 10, 10, [1, 2, 3, 4])


def test_read_model_txt(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "# comment\n1 SIMPLE_RADIAL 1920 1088 1090.5 960 544 -0.0021\n2 PINHOLE 1600 905 907.8 907.9 800 452.5\n")
    # image 2 has NO observations, so COLMAP writes an empty POINTS2D line: parity must survive it
    (tmp_path / "images.txt").write_text(
        "# comment\n1 1 0 0 0 0.5 -0.25 3 1 v0.jpg\n100.5 200.5 7 101.5 201.5 -1\n"
        "2 0.7071 0 0.7071 0 1 2 3 2 v1.jpg\n\n"
        "3 1 0 0 0 4 5 6 1 v2.jpg\n1.5 2.5 9\n")
    cams, imgs = ci.read_model_txt(str(tmp_path))
    assert set(cams) == {1, 2} and cams[1].model == "SIMPLE_RADIAL" and cams[2].fy == 907.9
    assert set(imgs) == {"v0.jpg", "v1.jpg", "v2.jpg"}
    assert imgs["v0.jpg"].tvec == [0.5, -0.25, 3] and imgs["v0.jpg"].camera_id == 1
    assert imgs["v2.jpg"].tvec == [4, 5, 6]          # would be mis-parsed if the blank line desynced
    assert np.allclose(np.asarray(imgs["v0.jpg"].R), np.eye(3))
    (tmp_path / "odd.txt").write_text("1 1 0 0 0 0 0 0 1 a.jpg\n")
    with pytest.raises(AssertionError):
        ci.read_images_txt(str(tmp_path / "odd.txt"))


def test_read_colmap_array(tmp_path):
    p = tmp_path / "d.bin"
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    p.write_bytes(b"3&2&1&" + struct.pack("<6f", *vals))
    got, w, h, c = ci.read_colmap_array(str(p))
    assert (w, h, c) == (3, 2, 1) and got == vals
    d = ci.read_depth_map(str(p))
    assert d.shape == (2, 3) and d[1, 2] == 6.0
    (tmp_path / "bad.bin").write_bytes(b"3&2&1&" + struct.pack("<3f", 1, 2, 3))
    with pytest.raises(AssertionError):
        ci.read_colmap_array(str(tmp_path / "bad.bin"))


# ------------------------------------------------------------------ pixel maps
def test_pixel_maps_are_inverse_of_each_other():
    orig = ci.Camera(1, "SIMPLE_RADIAL", 200, 160, [100, 100, 80, -0.03])
    und = ci.Camera(2, "PINHOLE", 100, 80, [50, 50, 50, 40])
    h, w = und.height, und.width
    uo, vo = warp.undistorted_to_original_map(und, orig, h, w)
    back = warp.original_to_undistorted_map(orig, und, orig.height, orig.width)
    uu, _ = warp.sample_nearest(back[0], uo, vo)
    vv, _ = warp.sample_nearest(back[1], uo, vo)
    u, v = warp._grid(h, w)
    inside = (uo >= 1) & (uo < orig.width - 1) & (vo >= 1) & (vo < orig.height - 1)
    assert inside.mean() > 0.5
    assert np.abs(uu - u)[inside].max() < 1.5 and np.abs(vv - v)[inside].max() < 1.5
    # with no distortion the map is the exact affine rescale
    plain = ci.Camera(3, "PINHOLE", 200, 160, [100, 100, 100, 80])
    uo2, vo2 = warp.undistorted_to_original_map(und, plain, h, w)
    assert np.allclose(uo2, u * 2) and np.allclose(vo2, v * 2)


# ------------------------------------------------------------------ two-view warp
def _pair(baseline=0.5, depth=5.0, h=80, w=100, f=100.0):
    """Two pinhole cameras with parallel axes; camera 2 sits `baseline` to the +x of camera 1."""
    cam = ci.Camera(1, "PINHOLE", w, h, [f, f, w / 2.0, h / 2.0])
    ref = ci.Image(1, [1, 0, 0, 0], [0, 0, 0], 1, "ref.jpg")
    tgt = ci.Image(2, [1, 0, 0, 0], [-baseline, 0, 0], 1, "tgt.jpg")   # X_cam2 = X_world - (b,0,0)
    dref = np.full((h, w), depth, dtype="float32")
    dtgt = np.full((h, w), depth, dtype="float32")
    return cam, ref, tgt, dref, dtgt


def test_forward_and_backward_warp_shift_by_the_expected_disparity():
    f, b, z = 100.0, 0.5, 5.0
    cam, ref, tgt, dref, dtgt = _pair(baseline=b, depth=z, f=f)
    shift = f * b / z                                                  # 10 px, object moves to -x in cam 2
    m = np.zeros((80, 100), bool)
    m[30:50, 40:60] = True
    fwd = warp.forward_warp(m, cam, ref, dref, cam, tgt, dtgt)
    back = warp.backward_warp(m, cam, ref, dref, cam, tgt, dtgt)
    expect = np.zeros_like(m)
    expect[30:50, int(40 - shift):int(60 - shift)] = True
    assert warp.iou(fwd, expect) > 0.95, warp.iou(fwd, expect)
    assert warp.iou(back, expect) > 0.95, warp.iou(back, expect)
    assert warp.iou(fwd, back) > 0.9
    assert warp.iou(m, expect) < 0.6                                    # the shift is real, not a no-op


def test_zero_baseline_is_the_identity():
    cam, ref, tgt, dref, dtgt = _pair(baseline=0.0)
    m = np.zeros((80, 100), bool)
    m[20:60, 20:60] = True
    assert warp.iou(warp.backward_warp(m, cam, ref, dref, cam, tgt, dtgt), m) > 0.99
    assert warp.iou(warp.forward_warp(m, cam, ref, dref, cam, tgt, dtgt), m) > 0.99


def test_occlusion_test_drops_hidden_samples():
    cam, ref, tgt, dref, dtgt = _pair(baseline=0.5, depth=5.0)
    m = np.zeros((80, 100), bool)
    m[30:50, 40:60] = True
    dtgt_occ = dtgt.copy()
    dtgt_occ[:, :] = 1.0                                                # a wall right in front of camera 2
    fwd = warp.forward_warp(m, cam, ref, dref, cam, tgt, dtgt_occ)
    assert not fwd.any()
    dref_occ = dref.copy()
    dref_occ[:, :] = 1.0                                                # the surface is hidden in the reference
    back = warp.backward_warp(m, cam, ref, dref_occ, cam, tgt, dtgt)
    assert not back.any()
    # the reference sits 10 units BEHIND the surface, so every target point has z_ref < 0 there
    behind = ci.Image(3, [1, 0, 0, 0], [0, 0, -10.0], 1, "b.jpg")
    pts, _ = warp.unproject(cam, dtgt)
    in_ref = warp.transfer(pts, *warp.pose_matrices(tgt), *warp.pose_matrices(behind))
    assert (warp.project(cam, in_ref)[2] < 0).all()                     # the fixture really is behind
    assert not warp.backward_warp(m, cam, behind, dref, cam, tgt, dtgt).any()


def test_missing_depth_is_where_forward_and_backward_differ():
    cam, ref, tgt, dref, dtgt = _pair()
    m = np.zeros((80, 100), bool)
    m[30:50, 40:60] = True
    holes = dtgt.copy()
    holes[35:45, :] = 0.0                                               # target depth missing in a band
    back = warp.backward_warp(m, cam, ref, dref, cam, tgt, holes)
    fwd = warp.forward_warp(m, cam, ref, dref, cam, tgt, holes)
    assert not back[35:45, :].any() and fwd[35:45, :].any()              # forward fills what backward cannot
    assert warp.iou(fwd | back, fwd) == 1.0
    empty = np.zeros_like(m)
    assert not warp.forward_warp(empty, cam, ref, dref, cam, tgt, dtgt).any()
    assert not warp.backward_warp(empty, cam, ref, dref, cam, tgt, dtgt).any()


def test_iou_and_centre_offset():
    a = np.zeros((20, 20), bool); a[5:15, 5:15] = True
    b = a.copy()
    assert warp.iou(a, b) == 1.0 and warp.iou(a, ~a) == 0.0
    assert warp.iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0
    assert warp.centre_offset(a, b) == 0.0
    c = np.zeros((20, 20), bool); c[5:15, 9:19] = True                   # centroid 4 px away
    d = warp.centre_offset(c, a)
    assert abs(d - 4.0 / (2 * (100 / np.pi) ** 0.5)) < 1e-9
    assert warp.centre_offset(np.zeros((4, 4), bool), a) is None

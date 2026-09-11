"""S1-E4c: alpha-normalised render depth, and that the arm reuses E4a's rules unchanged."""
import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("numpy")


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gs_depth = _load("MVSeed/geo/gs_depth.py", "gs_depth")
e4c = _load("MVSeed/geo/e4c_project.py", "e4c_project")
e4a = e4c.e4a


def test_metric_depth_is_alpha_over_inverse_depth():
    np = pytest.importorskip("numpy")
    # one Gaussian at depth 4 covering the pixel fully: alpha 1, inv 1/4 -> depth 4
    assert gs_depth.metric_depth(np.array([1.0]), np.array([0.25]))[0] == pytest.approx(4.0)
    # half-covered pixel on the same surface: both accumulators halve, the ratio does not
    assert gs_depth.metric_depth(np.array([0.5]), np.array([0.125]))[0] == pytest.approx(4.0)


def test_metric_depth_rejects_thin_and_empty_pixels():
    np = pytest.importorskip("numpy")
    alpha = np.array([0.49, 0.51, 1.0, 0.0])
    inv = np.array([0.125, 0.125, 0.0, 0.0])
    d = gs_depth.metric_depth(alpha, inv)
    assert d[0] == 0.0                      # too transparent: below ALPHA_MIN
    assert d[1] == pytest.approx(0.51 / 0.125)
    assert d[2] == 0.0 and d[3] == 0.0      # nothing rendered -> no depth, like an MVS hole
    assert d.dtype == np.float32


def test_alpha_min_is_the_registered_constant():
    assert gs_depth.ALPHA_MIN == 0.5 and gs_depth.ITERATION == 30000


def test_e4c_reuses_e4a_gate_and_arms():
    """The prereg says only the depth source changes; guard the rest against drift."""
    assert (e4a.GATE_MIN, e4a.GATE_N, e4a.J_PASS) == (8, 16, 0.5)
    assert e4a.ARMS == ("back", "fwd", "union") and e4a.VERDICT_ARM == "union"
    assert e4c.OUT.endswith("e4c_project.json") and e4c.PARTS.endswith("e4c_project_parts")
    assert e4c.DEPTH_DIR == "depth"      # the linear read-out is a --linear diagnostic, never the default


def test_linear_readout_scales_through_the_renderer_clamp():
    """The rasteriser clamps its output to [0, 1], so the linear diagnostic must carry z/zmax."""
    src = open(os.path.join(REPO, "MVSeed", "geo", "gs_depth.py"), encoding="utf-8").read()
    assert "zmax = float(z.max())" in src and "* zmax" in src


def test_load_scene_geometry_is_none_without_a_trained_model(tmp_path, monkeypatch):
    monkeypatch.setattr(e4c, "CACHE", str(tmp_path))
    assert e4c.load_scene_geometry("Nowhere", ["v0"]) is None


def test_load_scene_geometry_skips_cameras_without_a_render(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    scene = "Toy"
    p = {k: os.path.join(str(tmp_path), scene, *v) for k, v in
         dict(sparse=("pose", "sparse", "0", "txt"), und=("undist", "sparse", "0", "txt"),
              depth=("gs", "depth")).items()}
    for d in p.values():
        os.makedirs(d)
    cam_orig = "1 SIMPLE_RADIAL 8 6 4.0 4.0 3.0 0.0\n"
    cam_und = "1 PINHOLE 8 6 4.0 4.0 4.0 3.0\n"
    pts = "1.0 1.0 -1\n"          # COLMAP writes a POINTS2D line for every image
    imgs = "1 1 0 0 0 0 0 0 1 v0.png\n" + pts + "2 1 0 0 0 0 0 0 1 v1.png\n" + pts
    open(os.path.join(p["sparse"], "cameras.txt"), "w").write(cam_orig)
    open(os.path.join(p["sparse"], "images.txt"), "w").write(imgs)
    open(os.path.join(p["und"], "cameras.txt"), "w").write(cam_und)
    open(os.path.join(p["und"], "images.txt"), "w").write(imgs)
    np.savez_compressed(os.path.join(p["depth"], "v0.npz"), depth=np.ones((6, 8), "float32"))
    monkeypatch.setattr(e4c, "CACHE", str(tmp_path))
    geo = e4c.load_scene_geometry(scene, ["v0", "v1"])
    assert set(geo) == {"v0"}                       # v1 has a pose but no render -> not available
    assert geo["v0"]["depth"].shape == (6, 8)


def test_prereg_registers_the_same_gate_as_e4a():
    text = open(os.path.join(REPO, "docs", "stage1-E4c-prereg.md"), encoding="utf-8").read()
    assert "16쌍 중 `J_union` > 0.5 가 ≥ 8쌍" in text
    assert "ALPHA_MIN = 0.5" in text

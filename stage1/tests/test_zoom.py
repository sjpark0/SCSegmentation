"""S1-E10 (docs/stage1-E10-prereg.md): torch-free parts of the crop-zoom pass, the preset / tag registries and the
in-process argument plumbing.  Host venv (numpy, cv2)."""
import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from geomem import pointmap, zoom  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem import stage1 as S  # noqa: E402

SIZES = [(1080, 1920), (1088, 1920), (1088, 2048), (1920, 2560), (2028, 2704)]   # every MUVOD scene resolution (meta.json)


def test_band_px_is_one_288_cell_of_the_longer_side():
    assert [zoom.band_px(hw) for hw in SIZES] == [7, 7, 8, 9, 10]
    assert zoom.band_px((1080, 1920)) == math.ceil(1920 / 288)


def test_window_for_fixed_half_window_clamped_and_containing_the_bbox():
    rng = np.random.default_rng(0)
    for hw in SIZES:
        H, W = hw
        wh, ww = H // 2, W // 2
        for _ in range(300):
            y0 = int(rng.integers(0, H - 1)); y1 = int(rng.integers(y0 + 1, H + 1))
            x0 = int(rng.integers(0, W - 1)); x1 = int(rng.integers(x0 + 1, W + 1))
            win = zoom.window_for((y0, y1, x0, x1), hw)
            fits = (y1 - y0) * 1.2 <= wh and (x1 - x0) * 1.2 <= ww
            assert (win is not None) == fits
            if win is None:
                continue
            wy0, wy1, wx0, wx1 = win
            assert (wy1 - wy0, wx1 - wx0) == (wh, ww) and wy0 >= 0 and wx0 >= 0 and wy1 <= H and wx1 <= W
            assert wy0 <= y0 and y1 <= wy1 and wx0 <= x0 and x1 <= wx1
            # the bbox never comes closer than wh/12 (>= 45 px > band_px) to a window edge that is not an image edge
            r = zoom.band_px(hw)
            if wy0 > 0: assert y0 - wy0 > r
            if wy1 < H: assert wy1 - y1 > r
            if wx0 > 0: assert x0 - wx0 > r
            if wx1 < W: assert wx1 - x1 > r


def test_window_for_centres_when_there_is_room():
    assert zoom.window_for((500, 580, 900, 1020), (1080, 1920)) == (270, 810, 480, 1440)
    assert zoom.window_for((0, 80, 0, 120), (1080, 1920)) == (0, 540, 0, 960)             # pushed inside at the corner
    assert zoom.window_for((0, 541, 0, 100), (1080, 1920)) is None                          # 541 * 1.2 > 540


def test_band_mask_and_paste_touch_only_the_band():
    full = np.zeros((120, 160), bool); full[30:80, 40:110] = True
    r = 7
    band = zoom.band_mask(full, r)
    yy, xx = np.mgrid[0:120, 0:160]
    inside_dil = (yy >= 30 - r) & (yy < 80 + r) & (xx >= 40 - r) & (xx < 110 + r)
    inside_ero = (yy >= 30 + r) & (yy < 80 - r) & (xx >= 40 + r) & (xx < 110 - r)
    assert np.array_equal(band, inside_dil & ~inside_ero)
    canvas = np.zeros_like(full); canvas[20:90, 30:120] = True          # a crop mask grown by 10 px everywhere
    final = zoom.paste(full, canvas, r)
    assert np.array_equal(final[~band], full[~band]) and np.array_equal(final[band], canvas[band])
    assert final.sum() > full.sum()
    canvas2 = np.zeros_like(full)                                       # an empty crop mask: only the band can be removed
    final2 = zoom.paste(full, canvas2, r)
    assert np.array_equal(final2, full & ~band) and final2.any()


def test_verdict_order():
    assert zoom.verdict(-1.0, True, 0.9, True) == "kept_absent"
    assert zoom.verdict(1.0, False, 0.9, True) == "kept_absent"
    assert zoom.verdict(1.0, True, 0.4, True) == "kept_iou"
    assert zoom.verdict(1.0, True, 0.9, False) == "kept_empty"
    assert zoom.verdict(1.0, True, 0.9, True) == "accepted"
    assert zoom.TAU_IOU == 0.5 and zoom.ZOOM == 2 and zoom.MARGIN == 0.10


def test_mask_iou_and_bbox():
    a = np.zeros((10, 10), bool); a[2:6, 2:6] = True
    b = np.zeros((10, 10), bool); b[4:8, 4:8] = True
    assert zoom.mask_iou(a, b) == pytest.approx(4 / 28)
    assert zoom.mask_iou(a, np.zeros_like(a)) == 0.0 and zoom.mask_iou(np.zeros_like(a), np.zeros_like(a)) == 0.0
    assert zoom.bbox_of(a) == (2, 6, 2, 6) and zoom.bbox_of(np.zeros_like(a)) is None


def test_cell_positions_full_window_is_the_integer_lattice_exactly():
    for hw in SIZES:
        H, W = hw
        t_x, t_y = zoom.cell_positions((0, H, 0, W), hw)
        j = np.tile(np.arange(72, dtype=np.float32), 72); i = np.repeat(np.arange(72, dtype=np.float32), 72)
        assert np.array_equal(t_x, j) and np.array_equal(t_y, i)
    # a 2x window at (y0, x0): crop cell j sits at full-view cell x0*72/W + (j + 0.5)/2 - 0.5
    H, W = 1080, 1920
    t_x, t_y = zoom.cell_positions((270, 810, 480, 1440), (H, W))
    j = np.tile(np.arange(72), 72); i = np.repeat(np.arange(72), 72)
    assert np.allclose(t_x, 480 * 72 / W + (j + 0.5) / 2 - 0.5, atol=1e-5)
    assert np.allclose(t_y, 270 * 72 / H + (i + 0.5) / 2 - 0.5, atol=1e-5)


def test_rides_edge():
    hw = (100, 200)
    canvas = np.zeros(hw, bool); canvas[40:60, 60:90] = True
    assert not zoom.rides_edge(canvas, (25, 75, 50, 150), hw)
    canvas[25, 70] = True
    assert zoom.rides_edge(canvas, (25, 75, 50, 150), hw)
    canvas2 = np.zeros(hw, bool); canvas2[0, 70] = True                  # touching an IMAGE edge does not count
    assert not zoom.rides_edge(canvas2, (0, 50, 50, 150), hw)


def _synthetic_cache(tmp_path, h=27, w=48):
    cache = tmp_path / "pi3"
    cache.mkdir()
    rng = np.random.default_rng(1)
    K = np.array([[50.0, 0, w / 2 - 0.5], [0, 50.0, h / 2 - 0.5], [0, 0, 1.0]])
    for cam, tz in (("ref", 0.0), ("v", 0.3)):
        depth = 2.0 + rng.random((h, w)) * 0.5
        depth[rng.random((h, w)) < 0.1] = 0.0                               # some invalid pixels
        c2w = np.eye(4); c2w[0, 3] = tz
        np.savez(cache / f"{cam}.npz", depth=depth, conf=np.ones_like(depth), K=K, c2w=c2w)
    return str(cache)


def test_window_grid_full_window_reproduces_view_grid(tmp_path):
    cache = _synthetic_cache(tmp_path)
    for cam in ("ref", "v"):
        vg = pointmap.view_grid(cache, cam, "ref")
        src = pointmap.ref_frame_points(cache, cam, "ref")
        wg = pointmap.window_grid(src, (0, 1080, 0, 1920), (1080, 1920))
        assert np.array_equal(vg["p"], wg["p"]) and np.array_equal(vg["valid"], wg["valid"]) and vg["scale"] == wg["scale"]
        assert vg["p"].shape == (72, 72, 3) and vg["p"].dtype == np.float32


def test_window_grid_window_pools_the_window_only(tmp_path):
    cache = _synthetic_cache(tmp_path, h=72, w=72)
    src = pointmap.ref_frame_points(cache, "v", "ref")
    p, valid, _ = src
    assert pointmap.window_to_work((270, 810, 480, 1440), (1080, 1920), (72, 72)) == (18, 54, 18, 54)
    wg = pointmap.window_grid(src, (270, 810, 480, 1440), (1080, 1920))
    g, cv = pointmap.pool_grid(p[18:54, 18:54], valid[18:54, 18:54], 72)
    assert np.array_equal(wg["p"], g.astype(np.float32)) and np.array_equal(wg["valid"], cv)
    with pytest.raises(AssertionError):
        pointmap.window_to_work((0, 1, 0, 1), (1080, 1920), (27, 48))      # rounds to an empty window


def test_presets_and_tag_registries():
    assert S.PRESETS["geoset_rep"] == dict(arm="geoset_pred_ft", gate="reproj")
    assert S.PRESETS["geoset_rep_z2"] == dict(arm="geoset_pred_ft", gate="reproj", zoom=2, zoom_rope="grid")
    assert S.PRESETS["geoset_rep_z2r"] == dict(arm="geoset_pred_ft", gate="reproj", zoom=2, zoom_rope="remap")
    assert list(S.PRESETS)[-7:] == ["geoset_rep_z2", "geoset_rep_z2r", "geoset_rep_x", "geoset_rep_xm", "geoset_cutall_rep_x",
                                    "geoset_rep_xs", "geoset_rep_xs_ab"]      # S1-E15 main preset appended last
    main = os.path.dirname(ROOT)
    spec = importlib.util.spec_from_file_location("stage1_supply", os.path.join(main, "SCSam3", "stage1_supply.py"))
    supply = importlib.util.module_from_spec(spec); spec.loader.exec_module(supply)
    assert tuple(S.PRESETS) == supply.TAGS
    spec2 = importlib.util.spec_from_file_location("build_geoset_seeds", os.path.join(main, "MVSeed", "build_geoset_seeds.py"))
    bgs = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(bgs)
    assert set(bgs.TAGS) == set(supply.TAGS)


def test_check_gate_out_refuses_the_arms_own_folder_with_zoom():
    R.check_gate_out("geoset_pred_ft", "reproj", "geoset_pred_ft_rep_z2_e10", "pred", 2)
    R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None)
    with pytest.raises(SystemExit):
        R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", 2)


@pytest.mark.parametrize("tag", ["geoset_rep", "geoset_rep_z2", "geoset_rep_z2r", "geoset_rep_x", "geoset_rep_xm", "geoset_rep_xs"])
def test_stage1_run_forwards_every_preset_key(tag, monkeypatch):
    captured = {}

    def fake_decode(scene, args, sink=None):
        captured["args"] = args
        sink.json("final", "round1_conf.json", {})
        if getattr(args, "zoom", None) is not None:
            sink.json("final", "zoom_conf.json", {"cam": {}})
        if getattr(args, "xcl", "none") != "none":
            sink.json("final", "xcl_conf.json", {"cam": {}})
        return {"arm": args.arm}

    scene = SimpleNamespace(name="X", ref_cam="c0", objects=[], ref_labels=np.zeros((4, 4), np.int32), hw=(4, 4))
    monkeypatch.setattr(S.Stage1, "load", lambda self: None)
    monkeypatch.setattr(common, "load_scene", lambda data, name: scene)
    monkeypatch.setattr(R, "decode_scene", fake_decode)
    res = S.Stage1(tag).run("X")
    a = captured["args"]
    p = S.PRESETS[tag]
    assert a.arm == p["arm"] and a.gate == p["gate"] and a.out == S.FINAL and a.r1_out == S.ROUND1 and a.round2 == "pred"
    assert getattr(a, "zoom", None) == p.get("zoom") and getattr(a, "zoom_rope", None) == p.get("zoom_rope")
    assert getattr(a, "xcl", "none") == p.get("xcl", "none")
    assert (res["zoom_conf"] is not None) == ("zoom" in p)
    assert (res["xcl_conf"] is not None) == ("xcl" in p)


def test_check_preflight_reports_zoom(monkeypatch, tmp_path):
    (tmp_path / "X").mkdir(); (tmp_path / "X" / "meta.json").write_text("{}")
    ck = tmp_path / "runs" / "geoset_pred_ft" / "fold0"; ck.mkdir(parents=True)
    (ck / "config.json").write_text('{"fold": 0, "scenes": [], "ckpt_sha256": "x"}')
    monkeypatch.setattr(R, "checkpoint_for", lambda scene, arm, runs, folds: str(ck / "ckpt.pt"))
    import geomem.folds
    monkeypatch.setattr(geomem.folds, "load_folds", lambda: {})
    out = S.check_preflight("geoset_rep_z2r", "X", data=str(tmp_path), runs=str(tmp_path / "runs"))
    assert out["zoom"] == 2 and out["zoom_rope"] == "remap" and out["arm"] == "geoset_pred_ft"
    out0 = S.check_preflight("geoset_rep", "X", data=str(tmp_path), runs=str(tmp_path / "runs"))
    assert out0["zoom"] is None and out0["zoom_rope"] is None

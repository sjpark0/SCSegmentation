"""S1-E4a summary logic: registered denominators, exact-IoU pass rule and the 132-row hallucination rate."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("numpy")
spec = importlib.util.spec_from_file_location("e4a_project", os.path.join(REPO, "MVSeed", "geo", "e4a_project.py"))
e4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4a)
warp = e4a.warp


def _row(scene, cam, obj, gt_px, j, rig="pinhole", available=True, px=100, tag=None):
    r = dict(scene=scene, cam=cam, obj=obj, ref_cam="c0", gt_px=gt_px, rig=rig, available=available,
             ref_registered=available, tgt_registered=available, ref_gt_px=max(gt_px, 1),
             centre_offset=0.1 if j is not None else None, tgt_depth_frac=0.9 if available else None)
    for a in e4a.ARMS:
        r[f"px_{a}"] = px
        r[f"J_{a}"] = None if j is None else round(j, 4)
        r[f"pass_{a}"] = bool(j is not None and j > e4a.J_PASS)
    if tag:
        r["set"], r["size_bin"] = tag, (None if gt_px == 0 else "<2000" if gt_px < 2000 else
                                        "2000-10000" if gt_px < 10000 else ">=10000")
    return r


def _population(pass_pin=8, n_empty=132, halluc=20, unavailable_empty=27):
    rows = []
    for i in range(16):                                        # F0 & pinhole & >= 2000 px
        rows.append(_row("A", "c1", i, 5000, 0.9 if i < pass_pin else 0.1, tag="F0"))
    for i in range(3):                                         # F0 & hemisphere & >= 2000 px
        rows.append(_row("H", "c1", i, 5000, 0.2, rig="hemisphere", available=False, px=0, tag="F0"))
    for i in range(22):
        rows.append(_row("A", "c2", i, 500, 0.6, tag="F0"))
    for i in range(623):
        rows.append(_row("B", "c1", i, 3000, 0.8, tag="N0"))
    for i in range(n_empty):
        av = i >= unavailable_empty
        hal = av and i < unavailable_empty + halluc          # the hallucinating rows are available ones
        rows.append(_row("C", "c1", i, 0, None, available=av, px=(100 if hal else 0), tag="gt_empty"))
    return rows


def test_registered_denominators_and_gate():
    s = e4a.summarize(_population(pass_pin=8))
    assert s["denominators"]["match"] is True
    assert s["f0_ge2k_pinhole"]["n"] == 16 and s["f0_ge2k"]["n"] == 19 and s["f0_hemisphere"]["n"] == 3
    assert s["gates"]["gate_projection"]["pass"] is True and s["gates"]["gate_projection"]["value"] == 8
    assert s["gates"]["gate_projection"]["envelope_points"] == 0.25
    assert s["gates"]["verdict"].startswith("proceed")
    assert e4a.summarize(_population(pass_pin=7))["gates"]["gate_projection"]["pass"] is False
    # an incomplete run cannot pass: the n == GATE_N guard
    short = [r for r in _population(pass_pin=16) if not (r["set"] == "F0" and r["gt_px"] == 5000 and r["obj"] > 12)]
    s2 = e4a.summarize(short)
    assert s2["f0_ge2k_pinhole"]["n"] == 13 and s2["gates"]["gate_projection"]["pass"] is False
    assert s2["denominators"]["match"] is False


def test_pass_uses_the_exact_iou_not_the_rounded_value():
    rows = _population(pass_pin=0)
    marginal = rows[0]
    for a in e4a.ARMS:                                          # true IoU 0.500049 -> rounds to 0.5
        marginal[f"J_{a}"], marginal[f"pass_{a}"] = round(0.500049, 4), True
    assert marginal["J_union"] == 0.5
    assert e4a.summarize(rows)["f0_ge2k_pinhole"]["pass"] == 1   # the rounded value would have failed "> 0.5"


def test_hallucination_denominator_is_the_registered_132():
    s = e4a.summarize(_population(halluc=20, unavailable_empty=27))
    e = s["gt_empty"]
    assert e["n"] == 132 and e["available"] == 105 and e["halluc"] == 20
    assert e["halluc_rate"] == round(20 / 132, 4)               # registered population, comparable with E2
    assert e["halluc_rate_available"] == round(20 / 105, 4)
    assert e["halluc_rate"] < e["halluc_rate_available"]


def test_unavailable_rows_stay_in_the_population_and_fail():
    rows = _population(pass_pin=16)
    for r in rows:
        if r["set"] == "F0" and r["rig"] == "pinhole" and r["gt_px"] == 5000 and r["obj"] < 4:
            r.update(available=False, px_union=0, J_union=0.0, pass_union=False,
                     J_back=0.0, pass_back=False, J_fwd=0.0, pass_fwd=False)
    s = e4a.summarize(rows)
    g = s["f0_ge2k_pinhole"]
    assert g["n"] == 16 and g["available"] == 12 and g["pass"] == 12
    assert s["gates"]["gate_projection"]["pass"] is True        # 12 >= 8, denominator still 16


# ----------------------------------------------------------------------------- S1-E4b (Pi3X provider)
e4b_spec = importlib.util.spec_from_file_location("e4b_project", os.path.join(REPO, "MVSeed", "geo", "e4b_project.py"))
e4b = importlib.util.module_from_spec(e4b_spec)
e4b_spec.loader.exec_module(e4b)
np = pytest.importorskip("numpy")


def test_target_size_keeps_aspect_under_the_pixel_limit():
    for w0, h0 in ((1920, 1080), (2560, 1920), (2704, 2028), (2048, 1088)):
        w, h = e4b.target_size(w0, h0)
        assert w % 14 == 0 and h % 14 == 0
        assert w * h <= e4b.PIXEL_LIMIT * 1.05
        assert abs((w / h) / (w0 / h0) - 1) < 0.03
    assert e4b.target_size(10, 10, pixel_limit=1) == (14, 14)          # never below one patch


def test_rotmat_to_qvec_round_trips_through_colmap():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=3)
        t = np.linalg.norm(a)
        k = a / t
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * (Kx @ Kx)    # Rodrigues
        back = np.asarray(e4a.ci.qvec_to_rotmat(e4b.rotmat_to_qvec(R)))
        assert np.allclose(back, R, atol=1e-9), np.abs(back - R).max()
    assert np.allclose(np.asarray(e4a.ci.qvec_to_rotmat(e4b.rotmat_to_qvec(np.eye(3)))), np.eye(3))


def test_cameras_from_pi3_conventions():
    K = np.array([[500.0, 0, 335.5], [0, 500.0, 188.5], [0, 0, 1]])
    R_c2w = np.asarray(e4a.ci.qvec_to_rotmat([0.5 ** 0.5, 0, 0.5 ** 0.5, 0]))
    T = np.eye(4); T[:3, :3] = R_c2w; T[:3, 3] = [1.0, -2.0, 3.0]
    cam, img = e4b.cameras_from_pi3(K, T, 672, 378)
    assert (cam.fx, cam.fy) == (500.0, 500.0)
    assert (cam.cx, cam.cy) == (336.0, 189.0)                          # +0.5: index -> pixel-centre
    assert cam.model == "PINHOLE" and not cam.has_distortion
    R, t = warp.pose_matrices(img)
    assert np.allclose(R, R_c2w.T)                                     # world-to-camera
    centre = -R.T @ t                                                  # camera centre back in world
    assert np.allclose(centre, [1.0, -2.0, 3.0], atol=1e-9)
    # a point 5 in front of the camera projects to the principal point
    world = T[:3, :3] @ np.array([0, 0, 5.0]) + T[:3, 3]
    u, v, z = warp.project(cam, (world @ R.T + t)[None])
    assert abs(u[0] - cam.cx) < 1e-6 and abs(v[0] - cam.cy) < 1e-6 and abs(z[0] - 5.0) < 1e-9

"""S1-E4a summary logic: registered denominators, exact-IoU pass rule and the 132-row hallucination rate."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("numpy")
spec = importlib.util.spec_from_file_location("e4a_project", os.path.join(REPO, "MVSeed", "geo", "e4a_project.py"))
e4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4a)


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

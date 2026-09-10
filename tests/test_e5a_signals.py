"""S1-E5a: the ground-truth-free seed signals and the matched-false-positive readout (pure logic)."""
import importlib.util
import math
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("numpy")
import numpy as np  # noqa: E402

spec = importlib.util.spec_from_file_location("e5a_signals", os.path.join(REPO, "MVSeed", "geo", "e5a_signals.py"))
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)


def test_equivalent_diameter_and_centre_ratio():
    assert e5.equivalent_diameter(0) == 0.0
    assert abs(e5.equivalent_diameter(math.pi * 25) - 10.0) < 1e-9
    a = np.zeros((40, 40), bool); a[10:20, 10:20] = True          # 100 px, diameter 11.28
    assert e5.centre_ratio(a, a) == 0.0
    b = np.zeros((40, 40), bool); b[10:20, 21:31] = True          # same size, centroid 11 px away
    assert abs(e5.centre_ratio(a, b) - 11.0 / e5.equivalent_diameter(100)) < 1e-9
    empty = np.zeros((40, 40), bool)
    assert math.isinf(e5.centre_ratio(a, empty)) and math.isinf(e5.centre_ratio(empty, a))
    big = np.zeros((40, 40), bool); big[0:40, 0:40] = True        # a large projection forgives the same offset
    assert e5.centre_ratio(a, big) < e5.centre_ratio(a, b)


def test_p5_area_rule():
    assert e5.p5_threshold([1000, 2000, 3000]) == max(64.0, 0.05 * 2000)
    assert e5.p5_threshold([1000, 0, 0]) == 64.0                   # fewer than three non-zero -> 64
    assert e5.p5_threshold([100, 200, 300]) == 64.0                # the 5 % of the median floor
    assert e5.p5_threshold([]) == 64.0
    T = e5.p5_threshold([10000, 20000, 30000])                     # 1000
    assert e5.p5_flag(seed_area=500, prompt_area=5000, threshold=T) is True
    assert e5.p5_flag(seed_area=5000, prompt_area=5000, threshold=T) is False
    assert e5.p5_flag(seed_area=0, prompt_area=10, threshold=T) is False    # tiny prompt is not this rule's job


def _row(kind, size_bin=">=10000", rig="pinhole", **sig):
    s = dict(P5area=False, S2a_iou=0.9, S2b_iou=0.9, S2max_iou=0.9, S1_rev=0.9, S2a_ctr=0.1, S2b_ctr=0.1)
    s.update(sig)
    r = dict(scene="S", cam="c", obj=1, gt_px=20000, seed_px=1000, size_bin=size_bin, rig=rig, signals=s)
    r["set"] = "F0" if kind in ("misplaced", "empty") else ("N0" if kind == "normal" else "gt_empty")
    if r["set"] == "F0":
        r["f0_kind"] = kind
    if kind == "gt_empty":
        r["gt_px"], r["seed_px"], r["size_bin"] = 0, 5, None
    return r


def test_flagged_operating_points():
    assert e5.flagged(_row("normal", S2b_iou=0.49), "S2b_iou") and not e5.flagged(_row("normal", S2b_iou=0.5), "S2b_iou")
    assert e5.flagged(_row("normal", S2b_ctr=1.01), "S2b_ctr") and not e5.flagged(_row("normal", S2b_ctr=1.0), "S2b_ctr")
    assert e5.flagged(_row("normal", S2b_ctr=None), "S2b_ctr")     # inf (no projection) is flagged
    assert e5.flagged(_row("normal", S2a_iou=None), "S2a_iou")
    assert e5.flagged(_row("normal", P5area=True), "P5area") and not e5.flagged(_row("normal"), "P5area")


def test_summarize_counts_hits_and_false_positives():
    rows = ([_row("misplaced", S2b_ctr=3.0, S2max_iou=0.1) for _ in range(20)]
            + [_row("misplaced") for _ in range(18)]
            + [_row("empty", P5area=True, S2max_iou=0.0, S2b_ctr=None) for _ in range(3)]
            + [_row("normal", P5area=True) for _ in range(11)]
            + [_row("normal") for _ in range(612)]
            + [_row("gt_empty", S2max_iou=0.0) for _ in range(9)])
    s = e5.summarize(rows)
    assert (s["n_misplaced"], s["n_empty_seed"], s["n_normal"]) == (38, 3, 623)
    assert s["n_gt_empty"] == 9 and s["n_gt_empty_nonempty_seed"] == 9
    assert s["operating_point"]["P5area"] == dict(s["operating_point"]["P5area"],
                                                  hits=0, fp=11, empty_seed_hits=3)
    assert s["operating_point"]["S2b_ctr"]["hits"] == 20 and s["operating_point"]["S2b_ctr"]["fp"] == 0
    assert s["operating_point"]["S2max_iou"]["hits"] == 20
    assert s["operating_point"]["S2max_iou"]["gt_empty_hits"] == 9
    assert s["matched_fp"]["target_fp"] == 11
    assert s["combinations"]["P5area OR S2max_iou"]["hits"] == 20
    assert s["combinations"]["P5area OR S2max_iou"]["fp"] == 11


def test_matched_threshold_sweeps_to_the_allowed_false_positives():
    # scores: misplaced are low, normals mostly high with a few low ones
    pos = [_row("misplaced", S2b_iou=v) for v in (0.0, 0.05, 0.1, 0.2, 0.45, 0.8)]
    neg = [_row("normal", S2b_iou=v) for v in (0.03, 0.3, 0.6, 0.7, 0.9, 0.95)]
    m0 = e5.matched_threshold(pos, neg, "S2b_iou", target_fp=0)
    assert m0["fp"] == 0 and m0["hits"] == 1                       # only below 0.03 is free of normals
    m1 = e5.matched_threshold(pos, neg, "S2b_iou", target_fp=1)
    assert m1["fp"] <= 1 and m1["hits"] == 4                       # below 0.3 catches four
    m2 = e5.matched_threshold(pos, neg, "S2b_iou", target_fp=2)
    assert m2["fp"] <= 2 and m2["hits"] == 5
    # centre signals flag HIGH values, and a missing projection (None) sits at the flagged end
    posc = [_row("misplaced", S2b_ctr=v) for v in (5.0, 4.0, 2.0)] + [_row("misplaced", S2b_ctr=None)]
    negc = [_row("normal", S2b_ctr=v) for v in (0.1, 0.2, 3.0)]
    mc = e5.matched_threshold(posc, negc, "S2b_ctr", target_fp=0)
    assert mc["fp"] == 0 and mc["hits"] == 3                       # above 3.0: 5.0, 4.0 and the missing one

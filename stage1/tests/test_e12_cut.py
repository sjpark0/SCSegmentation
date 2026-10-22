"""S1-E12 (docs/stage1-E12-prereg.md section 2): arm plumbing, presets / TAGS, the half-plane cut rule and its RNG
stream.  Host venv (numpy); geomem.train imports torch, so its flag table is checked in tests_torch/test_e12_cut_torch.py."""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from geomem import arms  # noqa: E402
from geomem.data import GRID, cut_rng, fov_cut, memory_rng  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem import stage1 as S  # noqa: E402

MAIN = os.path.dirname(ROOT)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ arms
def test_cut_arms_are_predicted_memory_set2_arms():
    assert arms.CUT_ARMS == ("geoset_cut0_ft", "geoset_cutall_ft")
    assert arms.CUT_PROB == {"geoset_cut0_ft": 0.0, "geoset_cutall_ft": 1.0}
    for a in arms.CUT_ARMS:
        # PRED_MEM_ARMS does not derive from SET2_ARMS: without the explicit entry the arm would train with GT memory
        assert a in arms.SET2_ARMS and a in arms.PRED_MEM_ARMS and a in arms.NO_EMBED_ARMS
        assert a not in arms.DEC_ARMS and a not in arms.BAND_ARMS
        assert arms.base_arm(a) == a
        assert a in R.ARMS and a in R.SET_ARMS


# ------------------------------------------------------------------ presets / TAGS
def test_presets_and_tags():
    assert S.PRESETS["geoset_cutall_rep_x"] == dict(arm="geoset_cutall_ft", gate="reproj", xcl="out")
    assert S.PRESETS["geoset_rep_x"] == dict(arm="geoset_pred_ft", gate="reproj", xcl="out")      # the adopted preset untouched
    assert list(S.PRESETS)[-3:] == ["geoset_cutall_rep_x", "geoset_rep_xs", "geoset_rep_xs_ab"]   # S1-E14 S after M+, S1-E15 main last
    assert all(p["arm"] != "geoset_cut0_ft" for p in S.PRESETS.values())                       # C0' never reaches stage 2
    supply = _load("stage1_supply", os.path.join(MAIN, "SCSam3", "stage1_supply.py"))
    bgs = _load("build_geoset_seeds", os.path.join(MAIN, "MVSeed", "build_geoset_seeds.py"))
    assert tuple(S.PRESETS) == supply.TAGS and set(bgs.TAGS) == set(supply.TAGS)


# ------------------------------------------------------------------ the cut rule
def _rect(b, y0, y1, x0, x1):
    c = np.zeros((b, GRID, GRID), bool)
    c[:, y0:y1, x0:x1] = True
    return c


def test_fov_cut_drops_every_cell_on_one_side_of_the_cut_line():
    cells = _rect(1, 10, 30, 20, 50)                                   # 20 x 30 = 600 cells
    for seed in range(40):
        g = np.random.default_rng(seed)
        drop, recs = fov_cut(cells, g)
        r = recs[0]
        assert r["eligible"] and r["side"] in (0, 1, 2, 3) and 0.3 <= r["f"] <= 0.7
        d = drop[0].reshape(GRID, GRID)
        coord = {0: np.mgrid[0:GRID, 0:GRID][1], 1: -np.mgrid[0:GRID, 0:GRID][1],
                 2: np.mgrid[0:GRID, 0:GRID][0], 3: -np.mgrid[0:GRID, 0:GRID][0]}[r["side"]]
        if r["dropped"]:
            # the dropped set is exactly {coord < threshold}: object AND background cells on that side
            thr = coord[~d].min()
            assert (d == (coord < thr)).all()
            assert d.sum() > r["dropped"]                                  # background cells are dropped too
        kept = int((cells[0] & ~d).sum())
        assert kept >= r["f"] * 600 - 1e-9 and kept >= 6
        assert r["dropped"] == 600 - kept


def test_fov_cut_kept_fraction_bounds_with_ties():
    cells = _rect(1, 0, 20, 0, 40)                                      # 800 cells; a cut line holds 20 (side 0/1) or 40 (2/3)
    for seed in range(200):
        drop, recs = fov_cut(cells, np.random.default_rng(seed))
        r = recs[0]
        kept = r["n"] - r["dropped"]
        line = 20 if r["side"] in (0, 1) else 40
        assert r["f"] * r["n"] <= kept <= r["f"] * r["n"] + line


def test_fov_cut_small_objects_draw_nothing():
    cells = np.zeros((3, GRID, GRID), bool)
    cells[0, 0:4, 0:4] = True                                           # 16 cells < 20: ineligible
    cells[1, 10:30, 10:30] = True                                       # eligible
    cells[2, 50:54, 50:54] = True                                       # ineligible
    g1, g2 = np.random.default_rng(7), np.random.default_rng(7)
    drop, recs = fov_cut(cells, g1)
    assert [r["eligible"] for r in recs] == [False, True, False]
    assert not drop[0].any() and not drop[2].any()
    # exactly two draws (side, f) for the one eligible object
    side, f = int(g2.integers(4)), float(g2.uniform(0.3, 0.7))
    assert recs[1]["side"] == side and recs[1]["f"] == f
    assert g1.random() == g2.random()


def test_fov_cut_no_cut_when_one_line_holds_the_quantile():
    cells = np.zeros((1, GRID, GRID), bool)
    cells[0, 10:40, 33] = True                                          # a vertical line of 30 cells
    g = np.random.default_rng(0)
    for _ in range(50):
        drop, recs = fov_cut(cells, g)
        if recs[0]["side"] in (0, 1):                                   # left/right: every cell has the same x
            assert recs[0]["dropped"] == 0 and not drop[0].any()


def test_fov_cut_shape_and_token_index():
    cells = _rect(2, 0, 72, 0, 72)
    drop, _ = fov_cut(cells, np.random.default_rng(3))
    assert drop.shape == (2, GRID * GRID) and drop.dtype == bool
    grid = np.arange(GRID * GRID).reshape(GRID, GRID)                   # token index y * 72 + x (row-major)
    assert grid[5, 7] == 5 * 72 + 7


def test_cut_rng_is_its_own_stream():
    a, b = cut_rng(0, 12), cut_rng(0, 12)
    assert a.random() == b.random()
    assert cut_rng(0, 12).random() != memory_rng(0, 12).random()
    shared = np.random.default_rng(0)
    before = shared.bit_generator.state
    cut_rng(0, 5).random(); memory_rng(0, 5).random()
    assert shared.bit_generator.state == before                         # neither touches the shared training rng


def test_cut_rng_draw_order_and_distribution():
    """Coin first, then (side, f) per eligible object, all from ONE generator per step."""
    cells = _rect(2, 10, 40, 10, 40)
    sides, fs = [], []
    for step in range(10_000):
        g = cut_rng(0, step)
        g_ref = cut_rng(0, step)
        coin = g.random()
        assert coin == g_ref.random()
        _, recs = fov_cut(cells, g)
        assert recs[0]["side"] == int(g_ref.integers(4)) and recs[0]["f"] == float(g_ref.uniform(0.3, 0.7))
        sides += [r["side"] for r in recs]
        fs += [r["f"] for r in recs]
    freq = np.bincount(sides, minlength=4) / len(sides)
    assert np.all(np.abs(freq - 0.25) < 0.02) and abs(np.mean(fs) - 0.5) < 0.01

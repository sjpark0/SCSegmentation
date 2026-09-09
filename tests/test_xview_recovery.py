"""T10: eval/xview_recovery.py implements docs/xview-recovery-prereg.md section 1 verbatim."""
import math
import os
import sys

import pytest

from conftest import REPO

sys.path.insert(0, REPO)
from eval import xview_recovery as xr  # noqa: E402


def test_runs():
    assert xr.runs([]) == []
    assert xr.runs([True, True, False, True]) == [(0, 2), (3, 4)]
    assert xr.runs([False, False]) == []
    assert xr.runs([True] * 5) == [(0, 5)]


def test_stretch_needs_three_consecutive_gt_present_lost_frames():
    gt = [1] * 8
    lose = [0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.9]      # runs of 3 and 2
    hold = [0.9] * 8
    st = xr.stretches(lose, hold, gt)
    assert [(s["start"], s["end"]) for s in st] == [(0, 3)]
    assert st[0]["kind"] == "never"                     # GT from frame 0, never held before


def test_stretch_needs_two_held_frames_on_the_other_side():
    gt = [1] * 6
    lose = [0.0] * 6
    hold = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0]               # only one frame held
    assert xr.stretches(lose, hold, gt) == []
    hold[1] = 0.9
    assert len(xr.stretches(lose, hold, gt)) == 1


def test_gt_absent_frames_break_runs_and_define_entry():
    gt = [0, 0, 5, 5, 5, 5]
    lose = [0.0] * 6
    hold = [0.0, 0.0, 0.9, 0.9, 0.9, 0.9]
    st = xr.stretches(lose, hold, gt)
    assert [(s["start"], s["end"], s["kind"]) for s in st] == [(2, 6, "entry")]


def test_occlusion_means_the_loser_held_it_before():
    gt = [1] * 8
    lose = [0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9]
    hold = [0.9] * 8
    st = xr.stretches(lose, hold, gt)
    assert [(s["start"], s["end"], s["kind"]) for s in st] == [(2, 5, "occlusion")]
    assert st[0]["gain"] == pytest.approx(0.9)


def test_in_between_values_are_neither_held_nor_lost():
    gt = [1] * 5
    lose = [0.3] * 5                                     # not < 0.1
    hold = [0.9] * 5
    assert xr.stretches(lose, hold, gt) == []
    lose = [0.0] * 5
    hold = [0.4] * 5                                     # not > 0.5
    assert xr.stretches(lose, hold, gt) == []


def test_binomial_two_sided():
    assert xr.binom_two_sided(3, 3) == pytest.approx(0.25)
    assert xr.binom_two_sided(6, 6) == pytest.approx(2 / 64)
    assert xr.binom_two_sided(3, 4) == pytest.approx(0.625)
    assert xr.binom_two_sided(2, 4) == pytest.approx(1.0)
    assert math.isnan(xr.binom_two_sided(0, 0))


def test_verdict_rules(capsys):
    def verdict(n_rec, n_ind):
        ev = [dict(direction="recovered", kind="occlusion", gain=0.5, scene="s", cam="c", obj=1,
                   start=0, end=3, held=2)] * n_rec + \
             [dict(direction="induced", kind="occlusion", gain=0.5, scene="s", cam="c", obj=1,
                   start=0, end=3, held=2)] * n_ind
        r = xr.summarise("t", ev, 0)
        capsys.readouterr()
        return r["verdict"]
    assert verdict(3, 0) == "지지·검정력 부족"
    assert verdict(6, 0) == "확인"
    assert verdict(3, 1) == "미확인"
    assert verdict(2, 2) == "기각"
    assert verdict(5, 1) == "미확인"                     # p = 0.219
    assert verdict(9, 1) == "확인"                       # p = 0.021

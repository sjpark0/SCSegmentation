"""T5: SCSam3/runMVSeg.py resolve_run (+ argparse) for every canonical MVSeg dataset.

Torch-free: runMVSeg.py imports only argparse/json/os/sys at module level.  Covers the
legacy resolution table (SPEC 2.6), closure session counts, the XW defaults and every
refusal, including the F1 (lazy/empty closure), F2 (mirror guard) and F3 (XW+all
default name) amendments.
"""
import argparse
import os
import sys

import pytest

from conftest import REPO, load_runmvseg

mod = load_runmvseg()
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CANONICAL_K = {"AlexaMeadeExhibit": 4, "AlexaMeadeFacePaint": 9, "Barn": 11, "Blocks": 10,
               "Breakfast": 10, "Carpark": 9, "CoffeeMartini": 15, "Dog": 4, "Fencing": 10,
               "FlameSteak": 17, "Frog": 10, "MATF": 10, "Painter": 16, "PoznanStreet": 9,
               "Welder": 4}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in mod.XVIEW_ENV:
        monkeypatch.delenv(k, raising=False)


def ns(**kw):
    d = dict(algo="MVOpt", xview_window=None, xview_hygiene=False, track_cams=None, out=None)
    d.update(kw)
    return argparse.Namespace(**d)


def cams(name):
    c = mod.load_config(CONFIG, name)
    cam_names = [mod.cam_name(x, c["prefix"], c["prefix1"]) for x in c["perms"]]
    written = [mod.cam_name(x, c["prefix"], c["prefix1"]) for x in c["cam_list"]]
    return cam_names, written


def scored(cam_names, written):
    return [i for i, n in enumerate(cam_names) if n in written]


@pytest.mark.parametrize("ds", sorted(CANONICAL_K))
def test_legacy_table(ds):
    cam_names, written = cams(ds)
    n, sc = len(cam_names), scored(cam_names, written)
    r = mod.resolve_run(ns(algo="MVOpt"), cam_names, written)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3MVOpt", "all", list(range(n)))
    assert r["xview_kwargs"] == {} and r["lineage"] == "legacy" and r["xview_on"] is False
    assert r["xview_window"] is None and r["scored_idx"] == sc
    r = mod.resolve_run(ns(algo="OneStageNew"), cam_names, written)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3OneStageNew", "all", list(range(n)))
    r = mod.resolve_run(ns(algo="OneStage"), cam_names, written)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3OneStage", "written", sc)
    # explicit --track-cams / --out behave exactly as runMVSeg.py:269 and :308-310 did
    r = mod.resolve_run(ns(algo="MVOpt", track_cams="written"), cam_names, written)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3MVOpt", "written", sc)
    r = mod.resolve_run(ns(algo="OneStage", track_cams="all"), cam_names, written)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3OneStage", "all", list(range(n)))
    r = mod.resolve_run(ns(algo="MVOpt", out="SegMaskSam3MVOpt_guard"), cam_names, written)
    assert (r["out_name"], r["track_mode"]) == ("SegMaskSam3MVOpt_guard", "all")
    r = mod.resolve_run(ns(algo="OneStageNew", out="X"), cam_names, written)
    assert r["out_name"] == "X"


@pytest.mark.parametrize("ds", sorted(CANONICAL_K))
def test_closure_session_count(ds):
    cam_names, written = cams(ds)
    K = CANONICAL_K[ds]
    assert max(scored(cam_names, written)) + 1 == K
    r = mod.resolve_run(ns(algo="MVOpt", xview_window=4), cam_names, written)
    assert (r["track_mode"], r["track_idx"]) == ("closure", list(range(K)))
    r = mod.resolve_run(ns(algo="OneStage", track_cams="closure"), cam_names, written)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3OneStageC", "closure", list(range(K)))
    assert r["xview_kwargs"] == {} and r["lineage"] == "legacy"


def test_xview_defaults():
    cam_names, written = cams("Welder")
    r = mod.resolve_run(ns(xview_window=0), cam_names, written)
    assert r["out_name"] == "SegMaskSam3XW0" and r["track_mode"] == "closure"
    assert r["xview_kwargs"] == dict(cross_view_window=0, cross_view_hygiene=True)
    assert r["lineage"] == "XW0" and r["xview_on"] is True and r["xview_window"] == 0
    assert r["track_idx"] == list(range(4))
    r = mod.resolve_run(ns(xview_hygiene=True), cam_names, written)
    assert r["out_name"] == "SegMaskSam3XW4" and r["lineage"] == "XW4"
    assert r["xview_kwargs"] == dict(cross_view_window=4, cross_view_hygiene=True)
    r = mod.resolve_run(ns(xview_window=2, xview_hygiene=True), cam_names, written)
    assert r["out_name"] == "SegMaskSam3XW2" and r["xview_window"] == 2
    for W in range(0, 7):
        r = mod.resolve_run(ns(xview_window=W), cam_names, written)
        assert r["out_name"] == f"SegMaskSam3XW{W}" and r["track_mode"] == "closure"


def test_xview_all_default_name_f3():
    cam_names, written = cams("Welder")
    n = len(cam_names)
    r = mod.resolve_run(ns(xview_window=4, track_cams="all"), cam_names, written)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3XW4all", "all", list(range(n)))
    r = mod.resolve_run(ns(xview_window=0, track_cams="all"), cam_names, written)
    assert r["out_name"] == "SegMaskSam3XW0all"
    r = mod.resolve_run(ns(xview_window=4, track_cams="all", out="SegMaskSam3XW4all"), cam_names, written)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3XW4all", "all", list(range(n)))
    r = mod.resolve_run(ns(xview_window=4, track_cams="all", out="SegMaskSam3XWmine"), cam_names, written)
    assert r["out_name"] == "SegMaskSam3XWmine"                     # explicit --out wins
    r = mod.resolve_run(ns(xview_window=4, track_cams="closure"), cam_names, written)
    assert r["out_name"] == "SegMaskSam3XW4"


def test_refusals(monkeypatch):
    cam_names, written = cams("Blocks")
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="OneStage", xview_window=2), cam_names, written)
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="OneStageNew", xview_hygiene=True), cam_names, written)
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="MVOpt", track_cams="closure"), cam_names, written)
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="OneStageNew", track_cams="closure"), cam_names, written)
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(xview_window=4, track_cams="written"), cam_names, written)
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(xview_window=4, out="SegMaskSam3MVOpt"), cam_names, written)
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(xview_window=4, out="SegMaskSam3OneStageNew"), cam_names, written)
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(xview_window=4, out="SegMaskSam3OneStageC"), cam_names, written)
    monkeypatch.setenv("SCSAM3_XVIEW_WINDOW", "2")
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(), cam_names, written)
    r = mod.resolve_run(ns(xview_window=2), cam_names, written)     # a flag makes it explicit
    assert r["lineage"] == "XW2"
    monkeypatch.delenv("SCSAM3_XVIEW_WINDOW")
    monkeypatch.setenv("SCSAM3_XVIEW_HYGIENE", "1")
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="OneStage"), cam_names, written)
    monkeypatch.setenv("SCSAM3_XVIEW_HYGIENE", "   ")                 # blank is not "set"
    assert mod.resolve_run(ns(), cam_names, written)["lineage"] == "legacy"


def test_mirror_guard_f2():
    cam_names, written = cams("Blocks")
    for algo in ("MVOpt", "OneStageNew", "OneStage"):
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(algo=algo, out="SegMaskSam3XW0"), cam_names, written)
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(algo=algo, out="SegMaskSam3XW4all"), cam_names, written)
    assert mod.resolve_run(ns(algo="MVOpt", out="SegMaskSam3X_legacy"), cam_names, written)["out_name"] == "SegMaskSam3X_legacy"
    assert mod.resolve_run(ns(algo="MVOpt", out="MySegMaskSam3XW4"), cam_names, written)["out_name"] == "MySegMaskSam3XW4"


def test_empty_cam_list_f1():
    cam_names, _ = cams("Blocks")
    n = len(cam_names)
    r = mod.resolve_run(ns(algo="MVOpt"), cam_names, [])
    assert (r["track_mode"], r["track_idx"], r["scored_idx"]) == ("all", list(range(n)), [])
    r = mod.resolve_run(ns(algo="OneStage"), cam_names, [])
    assert (r["track_mode"], r["track_idx"]) == ("written", [])
    r = mod.resolve_run(ns(algo="MVOpt", track_cams="written"), cam_names, [])
    assert r["track_idx"] == []
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="MVOpt", xview_window=4), cam_names, [])            # closure default
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="OneStage", track_cams="closure"), cam_names, [])
    r = mod.resolve_run(ns(algo="MVOpt", xview_window=4, track_cams="all"), cam_names, [])
    assert r["track_idx"] == list(range(n)) and r["out_name"] == "SegMaskSam3XW4all"
    # a written camera outside the perms list: scored_idx empty even with a cam_list
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="MVOpt", xview_window=0), cam_names, ["nonexistent"])


def test_parse_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks", "--algo", "MVOpt", "--xview-window", "3",
                                      "--track-cams", "closure"])
    a = mod.parse_args()
    assert (a.xview_window, a.track_cams, a.xview_hygiene) == (3, "closure", False)
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks", "--xview-window", "7"])
    with pytest.raises(SystemExit):
        mod.parse_args()
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks", "--xview-window", "-1"])
    with pytest.raises(SystemExit):
        mod.parse_args()
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks"])
    a = mod.parse_args()
    assert a.xview_window is None and a.xview_hygiene is False and a.track_cams is None
    assert mod.XVIEW_MAX_WINDOW == 6 and mod.XVIEW_ALGOS == ("MVOpt",)

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
    d = dict(algo="MVOpt", xview_window=None, xview_hygiene=False, track_cams=None, out=None,
             xview_mode=None, xview_gate=False, xview_ptr=False, xview_tpos_shift=None)
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


# ------------------------------------------------------------------ P4: --xview-mode
# (N, K = max(scored)+1, B/C reach = min(N, K+20), E reach = min(N, K+40)) at W=1, nf=21
MODE_REACH = {"AlexaMeadeExhibit": (45, 4, 24, 44), "AlexaMeadeFacePaint": (46, 9, 29, 46),
              "Barn": (15, 11, 15, 15), "Blocks": (10, 10, 10, 10), "Breakfast": (15, 10, 15, 15),
              "Carpark": (9, 9, 9, 9), "CoffeeMartini": (18, 15, 18, 18), "Dog": (41, 4, 24, 41),
              "Fencing": (10, 10, 10, 10), "FlameSteak": (21, 17, 21, 21), "Frog": (13, 10, 13, 13),
              "MATF": (10, 10, 10, 10), "Painter": (16, 16, 16, 16), "PoznanStreet": (9, 9, 9, 9),
              "Welder": (46, 4, 24, 44)}


@pytest.mark.parametrize("ds", sorted(MODE_REACH))
def test_mode_table(ds):
    cam_names, written = cams(ds)
    N, K, BC, E = MODE_REACH[ds]
    assert len(cam_names) == N and K == CANONICAL_K[ds]
    for M in "BCDE":
        r = mod.resolve_run(ns(xview_window=1, xview_mode=M), cam_names, written, num_frame=21)
        assert r["out_name"] == f"SegMaskSam3XW1{M}" and r["lineage"] == f"XW1{M}"
        assert r["xview_kwargs"] == dict(cross_view_window=1, cross_view_hygiene=True, cross_view_mode=M)
        assert r["two_pass"] == (M == "E") and (r["xview_mode"], r["xview_mode_eff"]) == (M, M)
        reach = {"B": BC, "C": BC, "D": K, "E": E}[M]
        assert r["track_mode"] == "closure" and r["track_idx"] == list(range(reach))
        assert r["closure_reach"] == reach and r["xview_on"] is True and r["xview_window"] == 1
        r4 = mod.resolve_run(ns(xview_window=4, xview_mode=M), cam_names, written, num_frame=21)
        assert r4["out_name"] == f"SegMaskSam3XW4{M}" and r4["lineage"] == f"XW4{M}"
        assert len(r4["track_idx"]) == (N if M in "BCE" else K)
    r = mod.resolve_run(ns(xview_window=1, xview_mode="A"), cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XW1" and r["lineage"] == "XW1" and r["track_idx"] == list(range(K))
    assert r["xview_kwargs"] == dict(cross_view_window=1, cross_view_hygiene=True, cross_view_mode="A")
    assert r["two_pass"] is False and r["xview_mode_eff"] == "A" and r["closure_reach"] == K
    # no mode: the Phase 2 resolution, the new keys at their neutral values
    r = mod.resolve_run(ns(xview_window=1), cam_names, written, num_frame=21)
    assert r["xview_kwargs"] == dict(cross_view_window=1, cross_view_hygiene=True)
    assert (r["xview_mode"], r["xview_mode_eff"], r["two_pass"], r["closure_reach"]) == (None, "A", False, K)
    assert r == mod.resolve_run(ns(xview_window=1), cam_names, written)          # num_frame unused
    r = mod.resolve_run(ns(), cam_names, written)
    assert (r["xview_mode"], r["xview_mode_eff"], r["two_pass"], r["closure_reach"]) == (None, None, False, None)


def test_mode_all_and_refusals():
    cam_names, written = cams("Welder")
    n = len(cam_names)
    r = mod.resolve_run(ns(xview_window=1, xview_mode="B", track_cams="all"), cam_names, written, num_frame=21)
    assert (r["out_name"], r["track_mode"], r["track_idx"]) == ("SegMaskSam3XW1Ball", "all", list(range(n)))
    assert r["closure_reach"] is None and r["lineage"] == "XW1B"
    r = mod.resolve_run(ns(xview_window=1, xview_mode="E", track_cams="all", out="SegMaskSam3XWmine"),
                        cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XWmine" and r["two_pass"] is True
    with pytest.raises(SystemExit):                                   # mode without an XW flag
        mod.resolve_run(ns(xview_mode="B"), cam_names, written, num_frame=21)
    for M in "BCDE":                                                  # W=0 reads no neighbour
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(xview_window=0, xview_mode=M), cam_names, written, num_frame=21)
    assert mod.resolve_run(ns(xview_window=0, xview_mode="A"), cam_names, written)["out_name"] == "SegMaskSam3XW0"
    for algo in ("OneStage", "OneStageNew"):                          # MVOpt only
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(algo=algo, xview_window=1, xview_mode="B"), cam_names, written, num_frame=21)
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(algo=algo, xview_mode="B"), cam_names, written, num_frame=21)
    with pytest.raises(SystemExit):                                   # written renumbers neighbours
        mod.resolve_run(ns(xview_window=1, xview_mode="E", track_cams="written"), cam_names, written, num_frame=21)
    for M in "BCE":                                                   # the cone needs num_frame
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(xview_window=1, xview_mode=M), cam_names, written)
        r = mod.resolve_run(ns(xview_window=1, xview_mode=M, track_cams="all"), cam_names, written)
        assert r["track_idx"] == list(range(n))                       # all: no cone, no num_frame
    for M in "AD":
        assert mod.resolve_run(ns(xview_window=1, xview_mode=M), cam_names, written)["track_idx"] == list(range(4))
    with pytest.raises(SystemExit):                                   # existing legacy-folder guard
        mod.resolve_run(ns(xview_window=1, xview_mode="B", out="SegMaskSam3MVOpt"), cam_names, written, num_frame=21)
    r = mod.resolve_run(ns(algo="OneStage", track_cams="closure"), cam_names, written, num_frame=21)
    assert r["track_idx"] == list(range(4)) and r["closure_reach"] == 4 and r["xview_mode_eff"] is None


def test_closure_reach():
    f = mod.closure_reach
    assert f("A", 1, 3, 45, 21) == 4 and f("B", 1, 3, 45, 21) == 24 and f("E", 1, 3, 45, 21) == 44
    assert f("E", 1, 3, 41, 21) == 41 and f("B", 4, 3, 46, 21) == 46 and f("D", 1, 9, 10, 21) == 10
    assert f("C", 1, 3, 45, 21) == 24 and f("C", 2, 3, 45, 11) == 24 and f("E", 2, 3, 46, 11) == 44
    assert f("A", None, 3, 45, None) == 4 and f("D", 1, 3, 45, None) == 4    # growth 0: no num_frame
    with pytest.raises(SystemExit):
        f("B", 1, 3, 45, None)
    assert mod.XVIEW_CLOSURE_GROWTH == {"A": 0, "B": 1, "C": 1, "D": 0, "E": 2}
    assert mod.XVIEW_TWO_PASS == ("E",) and mod.XVIEW_LEGACY_MODE == "A"


def test_parse_args_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks", "--xview-window", "1", "--xview-mode", "b"])
    with pytest.raises(SystemExit):
        mod.parse_args()
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks", "--xview-window", "1", "--xview-mode", "E"])
    a = mod.parse_args()
    assert (a.xview_mode, a.xview_window, a.xview_hygiene) == ("E", 1, False)
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks"])
    assert mod.parse_args().xview_mode is None
    assert mod.XVIEW_MODES == ("A", "B", "C", "D", "E")


# ------------------------------------------------- P12: --xview-gate/--xview-ptr/-tpos-shift
# (flags to set on the namespace, folder/lineage suffix, the kwargs they add)
KNOB_TABLE = [(dict(xview_gate=True), "G", dict(cross_view_gate=True)),
              (dict(xview_ptr=True), "P", dict(cross_view_ptr=True)),
              (dict(xview_gate=True, xview_ptr=True), "GP",
               dict(cross_view_gate=True, cross_view_ptr=True)),
              (dict(xview_tpos_shift=2), "S2", dict(cross_view_tpos_shift=2)),
              (dict(xview_tpos_shift=4), "S4", dict(cross_view_tpos_shift=4)),
              (dict(xview_gate=True, xview_ptr=True, xview_tpos_shift=2), "GPS2",
               dict(cross_view_gate=True, cross_view_ptr=True, cross_view_tpos_shift=2))]


@pytest.mark.parametrize("ds", sorted(CANONICAL_K))
def test_knob_table(ds):
    cam_names, written = cams(ds)
    K = CANONICAL_K[ds]
    for flags, suffix, kwargs in KNOB_TABLE:
        r = mod.resolve_run(ns(xview_window=1, **flags), cam_names, written, num_frame=21)
        assert r["out_name"] == f"SegMaskSam3XW1{suffix}", (ds, suffix)
        assert r["lineage"] == f"XW1{suffix}", (ds, suffix)
        assert r["xview_kwargs"] == dict(cross_view_window=1, cross_view_hygiene=True, **kwargs)
        assert (r["track_mode"], r["track_idx"]) == ("closure", list(range(K)))   # cone unchanged
        assert r["closure_reach"] == K and r["two_pass"] is False
        assert (r["xview_gate"], r["xview_ptr"], r["xview_tpos_shift"]) == (
            flags.get("xview_gate", False), flags.get("xview_ptr", False),
            flags.get("xview_tpos_shift", None) or 0)
        # an explicit --out still wins, and the mode letter goes before the knob suffix
        assert mod.resolve_run(ns(xview_window=1, out="SegMaskSam3XWmine", **flags),
                               cam_names, written, num_frame=21)["out_name"] == "SegMaskSam3XWmine"
    r = mod.resolve_run(ns(xview_window=1, xview_mode="C", xview_gate=True, xview_ptr=True),
                        cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XW1CGP" and r["lineage"] == "XW1CGP"
    assert r["xview_kwargs"] == dict(cross_view_window=1, cross_view_hygiene=True,
                                     cross_view_mode="C", cross_view_gate=True, cross_view_ptr=True)
    r = mod.resolve_run(ns(xview_window=1, xview_gate=True, xview_ptr=True, track_cams="all"),
                        cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XW1GPall" and r["lineage"] == "XW1GP"
    r = mod.resolve_run(ns(xview_window=4, xview_tpos_shift=2), cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XW4S2" and r["lineage"] == "XW4S2"
    r = mod.resolve_run(ns(xview_window=2, xview_tpos_shift=4), cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XW2S4"
    # no knob: the P4 resolution, the three new keys at their neutral values
    r = mod.resolve_run(ns(xview_window=1), cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XW1"
    assert (r["xview_gate"], r["xview_ptr"], r["xview_tpos_shift"]) == (False, False, 0)
    r = mod.resolve_run(ns(), cam_names, written)
    assert (r["xview_gate"], r["xview_ptr"], r["xview_tpos_shift"]) == (False, False, 0)


def test_knob_refusals():
    cam_names, written = cams("Blocks")
    for flags, _, _ in KNOB_TABLE:
        with pytest.raises(SystemExit):                       # knobs need an XW flag
            mod.resolve_run(ns(**flags), cam_names, written, num_frame=21)
        with pytest.raises(SystemExit):                       # W=0 has no neighbour token
            mod.resolve_run(ns(xview_window=0, **flags), cam_names, written, num_frame=21)
        for algo in ("OneStage", "OneStageNew"):              # MVOpt only
            with pytest.raises(SystemExit):
                mod.resolve_run(ns(algo=algo, xview_window=1, **flags), cam_names, written, num_frame=21)
            with pytest.raises(SystemExit):
                mod.resolve_run(ns(algo=algo, **flags), cam_names, written, num_frame=21)
        with pytest.raises(SystemExit):                       # written renumbers neighbours
            mod.resolve_run(ns(xview_window=1, track_cams="written", **flags), cam_names, written,
                            num_frame=21)
        with pytest.raises(SystemExit):                       # existing legacy-folder guard
            mod.resolve_run(ns(xview_window=1, out="SegMaskSam3MVOpt", **flags), cam_names, written,
                            num_frame=21)
    for W, s in ((3, 4), (6, 1), (4, 3), (2, 5)):             # W + S > 6 (S=6 is argparse's job)
        with pytest.raises(SystemExit):
            mod.resolve_run(ns(xview_window=W, xview_tpos_shift=s), cam_names, written, num_frame=21)
    for W, s in ((1, 5), (2, 4), (4, 2)):                     # the bound itself is allowed
        assert mod.resolve_run(ns(xview_window=W, xview_tpos_shift=s), cam_names, written,
                               num_frame=21)["lineage"] == f"XW{W}S{s}"
    # a namespace without the three attributes (a legacy caller) resolves as before
    legacy = argparse.Namespace(algo="MVOpt", xview_window=1, xview_hygiene=False,
                                track_cams=None, out=None, xview_mode=None)
    r = mod.resolve_run(legacy, cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XW1" and r["lineage"] == "XW1"
    assert (r["xview_gate"], r["xview_ptr"], r["xview_tpos_shift"]) == (False, False, 0)


def test_parse_args_knobs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks", "--xview-window", "1",
                                      "--xview-gate", "--xview-ptr"])
    a = mod.parse_args()
    assert (a.xview_gate, a.xview_ptr, a.xview_tpos_shift) == (True, True, None)
    for bad in ("0", "6", "-1"):
        monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks", "--xview-window", "1",
                                          "--xview-tpos-shift", bad])
        with pytest.raises(SystemExit):
            mod.parse_args()
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks", "--xview-window", "1",
                                      "--xview-tpos-shift", "2"])
    a = mod.parse_args()
    assert a.xview_tpos_shift == 2 and a.xview_gate is False
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Blocks"])
    a = mod.parse_args()
    assert (a.xview_gate, a.xview_ptr, a.xview_tpos_shift) == (False, False, None)
    assert mod.XVIEW_TPOS_SHIFTS == (1, 2, 3, 4, 5)

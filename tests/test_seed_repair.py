"""T9: seed_repair.py (torch-free) and the --repair-seeds runner plumbing.

The rules under test are the ones pre-registered in docs/phase5-seed-repair-prereg.md
section 1 as amended by section 8 (after the Welder smoke test, before the sweep).  The
census golden ties the implementation to the proxy numbers written into that amendment.
"""
import argparse
import json
import os
import sys

import pytest

from conftest import REPO, load_runmvseg

import seed_repair as sr

CENSUS = os.path.join(REPO, "Data", "MVSeg", "seed_census_muvod.json")


# ------------------------------------------------------------------ rule 3: threshold
def test_threshold_is_relative_to_the_prompt():
    assert sr.threshold(5983) == pytest.approx(598.3)
    assert sr.threshold(100) == 64                     # never below the pixel floor
    assert sr.threshold(640) == 64
    assert sr.threshold(641) == pytest.approx(64.1)


# ------------------------------------------------------------------ rules 1-3: detect
# Welder, views 0..3 = camera_0001 (reference) .. camera_0004, plus a far view 7 that
# holds a blob for obj 3 and nothing for the rest -- the shape the smoke test showed.
SEEDS = {
    0: {1: 5963, 2: 6745, 3: 3158},
    1: {1: 2197, 2: 1341, 3: 3000},
    2: {1: 6911, 2: 6496, 3: 24949},
    3: {1: 299, 2: 181, 3: 2633},
    7: {1: 0, 2: 0, 3: 1581250},
}
REF = {1: 5983, 2: 6766, 3: 3211}
TRACKED = [0, 1, 2, 3]


def test_detect_flags_the_welder_pairs():
    flags = sr.detect(SEEDS, REF, ref_view=0, views=TRACKED)
    assert [(f.view, f.obj, f.area) for f in flags] == [(3, 1, 299), (3, 2, 181)]
    assert flags[0].threshold == pytest.approx(598.3)
    assert flags[1].threshold == pytest.approx(676.6)


def test_detect_scope_excludes_untracked_views():
    all_views = sr.detect(SEEDS, REF, 0)
    assert (7, 1) in {(f.view, f.obj) for f in all_views}       # a blank far view gets flagged
    scoped = sr.detect(SEEDS, REF, 0, views=TRACKED)
    assert scoped and all(f.view in TRACKED for f in scoped)


def test_detect_never_flags_the_reference_view():
    seeds = dict(SEEDS)
    seeds[0] = {1: 1, 2: 1, 3: 1}
    assert [(f.view, f.obj) for f in sr.detect(seeds, REF, 0, views=TRACKED)] == [(3, 1), (3, 2)]


def test_detect_skips_objects_without_a_real_prompt():
    ref = dict(REF)
    ref[1] = 63
    assert [(f.view, f.obj) for f in sr.detect(SEEDS, ref, 0, views=TRACKED)] == [(3, 2)]
    ref = {2: 6766, 3: 3211}
    assert [(f.view, f.obj) for f in sr.detect(SEEDS, ref, 0, views=TRACKED)] == [(3, 2)]


def test_detect_treats_a_missing_object_as_zero_area():
    seeds = {0: SEEDS[0], 1: {1: 5900, 3: 3000}}
    assert [(f.view, f.obj, f.area) for f in sr.detect(seeds, REF, 0)] == [(1, 2, 0)]


def test_a_blob_elsewhere_does_not_move_the_bar():
    # a per-view-median rule would have set view 3's bar from the 1.58M px blob; the
    # relative rule keeps obj 3's bar at 0.10 x 3211 and leaves its 2633 px seed alone
    seeds = {0: SEEDS[0], 3: {1: 299, 2: 181, 3: 2633, 4: 1581250}}
    ref = dict(REF)
    ref[4] = 3000
    assert [(f.view, f.obj) for f in sr.detect(seeds, ref, 0)] == [(3, 1), (3, 2)]


# ------------------------------------------------------------------ rule 4: donor
def test_donor_band():
    assert sr.in_band(6911, 5983)
    assert not sr.in_band(1581250, 5983)               # blob: above 4x
    assert not sr.in_band(1341, 6766)                  # below 0.25x
    assert sr.in_band(64, 100) and not sr.in_band(63, 100)


def test_pick_donor_largest_in_band_within_window():
    donors = [1, 2, 3]                                 # tracked minus the reference
    assert sr.pick_donor(SEEDS, 3, 1, REF[1], candidates=donors) == (2, 6911)
    assert sr.pick_donor(SEEDS, 3, 2, REF[2], candidates=donors) == (2, 6496)   # view 1 is below band


def test_pick_donor_never_takes_the_blob():
    seeds = {0: {3: 3158}, 3: {3: 30}, 7: {3: 1581250}, 5: {3: 4000}}
    assert sr.pick_donor(seeds, 3, 3, 3211) == (5, 4000)
    seeds[5][3] = 0
    assert sr.pick_donor(seeds, 3, 3, 3211) == (0, 3158)     # reference allowed when not excluded
    assert sr.pick_donor(seeds, 3, 3, 3211, candidates=[5, 7]) is None


def test_pick_donor_window_then_fallback_then_none():
    seeds = {0: {1: 6000}, 9: {1: 4}, 12: {1: 5000}}
    assert sr.pick_donor(seeds, 9, 1, 6000) == (12, 5000)
    seeds = {0: {1: 6000}, 9: {1: 4}, 17: {1: 5000}}
    assert sr.pick_donor(seeds, 9, 1, 6000) == (17, 5000)
    seeds = {0: {1: 6000}, 9: {1: 4}, 18: {1: 5000}}
    assert sr.pick_donor(seeds, 9, 1, 6000) is None


def test_pick_donor_excludes_and_tie_breaks():
    seeds = {0: {1: 5000}, 1: {1: 5000}, 2: {1: 4}, 3: {1: 5000}, 4: {1: 5000}}
    assert sr.pick_donor(seeds, 2, 1, 6000) == (1, 5000)
    assert sr.pick_donor(seeds, 2, 1, 6000, exclude={1}) == (3, 5000)
    assert sr.pick_donor(seeds, 2, 1, 6000, exclude={1, 3}) == (0, 5000)


# ------------------------------------------------------------------ plan / rounds
def test_plan_round_one_and_two():
    donors = [1, 2, 3]
    repairs, unrep = sr.plan(SEEDS, REF, 0, views=TRACKED, donors=donors)
    assert [(r.view, r.obj, r.donor) for r in repairs] == [(3, 1, 2), (3, 2, 2)]
    assert unrep == []
    used = {(3, 1): {2}, (3, 2): {2}}
    repairs, unrep = sr.plan(SEEDS, REF, 0, views=TRACKED, donors=donors, used=used)
    assert [(r.view, r.obj, r.donor) for r in repairs] == [(3, 1, 1)]        # 2197 in band
    assert [(u.view, u.obj) for u in unrep] == [(3, 2)]                      # 1341 below band


def test_plan_flags_of_the_same_object_never_donate_to_each_other():
    seeds = {0: {1: 6000}, 1: {1: 5}, 2: {1: 6}, 3: {1: 5900}}
    repairs, unrep = sr.plan(seeds, {1: 6000}, 0, donors=[1, 2, 3])
    assert [(r.view, r.donor) for r in repairs] == [(1, 3), (2, 3)]


def test_plan_reports_unrepairable():
    seeds = {0: {1: 6000, 2: 6000}, 10: {1: 5, 2: 5}, 20: {1: 6000, 2: 6000}}
    repairs, unrep = sr.plan(seeds, {1: 6000, 2: 6000}, 0)
    assert repairs == []
    assert [(u.view, u.obj) for u in unrep] == [(10, 1), (10, 2)]


def test_as_records():
    r = sr.Repair(3, 1, 299, 598.3, 2, 6911)
    assert sr.as_records([r]) == [dict(view=3, obj=1, area=299, threshold=598.3, donor=2,
                                       donor_area=6911)]


# ------------------------------------------------------------------ census golden
def _proxy(d):
    """Per scene: (camera names, seed dict, ref areas, ref view, rows) from the census."""
    rows = [r for r in d["rows"] if r["method"] == "SegMaskSam3XW1GPS4M"]
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["dataset"], []).append(r)
    for scene, rs in by_scene.items():
        cams = sorted({r["camera"] for r in rs})
        idx = {c: i for i, c in enumerate(cams)}
        seeds = {idx[c]: {} for c in cams}
        ref_areas = {}
        for r in rs:
            seeds[idx[r["camera"]]][r["obj"]] = r["pred_area"]
            if r["reachable"]:
                ref_areas[r["obj"]] = r["ref_gt_area"]
        yield scene, cams, seeds, ref_areas, idx[d["datasets"][scene]["ref_cam"]], rs


def test_census_proxy_detect_matches_the_amendment():
    """prereg section 8.4, rules 1-3 only: 26 scoreable flags, 14 of 15 true degenerates,
    the miss is MATF S1_CAM_1 obj 26 (misaligned, out of scope by design)."""
    if not os.path.isfile(CENSUS):
        pytest.skip("census not generated")
    d = json.load(open(CENSUS))
    deg = set(d["degenerate_classes"])
    total, hit, truth, missed = 0, 0, 0, []
    for scene, cams, seeds, ref_areas, ref_view, rs in _proxy(d):
        rowset = {(r["camera"], r["obj"]) for r in rs}
        got = {(cams[f.view], f.obj) for f in sr.detect(seeds, ref_areas, ref_view)
               if (cams[f.view], f.obj) in rowset}
        total += len(got)
        for r in rs:
            if r["cls"] in deg:
                truth += 1
                if (r["camera"], r["obj"]) in got:
                    hit += 1
                else:
                    missed.append((scene, r["camera"], r["obj"]))
    assert total == 26
    assert (hit, truth) == (14, 15)
    assert missed == [("MATF", "S1_CAM_1", 26)]


def test_census_proxy_plan_is_a_partition_of_detect_with_in_band_non_reference_donors():
    """Rule 4 on the proxy: every flag is either repaired or unrepairable, no donor is the
    reference or another flagged view of the same object, and every donor is in band.
    The count itself is not pinned: the proxy has two candidate donors per scene, the
    real pool is the tracked set (prereg section 8.3)."""
    if not os.path.isfile(CENSUS):
        pytest.skip("census not generated")
    d = json.load(open(CENSUS))
    for scene, cams, seeds, ref_areas, ref_view, rs in _proxy(d):
        donors = [v for v in seeds if v != ref_view]
        flags = sr.detect(seeds, ref_areas, ref_view)
        repairs, unrep = sr.plan(seeds, ref_areas, ref_view, donors=donors,
                                 window=10**6, fallback=10**6)
        assert {(x.view, x.obj) for x in repairs} | {(x.view, x.obj) for x in unrep} == \
               {(f.view, f.obj) for f in flags}
        assert not ({(x.view, x.obj) for x in repairs} & {(x.view, x.obj) for x in unrep})
        flagged = {}
        for f in flags:
            flagged.setdefault(f.obj, set()).add(f.view)
        for r in repairs:
            assert r.donor != ref_view and r.donor not in flagged[r.obj]
            assert sr.in_band(r.donor_area, ref_areas[r.obj])


# ------------------------------------------------------------------ runner plumbing
mod = load_runmvseg()
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")


def ns(**kw):
    d = dict(algo="MVOpt", xview_window=None, xview_hygiene=False, track_cams=None, out=None,
             xview_mode=None, xview_gate=False, xview_ptr=False, xview_tpos_shift=None,
             ref_cam=None, repair_seeds=False)
    d.update(kw)
    return argparse.Namespace(**d)


def cams(name):
    c = mod.load_config(CONFIG, name)
    names = [mod.cam_name(x, c["prefix"], c["prefix1"]) for x in c["perms"]]
    return names, [mod.cam_name(x, c["prefix"], c["prefix1"]) for x in c["cam_list"]]


def test_repair_suffix_follows_the_reference_suffix():
    cam_names, written = cams("Welder")
    base = ns(xview_window=1, xview_gate=True, xview_ptr=True, xview_tpos_shift=4)
    r = mod.resolve_run(base, cam_names, written, num_frame=21, ref_suffix="M")
    assert r["out_name"] == "SegMaskSam3XW1GPS4M" and r["repair_seeds"] is False
    r = mod.resolve_run(ns(xview_window=1, xview_gate=True, xview_ptr=True, xview_tpos_shift=4,
                           repair_seeds=True), cam_names, written, num_frame=21, ref_suffix="M")
    assert r["out_name"] == "SegMaskSam3XW1GPS4MRp" and r["repair_seeds"] is True
    r = mod.resolve_run(ns(xview_window=0, repair_seeds=True), cam_names, written, num_frame=21)
    assert r["out_name"] == "SegMaskSam3XW0Rp"
    r = mod.resolve_run(ns(xview_window=1, out="Whatever", repair_seeds=True), cam_names, written,
                        num_frame=21, ref_suffix="M")
    assert r["out_name"] == "Whatever" and r["repair_seeds"] is True


def test_repair_refused_without_a_spatial_pass():
    cam_names, written = cams("Welder")
    with pytest.raises(SystemExit):
        mod.resolve_run(ns(algo="OneStage", repair_seeds=True), cam_names, written)


def test_parse_args_repair_seeds(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Welder"])
    assert mod.parse_args().repair_seeds is False
    monkeypatch.setattr(sys, "argv", ["runMVSeg.py", "Welder", "--repair-seeds"])
    assert mod.parse_args().repair_seeds is True

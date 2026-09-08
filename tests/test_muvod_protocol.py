"""T7: eval/report_jf.py --muvod, the MUVOD evaluation protocol.

MUVOD (arXiv:2507.07519) is the benchmark our data comes from, so the outward-facing
numbers use its protocol, not ours: score the three annotated cameras, average the
cameras into a scene score, average the scenes; "basic" keeps only the objects visible
in c_ini's reference frame, "complete" keeps them all.  These tests pin the object
filter, the aggregation, the guard rails, and the one consistency check the published
tables give us for free.
"""
import json
import math
import os
import subprocess
import sys

import pytest

from conftest import REPO, load_report_jf

mod = load_report_jf()
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")


# --------------------------------------------------------------- synthetic raw file
def entry(dataset, cam, method, objects, j, f, seed_ids, ref_cam, n_frames=4):
    """One eval_jf.py schema-3 entry with constant per-frame scores."""
    n = len(objects)
    return {
        "dataset": dataset, "camera": cam, "method": method,
        "n_frames": n_frames, "objects": list(objects),
        "meta": {"h": 20, "w": 20, "bound_pix": 1,
                 "frames": list(range(n_frames)), "start_frame": 0, "n_views": 3,
                 "view_index": ["a", "b", "c"].index(cam), "seed_ids": seed_ids[cam],
                 "ref": {r: {"cam": ref_cam[r], "view_index": 0,
                             "seed_ids": seed_ids[ref_cam[r]], "max_id": 9}
                         for r in ("maxid", "count", "muvod")}},
        "result": {
            "J_all": [j[o] for o in objects], "F_all": [f[o] for o in objects],
            "J_inner": [j[o] for o in objects], "F_inner": [f[o] for o in objects],
            "missing_files": 0,
            "per_frame": {"J": [[j[o]] * n_frames for o in objects],
                          "F": [[f[o]] * n_frames for o in objects],
                          "gt_area": [[10] * n_frames for _ in objects],
                          "pred_area": [[10] * n_frames for _ in objects],
                          "pred_missing": [[False] * n_frames for _ in objects]},
        },
    }


@pytest.fixture(scope="module")
def store():
    # one scene, three cameras.  c_ini is 'b'; the max-id rule would pick 'a'.
    # object 3 is missing from c_ini's reference frame and is scored 0 everywhere,
    # so basic and complete must differ, and by a predictable amount.
    seed = {"a": [1, 2, 3], "b": [1, 2], "c": [1, 2, 3]}
    ref = {"maxid": "a", "count": "a", "muvod": "b"}
    j = {1: 0.8, 2: 0.6, 3: 0.0}
    f = {1: 1.0, 2: 0.4, 3: 0.0}
    es = [entry("Fencing", cam, "M1", [1, 2, 3], j, f, seed, ref) for cam in "abc"]
    return mod.Store({(e["dataset"], e["camera"], e["method"]): e for e in es})


BASIC = mod.Cfg(**mod.MUVOD_CFG).replace(objects="basic")
COMPLETE = mod.Cfg(**mod.MUVOD_CFG)


def test_basic_drops_objects_outside_c_ini(store):
    # basic: objects 1 and 2 -> J = (0.8+0.6)/2 = 0.7, F = (1.0+0.4)/2 = 0.7
    ds, cs = mod.score_all(store, ["M1"], ["Fencing"], BASIC)
    for cam in "abc":
        assert cs[("Fencing", cam, "M1")] == pytest.approx((0.7, 0.7, 0.7))
    assert ds[("Fencing", "M1")] == pytest.approx((0.7, 0.7, 0.7))
    # complete: object 3 joins at 0 -> J = 1.4/3, F = 1.4/3
    ds, _ = mod.score_all(store, ["M1"], ["Fencing"], COMPLETE)
    assert ds[("Fencing", "M1")][1] == pytest.approx(1.4 / 3)
    assert ds[("Fencing", "M1")][0] == pytest.approx(1.4 / 3)


def test_object_filter_follows_the_reference_rule(store):
    assert store.object_filter("Fencing", COMPLETE) is None
    assert store.object_filter("Fencing", BASIC) == frozenset({1, 2})
    # under our own rule the reference is 'a', whose seed frame holds all three
    assert store.object_filter("Fencing", BASIC.replace(ref_rule="maxid")) == \
        frozenset({1, 2, 3})


def test_muvod_aggregation_is_per_camera_then_per_scene(store):
    # give one camera a different score and check the scene score is the camera mean,
    # not the object-pooled mean
    e = dict(store.E[("Fencing", "c", "M1")])
    r = json.loads(json.dumps(e["result"]))
    r["J_all"] = [0.2, 0.2, 0.0]
    r["F_all"] = [0.2, 0.2, 0.0]
    e = dict(e, result=r)
    swapped = dict(store.E)
    swapped[("Fencing", "c", "M1")] = e
    s2 = mod.Store(swapped)
    ds, cs = mod.score_all(s2, ["M1"], ["Fencing"], BASIC)
    assert cs[("Fencing", "c", "M1")][0] == pytest.approx(0.2)
    assert ds[("Fencing", "M1")][0] == pytest.approx((0.7 + 0.7 + 0.2) / 3)


def test_missing_reference_rule_is_a_clear_error(store):
    stripped = json.loads(json.dumps(store.E[("Fencing", "a", "M1")]))
    del stripped["meta"]["ref"]["muvod"]
    s2 = mod.Store({("Fencing", "a", "M1"): stripped})
    with pytest.raises(SystemExit) as exc:
        s2.object_filter("Fencing", BASIC)
    assert "schema 3" in str(exc.value)


def test_table_shape_and_baseline_column(store):
    t = mod.muvod_table(store, ["M1"], ["Fencing"], BASIC, "basic")
    assert t.columns == ["M1", mod.MUVOD_COL, f"M1 - {mod.MUVOD_COL}"]
    (label, cells), = t.rows
    assert label == "Fencing"
    assert cells[0] == pytest.approx(70.0)          # percent, not a fraction
    assert cells[1] == mod.MUVOD_BASELINE["basic"]["Fencing"] == 85.7
    assert cells[2] == pytest.approx(70.0 - 85.7)
    # a short table still shows what the paper's own global was over all 17
    assert ("MUVOD global (17 scenes)", [None, 79.4, None]) in t.foot
    assert t.foot[0][0] == "Global (1 scenes)"


def test_published_tables_are_transcribed_completely():
    for setting in ("basic", "complete"):
        assert sorted(mod.MUVOD_BASELINE[setting]) == sorted(mod.MUVOD_ORDER)
        assert len(mod.MUVOD_ORDER) == 17
        vals = list(mod.MUVOD_BASELINE[setting].values())
        assert mod.MUVOD_GLOBAL[setting] == pytest.approx(sum(vals) / len(vals), abs=0.05)
    # complete can never beat basic: it adds objects the model was never prompted with
    for d in mod.MUVOD_ORDER:
        assert mod.MUVOD_BASELINE["complete"][d] <= mod.MUVOD_BASELINE["basic"][d]


def test_c_ini_explains_where_the_published_tables_agree():
    """MUVOD's basic and complete scores are equal exactly when c_ini's reference frame
    already holds every labelled object.  Our own c_ini must reproduce that pattern in
    one direction: wherever the published scores differ, c_ini must be missing an
    object.  (The converse needs the ground truth and is checked by the report's
    'object sets per scene' table on real data.)"""
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    census = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")
    if not os.path.isfile(census):
        pytest.skip("object-set census not generated yet")
    counts = json.load(open(census))
    for d in mod.MUVOD_ORDER:
        equal = mod.MUVOD_BASELINE["basic"][d] == mod.MUVOD_BASELINE["complete"][d]
        holds_all = counts[d]["c_ini_holds_all"]
        assert holds_all == equal, (
            f"{d}: c_ini={counts[d]['c_ini']} holds all objects = {holds_all}, but "
            f"MUVOD's basic and complete scores are {'equal' if equal else 'different'}")


def test_muvod_refuses_flags_that_change_the_protocol():
    for extra in (["--aggregation", "sequence"], ["--objects", "basic"],
                  ["--ref-rule", "muvod"], ["--area-weighted"], ["--frames", "inner"]):
        p = subprocess.run([sys.executable, os.path.join(REPO, "eval", "report_jf.py"),
                            "--muvod"] + extra, capture_output=True, text=True)
        assert p.returncode != 0 and "fixes the protocol" in p.stderr + p.stdout, extra

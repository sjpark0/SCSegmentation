"""T10: MVSeed/seed_regression.py -- S1-E0 readings and the S1-F0 freeze (host, stdlib only).

A two-scene synthetic input (one plain ring, one hemisphere ring with `perms` gaps) pins
the pairing, the frames-1-20 regression, the F0/N0 split with its labels, and the view
distance; the CLI test checks the freeze files, their input digests and the write-once
guard.  The last test reads the real baseline against Data/MVSeg/jf_e4.json and must
reproduce docs/stage1-headroom.md (0.964 / 0.027 / 0.933; 45 = 7 + 38); it skips when
those files are not present.
"""
import hashlib
import json
import math
import os
import sys

import pytest

from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "MVSeed"))
import seed_regression as sr  # noqa: E402

METHOD = "M"
CFG = {
    # a plain ring: v0..v4, reference v2 (index 2)
    "Alpha": {"start_frame": 0, "cam_list": [0, 2, 4], "c_ini": 2, "num_frame": 21,
              "folder": "Alpha", "num_cam": 5, "start_cam": 0, "prefix": "v", "prefix1": 0},
    # a hemisphere name with gaps in the ring: camera_0005 is index 3, not 4
    "Welder": {"start_frame": 0, "cam_list": [1, 3, 5], "c_ini": 1, "num_frame": 21,
               "folder": "Welder", "perms": [1, 2, 3, 5, 8], "start_cam": 1,
               "prefix": "camera_", "prefix1": 4},
}


def row(cam, obj, J, gt, pred):
    return dict(cam=cam, obj=obj, J=J, F=J, gt=gt, pred=pred)


SEEDS = {"run1": {
    "Alpha": {"c_ini": "v2", "rows": [
        row("v2", 1, 0.95, 5000, 4900),        # reference
        row("v0", 1, 0.0, 1500, 0),            # F0 empty,      <2000,      pinhole, d2
        row("v0", 2, 0.3, 3000, 1200),         # F0 misplaced,  2000-10000, pinhole, d2; recovers at frame 0
        row("v0", 3, 1.0, 0, 0),               # GT empty: not in the jf object list, outside the population
        row("v4", 1, 0.9, 12000, 11000),       # N0, >=10000; fails only after re-prediction
        row("v4", 2, 0.5, 2000, 1900),         # N0 (J == tau stays N0), 2000-10000 (edge)
    ]},
    "Welder": {"c_ini": "camera_0001", "rows": [
        row("camera_0001", 1, 1.0, 100, 100),          # reference
        row("camera_0003", 1, 0.2, 1999, 500),         # F0 misplaced, <2000,   hemisphere, d2
        row("camera_0005", 1, 0.88, 9999, 9000),       # N0, 2000-10000 (edge), hemisphere, d3
        row("camera_0005", 2, 0.15, 10000, 2000),      # F0 misplaced, >=10000, hemisphere, d3
    ]},
}}


def entry(scene, cam, objs, frame0, later):
    """One eval_jf entry with 3 frames: J = [frame0, later, later] per object."""
    return {"dataset": scene, "camera": cam, "method": METHOD, "objects": objs,
            "result": {"per_frame": {"J": [[f, l, l] for f, l in zip(frame0, later)]}}}


# frames-1-20 mean = 0.5 * raw J + 0.1 on every population pair, so the raw fit is exact
JF = [
    entry("Alpha", "v2", [1], [0.95], [0.9]),
    entry("Alpha", "v0", [1, 2], [0.0, 0.55], [0.1, 0.25]),
    entry("Alpha", "v4", [2, 1], [0.5, 0.4], [0.35, 0.55]),      # object order differs from the rows
    entry("Welder", "camera_0001", [1], [1.0], [0.99]),
    entry("Welder", "camera_0003", [1], [0.2], [0.2]),
    entry("Welder", "camera_0005", [1, 2], [0.86, 0.15], [0.54, 0.175]),
    dict(entry("Alpha", "v0", [1, 2], [0.9, 0.9], [0.9, 0.9]), method="other"),  # other method: ignored
]


def current_layout(seeds=SEEDS, baseline=None):
    """SEEDS rewritten in score_seeds.py's current layout: runs/scenes, gt_px/pred_px rows."""
    def scenes(run):
        return {sc: dict(v, rows=[dict(scene=sc, cam=r["cam"], obj=r["obj"], J=r["J"], F=r["F"],
                                       gt_px=r["gt"], pred_px=r["pred"]) for r in v["rows"]])
                for sc, v in run.items()}
    doc = {"meta": {"populations": ["gt_present"]},
           "runs": {run: {"summary": {}, "scenes": scenes(v)} for run, v in seeds.items()}}
    if baseline:
        doc["baseline"] = {"run": baseline, "summary": {}, "scenes": scenes(seeds["run1"])}
    return doc


@pytest.fixture(scope="module")
def paired():
    pairs, unmatched = sr.pair(SEEDS["run1"], JF, METHOD, CFG)
    return pairs, unmatched


def by_key(pairs):
    return {(p.scene, p.cam, p.obj): p for p in pairs}


# ------------------------------------------------------------------ seeds file layouts
def test_seed_runs_reads_both_layouts():
    assert sr.seed_runs(SEEDS) == SEEDS                                   # legacy: {run: scenes}
    assert sr.seed_runs(dict(SEEDS, meta={"x": 1})) == SEEDS              # a meta block is not a run
    cur = sr.seed_runs(current_layout(baseline="ctrl"))
    assert sorted(cur) == ["ctrl", "run1"]
    assert cur["run1"]["Alpha"]["c_ini"] == "v2"
    assert [sr.row_px(r) for r in cur["run1"]["Alpha"]["rows"]] == \
        [sr.row_px(r) for r in SEEDS["run1"]["Alpha"]["rows"]]
    assert sr.seed_runs(current_layout())["run1"]["Welder"]["rows"][1]["gt_px"] == 1999


def test_pairing_is_the_same_under_both_layouts(paired):
    legacy, _ = paired
    cur, unmatched = sr.pair(sr.seed_runs(current_layout())["run1"], JF, METHOD, CFG)
    assert unmatched == [("Alpha", "v0", 3, 0)]
    assert [p.record() for p in cur] == [p.record() for p in legacy]


# ------------------------------------------------------------------ view distance
def test_view_index_map_plain_and_perms():
    names, ci = sr.view_index_map(CFG["Alpha"])
    assert names == {"v0": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4} and ci == 2
    names, ci = sr.view_index_map(CFG["Welder"])
    assert names == {"camera_0001": 0, "camera_0002": 1, "camera_0003": 2,
                     "camera_0005": 3, "camera_0008": 4} and ci == 0


def test_view_index_map_matches_e4_judge_on_the_real_config():
    if not os.path.isfile(sr.CONFIG):
        pytest.skip("SCSam3/demo/MVSeg.json not present")
    cfg = json.load(open(sr.CONFIG, encoding="utf-8"))
    names, ci = sr.view_index_map(cfg["CoffeeMartini"])           # the ring with perms
    assert names["cam16"] == ci == 14 and names["cam02"] == 2 and "cam03" not in names
    names, ci = sr.view_index_map(cfg["Barn"])
    assert names["v7"] == ci == 7 and names["v10"] == 10


def test_scene_config_by_key_or_folder():
    assert sr.scene_config(CFG, "Alpha") is CFG["Alpha"]
    cfg = {"AlphaKey": dict(CFG["Alpha"], folder="Alpha")}
    assert sr.scene_config(cfg, "Alpha") is cfg["AlphaKey"]
    with pytest.raises(KeyError):
        sr.scene_config(CFG, "Nope")


# ------------------------------------------------------------------ pairing
def test_pairing_keys_and_fields(paired):
    pairs, unmatched = paired
    assert unmatched == [("Alpha", "v0", 3, 0)]
    k = by_key(pairs)
    assert len(k) == 9
    p = k[("Alpha", "v4", 1)]                       # found by id, not by position
    assert (p.gt_px, p.raw_pred_px, p.raw_J, p.frame0_J) == (12000, 11000, 0.9, 0.4)
    assert p.later_J == pytest.approx(0.55)
    assert not p.ref and p.view_distance == 2
    assert k[("Alpha", "v2", 1)].ref and k[("Alpha", "v2", 1)].view_distance == 0
    assert k[("Welder", "camera_0005", 2)].view_distance == 3      # index 3 in the perms ring
    assert k[("Welder", "camera_0003", 1)].view_distance == 2


def test_population_is_nonref_with_ground_truth(paired):
    pairs, _ = paired
    pop = sorted((p.scene, p.cam, p.obj) for p in pairs if p.in_population)
    assert pop == [("Alpha", "v0", 1), ("Alpha", "v0", 2), ("Alpha", "v4", 1), ("Alpha", "v4", 2),
                   ("Welder", "camera_0003", 1), ("Welder", "camera_0005", 1),
                   ("Welder", "camera_0005", 2)]


def test_pairing_refuses_a_missing_camera_and_a_single_frame():
    with pytest.raises(KeyError):
        sr.pair(SEEDS["run1"], [e for e in JF if e["camera"] != "v4"], METHOD, CFG)
    one = [dict(e, result={"per_frame": {"J": [[0.5] for _ in e["objects"]]}}) for e in JF]
    with pytest.raises(ValueError):
        sr.pair(SEEDS["run1"], one, METHOD, CFG)


# ------------------------------------------------------------------ reading (1)
def test_diff_summary(paired):
    pairs, _ = paired
    s = sr.diff_summary([p for p in pairs if p.in_population])
    assert s["n"] == 7
    assert s["delta_mean"] == pytest.approx((0.25 - 0.5 - 0.02) / 7)
    assert s["delta_median"] == 0
    assert s["n_abs_delta_gt_0.05"] == 2
    assert s["n_identical"] == 4
    assert s["n_new_fail_at_frame0"] == 1                 # Alpha v4 obj 1: 0.9 -> 0.4
    assert s["n_recovered_at_frame0"] == 1                # Alpha v0 obj 2: 0.3 -> 0.55
    assert (s["n_raw_fail"], s["n_frame0_fail"]) == (4, 4)
    assert s["n_raw_in_boundary"] == 1                    # the J == 0.5 pair
    assert sr.diff_summary([])["n"] == 0 and sr.diff_summary([])["delta_mean"] is None


# ------------------------------------------------------------------ reading (2)
def test_linfit_exact_line_and_degenerate():
    fit = sr.linfit([0.0, 0.5, 1.0, 2.0], [0.1, 0.35, 0.6, 1.1])
    assert fit["slope"] == pytest.approx(0.5) and fit["intercept"] == pytest.approx(0.1)
    assert fit["r2"] == pytest.approx(1.0) and fit["n"] == 4
    fit = sr.linfit([1, 2, 3], [1, 2, 2])                 # by hand: slope 0.5, icpt 2/3, R2 0.75
    assert fit["slope"] == pytest.approx(0.5) and fit["intercept"] == pytest.approx(2 / 3)
    assert fit["r2"] == pytest.approx(0.75)
    flat = sr.linfit([1, 1, 1], [1, 2, 3])                # no spread in x: nothing to fit
    assert math.isnan(flat["slope"]) and math.isnan(flat["r2"])
    assert sr.linfit([1], [1])["n"] == 1 and math.isnan(sr.linfit([1], [1])["slope"])


def test_regression_on_the_synthetic_pairs(paired):
    pairs, _ = paired
    pop = [p for p in pairs if p.in_population]
    ys = [p.later_J for p in pop]
    raw = sr.linfit([p.raw_J for p in pop], ys)
    assert raw["slope"] == pytest.approx(0.5) and raw["intercept"] == pytest.approx(0.1)
    assert raw["r2"] == pytest.approx(1.0)
    f0 = sr.linfit([p.frame0_J for p in pop], ys)
    assert f0["r2"] < 1.0                                 # re-prediction moved two pairs


def test_reproduces_headroom_rounds_to_three_decimals():
    assert sr.reproduces_headroom({"slope": 0.96366, "intercept": 0.02711, "r2": 0.93304})
    assert not sr.reproduces_headroom({"slope": 0.96366, "intercept": 0.02711, "r2": 0.9336})
    assert not sr.reproduces_headroom({"slope": float("nan"), "intercept": 0.027, "r2": 0.933})


# ------------------------------------------------------------------ reading (3)
def test_labels():
    assert [sr.size_bin(x) for x in (0, 1999, 2000, 9999, 10000)] == \
        ["<2000", "<2000", "2000-10000", "2000-10000", ">=10000"]
    assert sr.rig("Welder") == sr.rig("Dog") == "hemisphere" and sr.rig("Barn") == "pinhole"
    assert sr.kind(0, 0.0) == "empty" and sr.kind(0, 0.9) == "empty"
    assert sr.kind(10, 0.49) == "misplaced" and sr.kind(10, 0.5) == "ok"


def test_split_and_decompose(paired):
    pairs, _ = paired
    f0, n0 = sr.split(pairs)
    assert sorted((p.cam, p.obj) for p in f0) == [("camera_0003", 1), ("camera_0005", 2),
                                                  ("v0", 1), ("v0", 2)]
    assert sorted((p.cam, p.obj) for p in n0) == [("camera_0005", 1), ("v4", 1), ("v4", 2)]
    d = sr.decompose(f0)
    assert d["n"] == 4
    assert d["kind"] == {"empty": 1, "misplaced": 3}
    assert d["size_bin"] == {"<2000": 2, "2000-10000": 1, ">=10000": 1}
    assert d["rig"] == {"pinhole": 2, "hemisphere": 2}
    assert d["pinhole_ge2000"] == 1                       # Alpha v0 obj 2 only
    assert d["view_distance"] == {"2": 3, "3": 1}
    assert d["scene"] == {"Alpha": 2, "Welder": 2}
    d = sr.decompose(n0)
    assert d["kind"] == {"ok": 3} and d["size_bin"] == {"<2000": 0, "2000-10000": 2, ">=10000": 1}
    assert d["pinhole_ge2000"] == 2


def test_record_fields(paired):
    pairs, _ = paired
    r = by_key(pairs)[("Welder", "camera_0005", 2)].record()
    assert r == {"scene": "Welder", "cam": "camera_0005", "obj": 2, "gt_px": 10000,
                 "raw_pred_px": 2000, "raw_J": 0.15, "frame0_J": 0.15,
                 "frames1_20_J": pytest.approx(0.175), "kind": "misplaced",
                 "size_bin": ">=10000", "rig": "hemisphere", "view_distance": 3}


# ------------------------------------------------------------------ CLI / freeze files
def write_inputs(tmp_path, seeds=SEEDS):
    paths = {}
    for name, obj in (("seeds.json", seeds), ("jf.json", JF), ("cfg.json", CFG)):
        p = tmp_path / name
        p.write_text(json.dumps(obj), encoding="utf-8")
        paths[name] = str(p)
    return paths


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_cli_writes_the_freeze_with_digests(tmp_path, capsys):
    paths = write_inputs(tmp_path)
    out = tmp_path / "runs"
    argv = ["--seeds", paths["seeds.json"], "--jf", paths["jf.json"], "--method", METHOD,
            "--config", paths["cfg.json"], "--out-dir", str(out),
            "--report", str(tmp_path / "readings.json")]
    assert sr.main(argv) == 0                           # --run omitted: the file holds one run
    text = capsys.readouterr().out
    assert "재현 불일치" in text and "seeds_equal_frame0" not in text
    f0 = json.load(open(out / "F0.json"))
    n0 = json.load(open(out / "N0.json"))
    assert (f0["set"], f0["n"], f0["rule"]) == ("F0", 4, "raw seed J < 0.5")
    assert (n0["set"], n0["n"], n0["rule"]) == ("N0", 3, "raw seed J >= 0.5")
    for rec in (f0, n0):
        assert rec["schema"] == sr.SCHEMA and rec["schema_version"] == sr.SCHEMA_VERSION
        assert rec["inputs"]["seeds"] == {"path": paths["seeds.json"], "sha256": sha(paths["seeds.json"]),
                                          "run": "run1"}
        assert rec["inputs"]["jf"]["sha256"] == sha(paths["jf.json"])
        assert rec["inputs"]["jf"]["method"] == METHOD
        assert rec["inputs"]["config"]["sha256"] == sha(paths["cfg.json"])
        assert rec["population"] == {"definition": rec["population"]["definition"], "n": 7,
                                     "n_reference": 2, "n_gt_empty": 1, "n_unmatched": 1}
        assert rec["seeds_equal_frame0"] is False
        assert rec["generated_at"] and rec["generated_by"] == "MVSeed/seed_regression.py"
        keys = [(p["scene"], p["cam"], p["obj"]) for p in rec["pairs"]]
        assert keys == sorted(keys)
        assert all(set(p) == {"scene", "cam", "obj", "gt_px", "raw_pred_px", "raw_J", "frame0_J",
                              "frames1_20_J", "kind", "size_bin", "rig", "view_distance"}
                   for p in rec["pairs"])
    assert f0["summary"]["pinhole_ge2000"] == 1
    rep = json.load(open(tmp_path / "readings.json"))
    assert rep["fit"]["frames1_20_on_raw"]["slope"] == pytest.approx(0.5)
    assert rep["fit"]["reproduces_headroom"] is False
    assert rep["diff"]["nonref"]["n"] == 7 and rep["diff"]["ref"]["n"] == 2
    assert rep["F0"]["n"] == 4 and rep["N0"]["n"] == 3
    # the freeze is written once
    with pytest.raises(SystemExit):
        sr.main(argv)
    assert sr.main(argv + ["--overwrite"]) == 0


def test_cli_refuses_bad_run_method_and_unpaired_ground_truth(tmp_path):
    paths = write_inputs(tmp_path)
    base = ["--seeds", paths["seeds.json"], "--jf", paths["jf.json"], "--config", paths["cfg.json"]]
    with pytest.raises(SystemExit):
        sr.main(base + ["--method", METHOD, "--run", "nope", "--out-dir", str(tmp_path / "a")])
    with pytest.raises(SystemExit):
        sr.main(base + ["--method", "absent", "--out-dir", str(tmp_path / "b")])
    (tmp_path / "two").mkdir()
    paths = write_inputs(tmp_path / "two", current_layout(baseline="ctrl"))
    with pytest.raises(SystemExit):                       # --run required: run1 and ctrl
        sr.main(["--seeds", paths["seeds.json"], "--jf", paths["jf.json"], "--config",
                 paths["cfg.json"], "--method", METHOD, "--out-dir", str(tmp_path / "c")])
    assert sr.main(["--seeds", paths["seeds.json"], "--run", "ctrl", "--jf", paths["jf.json"],
                    "--config", paths["cfg.json"], "--method", METHOD,
                    "--out-dir", str(tmp_path / "c")]) == 0
    assert json.load(open(tmp_path / "c" / "F0.json"))["inputs"]["seeds"]["run"] == "ctrl"
    # a seed row with ground truth that the jf file has no object for must abort
    seeds = json.loads(json.dumps(SEEDS))
    seeds["run1"]["Alpha"]["rows"].append(row("v0", 9, 0.7, 500, 400))
    (tmp_path / "bad").mkdir()
    paths = write_inputs(tmp_path / "bad", seeds)
    with pytest.raises(SystemExit):
        sr.main(["--seeds", paths["seeds.json"], "--jf", paths["jf.json"], "--config",
                 paths["cfg.json"], "--method", METHOD, "--out-dir", str(tmp_path / "d")])
    assert not (tmp_path / "d" / "F0.json").exists()


def test_cli_flags_a_frame0_seed_file(tmp_path, capsys):
    """When the seeds ARE the frame-0 values (baseline_seedJ.json today) both fits coincide
    and the outputs say so."""
    seeds = json.loads(json.dumps(SEEDS))
    f0 = {(e["camera"], o): e["result"]["per_frame"]["J"][i][0]
          for e in JF if e["method"] == METHOD for i, o in enumerate(e["objects"])}
    for v in seeds["run1"].values():
        for r in v["rows"]:
            r["J"] = f0.get((r["cam"], r["obj"]), r["J"])
    paths = write_inputs(tmp_path, seeds)
    out = tmp_path / "runs"
    assert sr.main(["--seeds", paths["seeds.json"], "--jf", paths["jf.json"], "--config",
                    paths["cfg.json"], "--method", METHOD, "--out-dir", str(out),
                    "--report", str(tmp_path / "r.json")]) == 0
    assert "seeds_equal_frame0" in capsys.readouterr().out
    assert json.load(open(out / "F0.json"))["seeds_equal_frame0"] is True
    rep = json.load(open(tmp_path / "r.json"))
    assert rep["fit"]["frames1_20_on_raw"] == rep["fit"]["frames1_20_on_frame0"]
    assert rep["diff"]["nonref"]["n_identical"] == 7


# ------------------------------------------------------------------ golden: headroom section 2
BASELINE = os.path.join(REPO, "MVSeed", "runs", "baseline_seedJ.json")


def test_baseline_reproduces_stage1_headroom(tmp_path, capsys):
    for p in (BASELINE, sr.DEFAULT_JF, sr.CONFIG):
        if not os.path.isfile(p):
            pytest.skip(f"{p} not present")
    out = tmp_path / "runs"
    assert sr.main(["--seeds", BASELINE, "--run", sr.METHOD, "--jf", sr.DEFAULT_JF,
                    "--method", sr.METHOD, "--out-dir", str(out),
                    "--report", str(tmp_path / "readings.json")]) == 0
    assert "재현 OK" in capsys.readouterr().out
    rep = json.load(open(tmp_path / "readings.json"))
    assert rep["population"]["n"] == 664 and rep["population"]["n_reference"] == 398
    assert rep["seeds_equal_frame0"] is True
    fit = rep["fit"]["frames1_20_on_frame0"]
    assert (round(fit["slope"], 3), round(fit["intercept"], 3), round(fit["r2"], 3)) == (0.964, 0.027, 0.933)
    assert rep["fit"]["frames1_20_on_raw"] == fit
    assert rep["diff"]["nonref"]["mean_raw_J"] == pytest.approx(0.836, abs=5e-4)
    assert rep["diff"]["ref"]["mean_raw_J"] == pytest.approx(0.951, abs=5e-4)
    assert rep["diff"]["nonref"]["n_raw_in_boundary"] == 27                # plan section 2
    f0 = json.load(open(out / "F0.json"))
    assert f0["n"] == 45 and f0["summary"]["kind"] == {"empty": 7, "misplaced": 38}
    assert f0["summary"]["size_bin"]["<2000"] == 23 and f0["summary"]["rig"]["hemisphere"] == 13
    assert f0["summary"]["pinhole_ge2000"] == 18                             # plan section 5, G-geo
    assert f0["summary"]["view_distance"].get("1", 0) == 0                   # headroom 3.2: none at d=1
    assert json.load(open(out / "N0.json"))["n"] == 619
    assert f0["inputs"]["seeds"]["path"] == "MVSeed/runs/baseline_seedJ.json"

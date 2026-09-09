"""MVSeed/score_seeds.py: populations, row fields and the paired-delta plumbing.

Container only (cv2).  A synthetic MVSeg root in a tmp dir: a pinhole scene "Toy" with
a perms ring and a hemisphere scene "Dog" with a start_cam.. ring, three objects whose
ground-truth areas sit one per size bin, one camera whose ground truth lacks an object
(a gt_empty row), and two runs whose predictions are the ground-truth rectangles
shifted sideways -- so every J is a hand-checkable (W-s)/(W+s).  J and F themselves are
pinned to eval/eval_jf.py by running its eval_camera on the same files; the delta
statistics are pinned to eval/report_jf.py by recomputing boot_ci / wilcoxon_p on the
deltas read back from the rows.
"""
import importlib.util
import json
import os
import sys

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from conftest import REPO  # noqa: E402

sys.path.insert(0, REPO)
from eval import eval_jf, report_jf  # noqa: E402


def load_score_seeds():
    spec = importlib.util.spec_from_file_location(
        "score_seeds", os.path.join(REPO, "MVSeed", "score_seeds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load_score_seeds()

H = W = 200
# object id -> (row0, row1, col0, col1): 10000 px, 2500 px, 400 px -- one per size bin
RECT = {1: (0, 100, 0, 100), 2: (120, 170, 0, 50), 3: (150, 170, 150, 170)}
BASE, NEW, NONE = "MVSeed_base", "MVSeed_new", "MVSeed_none"

CONFIG = {
    "Toy": {"start_frame": 0, "cam_list": [2, 5, 9], "c_ini": 5, "num_frame": 21,
            "folder": "Toy", "perms": [1, 2, 3, 5, 6, 8, 9], "start_cam": 1,
            "prefix": "cam", "prefix1": 2},
    "Dog": {"start_frame": 0, "cam_list": [1, 3, 4], "c_ini": 1, "num_frame": 21,
            "folder": "Dog", "num_cam": 5, "start_cam": 1,
            "prefix": "camera_", "prefix1": 4},
}
CENSUS = {"Toy": {"c_ini": "cam05", "seed_ids_c_ini": [1, 2, 3]},
          "Dog": {"c_ini": "camera_0001", "seed_ids_c_ini": [1, 2]}}
# which objects each camera's ground truth holds
GT_OBJS = {("Toy", "cam05"): [1, 2, 3], ("Toy", "cam02"): [1, 2, 3], ("Toy", "cam09"): [1, 2],
           ("Dog", "camera_0001"): [1, 2], ("Dog", "camera_0003"): [1, 2],
           ("Dog", "camera_0004"): [1, 2]}
# prediction = ground-truth rectangle shifted right by s columns; "blob" = a stray
# 5x5 square on an empty ground truth; None = no file written
SHIFT = {
    BASE: {("Toy", "cam05"): {1: 0, 2: 0, 3: 0},
           ("Toy", "cam02"): {1: 50, 2: 10, 3: 0},
           ("Toy", "cam09"): {1: 30, 2: 25, 3: "blob"},
           ("Dog", "camera_0001"): {1: 0, 2: 0},
           ("Dog", "camera_0003"): {1: 10, 2: 5},
           ("Dog", "camera_0004"): {1: 0, 2: 10}},
    NEW: {("Toy", "cam05"): {1: 0, 2: 0, 3: 0},
          ("Toy", "cam02"): {1: 10, 2: 10, 3: 4},
          ("Toy", "cam09"): {1: 30, 2: 25, 3: None},
          ("Dog", "camera_0001"): {1: 0, 2: 0},
          ("Dog", "camera_0003"): {1: 0, 2: 25},
          ("Dog", "camera_0004"): {1: 0, 2: 0}},
}


def shifted_j(obj, s):
    """IoU of a rectangle with itself shifted by s columns."""
    r0, r1, c0, c1 = RECT[obj]
    w = c1 - c0
    return (w - s) / (w + s)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    base = tmp_path_factory.mktemp("mvseg")
    for (scene, cam), objs in GT_OBJS.items():
        gt = np.zeros((H, W), np.uint8)
        for o in objs:
            r0, r1, c0, c1 = RECT[o]
            gt[r0:r1, c0:c1] = o
        d = base / scene / "Mask" / cam
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "000000.png"), gt)
    for run, table in SHIFT.items():
        for (scene, cam), shifts in table.items():
            d = base / scene / run / cam / "0"
            d.mkdir(parents=True)
            for o, s in shifts.items():
                if s is None:
                    continue
                pm = np.zeros((H, W), np.uint8)
                if s == "blob":
                    pm[10:15, 180:185] = 255
                else:
                    r0, r1, c0, c1 = RECT[o]
                    pm[r0:r1, c0 + s:c1 + s] = 255
                cv2.imwrite(str(d / f"{o}.png"), pm)
    (base / "MVSeg.json").write_text(json.dumps(CONFIG))
    (base / "census.json").write_text(json.dumps(CENSUS))
    return base


def cli(root, *extra):
    return ["--data-root", str(root), "--config", str(root / "MVSeg.json"),
            "--census", str(root / "census.json")] + list(extra)


@pytest.fixture(scope="module")
def out(root):
    return mod.main(cli(root, "--runs", NEW, "--baseline", BASE))


def rows_of(out, run):
    """(scene, cam, obj) -> row of `run`, whether it was a --runs entry or the baseline."""
    block = out["runs"].get(run) or out["baseline"]
    return {(r["scene"], r["cam"], r["obj"]): r
            for v in block["scenes"].values() for r in v["rows"]}


# ----------------------------------------------------------------- J and F
def test_j_matches_the_shifted_rectangles(out):
    rows = rows_of(out, NEW)
    for (scene, cam), shifts in SHIFT[NEW].items():
        for o, s in shifts.items():
            if isinstance(s, int) and o in GT_OBJS[(scene, cam)]:
                assert rows[(scene, cam, o)]["J"] == pytest.approx(shifted_j(o, s))
    # empty ground truth: no file -> J = 1 (eval_jf convention); base's blob -> J = 0
    assert rows[("Toy", "cam09", 3)]["J"] == 1.0 and rows[("Toy", "cam09", 3)]["pred_px"] == 0
    base = rows_of(out, BASE)
    assert base[("Toy", "cam09", 3)]["J"] == 0.0 and base[("Toy", "cam09", 3)]["pred_px"] == 25


def test_j_and_f_equal_eval_jf(root, out, monkeypatch):
    """eval_jf.eval_camera on the same files, object set = that camera's GT."""
    monkeypatch.setattr(eval_jf, "ROOT", str(root))
    for run in (BASE, NEW):
        per_scene = mod.score_run(run, ["Dog", "Toy"], CONFIG, CENSUS, str(root))
        rows = {(r["scene"], r["cam"], r["obj"]): r
                for v in per_scene.values() for r in v["rows"]}
        for (scene, cam), objs in GT_OBJS.items():
            e = eval_jf.eval_camera((scene, cam, run, None))
            assert e["objects"] == objs
            for oi, o in enumerate(objs):
                assert rows[(scene, cam, o)]["J"] == e["result"]["per_frame"]["J"][oi][0]
                assert rows[(scene, cam, o)]["F"] == e["result"]["per_frame"]["F"][oi][0]
                assert rows[(scene, cam, o)]["gt_px"] == e["result"]["per_frame"]["gt_area"][oi][0]
                assert rows[(scene, cam, o)]["pred_px"] == e["result"]["per_frame"]["pred_area"][oi][0]


def test_missing_run_scores_empty_predictions(root):
    per_scene = mod.score_run(NONE, ["Toy"], CONFIG, CENSUS, str(root))
    for r in per_scene["Toy"]["rows"]:
        assert r["pred_px"] == 0
        assert r["J"] == (1.0 if r["gt_px"] == 0 else 0.0)
        assert r["F"] == (1.0 if r["gt_px"] == 0 else 0.0)


# ------------------------------------------------------------- populations
def test_population_split(out):
    for run, halluc in ((BASE, 1), (NEW, 0)):
        s = (out["baseline"] if run == BASE else out["runs"][run])["summary"]
        assert s["gt_present"]["n"] == 9          # Toy 5 + Dog 4
        assert s["all"]["n"] == 10                # + the one empty-GT row
        assert s["gt_empty"]["n"] == 1
        assert s["gt_empty"]["halluc"] == halluc
        assert s["ref"]["n"] == 5                 # Toy 3 + Dog 2, all J = 1
        assert s["ref"]["J"] == 1.0
    b = out["baseline"]["summary"]
    assert b["gt_present"]["fail"] == 2 and b["gt_present"]["band"] == 1
    assert b["all"]["fail"] == 3                  # the blob row (J = 0) joins in `all`
    n = out["runs"][NEW]["summary"]
    assert n["gt_present"]["fail"] == 2 and n["gt_present"]["band"] == 1
    # the headline J is pooled over pairs; the scene mean is a separate field
    js = [r["J"] for r in rows_of(out, NEW).values() if not r["ref"] and r["gt_px"] > 0]
    assert n["gt_present"]["J"] == pytest.approx(sum(js) / len(js))
    toy = [r["J"] for k, r in rows_of(out, NEW).items()
           if k[0] == "Toy" and not r["ref"] and r["gt_px"] > 0]
    dog = [r["J"] for k, r in rows_of(out, NEW).items()
           if k[0] == "Dog" and not r["ref"] and r["gt_px"] > 0]
    assert n["gt_present"]["J_scene_mean"] == pytest.approx(
        (sum(toy) / len(toy) + sum(dog) / len(dog)) / 2)


def test_legacy_scene_keys_keep_the_old_meaning(out):
    v = out["runs"][NEW]["scenes"]["Toy"]
    js = [r["J"] for r in v["rows"]]
    non = [r["J"] for r in v["rows"] if r["cam"] != "cam05"]
    assert v["c_ini"] == "cam05" and v["n"] == 9 and v["n_nonref"] == 6
    assert v["J_all"] == pytest.approx(sum(js) / 9)
    assert v["J_nonref"] == pytest.approx(sum(non) / 6)
    assert v["fail_nonref"] == sum(1 for j in non if j < 0.5)


# -------------------------------------------------------------- row fields
def test_size_bin_edges():
    assert mod.size_bin(0) is None
    assert mod.size_bin(1) == "<2000" and mod.size_bin(1999) == "<2000"
    assert mod.size_bin(2000) == "2000-10000" and mod.size_bin(9999) == "2000-10000"
    assert mod.size_bin(10000) == ">=10000"


def test_row_fields(out):
    rows = rows_of(out, NEW)
    for (scene, cam, o), r in rows.items():
        assert r["rig"] == ("hemisphere" if scene == "Dog" else "pinhole")
        assert r["ref"] == (cam == CENSUS[scene]["c_ini"])
        assert r["fail"] == (r["J"] < 0.5)
        assert r["boundary_band"] == (0.4 <= r["J"] < 0.6)
        if o in GT_OBJS[(scene, cam)]:
            r0, r1, c0, c1 = RECT[o]
            assert r["gt_px"] == (r1 - r0) * (c1 - c0)
            assert r["size_bin"] == {1: ">=10000", 2: "2000-10000", 3: "<2000"}[o]
        else:
            assert r["gt_px"] == 0 and r["size_bin"] is None
    # view distance: |ring index - ring index of c_ini|; perms ring and start_cam ring
    dist = {k: r["view_distance"] for k, r in rows.items()}
    assert dist[("Toy", "cam05", 1)] == 0 and dist[("Toy", "cam02", 1)] == 2 \
        and dist[("Toy", "cam09", 1)] == 3
    assert dist[("Dog", "camera_0001", 1)] == 0 and dist[("Dog", "camera_0003", 1)] == 2 \
        and dist[("Dog", "camera_0004", 1)] == 3
    assert rows[("Toy", "cam09", 1)]["boundary_band"]            # J = 70/130


def test_view_index_map_follows_e4_judge():
    assert mod.view_index_map(CONFIG["Toy"]) == {
        "cam01": 0, "cam02": 1, "cam03": 2, "cam05": 3, "cam06": 4, "cam08": 5, "cam09": 6}
    assert mod.view_index_map(CONFIG["Dog"])["camera_0005"] == 4
    assert mod.rig_of("Welder") == "hemisphere" and mod.rig_of("Fencing") == "pinhole"


def test_breakdowns(out):
    g = out["runs"][NEW]["summary"]["gt_present"]
    assert {k: v["n"] for k, v in g["by_size"].items()} == \
        {"<2000": 1, "2000-10000": 4, ">=10000": 4}
    assert {k: v["n"] for k, v in g["by_rig"].items()} == {"pinhole": 5, "hemisphere": 4}
    assert {k: v["n"] for k, v in g["by_view_distance"].items()} == {"2": 5, "3": 4}
    assert {k: v["n"] for k, v in g["by_scene"].items()} == {"Dog": 4, "Toy": 5}
    assert list(g["by_size"]) == ["<2000", "2000-10000", ">=10000"]    # fixed order


def test_c_ini_mismatch_is_an_error(root):
    census = {"Toy": dict(CENSUS["Toy"], c_ini="cam02")}
    with pytest.raises(SystemExit, match="c_ini differs"):
        mod.score_run(NEW, ["Toy"], CONFIG, census, str(root))


# ------------------------------------------------------------------ paired
def expected_deltas(out):
    """(scene, cam, obj) -> J_new - J_base over the gt_present rows, in row order."""
    new, base = rows_of(out, NEW), rows_of(out, BASE)
    return {k: new[k]["J"] - base[k]["J"] for k in new if not new[k]["ref"] and new[k]["gt_px"] > 0}


def test_paired_delta_matches_report_jf(out):
    p = out["paired"][NEW]
    d = expected_deltas(out)
    assert p["n_run"] == p["n_paired"] == 9
    s = p["subsets"]["gt_present"]
    assert s["n"] == 9 and s["n_scenes"] == 2
    assert s["mean"] == pytest.approx(sum(d.values()) / 9)
    assert s["mean"] == pytest.approx((shifted_j(1, 10) - shifted_j(1, 50)      # Toy cam02 1
                                       + shifted_j(3, 4) - 1                     # Toy cam02 3
                                       + 1 - shifted_j(1, 10)                    # Dog 0003 1
                                       + shifted_j(2, 25) - shifted_j(2, 5)      # Dog 0003 2
                                       + 1 - shifted_j(2, 10)) / 9)              # Dog 0004 2
    groups = [[v for k, v in d.items() if k[0] == sc] for sc in ("Dog", "Toy")]
    assert (s["ci_lo"], s["ci_hi"]) == report_jf.boot_ci(groups)
    assert s["ci_lo"] < s["mean"] < s["ci_hi"]
    p_w, how = report_jf.wilcoxon_p(list(d.values()))
    assert s["wilcoxon_p"] == p_w and s["wilcoxon"] == how == "exact, m=5"
    assert (s["wins"], s["ties"], s["losses"]) == (3, 4, 2)
    assert (s["fail_base"], s["fail_run"], s["recovered"], s["new_fail"]) == (2, 2, 1, 1)
    assert (s["band_base"], s["band_run"]) == (1, 1)


def test_paired_subsets(out):
    p = out["paired"][NEW]
    N, F = p["subsets"]["N"], p["subsets"]["F"]
    assert N["n"] == 7 and F["n"] == 2
    assert F["recovered"] == 1 and F["fail_run"] == 1 and F["new_fail"] == 0
    assert N["new_fail"] == 1 and N["recovered"] == 0
    assert F["mean"] == pytest.approx((shifted_j(1, 10) - shifted_j(1, 50)) / 2)
    assert {k: v["n"] for k, v in p["by_size"].items()} == \
        {"<2000": 1, "2000-10000": 4, ">=10000": 4}
    assert p["by_size"]["<2000"]["mean"] == pytest.approx(shifted_j(3, 4) - 1)
    assert p["by_size"]["<2000"]["wilcoxon"] == "exact, m=1"
    assert {k: v["n"] for k, v in p["by_rig"].items()} == {"pinhole": 5, "hemisphere": 4}
    assert {k: v["n"] for k, v in p["by_view_distance"].items()} == {"2": 5, "3": 4}
    assert p["by_scene"]["Toy"]["n"] == 5 and p["by_scene"]["Dog"]["n"] == 4
    # a single-scene subset is a plain bootstrap over that scene's deltas
    d = expected_deltas(out)
    toy = [v for k, v in d.items() if k[0] == "Toy"]
    assert (p["by_scene"]["Toy"]["ci_lo"], p["by_scene"]["Toy"]["ci_hi"]) == \
        report_jf.boot_ci([toy])
    # monitors: the hallucination went away, the c_ini seeds are untouched
    assert p["gt_empty"] == {"n": 1, "halluc_base": 1, "halluc_run": 0}
    assert p["ref"] == {"n": 5, "max_abs_delta": 0.0}


def test_empty_subset_is_nan_not_a_crash():
    s = mod.delta_stats([])
    assert s["n"] == 0 and s["mean"] != s["mean"] and s["wilcoxon"] == "none, m=0"
    assert mod.no_nan(s)["mean"] is None


def test_partial_baseline_pairs_only_the_common_rows(root):
    """Baseline scored on one scene: pairs come from that scene alone, the rest is counted."""
    new = mod.all_rows(mod.score_run(NEW, ["Dog", "Toy"], CONFIG, CENSUS, str(root)))
    base = mod.all_rows(mod.score_run(BASE, ["Dog"], CONFIG, CENSUS, str(root)))
    p = mod.paired_summary(new, base)
    assert p["n_run"] == 9 and p["n_paired"] == 4
    assert list(p["by_scene"]) == ["Dog"]


# --------------------------------------------------------------------- CLI
def test_cli_writes_json_and_prints_korean_tables(root, tmp_path, capsys):
    out_path = tmp_path / "runs" / "score.json"
    mod.main(cli(root, "--runs", NEW, "--baseline", BASE, "--out", str(out_path)))
    text = capsys.readouterr().out
    assert f"== {NEW}: 2장면 ==" in text
    assert "gt_present" in text and "gt_empty" in text and "환각" in text
    assert f"== ΔJ = {NEW} − {BASE}" in text and "N (기준 J≥0.5)" in text
    assert "하락 > 0.005" in text                          # Dog's mean delta is below -0.005
    data = json.load(open(out_path))
    assert set(data) == {"meta", "runs", "baseline", "paired"}
    assert data["baseline"]["run"] == BASE
    assert data["paired"][NEW]["subsets"]["gt_present"]["n"] == 9
    assert data["meta"]["size_bins"] == ["<2000", "2000-10000", ">=10000"]
    assert "random.Random(0)" in data["meta"]["boot"] and "10000" in data["meta"]["boot"]
    # the row table carries every section-2 field
    row = data["runs"][NEW]["scenes"]["Toy"]["rows"][0]
    for key in ("scene", "cam", "obj", "ref", "J", "F", "gt_px", "pred_px", "size_bin",
                "rig", "fail", "boundary_band", "view_distance"):
        assert key in row


def test_cli_without_baseline_has_no_paired_block(root, capsys):
    o = mod.main(cli(root, "--runs", BASE, "--scenes", "Toy"))
    assert "paired" not in o and "baseline" not in o
    assert list(o["runs"][BASE]["scenes"]) == ["Toy"]
    assert "ΔJ" not in capsys.readouterr().out

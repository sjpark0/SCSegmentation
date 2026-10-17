"""plenoptic/score_full.py and plenoptic/build_full.merge_back on a synthetic two-camera run (host, stdlib only)."""
import json
import os

from conftest import REPO, _load

sf = _load("score_full", os.path.join(REPO, "plenoptic", "score_full.py"))


def _entry(cam, objs, J, F, gt, frames=(900,)):
    return dict(method="M", camera=cam, objects=objs, meta=dict(frames=list(frames)),
                result=dict(per_frame=dict(J=[[j] for j in J], F=[[f] for f in F], gt_area=[[g] for g in gt],
                                           pred_missing=[[False] for _ in objs])))


def test_frame900_readings():
    entries = [_entry("cam_06", [1, 2], [1.0, 1.0], [1.0, 1.0], [10, 10]),
               _entry("cam_07", [1, 2], [0.9, 0.3], [0.8, 0.2], [10, 10]),
               _entry("cam_22", [1, 2], [0.7, 1.0], [0.6, 1.0], [10, 0]),       # person 2 not in this view's GT
               dict(method="other", camera="cam_07", objects=[1], meta=dict(frames=[900]), result=None)]
    r = sf.frame900(entries, "M")
    assert r["n_cameras"] == 3 and r["n_present"] == 3 and r["n_gt_empty"] == 1
    assert abs(r["Q1_mean_J"] - (0.9 + 0.3 + 0.7) / 3) < 1e-12
    assert r["Q3_fail_lt05"] == 1 and r["Q3_worst"][0] == (0.3, "cam_07", 2)
    assert set(r["Q2_by_distance"]) == {1, 16}                                  # cam_22 is opposite cam_06 on 32
    cams = [1.0, (0.85 + 0.25) / 2, (0.65 + 1.0) / 2]
    assert abs(r["Q5_scene_JF"] - 100 * sum(cams) / 3) < 1e-9
    assert r["ref_J"] == {1: 1.0, 2: 1.0}


def test_census_counts_zero_stretches():
    va = dict(seed={"0": {"1": 100, "2": 0}},
              tracked={"0": {str(t): {"1": (0 if 903 <= t <= 905 else 5 if t == 907 else 100), "2": 0}
                             for t in range(901, 911)}})
    c = sf.census(va, range(901, 911))
    assert c["n_pairs"] == 1                                                    # person 2 had no seed: skipped
    assert c["per_pair"]["v0/1"] == dict(zero=3, longest_zero=3, below_0p1_seed=4)
    assert abs(c["mean_zero_fraction"] - 0.3) < 1e-12


def test_prompt_identity_and_merge_back(tmp_path):
    bf = _load("build_full", os.path.join(REPO, "plenoptic", "build_full.py"))
    fwd, bwd = tmp_path / "F" / "M", tmp_path / "B" / "M"
    for cam in ("cam_00", "cam_01"):
        (fwd / cam / "900").mkdir(parents=True)
        (fwd / cam / "901").mkdir()
        for i in (0, 1, 900):
            (bwd / cam / str(i)).mkdir(parents=True)
            for o in (1, 2):
                (bwd / cam / str(i) / f"{o}.png").write_bytes(f"{cam}{i}{o}".encode())
        for o in (1, 2):
            (fwd / cam / "900" / f"{o}.png").write_bytes(f"{cam}0{o}".encode())
    (fwd / "cam_01" / "900" / "2.png").write_bytes(b"other")
    p = sf.prompt_identity(str(fwd), str(bwd))
    assert p["identical"] == 3 and p["different_list"] == ["cam_01/2.png"] and not p["in_one_only"]
    bf.merge_back(str(bwd), str(fwd))
    assert (fwd / "cam_00" / "899" / "1.png").read_bytes() == b"cam_0011"           # i = 1 -> source 899
    assert (fwd / "cam_00" / "0" / "2.png").read_bytes() == b"cam_009002"           # i = 900 -> source 0
    assert (fwd / "cam_01" / "900" / "2.png").read_bytes() == b"other"              # the forward prompt frame is kept
    assert os.stat(fwd / "cam_00" / "0" / "2.png").st_ino == os.stat(bwd / "cam_00" / "900" / "2.png").st_ino
    bf.merge_back(str(bwd), str(fwd))                                               # idempotent (rerun after a stop)


def test_main_writes_the_readings(tmp_path, capsys):
    fwd = tmp_path / "M"
    (fwd / "cam_06" / "900").mkdir(parents=True)
    va = dict(seed={"6": {"1": 50}}, tracked={"6": {str(t): {"1": 50} for t in range(900, 903)}})
    json.dump(dict(provenance=dict(start_frame=900, num_frame=3, view_areas=va, git_rev="x", ref_snap=None,
                                   crop_small=None)), open(fwd / "MANIFEST.json", "w"))
    jf = tmp_path / "jf.json"
    json.dump([_entry("cam_06", [1], [1.0], [1.0], [50]), _entry("cam_07", [1], [0.8], [0.7], [50])], open(jf, "w"))
    import sys
    argv, sys.argv = sys.argv, ["score_full.py", "--jf", str(jf), "--method", "M", "--fwd", str(fwd),
                                "--out", str(tmp_path / "s.json")]
    try:
        sf.main()
    finally:
        sys.argv = argv
    s = json.load(open(tmp_path / "s.json"))
    assert s["frame900"]["Q1_mean_J"] == 0.8 and s["census_forward"]["n_frames"] == 2
    assert s["census_forward"]["mean_zero_fraction"] == 0.0 and s["run_forward"]["git_rev"] == "x"
    assert "Q1 mean J 0.8000" in capsys.readouterr().out


def test_merge_back_runs_under_the_plain_host_python(tmp_path):
    """run_full.sh calls merge_back with the host's python3, which has no numpy (2026-10-18 chain stop)."""
    import shutil
    import subprocess
    py = shutil.which("python3", path="/usr/bin:/bin")
    if py is None:
        import pytest
        pytest.skip("no /usr/bin/python3")
    b, f = tmp_path / "B", tmp_path / "F"
    (b / "cam_00" / "1").mkdir(parents=True)
    (b / "cam_00" / "1" / "3.png").write_bytes(b"x")
    (f / "cam_00").mkdir(parents=True)
    r = subprocess.run([py, "-I", os.path.join(REPO, "plenoptic", "build_full.py"), "merge_back", str(b), str(f)],
                       capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stderr
    assert (f / "cam_00" / "899" / "3.png").read_bytes() == b"x"

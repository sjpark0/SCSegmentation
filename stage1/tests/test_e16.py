"""S1-E16 (docs/stage1-E16-prereg.md): the geometry-free control arm, the --nbr neighbour rules and the DEV override."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from geomem import arms
from methods.geomem import dev_decode as D
from methods.geomem import run as R
from methods.geomem import stage1 as S


def test_nogeo_arm_sets_and_flags():
    assert arms.NOGEO_ARMS == ("nogeo_pred_ft",)
    a = "nogeo_pred_ft"
    assert a in R.ARMS and a in R.SET_ARMS and a in arms.PRED_MEM_ARMS and a in arms.NO_EMBED_ARMS
    assert a not in arms.SET2_ARMS and a not in R.PAIR_ARMS and a not in arms.DEC_ARMS and a not in arms.FUSE_ARMS
    T = pytest.importorskip("geomem.train")                  # torch: the training module (scsam3 / torch test runs)
    assert a in T.ARMS and a in T.FT_ARMS and a in T.SET_ARMS and a not in T.BIAS_ARMS and a not in T.PAIR_ARMS
    from geomem import wrapper as W
    assert a in W.FT_ARMS and a not in W.BIAS_ARMS                # wrapper.freeze: attention trainable, no GeoBias
    f = T.arm_flags(a)
    assert f["use_bias"] is False and f["use_geometry"] is False and f["no_rope"] is False
    assert f["extra_mem_prob"] == T.EXTRA_MEM_PROB and f["extra_mem_pred"] is True and f["train_decoder"] is False
    g = T.arm_flags("geoset_pred_ft")                       # the recipe it copies: same except the bias
    assert g["use_bias"] is True and g["use_geometry"] is False
    assert {k: v for k, v in f.items() if k != "use_bias"} == {k: v for k, v in g.items() if k != "use_bias"}
    assert list(S.PRESETS)[-2:] == ["geoset_rep_xs", "geoset_rep_xs_ab"]   # S1-E15 I5: PRESETS untouched by E16


def test_index_coords_and_candidates():
    c = R.camera_index_coords(["cam9", "cam0", "camera_0012", "cam3"])
    assert c["cam9"][0] == 9 and c["camera_0012"][0] == 12 and c["cam0"][0] == 0
    assert R.select_candidates("cam3", c, k=2) == ["cam0", "cam9"]      # |3-0| = 3 < |3-9| = 6 < |3-12|
    assert R.select_neighbours("cam3", c, {"cam0": False, "cam9": True, "camera_0012": True}, k=2) == ["cam9"]
    with pytest.raises(ValueError):
        R.camera_index_coords(["left", "right"])
    with pytest.raises(ValueError):
        R.camera_coords("nope", None, SimpleNamespace(cams=["cam0"], image_paths={}))


def test_pi3_coords_are_the_centres(tmp_path):
    for c, t in (("cam0", [0, 0, 1]), ("cam1", [2, 0, 1])):
        c2w = np.eye(4)
        c2w[:3, 3] = t
        np.savez(tmp_path / f"{c}.npz", c2w=c2w)
    sc = SimpleNamespace(cams=["cam0", "cam1"], image_paths={})
    got = R.camera_coords("pi3", str(tmp_path), sc)
    assert got.keys() == {"cam0", "cam1"} and np.allclose(got["cam1"], [2, 0, 1])
    assert all(np.allclose(got[c], R.camera_centres(str(tmp_path), sc.cams)[c]) for c in sc.cams)


def test_check_gate_out_refuses_nbr_under_the_arm():
    R.check_gate_out("nogeo_pred_ft", "obj", "e16_x", xcl="outs", nbr="index")
    R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", nbr="pi3")
    R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", nbr=None)
    with pytest.raises(SystemExit):
        R.check_gate_out("nogeo_pred_ft", "view", "nogeo_pred_ft", nbr="index")


def test_e16_presets(tmp_path):
    assert set(D.E16_ARMS) == {"NG-idx", "NG-feat", "NG-pi3", "G", "G-bias0", "AB"}
    ng = D.e16_preset("NG-idx")
    assert ng["arm"] == "nogeo_pred_ft" and ng["gate"] == "obj" and ng["xcl"] == "outs" and ng["nbr"] == "index"
    assert D.e16_preset("NG-feat")["nbr"] == "feat" and D.e16_preset("NG-pi3")["gate"] == "reproj" and "nbr" not in D.e16_preset("NG-pi3")
    g = D.e16_preset("G")
    assert {k: v for k, v in g.items() if not k.startswith("_")} == S.PRESETS["geoset_rep_xs"]
    assert D.e16_preset("G-bias0")["zero_bias"] is True and R.e15_options(SimpleNamespace(**D.e16_preset("G-bias0"))) is None
    ab = D.e16_preset("AB", str(tmp_path))
    assert ab["arm"] == "geoset_pred_ft" and ab["relay"] == 60.0 and ab["id3d"] == "cstar" and ab["runs"] == str(tmp_path)
    assert R.e15_options(SimpleNamespace(**ab))["row"] == "AB"
    # the seed label comes from <runs>/<arm>/fold0/config.json
    d = tmp_path / "nogeo_pred_ft" / "fold0"
    d.mkdir(parents=True)
    json.dump({"seed": 2}, open(d / "config.json", "w"))
    assert D.seed_of_runs(str(tmp_path), "nogeo_pred_ft") == 2 and D.seed_of_runs(str(tmp_path), "geoset_pred_ft") == 0


def test_e16_prepare_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DATA", D.DATA)                       # real meta (index only), roots under tmp
    with pytest.raises(SystemExit):                              # DEV-B refused
        D.prepare(["--e16-arm", "NG-idx", "--split", "B", "--root", str(tmp_path / "b"), "--scenes", "Dog"])
    with pytest.raises(SystemExit):                              # no E15 flags with an E16 arm
        D.prepare(["--e16-arm", "G", "--relay", "60", "--root", str(tmp_path / "r"), "--scenes", "Dog"])
    with pytest.raises(SystemExit):                              # not both overrides
        D.prepare(["--e16-arm", "G", "--e15-arm", "AB", "--root", str(tmp_path / "x"), "--scenes", "Dog"])
    a, preset, rec, scenes, _ = D.prepare(["--e16-arm", "NG-feat", "--runs", str(tmp_path / "runs"), "--seed-tag", "s1",
                                           "--root", str(tmp_path / "ok"), "--scenes", "Dog"])
    assert preset["nbr"] == "feat" and preset["runs"] == str(tmp_path / "runs") and rec["e16_arm"] == "NG-feat"
    assert scenes == ["Dog"] and json.load(open(tmp_path / "ok" / "index.json"))["e16_arm"] == "NG-feat"
    a, preset, rec, _, _ = D.prepare(["--e16-arm", "G", "--seed-tag", "s0", "--scenes", "Dog", "--ref-rot", "1",
                                      "--root", str(tmp_path / "g")])
    assert preset["arm"] == "geoset_pred_ft" and "nbr" not in preset
    # the default root name carries the arm, the seed tag, the split and the rotation
    a2 = D._parser().parse_args(["--e16-arm", "NG-idx", "--seed-tag", "s2", "--ref-rot", "2"])
    assert a2.e16_arm == "NG-idx" and a2.seed_tag == "s2"

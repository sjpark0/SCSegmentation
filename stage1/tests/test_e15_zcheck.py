"""S1-E15 Z-checks (stage1/scripts/e15/e15_zcheck.py, docs/stage1-E15-prereg.md §8): Z4 / Z-R / Z-I2 / Z-sim and the T06
I1 / I2 comparisons on synthetic decode folders (host)."""
import importlib.util
import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e15_synth as SY  # noqa: E402
from geomem import id3d  # noqa: E402

spec = importlib.util.spec_from_file_location("e15_zcheck", os.path.join(ROOT, "scripts", "e15", "e15_zcheck.py"))
Z = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Z)

CFG = dict(relay_theta=60.0, id3d="vote", id3d_scope="relay", relay_mode="seq", relay_order="hop", relay_mem="pool",
           relay_trust="id3d", flagged="keep")


def _w(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, bytes):
        open(path, "wb").write(data)
    else:
        json.dump(data, open(path, "w"), indent=1)


def _frame(d, evals=("e1", "e2"), r1=("e1", "e2", "n1"), png=b"p", r1png=b"r", xcl=None, conf=None, cfg=None, rel=None):
    """One dev_decode frame folder: eval finals, the c_ini GT, _r1 masks, round1_conf / xcl_conf / config, DONE."""
    for c in evals:
        _w(os.path.join(d, c, "1.png"), png if not isinstance(png, dict) else png.get(c, b"p"))
    _w(os.path.join(d, "ref", "1.png"), b"gt")
    for c in r1:
        _w(os.path.join(d, "_r1", c, "1.png"), r1png if not isinstance(r1png, dict) else r1png.get(c, b"r"))
    _w(os.path.join(d, "round1_conf.json"), conf or {c: {"1": dict(presence=1.0, iou=0.9, overlap=0.8, n_valid=99)} for c in r1})
    _w(os.path.join(d, "xcl_conf.json"), xcl or {c: {"1": dict(px=3)} for c in evals})
    if cfg is not None:
        _w(os.path.join(d, "config.json"), cfg)
    if rel is not None:
        _w(os.path.join(d, "relay.json"), rel)
    _w(os.path.join(d, "DONE"), b"")


def _root(base, frames, **kw):
    scenes = {}
    for s, t in frames:
        scenes.setdefault(s, dict(frames=[], eval_cams=["e1", "e2"], basic=[1]))["frames"].append(t)
    _w(os.path.join(base, "index.json"), dict(scenes=scenes))
    for s, t in frames:
        _frame(os.path.join(base, s, f"{t:06d}"), **kw)


def _summary(n_all_60=0, n_all_55=3):
    return dict(in_F={"60": dict(A=dict(n_all_in_F=n_all_60)), "55": dict(A=dict(n_all_in_F=n_all_55))})


def test_tree_compare_and_config_mismatch(tmp_path):
    _frame(str(tmp_path / "a"))
    _frame(str(tmp_path / "b"))
    assert Z.tree_compare(str(tmp_path / "a"), str(tmp_path / "b")) == []
    _frame(str(tmp_path / "c"), r1png={"n1": b"X"})
    assert Z.tree_compare(str(tmp_path / "a"), str(tmp_path / "c")) == [("png", "_r1/n1/1.png")]
    _frame(str(tmp_path / "d"), xcl={"e1": {"1": dict(px=4)}, "e2": {"1": dict(px=3)}})
    assert Z.tree_compare(str(tmp_path / "a"), str(tmp_path / "d")) == [("xcl_conf", "*")]
    assert Z.config_mismatch(CFG, 60.0, "vote") == []
    assert Z.config_mismatch(dict(CFG, relay_theta=55.0), 60.0, "vote") == [("relay_theta", 55.0)]
    assert Z.config_mismatch(dict(CFG, id3d="hybrid", id3d_scope="global"), 60.0, "vote") == [("id3d", "hybrid"), ("row", "G")]
    assert Z.config_mismatch(dict(CFG, id3d_scope="global"), 60.0, "vote", row="G") == []
    assert Z.config_mismatch({}, 60.0, "vote") == [("config", "no relay_theta")]


def test_zr_full_and_case_z(tmp_path, monkeypatch):
    """Z-R at 60: every PNG / round1_conf / xcl_conf byte-identical (no DEV-A camera in F); at 55 (in [A_E, A_all)):
    the case-Z rule -- a relayed non-eval camera's round-2 output (never written) does not matter, eval cams do."""
    monkeypatch.setattr(Z, "E15_DEV", str(tmp_path / "e15"))
    frames = [("S", 4), ("T", 8)]
    _root(str(tmp_path / "c0"), frames)
    assert Z.zr(60.0, "vote", _summary(), str(tmp_path / "c0")) is None                       # not decoded
    _root(str(tmp_path / "e15" / "AB_60_vote_A"), frames, cfg=CFG)
    r = Z.zr(60.0, "vote", _summary(), str(tmp_path / "c0"))
    assert r["pass"] and r["mode"] == "full" and r["n_frames"] == r["n_expected"] == 2 and r["family"] == "vote"
    _frame(str(tmp_path / "e15" / "AB_60_vote_A" / "T" / "000008"), r1png={"n1": b"X"}, cfg=CFG)  # a non-eval r1 mask
    r = Z.zr(60.0, "vote", _summary(), str(tmp_path / "c0"))
    assert not r["pass"] and r["diffs"] == {"T/8": [("png", "_r1/n1/1.png")]}
    # case Z at 55: an eval cam's PNG must match; a whole-file round1_conf difference fails; the xcl entry of a non-eval
    # cam may differ
    c55 = dict(CFG, relay_theta=55.0)
    _root(str(tmp_path / "e15" / "AB_55_vote_A"), frames, cfg=c55,
          xcl={"e1": {"1": dict(px=3)}, "e2": {"1": dict(px=3)}, "n1": {"1": dict(px=9)}})
    r = Z.zr(55.0, "vote", _summary(), str(tmp_path / "c0"))
    assert r["pass"] and r["mode"] == "case_z"
    _frame(str(tmp_path / "e15" / "AB_55_vote_A" / "S" / "000004"), png={"e2": b"Q"}, cfg=c55)
    assert Z.zr(55.0, "vote", _summary(), str(tmp_path / "c0"))["n_frames_differing"] == 1
    # a frame not decoded, or decoded with another family, fails
    os.remove(tmp_path / "e15" / "AB_60_vote_A" / "S" / "000004" / "DONE")
    _frame(str(tmp_path / "e15" / "AB_60_vote_A" / "T" / "000008"), cfg=dict(CFG, id3d="hybrid"))
    r = Z.zr(60.0, "vote", _summary(), str(tmp_path / "c0"))
    assert not r["pass"] and r["n_frames"] == 1 and set(r["diffs"]) == {"S/4", "T/8"}
    assert Z.zr_thetas(47.1, 58.12) == [60.0, 50.0, 55.0] and Z.zr_thetas(47.1, 50.0) == [60.0]


def test_zi2_star_views_and_round1(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, "E15_DEV", str(tmp_path / "e15"))
    frames = [("S", 4)]
    _root(str(tmp_path / "c0"), frames)
    rel = dict(S=["e1", "n1"], F=["e2"], active=True)
    c20 = dict(CFG, relay_theta=20.0)
    root = tmp_path / "e15" / "stress_20_vote_A"
    _root(str(root), frames, cfg=c20, rel=rel, png={"e2": b"relayed"}, r1=("e1", "e2", "n1", "extra"))
    r = Z.zi2("vote", str(tmp_path / "c0"))
    assert r["pass"] and r["n_star_eval_views"] == 1 and r["n_far_eval_views"] == 1 and r["n_frames_with_F"] == 1
    _frame(str(root / "S" / "000004"), cfg=c20, rel=rel, png={"e1": b"changed"})            # a star eval cam changed
    assert not Z.zi2("vote", str(tmp_path / "c0"))["pass"]
    _frame(str(root / "S" / "000004"), cfg=c20, rel=rel, r1png={"n1": b"X"})                 # a C0 round-1 mask changed
    assert Z.zi2("vote", str(tmp_path / "c0"))["diffs"]["S/4"] == [("r1_png", "n1")]


def test_z4_dog(tmp_path):
    frames = [("Dog", 4), ("Dog", 8)]
    _root(str(tmp_path / "c0"), frames)
    assert Z.z4(str(tmp_path / "z4"), str(tmp_path / "c0")) is None
    _root(str(tmp_path / "z4"), frames)
    r = Z.z4(str(tmp_path / "z4"), str(tmp_path / "c0"))
    assert not r["pass"] and r["diffs"]["_validate"] == [("missing", "_validate.json")]
    _w(str(tmp_path / "z4" / "_validate.json"), {"Dog/4": dict(equal=True), "Blocks/3": dict(equal=False)})
    r = Z.z4(str(tmp_path / "z4"), str(tmp_path / "c0"))
    assert r["pass"] and r["validate"] == ["Dog/4"] and r["n_frames"] == 2
    _frame(str(tmp_path / "z4" / "Dog" / "000008"), conf={"e1": {"1": dict(presence=0.5)}})
    assert not Z.z4(str(tmp_path / "z4"), str(tmp_path / "c0"))["pass"]


def test_zsim_one_on_the_synthetic_ring(tmp_path):
    """Z-sim: the decode's round-1 states = id3d.rejudge of the written PNGs; one altered state is reported."""
    d = SY.write_scene(str(tmp_path))
    lab = d["labels"][d["ref"]]
    masks = np.stack([lab == 1, lab == 2])
    cams = d["cams"][1:6]
    r1 = tmp_path / "r1"
    conf = {}
    for i, c in enumerate(cams):
        L = d["labels"][c]
        os.makedirs(r1 / c)
        for o in (1, 2):
            m = (L == (3 - o)) if i % 2 else (L == o)                                      # odd cams: swapped masks
            if m.any():
                cv2.imwrite(str(r1 / c / f"{o}.png"), m.astype(np.uint8) * 255)
        conf[c] = {str(o): dict(presence=1.0, iou=0.9, overlap=None, n_valid=0) for o in (1, 2)}
    host = id3d.rejudge(d["cache"], d["ref"], masks, [1, 2], "vote", str(r1), conf, cams, (SY.H, SY.W))
    rec = dict(family="vote", consts={}, rig_ok=True, round1=json.loads(json.dumps(host)))
    assert Z.zsim_one(d["cache"], d["ref"], masks, [1, 2], rec, str(r1), conf, (SY.H, SY.W)) == []
    assert any(s == id3d.FLAG for c in cams[1::2] for s in host[c].values())
    rec["round1"][cams[0]]["1"] = id3d.FLAG
    diff = Z.zsim_one(d["cache"], d["ref"], masks, [1, 2], rec, str(r1), conf, (SY.H, SY.W))
    assert diff == [(cams[0], "1", id3d.FLAG, host[cams[0]]["1"])]
    # cstar records its constants with "inf": they are read back as floats
    k = dict(SY.P1_CSTAR)
    hc = id3d.rejudge(d["cache"], d["ref"], masks, [1, 2], "cstar", str(r1), conf, cams, (SY.H, SY.W), consts=k)
    rc = dict(family="cstar", consts={kk: ("inf" if math.isinf(v) else v) for kk, v in k.items()}, rig_ok=True, round1=hc)
    assert Z._consts(rc) == k and Z.zsim_one(d["cache"], d["ref"], masks, [1, 2], rc, str(r1), conf, (SY.H, SY.W)) == []


def test_t06_invariants(tmp_path):
    res = tmp_path / "Result"
    for name in ("e15_C0", "e15_AB_60_vote"):
        for c in ("cam_00", "cam_01", "cam_13"):
            _w(str(res / name / c / "1.png"), b"far" if (c == "cam_13" and name != "e15_C0") else b"m")
            _w(str(res / (name + "_r1") / c / "1.png"), b"r")
        _w(str(res / name / "round1_conf.json"), {"cam_00": {"1": dict(presence=1.0)}})
    _w(str(res / "e15_AB_60_vote" / "relay.json"), dict(options=dict(scope="relay"), S=["cam_00", "cam_01"], F=["cam_13"],
                                                         active=True))
    assert Z.t06_invariants("AB_60_vote", str(res)) == []                  # only the far view changed
    _w(str(res / "e15_AB_60_vote" / "cam_01" / "1.png"), b"X")             # a star view changed: I2
    assert Z.t06_invariants("AB_60_vote", str(res)) == [("I2 png", "cam_01/1.png")]
    _w(str(res / "e15_AB_60_vote" / "relay.json"), dict(options=dict(scope="global"), S=["cam_00", "cam_01"], F=["cam_13"],
                                                         active=True))
    assert Z.t06_invariants("AB_60_vote", str(res)) == []                  # scope G: I2 does not apply
    _w(str(res / "e15_AB_60_vote_r1" / "cam_00" / "1.png"), b"Y")           # I1: a round-1 mask
    _w(str(res / "e15_AB_60_vote" / "round1_conf.json"), {"cam_00": {"1": dict(presence=0.5)}})
    assert Z.t06_invariants("AB_60_vote", str(res)) == [("I1 r1_png", "cam_00/1.png"), ("I1 round1_conf", "*")]


def test_run_all_records_missing_and_failures(tmp_path, monkeypatch):
    out = tmp_path / "runs"
    _w(str(out / "select.json"), dict(s_star=dict(family="vote")))
    _w(str(out / "angles.json"), dict(summary=dict(A_E=dict(a=47.1), A_all=dict(a=58.12), **_summary())))
    monkeypatch.setattr(Z.SEL, "OUT", str(out))
    monkeypatch.setattr(Z.SEL, "ZCHECK", str(out / "zcheck.json"))
    monkeypatch.setattr(Z, "E15_DEV", str(tmp_path / "e15"))
    monkeypatch.setattr(Z, "Z4_ROOT", str(tmp_path / "e15" / "_z4"))
    monkeypatch.setattr(Z, "DEV_C0_A", str(tmp_path / "c0"))
    monkeypatch.setattr(Z, "T06_RESULT", str(tmp_path / "none"))
    _root(str(tmp_path / "c0"), [("S", 4)])
    _root(str(tmp_path / "e15" / "AB_60_vote_A"), [("S", 4)], cfg=CFG)
    r = Z.run_all()
    assert r["ZR"]["60"]["pass"] and r["pass_present"] and not r["stage_pass"]
    assert r["missing"] == ["Z4", "ZI2", "Zsim T06", "Zsim DEV-A G", "ZT06"]
    assert r["ZR"]["60"]["root"] == os.path.relpath(str(tmp_path / "e15" / "AB_60_vote_A"), Z.S1)
    assert r["ZR"]["60"]["n_files"] == 10 and len(r["ZR"]["60"]["files_sha256"]) == 64
    assert json.load(open(out / "zcheck.json"))["ZR"]["60"]["pass"] is True
    # what Step 2 reads: the Z stage is incomplete -> need_zcheck lists the missing items
    assert Z.SEL.zcheck_missing(r, 60.0, 58.12, "vote") == r["missing"]
    _frame(str(tmp_path / "e15" / "AB_60_vote_A" / "S" / "000004"), png={"e1": b"Q"}, cfg=CFG)
    with pytest.raises(SystemExit) as e:
        Z.run_all()
    assert e.value.code == 2 and json.load(open(out / "zcheck.json"))["ZR"]["60"]["pass"] is False


def test_zt06_covers_every_t06_decode(tmp_path):
    res = tmp_path / "Result"
    assert Z.zt06(str(res)) is None
    for name in ("e15_C0", "e15_AB_60_vote", "e15_G_60_vote"):
        _w(str(res / name / "cam_00" / "1.png"), b"m" if name != "e15_G_60_vote" else b"changed")   # G changes star views
        _w(str(res / (name + "_r1") / "cam_00" / "1.png"), b"r")
        _w(str(res / name / "round1_conf.json"), {"cam_00": {}})
    _w(str(res / "e15_AB_60_vote" / "relay.json"), dict(options=dict(scope="relay"), S=["cam_00"], F=[], active=True))
    _w(str(res / "e15_G_60_vote" / "relay.json"), dict(options=dict(scope="global"), S=["cam_00"], F=[], active=True))
    r = Z.zt06(str(res))
    assert r["pass"] and sorted(r["runs"]) == ["AB_60_vote", "G_60_vote"] and r["runs"]["AB_60_vote"]["n_files"] == 4
    _w(str(res / "e15_AB_60_vote" / "cam_00" / "1.png"), b"X")
    r = Z.zt06(str(res))
    assert not r["pass"] and r["runs"]["AB_60_vote"]["diffs"] == [("I2 png", "cam_00/1.png")]

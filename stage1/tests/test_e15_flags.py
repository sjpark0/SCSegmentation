"""S1-E15 (docs/stage1-E15-prereg.md §3.7, §4, §8 I5-I9, §11.1): the opt-in decode flags, the §4 whitelist (L6), the
--xcl / theta rules, check_gate_out, the DEV restriction rule (I6), the recorded dev_decode override (M2, I8) and the
reference rotation (U1, protocol §4) (host)."""
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e15_synth as SY  # noqa: E402
from geomem import devset, id3d  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem import run as R  # noqa: E402
from methods.geomem import stage1 as S  # noqa: E402

DATA = os.path.join(ROOT, "Data")
needs_data = pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "Blocks", "meta.json")), reason="needs stage1/Data")
# The Step-1 commit (prereg §6.2 L5) changes geomem/id3d.py CSTAR = c_best / CSTAR_SELECTED = True and adds
# scripts/e15/runs/{angles,step0,select}.json.  Its ONLY intended test update is this line (False -> True): the tests
# below then check id3d.py against the committed select.json instead of the pre-selection state.  No other test pins
# the A-c* values (the rule tests use e15_synth.P1_CSTAR).
STEP1_COMMITTED = True


def _args(*extra):
    return common.parse_args(R.__doc__, R._extra, argv=["--scene", "X", "--out", "y", *extra])


def _r(*extra):
    """A relay decode's CLI args: --xcl outs (the only exclusivity rule E15 accepts) + the given flags."""
    return _args("--xcl", "outs", *extra)


def test_every_e15_flag_is_off_by_default():
    a = _args()
    assert all(getattr(a, k) is None for k in R.E15_KEYS)
    assert R.e15_options(a) is None
    assert R.e15_options(SimpleNamespace(arm="geoset_pred_ft", gate="reproj")) is None   # presets / dev_decode args


def test_e15_flags_need_relay():
    for extra in (["--id3d", "vote"], ["--flagged", "empty"], ["--relay-mode", "jacobi"], ["--gt-dir", "x"]):
        with pytest.raises(ValueError, match="--relay"):
            R.e15_options(_r(*extra))
    assert R.e15_options(_r("--id3d", "off")) is None                        # "off" is the default, not a stray flag


# the §4 whitelist rows (and the AB variants) as CLI flag sets -> the resolved options
ARMS = {
    "AB": (["--relay", "60", "--id3d", "vote"], "AB",
           dict(scope="relay", mode="seq", order="hop", mem="pool", trust="id3d", flagged="keep")),
    "AB_theta35_cstar_like": (["--relay", "35", "--id3d", "hybrid"], "AB", dict(theta=35.0, family="hybrid")),
    "stress": (["--relay", "20", "--id3d", "vote_inf"], "AB", dict(theta=20.0, family="vote_inf")),
    "theta_inf": (["--relay", "inf", "--id3d", "c0"], "AB", dict(theta=math.inf)),
    "A": (["--relay", "60", "--id3d", "vote", "--relay-mode", "jacobi"], "A", dict(mode="jacobi", trust="id3d")),
    "B-rep": (["--relay", "60"], "B-rep", dict(family=None, trust="reproj")),
    "B-obj": (["--relay", "60", "--relay-trust", "obj"], "B-obj", dict(family=None, trust="obj")),
    "ABc": (["--relay", "60", "--id3d", "vote", "--flagged", "chain", "--chain-dir", "/c"], "ABc",
            dict(flagged="chain", chain_dir="/c")),
    "Ac": (["--relay", "60", "--id3d", "vote", "--relay-mode", "jacobi", "--flagged", "chain", "--chain-dir", "/c"], "Ac",
           dict(mode="jacobi", flagged="chain")),
    "ABe": (["--relay", "60", "--id3d", "vote", "--flagged", "empty"], "ABe", dict(flagged="empty")),
    "ABnh": (["--relay", "60", "--id3d", "vote", "--relay-order", "angle"], "ABnh", dict(order="angle")),
    "ABml": (["--relay", "60", "--id3d", "vote", "--relay-mem", "latest"], "ABml", dict(mem="latest")),
    "G": (["--relay", "60", "--id3d", "vote", "--id3d-scope", "global"], "G", dict(scope="global")),
    "GN": (["--relay", "60", "--id3d", "vote", "--id3d-scope", "global_nf"], "GN", dict(scope="global_nf")),
    "oracle": (["--relay", "60", "--relay-trust", "gt", "--gt-dir", "/gt/{cam}/000900.png"], "oracle",
               dict(family=None, trust="gt")),
}


@pytest.mark.parametrize("name", sorted(ARMS))
def test_registered_arms_resolve(name):
    argv, row, expect = ARMS[name]
    o = R.e15_options(_r(*argv))
    assert o["row"] == row
    for k, v in expect.items():
        assert o[k] == v, (name, k, o[k], v)


def test_whitelist_is_the_registered_table():
    """§4: exactly 12 rows; every row is reachable from the CLI and resolves back to itself."""
    assert len(R.E15_ROWS) == 12 and len(set(R.E15_ROWS.values())) == 12
    assert set(R.E15_ROWS) == {"AB", "A", "B-rep", "B-obj", "ABc", "Ac", "ABe", "ABnh", "ABml", "G", "GN", "oracle"}
    assert {row for _, row, _ in ARMS.values()} == set(R.E15_ROWS)


@pytest.mark.parametrize("argv, msg", [
    (["--relay", "60", "--relay-trust", "id3d"], "needs --id3d"),
    (["--relay", "60", "--id3d", "vote", "--relay-trust", "reproj"], "B arms"),
    (["--relay", "60", "--relay-trust", "gt"], "--gt-dir"),
    (["--relay", "60", "--id3d", "vote", "--gt-dir", "/g"], "--relay-trust gt"),
    (["--relay", "60", "--id3d-scope", "global"], "needs --id3d"),
    (["--relay", "60", "--id3d", "vote", "--id3d-scope", "global", "--flagged", "empty"], "scope relay only"),
    (["--relay", "60", "--relay-mode", "jacobi"], "arm A"),
    (["--relay", "60", "--id3d", "vote", "--relay-mode", "jacobi", "--relay-order", "angle"], "relay \\(seq\\)"),
    (["--relay", "60", "--relay-mem", "latest"], "PASS Fix A"),
    (["--relay", "60", "--flagged", "empty"], "Fix A flags"),
    (["--relay", "60", "--id3d", "vote", "--flagged", "chain"], "go together"),
    (["--relay", "60", "--id3d", "vote", "--chain-dir", "/c"], "go together"),
    (["--relay", "-1"], ">= 0"),
    (["--relay", "nan"], ">= 0"),
])
def test_invalid_combinations_are_refused(argv, msg):
    with pytest.raises(ValueError, match=msg):
        R.e15_options(_r(*argv))


@pytest.mark.parametrize("argv", [
    ["--relay", "60", "--id3d", "vote", "--id3d-scope", "global", "--relay-mode", "jacobi"],          # global + jacobi
    ["--relay", "60", "--id3d", "vote", "--id3d-scope", "global_nf", "--relay-mode", "jacobi"],
    ["--relay", "60", "--relay-trust", "gt", "--gt-dir", "/g", "--relay-mode", "jacobi"],              # gt + jacobi
    ["--relay", "60", "--id3d", "vote", "--relay-mode", "jacobi", "--flagged", "empty"],               # empty + jacobi
    ["--relay", "60", "--id3d", "vote", "--relay-mem", "latest", "--flagged", "chain", "--chain-dir", "/c"],   # chain + latest
    ["--relay", "60", "--id3d", "vote", "--relay-mem", "latest", "--flagged", "empty"],
    ["--relay", "60", "--id3d", "vote", "--relay-order", "angle", "--relay-mem", "latest"],
    ["--relay", "60", "--id3d", "vote", "--relay-order", "angle", "--flagged", "empty"],
    ["--relay", "60", "--relay-order", "angle"],                                                       # B-rep, angle order
    ["--relay", "60", "--relay-trust", "obj", "--relay-order", "angle"],                              # B-obj, angle order
    ["--relay", "60", "--relay-trust", "gt", "--gt-dir", "/g", "--relay-order", "angle"],             # oracle, angle order
    ["--relay", "60", "--relay-trust", "gt", "--gt-dir", "/g", "--relay-mem", "latest"],
    ["--relay", "60", "--relay-trust", "gt", "--gt-dir", "/g", "--flagged", "empty"],
    ["--relay", "60", "--id3d", "vote", "--relay-trust", "gt", "--gt-dir", "/g"],                     # gt with a family
    ["--relay", "60", "--id3d", "vote", "--id3d-scope", "global", "--relay-order", "angle"],
    ["--relay", "60", "--id3d", "vote", "--id3d-scope", "global_nf", "--relay-mem", "latest"],
])
def test_combinations_outside_the_whitelist_are_refused(argv):
    with pytest.raises(ValueError, match="not a registered arm"):
        R.e15_options(_r(*argv))


@pytest.mark.parametrize("xcl", ["none", "out", "outmem", "out_nocap"])
def test_relay_needs_xcl_outs(xcl):
    with pytest.raises(ValueError, match="--xcl outs"):
        R.e15_options(_args("--xcl", xcl, "--relay", "60", "--id3d", "vote"))
    with pytest.raises(ValueError, match="--xcl outs"):
        R.e15_options(SimpleNamespace(arm="geoset_pred_ft", gate="reproj", xcl=xcl, relay=60.0, id3d="vote"))
    with pytest.raises(ValueError, match="--xcl outs"):
        R.e15_options(SimpleNamespace(arm="geoset_pred_ft", gate="reproj", relay=60.0, id3d="vote"))   # no xcl = none


@pytest.mark.parametrize("theta", ["20", "35", "40", "45", "50", "55", "60", "inf"])
def test_registered_thresholds_pass(theta):
    assert R.e15_options(_r("--relay", theta, "--id3d", "vote"))["theta"] == float(theta)


@pytest.mark.parametrize("theta", ["0", "30", "47.2", "58.12", "65", "90", "181", "1e9"])
def test_other_thresholds_are_refused(theta):
    with pytest.raises(ValueError, match="registered threshold"):
        R.e15_options(_r("--relay", theta, "--id3d", "vote"))


def test_cstar_state_follows_the_step1_commit(monkeypatch):
    """Before the Step-1 commit cstar is refused; after it, id3d.CSTAR / CSTAR_SELECTED equal the committed select.json
    (c_best; CSTAR_SELECTED stays False when there is no feasible A-c* candidate).  See STEP1_COMMITTED."""
    if not STEP1_COMMITTED:
        assert id3d.CSTAR_SELECTED is False
        with pytest.raises(ValueError, match="not selected"):
            R.e15_options(_r("--relay", "60", "--id3d", "cstar"))
    else:
        sel = json.load(open(R.E15_SELECT))
        assert R.e15_git_committed(R.E15_SELECT)[0]
        assert id3d.CSTAR_SELECTED is bool(sel.get("c_best"))
        if sel.get("c_best"):
            assert {k: float(v) for k, v in sel["c_best"]["consts"].items()} == {k: float(v) for k, v in id3d.CSTAR.items()}
    monkeypatch.setattr(id3d, "CSTAR_SELECTED", True)
    assert R.e15_options(_r("--relay", "60", "--id3d", "cstar"))["family"] == "cstar"
    monkeypatch.setattr(id3d, "CSTAR_SELECTED", False)
    with pytest.raises(ValueError, match="not selected"):
        R.e15_options(_r("--relay", "60", "--id3d", "cstar"))


def _git(repo, *a):
    import subprocess
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)


def _select(path, s_star="vote", best=None, c_best=None):
    best = best if best is not None else {"A-v": dict(family=s_star), "A-c*": None, "A-h": dict(family="hybrid"), "A-c0": None}
    json.dump(dict(s_star=s_star and dict(family=s_star), best_per_family=best, c_best=c_best), open(path, "w"))


def test_fix_a_decodes_need_the_committed_step1_record(tmp_path, monkeypatch):
    """Review 0 #8 / 1 #4: any Fix A family needs scripts/e15/runs/select.json committed and unmodified (§6.2 L5, §10),
    a feasible Step 1, s* or a family's best candidate, and for cstar the committed c_best in id3d.CSTAR."""
    assert R.E15_SELECT == os.path.join(ROOT, "scripts", "e15", "runs", "select.json")
    assert R.e15_require_step1(None) is None                                  # B arms / oracle: no Fix A
    p = tmp_path / "runs" / "select.json"
    p.parent.mkdir()
    for fam in id3d.FAMILIES:
        with pytest.raises(ValueError, match="Step-1 commit"):
            R.e15_require_step1(fam, str(p))                                  # missing
    _git(tmp_path, "init", "-q")
    _select(p)
    with pytest.raises(ValueError, match="not tracked"):
        R.e15_require_step1("vote", str(p))
    _git(tmp_path, "add", "runs/select.json")
    with pytest.raises(ValueError, match="differs from HEAD"):               # staged, not committed
        R.e15_require_step1("vote", str(p))
    _git(tmp_path, "commit", "-q", "-m", "step 1")
    assert R.e15_require_step1("vote", str(p))["s_star"]["family"] == "vote"
    assert R.e15_require_step1("hybrid", str(p))                              # a family's best (Step 2b / Step 3 s')
    with pytest.raises(ValueError, match="best feasible"):
        R.e15_require_step1("c0", str(p))                                     # no feasible A-c0 candidate
    _select(p, s_star="vote", best={"A-v": dict(family="vote")})
    with pytest.raises(ValueError, match="differs from HEAD"):               # modified after the commit
        R.e15_require_step1("vote", str(p))
    k = dict(SY.P1_CSTAR, beta=1.5)
    _select(p, s_star="cstar", best={"A-c*": dict(family="cstar")}, c_best=dict(consts={**k, "gamma": "inf", "phi_rig": "inf"}))
    _git(tmp_path, "commit", "-q", "-am", "cstar")
    monkeypatch.setattr(id3d, "CSTAR_SELECTED", False)
    with pytest.raises(ValueError, match="CSTAR_SELECTED"):
        R.e15_require_step1("cstar", str(p))
    monkeypatch.setattr(id3d, "CSTAR_SELECTED", True)
    monkeypatch.setattr(id3d, "CSTAR", dict(SY.P1_CSTAR))
    with pytest.raises(ValueError, match="c_best"):
        R.e15_require_step1("cstar", str(p))                                  # id3d.py not at the committed c_best
    monkeypatch.setattr(id3d, "CSTAR", k)
    assert R.e15_require_step1("cstar", str(p))
    _select(p, s_star=None, best={})
    _git(tmp_path, "commit", "-q", "-am", "infeasible")
    with pytest.raises(ValueError, match="no feasible"):
        R.e15_require_step1("vote", str(p))
    monkeypatch.setattr(R, "subprocess", SimpleNamespace(run=lambda *a, **k: (_ for _ in ()).throw(OSError("no git"))))
    assert R.e15_git_committed(str(p)) == (False, "git unavailable (no git)")


def test_theta_20_is_the_stress_row_only():
    """Review 0 #7: the decode CLIs take theta 20 for the AB row (the stress arm) only."""
    R.e15_cli_check(None)
    R.e15_cli_check(R.e15_options(_r("--relay", "20", "--id3d", "vote")))
    R.e15_cli_check(R.e15_options(_r("--relay", "35")))
    for extra in (["--relay", "20"], ["--relay", "20", "--relay-trust", "obj"], ["--relay", "20", "--id3d", "vote",
                                                                                    "--relay-mode", "jacobi"],
                  ["--relay", "20", "--id3d", "vote", "--id3d-scope", "global"]):
        with pytest.raises(ValueError, match="stress arm only"):
            R.e15_cli_check(R.e15_options(_r(*extra)))


def test_run_cli_refuses_before_loading_the_model(monkeypatch):
    """run.py main(): theta 20 (the stress arm is DEV-A only; no T06 arm has it, I7) and Fix-A decodes before the Step-1
    commit stop before any model load."""
    monkeypatch.setattr(R, "E15_SELECT", os.path.join(ROOT, "no_such_dir", "select.json"))
    for extra, msg in ((["--relay", "20", "--relay-trust", "obj"], "stress arm, registered on DEV-A only"),
                       (["--relay", "20", "--id3d", "vote"], "no run.py / T06 arm has theta 20"),    # I7: even the AB row
                       (["--relay", "60", "--id3d", "vote"], "Step-1 commit"),
                       (["--relay", "47", "--id3d", "vote"], "registered threshold")):
        monkeypatch.setattr(sys, "argv", ["run.py", "--scene", "T06_flag_ganggang_F", "--out", "e15_x", "--arm", "geoset_pred_ft",
                                          "--gate", "reproj", "--xcl", "outs", *extra])
        with pytest.raises(SystemExit, match=msg):
            R.main()


def test_cstar_constants_lie_on_the_registered_grid():
    assert set(id3d.CSTAR) == set(id3d.CSTAR_GRID)
    assert all(id3d.CSTAR[k] in id3d.CSTAR_GRID[k] for k in id3d.CSTAR)
    assert math.prod(len(v) for v in id3d.CSTAR_GRID.values()) == 648


def test_check_gate_out_refuses_the_arms_own_folder_with_relay():
    with pytest.raises(SystemExit):
        R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, "none", False, 60.0)
    with pytest.raises(SystemExit):
        R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, "none", False, math.inf)
    R.check_gate_out("geoset_pred_ft", "reproj", "e15_ab", "pred", None, "outs", False, 60.0)
    R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, "none")             # flags off: as before
    R.check_gate_out("geoset_pred_ft", "view", "geoset_pred_ft", "pred", None, "none", False, None)


def test_presets_unchanged_until_the_second_registration():
    """I5: the main preset geoset_rep_xs_ab is appended after geoset_rep_xs at the second commit (§6.6), and it is the
    only E15 preset; no earlier preset carries an E15 key (their decodes stay the flags-off path)."""
    names = list(S.PRESETS)
    i = names.index("geoset_rep_xs")
    assert names[:i + 1] == ["geoset_rep", "geoset_abs", "geoset_dec_rep", "geoset_decb_rep", "geoalign_rep", "geoset_rep_z2",
                             "geoset_rep_z2r", "geoset_rep_x", "geoset_rep_xm", "geoset_cutall_rep_x", "geoset_rep_xs"]
    assert all(not (set(S.PRESETS[n]) & set(R.E15_KEYS)) for n in names[:i + 1])
    assert names[i + 1:] in ([], ["geoset_rep_xs_ab"])
    for n in names[i + 1:]:
        o = R.e15_options(SimpleNamespace(**S.PRESETS[n]))
        assert o is not None and o["row"] == "AB" and o["scope"] == "relay"


def test_e15_choices_match_the_registration():
    a = _r("--relay", "60")
    import argparse
    ap = argparse.ArgumentParser()
    R._extra(ap)
    ch = {x.dest: tuple(x.choices) for x in ap._actions if x.choices}
    assert ch["id3d"] == ("off", "c0", "cstar", "vote", "vote_inf", "hybrid")
    assert ch["id3d_scope"] == ("relay", "global", "global_nf")
    assert ch["relay_mode"] == ("seq", "jacobi") and ch["relay_order"] == ("hop", "angle")
    assert ch["relay_mem"] == ("pool", "latest") and ch["relay_trust"] == ("id3d", "reproj", "obj", "gt")
    assert ch["flagged"] == ("keep", "chain", "empty")
    assert a.relay == 60.0 and R.E15_XCL == ("outs",)
    assert R.E15_FOLD == {"T06_flag_ganggang_F": 2}                                  # I9


def test_in_process_stage1_passes_relay_to_check_gate_out(monkeypatch):
    """Review 0 #8: Stage1.run checks the preset's relay key and keeps relay.json / id3d_conf.json."""
    seen = {}

    def fake_check(*a, **k):
        seen["relay"] = k.get("relay")

    def fake_decode(scene, args, sink=None):
        sink.json("final", "relay.json", {"F": []})
        sink.json("final", "id3d_conf.json", {"final": {}})
        return {"arm": args.arm}
    monkeypatch.setattr(R, "check_gate_out", fake_check)
    monkeypatch.setattr(R, "decode_scene", fake_decode)
    monkeypatch.setitem(S.PRESETS, "_e15_probe", dict(arm="geoset_pred_ft", gate="reproj", xcl="outs", relay=60.0, id3d="vote"))
    st = S.Stage1("_e15_probe")
    monkeypatch.setattr(st, "load", lambda: None)
    sc = common.Scene(name="Syn", cams=["a", "b"], ref_cam="a", mask_cams=["a"], hw=(4, 4), image_paths={},
                      ref_labels=np.ones((4, 4), np.uint8), objects=[1])
    monkeypatch.setattr(common, "load_scene", lambda data, name: sc)
    res = st.run("Syn")
    assert seen["relay"] == 60.0 and res["relay"] == {"F": []} and res["id3d_conf"] == {"final": {}}


# ------------------------------------------------------------------ dev_decode: I6, the recorded override, rotation
def test_dev_decode_restriction_rule(tmp_path):
    from methods.geomem import dev_decode as D
    d = SY.write_scene(str(tmp_path))
    lab = d["labels"][d["ref"]]
    sc = SimpleNamespace(cams=d["cams"], ref_cam=d["ref"], ref_labels=lab, objects=[1, 2])
    g = R.relay_geometry(d["cache"], d["cams"], d["ref"], np.stack([lab == 1, lab == 2]), 45.0)
    assert g["S"] == ["cam01", "cam11"] and "cam02" in g["F"]
    star, far = ["cam01", "cam11"], ["cam01", "cam02"]
    assert D.e15_may_restrict({"arm": "geoset_pred_ft"}, sc, d["cache"], far) == (True, [])        # no relay
    assert D.e15_may_restrict({"relay": 45.0}, sc, d["cache"], star) == (True, [])
    assert D.e15_may_restrict({"relay": 45.0, "id3d_scope": "global"}, sc, d["cache"], star) == (True, [])
    assert D.e15_may_restrict({"relay": 45.0}, sc, d["cache"], far) == (False, ["cam02"])          # an eval cam in F
    assert D.e15_may_restrict({"relay": 45.0, "id3d_scope": "global_nf"}, sc, d["cache"], star) == (False, [])
    assert D.e15_may_restrict({"relay": math.inf}, sc, d["cache"], far) == (True, [])


def test_relay_geometry_without_a_valid_cini_point(tmp_path):
    """Review 0 #6: no valid Pi3 point at c_ini -> no anchor, F empty, a recorded reason (the decode is C0), no crash."""
    d = SY.write_scene(str(tmp_path))
    p = os.path.join(d["cache"], f"{d['ref']}.npz")
    z = dict(np.load(p))
    z["conf"] = np.zeros_like(z["conf"])
    np.savez(p, **z)
    lab = d["labels"][d["ref"]]
    g = R.relay_geometry(d["cache"], d["cams"], d["ref"], np.stack([lab == 1, lab == 2]), 20.0)
    assert g["X"] is None and g["F"] == [] and g["S"] == d["cams"][1:] and "no valid Pi3" in g["fallback"]
    from methods.geomem import dev_decode as D
    sc = SimpleNamespace(cams=d["cams"], ref_cam=d["ref"], ref_labels=lab, objects=[1, 2])
    assert D.e15_may_restrict({"relay": 20.0}, sc, d["cache"], ["cam05"]) == (True, [])


def test_check_restricted_asserts_i6():
    from methods.geomem import dev_decode as D
    from methods.geomem.stage1 import MemorySink
    sink = MemorySink()
    sink.json("final", "relay.json", dict(options=dict(scope="relay"), F=["c2"]))
    D.check_restricted(sink, "S", 4, ["c1", "c3"], True)                     # no eval cam in F
    D.check_restricted(sink, "S", 4, ["c2"], False)                          # a full decode may have one
    with pytest.raises(AssertionError):
        D.check_restricted(sink, "S", 4, ["c2"], True)
    sink.json("final", "relay.json", dict(options=dict(scope="global_nf"), F=[]))
    with pytest.raises(AssertionError):
        D.check_restricted(sink, "S", 4, ["c1"], True)                       # GN never restricts
    D.check_restricted(MemorySink(), "S", 4, ["c2"], True)                   # no relay: nothing to check


def test_write_frame_keeps_the_e15_records(tmp_path):
    from methods.geomem import dev_decode as D
    from methods.geomem.stage1 import MemorySink
    sink = MemorySink()
    m = np.zeros((4, 4), bool)
    m[1, 1] = True
    sink.mask("final", "c1", 1, m)
    sink.mask("round1", "c2", 1, m)
    sink.json("final", "round1_conf.json", {"c1": {}})
    sink.json("final", "xcl_conf.json", {"c1": {"1": {}}, "c2": {"1": {}}})
    sink.json("final", "relay.json", {"theta": "inf", "F": []})
    sink.json("final", "id3d_conf.json", {"final": {}})
    sc = SimpleNamespace(name="S", ref_cam="c0", ref_labels=np.ones((4, 4), np.uint8), objects=[1])
    cfg = dict(eval_cams=["c1"], e15_arm="AB", e15_options=dict(theta="inf"))
    assert D.write_frame(str(tmp_path), sc, 4, sink, cfg) == 1
    d = tmp_path / "S" / "000004"
    assert json.load(open(d / "relay.json")) == {"theta": "inf", "F": []}
    assert json.load(open(d / "id3d_conf.json")) == {"final": {}}
    assert json.load(open(d / "xcl_conf.json")) == {"c1": {"1": {}}}            # eval cams only
    assert json.load(open(d / "config.json"))["e15_arm"] == "AB" and (d / "DONE").is_file()
    plain = MemorySink()
    plain.json("final", "round1_conf.json", {})
    D.write_frame(str(tmp_path / "p"), sc, 4, plain, dict(eval_cams=[]))
    assert not (tmp_path / "p" / "S" / "000004" / "relay.json").exists()


def _flags(**kw):
    out = {k: None for k in R.E15_KEYS}
    out.update(kw)
    return out


def _runs(tmp_path, monkeypatch, s_star="vote", theta_s="55", step2_extra=None, step2b=None, A_E=47.1, A_all=58.12):
    """scripts/e15/runs/{select,angles,step2,step2b}.json stand-ins for the I11 bundle table (theta_s None: no step2.json)."""
    from methods.geomem import dev_decode as D
    d = tmp_path / "runs"
    d.mkdir(exist_ok=True)
    for f in d.iterdir():
        f.unlink()
    json.dump(dict(s_star=dict(family=s_star), best_per_family={}), open(d / "select.json", "w"))
    json.dump(dict(summary=dict(A_E=dict(a=A_E), A_all=dict(a=A_all))), open(d / "angles.json", "w"))
    if theta_s is not None:
        json.dump(dict(family=s_star, rule=dict(theta_s=theta_s, **(step2_extra or {}))), open(d / "step2.json", "w"))
    if step2b is not None:
        json.dump(step2b, open(d / "step2b.json", "w"))
    monkeypatch.setattr(D, "E15_RUNS", str(d))
    return D


def _swap(to_family="hybrid", done=True, **extra):
    """A step2b.json record: s' = to_family; done = Step 2b finished with the swap kept."""
    rec = dict(rule=dict(swap=done, to="A-x"), alternatives={"A-x": dict(family=to_family)}, **extra)
    if done:
        rec["main_family"] = to_family
    return rec


def test_override_records_the_arm_and_refuses_dev_b(tmp_path, monkeypatch):
    D = _runs(tmp_path, monkeypatch)
    preset, o = D.e15_override("AB", _flags(relay=35.0, id3d="vote"), "A")
    assert preset == dict(S.PRESETS["geoset_rep_xs"], relay=35.0, id3d="vote") and o["row"] == "AB"
    rec = D.e15_record("AB", o)
    assert rec["e15_arm"] == "AB" and rec["e15_row"] == "AB" and rec["e15_options"]["theta"] == 35.0
    assert set(rec["e15_options"]) >= {"theta", "family", "scope", "mode", "order", "mem", "trust", "flagged", "row"}
    json.dumps(rec, allow_nan=False)
    assert D.e15_root("AB", o, "A").endswith(os.path.join("_dev", "e15", "AB_35_vote_A"))
    assert D.e15_root("AB", o, "A", 2).endswith("AB_35_vote_A_rot2")
    oi = dict(o, theta=math.inf)
    assert D.e15_root("AB", oi, "A").endswith("AB_inf_vote_A") and D.e15_record("AB", oi)["e15_options"]["theta"] == "inf"
    ob = R.e15_options(SimpleNamespace(**dict(S.PRESETS["geoset_rep_xs"], relay=60.0)))
    assert D.e15_root("B-rep", ob, "A", 1).endswith("B-rep_60_off_A_rot1")
    with pytest.raises(SystemExit, match="split B"):                                           # I8
        D.e15_override("AB", _flags(relay=35.0, id3d="vote"), "B")


@pytest.mark.parametrize("arm, flags, msg", [
    ("A", dict(relay=35.0, id3d="vote"), "AB row"),                       # the flags are AB's, not A's
    ("G", dict(relay=60.0, id3d="vote", relay_mode="jacobi"), "A row"),
    ("AB", dict(relay=20.0, id3d="vote"), "stress"),                     # theta 20 is the stress arm
    ("stress", dict(relay=35.0, id3d="vote"), "stress"),
    ("ABc", dict(relay=60.0, id3d="vote", flagged="chain", chain_dir="/c"), "T06 only"),
    ("oracle", dict(relay=60.0, relay_trust="gt", gt_dir="/g"), "T06 only"),
    ("AB", dict(id3d="vote"), "--relay"),
    ("AB", dict(relay=47.0, id3d="vote"), "registered threshold"),
    ("AB", dict(relay=60.0, id3d="vote", relay_mode="jacobi", flagged="empty"), "not a registered arm"),
])
def test_override_refuses_unregistered_runs(arm, flags, msg, tmp_path, monkeypatch):
    D = _runs(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match=msg):
        D.e15_override(arm, _flags(**flags), "A")


A_JAC = dict(relay_mode="jacobi")


def _ok(D, arm, flags, rot, family=None):
    fl = dict(flags, **({"id3d": family} if family else {}))
    return D.e15_override(arm, _flags(**fl), "A", rot)[1]


def test_override_bundles_before_and_after_step2(tmp_path, monkeypatch):
    """§4 I11: the registered (arm, set, theta, family) bundles, s* = vote, A_E 47.1, A_all 58.12."""
    D = _runs(tmp_path, monkeypatch, theta_s=None)                           # Step 2 not run yet
    assert set(D.E15_DEV_ARMS.values()) <= set(R.E15_ROWS)
    for th in D.E15_GRID:                                                    # Step 2 DEV-A' grid, s*
        assert _ok(D, "AB", dict(relay=th), 1, "vote")["row"] == "AB" and _ok(D, "AB", dict(relay=th), 2, "vote")
    for th in (35.0, 40.0, 45.0, 50.0, 55.0, 60.0):                         # DEV-A: < A_E (Step 2) and Z-R (60, [A_E, A_all))
        assert _ok(D, "AB", dict(relay=th), 0, "vote")
    assert _ok(D, "stress", dict(relay=20.0), 0, "vote") and _ok(D, "G", dict(relay=60.0, id3d_scope="global"), 0, "vote")
    assert _ok(D, "GN", dict(relay=60.0, id3d_scope="global_nf"), 0, "vote")
    refused = [("AB", dict(relay=60.0), 1, "hybrid", "finite grid with s"),       # not s*
               ("AB", dict(relay=math.inf), 1, "vote", "finite grid"),
               ("AB", dict(relay=45.0), 0, "hybrid", "grid theta < A_E"),
               ("stress", dict(relay=20.0), 1, "vote", "stress = DEV-A"),
               ("stress", dict(relay=20.0), 0, "hybrid", "stress = DEV-A"),
               ("G", dict(relay=60.0, id3d_scope="global"), 1, "vote", "G = DEV-A, theta 60"),
               ("G", dict(relay=35.0, id3d_scope="global"), 0, "vote", "G = DEV-A, theta 60"),
               ("GN", dict(relay=60.0, id3d_scope="global_nf"), 2, "vote", "GN = DEV-A, theta 60"),
               ("A", dict(relay=55.0, **A_JAC), 1, "vote", "Step 2 / 2b are not finished"),
               ("A", dict(relay=55.0, **A_JAC), 0, "vote", "A = DEV-A' "),
               ("B-rep", dict(relay=55.0), 1, None, "Step 2 is not finished"),
               ("B-obj", dict(relay=55.0, relay_trust="obj"), 0, None, r"DEV-A' \(--ref-rot 1, 2\) only")]
    for arm, flags, rot, fam, msg in refused:
        with pytest.raises(SystemExit, match=msg):
            _ok(D, arm, flags, rot, fam)
    # Step 2 finished at theta_s* = 55, Step 2b with no swap (main = s*)
    D = _runs(tmp_path, monkeypatch, theta_s="55", step2b=dict(rule=dict(swap=False, to=None), alternatives={},
                                                               main_family="vote"))
    assert _ok(D, "B-rep", dict(relay=55.0), 1) and _ok(D, "B-obj", dict(relay=55.0, relay_trust="obj"), 2)
    assert _ok(D, "A", dict(relay=55.0, **A_JAC), 1, "vote") and _ok(D, "A", dict(relay=60.0, **A_JAC), 2, "vote")
    for arm, flags, rot, fam, msg in (("B-rep", dict(relay=60.0), 1, None, "theta_s\\* = 55 only"),
                                      ("A", dict(relay=45.0, **A_JAC), 1, "vote", "theta_s"),
                                      ("A", dict(relay=60.0, **A_JAC), 1, "hybrid", "s_main"),
                                      ("AB", dict(relay=60.0), 1, "hybrid", "60 with s' after a swap")):
        with pytest.raises(SystemExit, match=msg):
            _ok(D, arm, flags, rot, fam)


def test_override_bundles_of_a_swap(tmp_path, monkeypatch):
    """§6.5 / §6.7: s' re-checks at theta_s* (DEV-A', and DEV-A when theta_s* < A_E); after the swap AB(s', 60) on DEV-A'
    and A(s_main = s', 60 / theta_s*), Z-R with s' too."""
    D = _runs(tmp_path, monkeypatch, theta_s="45", step2b=_swap(done=False, need_devA2="AB_45_hybrid_A_rot{1,2}"))
    assert _ok(D, "AB", dict(relay=45.0), 1, "hybrid") and _ok(D, "AB", dict(relay=45.0), 0, "hybrid")   # 45 < A_E
    with pytest.raises(SystemExit, match="s_main"):
        _ok(D, "A", dict(relay=45.0, **A_JAC), 1, "hybrid")                  # Step 2b not finished
    with pytest.raises(SystemExit):
        _ok(D, "AB", dict(relay=60.0), 1, "hybrid")                          # no swap yet
    D = _runs(tmp_path, monkeypatch, theta_s="55", step2b=_swap())
    assert _ok(D, "AB", dict(relay=55.0), 2, "hybrid") and _ok(D, "AB", dict(relay=60.0), 1, "hybrid")
    assert _ok(D, "A", dict(relay=60.0, **A_JAC), 1, "hybrid") and _ok(D, "AB", dict(relay=60.0), 0, "hybrid")   # Z-R s'
    assert _ok(D, "AB", dict(relay=55.0), 0, "hybrid")                     # Z-R case Z at theta_s* in [A_E, A_all)
    with pytest.raises(SystemExit, match="grid theta < A_E"):
        _ok(D, "AB", dict(relay=45.0), 0, "hybrid")                          # s' at a theta < A_E other than theta_s*
    with pytest.raises(SystemExit, match="s_main"):
        _ok(D, "A", dict(relay=60.0, **A_JAC), 1, "vote")                    # A runs with s_main = s'
    # theta_s* = inf (relay closed): s_main = s*, the (b) proxy A(s*, 60) is registered, the theta_s* arms are not
    D = _runs(tmp_path, monkeypatch, theta_s="inf")
    assert _ok(D, "A", dict(relay=60.0, **A_JAC), 1, "vote")
    with pytest.raises(SystemExit, match="Step 2 is not finished or chose inf"):
        _ok(D, "B-rep", dict(relay=60.0), 1)


def test_override_bundles_wait_for_a_finished_step2(tmp_path, monkeypatch):
    for extra in (dict(need_devA="45"), dict(need_zcheck=["ZR 55"])):
        D = _runs(tmp_path, monkeypatch, theta_s=None, step2_extra=extra)
        import json as _j
        _j.dump(dict(family="vote", rule=dict(theta_s=None, **extra)), open(tmp_path / "runs" / "step2.json", "w"))
        assert D.e15_selection()["theta_s"] is None
        with pytest.raises(SystemExit, match="not finished"):
            _ok(D, "B-rep", dict(relay=55.0), 1)
    D = _runs(tmp_path, monkeypatch)
    os.remove(tmp_path / "runs" / "angles.json")
    with pytest.raises(SystemExit, match="angles.json"):
        _ok(D, "AB", dict(relay=60.0), 0, "vote")
    os.remove(tmp_path / "runs" / "select.json")
    assert D.e15_selection()["s_star"] is None
    with pytest.raises(SystemExit):
        _ok(D, "AB", dict(relay=60.0), 1, "vote")


@needs_data
def test_dev_decode_prepares_the_dev_index_on_the_real_data(tmp_path, monkeypatch):
    """Review 0 #1: dev_decode's index and default scenes are the 17 MUVOD fold scenes, never the Data/ listing (T06 has no
    train_frames and one annotated camera); main() gets to the model load on the real listing.  Nothing is decoded."""
    from geomem.folds import load_folds
    from methods.geomem import dev_decode as D
    folds = sorted(load_folds())
    assert devset.dev_scenes() == folds and len(folds) == 17
    assert set(common.all_scenes(DATA)) >= set(folds)                         # the listing may hold more (T06)
    for argv, n_frames in (([], 85), (["--ref-rot", "1"], 85), (["--ref-rot", "2"], 85), (["--split", "B"], 255)):
        root = tmp_path / ("r" + "_".join(argv))
        a, preset, rec, scenes, idx = D.prepare(["--preset", "geoset_rep_xs", "--root", str(root), *argv])
        assert scenes == folds and sorted(idx) == folds and rec is None and preset == "geoset_rep_xs"
        assert sum(len(v["frames"]) for v in idx.values()) == n_frames
        written = json.load(open(root / "index.json"))
        assert sorted(written["scenes"]) == folds and written.get("rot", 0) == (int(argv[1]) if "--ref-rot" in argv else 0)
    a, _, _, scenes, _ = D.prepare(["--preset", "geoset_rep_xs", "--root", str(tmp_path / "d"), "--scenes", "Dog"])
    assert scenes == ["Dog"]
    with pytest.raises(SystemExit):
        D.prepare(["--preset", "geoset_rep_xs", "--root", str(tmp_path / "t"), "--scenes", "T06_flag_ganggang_F"])
    if "geoset_rep_xs_ab" not in S.PRESETS:
        with pytest.raises(SystemExit):                                       # DEV-B' before the second commit
            D.prepare(["--preset", "geoset_rep_xs", "--root", str(tmp_path / "b"), "--split", "B", "--ref-rot", "1"])
    monkeypatch.setattr(R, "E15_SELECT", str(tmp_path / "none" / "select.json"))
    with pytest.raises(SystemExit):                                           # a Fix-A override before the Step-1 commit
        D.prepare(["--preset", "geoset_rep_xs", "--root", str(tmp_path / "f"), "--e15-arm", "AB", "--relay", "60",
                   "--id3d", "vote"])

    class Loaded(Exception):
        pass

    def stop():
        raise Loaded
    monkeypatch.setattr(D, "_load_model", stop)
    root = tmp_path / "main"
    with pytest.raises(Loaded):
        D.main(["--preset", "geoset_rep_xs", "--root", str(root)])
    assert sorted(json.load(open(root / "index.json"))["scenes"]) == folds


def _meta(tmp_path, name="Rot", mask_cams=("c0", "c1", "c2"), ref="c1"):
    d = tmp_path / name
    d.mkdir(parents=True)
    json.dump(dict(ref_frame=0, ref_cam=ref, cams=list(mask_cams) + ["c3"], mask_cams=list(mask_cams),
                   train_frames=list(range(21))), open(d / "meta.json", "w"))
    return str(tmp_path)


def test_reference_rotation_definitions(tmp_path, monkeypatch):
    data = _meta(tmp_path)
    assert devset.ROTS == (0, 1, 2)
    assert devset.ref_cam(data, "Rot", 0) == "c1" and devset.eval_cams(data, "Rot", 0) == ["c0", "c2"]
    assert devset.ref_cam(data, "Rot", 1) == "c0" and devset.eval_cams(data, "Rot", 1) == ["c1", "c2"]
    assert devset.ref_cam(data, "Rot", 2) == "c2" and devset.eval_cams(data, "Rot", 2) == ["c0", "c1"]
    assert devset.eval_cams(data, "Rot") == devset.eval_cams(data, "Rot", 0)
    with pytest.raises(AssertionError):
        devset.ref_cam(data, "Rot", 3)
    monkeypatch.setattr(devset, "basic_set", lambda s: [1, 2])
    i0, i2 = devset.index(data, ["Rot"], "A"), devset.index(data, ["Rot"], "A", 2)
    assert "rot" not in i0 and "orig_ref_cam" not in i0["scenes"]["Rot"]                  # rotation 0 = as before
    assert i2["rot"] == 2 and i2["scenes"]["Rot"]["ref_cam"] == "c2" and i2["scenes"]["Rot"]["orig_ref_cam"] == "c1"
    assert i2["scenes"]["Rot"]["eval_cams"] == ["c0", "c1"] and i2["scenes"]["Rot"]["frames"] == i0["scenes"]["Rot"]["frames"]


@needs_data
def test_frame_scene_rotation_on_muvod():
    """Blocks (mask_cams cam0, cam4, cam9; c_ini cam4): rotation 1 -> input cam0 (its GT at t, basic set), eval cam4 /
    cam9; objects = the basic objects present at cam0."""
    from methods.geomem import dev_decode as D
    from mvseg2d.io import load_labelmap
    basic = devset.basic_set("Blocks")
    t = devset.dev_frames(DATA, "Blocks", "A")[0]
    s0, s1 = D.frame_scene("Blocks", t, basic), D.frame_scene("Blocks", t, basic, 1)
    assert s0.ref_cam == "cam4" and s1.ref_cam == "cam0" and D.eval_cams("Blocks", 1) == ["cam4", "cam9"]
    lab = load_labelmap(os.path.join(DATA, "Blocks", "frames", f"{t:06d}", "masks", "cam0.png"))
    expect = sorted(int(o) for o in np.unique(lab) if o and int(o) in basic)
    assert s1.objects == expect and np.array_equal(s1.ref_labels, np.where(np.isin(lab, basic), lab, 0))
    assert s1.cams == s0.cams and s1.image_paths == s0.image_paths


def test_dev_compare_pairs_rotated_rows(tmp_path):
    from mvseg2d import dev_compare as C

    def rows(jf):
        out = []
        for ref, cams in (("r1", ("a", "b")), ("r2", ("a", "c"))):
            for cam in cams:
                out.append(dict(scene="S", t=4, ref=ref, cam=cam, obj=1, J=jf, F=jf, JF=jf, kind="x"))
        return out
    for name, jf in (("b1", 0.5), ("c1", 0.6)):
        json.dump(dict(rows=[r for r in rows(jf) if r["ref"] == "r1"]), open(tmp_path / f"{name}.json", "w"))
        json.dump(dict(rows=[r for r in rows(jf) if r["ref"] == "r2"]), open(tmp_path / f"{name[0]}2.json", "w"))
    base = C.load(f"{tmp_path / 'b1.json'},{tmp_path / 'b2.json'}")
    cand = C.load(f"{tmp_path / 'c1.json'},{tmp_path / 'c2.json'}")
    assert {r["view"] for r in base} == {"4/r1/a", "4/r1/b", "4/r2/a", "4/r2/c"}         # cam "a" twice, distinct views
    res = C.compare(base, cand, n=100)
    assert res["n_pairs"] == 4 and res["paired"]["JF"]["delta"] == pytest.approx(0.1)
    assert C.pair_key(dict(scene="S", t=4, cam="a", obj=1)) == ("S", 4, "", "a", 1)
    with pytest.raises(SystemExit, match="duplicated"):
        C.compare(base + base[:1], cand, n=10)


def test_decode_frame_restriction_override_and_rotation(tmp_path, monkeypatch):
    """dev_decode.decode_frame end to end with a stand-in decode_scene: the I6 restriction from the inputs, the base
    preset name recorded for a preset dict, the rotation fields, and the I6 assertion on the decode's relay record."""
    from methods.geomem import dev_decode as D
    d = SY.write_scene(str(tmp_path))
    meta_p = os.path.join(d["data"], "Syn", "meta.json")
    meta = json.load(open(meta_p))
    meta["mask_cams"] = ["cam00", "cam01", "cam11"]
    json.dump(meta, open(meta_p, "w"))
    monkeypatch.setattr(D, "DATA", d["data"])
    monkeypatch.setattr(D, "pi3_frame_cache", lambda data, scene, t: d["cache"])
    monkeypatch.setattr(R, "cache_dir", R.cache_dir)                         # decode_frame re-points it
    seen = {}

    def fake_decode(sc_run, args, sink=None):
        o = R.e15_options(args)
        seen.update(cams=list(sc_run.cams), ref=sc_run.ref_cam, relay=getattr(args, "relay", None))
        if o is not None:
            masks = np.stack([sc_run.ref_labels == q for q in sc_run.objects])
            g = R.relay_geometry(d["cache"], sc_run.cams, sc_run.ref_cam, masks, o["theta"])
            sink.json("final", "relay.json", dict(options=dict(scope=o["scope"]), F=seen.get("force_F", g["F"]), S=g["S"]))
        return {"arm": args.arm}
    monkeypatch.setattr(R, "decode_scene", fake_decode)

    def scene(rot):
        ref = devset.ref_cam(d["data"], "Syn", rot)
        lab = d["labels"][ref]
        return common.Scene(name="Syn", cams=d["cams"], ref_cam=ref, mask_cams=meta["mask_cams"], hw=(SY.H, SY.W),
                            image_paths={}, ref_labels=lab, objects=[1, 2])
    ab = dict(S.PRESETS["geoset_rep_xs"], relay=45.0, id3d="vote")
    # rot 0: eval cams cam01 / cam11 (30 deg) are star views at 45 -> the restricted set (c_ini + evals + k3)
    _, cfg = D.decode_frame(scene(0), 4, restrict=True, preset=ab)
    assert cfg["restricted"] and set(seen["cams"]) < set(d["cams"]) and cfg["preset"] == "geoset_rep_xs"
    assert cfg["eval_cams"] == ["cam01", "cam11"] and "ref_rot" not in cfg
    # rot 1: input cam01, eval cam00 / cam11; cam11 is 60 deg from cam01 -> F -> the full camera set (I6)
    _, cfg = D.decode_frame(scene(1), 4, restrict=True, preset=ab, rot=1)
    assert not cfg["restricted"] and seen["cams"] == d["cams"] and seen["ref"] == "cam01"
    assert cfg["eval_cams"] == ["cam00", "cam11"] and (cfg["ref_rot"], cfg["ref_cam"], cfg["orig_ref_cam"]) == (1, "cam01", "cam00")
    # a plain preset name is recorded as is and never consults the relay geometry
    _, cfg = D.decode_frame(scene(0), 4, restrict=True, preset="geoset_rep_xs")
    assert cfg["preset"] == "geoset_rep_xs" and cfg["restricted"] and seen["relay"] is None
    # the I6 assertion fires when a restricted decode reports an eval cam in F
    seen["force_F"] = ["cam01"]
    with pytest.raises(AssertionError):
        D.decode_frame(scene(0), 4, restrict=True, preset=ab)
    with pytest.raises(AssertionError):                                       # the scene must carry the rotation's c_ini
        D.decode_frame(scene(0), 4, restrict=True, preset="geoset_rep_xs", rot=1)


# ------------------------------------------------------------------ run_e15.sh refusals (never reach docker)
RUNNER = os.path.join(ROOT, "scripts", "e15", "run_e15.sh")


def _runner(tmp_path, *argv):
    """run_e15.sh with a stand-in `docker` first on PATH that only records a call: these refusals must exit before any
    container (GPU or CPU) starts."""
    import subprocess
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    marker = tmp_path / "docker_called"
    (bin_ / "docker").write_text(f"#!/bin/sh\ntouch {marker}\nexit 99\n")
    (bin_ / "docker").chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}")
    r = subprocess.run(["bash", RUNNER, *argv], capture_output=True, text=True, env=env, timeout=60)
    return r.returncode, r.stdout + r.stderr, marker.exists()


@pytest.mark.parametrize("argv, msg", [
    (("t06", "stress", "20", "vote"), "DEV-A only"),                       # review 0 #7: the stress arm is not a T06 arm
    (("t06", "AB", "20", "vote"), "DEV-A only"),
    (("dev-c0", "A", "0"), "rot 1 or 2"),
    (("c0-rescore", "C"), "split A or B"),
])
def test_runner_refuses_before_any_container(tmp_path, argv, msg):
    code, out, docker = _runner(tmp_path, *argv)
    assert code == 2 and msg in out and not docker


def test_runner_guards_fix_a_and_dev_b(tmp_path):
    """Review 0 #8 / 1 #4: run_e15.sh checks the Step-1 commit on the host before a Fix-A decode; DEV-B C0 re-scoring
    waits for the second registration commit."""
    if os.path.isfile(R.E15_SELECT):
        pytest.skip("runs/select.json exists: the guard would let the decode start")
    for argv in (("dev", "AB", "60", "vote", "0"), ("t06", "AB", "60", "hybrid")):
        code, out, docker = _runner(tmp_path, *argv)
        assert code != 0 and not docker and ("Step-1 commit" in out or "run t06-c0 first" in out)
    code, out, docker = _runner(tmp_path, "dev", "AB", "60", "vote", "0")
    assert "Step-1 commit" in out
    if "geoset_rep_xs_ab" not in S.PRESETS:
        code, out, docker = _runner(tmp_path, "c0-rescore", "B")
        assert code == 2 and "second registration commit" in out and not docker

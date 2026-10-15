"""S1-E14 judge (MVSeed/e14_judge.py, docs/stage1-E14-prereg.md section 4-5): the registered rules on synthetic inputs (host)."""
import importlib.util
import json
import os

import cv2
import numpy as np
import pytest

MAIN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("e14_judge", os.path.join(MAIN, "MVSeed", "e14_judge.py"))
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)


def _jf(delta, lo, per_scene):
    return dict(delta=delta, ci=[lo, lo + 0.01], per_scene=per_scene)


def test_constants_as_registered():
    assert (J.SHIFT, J.TAU_KEEP, J.AUDIT_B, J.B3_FLOOR, J.AUDIT_KF, J.SAFETY, J.DEFENCE) == (0.5, 0.5, 0.006, -0.010, 0.008, -0.01, -1.0)
    assert (J.BASE_PRESET, J.ARM_PRESET, J.C0, J.ARM) == ("geoset_rep_x", "geoset_rep_xs", "geoset_pred_ft_rep_x_e11",
                                                            "geoset_pred_ft_rep_xs_e14")
    assert J.HEADLINE == "SegMaskSam3XW1CGPS4MSdgeoset_rep_xFsCs10kG2Rs" and J.METHOD2 == "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs"


def test_dev_b_rule_boundaries():
    ok = J.dev_b_rule(_jf(0.003, 0.001, {"a": 0.001, "b": -0.010}), dict(delta=0.0))
    assert ok["pass"] and ok["B3"] and ok["B2"] and ok["worst_scene"] == "b"            # scene exactly -0.010 passes; good 0 passes
    assert not J.dev_b_rule(_jf(0.003, 0.0, {"a": 0.001}), dict(delta=0.001))["B1"]    # CI lower bound exactly 0 fails
    assert not J.dev_b_rule(_jf(0.003, 0.001, {"a": -0.0100001}), dict(delta=0.001))["B3"]
    assert not J.dev_b_rule(_jf(0.003, 0.001, {"a": 0.001}), dict(delta=-1e-9))["B2"]
    nan = float("nan")
    assert not J.dev_b_rule(_jf(nan, nan, {"a": 0.001}), dict(delta=0.001))["pass"]
    assert not J.dev_b_rule(_jf(0.003, 0.001, {"a": nan}), dict(delta=0.001))["B3"]
    assert not J.dev_b_rule(_jf(0.003, 0.001, {"a": 0.001}), dict(delta=nan))["B2"]
    assert not J.dev_b_rule(_jf(0.003, 0.001, {}), dict(delta=0.001))["B3"]


def test_audit_and_keyframe_gate():
    assert not J.audit_needed(0.006, J.AUDIT_B) and J.audit_needed(0.00601, J.AUDIT_B) and J.audit_needed(float("nan"), J.AUDIT_B)
    assert not J.audit_needed(0.008, J.AUDIT_KF) and J.audit_needed(0.0081, J.AUDIT_KF)
    g = J.keyframe_gate(_jf(0.002, 0.0001, {"a": -0.01, "b": 0.003}))
    assert g["pass"] and g["worst_delta"] == -0.01
    assert not J.keyframe_gate(_jf(0.002, 0.0, {"a": 0.0}))["G1"]
    assert not J.keyframe_gate(_jf(0.002, 0.001, {"a": -0.0101}))["G2"]


def test_pair_check_and_object_class():
    rows = [dict(scene="S", t=1, cam="c", obj=o, JF=0.9) for o in (1, 2)]
    assert J.check_pairs(rows, list(rows))["ok"]
    assert not J.check_pairs(rows, rows[:1])["ok"]
    assert not J.check_pairs(rows, rows + rows[:1])["ok"]                                # a duplicated pair
    assert J.object_class(dict(exempt=False), dict(exempt=False)) == "same"
    assert J.object_class(dict(exempt=True), dict(exempt=True)) == "same"
    assert J.object_class(dict(exempt=False), dict(exempt=True)) == "on"
    assert J.object_class(dict(exempt=True), dict(exempt=False)) == "off"


def test_emul_keep_rule():
    L = np.full((2, 4, 16), -1024.0, np.float32)
    L[0, :, 0:4] = 1.0; L[0, :, 4:6] = -0.3; L[0, :, 6:8] = -0.6
    L[1, :, 2:4] = 0.8; L[1, :, 8:14] = 1.0; L[1, :, 14:16] = -0.3
    keep, exempt = J.emul_keep(L)
    assert keep[0, 0].tolist() == [True] * 6 + [False] * 10 and keep[1, 0].tolist() == [False] * 8 + [True] * 8
    assert exempt.tolist() == [False, False]
    keep0, _ = J.emul_keep(L, shift=0.0)
    assert keep0[0, 0].tolist() == [True] * 4 + [False] * 12
    nested = np.full((2, 8, 8), -1024.0, np.float32)
    nested[0] = 3.0; nested[1, 2:6, 2:6] = 1.0; nested[1, 0:2, :] = -0.2
    k, e = J.emul_keep(nested)
    assert e.tolist() == [False, True] and k[1].sum() == 32


# ------------------------------------------------------------------ frame_check on a synthetic DEV frame
def _frame(root, xcl, masks, exempt, shift=None, r1=None):
    d = os.path.join(root, "S", "000001")
    os.makedirs(os.path.join(d, "_r1", "c1"), exist_ok=True)
    cv2.imwrite(os.path.join(d, "_r1", "c1", "1.png"), (r1 if r1 is not None else np.ones((6, 8), np.uint8)) * 255)
    cfg = dict(xcl=xcl, xcl_tau_keep=0.5, objects=[1, 2], eval_cams=["c1"], seconds=1.0)
    if shift is not None:
        cfg["xcl_shift"] = shift
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    json.dump({"c1": {}}, open(os.path.join(d, "round1_conf.json"), "w"))
    recs = {}
    for o, m in masks.items():
        if m.any():
            os.makedirs(os.path.join(d, "c1"), exist_ok=True)
            cv2.imwrite(os.path.join(d, "c1", f"{o}.png"), m.astype(np.uint8) * 255)
        recs[str(o)] = dict(px=int(m.sum()) + 1, lost=0, exempt=exempt[o], kept=int(m.sum()), removed=0)
    json.dump({"c1": recs}, open(os.path.join(d, "xcl_conf.json"), "w"))
    open(os.path.join(d, "DONE"), "w").close()


@pytest.fixture
def dev(tmp_path, monkeypatch):
    data = tmp_path / "Data"
    gtd = data / "S" / "frames" / "000001" / "masks"; gtd.mkdir(parents=True)
    gt = np.zeros((6, 8), np.uint8); gt[:, :4] = 1; gt[:, 4:] = 2
    cv2.imwrite(str(gtd / "c1.png"), gt)
    monkeypatch.setattr(J, "DATA", str(data))
    return tmp_path


def _masks(cols1, cols2):
    m1, m2 = np.zeros((6, 8), bool), np.zeros((6, 8), bool)
    m1[:, cols1[0]:cols1[1]] = True; m2[:, cols2[0]:cols2[1]] = True
    return {1: m1, 2: m2}


def test_frame_check_superset_and_mechanism(dev):
    base, arm = str(dev / "base"), str(dev / "arm")
    _frame(base, "out", _masks((0, 3), (6, 8)), {1: False, 2: False})
    _frame(arm, "outs", _masks((0, 4), (5, 8)), {1: False, 2: False}, shift=0.5)
    r = J.frame_check(("S", 1, base, arm))
    assert r["bad"] == []
    c = r["counts"]
    assert c["added_px"] == 12 and c["added_unclaimed"] == 12 and c["added_own"] == 12 and c["removed_same"] == 0
    assert c["newly_claimed_px"] == 12 and c["newly_claimed_correct"] == 12 and c["object_views"] == 2


def test_frame_check_flags_a_removal_unless_exempt_flipped_off(dev):
    base, arm = str(dev / "base"), str(dev / "arm")
    _frame(base, "out", _masks((0, 4), (4, 8)), {1: False, 2: True})
    _frame(arm, "outs", _masks((0, 3), (5, 8)), {1: False, 2: False}, shift=0.5)
    bad = J.frame_check(("S", 1, base, arm))["bad"]
    kinds = [b[0] for b in bad]
    assert kinds == ["I2-removed"] and bad[0][4] == 1                              # object 1 (same flag) lost pixels: a bug
    c = J.frame_check(("S", 1, base, arm))["counts"]
    assert c["class_off"] == 1 and c["removed_off"] == 6                           # object 2 (exempt -> not): legitimate


def test_frame_check_config_round1_and_keyset(dev):
    base, arm = str(dev / "base"), str(dev / "arm")
    _frame(base, "out", _masks((0, 3), (6, 8)), {1: False, 2: False})
    m = _masks((0, 3), (6, 8)); m[2][:] = False
    r1 = np.zeros((6, 8), np.uint8)
    _frame(arm, "outs", m, {1: False, 2: False}, shift=0.25, r1=r1)
    kinds = {b[0] for b in J.frame_check(("S", 1, base, arm))["bad"]}
    assert {"I4", "I1-png", "I3-keyset", "I2-removed"} <= kinds


def test_devb_state():
    assert J.devb_state(None) == "missing" and J.devb_state({}) == "missing"
    assert J.devb_state(dict(audit_required=True, audited=False, delta=0.01)) == "audit"
    assert J.devb_state(dict(pairs={}, invariants={}, note="invariant violated")) == "bug"
    assert J.devb_state(dict(rule={"pass": True}, audit_required=False)) == "pass"
    assert J.devb_state(dict(rule={"pass": True}, audit_required=True, audited=True)) == "pass"
    assert J.devb_state(dict(rule={"pass": False}, audit_required=False)) == "fail"


def test_stage_b_audit_hold_saves_no_rule(tmp_path, monkeypatch):
    """Section 4: above +0.006 the judge stops (exit 4) and saves neither the rule nor the contrasts."""
    def rows(delta):
        out = []
        for i, sc in enumerate(J.SCENES):
            for t in (1, 2):
                for o in (1, 2):
                    jf = 0.9 + delta
                    out.append(dict(scene=sc, t=t, cam="c", obj=o, J=jf, F=jf, JF=jf, view=f"{t}/c", kind="boundary/mixed"))
        return out
    base, arm = rows(0.0), rows(0.01)
    monkeypatch.setattr(J, "OUT", str(tmp_path / "e14_judge.json"))
    monkeypatch.setattr(J, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(J, "_load_rows", lambda split, preset: (base, {}) if preset == J.BASE_PRESET else (arm, {}))
    monkeypatch.setattr(J, "dev_invariants", lambda split: dict(ok=True, n_bad=0, bad=[]))
    with pytest.raises(SystemExit) as e:
        J.stage_b(audited=False)
    assert e.value.code == 4
    saved = json.load(open(tmp_path / "e14_judge.json"))["stageB"]
    assert "rule" not in saved and "contrast" not in saved and J.devb_state(saved) == "audit"

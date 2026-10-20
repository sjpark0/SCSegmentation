"""S1-E15 judge (stage1/scripts/e15/e15_judge.py, docs/stage1-E15-prereg.md §5.3, §7, §8): the registered rules on
synthetic inputs (host)."""
import importlib.util
import json
import math
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("e15_judge", os.path.join(ROOT, "scripts", "e15", "e15_judge.py"))
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)


def test_constants_as_registered():
    assert (J.DELTA, J.SCENE_FLOOR, J.AUDIT_B, J.AUDIT_B2_WIDE, J.G1_J, J.G1_WRONG) == (-0.003, -0.010, 0.006, 0.05, 0.89, 5)
    assert (J.H_MIN, J.H3_SHARE, J.H2_SHARE) == (8, 0.5, 0.25)
    assert (J.C0_PRESET, J.MAIN_PRESET) == ("geoset_rep_xs", "geoset_rep_xs_ab")


def _frame(d, eval_pngs, r1, xcl, extra_cam=False):
    os.makedirs(d, exist_ok=True)
    for cam, objs in eval_pngs.items():
        os.makedirs(os.path.join(d, cam), exist_ok=True)
        for o, b in objs.items():
            open(os.path.join(d, cam, f"{o}.png"), "wb").write(b)
    for cam, objs in r1.items():
        os.makedirs(os.path.join(d, "_r1", cam), exist_ok=True)
        for o, b in objs.items():
            open(os.path.join(d, "_r1", cam, f"{o}.png"), "wb").write(b)
    conf = {cam: {"1": dict(iou=0.9, presence=1.0, overlap=0.8, n_valid=100)} for cam in r1}
    if extra_cam:
        conf["cX"] = {"1": dict(iou=0.1, presence=1.0, overlap=None, n_valid=0)}
    json.dump(conf, open(os.path.join(d, "round1_conf.json"), "w"), indent=1)
    json.dump(xcl, open(os.path.join(d, "xcl_conf.json"), "w"), indent=1)


def test_frame_compare_case_z_and_invariants(tmp_path):
    ev = {"c1": {1: b"a"}, "c2": {1: b"b"}}
    r1 = {"c1": {1: b"r"}, "c3": {1: b"s"}}
    xcl = {"c1": {"1": {"px": 3}}, "c2": {"1": {"px": 4}}}
    _frame(tmp_path / "c0", ev, r1, xcl)
    _frame(tmp_path / "same", ev, r1, dict(xcl, c9={"1": {"px": 1}}))           # a non-eval cam's record may differ
    assert J.frame_compare(str(tmp_path / "c0"), str(tmp_path / "same"), ["c1", "c2"], True, []) == []
    _frame(tmp_path / "png", {"c1": {1: b"a"}, "c2": {1: b"X"}}, r1, xcl)
    assert J.frame_compare(str(tmp_path / "c0"), str(tmp_path / "png"), ["c1", "c2"], True, []) == [("png", "c2")]
    _frame(tmp_path / "x", ev, r1, dict(xcl, c2={"1": {"px": 5}}))
    assert J.frame_compare(str(tmp_path / "c0"), str(tmp_path / "x"), ["c1", "c2"], True, []) == [("xcl_conf", "c2")]
    _frame(tmp_path / "r", ev, r1, xcl, extra_cam=True)                       # case Z: round1_conf compared as bytes
    assert J.frame_compare(str(tmp_path / "c0"), str(tmp_path / "r"), ["c1", "c2"], True, []) == [("round1_conf", "*")]
    # not case Z (a full decode): C0's round-1 cams by entry and PNG bytes; only the star eval cams' PNGs
    assert J.frame_compare(str(tmp_path / "c0"), str(tmp_path / "r"), ["c1", "c2"], False, ["c1"]) == []
    assert J.frame_compare(str(tmp_path / "c0"), str(tmp_path / "png"), ["c1", "c2"], False, ["c1"]) == []
    assert J.frame_compare(str(tmp_path / "c0"), str(tmp_path / "png"), ["c1", "c2"], False, ["c1", "c2"]) == [("png", "c2")]
    _frame(tmp_path / "r1bad", ev, {"c1": {1: b"Q"}, "c3": {1: b"s"}}, xcl)
    assert J.frame_compare(str(tmp_path / "c0"), str(tmp_path / "r1bad"), ["c1", "c2"], False, []) == [("r1_png", "c1")]


def test_case_z_from_angles():
    summ = {"in_F": {"45": {"B": {"n_eval_in_F": 3}}, "50": {"B": {"n_eval_in_F": 0}}}}
    assert J.case_z(summ, 50.0) and not J.case_z(summ, 45.0) and J.case_z(summ, math.inf)
    with pytest.raises(ValueError):
        J.case_z(summ, 55.0)


def _jf(lo, per):
    return dict(delta=0.0, ci=[lo, lo + 0.01], per_scene=per)


def test_rule_b():
    ok = dict(n_pairs=10, dJ=0.0, other_arm=3, other_c0=3)
    r = J.rule_b(_jf(-0.0029, {"a": -0.010}), _jf(-0.002, {"a": 0.0}), ok, True)
    assert r["pass"] and r["labels"]["B1"] == r["labels"]["B2"] == "vacuous (case Z)" and r["fail_type"] is None
    r = J.rule_b(_jf(-0.003, {"a": 0.0}), _jf(0.0, {"a": 0.0}), ok, False)
    assert not r["B1"] and r["fail_type"] == "DEV-B" and r["labels"]["B1"] is None
    r = J.rule_b(_jf(0.0, {"a": -0.0101}), _jf(0.0, {"a": 0.0}), ok, False)
    assert not r["B2"] and r["fail_type"] == "DEV-B"
    r = J.rule_b(_jf(0.0, {"a": 0.0}), _jf(0.0, {"a": -0.011}), ok, False)
    assert not r["B3"] and r["fail_type"] == "DEV-B'"
    r = J.rule_b(_jf(-0.01, {"a": 0.0}), _jf(0.0, {"a": 0.0}), dict(ok, dJ=-0.0001), False)
    assert not r["B4"] and r["fail_type"] == "DEV-B'"                          # DEV-B' failures route first
    assert not J.rule_b(_jf(0.0, {"a": 0.0}), _jf(0.0, {"a": 0.0}), dict(ok, other_arm=4), False)["B4"]
    r = J.rule_b(_jf(0.0, {"a": 0.0}), _jf(0.0, {"a": 0.0}), dict(n_pairs=0), True)
    assert r["B4"] and "vacuous" in r["B4_label"]
    assert not J.rule_b(_jf(float("nan"), {"a": 0.0}), _jf(0.0, {"a": 0.0}), ok, False)["B1"]


def test_audit_needed():
    assert not J.audit_needed(0.006, 0.05) and J.audit_needed(0.00601, None) and J.audit_needed(-0.0061, None)
    assert J.audit_needed(0.0, 0.0501) and not J.audit_needed(0.0, None) and J.audit_needed(float("nan"), None)



def _b(fail=None, z=False):
    return dict(**{"pass": fail is None}, fail_type=fail, case_z=z)


GOOD, BAD = dict(J_all=0.90, wrong_all=5), dict(J_all=0.88, wrong_all=5)
PROXY = {"Ac": dict(ok=True, label="A(vote, 60) on DEV-A'"), "ABc": dict(ok=True, label="AB(vote, 60) on DEV-A'")}


def test_g1_main_and_vacuous_label():
    r = J.g1_route(True, 60.0, _b(z=True), GOOD, GOOD, BAD, BAD, 47.1, 0)
    assert r["g1_pass"] and r["route"] == "main" and "vacuous" in r["labels"]["1"]
    r = J.g1_route(True, 45.0, _b(), GOOD, GOOD, BAD, BAD, 47.1, 0)
    assert r["g1_pass"] and r["labels"]["1"] is None
    assert not J.g1_route(True, 45.0, _b(), dict(J_all=0.90, wrong_all=6), BAD, BAD, BAD, 47.1, 0, proxy_b=PROXY)["g1_pass"]


def test_g1_routing_table():
    # DEV-B only (theta_s* < A_E): (a) at theta 60 decides with the AB(60) T06 value
    r = J.g1_route(True, 45.0, _b("DEV-B"), GOOD, GOOD, BAD, BAD, 47.1, 0)
    assert r["g1_pass"] and r["route"].startswith("fallback (a)") and "unverified" in r["labels"]["1"]
    r = J.g1_route(True, 45.0, _b("DEV-B"), GOOD, BAD, GOOD, BAD, 47.1, 0, proxy_b=PROXY)     # (a) misses -> (b)
    assert r["g1_pass"] and r["route"] == "fallback (b): Ac(60)" and r["C2"] == "detection" and r["tried"] == ["a", "b"]
    # T06 misses: (b) directly; Ac vs ABc by J_all, then fewer wrong, then Ac
    r = J.g1_route(True, 60.0, _b(), BAD, GOOD, dict(J_all=0.90, wrong_all=4), dict(J_all=0.91, wrong_all=5), 47.1, 0,
                   proxy_b=PROXY)
    assert r["route"] == "fallback (b): ABc(60)" and r["tried"] == ["b"]
    r = J.g1_route(True, 60.0, _b(), BAD, GOOD, dict(J_all=0.90, wrong_all=4), dict(J_all=0.90, wrong_all=4), 47.1, 0,
                   proxy_b=PROXY)
    assert r["b_arm"] == "Ac"
    # both DEV-B and T06: (a)'s AB(60) misses too -> (b)
    r = J.g1_route(True, 45.0, _b("DEV-B"), BAD, BAD, BAD, dict(J_all=0.95, wrong_all=1), 47.1, 0, proxy_b=PROXY)
    assert r["g1_pass"] and r["b_arm"] == "ABc"
    # (b) also misses: G1 fails, C2 'collapse analysis'
    r = J.g1_route(True, 60.0, _b(), BAD, BAD, BAD, BAD, 47.1, 0, proxy_b=PROXY)
    assert not r["g1_pass"] and r["C2"] == "collapse analysis"
    # DEV-B' failure: no fallback, even when every T06 value passes
    r = J.g1_route(True, 45.0, _b("DEV-B'"), GOOD, GOOD, GOOD, GOOD, 47.1, 0)
    assert not r["g1_pass"] and "DEV-B'" in r["route"]
    # Step 1 failed: G1 fails; theta_s* = inf: (b)
    assert not J.g1_route(False, math.inf, None, None, None, None, None, 47.1, 0)["g1_pass"]
    r = J.g1_route(True, math.inf, None, None, None, GOOD, BAD, 47.1, 0, proxy_b=PROXY)
    assert r["g1_pass"] and r["route"] == "fallback (b): Ac(60)" and r["conditions"] == {}
    assert "DEV-A' proxy = A(vote, 60)" in r["labels"]["1"]
    with pytest.raises(ValueError):
        J.g1_route(True, 45.0, _b("DEV-B"), GOOD, GOOD, BAD, BAD, 61.0, 0)     # (a) needs A_E < 60
    with pytest.raises(ValueError):
        J.g1_route(True, 60.0, _b(), BAD, GOOD, None, BAD, 47.1, 0, proxy_b=PROXY)


def test_g1_fallback_gates():
    """§7.3 (2nd review): (a) only when theta 60 is allowed (§6.4), else (b); (b)'s candidates are those whose DEV-A'
    proxy passes (Ac -> A(s_main, 60), ABc -> AB(s_main, 60)), then the T06 pick; none -> G1 fails."""
    # theta 60 not allowed (e.g. it failed the DEV-A' conditions): (a) is skipped even though AB(60) passes on T06
    r = J.g1_route(True, 45.0, _b("DEV-B"), GOOD, GOOD, GOOD, BAD, 47.1, 0, allowed60=False, proxy_b=PROXY)
    assert r["g1_pass"] and r["route"] == "fallback (b): Ac(60)" and r["tried"] == ["b"] and "a_skipped" in r
    assert "unverified in-domain" in r["labels"]["1"] and PROXY["Ac"]["label"] in r["labels"]["1"]
    # the better T06 candidate fails its proxy: the other candidate is picked (it passes T06 here)
    no_ac = dict(PROXY, Ac=dict(ok=False, label="A(vote, 60) on DEV-A'"))
    good2 = dict(J_all=0.895, wrong_all=4)
    r = J.g1_route(True, 60.0, _b(), BAD, GOOD, GOOD, good2, 47.1, 0, proxy_b=no_ac)
    assert r["g1_pass"] and r["b_arm"] == "ABc" and r["b_candidates"] == ["ABc"]
    r = J.g1_route(True, 60.0, _b(), BAD, GOOD, GOOD, BAD, 47.1, 0, proxy_b=no_ac)
    assert not r["g1_pass"] and r["b_arm"] == "ABc" and "also misses" in r["route"]
    # no candidate passes its proxy: G1 fails although the T06 values pass
    none = {a: dict(ok=False, label=a) for a in ("Ac", "ABc")}
    r = J.g1_route(True, 60.0, _b(), BAD, GOOD, GOOD, GOOD, 47.1, 0, proxy_b=none)
    assert not r["g1_pass"] and "no candidate" in r["route"] and r["C2"] == "collapse analysis"
    # a proxy not decoded yet: refuse to judge
    with pytest.raises(ValueError, match="proxy"):
        J.g1_route(True, 60.0, _b(), BAD, GOOD, GOOD, BAD, 47.1, 0)
    with pytest.raises(ValueError, match="proxy"):
        J.g1_route(True, 60.0, _b(), BAD, GOOD, GOOD, BAD, 47.1, 0, proxy_b=dict(PROXY, ABc=dict(ok=None, label="x")))
    # the main route and (a) never read the proxy
    assert J.g1_route(True, 45.0, _b("DEV-B"), GOOD, GOOD, BAD, BAD, 47.1, 0)["route"].startswith("fallback (a)")


def test_h23_rule():
    pairs = [(f"c{i:02d}", 1) for i in range(10)]
    neigh = {p: ["n"] for p in pairs}
    cov_n = {("n", 1): 1}
    r1 = {p: 2 for p in pairs}                                                  # round 1 took the other person everywhere
    assert J.h23(pairs, neigh, {p: 2 for p in pairs[:5]}, r1, cov_n)["verdict"] == "H3 supported"      # 5 >= 0.5 * 10
    assert J.h23(pairs, neigh, {p: 2 for p in pairs[:2]}, r1, cov_n)["verdict"] == "H2 supported"      # 2 <= 2.5
    assert J.h23(pairs, neigh, {p: 2 for p in pairs[:3]}, r1, cov_n)["verdict"] == "mixed"
    r = J.h23(pairs, neigh, {}, r1, {("n", 1): 2})                               # the neighbour memory was wrong: no P_good
    assert r["P_good"] == 0 and r["verdict"].startswith("undecidable")
    r = J.h23(pairs, {p: [] for p in pairs}, {}, r1, cov_n)                      # no trusted neighbour memory
    assert r["P_good"] == 0
    assert J.h23(pairs[:7], neigh, {}, r1, cov_n)["verdict"].startswith("undecidable")   # w1 = 7 < 8


def test_choice_report():
    recs = [dict(summary=dict(by_choice={"X": 5, "keep": 3}, by_choice_x_state={"keep/EMPTY": 2, "keep/FLAG": 1},
                              keep_absent=2, keep_flagged=1), chain=None),
            dict(summary=dict(by_choice={"chain": 2}, by_choice_x_state={"chain/FLAG": 2}, keep_absent=0, keep_flagged=0),
                 chain=dict(n_resolvable=10, n_chosen=2, n_missing=2))]
    r = J.choice_report(recs)
    assert r["by_choice"] == {"X": 5, "keep": 3, "chain": 2} and r["keep_absent"] == 2 and r["keep_flagged"] == 1
    assert r["chain"]["n_missing"] == 2 and "invalid" in r["chain"]["note"]
    assert J.choice_report([])["chain"] is None


def test_theta_and_family():
    assert J.theta_and_family(dict(rule=dict(theta_s="60"), family="vote"), dict()) == (60.0, "vote")
    # final check L3: a finite theta_s* needs the Step-2b record (prereg §6.5)
    with pytest.raises(SystemExit, match="step2b"):
        J.theta_and_family(dict(rule=dict(theta_s="60"), family="vote"), None)
    assert J.theta_and_family(dict(rule=dict(theta_s="45"), family="vote"), dict(main_family="cstar")) == (45.0, "cstar")
    th, _ = J.theta_and_family(dict(rule=dict(theta_s=None), family="vote"), None)
    assert math.isinf(th)
    # review 0 #4: an unfinished Step 2 / 2b (waiting for a decode) is not "relay closed"
    for rule in (dict(theta_s=None, need_devA="45"), dict(theta_s=None, need_zcheck=["60"])):
        with pytest.raises(SystemExit, match="unfinished"):
            J.theta_and_family(dict(rule=rule, family="vote"), None)
    with pytest.raises(SystemExit, match="unfinished"):
        J.theta_and_family(dict(rule=dict(theta_s="45"), family="vote"), dict(need_devA2="AB_45_hybrid_A_rot{1,2}"))


def test_config_check_and_main_preset(monkeypatch, tmp_path):
    """Review 0 #3: the decoded main preset's relay_theta / id3d / row must be the selection's."""
    cfg = dict(relay_theta=60.0, id3d="vote", id3d_scope="relay", relay_mode="seq", relay_order="hop", relay_mem="pool",
               relay_trust="id3d", flagged="keep")
    assert J.config_check(cfg, 60.0, "vote") == []
    assert J.config_check(cfg, 55.0, "vote") == [("relay_theta", 60.0)]
    assert J.config_check(dict(cfg, id3d="hybrid"), 60.0, "vote") == [("id3d", "hybrid")]
    assert J.config_check(dict(cfg, relay_mode="jacobi"), 60.0, "vote") == [("row", "A")]
    assert J.config_check(dict(cfg, relay_theta="inf"), math.inf, "vote") == []
    assert J.config_check({}, 60.0, "vote") == [("relay_theta", None)]
    from methods.geomem import stage1 as S
    monkeypatch.setattr(J, "DATA", str(tmp_path))
    assert J.check_main_preset(60.0, "vote", rots=()) == ({"PRESETS": [("missing", "geoset_rep_xs_ab")]}
                                                           if "geoset_rep_xs_ab" not in S.PRESETS else {})
    monkeypatch.setitem(S.PRESETS, "geoset_rep_xs_ab", dict(S.PRESETS["geoset_rep_xs"], relay=60.0, id3d="vote"))
    assert J.check_main_preset(60.0, "vote", rots=()) == {}
    assert J.check_main_preset(45.0, "vote", rots=())["PRESETS"] == [("theta", 60.0)]
    f = tmp_path / "_dev" / "geoset_rep_xs_ab_B" / "Blocks" / "000005"
    f.mkdir(parents=True)
    json.dump(dict(cfg, id3d="cstar"), open(f / "config.json", "w"))
    (f / "DONE").touch()
    bad = J.check_main_preset(60.0, "vote", rots=(0,))
    assert list(bad) == [os.path.join("_dev", "geoset_rep_xs_ab_B", "Blocks", "000005")]


def test_state_ratios_and_hallucinations():
    recs = [dict(round1={"a": {"1": "PASS", "2": "FLAG"}, "b": {"1": "EMPTY", "2": "PASS"}},
                 round2={"a": {"1": "PASS", "2": "PASS"}, "f": {"1": "FLAG", "2": "ABSTAIN"}},
                 relay={"g": {"1": "PASS", "2": "FLAG"}},
                 final={"g": {"1": dict(state="PASS"), "2": dict(state=None)}})]
    r = J.state_ratios(recs, [["a", "b"]])
    assert r["round1"]["counts"] == dict(PASS=2, FLAG=1, ABSTAIN=0, EMPTY=1) and r["round1"]["ratios"]["PASS"] == 0.5
    assert r["round2_S"]["n"] == 2 and r["round2_F"]["counts"]["FLAG"] == 1 and r["relay"]["counts"]["FLAG"] == 1
    assert r["final"]["counts"]["None"] == 1 and "gn" not in r
    pred = {("c", 1): 50, ("c", 2): 7, ("d", 1): 0, ("d", 2): 3}
    gt = {("c", 1): True, ("c", 2): False, ("d", 1): True, ("d", 2): False}
    c0 = {("c", 1): 40, ("c", 2): 0, ("d", 1): 9, ("d", 2): 5}
    h = J.hallucinations(pred, gt, c0)
    assert (h["n_absent"], h["arm"], h["c0"], h["newly_nonempty"], h["newly_nonempty_absent"]) == (2, 2, 1, 1, 1)
    assert (h["arm_px"], h["c0_px"]) == (10, 5)
    rows = [dict(d=8, p=1, covers=1, J=0.8), dict(d=9, p=2, covers=1, J=0.1), dict(d=3, p=1, covers=1, J=0.99),
            dict(d=7, p=3, covers=3, J=0.9)]
    assert J.far_correct_j(rows) == dict(n=2, J=pytest.approx(0.85), c0_diagnosis=0.842)


def test_pairs_and_subsets():
    rows = [dict(scene="S", t=4, ref="r", cam="a", obj=o) for o in (1, 2)]
    assert J.check_pairs(rows, list(rows))["ok"] and not J.check_pairs(rows, rows[:1])["ok"]
    assert not J.check_pairs(rows, rows + rows[:1])["ok"]
    keys = J.wide_keys(rows + [dict(scene="S", t=4, ref="r", cam="b", obj=1)], {("S", 4, "r", "a"): 70.0, ("S", 4, "r", "b"): 50.0}, 60.0)
    assert keys == {("S", 4, "r", "a", 1), ("S", 4, "r", "a", 2)} and len(J.subset(rows, keys)) == 2
    assert np.isclose(J.SEL.WIDE, 55.0)


def test_stage_g1_end_to_end(tmp_path, monkeypatch):
    """--stage G1 on synthetic records: the main route; the decoded T06 main config must be the selection's; (b) waits
    for its DEV-A' proxies."""
    out, res = tmp_path / "runs", tmp_path / "Result"
    out.mkdir()
    monkeypatch.setattr(J, "OUT", str(out))
    monkeypatch.setattr(J, "JUDGE", str(out / "e15_judge.json"))
    monkeypatch.setattr(J, "T06_RESULT", str(res))
    monkeypatch.setattr(J.SEL, "SCORES", str(tmp_path / "scores"))
    json.dump(dict(s_star=dict(family="vote")), open(out / "select.json", "w"))
    json.dump(dict(summary=dict(A_E=dict(a=47.1), A_all=dict(a=58.12), in_F={"60": dict(B=dict(n_eval_in_F=0))}),
                   entries=[]), open(out / "angles.json", "w"))
    ab60 = dict(safe=dict(ok=True, ci_lo=0.0, worst_scene=0.0), wide_dJ=0.01)
    json.dump(dict(family="vote", rule=dict(theta_s="60", allowed=["60", "inf"]), devA2={"60": ab60}),
              open(out / "step2.json", "w"))
    json.dump(dict(main_family="vote", rule=dict(swap=False)), open(out / "step2b.json", "w"))
    json.dump(dict(stageB=dict(case_z=True, rule={"pass": True, "fail_type": None})),
              open(out / "e15_judge.json", "w"))
    rows = [dict(cam="cam_20", d=10, p=1, J=0.9, covers=1)]
    arms = {"AB_60_vote": dict(J_all=0.91, wrong_all=3, Jfull_all=0.9, J_far=0.88, empty_all=0, empty_far=0, wrong_far=3)}
    json.dump(dict(arms=arms, rows={"AB_60_vote": rows}), open(tmp_path / "t06.json", "w"))
    cfg = dict(relay_theta=60.0, id3d="vote", id3d_scope="relay", relay_mode="seq", relay_order="hop", relay_mem="pool",
               relay_trust="id3d", flagged="keep")
    (res / "e15_AB_60_vote").mkdir(parents=True)
    json.dump(dict(config=cfg), open(res / "e15_AB_60_vote" / "RUN.json", "w"))
    J.stage_g1(str(tmp_path / "t06.json"))
    g = json.load(open(out / "e15_judge.json"))["G1"]
    assert g["g1_pass"] and g["route"] == "main" and g["allowed60"] is True and "vacuous" in g["labels"]["1"]
    assert g["reported"]["far_correct_J"]["J"] == 0.9 and g["proxy_b_all"]["ABc"]["ok"] is True
    assert g["proxy_b_all"]["Ac"]["ok"] is None                                  # A(vote, 60) not decoded here
    json.dump(dict(config=dict(cfg, id3d="hybrid")), open(res / "e15_AB_60_vote" / "RUN.json", "w"))
    with pytest.raises(SystemExit, match="disagrees with the selection"):
        J.stage_g1(str(tmp_path / "t06.json"))
    json.dump(dict(config=cfg), open(res / "e15_AB_60_vote" / "RUN.json", "w"))
    arms["AB_60_vote"].update(J_all=0.85)                                        # T06 misses -> (b) needs its proxies
    arms.update({"Ac_60_vote": dict(J_all=0.9, wrong_all=2), "ABc_60_vote": dict(J_all=0.88, wrong_all=2)})
    json.dump(dict(arms=arms, rows={"AB_60_vote": rows}), open(tmp_path / "t06.json", "w"))
    with pytest.raises(ValueError, match="proxy of Ac"):
        J.stage_g1(str(tmp_path / "t06.json"))


def _stage_b_world(tmp_path, monkeypatch, theta="60", arm_jf=0.8, drop_field=False):
    """A synthetic DEV-B / DEV-B' world for --stage B: 2 scenes x 1 frame, eval cams e1 / e2 (DEV-B'), case Z at 60."""
    from methods.geomem import stage1 as S
    out, data, scores = tmp_path / "runs", tmp_path / "Data", tmp_path / "scores"
    for d in (out, scores):
        d.mkdir(parents=True)
    monkeypatch.setattr(J, "OUT", str(out))
    monkeypatch.setattr(J, "JUDGE", str(out / "e15_judge.json"))
    monkeypatch.setattr(J, "DATA", str(data))
    monkeypatch.setattr(J.SEL, "SCORES", str(scores))
    monkeypatch.setattr(J.SEL, "C0_E14", str(data / "_dev" / "geoset_rep_xs_{split}" / "dev_pairs.json"))
    monkeypatch.setitem(S.PRESETS, "geoset_rep_xs_ab", dict(S.PRESETS["geoset_rep_xs"], relay=float(theta), id3d="vote"))
    ents = [dict(set="B'", scene=sc, t=5, ref=f"r{r}", rot=r, eval={"e1": 70.0, "e2": 20.0}, all={})
            for sc in ("S", "T") for r in (1, 2)]
    json.dump(dict(summary=dict(A_E=dict(a=47.1), A_all=dict(a=58.12), in_F={theta: dict(B=dict(n_eval_in_F=0))}),
                   entries=ents), open(out / "angles.json", "w"))
    json.dump(dict(family="vote", rule=dict(theta_s=theta, allowed=[theta])), open(out / "step2.json", "w"))
    json.dump(dict(main_family="vote", rule=dict(swap=False)), open(out / "step2b.json", "w"))
    cfg = dict(relay_theta=float(theta), id3d="vote", id3d_scope="relay", relay_mode="seq", relay_order="hop",
               relay_mem="pool", relay_trust="id3d", flagged="keep")

    def rows(rot, jf):
        out_ = []
        for sc in ("S", "T"):
            for cam in ("e1", "e2"):
                r = dict(scene=sc, t=5, cam=cam, obj=1, J=jf, F=jf, JF=jf, kind="x", seed_px=10, gt_px=20, covers=1,
                         other_obj_mask=False, view=f"5/{cam}" if not rot else f"5/r{rot}/{cam}")
                if rot:
                    r["ref"] = f"r{rot}"
                out_.append(r)
        return out_
    empty = [dict(scene="S", t=5, cam="e1", obj=2, seed_px=0)]
    for preset, jf in (("geoset_rep_xs", 0.8), ("geoset_rep_xs_ab", arm_jf)):
        for rot in (0, 1, 2):
            name = f"{preset}_B" + (f"_rot{rot}" if rot else "")
            rr = rows(rot, jf)
            if drop_field and preset == "geoset_rep_xs_ab" and rot == 1:
                rr[0].pop("other_obj_mask")
            json.dump(dict(meta={}, rows=rr, empty_gt=[dict(e, **({"ref": f"r{rot}"} if rot else {})) for e in empty]),
                      open(scores / f"{name}.json", "w"))
            root = data / "_dev" / name
            json.dump(dict(scenes={sc: dict(frames=[5], eval_cams=["e1", "e2"]) for sc in ("S", "T")}),
                      open((root.mkdir(parents=True) or root) / "index.json", "w"))
            for sc in ("S", "T"):
                f = root / sc / "000005"
                for cam in ("e1", "e2", "_r1/e1", "_r1/e2"):
                    (f / cam).mkdir(parents=True)
                    (f / cam / "1.png").write_bytes(b"m")
                json.dump({"e1": {"1": dict(presence=1.0)}, "e2": {"1": dict(presence=1.0)}}, open(f / "round1_conf.json", "w"))
                json.dump({"e1": {"1": dict(px=1)}, "e2": {"1": dict(px=1)}}, open(f / "xcl_conf.json", "w"))
                if preset == "geoset_rep_xs_ab":
                    json.dump(cfg, open(f / "config.json", "w"))
                    json.dump(dict(S=["e1", "e2"], F=[], active=False), open(f / "relay.json", "w"))
                (f / "DONE").touch()
        if preset == "geoset_rep_xs":
            e14 = data / "_dev" / "geoset_rep_xs_B" / "dev_pairs.json"
            json.dump(dict(meta={}, rows=[{k: v for k, v in r.items() if k not in ("covers", "other_obj_mask")}
                                          for r in rows(0, 0.8)], empty_gt=empty), open(e14, "w"))
    return out


def test_stage_b_end_to_end(tmp_path, monkeypatch):
    """--stage B on a synthetic case-Z world: the pre-checks (I12), the C0 re-score regression, invariants, B1-B4 and the
    reports (state ratios, hallucinations)."""
    out = _stage_b_world(tmp_path, monkeypatch)
    J.stage_b()
    b = json.load(open(out / "e15_judge.json"))["stageB"]
    assert b["case_z"] and b["rule"]["pass"] and b["b4"]["n_pairs"] == 4 and b["b4"]["other_arm"] == 0
    assert b["reports"]["hallucinations"]["B"]["n_absent"] == 1 and b["reports"]["hallucinations"]["B"]["arm"] == 0
    assert set(b["reports"]["state_ratios"]) == {"B", "B2"}
    # the decoded main preset disagrees with the selection: stop before anything is read
    with pytest.raises(SystemExit):
        json.dump(dict(family="vote", rule=dict(theta_s="55", allowed=["55"])), open(out / "step2.json", "w"))
        J.stage_b()
    assert "main_preset_mismatch" in json.load(open(out / "e15_judge.json"))["stageB"]


def test_stage_b_refuses_rows_without_the_b4_field(tmp_path, monkeypatch):
    _stage_b_world(tmp_path, monkeypatch, drop_field=True)
    with pytest.raises(SystemExit, match="other_obj_mask"):
        J.stage_b()

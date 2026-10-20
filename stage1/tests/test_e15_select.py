"""S1-E15 selection script (stage1/scripts/e15/e15_select.py, docs/stage1-E15-prereg.md §6): the registered rules on
synthetic inputs (host, numpy only)."""
import importlib.util
import itertools
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e15_synth as SY  # noqa: E402
from geomem import id3d  # noqa: E402
from geomem.id3d import EMPTY  # noqa: E402

spec = importlib.util.spec_from_file_location("e15_select", os.path.join(ROOT, "scripts", "e15", "e15_select.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def test_constants_as_registered():
    assert (E.F1_ALL, E.F1_SCENE, E.F1_FISH, E.F2_ALL, E.F3_PASS, E.F3_WRONG) == (0.003, 0.05, 0.005, 0.003, 0.9, 3)
    assert (E.WRONG_C0, E.DEV_LO, E.DEV_SCENE, E.J_TOL, E.SWAP_J, E.WIDE, E.A_E_STOP) == (28, -0.003, -0.010, 0.01, 0.02, 55.0, 60.0)
    assert E.THETA_GRID == (35.0, 40.0, 45.0, 50.0, 55.0, 60.0, math.inf) and E.STRESS == 20.0
    assert E.FAMILY_ORDER == ("A-v", "A-c*", "A-h", "A-c0") and set(E.FISHEYE) == {"AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder"}
    from methods.geomem import run as R
    assert set(E.THETAS_F) | {math.inf} == set(R.E15_THETAS)


def test_candidates_are_the_registered_652():
    c = E.candidates()
    assert len(c) == 652 and [x["index"] for x in c] == list(range(652))
    assert [x["family"] for x in c].count("cstar") == 648 and len({E.cand_name(x) for x in c}) == 652
    grid = [dict(zip(id3d.CSTAR_GRID, v)) for v in itertools.product(*id3d.CSTAR_GRID.values())]
    assert [x["consts"] for x in c if x["family"] == "cstar"] == grid
    assert E.run_name("AB", 35.0, "vote") == "AB_35_vote" and E.run_name("B-rep", math.inf, None) == "B-rep_inf_off"


def test_vote_from_counts_equals_vote_decide():
    rng = np.random.default_rng(3)
    models = {1: dict(has=True), 2: dict(has=True), 3: dict(has=False), 7: dict(has=True)}
    for _ in range(300):
        nq = int(rng.integers(1, 60))
        lab = rng.choice([0, 1, 2, 7], size=nq, p=rng.dirichlet(np.ones(4)))
        n = int(rng.choice([5, nq + 20]))
        o = int(rng.choice([1, 2, 3, 7, 9]))
        counts = {int(k): int(v) for k, v in zip(*np.unique(lab, return_counts=True))}
        assert E.vote_from_counts(counts, nq, n, o, models) == id3d.vote_decide(lab, n, o, models)[0]


@pytest.fixture(scope="module")
def ring(tmp_path_factory):
    d = SY.write_scene(str(tmp_path_factory.mktemp("e15sel")))
    lab = d["labels"][d["ref"]]
    d["masks"] = np.stack([lab == 1, lab == 2])
    return d


def test_decide_from_rows_equals_the_decode_judge(ring, monkeypatch):
    """A row (mask_quantities of the lifted mask) decided by `decide` = Judge.statuses of the decode, every family,
    A-c* over a slice of the grid with both rig outcomes; EMPTY by presence / no pixel."""
    from geomem.pointmap import ref_frame_points
    P_r, valid_r, _ = ref_frame_points(ring["cache"], ring["ref"], ring["ref"])
    models = id3d.ref_models(ring["masks"], [1, 2], P_r, valid_r)
    vref = id3d.vote_reference(models)
    rng = np.random.default_rng(0)
    rows, preds = [], []
    for c in ring["cams"][1:]:
        L = ring["labels"][c]
        P, v, _ = ref_frame_points(ring["cache"], c, ring["ref"])
        noise = rng.random(L.shape) < 0.05
        for i, (o, m) in enumerate([(1, L == 1), (2, L == 2), (1, L == 2), (2, (L == 1) | noise), (1, np.zeros_like(L, bool))]):
            q = E.mask_quantities(id3d.lift(id3d.mask_to_work(m, P.shape[:2]), P, v), models, vref)
            nv, ov = int(m.sum() * rng.random()), float(rng.choice([float("nan"), 0.2, 0.8]))
            rows.append(dict(q, o=o, n_px=int(m.sum()), presence=float(rng.choice([2.0, -1.0], p=[0.9, 0.1])), n_valid=nv,
                             overlap=ov, cam=c))
            preds.append(m)
    for phi_med in (0.0, 0.5):
        frame = dict(models=E.light_models(models), rig_phi=phi_med)
        cands = [dict(family=f, consts=None) for f in ("c0", "vote", "vote_inf", "hybrid")]
        cands += [x for x in E.candidates() if x["family"] == "cstar"][::37]
        for cand in cands:
            consts = cand["consts"]
            rig_ok = cand["family"] != "cstar" or math.isinf(consts["phi_rig"]) or phi_med <= consts["phi_rig"]
            J = id3d.Judge(ring["cache"], ring["ref"], ring["masks"], [1, 2], cand["family"], consts=consts, rig_ok=rig_ok)
            for r, m in zip(rows, preds):
                o = r["o"]
                pred = np.zeros((2,) + m.shape, bool)
                pred[o - 1] = m
                pres = [r["presence"]] * 2
                st, _ = J.statuses(r["cam"], pred, pres, [r["n_valid"]] * 2, [r["overlap"]] * 2)
                assert E.decide(r, cand, frame) == st[o - 1], (cand, r["cam"], o)
    assert E.decide(dict(rows[0], presence=-1.0), dict(family="vote", consts=None), frame) == EMPTY


def _m(**kw):
    base = dict(gt_ff=0, gt_n=1000, gt_scene_max=0.0, fish_max=0.0, r1_ff=0, r1_n=1000, t06_correct_pass=79, t06_correct_n=79,
                t06_wrong_pass=0, t06_wrong_flag=20, t06_wrong_n=31, swap_flag=10, swap_n=20)
    base.update(kw)
    base.update(E.feasibility(base))
    return base


def test_feasibility_boundaries():
    assert _m()["feasible"]
    assert _m(gt_ff=3)["F1"] and not _m(gt_ff=4)["F1"]                      # 0.3 % of 1000
    assert _m(gt_scene_max=0.05)["F1"] and not _m(gt_scene_max=0.0501)["F1"]
    assert _m(fish_max=0.005)["F1"] and not _m(fish_max=0.0051)["F1"]
    assert _m(r1_ff=3)["F2"] and not _m(r1_ff=4)["F2"]
    assert _m(t06_correct_pass=72, t06_correct_n=80)["F3"] and not _m(t06_correct_pass=71, t06_correct_n=80)["F3"]
    assert _m(t06_wrong_pass=3)["F3"] and not _m(t06_wrong_pass=4)["F3"]
    assert not _m(gt_n=0)["F1"] and not _m(r1_n=0)["F2"] and not _m(t06_correct_n=0)["F3"]


def _c(family, i, **consts):
    k = None
    if family == "cstar":
        k = dict(beta=1.0, kappa=1.25, sigma=0.05, rho_env=0.5, b_max=0.01, gamma=math.inf, phi_rig=math.inf)
        k.update(consts)
    return dict(family=family, consts=k, index=i)


def test_choose_order_and_tie_breaks():
    # objective first (an infeasible candidate never counts)
    best, log = E.choose([(_c("vote", 649), _m(t06_wrong_flag=29)), (_c("c0", 0), _m(t06_wrong_flag=30, gt_ff=50)),
                          (_c("cstar", 1), _m(t06_wrong_flag=24))])
    assert best[0]["family"] == "vote" and log["n_feasible"] == 2
    # tie-break 1: swap detection
    best, _ = E.choose([(_c("vote", 649), _m(swap_flag=9)), (_c("cstar", 1), _m(swap_flag=10))])
    assert best[0]["family"] == "cstar"
    # tie-break 2: fewer components (gamma / b_max / phi_rig finite count as on)
    best, _ = E.choose([(_c("cstar", 1, b_max=0.01, gamma=2.0), _m()), (_c("cstar", 2, b_max=math.inf, gamma=2.0), _m())])
    assert best[0]["index"] == 2
    # tie-break 3 within A-c*: larger beta, then kappa, then sigma; other families untouched; then 4: family order
    pool = [(_c("cstar", 1, beta=1.0, b_max=math.inf), _m()), (_c("cstar", 2, beta=1.5, kappa=1.0, b_max=math.inf), _m()),
            (_c("cstar", 3, beta=1.5, kappa=1.25, b_max=math.inf), _m()), (_c("hybrid", 651), _m())]
    best, log = E.choose(pool)
    assert best[0]["index"] == 3 and log["after_tie3_conservative"] == 2 and log["after_tie4_family"] == 1
    best, _ = E.choose(pool + [(_c("vote_inf", 650), _m())])
    assert best[0]["family"] == "vote_inf"                                      # A-v first in the family order
    # a remaining tie: the registered grid order decides, and it is recorded
    best, log = E.choose([(_c("vote_inf", 650), _m()), (_c("vote", 649), _m())])
    assert best[0]["family"] == "vote" and log["residual_tie"] == ["vote", "vote_inf"]
    assert E.choose([(_c("vote", 649), _m(gt_ff=50))]) == (None, dict(n_feasible=0))


def test_best_per_family():
    scored = [(_c("vote", 649), _m(t06_wrong_flag=29)), (_c("vote_inf", 650), _m(t06_wrong_flag=30)),
              (_c("cstar", 1), _m(t06_wrong_flag=24)), (_c("cstar", 2, beta=1.5), _m(t06_wrong_flag=24, gt_ff=40)),
              (_c("hybrid", 651), _m(gt_ff=40)), (_c("c0", 0), _m(t06_wrong_flag=2))]
    b = E.best_per_family(scored)
    assert b["A-v"]["family"] == "vote_inf" and b["A-c*"]["consts"]["beta"] == 1.0 and b["A-h"] is None and b["A-c0"]["family"] == "c0"


def _cmp(lo, per):
    return dict(paired=dict(JF=dict(delta=0.0, ci=[lo, lo + 0.01], per_scene=per)))


def test_dev_safe():
    assert E.dev_safe(_cmp(-0.0029, {"a": -0.01}))["ok"]
    assert not E.dev_safe(_cmp(-0.003, {"a": 0.0}))["ok"]                     # the lower bound must be > -0.003
    assert not E.dev_safe(_cmp(0.0, {"a": -0.0101}))["ok"]
    assert not E.dev_safe(_cmp(float("nan"), {"a": 0.0}))["ok"]


def _t06(**j):
    out = {th: dict(J_all=0.742, wrong_all=28) for th in E.THETA_GRID}
    for k, v in j.items():
        out[float(k[1:]) if k != "inf" else math.inf] = v
    return out


def _dev2(bad=(), wide_neg=()):
    return {th: dict(safe=dict(ok=th not in bad), wide_dJ=-0.01 if th in wide_neg else 0.01) for th in E.THETA_GRID if not math.isinf(th)}


def _zc(family="vote", ok=True, thetas=("60",), drop=()):
    """A runs/zcheck.json: the whole §6.3b Z stage passing (or Z-R failing), without the items in `drop`."""
    zc = dict(family=family, Z4={"pass": True}, ZI2=dict(family=family, **{"pass": True}), ZT06={"pass": True},
              Zsim=dict(T06={"pass": True}, DEV_A_G=dict(family=family, **{"pass": True})),
              ZR={t: dict(theta=t, family=family, mode="full" if t == "60" else "case_z", **{"pass": ok}) for t in thetas})
    for d in drop:
        if d.startswith("Zsim"):
            zc["Zsim"].pop("T06" if d == "Zsim T06" else "DEV_A_G")
        else:
            zc.pop(d)
    return zc


def test_theta_rule():
    t06 = _t06(t60=dict(J_all=0.90, wrong_all=6), t55=dict(J_all=0.905, wrong_all=6), t45=dict(J_all=0.91, wrong_all=5),
               t35=dict(J_all=0.93, wrong_all=30))
    kw = dict(zcheck=_zc(), A_all=58.12, family="vote")
    r = E.theta_rule(t06, _dev2(), {}, 47.1, **kw)
    assert "35" not in r["allowed"] and r["J_star"] == 0.91                 # theta 35 makes more wrong persons than C0
    assert r["C"] == ["60", "55", "45"] and r["theta_s"] == "60"            # the largest theta within 0.01 of J*
    assert r["checked"][0]["zcheck"] == list(E.ZSTAGE)
    r = E.theta_rule(t06, _dev2(bad=(60.0, 55.0)), {}, 47.1, **kw)
    assert r["C"] == ["45"] and r["need_devA"] == "45" and r["theta_s"] is None   # theta < A_E: DEV-A full decode first
    r = E.theta_rule(t06, _dev2(bad=(60.0, 55.0)), {45.0: dict(ok=True, ci_lo=0.0, worst_scene=0.0)}, 47.1, **kw)
    assert r["theta_s"] == "45"
    r = E.theta_rule(t06, _dev2(bad=(60.0, 55.0)), {45.0: dict(ok=False, ci_lo=-0.01, worst_scene=0.0)}, 47.1, **kw)
    assert r["theta_s"] is None and math.isinf(r["theta_s_value"])
    r = E.theta_rule(t06, _dev2(wide_neg=(60.0, 55.0, 45.0)), {}, 47.1, **kw)   # wide DEV-A' pairs got worse there
    assert r["allowed"] == ["40", "50", "inf"] and r["C"] == ["inf", "50", "40"] and r["theta_s"] == "inf"
    r = E.theta_rule(_t06(), _dev2(), {}, 47.1, **kw)                        # nothing beats C0: the largest = inf
    assert r["theta_s"] == "inf" and "relay closed" in r["note"]
    with pytest.raises(SystemExit):
        E.theta_rule({60.0: dict(J_all=0.9, wrong_all=5)}, _dev2(), {}, 47.1, **kw)


def test_theta_rule_needs_the_z_stage_beyond_a_e():
    """§6.3b / §6.4: theta >= A_E (inf included) is DEV-A-safe only with the whole Z stage passing in runs/zcheck.json
    (+ the case-Z Z-R at a theta in [A_E, A_all)); a failing check is a bug."""
    t06 = _t06(t60=dict(J_all=0.90, wrong_all=6), t55=dict(J_all=0.92, wrong_all=6))
    r = E.theta_rule(t06, _dev2(), {}, 47.1, zcheck=None, A_all=58.12, family="vote")
    assert r["theta_s"] is None and r["need_zcheck"] == list(E.ZSTAGE) + ["ZR 55"] and r["C"] == ["55"]
    r = E.theta_rule(t06, _dev2(), {}, 47.1, zcheck=_zc(), A_all=58.12, family="vote")
    assert r["need_zcheck"] == ["ZR 55"]                                       # 55 in [A_E, A_all): its own case-Z check
    r = E.theta_rule(t06, _dev2(), {}, 47.1, zcheck=_zc(thetas=("60", "55")), A_all=58.12, family="vote")
    assert r["theta_s"] == "55" and r["checked"][0]["zcheck"][-1] == "ZR 55"
    r = E.theta_rule(t06, _dev2(), {}, 47.1, zcheck=_zc(thetas=("60",)), A_all=54.0, family="vote")
    assert r["theta_s"] == "55"                                                # 55 >= A_all: Z-R at 60 covers it
    for item in ("Z4", "ZI2", "ZT06", "Zsim T06", "Zsim DEV-A G"):
        r = E.theta_rule(t06, _dev2(), {}, 47.1, zcheck=_zc(thetas=("60", "55"), drop=(item,)), A_all=58.12, family="vote")
        assert r["need_zcheck"] == [item], item
    r = E.theta_rule(t06, _dev2(), {}, 47.1, zcheck=_zc(family="hybrid", thetas=("60", "55")), A_all=58.12, family="vote")
    assert r["need_zcheck"] == list(E.ZSTAGE) + ["ZR 55"]                      # another family's Z stage does not count
    with pytest.raises(SystemExit) as e:
        E.theta_rule(t06, _dev2(), {}, 47.1, zcheck=_zc(ok=False, thetas=("60", "55")), A_all=58.12, family="vote")
    assert e.value.code == 2
    r = E.theta_rule(_t06(), _dev2(), {}, 47.1, zcheck=None, A_all=58.12, family="vote")
    assert r["need_zcheck"] == list(E.ZSTAGE)                                  # inf too (§6.3b "∞ 포함")
    assert E.zcheck_missing(_zc(), math.inf, 58.12, "vote") == [] and E.zcheck_missing(_zc(), 50.0, 58.12, "vote") == ["ZR 50"]


def test_swap_rule():
    main = dict(J_all=0.90, wrong_all=6)
    assert E.swap_rule(main, {"A-v": dict(J_all=0.919, wrong_all=2)})["swap"] is False      # < +0.02
    assert E.swap_rule(main, {"A-v": dict(J_all=0.92, wrong_all=6)})["swap"] is False       # not fewer wrong
    r = E.swap_rule(main, {"A-v": dict(J_all=0.92, wrong_all=5), "A-h": dict(J_all=0.93, wrong_all=5), "A-c0": None})
    assert r == dict(swap=True, to="A-h", qualifying=["A-h", "A-v"])
    r = E.swap_rule(main, {"A-v": dict(J_all=0.93, wrong_all=5), "A-h": dict(J_all=0.93, wrong_all=5)})
    assert r["to"] == "A-v"                                                    # tie -> the family order


def _entry(st, scene, t, ref, ev, al, rot=0):
    return dict(set=st, scene=scene, t=t, rot=rot, ref=ref, kind="obj", fallback=None, eval=ev, all=al, eval_scene=ev,
                anchor_dist=0.0, max_dangle=0.0)


def test_angle_summary():
    es = [_entry("A", "S", 4, "c1", {"c0": 30.0, "c2": 47.1}, {"c0": 30.0, "c2": 47.1, "c3": 58.1}),
          _entry("B", "S", 5, "c1", {"c0": 31.0, "c2": 46.0}, {"c0": 31.0, "c2": 46.0, "c3": 58.12}),
          _entry("K", "S", 0, "c1", {"c0": 29.0, "c2": 45.5}, {"c0": 29.0, "c2": 45.5, "c3": 57.0}),
          _entry("A'", "S", 4, "c0", {"c1": 30.0, "c2": 77.0}, {"c1": 30.0, "c2": 77.0, "c3": 85.0}, 1)]
    s = E.angle_summary(es)
    assert s["A_E"]["a"] == 47.1 and s["A_E"]["set"] == "A" and s["A_all"]["a"] == 58.12 and s["A_all"]["set"] == "B"
    assert s["in_F"]["45"]["A"]["n_eval_in_F"] == 1 and s["in_F"]["45"]["K"]["n_eval_in_F"] == 1
    assert s["in_F"]["50"]["B"]["n_eval_in_F"] == 0 and s["in_F"]["55"]["B"]["n_all_in_F"] == 1
    assert s["in_F"]["60"] == {st: dict(eval_in_F=[], n_eval_in_F=0, n_all_in_F=0) for st in ("A", "B", "K")}
    assert s["rotated"]["A'"]["n_wide"] == 1 and s["rotated"]["A'"]["max"] == 77.0          # rotated sets never set A_E


def test_n2_pairs(monkeypatch):
    es = [_entry("A", "S", 2, "c1", {"c0": 30.0, "c2": 50.0}, {}), _entry("A'", "S", 2, "c0", {"c1": 30.0, "c2": 80.0}, {}, 1),
          _entry("A'", "S", 2, "c2", {"c0": 80.0, "c1": 50.0}, {}, 2), _entry("K", "S", 0, "c1", {"c0": 29.0, "c2": 49.0}, {})]
    monkeypatch.setattr(E, "load_meta", lambda data, s: dict(mask_cams=["c0", "c1", "c2"]))
    import geomem.cache as gc
    monkeypatch.setattr(gc, "train_frames", lambda data, s: [0, 1, 2])
    r = E.n2_pairs(es, ["S"], stride=2)                                        # frames 0 and 2
    assert r["n"] == 8 and r["excluded_rotated_keyframe_pairs"] == 4          # keyframe: c_ini's pairs only
    assert r["per_scene"]["S"]["max"] == 80.0 and r["frac_over_45"] == pytest.approx(5 / 8)


def _rows(scene, jf, views=(("4", "c1", "a"),)):
    return [dict(scene=scene, t=int(t), ref=ref, cam=cam, obj=1, J=jf, F=jf, JF=jf, view=f"{t}/{ref}/{cam}") for t, ref, cam in views]


def test_fill_from_c0_and_wide_delta(tmp_path, monkeypatch):
    base = _rows("S", 0.5, (("4", "c1", "a"), ("4", "c1", "b"))) + _rows("T", 0.6)
    cand = _rows("S", 0.7, (("4", "c1", "a"), ("4", "c1", "b")))
    ang = {("S", 4, "c1", "a"): 70.0, ("S", 4, "c1", "b"): 30.0, ("T", 4, "c1", "a"): 40.0}
    import json
    monkeypatch.setattr(E, "ZCHECK", str(tmp_path / "zcheck.json"))
    with pytest.raises(SystemExit, match="Z-I2"):                              # §6.4: the fill rule waits for Z-I2
        E.fill_from_c0(base, cand, ang, 45.0)
    assert E.fill_from_c0(base, base, ang, 45.0)[1] == []                     # nothing to fill: no Z-I2 needed
    json.dump(dict(ZI2={"pass": False}), open(tmp_path / "zcheck.json", "w"))
    with pytest.raises(SystemExit, match="Z-I2"):
        E.fill_from_c0(base, cand, ang, 45.0)
    json.dump(dict(ZI2={"pass": True}), open(tmp_path / "zcheck.json", "w"))
    rows, filled = E.fill_from_c0(base, cand, ang, 45.0)
    assert filled == ["T"] and len(rows) == 3 and [r["JF"] for r in rows if r["scene"] == "T"] == [0.6]
    with pytest.raises(SystemExit, match="lacks scenes"):
        E.fill_from_c0(base, cand, ang, 35.0)                                  # T's view (40) is in F at 35
    assert E.wide_subset_delta(base, rows, ang, 45.0) == pytest.approx(0.2)    # only S/a (70 > 45)
    assert E.wide_subset_delta(base, rows, ang, 75.0) is None


def test_t06_d1_manifest(tmp_path):
    """Review 0 #9 / 1 #6: the T06 C0 input is the persisted D1 decode, checked file by file against the tracked manifest."""
    assert E.T06_C0 == os.path.join(E.WORK, "d1_ref") and "/tmp" not in E.T06_C0                 # prereg §6.1
    assert E.T06_D1_MANIFEST == os.path.join(ROOT, "scripts", "e15", "design", "d1_t06_c0.sha256")
    assert os.path.isfile(E.T06_D1_MANIFEST)
    lines = open(E.T06_D1_MANIFEST).read().split("\n")
    rel = [ln.split("  ", 1)[1] for ln in lines if ln]
    assert rel == sorted(rel) and {r.split("/")[0] for r in rel} == {"preset", "preset_r1"}
    assert {"preset/round1_conf.json", "preset/xcl_conf.json"} <= set(rel) and len(rel) == 377
    for d in ("preset/cam_00", "preset_r1/cam_00"):
        os.makedirs(tmp_path / d)
    (tmp_path / "preset" / "cam_00" / "1.png").write_bytes(b"a")
    (tmp_path / "preset_r1" / "cam_00" / "1.png").write_bytes(b"b")
    man = tmp_path / "m.sha256"
    man.write_text(f"{E.hashlib.sha256(b'a').hexdigest()}  preset/cam_00/1.png\n"
                   f"{E.hashlib.sha256(b'b').hexdigest()}  preset_r1/cam_00/1.png\n")
    assert E.check_manifest(str(tmp_path), str(man)) == dict(ok=True, n=2, bad=[])
    (tmp_path / "preset" / "cam_00" / "2.png").write_bytes(b"c")
    (tmp_path / "preset_r1" / "cam_00" / "1.png").write_bytes(b"X")
    r = E.check_manifest(str(tmp_path), str(man))
    assert not r["ok"] and sorted(r["bad"]) == [("preset/cam_00/2.png", "not in the manifest"),
                                                ("preset_r1/cam_00/1.png", "sha256 differs")]
    with pytest.raises(SystemExit, match="not the D1 decode"):
        E.require_t06_c0(str(tmp_path))


@pytest.mark.skipif(not os.path.isdir(E.T06_C0), reason="needs the persisted D1 decode")
def test_persisted_d1_matches_its_manifest():
    assert E.check_manifest()["ok"]


def _scoring(path, rows, empty=()):
    import json
    with open(path, "w") as f:
        json.dump(dict(meta={}, rows=rows, empty_gt=list(empty)), f)


def test_rescore_check(tmp_path):
    """Review 1 #9: the C0 re-scoring must reproduce the E14 J / F / J&F pair by pair and carry covers / other_obj_mask."""
    old = [dict(scene="S", t=4, cam="a", obj=o, J=0.5 + o / 10, F=0.6, JF=0.55 + o / 20) for o in (1, 2)]
    new = [dict(r, covers=r["obj"], other_obj_mask=False) for r in old]
    e = [dict(scene="S", t=4, cam="a", obj=3, seed_px=0)]
    _scoring(tmp_path / "old.json", old, e)
    _scoring(tmp_path / "new.json", new, e)
    r = E.rescore_check("A", str(tmp_path / "new.json"), str(tmp_path / "old.json"))
    assert r["ok"] and r["n_pairs"] == 2 and r["n_diff"] == 0
    _scoring(tmp_path / "bad.json", [dict(new[0], J=0.61)] + new[1:], e)
    r = E.rescore_check("A", str(tmp_path / "bad.json"), str(tmp_path / "old.json"))
    assert not r["ok"] and r["n_diff"] == 1 and r["diffs"][0][-1] == "J"
    _scoring(tmp_path / "nof.json", [dict(old[0])] + new[1:], e)
    assert E.rescore_check("A", str(tmp_path / "nof.json"), str(tmp_path / "old.json"))["missing_fields"] == 1
    _scoring(tmp_path / "less.json", new[:1], e)
    assert not E.rescore_check("A", str(tmp_path / "less.json"), str(tmp_path / "old.json"))["ok"]
    _scoring(tmp_path / "emp.json", new, [])
    assert not E.rescore_check("A", str(tmp_path / "emp.json"), str(tmp_path / "old.json"))["ok"]
    with pytest.raises(SystemExit, match="c0-rescore"):
        E.rescore_check("A", str(tmp_path / "none.json"), str(tmp_path / "old.json"))


@pytest.mark.skipif(not os.path.isfile(os.path.join(E.DATA, "_dev", "geoset_rep_xs_A", "dev_pairs.json")),
                    reason="needs the E14 DEV-A scoring")
def test_e14_scoring_against_itself_passes_the_rescore_check():
    """The regression check accepts the E14 file with the two fields added (what an exact re-score produces)."""
    import json
    p = E.C0_E14.format(split="A")
    assert E.rescore_check("A", p, p)["missing_fields"] == len(json.load(open(p))["rows"])


def _step2b_env(tmp_path, monkeypatch, alt_dev2_ok=True, alt_decoded=True):
    """A synthetic Step-2b world: s* = vote at theta_s* 60; the A-h best (hybrid) beats it on T06 by 0.03 with fewer wrong;
    DEV-A' C0 and AB(hybrid, 60) scorings over two scenes."""
    import json
    out, scores = tmp_path / "runs", tmp_path / "scores"
    out.mkdir(parents=True)
    scores.mkdir(parents=True)
    monkeypatch.setattr(E, "OUT", str(out))
    monkeypatch.setattr(E, "SCORES", str(scores))
    sel = dict(s_star=dict(family="vote"), best_per_family={"A-v": dict(family="vote", name="vote"),
                                                              "A-h": dict(family="hybrid", name="hybrid"),
                                                              "A-c*": None, "A-c0": None})
    json.dump(sel, open(out / "select.json", "w"))
    json.dump(dict(family="vote", rule=dict(theta_s="60", allowed=["60"])), open(out / "step2.json", "w"))
    ents = [dict(set="A'", scene=sc, t=4, ref=ref, rot=r, eval={c: a for c, a in cams.items()}, all={})
            for sc in ("S", "T") for r, ref, cams in ((1, "r1", {"a": 70.0, "b": 30.0}), (2, "r2", {"a": 75.0, "c": 20.0}))]
    json.dump(dict(summary=dict(A_E=dict(a=47.1), A_all=dict(a=58.12)), entries=ents), open(out / "angles.json", "w"))
    t06 = dict(arms={"AB_60_vote": dict(J_all=0.90, wrong_all=6), "AB_60_hybrid": dict(J_all=0.93, wrong_all=3)})
    json.dump(t06, open(tmp_path / "t06.json", "w"))

    def rows(rot, jf_wide):
        ref = f"r{rot}"
        out_ = []
        for sc in ("S", "T"):
            for cam in (("a", "b") if rot == 1 else ("a", "c")):
                v = jf_wide if cam == "a" else 0.5
                out_.append(dict(scene=sc, t=4, ref=ref, cam=cam, obj=1, J=v, F=v, JF=v, view=f"4/{ref}/{cam}"))
        return out_
    for r in (1, 2):
        _scoring(scores / f"geoset_rep_xs_A_rot{r}.json", rows(r, 0.5))
        if alt_decoded:
            _scoring(scores / f"AB_60_hybrid_A_rot{r}.json", rows(r, 0.6 if alt_dev2_ok else 0.3))
    return str(tmp_path / "t06.json")


def test_step2b_swap_rechecks_dev_a_prime(tmp_path, monkeypatch):
    """Review 1 #2: a swap to s' needs the §6.4 DEV-A' conditions of AB(s', theta_s*)."""
    p = _step2b_env(tmp_path, monkeypatch)
    r = E.step2b(p)
    assert r["rule"]["swap"] and r["main_family"] == "hybrid" and r["devA2_conditions"]["ok"]
    p = _step2b_env(tmp_path / "x", monkeypatch, alt_dev2_ok=False)
    r = E.step2b(p)
    assert not r["rule"]["swap"] and r["main_family"] == "vote" and "DEV-A'" in r["rule"]["note"]
    p = _step2b_env(tmp_path / "y", monkeypatch, alt_decoded=False)
    with pytest.raises(SystemExit) as e:
        E.step2b(p)
    assert e.value.code == 3 and E.load(os.path.join(E.OUT, "step2b.json"))["need_devA2"].startswith("AB_60_hybrid_A")


def test_step2b_refuses_an_unfinished_step2(tmp_path, monkeypatch):
    import json
    p = _step2b_env(tmp_path, monkeypatch)
    json.dump(dict(family="vote", rule=dict(theta_s=None, need_zcheck=["60"])), open(os.path.join(E.OUT, "step2.json"), "w"))
    with pytest.raises(SystemExit, match="unfinished"):
        E.step2b(p)

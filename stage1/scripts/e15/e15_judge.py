#!/usr/bin/env python3
"""S1-E15 judge (docs/stage1-E15-prereg.md §5.3, §7, §8).  Host (numpy).  Committed with the second registration, before
any DEV-B / DEV-B' decode (§11.2).

    --stage B    DEV-B and DEV-B' once (§7.1): the selection is final (Step 2 / 2b not waiting for a decode) and the main
                 preset geoset_rep_xs_ab and every decoded main-preset config carry theta_s*, the main family and the AB row
                 (exit 2 otherwise), pair sets (exit 3 if they differ), invariants I1 / I2 and, in case Z, the eval-cam PNG
                 bytes + eval-cam xcl_conf entries (by value) + whole round1_conf.json bytes against C0 (exit 2 = a bug: fix
                 and decode again), the stop-and-audit thresholds (exit 4, only those numbers saved, until --audited), then
                 B1-B4 (case Z: B1 / B2 "vacuous"; B4 needs other_obj_mask on every scored row) and the report-only values
                 (dJ / dF, per scene, state ratios, hallucinations and newly non-empty masks, time).
    --stage T06  T06 reports: §5.3 H2 / H3 rule (AB and A), §8 reports for every E15 T06 decode in the score json: the
                 J of the far pairs whose right person was found (C0 0.842), state ratios per stage, hallucinations
                 (non-empty masks of a person absent from the view's GT; count and pixels, with C0) and newly non-empty
                 masks vs C0, the reporting rules (choices split, absent objects, chain files, relay candidate counts);
                 and the state ratios of the DEV-A' override decodes (per scene and stage).
    --stage G1   G1 (§7.3): conditions 1-3, the failure-type routing table (fallbacks (a) / (b)), the C2 wording and the
                 condition-1 labels ("vacuous", "unverified").  Fallback (a) needs theta 60 in Step 2's allowed set (§6.4;
                 else (b)); (b) is gated by the §6.4 DEV-A' conditions of AB(60) (and A(60) for Ac when decoded) as an
                 in-domain proxy (chain masks do not exist on MUVOD) -- recorded as such.
All stages merge into stage1/scripts/e15/runs/e15_judge.json.  Names as e15_select.py (<arm>_<theta>_<family>).
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, S1)

_spec = importlib.util.spec_from_file_location("e15_select", os.path.join(HERE, "e15_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

DATA, OUT, WORK, SCORES = SEL.DATA, SEL.OUT, SEL.WORK, SEL.SCORES
JUDGE = os.path.join(OUT, "e15_judge.json")
C0_PRESET, MAIN_PRESET = "geoset_rep_xs", "geoset_rep_xs_ab"
T06_RESULT = os.path.join(DATA, SEL.T06, "Result")
DELTA, SCENE_FLOOR = -0.003, -0.010                  # B1 / B3 non-inferiority, B2 / B3 scene floor
AUDIT_B, AUDIT_B2_WIDE = 0.006, 0.05                 # §7.1 stop and audit
G1_J, G1_WRONG = 0.89, 5                             # §7.3 conditions 2 / 3
H_MIN, H3_SHARE, H2_SHARE = 8, 0.5, 0.25             # §5.3
C0_FAR_CORRECT_J = 0.842                             # §2.1 / §8: J of the far pairs with the right person, C0 (diagnosis)
STATES = ("PASS", "FLAG", "ABSTAIN", "EMPTY")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _save(stage, res):
    out = _load(JUDGE) if os.path.isfile(JUDGE) else {}
    out["prereg"] = "docs/stage1-E15-prereg.md"
    out[stage] = res
    SEL.save(JUDGE, out)


def _bytes(p):
    with open(p, "rb") as f:
        return f.read()


# ------------------------------------------------------------------------------------------------- pure rules (unit-tested)
def theta_and_family(step2, step2b):
    """theta_s* (float, inf = relay closed) and the main family (s*, or s' after a Step-2b swap).  Refuses a Step 2 / 2b
    still waiting for a decode (need_devA / need_zcheck / need_devA2): their theta_s None is not "relay closed"."""
    for name, rec in (("step2", step2["rule"]), ("step2b", step2b or {})):
        wait = {k: rec[k] for k in ("need_devA", "need_zcheck", "need_devA2") if rec.get(k)}
        if wait:
            raise SystemExit(f"{name}.json is unfinished {wait}: finish Step 2 / 2b before judging (prereg §6.4, §6.5)")
    th = step2["rule"].get("theta_s")
    theta = math.inf if th in (None, "inf") else float(th)
    if math.isfinite(theta) and step2b is None:          # §6.5: the Step-2b swap check is mandatory for a finite theta_s*
        raise SystemExit("finite theta_s* but runs/step2b.json is missing: run --step 2b before judging (prereg §6.5)")
    fam = (step2b or {}).get("main_family") or step2["family"]
    return theta, fam


def case_z(angles_summary, theta):
    """§7.1 case Z (scope R): no DEV-B eval cam in F at theta_s* (from the inputs, before decoding)."""
    if math.isinf(theta):
        return True
    rec = angles_summary["in_F"].get(SEL.theta_tag(theta))
    if rec is None:
        raise ValueError(f"angles.json has no F record for theta {theta}")
    return rec["B"]["n_eval_in_F"] == 0


def frame_compare(c0_dir, arm_dir, eval_cams, case_z_frame, star_cams):
    """One frame of the arm vs C0 -> list of differences (empty = equal).
    case Z: eval-cam PNG bytes, eval-cam xcl_conf entries by value, the whole round1_conf.json bytes.
    otherwise: I1 = round-1 PNG bytes and round1_conf entries of C0's decoded cams; I2 = the star eval cams' PNG bytes."""
    diffs = []

    def pngs(d, cam):
        p = os.path.join(d, cam)
        return {f: _bytes(os.path.join(p, f)) for f in sorted(os.listdir(p)) if f.endswith(".png")} if os.path.isdir(p) else {}
    cams = eval_cams if case_z_frame else star_cams
    for cam in cams:
        if pngs(c0_dir, cam) != pngs(arm_dir, cam):
            diffs.append(("png", cam))
    if case_z_frame:
        xa, xb = _load(os.path.join(c0_dir, "xcl_conf.json")), _load(os.path.join(arm_dir, "xcl_conf.json"))
        for cam in eval_cams:
            if xa.get(cam) != xb.get(cam):
                diffs.append(("xcl_conf", cam))
        if _bytes(os.path.join(c0_dir, "round1_conf.json")) != _bytes(os.path.join(arm_dir, "round1_conf.json")):
            diffs.append(("round1_conf", "*"))
    else:
        ra, rb = _load(os.path.join(c0_dir, "round1_conf.json")), _load(os.path.join(arm_dir, "round1_conf.json"))
        for cam in ra:
            if ra[cam] != rb.get(cam):
                diffs.append(("round1_conf", cam))
            if pngs(os.path.join(c0_dir, "_r1"), cam) != pngs(os.path.join(arm_dir, "_r1"), cam):
                diffs.append(("r1_png", cam))
    return diffs


def check_pairs(base, cand):
    from mvseg2d.dev_compare import pair_key
    kb, kc = Counter(pair_key(r) for r in base), Counter(pair_key(r) for r in cand)
    return dict(ok=kb == kc and all(v == 1 for v in kb.values()), n_base=len(base), n_cand=len(cand),
                only_base=len(set(kb) - set(kc)), only_cand=len(set(kc) - set(kb)))


def subset(rows, keys):
    from mvseg2d.dev_compare import pair_key
    return [r for r in rows if pair_key(r) in keys]


def wide_keys(rows, angles_by_view, theta):
    from mvseg2d.dev_compare import pair_key
    return {pair_key(r) for r in rows if angles_by_view.get((r["scene"], int(r["t"]), r.get("ref"), r["cam"]), -1.0) > theta}


def rule_b(jf_b, jf_b2, b4, z):
    """§7.1 B1-B4.  jf_b / jf_b2: the J&F paired records (delta, ci, per_scene) of DEV-B / DEV-B'; b4: dict(n_pairs, dJ,
    other_arm, other_c0) over the DEV-B' pairs whose eval view is in F (n_pairs 0: vacuous); z: case Z."""
    def finite(*x):
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in x)
    lo_b, worst_b = jf_b["ci"][0], min(jf_b["per_scene"].values(), default=float("nan"))
    lo_b2, worst_b2 = jf_b2["ci"][0], min(jf_b2["per_scene"].values(), default=float("nan"))
    r = dict(B1=bool(finite(lo_b) and lo_b > DELTA), B2=bool(finite(worst_b) and worst_b >= SCENE_FLOOR),
             B3=bool(finite(lo_b2, worst_b2) and lo_b2 > DELTA and worst_b2 >= SCENE_FLOOR),
             ci_lo_B=lo_b, worst_scene_B=worst_b, ci_lo_B2=lo_b2, worst_scene_B2=worst_b2)
    if b4["n_pairs"] == 0:
        r.update(B4=True, B4_label="vacuous (no DEV-B' eval view in F)")
    else:
        r["B4"] = bool(finite(b4["dJ"]) and b4["dJ"] >= 0 and b4["other_arm"] <= b4["other_c0"])
    r["labels"] = dict(B1="vacuous (case Z)" if z else None, B2="vacuous (case Z)" if z else None)
    r["pass"] = r["B1"] and r["B2"] and r["B3"] and r["B4"]
    r["fail_type"] = None if r["pass"] else ("DEV-B'" if not (r["B3"] and r["B4"]) else "DEV-B")
    return r


def audit_needed(dJF_b, dJ_b2_wide):
    """§7.1: DEV-B all-pair |dJ&F| > 0.006 or DEV-B' wide-pair dJ > +0.05 (nan counts as needing an audit)."""
    a = not (isinstance(dJF_b, float) and math.isfinite(dJF_b)) or abs(dJF_b) > AUDIT_B
    b = dJ_b2_wide is not None and (not math.isfinite(dJ_b2_wide) or dJ_b2_wide > AUDIT_B2_WIDE)
    return bool(a or b)


def t06_ok(score):
    return score is not None and score["J_all"] >= G1_J and score["wrong_all"] <= G1_WRONG


def g1_route(step1_ok, theta, b, t06_main, t06_ab60, t06_ac60, t06_abc60, A_E, devB_eval_in_F_60, allowed60=True,
             proxy_b=None):
    """§7.3 -> dict(pass, route, C2, condition labels).  b: the stage-B rule record (None when theta_s* = inf: no main
    arm); t06_*: dict(J_all, wrong_all) or None.  allowed60: theta 60 is in Step 2's allowed set (§6.4: T06 wrong_all and
    the DEV-A' conditions of AB(s_main, 60)) -- fallback (a) adopts AB(60) only then, otherwise the route goes to (b).
    proxy_b: {"Ac" / "ABc": dict(ok, label)} the DEV-A' proxy of each (b) candidate (§7.3: Ac -> A(s_main, 60), ABc ->
    AB(s_main, 60); ok = the three §6.4 DEV-A' conditions at 60; the chain fallback itself cannot run on MUVOD).  The (b)
    candidates are those whose proxy passes; the pick among them is by T06 J_all, then fewer wrong_all, then Ac; no
    candidate -> G1 fails.  A proxy not decoded yet (ok None) is an error (decode it first, §6.7)."""
    out = dict(conditions={}, labels={})
    if not step1_ok:
        return dict(out, g1_pass=False, route="G1 fails (Step 1: no feasible statistic, §6.2)", C2="collapse analysis")
    relay_closed = math.isinf(theta)
    if not relay_closed:
        out["conditions"] = {"1": b["pass"], "2": t06_main["J_all"] >= G1_J, "3": t06_main["wrong_all"] <= G1_WRONG}
        out["labels"]["1"] = "DEV-B vacuous (case Z); content = DEV-B' (B3, B4)" if b.get("case_z") else None
        if b["fail_type"] == "DEV-B'":                       # no fallback acts on the DEV-B' wide pairs (§7.3)
            return dict(out, g1_pass=False, route="G1 fails (DEV-B' B3/B4: no fallback)", C2="collapse analysis")
        if b["pass"] and t06_ok(t06_main):
            return dict(out, g1_pass=True, route="main", C2="safe on the ring (main arm)")
    devB_fail = (not relay_closed) and b["fail_type"] == "DEV-B"
    tried = []
    if devB_fail:                                            # (a): theta 60 (> A_E): MUVOD eval outputs = C0 by construction
        if not (A_E < 60 and devB_eval_in_F_60 == 0):
            raise ValueError("fallback (a) needs A_E < 60 and no DEV-B eval cam in F at 60 (angles.json)")
        if not allowed60:                                    # theta 60 failed the §6.4 conditions: (a) is not open
            out["a_skipped"] = "theta 60 is not in Step 2's allowed set (§6.4): fallback (a) not taken -> (b)"
        else:
            tried.append("a")
            if t06_ok(t06_ab60):
                return dict(out, g1_pass=True, route="fallback (a): scope R, theta 60", C2="safe on the ring (fallback a)",
                            tried=tried, labels=dict(out["labels"], **{"1": "DEV-B vacuous, DEV-B' unverified (theta 60: "
                                                                             "DEV-A' design value only)"}))
    tried.append("b")                                        # (b): identity check + chain fallback, theta 60
    t06b = {"Ac": t06_ac60, "ABc": t06_abc60}
    for arm in ("Ac", "ABc"):
        if t06b[arm] is None:
            raise ValueError(f"fallback (b) needs the T06 {arm}(60) score")
        g = (proxy_b or {}).get(arm)
        if g is None or g.get("ok") is None:
            raise ValueError(f"fallback (b): the DEV-A' proxy of {arm}(60) ({'A' if arm == 'Ac' else 'AB'}(s_main, 60)) is not "
                             "decoded / scored yet (§6.7)")
    cands = [a for a in ("Ac", "ABc") if proxy_b[a]["ok"]]
    out["b_candidates"] = cands
    if not cands:
        return dict(out, g1_pass=False, route="G1 fails ((b): no candidate passes its DEV-A' proxy)", C2="collapse analysis",
                    tried=tried, proxy_b=proxy_b)
    pick = sorted(cands, key=lambda a: (-t06b[a]["J_all"], t06b[a]["wrong_all"], a != "Ac"))[0]
    gate = proxy_b[pick]
    lab = dict(out["labels"], **{"1": "DEV-B vacuous, DEV-B' unverified (theta 60: DEV-A' design value only); (b): the chain "
                                      f"fallback is unverified in-domain, DEV-A' proxy = {gate['label']}"})
    if t06_ok(t06b[pick]):
        return dict(out, g1_pass=True, route=f"fallback (b): {pick}(60)", C2="detection", tried=tried, labels=lab, b_arm=pick,
                    proxy_b=proxy_b)
    return dict(out, g1_pass=False, route=f"G1 fails ((b) {pick}(60) also misses)", C2="collapse analysis", tried=tried,
                b_arm=pick, proxy_b=proxy_b)


def h23(pairs_far, neighbours_trusted, covers_arm, covers_r1, covers_neigh):
    """§5.3.  pairs_far: [(cam, obj)] of the relayed far pairs; neighbours_trusted[(cam, obj)] = neighbour views whose
    memory trusts obj; covers_*: {(cam, obj): covers} (score.py rule); covers_neigh: the outputs the neighbour memories
    were encoded from.  -> dict(P_good, w1, wX, verdict)."""
    good = [p for p in pairs_far if any(covers_neigh.get((m, p[1])) == p[1] for m in neighbours_trusted.get(p, []))]
    w1 = sum(covers_r1.get(p, 0) not in (0, p[1]) for p in good)
    wx = sum(covers_arm.get(p, 0) not in (0, p[1]) for p in good)
    if w1 < H_MIN:
        v = "undecidable (w1 < 8, report only)"
    elif wx >= H3_SHARE * w1:
        v = "H3 supported"
    elif wx <= H2_SHARE * w1:
        v = "H2 supported"
    else:
        v = "mixed"
    return dict(P_good=len(good), w1=int(w1), wX=int(wx), verdict=v)


def choice_report(id3d_confs):
    """§8 reporting rules: the final choices split by choice / X state, the absent objects ('keep' with X EMPTY and round 1
    not PASS) apart, chain files missing; config n_fallback is never reported as 'fallbacks'."""
    by, by_x, absent, flagged = Counter(), Counter(), 0, 0
    chain = dict(n_chosen=0, n_missing=0, runs_without_any=0)
    for rec in id3d_confs:
        s = rec.get("summary") or {}
        by.update(s.get("by_choice", {}))
        by_x.update(s.get("by_choice_x_state", {}))
        absent += s.get("keep_absent", 0)
        flagged += s.get("keep_flagged", 0)
        if rec.get("chain"):
            chain["n_chosen"] += rec["chain"]["n_chosen"]
            chain["n_missing"] += rec["chain"]["n_missing"]
    if chain["n_chosen"] and chain["n_missing"] == chain["n_chosen"]:
        chain["note"] = "every chain choice missed its PNG: the run equals the empty-mask arm -> invalid, fix the path (§8)"
    return dict(by_choice=dict(by), by_choice_x_state=dict(sorted(by_x.items())), keep_absent=absent, keep_flagged=flagged,
                chain=chain if any(chain.values()) else None,
                note="config n_fallback counts 'keep' and absent objects: not reported as fallbacks; gated_pairs_frac / "
                     "views_dropped are not compared with C0 (E15 counts chosen neighbours only); see relay counts")


def state_ratios(recs, star=None):
    """§8: PASS / FLAG / ABSTAIN / EMPTY counts and ratios per stage over id3d_conf.json records: round 1, round 2 of the
    star views (round2_S) and of Jacobi far views (round2_F), relay, GN re-decodes, final (the chosen state; None for a
    chain / empty fallback).  star: per record, the star views (relay.json S); None = every round-2 view is a star view."""
    from collections import defaultdict
    cnt = defaultdict(Counter)
    for i, rec in enumerate(recs):
        S = None if star is None else set(star[i])
        for stage in ("round1", "round2", "relay", "gn"):
            for cam, objs in (rec.get(stage) or {}).items():
                key = stage if stage != "round2" else ("round2_S" if S is None or cam in S else "round2_F")
                cnt[key].update(objs.values())
        for cam, objs in (rec.get("final") or {}).items():
            cnt["final"].update(str(x.get("state")) for x in objs.values())
    out = {}
    for stage, c in cnt.items():
        n = sum(c.values())
        out[stage] = dict(n=n, counts={k: int(c.get(k, 0)) for k in STATES + (("None",) if "None" in c else ())},
                          ratios={k: c.get(k, 0) / n for k in STATES} if n else {})
    return out


def hallucinations(pred, gt, c0):
    """§8: pred / c0: {(view, obj): predicted pixels}; gt: {(view, obj): GT non-empty} over the same keys.  -> the
    hallucinations (a non-empty prediction of an input object absent from the view's GT) of the arm and of C0, count and
    pixels, and the masks newly non-empty against C0 (all keys, and among the absent objects)."""
    keys = sorted(pred)
    absent = [k for k in keys if not gt[k]]
    return dict(n_keys=len(keys), n_absent=len(absent), arm=sum(pred[k] > 0 for k in absent),
                arm_px=int(sum(pred[k] for k in absent)), c0=sum(c0.get(k, 0) > 0 for k in absent),
                c0_px=int(sum(c0.get(k, 0) for k in absent)),
                newly_nonempty=sum(pred[k] > 0 and not c0.get(k, 0) for k in keys),
                newly_nonempty_absent=sum(pred[k] > 0 and not c0.get(k, 0) for k in absent))


def far_correct_j(rows):
    """§8 / H5: mean J of the far pairs (ring distance >= 7) whose mask covers the right person (score.py rows)."""
    far = [r for r in rows if r["d"] >= 7 and r["covers"] == r["p"]]
    return dict(n=len(far), J=float(sum(r["J"] for r in far) / len(far)) if far else None, c0_diagnosis=C0_FAR_CORRECT_J)


def config_check(cfg, theta, family, row="AB"):
    """A decoded config's E15 keys (run.RelayRound.config) -> list of mismatches with (theta, family, row)."""
    from methods.geomem import run as R
    if cfg is None or "relay_theta" not in cfg:
        return [("relay_theta", None)]
    th = math.inf if cfg["relay_theta"] == "inf" else float(cfg["relay_theta"])
    fam = None if cfg.get("id3d") in (None, "off") else cfg["id3d"]
    r = R.e15_row(dict(family=fam, scope=cfg.get("id3d_scope"), mode=cfg.get("relay_mode"), order=cfg.get("relay_order"),
                       mem=cfg.get("relay_mem"), trust=cfg.get("relay_trust"), flagged=cfg.get("flagged")))
    return [(k, v) for k, v, want in (("relay_theta", th, theta), ("id3d", fam, family), ("row", r, row)) if v != want]


def check_main_preset(theta, fam, rots=(0, 1, 2)):
    """Review 0 #3: the main preset and every decoded main-preset frame carry theta_s*, the main family and the AB row.
    -> {where: mismatches} (empty = consistent)."""
    from types import SimpleNamespace
    from methods.geomem import run as R
    from methods.geomem.stage1 import PRESETS
    bad = {}
    if MAIN_PRESET not in PRESETS:
        return {"PRESETS": [("missing", MAIN_PRESET)]}
    o = R.e15_options(SimpleNamespace(**PRESETS[MAIN_PRESET]))
    d = [("relay", None)] if o is None else \
        [(k, v) for k, v, want in (("theta", o["theta"], theta), ("family", o["family"], fam), ("row", o["row"], "AB"))
         if v != want]
    if d:
        bad["PRESETS"] = d
    for r in rots:
        root = _root(MAIN_PRESET, "B", r)
        for dp, _, fs in os.walk(root):
            if "config.json" in fs and "DONE" in fs:
                m = config_check(_load(os.path.join(dp, "config.json")), theta, fam)
                if m:
                    bad[os.path.relpath(dp, DATA)] = m
    return bad


# ------------------------------------------------------------------------------------------------- stage B (§7.1)
def _root(preset, split, rot=0):
    return os.path.join(DATA, "_dev", f"{preset}_{split}" + (f"_rot{rot}" if rot else ""))


def _scores(preset, split, rot=0):
    return SEL._score_rows(os.path.basename(_root(preset, split, rot)))


def invariants(rot, z, angles_rec, theta):
    """Every frame of the main-preset DEV-B (rot 0) / DEV-B' (rot 1, 2) decode against C0 -> {frame: diffs}."""
    c0, arm = _root(C0_PRESET, "B", rot), _root(MAIN_PRESET, "B", rot)
    idx = _load(os.path.join(arm, "index.json"))
    bad = {}
    for s, S in idx["scenes"].items():
        for t in S["frames"]:
            fa, fc = os.path.join(arm, s, f"{t:06d}"), os.path.join(c0, s, f"{t:06d}")
            if not os.path.isfile(os.path.join(fa, "DONE")) or not os.path.isfile(os.path.join(fc, "DONE")):
                bad[f"{s}/{t}"] = [("missing", "DONE")]
                continue
            rel = _load(os.path.join(fa, "relay.json")) if os.path.isfile(os.path.join(fa, "relay.json")) else None
            star = [c for c in S["eval_cams"] if rel is None or c in rel["S"] or not rel["active"]]
            d = frame_compare(fc, fa, S["eval_cams"], z and rot == 0, star)
            if d:
                bad[f"{s}/{t}"] = d
    return bad


def stage_b(audited=False):
    from mvseg2d.dev_compare import compare
    from mvseg2d.paired import paired_scene_bootstrap
    ang = _load(os.path.join(OUT, "angles.json"))
    s2 = _load(os.path.join(OUT, "step2.json"))
    s2b = _load(os.path.join(OUT, "step2b.json")) if os.path.isfile(os.path.join(OUT, "step2b.json")) else None
    theta, fam = theta_and_family(s2, s2b)
    if math.isinf(theta):
        _save("stageB", dict(note="theta_s* = inf: no main arm on DEV-B (relay closed); G1 goes to fallback (b)"))
        print("stage B: theta_s* = inf, nothing to judge")
        return
    mism = check_main_preset(theta, fam)
    if mism:
        _save("stageB", dict(theta_s=SEL.theta_tag(theta), family=fam, main_preset_mismatch=mism,
                             note="the main preset / its decodes disagree with step2 / step2b: stop (review 0 #3)"))
        print(f"stage B: the main preset disagrees with the selection (theta {theta}, {fam}): {list(mism.items())[:5]}")
        raise SystemExit(2)
    z = case_z(ang["summary"], theta)
    base_b, arm_b = SEL.c0_rows("B"), _scores(MAIN_PRESET, "B")
    base_b2 = _scores(C0_PRESET, "B", 1) + _scores(C0_PRESET, "B", 2)
    arm_b2 = _scores(MAIN_PRESET, "B", 1) + _scores(MAIN_PRESET, "B", 2)
    no_field = {k: sum(not ("covers" in r and "other_obj_mask" in r) for r in rows)
                for k, rows in (("C0 B", base_b), ("main B", arm_b), ("C0 B'", base_b2), ("main B'", arm_b2))}
    if any(no_field.values()):                               # I12: never count a missing field as False
        _save("stageB", dict(theta_s=SEL.theta_tag(theta), family=fam, rows_without_covers=no_field,
                             note="scored rows without covers / other_obj_mask: re-score with the current dev_score (I12)"))
        raise SystemExit(f"stage B: rows without covers / other_obj_mask {no_field}: re-score (prereg §7.1, I12)")
    pairs = dict(B=check_pairs(base_b, arm_b), B2=check_pairs(base_b2, arm_b2))
    if not (pairs["B"]["ok"] and pairs["B2"]["ok"]):
        _save("stageB", dict(pairs=pairs, note="pair sets differ: abort (§7.1)"))
        raise SystemExit(3)
    inv = {f"rot{r}": invariants(r, z, ang, theta) for r in (0, 1, 2)}
    if any(inv.values()):
        _save("stageB", dict(pairs=pairs, case_z=z, invariants=inv, note="invariant / case-Z difference: a bug, fix and decode again (§7.1, §8)"))
        raise SystemExit(2)
    cb, cb2 = compare(base_b, arm_b), compare(base_b2, arm_b2)
    rot_b = SEL.rotated_angles(ang, "B'")
    wide = wide_keys(base_b2, rot_b, SEL.WIDE - 1e-9)
    dJ_wide = paired_scene_bootstrap(subset(base_b2, wide), subset(arm_b2, wide), key="J", n=10_000, seed=0,
                                     view_key="view")["delta"] if wide else None
    if audit_needed(cb["paired"]["JF"]["delta"], dJ_wide) and not audited:
        _save("stageB", dict(pairs=pairs, case_z=z, audit_required=True, audited=False, dJF_B=cb["paired"]["JF"]["delta"],
                             dJ_B2_wide=dJ_wide))
        print(f"STOP AND AUDIT (§7.1): DEV-B dJ&F {cb['paired']['JF']['delta']:+.4f}, DEV-B' wide dJ {dJ_wide}")
        raise SystemExit(4)
    inF = wide_keys(base_b2, rot_b, theta)
    b4 = dict(n_pairs=len(inF))
    if inF:
        B, C = subset(base_b2, inF), subset(arm_b2, inF)
        no_field = sum("other_obj_mask" not in r for r in B + C)
        if no_field:                                         # review 0 #10: a scoring older than the field would count 0
            raise SystemExit(f"B4: {no_field} DEV-B' rows lack other_obj_mask: re-score with the current dev_score")
        b4.update(dJ=paired_scene_bootstrap(B, C, key="J", n=10_000, seed=0, view_key="view")["delta"],
                  other_c0=sum(bool(r["other_obj_mask"]) for r in B), other_arm=sum(bool(r["other_obj_mask"]) for r in C))
    rule = rule_b(cb["paired"]["JF"], cb2["paired"]["JF"], b4, z)
    rule["case_z"] = z
    reports = dict(dJ_B=cb["paired"]["J"]["delta"], dF_B=cb["paired"]["F"]["delta"], dJ_B2=cb2["paired"]["J"]["delta"],
                   dF_B2=cb2["paired"]["F"]["delta"], per_scene_B=cb["paired"]["JF"]["per_scene"],
                   per_scene_B2=cb2["paired"]["JF"]["per_scene"], dJ_B2_wide=dJ_wide, n_wide_B2=len(wide),
                   choices=choice_report(_frame_jsons(MAIN_PRESET, "B", "id3d_conf.json")),
                   state_ratios=_dev_state_ratios(),
                   hallucinations=dict(B=_dev_hallucinations(C0_PRESET, MAIN_PRESET, "B", (0,)),
                                       B2=_dev_hallucinations(C0_PRESET, MAIN_PRESET, "B", (1, 2))),
                   seconds={f"rot{r}": _seconds(r) for r in (0, 1, 2)})
    res = dict(theta_s=SEL.theta_tag(theta), family=fam, case_z=z, pairs=pairs, invariants=inv, rule=rule, b4=b4,
               audited=bool(audited), reports=reports)
    _save("stageB", res)
    print(f"stage B: case Z {z}; B1 {rule['B1']} B2 {rule['B2']} B3 {rule['B3']} B4 {rule['B4']} -> "
          f"{'PASS' if rule['pass'] else 'FAIL (' + rule['fail_type'] + ')'}")


def _frame_jsons(preset, split, name, rots=(0, 1, 2)):
    out = []
    for r in rots:
        root = _root(preset, split, r)
        for dp, _, fs in os.walk(root):
            if name in fs:
                out.append(_load(os.path.join(dp, name)))
    return out


def _frame_pairs(preset, split, rots):
    """(relay.json, id3d_conf.json) of every decoded frame that has an id3d_conf.json, by scene."""
    out = {}
    for r in rots:
        root = _root(preset, split, r)
        for dp, _, fs in os.walk(root):
            if "id3d_conf.json" in fs:
                scene = os.path.relpath(dp, root).split(os.sep)[0]
                rel = _load(os.path.join(dp, "relay.json")) if "relay.json" in fs else None
                out.setdefault(scene, []).append((rel, _load(os.path.join(dp, "id3d_conf.json"))))
    return out


def _dev_state_ratios(rots=(0, 1, 2)):
    """§8 state ratios of the main preset on DEV-B / DEV-B' (where Fix A ran), per scene and overall."""
    out = {}
    for name, rs in (("B", (0,)), ("B2", (1, 2))):
        per = _frame_pairs(MAIN_PRESET, "B", [r for r in rs if r in rots])
        allr = [x for v in per.values() for x in v]
        out[name] = dict(all=state_ratios([x[1] for x in allr], [x[0]["S"] if x[0] else [] for x in allr]),
                         per_scene={s: state_ratios([x[1] for x in v], [x[0]["S"] if x[0] else [] for x in v])
                                    for s, v in per.items()},
                         n_frames_with_fix_a=len(allr))
    return out


def _dev_hallucinations(c0_preset, arm_preset, split, rots):
    """§8 on DEV: the scorings' rows (GT non-empty) and empty_gt entries (GT empty): seed_px > 0 = a non-empty mask."""
    def recs(preset):
        if preset == C0_PRESET and rots == (0,):
            SEL.c0_rows(split)                               # the E14 regression check first
        out = []
        for r in rots:
            out += SEL._load_scoring(os.path.basename(_root(preset, split, r)))
        return out

    def table(rows):
        from mvseg2d.dev_compare import pair_key
        return {pair_key(r): (int(r["seed_px"]), r.get("gt_px", 0) > 0) for r in rows}
    a, c = table(recs(arm_preset)), table(recs(c0_preset))
    if set(a) != set(c):
        return dict(error="pair sets differ")
    return hallucinations({k: v[0] for k, v in a.items()}, {k: v[1] for k, v in a.items()}, {k: v[0] for k, v in c.items()})


def _seconds(rot):
    return dict(arm=sum(c.get("seconds", 0.0) for c in _frame_jsons(MAIN_PRESET, "B", "config.json", (rot,))),
                c0=sum(c.get("seconds", 0.0) for c in _frame_jsons(C0_PRESET, "B", "config.json", (rot,))))


# ------------------------------------------------------------------------------------------------- stage T06 (§5.3, §8)
def _t06_out(name):
    return os.path.join(T06_RESULT, f"e15_{name}")


def stage_t06(t06_scores, r1_arm):
    sc = _load(t06_scores)
    s2 = _load(os.path.join(OUT, "step2.json"))
    s2b = _load(os.path.join(OUT, "step2b.json")) if os.path.isfile(os.path.join(OUT, "step2b.json")) else None
    theta, fam = theta_and_family(s2, s2b)
    cov = {a: {(r["cam"], r["p"]): r["covers"] for r in rows} for a, rows in sc["rows"].items()}
    if r1_arm not in cov:
        raise SystemExit(f"T06 round-1 scores '{r1_arm}' missing from {t06_scores}")
    res = dict(theta_s=SEL.theta_tag(theta), family=fam, h23={}, reports={})
    main_run = os.path.join(_t06_out(SEL.run_name("AB", theta, fam)), "RUN.json")
    if not math.isinf(theta) and os.path.isfile(main_run):   # review 0 #3: the T06 main decode is the selection's
        m = config_check(_load(main_run).get("config"), theta, fam)
        if m:
            raise SystemExit(f"T06 {SEL.run_name('AB', theta, fam)}: the decode's config disagrees with the selection {m}")
    for arm in ("AB", "A"):
        name = SEL.run_name(arm, theta, fam)
        rp = os.path.join(_t06_out(name), "relay.json")
        if math.isinf(theta) or name not in cov or not os.path.isfile(rp):
            res["h23"][arm] = dict(note=f"{name}: no decode / score")
            continue
        rel = _load(rp)
        views = rel["views"] if arm == "AB" else rel["jacobi"]
        pairs = [k for k in cov[name] if k[0] in views]
        neigh = {k: [m for m, objs in views[k[0]].get("trusted", {}).items() if k[1] in objs] for k in pairs}
        res["h23"][arm] = h23(pairs, neigh, cov[name], cov[r1_arm], cov[name] if arm == "AB" else cov[r1_arm])
    nonempty = {}
    gt = t06_gt_nonempty()
    c0_dir = sc["arms"].get("C0", {}).get("mask_dir") or _t06_out("C0")
    nonempty["C0"] = t06_nonempty(c0_dir)
    for name in sorted(cov):
        rep = dict(far_correct_J=far_correct_j(sc["rows"][name]))
        d = _t06_out(name)
        if os.path.isdir(d):
            rel = _load(os.path.join(d, "relay.json")) if os.path.isfile(os.path.join(d, "relay.json")) else None
            rec = _load(os.path.join(d, "id3d_conf.json")) if os.path.isfile(os.path.join(d, "id3d_conf.json")) else None
            rep.update(choices=choice_report([rec] if rec else []), relay_counts=rel and rel.get("counts"),
                       n_relay=rel and len(rel.get("order", [])), unreachable=rel and rel.get("unreachable"),
                       orphan_views=rel and sum(v.get("orphan", False) for v in rel.get("views", {}).values()),
                       state_ratios=state_ratios([rec], [rel["S"] if rel else []]) if rec else None)
        mask_dir = sc["arms"].get(name, {}).get("mask_dir")
        if mask_dir and os.path.isdir(mask_dir) and name not in ("C0", "C0_r1"):
            rep["hallucinations"] = hallucinations(t06_nonempty(mask_dir), gt, nonempty["C0"])
        res["reports"][name] = rep
    res["devA2_state_ratios"] = _devA2_state_ratios()        # §8: the DEV-A' arms' state ratios (where Fix A ran)
    _save("stageT06", res)
    for arm, h in res["h23"].items():
        print(f"§5.3 {arm}: {h}")


def _devA2_state_ratios():
    """{run: per-scene and overall state ratios} of the DEV-A' override decodes Data/_dev/e15/<run>_A_rot{1,2}."""
    root = os.path.join(DATA, "_dev", "e15")
    runs = {}
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            for r in (1, 2):
                if d.endswith(f"_A_rot{r}"):
                    runs.setdefault(d[: -len(f"_rot{r}")], []).append(os.path.join(root, d))
    out = {}
    for name, dirs in runs.items():
        per = {}
        for top in dirs:
            for dp, _, fs in os.walk(top):
                if "id3d_conf.json" in fs:
                    scene = os.path.relpath(dp, top).split(os.sep)[0]
                    rel = _load(os.path.join(dp, "relay.json")) if "relay.json" in fs else None
                    per.setdefault(scene, []).append((rel, _load(os.path.join(dp, "id3d_conf.json"))))
        if per:
            allr = [x for v in per.values() for x in v]
            out[name] = dict(all=state_ratios([x[1] for x in allr], [x[0]["S"] if x[0] else [] for x in allr]),
                             per_scene={sc: state_ratios([x[1] for x in v], [x[0]["S"] if x[0] else [] for x in v])
                                        for sc, v in per.items()})
    return out


def t06_gt_nonempty():
    """{(cam, person): GT non-empty at 1/4 resolution (score.py's rule)} over the non-reference T06 cameras, persons 1-6."""
    from PIL import Image
    import numpy as np
    out = {}
    for i in range(32):
        cam = f"cam_{i:02d}"
        if cam == SEL.T06_REF:
            continue
        m = Image.open(SEL.T06_GT.format(cam=cam))
        q = np.array(m.resize((m.width // 4, m.height // 4), Image.NEAREST))
        for p in range(1, 7):
            out[(cam, p)] = bool((q == p).any())
    return out


def t06_nonempty(mask_dir):
    """{(cam, person): predicted pixels (full resolution, PNG > 127; absent PNG = 0)} for one T06 Result folder."""
    from PIL import Image
    import numpy as np
    out = {}
    for i in range(32):
        cam = f"cam_{i:02d}"
        if cam == SEL.T06_REF:
            continue
        for p in range(1, 7):
            f = os.path.join(mask_dir, cam, f"{p}.png")
            out[(cam, p)] = int((np.array(Image.open(f)) > 127).sum()) if os.path.isfile(f) else 0
    return out


# ------------------------------------------------------------------------------------------------- stage G1 (§7.3)
def stage_g1(t06_scores):
    sel = _load(os.path.join(OUT, "select.json"))
    ang = _load(os.path.join(OUT, "angles.json"))
    step1_ok = bool(sel.get("s_star"))
    s2 = _load(os.path.join(OUT, "step2.json")) if step1_ok else None
    s2b = _load(os.path.join(OUT, "step2b.json")) if os.path.isfile(os.path.join(OUT, "step2b.json")) else None
    theta, fam = theta_and_family(s2, s2b) if step1_ok else (math.inf, None)
    judge = _load(JUDGE) if os.path.isfile(JUDGE) else {}
    b = judge.get("stageB", {}).get("rule")
    if step1_ok and not math.isinf(theta) and b is None:
        raise SystemExit("stage B has not been judged (run --stage B first)")
    if b is not None:
        b = dict(b, case_z=judge["stageB"]["case_z"])
    arms = _load(t06_scores)["arms"] if step1_ok else {}

    def t06(arm, th):
        a = arms.get(SEL.run_name(arm, th, fam))
        return None if a is None else dict(J_all=a["J_all"], wrong_all=a["wrong_all"])
    main = None if (not step1_ok or math.isinf(theta)) else t06("AB", theta)
    if main is not None:                                     # review 0 #3: the scored T06 main decode is the selection's
        cfg = _load(os.path.join(_t06_out(SEL.run_name("AB", theta, fam)), "RUN.json")).get("config")
        m = config_check(cfg, theta, fam)
        if m:
            raise SystemExit(f"T06 {SEL.run_name('AB', theta, fam)}: the decode's config disagrees with the selection {m}")
    allowed60, proxy_b = True, None
    if step1_ok:
        allowed60, proxy_b = g1_inputs(s2, fam, ang, arms)
    r = g1_route(step1_ok, theta, b, main, t06("AB", 60.0), t06("Ac", 60.0), t06("ABc", 60.0),
                 ang["summary"]["A_E"]["a"], ang["summary"]["in_F"]["60"]["B"]["n_eval_in_F"], allowed60=allowed60,
                 proxy_b=proxy_b)
    r.update(theta_s=SEL.theta_tag(theta), family=fam, t06_main=main, allowed60=allowed60, proxy_b_all=proxy_b,
             reported=None if main is None else dict(
                 {k: arms[SEL.run_name("AB", theta, fam)].get(k) for k in ("Jfull_all", "J_far", "empty_all", "empty_far",
                                                                           "wrong_far")},
                 far_correct_J=far_correct_j(_load(t06_scores)["rows"][SEL.run_name("AB", theta, fam)])),
             definitions="J_all / wrong_all = ring_diag/d1_d5/score.py (1/4 resolution, 184 pairs); development targets, not "
                         "test evidence (§7.3)")
    _save("G1", r)
    print(f"G1: {'PASS' if r['g1_pass'] else 'FAIL'} via {r['route']}; C2 -> {r['C2']}; labels {r['labels']}")


def g1_inputs(s2, fam, ang, arms):
    """§6.7 / §7.3: allowed60 for fallback (a) and the (b) proxies, with fam = s_main.  No swap (fam = s*): theta 60 in
    step2.json `allowed`, and the ABc proxy = the Step-2 grid record of AB(s*, 60).  After a swap (fam = s'): allowed60 =
    AB(s', 60) has T06 wrong_all <= 28 and the three DEV-A' conditions; the ABc proxy = that DEV-A' record.  The Ac proxy
    = A(s_main, 60) on DEV-A' (θ_s* = 60: the A arm's θ_s* run)."""
    if fam == s2["family"]:
        allowed60 = "60" in s2["rule"].get("allowed", [])
        ab = s2["devA2"].get("60")
        ab = None if ab is None else dict(ab, ok=bool(ab["safe"]["ok"] and (ab["wide_dJ"] is None or ab["wide_dJ"] >= 0)))
    else:
        ab = SEL.devA2_conditions("AB", fam, 60.0, ang)
        t = arms.get(SEL.run_name("AB", 60.0, fam))
        allowed60 = bool(ab is not None and ab["ok"] and t is not None and t["wrong_all"] <= SEL.WRONG_C0)
    a60 = SEL.devA2_conditions("A", fam, 60.0, ang)
    proxy = {"ABc": dict(ok=None if ab is None else ab["ok"], label=f"AB({fam}, 60) on DEV-A'", record=ab),
             "Ac": dict(ok=None if a60 is None else a60["ok"], label=f"A({fam}, 60) on DEV-A'", record=a60)}
    return allowed60, proxy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("B", "T06", "G1"))
    ap.add_argument("--audited", action="store_true", help="stage B: the stop-and-audit check was done")
    ap.add_argument("--t06-scores", default=os.path.join(WORK, "t06_scores.json"), dest="t06_scores")
    ap.add_argument("--t06-r1", default="C0_r1", dest="t06_r1", help="score-json arm of the T06 C0 round-1 masks")
    a = ap.parse_args()
    if a.stage == "B":
        stage_b(a.audited)
    elif a.stage == "T06":
        stage_t06(a.t06_scores, a.t06_r1)
    else:
        stage_g1(a.t06_scores)


if __name__ == "__main__":
    main()

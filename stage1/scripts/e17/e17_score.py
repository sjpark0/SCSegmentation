#!/usr/bin/env python3
"""S1-E17 part 1 (docs/stage1-E17-prereg.md §3, §5): the C-ov arms x seeds on DEV-A / DEV-A' against C0 (geoset_rep_xs seed 0)
and AB (the main arm, E15 seed 0; DEV-A = C0 byte-identical), the T06 scores (final and round 1), the lambda rule.

    stage1/.venv/bin/python scripts/e17/e17_score.py [--select] [--out geomem/runs/_e17/summary.json]

--select applies the registered rule to the seed-0 scores and writes scripts/e17/runs/select.json (refuses to overwrite).
Inputs: geomem/runs/_e17/scores/<arm>_s<seed>_A[_rot<r>].json, the E15 scorings of C0 / AB, geomem/runs/_e16/scores/AB_s{1,2}_*
(the main arm's retrain noise), geomem/runs/_e17/t06_scores.json, scripts/e15/runs/angles.json (DEV-A' wide pairs).
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(S1, "scripts", "e15"))
import e15_select as SEL  # noqa: E402

WORK = os.path.join(S1, "geomem", "runs", "_e17")
SCORES = os.path.join(WORK, "scores")
E15_SCORES = os.path.join(S1, "geomem", "runs", "_e15", "scores")
E16_SCORES = os.path.join(S1, "geomem", "runs", "_e16", "scores")
SELECT = os.path.join(HERE, "runs", "select.json")
ARMS = ("C-ov2", "C-ov4")
LAMBDA = {"C-ov2": 2.0, "C-ov4": 4.0}
SEEDS = (0, 1, 2)
WIDE = 55.0


def score_path(arm, seed, rot):
    suf = "" if rot == 0 else f"_rot{rot}"
    if arm == "AB":                          # E15 seed 0: DEV-A identical to C0 (case Z), rotations AB_60_cstar
        return os.path.join(E15_SCORES, ("geoset_rep_xs_A.json" if rot == 0 else f"AB_60_cstar_A{suf}.json")) if seed == 0 \
            else os.path.join(E16_SCORES, f"AB_s{seed}_A{suf}.json")
    return os.path.join(SCORES, f"{arm}_s{seed}_A{suf}.json")


def rows(path):
    from mvseg2d.dev_compare import load
    return load(path)


def mean_jf(rs):
    from mvseg2d.paired import scene_means
    m = scene_means(rs, "JF", "view")
    return float(sum(m.values()) / len(m)) if m else float("nan")


def versus(base, cand, ang=None):
    from mvseg2d.dev_compare import compare
    c = compare(base, cand)
    p = c["paired"]
    out = dict(n=c["n_pairs"], dJF=p["JF"]["delta"], ci=p["JF"]["ci"], dJ=p["J"]["delta"], dF=p["F"]["delta"],
               worst_scene=min(p["JF"]["per_scene"].values()), per_scene=p["JF"]["per_scene"])
    if ang is not None:
        out["wide_dJ"] = SEL.wide_subset_delta(base, cand, ang, WIDE)
    return out


def one(arm, seed, base, ang_a2):
    """-> {A: {JF, vs_C0}, A2: {JF, vs_C0, vs_AB}} for the sets whose scores exist."""
    out = {}
    pa = score_path(arm, seed, 0)
    if os.path.isfile(pa):
        ca = rows(pa)
        out["A"] = dict(JF=mean_jf(ca), vs_C0=versus(base["C0_A"], ca))
    p1, p2 = score_path(arm, seed, 1), score_path(arm, seed, 2)
    if os.path.isfile(p1) and os.path.isfile(p2):
        ca2 = rows(p1) + rows(p2)
        out["A2"] = dict(JF=mean_jf(ca2), vs_C0=versus(base["C0_A2"], ca2, ang_a2), vs_AB=versus(base["AB_A2"], ca2, ang_a2))
    return out


def t06(path):
    if not os.path.isfile(path):
        return {}
    arms = json.load(open(path))["arms"]
    out = {}
    for k, v in arms.items():
        if k.endswith("_r1"):
            continue
        r1 = arms.get(k + "_r1") or {}
        out[k] = dict(J_all=v["J_all"], wrong_all=v["wrong_all"], J_far=v.get("J_far"), Jfull_all=v.get("Jfull_all"),
                      r1_J_all=r1.get("J_all"), r1_J_far=r1.get("J_far"), r1_wrong_all=r1.get("wrong_all"))
    return out


def msd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return dict(n=len(xs), mean=m, sd=sd)


def select_lambda(res):
    """Prereg §3: among the arms with seed-0 DEV-A and DEV-A' scores, the largest DEV-A' wide-pair dJ vs AB; ties (within 0.001)
    by the DEV-A dJ&F CI lower bound vs C0.  -> (chosen arm, table) or (None, table) when a score is missing."""
    table = {}
    for arm in ARMS:
        r = res.get(f"{arm}_s0", {})
        if "A" in r and "A2" in r:
            table[arm] = dict(wide_dJ_vs_AB=r["A2"]["vs_AB"]["wide_dJ"], ci_lo_A_vs_C0=r["A"]["vs_C0"]["ci"][0],
                              dJF_A2_vs_AB=r["A2"]["vs_AB"]["dJF"])
    if len(table) < len(ARMS):
        return None, table
    best = max(table, key=lambda k: table[k]["wide_dJ_vs_AB"])
    tied = [k for k in table if abs(table[k]["wide_dJ_vs_AB"] - table[best]["wide_dJ_vs_AB"]) <= 0.001]
    if len(tied) > 1:
        best = max(tied, key=lambda k: table[k]["ci_lo_A_vs_C0"])
    return best, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK, "summary.json"))
    ap.add_argument("--select", action="store_true", help="apply the lambda rule to the seed-0 scores -> runs/select.json")
    a = ap.parse_args()
    base = {"C0_A": rows(os.path.join(E15_SCORES, "geoset_rep_xs_A.json")),
            "C0_A2": rows(os.path.join(E15_SCORES, "geoset_rep_xs_A_rot1.json")) + rows(os.path.join(E15_SCORES, "geoset_rep_xs_A_rot2.json")),
            "AB_A2": rows(os.path.join(E15_SCORES, "AB_60_cstar_A_rot1.json")) + rows(os.path.join(E15_SCORES, "AB_60_cstar_A_rot2.json"))}
    ang_a2 = SEL.rotated_angles(json.load(open(os.path.join(S1, "scripts", "e15", "runs", "angles.json"))), "A'")
    t = t06(os.path.join(WORK, "t06_scores.json"))
    res = {}
    for arm in ARMS:
        for seed in SEEDS:
            r = one(arm, seed, base, ang_a2)
            tk = t.get(f"e17_{arm}_s{seed}")
            if r or tk:
                res[f"{arm}_s{seed}"] = dict(arm=arm, seed=seed, lam=LAMBDA[arm], **r, t06=tk)
    ref = {"C0": dict(t06=t.get("C0")), "AB_s0": dict(t06=t.get("AB_60_cstar"))}
    for seed in (1, 2):                     # the main arm's retrains (E16) as the noise reference
        r = one("AB", seed, base, ang_a2)
        if r:
            ref[f"AB_s{seed}"] = r
    chosen, table = select_lambda(res)
    if a.select:
        if os.path.isfile(SELECT):
            raise SystemExit(f"{SELECT} exists (the lambda choice is made once); delete it deliberately to redo")
        if chosen is None:
            raise SystemExit(f"seed-0 scores incomplete: {sorted(table)}")
        with open(SELECT, "w") as f:
            json.dump(dict(prereg="docs/stage1-E17-prereg.md §3", rule="max DEV-A' wide-pair dJ vs AB; tie (<= 0.001) -> DEV-A dJ&F CI lower bound vs C0",
                           wide_deg=WIDE, table=table, chosen=chosen, lam=LAMBDA[chosen]), f, indent=1)
        print(f"chosen {chosen} (lambda {LAMBDA[chosen]:g}) {table} -> {SELECT}")
        return
    per_arm = {arm: {st: {q: msd([res[k][st][q] if q == "JF" else res[k][st]["vs_AB" if st == "A2" else "vs_C0"].get(q)
                                 for k in res if res[k]["arm"] == arm and st in res[k]]) for q in ("JF", "dJF", "dJ", "wide_dJ")}
                     for st in ("A", "A2")} for arm in ARMS}
    for arm in ARMS:
        per_arm[arm]["t06"] = {q: msd([(res[k].get("t06") or {}).get(q) for k in res if res[k]["arm"] == arm])
                               for q in ("J_all", "wrong_all", "J_far", "r1_J_far", "r1_J_all")}
    out = dict(prereg="docs/stage1-E17-prereg.md §3, §5", C0="geoset_rep_xs seed 0 (E15 scorings)", AB="geoset_rep_xs_ab seed 0 (E15; DEV-A = C0)",
               wide_deg=WIDE, runs=res, reference=ref, per_arm=per_arm, select=dict(chosen=chosen, table=table),
               select_json=json.load(open(SELECT)) if os.path.isfile(SELECT) else None)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'run':12s} {'A JF':>7s} {'dJF/C0':>8s} {'CI':>19s} {'A2 JF':>7s} {'dJF/AB':>8s} {'wide/AB':>8s} {'T06 J':>6s} {'wr':>3s} {'r1 far':>7s}")
    for k, v in res.items():
        A, A2, T = v.get("A"), v.get("A2"), v.get("t06") or {}
        f = lambda x, w=8: (f"{x:+{w}.4f}" if isinstance(x, float) else f"{'-':>{w}s}")
        ci = f"[{A['vs_C0']['ci'][0]:+.4f},{A['vs_C0']['ci'][1]:+.4f}]" if A else "-"
        print(f"{k:12s} {A['JF'] if A else float('nan'):7.4f} {f(A['vs_C0']['dJF'] if A else None)} {ci:>19s} "
              f"{A2['JF'] if A2 else float('nan'):7.4f} {f(A2['vs_AB']['dJF'] if A2 else None)} {f(A2['vs_AB']['wide_dJ'] if A2 else None)} "
              f"{T.get('J_all', float('nan')):6.4f} {str(T.get('wrong_all', '-')):>3s} {T.get('r1_J_far') if T.get('r1_J_far') is not None else float('nan'):7.4f}")
    for k, v in ref.items():
        T = v.get("t06") or {}
        print(f"ref {k:8s} T06 J {T.get('J_all')} wrong {T.get('wrong_all')} r1 far {T.get('r1_J_far')}"
              + (f"  A2 wide/AB {v['A2']['vs_AB']['wide_dJ']:+.4f}" if "A2" in v else ""))
    print("select:", chosen, table)
    print("->", a.out)


if __name__ == "__main__":
    main()

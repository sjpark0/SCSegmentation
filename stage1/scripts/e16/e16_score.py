#!/usr/bin/env python3
"""S1-E16 report (docs/stage1-E16-prereg.md §5): every arm x seed on DEV-A / DEV-A' against C0 (geoset_rep_xs seed 0),
the T06 scores, the derived quantities per seed and their seed mean / sd.  Measurement only, no rule.

    stage1/.venv/bin/python scripts/e16/e16_score.py [--out geomem/runs/_e16/summary.json]

Inputs: geomem/runs/_e16/scores/<arm>_s<seed>_A[_rot<r>].json (dev_score), the E14 / E15 scorings of G seed 0 and AB
seed 0 (geomem/runs/_e15/scores/geoset_rep_xs_A*, AB_60_cstar_A*), geomem/runs/_e16/t06_scores.json (ring_diag score.py),
scripts/e15/runs/angles.json (DEV-A' view angles for the wide pairs).
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

WORK = os.path.join(S1, "geomem", "runs", "_e16")
SCORES = os.path.join(WORK, "scores")
E15_SCORES = os.path.join(S1, "geomem", "runs", "_e15", "scores")
ARMS = ("NG-idx", "NG-feat", "NG-pi3", "G", "G-bias0", "AB", "star")
SEEDS = (0, 1, 2)
WIDE = 55.0
DERIVED = {"bias_effect": ("G", "G-bias0"), "geometry_training": ("G", "NG-pi3"), "neighbour_gate_idx": ("NG-pi3", "NG-idx"),
           "neighbour_gate_feat": ("NG-pi3", "NG-feat"), "finetune_alone": ("NG-idx", "star"), "fixAB_on_G": ("AB", "G")}


def score_path(arm, seed, rot):
    suf = "" if rot == 0 else f"_rot{rot}"
    if arm == "G" and seed == 0:
        return os.path.join(E15_SCORES, f"geoset_rep_xs_A{suf}.json")
    if arm == "AB" and seed == 0:
        if rot == 0:        # E15 Z-R: AB(c*, 60) on DEV-A is byte-identical to C0 (runs/zcheck.json ZR 60) and was not scored
            return os.path.join(E15_SCORES, "geoset_rep_xs_A.json")
        return os.path.join(E15_SCORES, f"AB_60_cstar_A{suf}.json")
    return os.path.join(SCORES, f"{arm}_s{seed}_A{suf}.json")


def rows(path):
    from mvseg2d.dev_compare import load
    return load(path)


def seeds_of(arm):
    return (0,) if arm == "star" else SEEDS


def mean_jf(rs):
    from mvseg2d.paired import scene_means
    m = scene_means(rs, "JF", "view")
    return float(sum(m.values()) / len(m)) if m else float("nan")


def one(arm, seed, base_a, base_a2, ang_a2):
    """-> dict per set: n, JF (scene mean), dJF / dJ / dF + CI vs C0, worst scene, wide dJ (A')."""
    from mvseg2d.dev_compare import compare
    out = {}
    pa = score_path(arm, seed, 0)
    if os.path.isfile(pa):
        ca = rows(pa)
        c = compare(base_a, ca)
        p = c["paired"]
        out["A"] = dict(n=c["n_pairs"], JF=mean_jf(ca), dJF=p["JF"]["delta"], ci=p["JF"]["ci"], dJ=p["J"]["delta"], dF=p["F"]["delta"],
                        worst_scene=min(p["JF"]["per_scene"].values()), per_scene=p["JF"]["per_scene"])
    p1, p2 = score_path(arm, seed, 1), score_path(arm, seed, 2)
    if os.path.isfile(p1) and os.path.isfile(p2):
        ca2 = rows(p1) + rows(p2)
        c = compare(base_a2, ca2)
        p = c["paired"]
        out["A2"] = dict(n=c["n_pairs"], JF=mean_jf(ca2), dJF=p["JF"]["delta"], ci=p["JF"]["ci"], dJ=p["J"]["delta"], dF=p["F"]["delta"],
                         worst_scene=min(p["JF"]["per_scene"].values()), per_scene=p["JF"]["per_scene"],
                         wide_dJ=SEL.wide_subset_delta(base_a2, ca2, ang_a2, WIDE))
    return out


def t06(path):
    if not os.path.isfile(path):
        return {}
    arms = json.load(open(path))["arms"]
    return {k: dict(J_all=v["J_all"], wrong_all=v["wrong_all"], Jfull_all=v.get("Jfull_all"), J_far=v.get("J_far"))
            for k, v in arms.items()}


def msd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return dict(n=len(xs), mean=m, sd=sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK, "summary.json"))
    a = ap.parse_args()
    base_a = rows(os.path.join(E15_SCORES, "geoset_rep_xs_A.json"))
    base_a2 = rows(os.path.join(E15_SCORES, "geoset_rep_xs_A_rot1.json")) + rows(os.path.join(E15_SCORES, "geoset_rep_xs_A_rot2.json"))
    ang = json.load(open(os.path.join(S1, "scripts", "e15", "runs", "angles.json")))
    ang_a2 = SEL.rotated_angles(ang, "A'")
    t = t06(os.path.join(WORK, "t06_scores.json"))
    res = {}
    for arm in ARMS:
        for seed in seeds_of(arm):
            r = one(arm, seed, base_a, base_a2, ang_a2)
            key = f"e16_{arm}_s{seed}" if not (arm == "AB" and seed == 0) else "AB_60_cstar"
            tk = t.get(key) or (t.get("C0") if arm == "G" and seed == 0 else None)
            if r or tk:
                res[f"{arm}_s{seed}"] = dict(arm=arm, seed=seed, **r, t06=tk)
    per_arm = {}
    for arm in ARMS:
        ks = [f"{arm}_s{s}" for s in seeds_of(arm) if f"{arm}_s{s}" in res]
        per_arm[arm] = {st: {q: msd([res[k].get(st, {}).get(q) for k in ks]) for q in ("JF", "dJF", "dJ", "wide_dJ")} for st in ("A", "A2")}
        per_arm[arm]["t06"] = {q: msd([(res[k].get("t06") or {}).get(q) for k in ks]) for q in ("J_all", "wrong_all")}
    derived = {}
    for name, (x, y) in DERIVED.items():
        vals = {}
        for st in ("A", "A2"):
            ds = []
            for s in SEEDS:
                kx, ky = f"{x}_s{s}", f"{y}_s{0 if y == 'star' else s}"
                if kx in res and ky in res and st in res[kx] and st in res[ky]:
                    ds.append(res[kx][st]["JF"] - res[ky][st]["JF"])
            vals[st] = msd(ds)
        derived[name] = vals
    noise = {arm: msd([res[f"{arm}_s{s}"]["A"]["JF"] for s in SEEDS if f"{arm}_s{s}" in res and "A" in res[f"{arm}_s{s}"]])
             for arm in ("G", "AB", "NG-idx", "NG-feat", "NG-pi3")}
    out = dict(prereg="docs/stage1-E16-prereg.md §5", C0="geoset_rep_xs seed 0 (E14 / E15 scorings)", wide_deg=WIDE,
               runs=res, per_arm=per_arm, derived=derived, retrain_noise_sd=noise)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'run':18s} {'DEV-A JF':>9s} {'dJF':>8s} {'CI':>19s} {'A2 JF':>8s} {'dJF':>8s} {'wide dJ':>8s} {'T06 J':>7s} {'wrong':>5s}")
    for k, v in res.items():
        A, A2, T = v.get("A", {}), v.get("A2", {}), v.get("t06") or {}
        f = lambda x, w=8, d=4: (f"{x:+{w}.{d}f}" if isinstance(x, float) else f"{'-':>{w}s}")
        ci = f"[{A['ci'][0]:+.4f},{A['ci'][1]:+.4f}]" if A else "-"
        print(f"{k:18s} {A.get('JF', float('nan')):9.4f} {f(A.get('dJF'))} {ci:>19s} {A2.get('JF', float('nan')):8.4f} {f(A2.get('dJF'))} "
              f"{f(A2.get('wide_dJ'))} {T.get('J_all', float('nan')):7.4f} {str(T.get('wrong_all', '-')):>5s}")
    for n, v in derived.items():
        print(f"derived {n:22s} A {v['A']}  A' {v['A2']}")
    print("retrain noise sd (DEV-A JF):", {k: (None if v is None else v["sd"]) for k, v in noise.items()})
    print("->", a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Baseline harness validation on the DEV sets: J&F per run and the paired difference to C0 (geoset_rep_xs seed 0),
plus the wide-pair dJ on DEV-A' -- the same arithmetic as scripts/e16/e16_score.py.  Reporting only.

    stage1/.venv/bin/python scripts/baselines/summarize.py [--out geomem/runs/_baselines/summary.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(S1, "scripts", "e15"))
sys.path.insert(0, os.path.join(S1, "scripts", "e16"))
import e15_select as SEL  # noqa: E402
from e16_score import E15_SCORES, mean_jf, rows  # noqa: E402

SCORES = os.path.join(S1, "geomem", "runs", "_baselines", "scores")


def main():
    from mvseg2d.dev_compare import compare
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(S1, "geomem", "runs", "_baselines", "summary.json"))
    a = ap.parse_args()
    base_a = rows(os.path.join(E15_SCORES, "geoset_rep_xs_A.json"))
    base_a2 = rows(os.path.join(E15_SCORES, "geoset_rep_xs_A_rot1.json")) + rows(os.path.join(E15_SCORES, "geoset_rep_xs_A_rot2.json"))
    ang = SEL.rotated_angles(json.load(open(os.path.join(S1, "scripts", "e15", "runs", "angles.json"))), "A'")
    names = sorted({f[:-5].replace("_rot1", "").replace("_rot2", "") for f in os.listdir(SCORES) if f.endswith(".json")})
    out = {}
    print(f"{'run':16s} {'DEV-A JF':>9s} {'dJF vs C0':>10s} {'CI':>19s} {'A2 JF':>8s} {'dJF':>8s} {'wide dJ':>8s}")
    for n in names:
        r = {}
        p = os.path.join(SCORES, f"{n}.json")
        if os.path.isfile(p):
            c = compare(base_a, rows(p))["paired"]
            r["A"] = dict(JF=mean_jf(rows(p)), dJF=c["JF"]["delta"], ci=c["JF"]["ci"], dJ=c["J"]["delta"], dF=c["F"]["delta"],
                          per_scene=c["JF"]["per_scene"])
        p1, p2 = (os.path.join(SCORES, f"{n}_rot{k}.json") for k in (1, 2))
        if os.path.isfile(p1) and os.path.isfile(p2):
            ca2 = rows(p1) + rows(p2)
            c = compare(base_a2, ca2)["paired"]
            r["A2"] = dict(JF=mean_jf(ca2), dJF=c["JF"]["delta"], ci=c["JF"]["ci"], dJ=c["J"]["delta"], dF=c["F"]["delta"],
                           per_scene=c["JF"]["per_scene"], wide_dJ=SEL.wide_subset_delta(base_a2, ca2, ang, 55.0))
        out[n] = r
        A, A2 = r.get("A", {}), r.get("A2", {})
        f = lambda x: f"{x:+8.4f}" if isinstance(x, float) else f"{'-':>8s}"
        ci = f"[{A['ci'][0]:+.4f},{A['ci'][1]:+.4f}]" if A else "-"
        print(f"{n:16s} {A.get('JF', float('nan')):9.4f} {f(A.get('dJF')):>10s} {ci:>19s} {A2.get('JF', float('nan')):8.4f} {f(A2.get('dJF'))} {f(A2.get('wide_dJ'))}")
    out["C0"] = dict(A=dict(JF=mean_jf(base_a)), A2=dict(JF=mean_jf(base_a2)))
    print(f"{'C0 geoset_rep_xs':16s} {out['C0']['A']['JF']:9.4f} {'':>10s} {'':>19s} {out['C0']['A2']['JF']:8.4f}")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()

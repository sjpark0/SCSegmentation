#!/usr/bin/env python3
"""S1-E14 threshold-upsample report (host): the PREREG.md grid, picks, bootstrap CIs, risk and LOSO selection check.

    stage1/.venv/bin/python tu_report.py --parts <dir>/parts --out <dir>/summary.json > <dir>/report.txt
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from mvseg2d.paired import paired_scene_bootstrap  # noqa: E402

PAIRS = os.path.join(S1, "geomem/runs/_e13_dev/dev_pairs.json")
TAUS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
JBU = [f"jbu_s{ss:g}_r{'inf' if math.isinf(sr) else f'{sr:g}'}" for ss in (0.5, 1.0) for sr in (4.0, 8.0, 16.0, 32.0, math.inf)]
GF = [f"gf_r{r}_e{eps:g}" for r in (4, 8) for eps in (1e-4, 1e-3, 1e-2)]
BASE = "bil2|0"
ELIG_B = ["bic1"] + JBU + GF                     # (b), (c) eligible upsamplers (bil1 = control)


def vname(u, tau):
    return f"{u}|{tau:g}"


def load(parts):
    rows = []                                    # one per pair: key + per-variant J, F, tp, pp
    var = None
    for p in sorted(glob.glob(os.path.join(parts, "*.json"))):
        d = json.load(open(p))
        names = sorted(d["res"])
        var = names if var is None else var
        assert names == var, (p, set(names) ^ set(var))
        for i, o in enumerate(d["objs"]):
            r = dict(scene=d["scene"], t=d["t"], cam=d["cam"], obj=o, view=d["view"], gt_px=d["gt_px"][i])
            for n in var:
                v = d["res"][n]
                r[n] = (v["J"][i], v["F"][i], v["tp"][i], v["pp"][i])
            rows.append(r)
    rows.sort(key=lambda r: (r["scene"], r["t"], r["cam"], r["obj"]))
    return rows, var


def hmean(keys, scene, view, x):
    """pair -> view -> scene -> mean over scenes; returns (overall, {scene: mean})."""
    by = defaultdict(lambda: defaultdict(list))
    for s, v, xi in zip(scene, view, x):
        by[s][v].append(xi)
    sm = {s: float(np.mean([np.mean(a) for a in vv.values()])) for s, vv in by.items()}
    return float(np.mean(list(sm.values()))), sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=10_000)
    ap.add_argument("--partial", action="store_true", help="test on an incomplete parts dir (no 3 300 assertion)")
    a = ap.parse_args()
    rows, var = load(a.parts)
    ref = {(r["scene"], int(r["t"]), r["cam"], int(r["obj"])): r for r in json.load(open(PAIRS))["rows"]}
    assert a.partial or len(rows) == len(ref) == 3300, (len(rows), len(ref))
    n = len(rows)
    scene = [r["scene"] for r in rows]
    view = [r["view"] for r in rows]
    scenes = sorted(set(scene))
    gtpx = np.array([r["gt_px"] for r in rows], float)
    M = {}
    for v in var:
        a_ = np.array([r[v] for r in rows], float)
        J, F, tp, pp = a_.T
        M[v] = dict(J=J, F=F, JF=(J + F) / 2, rec=tp / gtpx, prec=np.where(pp > 0, tp / np.maximum(pp, 1), np.nan))
    jf_ref = np.array([ref[(r["scene"], int(r["t"]), r["cam"], int(r["obj"]))]["JF"] for r in rows])
    chk = float(np.abs(M[BASE]["JF"] - jf_ref).max())
    good = jf_ref >= 0.8
    fail = ~good
    base = M[BASE]["JF"]
    hb, hb_s = hmean(None, scene, view, base)

    def stats(v):
        x = M[v]["JF"]
        d = x - base
        h, hs = hmean(None, scene, view, x)
        hg, _ = hmean(None, [s for s, g in zip(scene, good) if g], [w for w, g in zip(view, good) if g], x[good] - base[good])
        hf, _ = hmean(None, [s for s, g in zip(scene, fail) if g], [w for w, g in zip(view, fail) if g], x[fail] - base[fail])
        return dict(hJF=h, dJF=h - hb, dJF_scene={s: hs[s] - hb_s[s] for s in hs}, dJF_pair=float(d.mean()),
                    dJ_pair=float((M[v]["J"] - M[BASE]["J"]).mean()), dF_pair=float((M[v]["F"] - M[BASE]["F"]).mean()),
                    dJF_good_pair=float(d[good].mean()), dJF_fail_pair=float(d[fail].mean()),
                    dJF_good_h=hg, dJF_fail_h=hf,
                    up=int((d > 0.01).sum()), down=int((d < -0.01).sum()),
                    down_good=int((d[good] < -0.01).sum()), down_fail=int((d[fail] < -0.01).sum()),
                    up_good=int((d[good] > 0.01).sum()), up_fail=int((d[fail] > 0.01).sum()),
                    d_1mrec_good=float(-(M[v]["rec"][good] - M[BASE]["rec"][good]).mean()),
                    d_1mprec_good=float(-np.nanmean(M[v]["prec"][good] - M[BASE]["prec"][good])))
    S = {v: stats(v) for v in var}

    def pick(cands, dscore=None):
        """argmax of hierarchical dJF (or of dscore[v]); ties within 1e-4 -> smaller |tau|, then fewer downs."""
        sc = {v: (S[v]["dJF"] if dscore is None else dscore[v]) for v in cands}
        top = max(sc.values())
        tied = [v for v in cands if sc[v] >= top - 1e-4]
        return sorted(tied, key=lambda v: (abs(float(v.split("|")[1])), S[v]["down"], -sc[v]))[0]

    fam = {"a_tau": [vname("bil2", t) for t in TAUS],
           "b_upsample": [vname(u, 0.0) for u in ELIG_B],
           "b_bic1": [vname("bic1", 0.0)],
           "b_jbu": [vname(u, 0.0) for u in JBU],
           "b_gf": [vname(u, 0.0) for u in GF],
           "c_joint": [vname(u, t) for u in ELIG_B for t in TAUS],
           "c_bic1_tau": [vname("bic1", t) for t in TAUS],
           "c_jbu_tau": [vname(u, t) for u in JBU for t in TAUS],
           "c_gf_tau": [vname(u, t) for u in GF for t in TAUS]}
    picks = {f: pick(c) for f, c in fam.items()}

    def boot(v, mask):
        ra = [dict(scene=scene[i], view=view[i], J=M[BASE]["J"][i], F=M[BASE]["F"][i], JF=base[i]) for i in range(n) if mask[i]]
        rb = [dict(scene=scene[i], view=view[i], J=M[v]["J"][i], F=M[v]["F"][i], JF=M[v]["JF"][i]) for i in range(n) if mask[i]]
        out = {}
        for k in ("JF", "J", "F"):
            b = paired_scene_bootstrap(ra, rb, key=k, n=a.nboot, seed=0, view_key="view")
            out[k] = dict(delta=b["delta"], ci=list(b["ci"]))
        return out

    allm = np.ones(n, bool)
    detail = {}
    for f, v in picks.items():
        d = M[v]["JF"] - base
        worst = np.argsort(d)[:5]
        bins = {}
        for lo, hi in ((0, 2e3), (2e3, 1e4), (1e4, 1e5), (1e5, 1e12)):
            m = good & (gtpx >= lo) & (gtpx < hi)
            bins[f"good_{lo:g}-{hi:g}"] = dict(n=int(m.sum()), dJF=float(d[m].mean()), down=int((d[m] < -0.01).sum()),
                                               up=int((d[m] > 0.01).sum()))
        detail[f] = dict(pick=v, stats=S[v], size_bins=bins, boot_all=boot(v, allm), boot_good=boot(v, good), boot_fail=boot(v, fail),
                         worst=[dict(scene=scene[i], view=view[i], obj=rows[i]["obj"], base=float(base[i]), d=float(d[i]),
                                     gt_px=int(gtpx[i])) for i in worst])
    for ctl in (vname("bil1", 0.0),):
        detail["control_" + ctl] = dict(pick=ctl, stats=S[ctl], boot_all=boot(ctl, allm), boot_good=boot(ctl, good),
                                        boot_fail=boot(ctl, fail))

    # leave-one-scene-out selection: pick on 16 scenes (mean of their scene deltas), score the held-out scene
    loso = {}
    for f, c in fam.items():
        if len(c) == 1 or len(scenes) < 2:
            continue
        per, chosen = {}, {}
        for s in scenes:
            sc = {v: float(np.mean([S[v]["dJF_scene"][q] for q in scenes if q != s])) for v in c}
            v = pick(c, sc)
            chosen[s] = v
            per[s] = S[v]["dJF_scene"][s]
        loso[f] = dict(dJF=float(np.mean(list(per.values()))), per_scene=per, chosen=chosen,
                       n_distinct=len(set(chosen.values())))

    out = dict(meta=dict(parts=os.path.abspath(a.parts), n_pairs=n, n_good=int(good.sum()), n_fail=int(fail.sum()),
                         base=BASE, base_hJF=hb, base_pairJF=float(base.mean()), check_max_abs_dJF_vs_dev_pairs=chk,
                         nboot=a.nboot, taus=TAUS),
               grid=S, picks=detail, loso=loso)
    json.dump(out, open(a.out, "w"), indent=1)

    # ------------------------------------------------------------------ text report
    P = print
    P(f"pairs {n} (good {good.sum()}, failing {fail.sum()}); base {BASE}: hier J&F {hb:.5f}, pair-mean {base.mean():.5f}; "
      f"max |JF - dev_pairs| = {chk:.2e}")
    P(f"base good-seed 1-recall {1 - M[BASE]['rec'][good].mean():.4f}, 1-precision {1 - np.nanmean(M[BASE]['prec'][good]):.4f}")
    P("\nGRID (hier dJF / pair dJF / good pair dJF / fail pair dJF / up / down(good,fail))")
    ups = ["bil2", "bil1"] + ELIG_B
    for u in ups:
        P(f"\n{u}")
        for t in TAUS:
            v = vname(u, t)
            s = S[v]
            P(f"  tau {t:+5.2f}: {s['dJF']:+.5f}  {s['dJF_pair']:+.5f}  good {s['dJF_good_pair']:+.5f}  fail {s['dJF_fail_pair']:+.5f}  "
              f"up {s['up']:4d} down {s['down']:4d} ({s['down_good']},{s['down_fail']})  d1-rec {s['d_1mrec_good']:+.4f} d1-prec {s['d_1mprec_good']:+.4f}")
    P("\nPICKS (scene-cluster paired bootstrap 95% CI)")
    for f, d in detail.items():
        s = d["stats"]
        ba, bg, bf = d["boot_all"], d["boot_good"], d["boot_fail"]
        P(f"\n{f}: {d['pick']}")
        P(f"  all : dJF {ba['JF']['delta']:+.5f} [{ba['JF']['ci'][0]:+.5f}, {ba['JF']['ci'][1]:+.5f}]  dJ {ba['J']['delta']:+.5f} [{ba['J']['ci'][0]:+.5f}, {ba['J']['ci'][1]:+.5f}]  dF {ba['F']['delta']:+.5f} [{ba['F']['ci'][0]:+.5f}, {ba['F']['ci'][1]:+.5f}]  pair-mean {s['dJF_pair']:+.5f}")
        P(f"  good: dJF {bg['JF']['delta']:+.5f} [{bg['JF']['ci'][0]:+.5f}, {bg['JF']['ci'][1]:+.5f}]  pair-mean {s['dJF_good_pair']:+.5f}")
        P(f"  fail: dJF {bf['JF']['delta']:+.5f} [{bf['JF']['ci'][0]:+.5f}, {bf['JF']['ci'][1]:+.5f}]  pair-mean {s['dJF_fail_pair']:+.5f}")
        P(f"  risk: down>0.01 {s['down']} (good {s['down_good']}, fail {s['down_fail']}); up>0.01 {s['up']} (good {s['up_good']}, fail {s['up_fail']})")
        P(f"  good-seed d(1-recall) {s['d_1mrec_good']:+.4f}  d(1-precision) {s['d_1mprec_good']:+.4f}")
        if "size_bins" in d:
            P("  good by gt_px: " + "; ".join(f"{k[5:]} n{b['n']} {b['dJF']:+.5f} (up {b['up']}, down {b['down']})" for k, b in d["size_bins"].items()))
        if "worst" in d:
            P("  worst: " + "; ".join(f"{w['scene']} {w['view']} o{w['obj']} {w['base']:.3f}{w['d']:+.3f}" for w in d["worst"]))
    P("\nLOSO selection (pick on 16 scenes, score held-out; mean over 17)")
    for f, l in loso.items():
        P(f"  {f:12s} dJF {l['dJF']:+.5f}  distinct picks {l['n_distinct']}: {sorted(set(l['chosen'].values()))}")


if __name__ == "__main__":
    main()

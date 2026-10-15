#!/usr/bin/env python3
"""S1-E13 candidate E2 report (host numpy; DEV only, the keyframe is never read).

    stage1/.venv/bin/python report_2ref.py --base ../_e13_dev/dev_pairs.json --arm both=both/dev_pairs.json \
        --arm r2=r2/dev_pairs.json --cover cover_B.json

Populations are fixed on the ADOPTED decode (base rows), as in dev_report.py:
    UC     = JF < 0.8 and gt_invis_frac >= 0.5 and recall < 0.85        (140 pairs, the task's dev under-cover population)
    CLEAN  = UC and precision >= 0.85                                    (67 pairs)
    REST   = all pairs not in UC
Paired deltas arm - base on identical (scene, t, cam, obj) rows; 95 % CIs by cluster bootstrap (10 000, seed 0) over scenes
and over (scene, cam, obj) objects.  Missed-pixel recovery uses the c_ini classes dev_score.py writes for both decodes
(the classes depend only on the frame's Pi3 and c_ini, not on the masks).
"""
import argparse
import json
from collections import defaultdict

import numpy as np

CLS = ("vis", "outside", "occluded", "nodepth")
KEY = ("scene", "t", "cam", "obj")
DOG_TREE = ("Dog", "camera_0004", 3)


def key(r):
    return tuple(r[k] for k in KEY)


def frac(a, b):
    return a / b if b else float("nan")


def boot(deltas, clusters, n=10000, seed=0):
    by = defaultdict(list)
    for d, c in zip(deltas, clusters):
        by[c].append(d)
    cs = sorted(by)
    sums = np.array([np.sum(by[c]) for c in cs])
    cnts = np.array([len(by[c]) for c in cs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cs), size=(n, len(cs)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def prec(r):
    return r["precision"] if r["precision"] is not None else np.nan


def compare(name, sel_keys, base, arm, out):
    b = [base[k] for k in sel_keys]
    a = [arm[k] for k in sel_keys]
    d = np.array([y["JF"] - x["JF"] for x, y in zip(b, a)])
    lo, hi = boot(d, [x["scene"] for x in b])
    lo2, hi2 = boot(d, [(x["scene"], x["cam"], x["obj"]) for x in b])
    Mb = sum(x["missed_px"] for x in b)
    Ma = sum(y["missed_px"] for y in a)
    Gb = sum(x["gt_px"] for x in b)
    FPb = sum(x["seed_px"] - (x["gt_px"] - x["missed_px"]) for x in b)
    FPa = sum(y["seed_px"] - (y["gt_px"] - y["missed_px"]) for y in a)
    res = dict(n=len(b), JF_base=float(np.mean([x["JF"] for x in b])), JF_arm=float(np.mean([y["JF"] for y in a])),
               dJF=float(d.mean()), dJF_ci_scene=(lo, hi), dJF_ci_object=(lo2, hi2),
               n_up=int((d > 0.005).sum()), n_down=int((d < -0.005).sum()),
               recall_base=float(np.mean([x["recall"] for x in b])), recall_arm=float(np.mean([y["recall"] for y in a])),
               precision_base=float(np.nanmean([prec(x) for x in b])), precision_arm=float(np.nanmean([prec(y) for y in a])),
               missed_base=Mb, missed_arm=Ma, gt_px=Gb, missed_reduction=frac(Mb - Ma, Mb), fp_base=FPb, fp_arm=FPa,
               fail_base=int(sum(x["JF"] < 0.8 for x in b)), fail_arm=int(sum(y["JF"] < 0.8 for y in a)))
    for c in CLS:
        mb, ma = sum(x[f"missed_{c}"] for x in b), sum(y[f"missed_{c}"] for y in a)
        pm = [frac(x[f"missed_{c}"] - y[f"missed_{c}"], x[f"missed_{c}"]) for x, y in zip(b, a) if x[f"missed_{c}"] > 0]
        res[f"recovered_{c}"] = dict(base=mb, arm=ma, pixel_weighted=frac(mb - ma, mb), pair_mean=float(np.mean(pm)) if pm else None)
    out[name] = res
    print(f"  {name:44s} n={len(b):4d}  J&F {res['JF_base']:.4f} -> {res['JF_arm']:.4f}  d {res['dJF']:+.4f} "
          f"[scene {lo:+.4f}, {hi:+.4f}] [object {lo2:+.4f}, {hi2:+.4f}]  up/down {res['n_up']}/{res['n_down']}  "
          f"fail(<0.8) {res['fail_base']}->{res['fail_arm']}")
    print(f"  {'':44s}        recall {res['recall_base']:.3f} -> {res['recall_arm']:.3f}  precision {res['precision_base']:.3f} -> "
          f"{res['precision_arm']:.3f}  missed px {Mb:,} -> {Ma:,} (net recovered {res['missed_reduction']:.3f})  "
          f"false-pos px {FPb:,} -> {FPa:,}")
    print(f"  {'':44s}        net recovered by c_ini class (pixel-weighted / pair-mean): " + "  ".join(
        f"{c} {res[f'recovered_{c}']['pixel_weighted']:.3f}/{(res[f'recovered_{c}']['pair_mean'] if res[f'recovered_{c}']['pair_mean'] is not None else float('nan')):.3f}"
        for c in CLS))
    return res


def cover(name, sel_keys, cov, out):
    rs = [cov[k] for k in sel_keys]
    M = sum(r["missed_px"] for r in rs)
    res = dict(n=len(rs), missed_px=M)
    line = []
    for q in ("visB", "onBgt", "onBgt_or_refgt"):
        pw = frac(sum(r[f"missed_{q}"] for r in rs), M)
        pm = float(np.mean([frac(r[f"missed_{q}"], r["missed_px"]) for r in rs if r["missed_px"] > 0]))
        res[q] = dict(pixel_weighted=pw, pair_mean=pm)
        line.append(f"{q} {pw:.3f}/{pm:.3f}")
    for c in CLS:
        mc = sum(r[f"missed_{c}"] for r in rs)
        res[f"{c}_share"] = frac(mc, M)
        res[f"{c}_visB"] = frac(sum(r[f"missed_{c}_visB"] for r in rs), mc)
        res[f"{c}_onBgt"] = frac(sum(r[f"missed_{c}_onBgt"] for r in rs), mc)
    res["B_absent_pairs"] = int(sum(r["B_gt_px"] == 0 for r in rs))
    out[name] = res
    print(f"  {name:44s} n={len(rs):4d} missed px {M:,}; (pixel-weighted/pair-mean) " + "  ".join(line)
          + f"; object absent from B's GT in {res['B_absent_pairs']} pairs")
    print(f"  {'':44s}  per c_ini class (share of missed | visible from B | on B GT): " + "  ".join(
        f"{c} {res[f'{c}_share']:.3f}|{res[f'{c}_visB']:.3f}|{res[f'{c}_onBgt']:.3f}" for c in CLS))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", default=[], help="name=path of a dev_score.py output")
    ap.add_argument("--cover", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    base_rows = json.load(open(a.base))["rows"]
    base = {key(r): r for r in base_rows}
    ALL = sorted(base)
    UC = [k for k in ALL if base[k]["JF"] < 0.8 and base[k]["gt_invis_frac"] >= 0.5 and base[k]["recall"] < 0.85]
    CLEAN = [k for k in UC if base[k]["precision"] is not None and base[k]["precision"] >= 0.85]
    UCx = [k for k in UC if (k[0], k[2], k[3]) != DOG_TREE]
    REST = sorted(set(ALL) - set(UC))
    FAIL = [k for k in ALL if base[k]["JF"] < 0.8]
    pops = [("ALL dev pairs", ALL), ("UC dev under-cover (140)", UC), ("UC without Dog camera_0004 tree", UCx),
            ("CLEAN UC & precision>=0.85", CLEAN), ("REST (not UC)", REST), ("FAIL base J&F<0.8", FAIL)]
    print(f"populations (fixed on the adopted decode): " + ", ".join(f"{n} {len(k)}" for n, k in pops))
    out = {"populations": {n: len(k) for n, k in pops}}
    if a.cover:
        cov = {key(r): r for r in json.load(open(a.cover))["rows"]}
        assert set(cov) == set(base), "cover rows must be the base rows"
        print("\n(A) GEOMETRIC ORACLE: adopted decode's missed pixels reachable from B (Pi3 reprojection X -> B, occlusion-checked)")
        out["cover"] = {}
        for n, ks in pops:
            cover(n, ks, cov, out["cover"])
        arms_cov = sorted({k[4:-5] for k in next(iter(cov.values())) if k.startswith("rec_") and k.endswith("_visB")})
        for an in arms_cov:
            print(f"  GROSS recovery by arm '{an}': adopted missed px now covered, split by whether B's GT reaches them")
            out["cover"][f"rec_{an}"] = {}
            for n, ks in pops:
                rs = [cov[k] for k in ks]
                M = sum(r["missed_px"] for r in rs)
                On = sum(r["missed_onBgt"] for r in rs)
                R_ = sum(r[f"rec_{an}"] for r in rs)
                Ron = sum(r[f"rec_{an}_onBgt"] for r in rs)
                d = dict(gross=frac(R_, M), in_onBgt=frac(Ron, On), in_not_onBgt=frac(R_ - Ron, M - On),
                         share_of_rec_on_Bgt=frac(Ron, R_))
                d.update({f"gross_{c}": frac(sum(r[f"rec_{an}_{c}"] for r in rs), sum(r[f"missed_{c}"] for r in rs)) for c in CLS})
                out["cover"][f"rec_{an}"][n] = d
                print(f"    {n:44s} gross {d['gross']:.3f} | of missed on B GT {d['in_onBgt']:.3f} | of missed NOT on B GT "
                      f"{d['in_not_onBgt']:.3f} | share of recovered on B GT {d['share_of_rec_on_Bgt']:.3f} | by c_ini class " +
                      " ".join(f"{c} {d[f'gross_{c}']:.3f}" for c in CLS))
        print("  (context) target GT reachable: " + ", ".join(
            f"{n}: visB {frac(sum(cov[k]['gt_visB'] for k in ks), sum(cov[k]['gt_px'] for k in ks)):.3f} "
            f"onBgt {frac(sum(cov[k]['gt_onBgt'] for k in ks), sum(cov[k]['gt_px'] for k in ks)):.3f}" for n, ks in pops[:2]))
    for spec in a.arm:
        name, path = spec.split("=", 1)
        arm = {key(r): r for r in json.load(open(path))["rows"]}
        assert set(arm) == set(base), (name, len(arm), len(base))
        print(f"\n(B) DECODE arm '{name}' vs adopted geoset_rep_x (paired, same rows)")
        out[name] = {}
        for n, ks in pops:
            compare(n, ks, base, arm, out[name])
        # per object group of UC, largest missed first
        grp = defaultdict(list)
        for k in UC:
            grp[(k[0], k[2], k[3])].append(k)
        print("  UC by (scene, cam, obj): frames  J&F base->arm  recall base->arm  missed px base->arm  oracle onBgt")
        for g, ks in sorted(grp.items(), key=lambda kv: -sum(base[k]["missed_px"] for k in kv[1])):
            Mb, Ma = sum(base[k]["missed_px"] for k in ks), sum(arm[k]["missed_px"] for k in ks)
            ob = frac(sum(cov[k]["missed_onBgt"] for k in ks), Mb) if a.cover else float("nan")
            print(f"    {g[0]:18s} {g[1]:12s} {g[2]:3d} {base[ks[0]]['label'] or '-':24s} {len(ks)}  "
                  f"{np.mean([base[k]['JF'] for k in ks]):.3f}->{np.mean([arm[k]['JF'] for k in ks]):.3f}  "
                  f"{np.mean([base[k]['recall'] for k in ks]):.2f}->{np.mean([arm[k]['recall'] for k in ks]):.2f}  "
                  f"{Mb:9,d}->{Ma:9,d}  {ob:.2f}")
        bys = defaultdict(list)
        for k in ALL:
            bys[k[0]].append(arm[k]["JF"] - base[k]["JF"])
        print("  per scene dJ&F (all pairs): " + ", ".join(f"{s} {np.mean(v):+.4f}" for s, v in sorted(bys.items())))
        out[name]["per_scene_dJF"] = {s: float(np.mean(v)) for s, v in bys.items()}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S1-E14 partition-view report (host): reads bd_partition.py output; good seed = J&F >= 0.8 of --good-from (default the
same file).  Scene-cluster bootstrap (17 scenes with replacement, 10 000 reps, seed 0) for the fill deltas.
    stage1/.venv/bin/python bd_part_report.py --part <part.json> [--good-from <part.json>] [--summary <json>]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

SIZE_BINS = ((0, 2000, "<2k"), (2000, 10000, "2k-10k"), (10000, 100000, "10k-100k"), (100000, 10 ** 12, ">=100k"))
KS = ("1", "2", "4", "8", "inf")


def key(r):
    return (r["scene"], r["t"], r["cam"], r["obj"])


def boot(rows, fn, reps=10000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    sc = sorted(by)
    rng = np.random.default_rng(seed)
    v = [fn([r for i in rng.integers(0, len(sc), len(sc)) for r in by[sc[i]]]) for _ in range(reps)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def S(rows, k):
    return float(sum(r[k] for r in rows))


def decomp(rows, label):
    fn_u = S(rows, "fn_unc_in") + S(rows, "fn_unc_out")
    fn_n = S(rows, "fn_nb_in") + S(rows, "fn_nb_out")
    fp_s = S(rows, "fp_seeded_in") + S(rows, "fp_seeded_out")
    fp_u = S(rows, "fp_unseeded_in") + S(rows, "fp_unseeded_out")
    tot = fn_u + fn_n + fp_s + fp_u
    d = dict(n=len(rows), err_px=tot, fn_unclaimed=fn_u / tot, fn_neighbour=fn_n / tot, fp_on_seeded=fp_s / tot,
             fp_on_unseeded=fp_u / tot, fp_overlap=(S(rows, "fp_seeded_both_in") + S(rows, "fp_seeded_both_out")) / tot)
    bi = sum(S(rows, k + "_in") for k in ("fn_unc", "fn_nb", "fp_seeded", "fp_unseeded"))
    d.update({f"in_{k}": S(rows, k + "_in") / bi for k in ("fn_unc", "fn_nb", "fp_seeded", "fp_unseeded")})
    bo = tot - bi
    d.update({f"out_{k}": S(rows, k + "_out") / bo for k in ("fn_unc", "fn_nb", "fp_seeded", "fp_unseeded")})
    d["in_band_share"] = bi / tot
    if len({r["scene"] for r in rows}) > 1:
        def _fs(q):
            a_ = sum(r["fn_unc_in"] + r["fn_unc_out"] for r in q)
            b_ = sum(r[k + "_in"] + r[k + "_out"] for r in q for k in ("fn_unc", "fn_nb", "fp_seeded", "fp_unseeded"))
            return a_ / b_
        d["fn_unclaimed_ci"] = boot(rows, _fs, 2000)
    pm = {k: [] for k in ("fn_unc", "fn_nb", "fp_seeded", "fp_unseeded")}
    for r in rows:
        e = sum(r[k + "_in"] + r[k + "_out"] for k in pm)
        if e:
            for k in pm:
                pm[k].append((r[k + "_in"] + r[k + "_out"]) / e)
    d.update({f"pairmean_{k}": float(np.mean(v)) for k, v in pm.items()})
    print(f"  {label:14s} n {len(rows):5d}  error px {int(tot):>11,}:  FN unclaimed {d['fn_unclaimed']:.3f}  FN claimed-by-neighbour {d['fn_neighbour']:.3f}"
          f"  FP on seeded-GT {d['fp_on_seeded']:.3f} (overlap {d['fp_overlap']:.4f})  FP on unseeded-GT {d['fp_on_unseeded']:.3f}"
          + (f"  [FN-unclaimed share CI {d['fn_unclaimed_ci'][0]:.3f}, {d['fn_unclaimed_ci'][1]:.3f}]" if "fn_unclaimed_ci" in d else ""))
    print(f"  {'':14s} inside +-2 ({bi / tot:.3f} of error px): FN unc {d['in_fn_unc']:.3f}  FN nb {d['in_fn_nb']:.3f}  FP seeded {d['in_fp_seeded']:.3f}  FP unseeded {d['in_fp_unseeded']:.3f}"
          f"   | outside: FN unc {d['out_fn_unc']:.3f}  FN nb {d['out_fn_nb']:.3f}  FP seeded {d['out_fp_seeded']:.3f}  FP unseeded {d['out_fp_unseeded']:.3f}")
    print(f"  {'':14s} pair-mean shares: FN unc {d['pairmean_fn_unc']:.3f}  FN nb {d['pairmean_fn_nb']:.3f}  FP seeded {d['pairmean_fp_seeded']:.3f}  FP unseeded {d['pairmean_fp_unseeded']:.3f}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", required=True)
    ap.add_argument("--good-from", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    P = json.load(open(a.part))
    rows, views = P["rows"], P["views"]
    ref = json.load(open(a.good_from))["rows"] if a.good_from else rows
    gk = {key(r) for r in ref if r["JF"] >= 0.8}
    good = [r for r in rows if key(r) in gk]
    bad = [r for r in rows if key(r) not in gk]
    out = dict(meta=dict(part=a.part, good_from=a.good_from))
    print(f"masks {P['meta']['masks']}: {len(views)} views, {len(rows)} pairs, good {len(good)}")
    V = lambda k: float(sum(v[k] for v in views))  # noqa: E731
    px, un = V("px"), V("unclaimed")
    print("\n(P1) predicted partition per view (pooled over the 170 DEV-A views)")
    print(f"  unclaimed px {int(un):,} = {un / px:.4f} of all px;  of them GT seeded {V('unclaimed_gt_seeded') / un:.3f}, GT unseeded-labelled {V('unclaimed_gt_unseeded') / un:.3f},"
          f" GT label 0 {V('unclaimed_gt_0') / un:.5f};  GT unseeded px overall {V('gt_unseeded_px') / px:.4f} of all px;  GT label-0 px overall {int(V('gt_label0_px')):,} = {V('gt_label0_px') / px:.6f} of all px")
    us = V("unclaimed_gt_seeded")
    print(f"  unclaimed px whose GT owner is seeded: {int(us):,} = {us / px:.4f} of all px; by gap width (distance to the nearest claimed px):")
    gaps = (("0", "1"), ("1", "2"), ("2", "4"), ("4", "8"), ("8", "inf"))
    out["views"] = dict(unclaimed_frac=un / px, unclaimed_gt_seeded=us / un, unclaimed_seeded_frac_of_px=us / px)
    for lo, hi in gaps:
        x = V(f"unc_gap{lo}_{hi}_seeded")
        y = V(f"unc_gap{lo}_{hi}")
        print(f"    ({lo},{hi}] px: {int(x):>11,} = {x / us:.3f} of the seeded-owner unclaimed px;  share of ALL unclaimed px in this width that is seeded-owner {x / y:.3f}")
        out["views"][f"gap{lo}_{hi}"] = dict(seeded=x, all=y)
    print(f"  single-claimed px with the WRONG seeded owner: {int(V('wrong_single_seeded')):,} = {V('wrong_single_seeded') / px:.4f} of px;"
          f"  single-claimed px whose GT is unseeded: {int(V('wrong_single_unseeded')):,};  multiply-claimed px {int(V('multi')):,}")
    print("\n(P2) error pixels of the pairs, by what is on the other side")
    out["decomp"] = {nm: decomp(rr, nm) for nm, rr in (("all DEV-A", rows), ("good seeds", good), ("failing", bad))}
    print("  good seeds by GT size:")
    out["decomp_size"] = {}
    for lo, hi, nm in SIZE_BINS:
        out["decomp_size"][nm] = decomp([r for r in good if lo <= r["gt_px"] < hi], nm)
    print("  good seeds by object type:")
    import os
    types = {}
    for sc in {r["scene"] for r in good}:
        pth = os.path.join("/home/sjpark/Documents/SCSegmentation/stage1/Data", sc, "objects_labels.json")
        types[sc] = {int(k): v.get("type") for k, v in json.load(open(pth)).items()} if os.path.isfile(pth) else {}
    out["decomp_type"] = {}
    for ty in ("dynamic", "static", "env"):
        out["decomp_type"][ty] = decomp([r for r in good if types[r["scene"]].get(r["obj"]) == ty], ty)
    fu = S(good, "fn_unc_in") + S(good, "fn_unc_out")
    print("  good seeds' unclaimed FN by gap width: " + "  ".join(f"({lo},{hi}] {S(good, f'fn_unc_gap{lo}_{hi}') / fu:.3f}" for lo, hi in gaps))
    print("\n(P3) GT-free nearest-claimed-mask fill of unclaimed px within k px (DEV-A); delta J&F vs the masks as decoded")
    out["fill"] = {}
    for nm, rr in (("all DEV-A", rows), ("good seeds", good), ("failing", bad)):
        base = np.mean([r["JF"] for r in rr])
        print(f"  {nm:12s} n {len(rr):5d} base J&F {base:.4f}")
        out["fill"][nm] = dict(base=base)
        for k in KS:
            dlt = np.mean([r[f"fill{k}"] - r["JF"] for r in rr])
            ci = boot(rr, lambda q: np.mean([r[f"fill{k}"] - r["JF"] for r in q]), a.reps)
            up = sum(r[f"fill{k}"] > r["JF"] + 0.01 for r in rr)
            dn = sum(r[f"fill{k}"] < r["JF"] - 0.01 for r in rr)
            add, ok = S(rr, f"fill{k}_add"), S(rr, f"fill{k}_add_ok")
            print(f"    fill k={k:>3s}: dJ&F {dlt:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  up/down (+-0.01) {up}/{dn}  px added {int(add):,} of which correct {ok / add if add else float('nan'):.3f}")
            out["fill"][nm][k] = dict(delta=dlt, ci=ci, up=up, down=dn, added=add, correct=ok / add if add else None)
        for k in ("2", "inf"):
            dlt = np.mean([r[f"ofill{k}"] - r["JF"] for r in rr])
            ci = boot(rr, lambda q: np.mean([r[f"ofill{k}"] - r["JF"] for r in q]), a.reps)
            print(f"    ORACLE fill (unclaimed px of the pair's own GT) k={k:>3s}: dJ&F {dlt:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
            out["fill"][nm][f"oracle{k}"] = dlt
    for k in KS:
        x, c, u = V(f"fill{k}_px"), V(f"fill{k}_correct_px"), V(f"fill{k}_to_unscored_px")
        print(f"  view-pooled fill k={k:>3s}: px given {int(x):,}, to the GT owner {c / x:.3f}, to masks of objects with no scored pair (GT empty here) {int(u):,}")
    if a.summary:
        json.dump(out, open(a.summary, "w"), indent=1)


if __name__ == "__main__":
    main()

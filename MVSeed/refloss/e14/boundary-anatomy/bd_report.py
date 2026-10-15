#!/usr/bin/env python3
"""S1-E14 boundary anatomy report (host): reads bd_anatomy.py output (+ bd_grid.py output when present) and prints the
DEV-A tables; writes a summary json.  Good seed = the ADOPTED preset's DEV-A pair J&F >= 0.8 (fixed by --good-from).

    stage1/.venv/bin/python bd_report.py --anat <anat.json> [--grid <grid.json>] [--good-from <anat.json>] [--summary <json>]
Pair means unless marked "pooled" (pixel sums over pairs).  CIs: scene-cluster bootstrap (17 scenes resampled with
replacement, pairs inside equally weighted, 10 000 reps, seed 0), ratio statistics recomputed per replicate.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

CLS = ("bg", "seeded", "unseeded")
SIZE_BINS = ((0, 2000, "<2k"), (2000, 10000, "2k-10k"), (10000, 100000, "10k-100k"), (100000, 10 ** 12, ">=100k"))
CELL_BINS = ((0, 25, "<25 cells"), (25, 100, "25-100"), (100, 1000, "100-1000"), (1000, 10 ** 12, ">=1000"))
DB = (0, 4, 16, 64)


def key(r):
    return (r["scene"], r["t"], r["cam"], r["obj"])


def boot(rows, fn, reps=10000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    scenes = sorted(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(scenes), len(scenes))
        rr = [r for i in pick for r in by[scenes[i]]]
        vals.append(fn(rr))
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def m(rows, k):
    return float(np.mean([r[k] for r in rows])) if rows else float("nan")


def s(rows, k):
    return float(np.sum([r[k] for r in rows]))


def f(x, n=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def stats(rows):
    """Core numbers of a group of pairs."""
    d = dict(n=len(rows))
    if not rows:
        return d
    for k in ("J", "F", "JF"):
        d[k] = m(rows, k)
    d["gap"] = 1 - d["JF"]
    d["1-recall"] = float(np.mean([1 - (r["gt_px"] - r["fn"]) / r["gt_px"] for r in rows]))
    d["1-precision"] = float(np.mean([(r["fp"] / r["seed_px"]) if r["seed_px"] else 1.0 for r in rows]))
    for rad in (1, 2, 4, 8):
        d[f"or{rad}"] = m(rows, f"or{rad}")
        d[f"orJ{rad}"] = m(rows, f"orJ{rad}")
        d[f"orF{rad}"] = m(rows, f"orF{rad}")
        d[f"close{rad}"] = (d[f"or{rad}"] - d["JF"]) / d["gap"] if d["gap"] > 0 else float("nan")
        ein = s(rows, f"fn_in{rad}") + s(rows, f"fp_in{rad}")
        etot = s(rows, "fn") + s(rows, "fp")
        d[f"err_in{rad}_pooled"] = ein / etot if etot else float("nan")
        d[f"fnshare_in{rad}_pooled"] = s(rows, f"fn_in{rad}") / ein if ein else float("nan")
        pm = [r[f"fn_in{rad}"] / (r[f"fn_in{rad}"] + r[f"fp_in{rad}"]) for r in rows if r[f"fn_in{rad}"] + r[f"fp_in{rad}"] > 0]
        d[f"fnshare_in{rad}_pairmean"] = float(np.mean(pm)) if pm else float("nan")
    for rad in (1, 2, 4):
        d[f"gainFN{rad}"] = m(rows, f"orFN{rad}") - d["JF"]
        d[f"gainFP{rad}"] = m(rows, f"orFP{rad}") - d["JF"]
        d[f"gain{rad}"] = d[f"or{rad}"] - d["JF"]
        d[f"gainFN_J{rad}"] = m(rows, f"orFN_J{rad}") - d["J"]
        d[f"gainFP_J{rad}"] = m(rows, f"orFP_J{rad}") - d["J"]
        d[f"gainFN_F{rad}"] = m(rows, f"orFN_F{rad}") - d["F"]
        d[f"gainFP_F{rad}"] = m(rows, f"orFP_F{rad}") - d["F"]
    blen = s(rows, "blen")
    d["disp_abs_pooled_px"] = (s(rows, "fn_in2") + s(rows, "fp_in2")) / blen
    d["disp_net_pooled_px"] = (s(rows, "fp_in2") - s(rows, "fn_in2")) / blen
    d["disp_abs_pairmean_px"] = float(np.mean([(r["fn_in2"] + r["fp_in2"]) / max(1, r["blen"]) for r in rows]))
    d["disp_net_pairmean_px"] = float(np.mean([(r["fp_in2"] - r["fn_in2"]) / max(1, r["blen"]) for r in rows]))
    if all("g288" in r for r in rows):
        for n in (288, 576):
            d[f"g{n}"] = m(rows, f"g{n}")
            d[f"g{n}_J"] = m(rows, f"g{n}_J")
            d[f"g{n}_F"] = m(rows, f"g{n}_F")
            d[f"gridshare{n}"] = (1 - d[f"g{n}"]) / d["gap"] if d["gap"] > 0 else float("nan")
        es = s(rows, "fn_s") + s(rows, "fp_s")
        d["seed_err_on_rt288_err_pooled"] = (s(rows, "fn_both") + s(rows, "fp_both")) / es if es else float("nan")
        eq = s(rows, "fn288") + s(rows, "fp288")
        d["rt288_err_covered_by_seed_err_pooled"] = (s(rows, "fn_both") + s(rows, "fp_both")) / eq if eq else float("nan")
        d["rt288_fnshare_pooled"] = s(rows, "fn288") / eq if eq else float("nan")
    return d


def line(name, d):
    if d["n"] == 0:
        return f"  {name:22s}    0"
    g = f"  g288 {f(d.get('g288'))} g576 {f(d.get('g576'))} gridshare288 {f(d.get('gridshare288'), 3)}" if "g288" in d else ""
    return (f"  {name:22s} {d['n']:5d}  J {f(d['J'])} F {f(d['F'])} J&F {f(d['JF'])} gap {f(d['gap'])} | "
            f"or1 {f(d['or1'])} or2 {f(d['or2'])} or4 {f(d['or4'])} | close2 {f(d['close2'], 3)} | "
            f"gainFN2 {f(d['gainFN2'])} gainFP2 {f(d['gainFP2'])} FNpx-share2 {f(d['fnshare_in2_pooled'], 3)} | "
            f"1-rec {f(d['1-recall'])} 1-prec {f(d['1-precision'])} | disp|.| {f(d['disp_abs_pooled_px'], 3)} net {f(d['disp_net_pooled_px'], 3)} px"
            f" | orJ2 {f(d['orJ2'])} orF2 {f(d['orF2'])}" + g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anat", required=True)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--good-from", default=None, help="anat json of the adopted preset defining the good-seed set")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    A = json.load(open(a.anat))
    rows = A["rows"]
    if a.grid and os.path.isfile(a.grid):
        G = {key(r): r for r in json.load(open(a.grid))["rows"]}
        for r in rows:
            r.update({k: v for k, v in G[key(r)].items() if k not in ("scene", "t", "cam", "obj")})
    goodset = None
    if a.good_from:
        goodset = {key(r) for r in json.load(open(a.good_from))["rows"] if r["JF"] >= 0.8}
    good = [r for r in rows if (key(r) in goodset if goodset is not None else r["JF"] >= 0.8)]
    gk = {key(r) for r in good}
    bad = [r for r in rows if key(r) not in gk]
    out = dict(meta=dict(anat=os.path.abspath(a.anat), grid=a.grid, good_from=a.good_from))
    print(f"masks: {A['meta']['masks']}")
    print(f"pairs {len(rows)}, good seeds (J&F >= 0.8) {len(good)}, failing {len(bad)}")
    print("\n(A) headline")
    for name, rr in (("all DEV-A", rows), ("good seeds", good), ("failing", bad)):
        d = stats(rr)
        out[name] = d
        print(line(name, d))
    d = out["good seeds"]
    print("\n  good seeds, band oracles (J&F / J / F) and closure of the gap:")
    for rad in (1, 2, 4, 8):
        print(f"    +-{rad} px: J&F {f(d[f'or{rad}'])}  J {f(d[f'orJ{rad}'])}  F {f(d[f'orF{rad}'])}  closes {f(d[f'close{rad}'], 3)} of gap {f(d['gap'])};"
              f"  error px inside band {f(d[f'err_in{rad}_pooled'], 3)} (pooled);  FN share of in-band error px pooled {f(d[f'fnshare_in{rad}_pooled'], 3)}"
              f" pair-mean {f(d[f'fnshare_in{rad}_pairmean'], 3)}")
    for rad in (1, 2, 4):
        g_all = d[f"gain{rad}"]
        print(f"    +-{rad} px  fix-under-only (FN) +{f(d[f'gainFN{rad}'])} [J +{f(d[f'gainFN_J{rad}'])} F {f(d[f'gainFN_F{rad}'], 4)}]"
              f"   fix-over-only (FP) +{f(d[f'gainFP{rad}'])} [J +{f(d[f'gainFP_J{rad}'])} F {f(d[f'gainFP_F{rad}'], 4)}]"
              f"   both +{f(g_all)};  FN/(FN+FP) of the one-sided gains {f(d[f'gainFN{rad}'] / (d[f'gainFN{rad}'] + d[f'gainFP{rad}']), 3)}")
    print(f"    mean boundary displacement inside +-2 px band: |.| {f(d['disp_abs_pooled_px'], 3)} px pooled ({f(d['disp_abs_pairmean_px'], 3)} pair-mean),"
          f" net (over - under) {f(d['disp_net_pooled_px'], 3)} px pooled ({f(d['disp_net_pairmean_px'], 3)} pair-mean)")
    if "g288" in d:
        print(f"    GT 288-grid round trip: J&F {f(d['g288'])} (J {f(d['g288_J'])} F {f(d['g288_F'])}), 1 - g288 = {f(1 - d['g288'])} = {f(d['gridshare288'], 3)} of the gap;"
              f"  576: {f(d['g576'])} (1 - g576 = {f(1 - d['g576'])} = {f(d['gridshare576'], 3)} of the gap)")
        print(f"    seed error px that are also 288-round-trip error px (same direction): {f(d['seed_err_on_rt288_err_pooled'], 3)} pooled;"
              f"  round-trip error px also seed error px: {f(d['rt288_err_covered_by_seed_err_pooled'], 3)};  round-trip error FN share {f(d['rt288_fnshare_pooled'], 3)}")
    # error px distribution over radii
    e = {rad: s(good, f"fn_in{rad}") + s(good, f"fp_in{rad}") for rad in (1, 2, 4, 8)}
    et = s(good, "fn") + s(good, "fp")
    fnt, fpt = s(good, "fn"), s(good, "fp")
    print(f"    all good-seed error px {int(et):,} (FN {int(fnt):,} = {f(fnt / et, 3)}, FP {int(fpt):,}); within +-1 {f(e[1] / et, 3)}, +-2 {f(e[2] / et, 3)}, +-4 {f(e[4] / et, 3)}, +-8 {f(e[8] / et, 3)}")
    print(f"    outside +-2: FN {int(fnt - s(good, 'fn_in2')):,}  FP {int(fpt - s(good, 'fp_in2')):,} (FN share {f((fnt - s(good, 'fn_in2')) / (et - e[2]), 3)})")
    # bootstrap CIs (good seeds)
    def r_close2(rr):
        jf = np.mean([r["JF"] for r in rr])
        return (np.mean([r["or2"] for r in rr]) - jf) / (1 - jf)

    def r_fnshare_gain2(rr):
        jf = np.mean([r["JF"] for r in rr])
        a_, b_ = np.mean([r["orFN2"] for r in rr]) - jf, np.mean([r["orFP2"] for r in rr]) - jf
        return a_ / (a_ + b_)

    def r_fnpx2(rr):
        x = sum(r["fn_in2"] for r in rr)
        return x / (x + sum(r["fp_in2"] for r in rr))
    cis = dict(JF=boot(good, lambda rr: np.mean([r["JF"] for r in rr]), a.reps),
               or2=boot(good, lambda rr: np.mean([r["or2"] for r in rr]), a.reps),
               gain2=boot(good, lambda rr: np.mean([r["or2"] - r["JF"] for r in rr]), a.reps),
               close2=boot(good, r_close2, a.reps), fnshare_gain2=boot(good, r_fnshare_gain2, a.reps),
               fnpx_share2=boot(good, r_fnpx2, a.reps),
               gainFN2_minus_gainFP2=boot(good, lambda rr: np.mean([r["orFN2"] - r["orFP2"] for r in rr]), a.reps))
    if "g288" in d:
        cis["g288"] = boot(good, lambda rr: np.mean([r["g288"] for r in rr]), a.reps)
        cis["gridshare288"] = boot(good, lambda rr: (1 - np.mean([r["g288"] for r in rr])) / (1 - np.mean([r["JF"] for r in rr])), a.reps)
    out["good_ci"] = cis
    print("    scene-cluster bootstrap 95 % CIs (good seeds): " + "; ".join(f"{k} [{f(v[0])}, {f(v[1])}]" for k, v in cis.items()))

    print("\n(B) good seeds by GT size (px)")
    out["by_size"] = {}
    for lo, hi, nm in SIZE_BINS:
        rr = [r for r in good if lo <= r["gt_px"] < hi]
        out["by_size"][nm] = st = stats(rr)
        print(line(nm, st))
    print("    by GT size in 288-grid cells (cell = (W/288)(H/288) px)")
    out["by_cells"] = {}
    for lo, hi, nm in CELL_BINS:
        rr = [r for r in good if lo <= r["gt_px"] / ((r["W"] / 288) * (r["H"] / 288)) < hi]
        out["by_cells"][nm] = st = stats(rr)
        print(line(nm, st))
    print("    by image size")
    out["by_image"] = {}
    for hw in sorted({(r["H"], r["W"]) for r in good}):
        rr = [r for r in good if (r["H"], r["W"]) == hw]
        out["by_image"][f"{hw[0]}x{hw[1]}"] = st = stats(rr)
        print(line(f"{hw[0]}x{hw[1]} bp{rr[0]['bp']}", st))

    print("    by object type (Data/<scene>/objects_labels.json type)")
    out["by_type"] = {}
    types = {}
    for sc in {r["scene"] for r in good}:
        pth = os.path.join("/home/sjpark/Documents/SCSegmentation/stage1/Data", sc, "objects_labels.json")
        types[sc] = {int(k): v.get("type") for k, v in json.load(open(pth)).items()} if os.path.isfile(pth) else {}
    for ty in ("dynamic", "static", "env", None):
        rr = [r for r in good if types[r["scene"]].get(r["obj"]) == ty]
        if rr:
            out["by_type"][str(ty)] = st = stats(rr)
            print(line(f"type {ty}", st))

    print("\n(C) frame border")
    out["border"] = {}
    for nm, rr in (("touches frame", [r for r in good if r["touches"]]), ("inside frame", [r for r in good if not r["touches"]])):
        out["border"][nm] = st = stats(rr)
        print(line(nm, st))
    print("    +-2 px band pixels by distance to the image edge d (pooled over good seeds):")
    tot_b, tot_e = s(good, "band2"), s(good, "fn_in2") + s(good, "fp_in2")
    out["border"]["dbins"] = {}
    for i, lo in enumerate(DB):
        nm = f"[{lo},{DB[i + 1] if i + 1 < len(DB) else 'inf'})"
        b, fn_, fp_ = s(good, f"b2_d{lo}"), s(good, f"fn2_d{lo}"), s(good, f"fp2_d{lo}")
        bnd = s(good, f"bnd_d{lo}")
        out["border"]["dbins"][nm] = dict(band_px=b, fn=fn_, fp=fp_, bnd=bnd)
        print(f"      d {nm:9s} band px share {f(b / tot_b, 4)}  inner-boundary px share {f(bnd / sum(s(good, f'bnd_d{x}') for x in DB), 4)}  error px share {f((fn_ + fp_) / tot_e, 4)}"
              f"  error rate (FN+FP)/band {f((fn_ + fp_) / b if b else float('nan'), 4)}  FN share {f(fn_ / (fn_ + fp_) if fn_ + fp_ else float('nan'), 3)}")
    gn, gf = m(good, "or2_near") - d["JF"], m(good, "or2_far") - d["JF"]
    print(f"    +-2 px oracle restricted to the band within 16 px of the frame: +{f(gn)}; beyond 16 px: +{f(gf)}; full band +{f(d['gain2'])}"
          f"  (near share of the restricted gains {f(gn / (gn + gf), 3)})")
    eo = et - e[2]
    print(f"    error px outside the +-2 band within 4 px of the frame: FN {int(s(good, 'fn_out2_edge4')):,} FP {int(s(good, 'fp_out2_edge4')):,}"
          f" = {f((s(good, 'fn_out2_edge4') + s(good, 'fp_out2_edge4')) / eo, 3)} of the out-of-band error px")
    out["border"].update(gain_near=gn, gain_far=gf)

    print("\n(D) neighbour across the GT boundary (0 bg = unlabelled, seeded = another decoded object, unseeded = another labelled object not decoded)")
    out["neighbour"] = {}
    for rho in (0, 3):
        print(f"  rho = {rho} px")
        nb = {}
        bl = sum(s(good, f"bnd_{c}_r{rho}") for c in CLS)
        gains = {c: m(good, f"or2_{c}_r{rho}") - d["JF"] for c in CLS}
        for c in CLS:
            b, fn_, fp_ = s(good, f"b2_{c}_r{rho}"), s(good, f"fn2_{c}_r{rho}"), s(good, f"fp2_{c}_r{rho}")
            bnd = s(good, f"bnd_{c}_r{rho}")
            fpo = s(good, f"fpout2_{c}_r{rho}")
            nb[c] = dict(bnd_share=bnd / bl, band_px=b, err_share=(fn_ + fp_) / tot_e, rate=(fn_ + fp_) / b if b else None,
                         fn_share=fn_ / (fn_ + fp_) if fn_ + fp_ else None, gain2=gains[c], fp_out=fpo,
                         pairs_with=int(sum(1 for r in good if r[f"bnd_{c}_r{rho}"] > 0)))
            print(f"    {c:9s} GT boundary px share {f(bnd / bl, 3)}  band-error px share {f((fn_ + fp_) / tot_e, 3)}  error rate/band px {f(nb[c]['rate'], 4)}"
                  f"  FN share {f(nb[c]['fn_share'], 3)}  restricted +-2 oracle +{f(gains[c])} ({f(gains[c] / d['gain2'], 3)} of full band)"
                  f"  FP px beyond the band on it {int(fpo):,}  pairs with such boundary {nb[c]['pairs_with']}")
        print(f"    sum of restricted gains {f(sum(gains.values()))} vs full band +{f(d['gain2'])}")
        out["neighbour"][f"rho{rho}"] = nb
        print(f"    good seeds by the share of their GT boundary shared with an UNSEEDED labelled object (rho {rho}):")
        bins = ((-1, 0, "0"), (0, 0.1, "(0,0.1]"), (0.1, 1.01, ">0.1"))
        out["neighbour"][f"rho{rho}_byunseeded"] = {}
        for lo, hi, nm in bins:
            rr = [r for r in good if lo < r[f"bnd_unseeded_r{rho}"] / max(1, sum(r[f"bnd_{c}_r{rho}"] for c in CLS)) <= hi]
            out["neighbour"][f"rho{rho}_byunseeded"][nm] = st = stats(rr)
            print(line("unseeded-shared " + nm, st))

    print("\n(E) per scene (good seeds)")
    out["by_scene"] = {}
    for sc in sorted({r["scene"] for r in good}):
        rr = [r for r in good if r["scene"] == sc]
        out["by_scene"][sc] = st = stats(rr)
        print(line(sc, st))
    if a.summary:
        json.dump(out, open(a.summary, "w"), indent=1)


if __name__ == "__main__":
    main()

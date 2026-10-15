#!/usr/bin/env python3
"""S1-E13 candidate D (geometry-grow) report (host).  Reads grow.py run dirs + _e13_dev/dev_pairs.json +
dev_undercover.json.  DEV ONLY.
    stage1/.venv/bin/python -I report.py main [--sens sens_*] [--json summary.json]"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

DEV = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_dev"
DATA = "/home/sjpark/Documents/SCSegmentation/stage1/Data"
F = ["dJ", "dF", "added", "on_o", "on_other", "on_bg", "add_out", "add_occ", "add_nod", "on_out", "on_occ", "on_nod", "on_other_seeded"]
ZERO = [0.0, 0.0] + [0] * 11


def boot(vals, clusters, n=10000, seed=0):
    by = defaultdict(list)
    for v, c in zip(vals, clusters):
        by[c].append(v)
    keys = sorted(by)
    sums = np.array([np.sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fr(a, b):
    return a / b if b else float("nan")


def load_run(d):
    frames = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "frames", "*.json")))]
    views = [v for fr_ in frames for v in fr_["views"]]
    return frames, views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sens", nargs="*", default=[])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    frames, views = load_run(a.run)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    dp = json.load(open(f"{DEV}/dev_pairs.json"))
    uc = json.load(open(f"{DEV}/dev_undercover.json"))["rows"]
    base = {(r["scene"], r["t"], r["cam"], r["obj"]): r for r in dp["rows"]}
    UC = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in uc}
    CLEAN = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in uc if r["precision"] is not None and r["precision"] >= 0.85}
    DOGTREE = {k for k in UC if k[0] == "Dog" and k[2] == "camera_0004" and k[3] == 3}
    types = {}
    for s in {v["scene"] for v in views}:
        lab = json.load(open(os.path.join(DATA, s, "objects_labels.json")))
        types[s] = {int(k): (vv.get("type"), vv.get("label")) for k, vv in lab.items()}
    objdiag = {(f_["scene"], f_["t"], int(o)): d for f_ in frames for o, d in f_["objdiag"].items()}
    # ------------------------------------------------------------------ consistency with the dev scorer
    pairs, funnel, maxd, clsbad = {}, {}, 0.0, 0
    for v in views:
        for o, (J, Fm) in v["base"].items():
            k = (v["scene"], v["t"], v["cam"], int(o))
            b = base[k]
            maxd = max(maxd, abs(J - b["J"]), abs(Fm - b["F"]))
            clsbad += v["gtcls"][o] != [b["gt_vis"], b["gt_outside"], b["gt_occluded"], b["gt_nodepth"]]
            pairs[k] = v["pairs"].get(o, {})
            if o in v["funnel"]:
                funnel[k] = v["funnel"][o]
    print(f"run {a.run} cfg {cfg['cfg']} variants {cfg['variants']}")
    print(f"frames {len(frames)} views {len(views)}; scored pairs {len(pairs)} (dev_pairs.json {len(base)}; equal sets {set(pairs) == set(base)}); "
          f"max |J,F - dev_pairs| = {maxd:.2e}; GT class-count mismatches vs dev_pairs: {clsbad}")
    assert set(pairs) == set(base) and maxd < 1e-12 and clsbad == 0
    secs = [f_["seconds"]["total"] for f_ in frames]
    print(f"cost: {np.sum(secs):.0f} s CPU-process time over {len(frames)} frames (mean {np.mean(secs):.1f} s, max {np.max(secs):.1f} s); "
          f"cams per frame {np.mean([f_['n_cams'] for f_ in frames]):.1f}; linked camera pairs {np.mean([f_['n_pairs_link'] for f_ in frames]):.1f}; "
          f"correspondences per frame {np.mean([f_['n_corr'] for f_ in frames]):,.0f}")
    keys = sorted(pairs)
    UCk = [k for k in keys if k in UC]
    CLk = [k for k in keys if k in CLEAN]
    summ = {}
    print("\ndJF = mean J&F change; CIs: all pairs scene-cluster, under-cover object-cluster bootstrap 95 % (10 000, seed 0)")
    for var in cfg["variants"]:
        rows = {k: dict(zip(F, pairs[k].get(var, ZERO))) for k in keys}
        d = {k: (r["dJ"] + r["dF"]) / 2 for k, r in rows.items()}
        S = {}
        for name, sel, clus in (("all", keys, lambda k: k[0]), ("uc140", UCk, lambda k: (k[0], k[2], k[3])),
                                ("clean67", CLk, lambda k: (k[0], k[2], k[3])),
                                ("not_uc", [k for k in keys if k not in UC], lambda k: k[0])):
            vals = [d[k] for k in sel]
            lo, hi = boot(vals, [clus(k) for k in sel])
            add = sum(rows[k]["added"] for k in sel)
            on = sum(rows[k]["on_o"] for k in sel)
            miss = sum(base[k]["missed_px"] for k in sel)
            S[name] = dict(n=len(sel), dJF=float(np.mean(vals)), ci=[lo, hi], dJ=float(np.mean([rows[k]["dJ"] for k in sel])),
                           dF=float(np.mean([rows[k]["dF"] for k in sel])), base_JF=float(np.mean([base[k]["JF"] for k in sel])),
                           n_changed=sum(rows[k]["added"] > 0 for k in sel), n_up=sum(d[k] > 0 for k in sel), n_down=sum(d[k] < 0 for k in sel),
                           sum_up=float(sum(max(0, d[k]) for k in sel)), sum_down=float(sum(min(0, d[k]) for k in sel)),
                           added=add, on_o=on, prec_pooled=fr(on, add),
                           prec_pair_mean=float(np.mean([rows[k]["on_o"] / rows[k]["added"] for k in sel if rows[k]["added"]]))
                           if any(rows[k]["added"] for k in sel) else float("nan"),
                           on_other=sum(rows[k]["on_other"] for k in sel), on_bg=sum(rows[k]["on_bg"] for k in sel),
                           on_other_seeded=sum(rows[k]["on_other_seeded"] for k in sel), missed=miss, recovered=fr(on, miss),
                           recovered_pair_mean=float(np.mean([rows[k]["on_o"] / base[k]["missed_px"] for k in sel if base[k]["missed_px"]])))
            for c, cn in ((1, "outside"), (2, "occluded"), (3, "nodepth")):
                ad = sum(rows[k][F[5 + c]] for k in sel)
                oc = sum(rows[k][F[8 + c]] for k in sel)
                mc = sum(base[k][f"missed_{cn}"] for k in sel)
                S[name][f"cls_{cn}"] = dict(added=ad, on_o=oc, prec=fr(oc, ad), missed=mc, recovered=fr(oc, mc))
        eg = sum(sum(v["empty_gt"].get(var, {}).values()) for v in views)
        S["empty_gt_added_px"] = eg
        S["all"]["prec_pooled_incl_emptygt"] = fr(S["all"]["on_o"], S["all"]["added"] + eg)
        nd = [k for k in UCk if k not in DOGTREE]
        S["uc140"]["recovered_wo_dogtree"] = fr(sum(rows[k]["on_o"] for k in nd), sum(base[k]["missed_px"] for k in nd))
        S["uc140"]["dJF_wo_dogtree"] = float(np.mean([d[k] for k in nd]))
        S["uc140"]["dogtree_dJF"] = float(np.mean([d[k] for k in UCk if k in DOGTREE]))
        S["uc140"]["dogtree_recovered"] = fr(sum(rows[k]["on_o"] for k in DOGTREE), sum(base[k]["missed_px"] for k in DOGTREE))
        summ[var] = S
        A, U, C, N = S["all"], S["uc140"], S["clean67"], S["not_uc"]
        print(f"\n[{var}]  ALL n={A['n']}: dJF {A['dJF']:+.5f} [{A['ci'][0]:+.5f}, {A['ci'][1]:+.5f}] (dJ {A['dJ']:+.5f} dF {A['dF']:+.5f}); "
              f"changed {A['n_changed']} up {A['n_up']} down {A['n_down']} (sum +{A['sum_up']:.3f} / {A['sum_down']:.3f}); "
              f"added {A['added']:,} px, precision pooled {A['prec_pooled']:.3f} (incl. {eg:,} px on empty-GT objects "
              f"{A['prec_pooled_incl_emptygt']:.3f}), pair-mean {A['prec_pair_mean']:.3f}; on other obj {fr(A['on_other'], A['added']):.3f} "
              f"(seeded at t {fr(A['on_other_seeded'], A['added']):.3f}), on unlabelled {fr(A['on_bg'], A['added']):.3f}")
        print("      by class (added px, precision): " + "; ".join(
            f"{cn} {A[f'cls_{cn}']['added']:,} {A[f'cls_{cn}']['prec']:.3f}" for cn in ("outside", "occluded", "nodepth")))
        print(f"   UC140: base {U['base_JF']:.4f} dJF {U['dJF']:+.5f} [{U['ci'][0]:+.5f}, {U['ci'][1]:+.5f}]  changed {U['n_changed']} up {U['n_up']} "
              f"down {U['n_down']}  added {U['added']:,} prec {U['prec_pooled']:.3f} (pair-mean {U['prec_pair_mean']:.3f}); on other obj "
              f"{fr(U['on_other'], U['added']):.3f}; missed px recovered {U['recovered']:.4f} (pair-mean {U['recovered_pair_mean']:.4f}; "
              f"w/o Dog tree {U['recovered_wo_dogtree']:.4f}; Dog tree {U['dogtree_recovered']:.4f}); dJF w/o Dog tree {U['dJF_wo_dogtree']:+.5f}, "
              f"Dog tree {U['dogtree_dJF']:+.5f}")
        print("      recovered by class: " + "; ".join(
            f"{cn} {U[f'cls_{cn}']['recovered']:.4f} of {U[f'cls_{cn}']['missed']:,} (added {U[f'cls_{cn}']['added']:,} prec {U[f'cls_{cn}']['prec']:.3f})"
            for cn in ("outside", "occluded", "nodepth")))
        print(f"   CLEAN67: base {C['base_JF']:.4f} dJF {C['dJF']:+.5f} [{C['ci'][0]:+.5f}, {C['ci'][1]:+.5f}] up {C['n_up']} down {C['n_down']}  "
              f"added {C['added']:,} prec {C['prec_pooled']:.3f} recovered {C['recovered']:.4f}")
        print(f"   NOT-UC n={N['n']}: dJF {N['dJF']:+.5f} [{N['ci'][0]:+.5f}, {N['ci'][1]:+.5f}] up {N['n_up']} down {N['n_down']} "
              f"added {N['added']:,} prec {N['prec_pooled']:.3f}")
    # ------------------------------------------------------------------ funnel (primary)
    if "mv" in cfg["variants"]:
        print("\nFUNNEL (primary mv): fraction of the missed px (GT_o & ~D_o) surviving each step")
        print("   valid-depth branch: missed & ref-invisible & valid -> appearance-accepted -> traversable (acc & ~cut) -> grown")
        print("   no-depth branch:    missed & no depth -> a splatted point lands -> splat source grown")
        print("   composition:        in G -> & ref-invisible -> & unclaimed by other D -> & 8-connected to D_o -> & uncontested (= recovered)")
        fun = {}
        for name, sel in (("all", keys), ("uc140", UCk), ("clean67", CLk), ("uc140 w/o Dog tree", [k for k in UCk if k not in DOGTREE]),
                          ("Dog tree", [k for k in UCk if k in DOGTREE])):
            tot = Counter()
            for k in sel:
                f = funnel.get(k)
                if f:
                    tot.update(f)
                else:
                    tot["missed"] += base[k]["missed_px"]
            m = tot["missed"]
            fun[name] = {kk: fr(vv, m) for kk, vv in tot.items()}
            print(f"   {name:20s} missed {m:>11,} | inv&valid {fr(tot['missed_invis_valid'], m):.4f} -> acc {fr(tot['acc'], m):.4f} -> trav "
                  f"{fr(tot['trav'], m):.4f} -> grown {fr(tot['grown_valid'], m):.4f} | nodepth {fr(tot['missed_nodepth'], m):.4f} -> splat "
                  f"{fr(tot['nod_splat_src'], m):.4f} -> grown {fr(tot['grown_nod'], m):.4f} | in G {fr(tot['in_G'], m):.4f} invis "
                  f"{fr(tot['invis'], m):.4f} unclaimed {fr(tot['unclaimed'], m):.4f} connected {fr(tot['connected'], m):.4f} "
                  f"uncontested {fr(tot['uncontested'], m):.4f}")
        summ["funnel"] = fun
    # ------------------------------------------------------------------ leak of the raw grown region beyond c_ini
    print("\nLEAK of the raw grown region R_o = G_o & ref-invisible vs target GT (scored pairs, non-empty R_o); classes at precision tau:")
    print("   clean (dom = o, prec >= tau) / spill (dom = o, prec < tau) / larger (dom != o, R holds >= 50 % of GT_o) / different (dom != o, < 50 %)")
    leak = {}
    for var in cfg["variants"]:
        recs, recs_uc, eg_n, eg_px = [], [], 0, 0
        for v in views:
            L = v["leak"].get(var, {})
            for o in v["objs"]:
                so = str(o)
                if so not in L:
                    continue
                rec = L[so]
                if v["gt_px"][so]:
                    k = (v["scene"], v["t"], v["cam"], o)
                    rec = dict(rec, gt=v["gt_px"][so], k=k)
                    recs.append(rec)
                    if k in UC:
                        recs_uc.append(rec)
                else:
                    eg_n += 1
                    eg_px += rec["s_px"]
        leak[var] = {}
        for tag, rr in (("all", recs), ("uc140", recs_uc)):
            for tau in (0.5, 0.75, 0.9):
                c = Counter()
                for r in rr:
                    prec = r["s_on_o"] / r["s_px"]
                    if r["dom"] == r["k"][3]:
                        c["clean" if prec >= tau else "spill"] += 1
                    elif r["s_on_o"] >= 0.5 * r["gt"]:
                        c["larger"] += 1
                    else:
                        c["different|" + ("bg" if r["dom"] == 0 else "obj")] += 1
                leak[var][f"{tag}|{tau}"] = dict(c)
            n = len(rr)
            spx = sum(r["s_px"] for r in rr)
            byc = np.sum([r["by_cls"] for r in rr], 0) if rr else np.zeros(3)
            onc = np.sum([r["on_o_cls"] for r in rr], 0) if rr else np.zeros(3)
            leak[var][f"{tag}|pix"] = dict(n=n, px=spx, prec_pix=fr(sum(r["s_on_o"] for r in rr), spx),
                                           prec_cls=[fr(onc[i], byc[i]) for i in range(3)], px_cls=[int(x) for x in byc],
                                           dom_ne_o=fr(sum(r["dom"] != r["k"][3] for r in rr), n),
                                           s_over_gt_median=float(np.median([r["s_px"] / r["gt"] for r in rr])) if rr else None)
            if not n:
                continue
            cl = {tt: leak[var][f"{tag}|{tt}"] for tt in (0.5, 0.75, 0.9)}

            def share(c, key):
                return fr(sum(v for kk, v in c.items() if kk.startswith(key)), n)
            P = leak[var][f"{tag}|pix"]
            print(f"   {var:10s} {tag:6s} n {n:5d}  clean {share(cl[0.75], 'clean'):.3f} [{share(cl[0.9], 'clean'):.3f}-{share(cl[0.5], 'clean'):.3f}]  "
                  f"spill {share(cl[0.75], 'spill'):.3f}  larger {share(cl[0.75], 'larger'):.3f}  different {share(cl[0.75], 'different'):.3f} "
                  f"(obj {share(cl[0.75], 'different|obj'):.3f} / unlabelled {share(cl[0.75], 'different|bg'):.3f}); raw px {spx:,} precision "
                  f"{P['prec_pix']:.3f} (outside {P['prec_cls'][0]:.3f} / occluded {P['prec_cls'][1]:.3f} / nodepth {P['prec_cls'][2]:.3f} of "
                  f"{P['px_cls'][0]:,} / {P['px_cls'][1]:,} / {P['px_cls'][2]:,} px); median |R|/|GT| {P['s_over_gt_median']:.2f}")
        leak[var]["empty_gt"] = dict(n=eg_n, px=eg_px)
        print(f"   {var:10s} raw regions on objects absent from the target GT: {eg_n} ({eg_px:,} px)")
    # ------------------------------------------------------------------ failure modes of the primary
    var = "mv" if "mv" in cfg["variants"] else cfg["variants"][0]
    rows = {k: dict(zip(F, pairs[k].get(var, ZERO))) for k in keys}
    d = {k: (r["dJ"] + r["dF"]) / 2 for k, r in rows.items()}
    print(f"\nFAILURE MODES [{var}] by object type (objects_labels.json, diagnostic only) and GT-free planar flag:")
    for name, fn in (("type", lambda k: types[k[0]].get(k[3], (None, None))[0]),
                     ("planar", lambda k: (objdiag.get((k[0], k[1], k[3])) or {}).get("planar"))):
        grp = defaultdict(list)
        for k in keys:
            grp[fn(k)].append(k)
        for g, ks in sorted(grp.items(), key=lambda kv: str(kv[0])):
            ad = sum(rows[k]["added"] for k in ks)
            print(f"   {name}={g!s:8s} n {len(ks):5d}  base {np.mean([base[k]['JF'] for k in ks]):.4f}  dJF {np.mean([d[k] for k in ks]):+.5f}  "
                  f"up {sum(d[k] > 0 for k in ks)} down {sum(d[k] < 0 for k in ks)}  added {ad:,} prec {fr(sum(rows[k]['on_o'] for k in ks), ad):.3f}  "
                  f"(UC {sum(k in UC for k in ks)})")
    # largest losses: which label grows onto which GT label
    print(f"\n   worst 15 pairs by dJF [{var}]:")
    for k in sorted(keys, key=lambda k: d[k])[:15]:
        r = rows[k]
        print(f"      {k[0]:18s} t={k[1]:3d} {k[2]:12s} obj {k[3]:3d} {base[k]['label'] or '-':24s} {types[k[0]].get(k[3], ('?',))[0]:8s} "
              f"base {base[k]['JF']:.3f} dJF {d[k]:+.4f} added {r['added']:,} prec {fr(r['on_o'], r['added']):.3f} other {fr(r['on_other'], r['added']):.3f}")
    print(f"\n   best 15 pairs by dJF [{var}]:")
    for k in sorted(keys, key=lambda k: -d[k])[:15]:
        r = rows[k]
        print(f"      {k[0]:18s} t={k[1]:3d} {k[2]:12s} obj {k[3]:3d} {base[k]['label'] or '-':24s} {types[k[0]].get(k[3], ('?',))[0]:8s} "
              f"base {base[k]['JF']:.3f} dJF {d[k]:+.4f} added {r['added']:,} prec {fr(r['on_o'], r['added']):.3f} {'UC' if k in UC else ''}")
    print(f"\nPRIMARY [{var}] per scene: n  base J&F  dJF  added px  precision  (UC pairs: dJF)")
    for sc in sorted({k[0] for k in keys}):
        sel = [k for k in keys if k[0] == sc]
        ad = sum(rows[k]["added"] for k in sel)
        u = [k for k in sel if k in UC]
        print(f"   {sc:20s} {len(sel):4d}  {np.mean([base[k]['JF'] for k in sel]):.4f}  {np.mean([d[k] for k in sel]):+.5f}  {ad:10,d}  "
              f"{fr(sum(rows[k]['on_o'] for k in sel), ad):.3f}   ({len(u)}: {np.mean([d[k] for k in u]) if u else float('nan'):+.4f})")
    print(f"\nPRIMARY [{var}] dev under-cover by (scene, cam, obj): n  base  dJF  recovered of missed  added prec  | funnel inv&valid acc trav grown")
    grp = defaultdict(list)
    for k in UCk:
        grp[(k[0], k[2], k[3])].append(k)
    for g, ks in sorted(grp.items(), key=lambda kv: -sum(base[k]["missed_px"] for k in kv[1])):
        M = sum(base[k]["missed_px"] for k in ks)
        ad = sum(rows[k]["added"] for k in ks)
        tot = Counter()
        for k in ks:
            tot.update(funnel.get(k, {}))
        print(f"   {g[0]:18s} {g[1]:12s} {g[2]:3d} {base[ks[0]]['label'] or '-':26s} {types[g[0]].get(g[2], ('?',))[0]:7s} {len(ks)}  "
              f"{np.mean([base[k]['JF'] for k in ks]):.3f}  {np.mean([d[k] for k in ks]):+.4f}  {fr(sum(rows[k]['on_o'] for k in ks), M):.3f} of {M:9,d}  "
              f"{fr(sum(rows[k]['on_o'] for k in ks), ad):.3f} ({ad:,})  | {fr(tot['missed_invis_valid'], M):.2f} {fr(tot['acc'], M):.2f} "
              f"{fr(tot['trav'], M):.2f} {fr(tot['grown_valid'], M):.2f}; nod {fr(tot['missed_nodepth'], M):.2f} splat {fr(tot['nod_splat_src'], M):.2f} "
              f"grown {fr(tot['grown_nod'], M):.2f}; conn {fr(tot['connected'], M):.3f}")
    # ------------------------------------------------------------------ sensitivity runs
    if a.sens:
        print("\nSENSITIVITY (one constant at a time; registered values in brackets)")
        sens = {}
        for sd in a.sens:
            fs, vs = load_run(sd)
            sc = json.load(open(os.path.join(sd, "config.json")))
            diff = {kk: vv for kk, vv in sc["cfg"].items() if cfg["cfg"][kk] != vv}
            P = {}
            for v in vs:
                for o in v["pairs"]:
                    P[(v["scene"], v["t"], v["cam"], int(o))] = v["pairs"][o]
            assert len(fs) == 85 and set(P) | set(base) == set(base), sd
            for var2 in sc["variants"]:
                rr = {k: dict(zip(F, P.get(k, {}).get(var2, ZERO))) for k in keys}
                dd = {k: (r["dJ"] + r["dF"]) / 2 for k, r in rr.items()}
                res = {}
                for name, sel in (("all", keys), ("uc140", UCk), ("clean67", CLk)):
                    ad = sum(rr[k]["added"] for k in sel)
                    res[name] = dict(dJF=float(np.mean([dd[k] for k in sel])), added=ad, prec=fr(sum(rr[k]["on_o"] for k in sel), ad),
                                     recovered=fr(sum(rr[k]["on_o"] for k in sel), sum(base[k]["missed_px"] for k in sel)))
                lo, hi = boot([dd[k] for k in keys], [k[0] for k in keys])
                res["all"]["ci"] = [lo, hi]
                lo, hi = boot([dd[k] for k in UCk], [(k[0], k[2], k[3]) for k in UCk])
                res["uc140"]["ci"] = [lo, hi]
                sens[f"{sd}|{var2}"] = dict(diff=diff, **res)
                print(f"   {os.path.basename(sd.rstrip('/')):14s} {str(diff):22s} {var2:10s} ALL dJF {res['all']['dJF']:+.5f} "
                      f"[{res['all']['ci'][0]:+.5f}, {res['all']['ci'][1]:+.5f}] added {res['all']['added']:>10,} prec {res['all']['prec']:.3f} | "
                      f"UC140 dJF {res['uc140']['dJF']:+.5f} [{res['uc140']['ci'][0]:+.5f}, {res['uc140']['ci'][1]:+.5f}] prec {res['uc140']['prec']:.3f} "
                      f"rec {res['uc140']['recovered']:.4f} | CLEAN67 dJF {res['clean67']['dJF']:+.5f}")
        summ["sensitivity"] = sens
    if a.json:
        json.dump(dict(summary=summ, leak=leak), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

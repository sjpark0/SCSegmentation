#!/usr/bin/env python3
"""S1-E13 candidate C report (host).  Reads compose_score.py output + _e13_dev/dev_pairs.json + dev_undercover.json.
DEV ONLY.  stage1/.venv/bin/python -I report.py rows.json [--json summary.json]"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

DEV = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_dev"
F = ["dJ", "dF", "added", "on_o", "on_other", "on_bg", "add_out", "add_occ", "add_nod", "on_out", "on_occ", "on_nod", "on_other_seeded"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    R = json.load(open(a.rows))
    dp = json.load(open(f"{DEV}/dev_pairs.json"))
    uc = json.load(open(f"{DEV}/dev_undercover.json"))["rows"]
    base = {(r["scene"], r["t"], r["cam"], r["obj"]): r for r in dp["rows"]}
    UC = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in uc}
    CLEAN = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in uc if r["precision"] is not None and r["precision"] >= 0.85}
    DOGTREE = {k for k in UC if k[0] == "Dog" and k[2] == "camera_0004" and k[3] == 3}
    views = R["views"]
    # ------------------------------------------------------------------ consistency with the dev scorer
    pairs, maxd, clsbad, n_ov = {}, 0.0, 0, 0
    for v in views:
        n_ov += v["n_dev_overlap"]
        for o, (J, Fm) in v["base"].items():
            k = (v["scene"], v["t"], v["cam"], int(o))
            b = base[k]
            maxd = max(maxd, abs(J - b["J"]), abs(Fm - b["F"]))
            g = v["gtcls"][o]
            clsbad += g != [b["gt_vis"], b["gt_outside"], b["gt_occluded"], b["gt_nodepth"]]
            pairs[k] = v["pairs"].get(o, {})
    scenes_in = {v["scene"] for v in views}
    exp = {k for k in base if k[0] in scenes_in}
    print(f"views {len(views)}; scored pairs {len(pairs)} (dev_pairs.json for these scenes: {len(exp)}; equal sets: {set(pairs) == exp}); "
          f"max |J,F - dev_pairs| = {maxd:.2e}; GT class-count mismatches vs dev_pairs: {clsbad}; dev-mask overlap px {n_ov}")
    assert set(pairs) == exp and maxd < 1e-9 and clsbad == 0
    keys = sorted(pairs)
    UCk = [k for k in keys if k in UC]
    CLk = [k for k in keys if k in CLEAN]
    summ = {}
    print("\nConfig = variant | min_proj | iou gate.  dJF = mean J&F change; CIs: all pairs scene-cluster, under-cover object-cluster "
          "bootstrap 95 % (10 000, seed 0)")
    for var in R["variants"]:
        for mp, gate in R["configs"]:
            key = f"{var}|{mp}|{gate}"
            rows = {k: dict(zip(F, pairs[k].get(key, [0.0] * 2 + [0] * 11))) for k in keys}
            d = {k: (r["dJ"] + r["dF"]) / 2 for k, r in rows.items()}
            S = {}
            for name, sel, clus in (("all", keys, lambda k: k[0]), ("uc140", UCk, lambda k: (k[0], k[2], k[3])),
                                    ("clean67", CLk, lambda k: (k[0], k[2], k[3])),
                                    ("not_uc", [k for k in keys if k not in UC], lambda k: k[0])):
                if not sel:
                    continue
                vals = [d[k] for k in sel]
                lo, hi = boot(vals, [clus(k) for k in sel])
                add = sum(rows[k]["added"] for k in sel)
                on = sum(rows[k]["on_o"] for k in sel)
                miss = sum(base[k]["missed_px"] for k in sel)
                S[name] = dict(n=len(sel), dJF=float(np.mean(vals)), ci=[lo, hi], dJ=float(np.mean([rows[k]["dJ"] for k in sel])),
                               dF=float(np.mean([rows[k]["dF"] for k in sel])),
                               base_JF=float(np.mean([base[k]["JF"] for k in sel])),
                               n_changed=sum(rows[k]["added"] > 0 for k in sel), n_up=sum(d[k] > 0 for k in sel),
                               n_down=sum(d[k] < 0 for k in sel), sum_up=float(sum(max(0, d[k]) for k in sel)),
                               sum_down=float(sum(min(0, d[k]) for k in sel)),
                               added=add, on_o=on, prec_pooled=fr(on, add),
                               prec_pair_mean=float(np.mean([rows[k]["on_o"] / rows[k]["added"] for k in sel if rows[k]["added"]]))
                               if any(rows[k]["added"] for k in sel) else float("nan"),
                               on_other=sum(rows[k]["on_other"] for k in sel), on_bg=sum(rows[k]["on_bg"] for k in sel),
                               on_other_seeded=sum(rows[k]["on_other_seeded"] for k in sel),
                               n_prompted=sum(any(f"funnel|{var}" == x for x in pairs[k]) for k in sel),
                               missed=miss, recovered=fr(on, miss),
                               recovered_pair_mean=float(np.mean([rows[k]["on_o"] / base[k]["missed_px"] for k in sel if base[k]["missed_px"]])))
                for c, cn in ((1, "outside"), (2, "occluded"), (3, "nodepth")):
                    ad = sum(rows[k][F[5 + c]] for k in sel)
                    oc = sum(rows[k][F[8 + c]] for k in sel)
                    mc = sum(base[k][f"missed_{cn}"] for k in sel)
                    S[name][f"cls_{cn}"] = dict(added=ad, on_o=oc, prec=fr(oc, ad), missed=mc, recovered=fr(oc, mc))
            # empty-GT additions (FP outside J&F)
            eg = sum(sum(v["empty_gt"].get(key, {}).values()) for v in views)
            S["empty_gt_added_px"] = eg
            S["all"]["prec_pooled_incl_emptygt"] = fr(S["all"]["on_o"], S["all"]["added"] + eg)
            if UCk:
                nd = [k for k in UCk if k not in DOGTREE]
                on = sum(rows[k]["on_o"] for k in nd)
                miss = sum(base[k]["missed_px"] for k in nd)
                S["uc140"]["recovered_wo_dogtree"] = fr(on, miss)
                S["uc140"]["dJF_wo_dogtree"] = float(np.mean([d[k] for k in nd]))
                S["uc140"]["dogtree_dJF"] = float(np.mean([d[k] for k in UCk if k in DOGTREE])) if DOGTREE & set(UCk) else None
            summ[key] = S
            A, U, C = S["all"], S.get("uc140"), S.get("clean67")
            print(f"\n[{key}]  ALL n={A['n']}: dJF {A['dJF']:+.5f} [{A['ci'][0]:+.5f}, {A['ci'][1]:+.5f}] (dJ {A['dJ']:+.5f} dF {A['dF']:+.5f}); "
                  f"changed {A['n_changed']} up {A['n_up']} down {A['n_down']} (sum +{A['sum_up']:.3f} / {A['sum_down']:.3f}); "
                  f"added {A['added']:,} px, precision pooled {A['prec_pooled']:.3f} (incl. {eg:,} px on empty-GT objects "
                  f"{A['prec_pooled_incl_emptygt']:.3f}), pair-mean {A['prec_pair_mean']:.3f}; on other obj {fr(A['on_other'], A['added']):.3f} "
                  f"(seeded at t {fr(A['on_other_seeded'], A['added']):.3f}), "
                  f"on unlabelled {fr(A['on_bg'], A['added']):.3f}")
            print("      by class (added px, precision): " + "; ".join(
                f"{cn} {A[f'cls_{cn}']['added']:,} {A[f'cls_{cn}']['prec']:.3f}" for cn in ("outside", "occluded", "nodepth")))
            if U:
                print(f"   UC140: prompted {U['n_prompted']}/{U['n']}; on other obj {fr(U['on_other'], U['added']):.3f} (seeded {fr(U['on_other_seeded'], U['added']):.3f})")
                print(f"   UC140: base {U['base_JF']:.4f} dJF {U['dJF']:+.5f} [{U['ci'][0]:+.5f}, {U['ci'][1]:+.5f}]  up {U['n_up']} down {U['n_down']}  "
                      f"added {U['added']:,} prec {U['prec_pooled']:.3f} (pair-mean {U['prec_pair_mean']:.3f}); missed px recovered "
                      f"{U['recovered']:.4f} (pair-mean {U['recovered_pair_mean']:.4f}; w/o Dog tree {U['recovered_wo_dogtree']:.4f}); "
                      f"dJF w/o Dog tree {U['dJF_wo_dogtree']:+.5f}, Dog tree {U['dogtree_dJF'] if U['dogtree_dJF'] is None else round(U['dogtree_dJF'], 5)}")
                print("      recovered by class: " + "; ".join(
                    f"{cn} {U[f'cls_{cn}']['recovered']:.4f} of {U[f'cls_{cn}']['missed']:,}" for cn in ("outside", "occluded", "nodepth")))
            if C:
                print(f"   CLEAN67: base {C['base_JF']:.4f} dJF {C['dJF']:+.5f} [{C['ci'][0]:+.5f}, {C['ci'][1]:+.5f}] up {C['n_up']} down {C['n_down']}  "
                      f"prec {C['prec_pooled']:.3f} recovered {C['recovered']:.4f}")
            if "not_uc" in S:
                N = S["not_uc"]
                print(f"   NOT-UC n={N['n']}: dJF {N['dJF']:+.5f} [{N['ci'][0]:+.5f}, {N['ci'][1]:+.5f}] up {N['n_up']} down {N['n_down']} "
                      f"added {N['added']:,} prec {N['prec_pooled']:.3f}")
    # ------------------------------------------------------------------ funnel (registered config)
    print("\nFUNNEL (registered config min_proj 20, no gate): fraction of missed px that survive each step")
    fun = {}
    for var in R["variants"]:
        for name, sel in (("all", keys), ("uc140", UCk), ("clean67", CLk)):
            tot = np.zeros(6)
            for k in sel:
                f = pairs[k].get(f"funnel|{var}")
                if f:
                    tot += np.array(f, float)
                else:
                    tot[0] += base[k]["missed_px"]
            fun[f"{var}|{name}"] = (tot / max(1, tot[0])).tolist()
            print(f"   {var:10s} {name:8s} missed {int(tot[0]):>11,}  in S {tot[1] / tot[0]:.4f}  & ref-invisible {tot[2] / tot[0]:.4f}  "
                  f"& unclaimed {tot[3] / tot[0]:.4f}  & 8-connected {tot[4] / tot[0]:.4f}  & uncontested {tot[5] / tot[0]:.4f}")
    # ------------------------------------------------------------------ leak of the raw segment
    print("\nLEAK of the raw SAM segment S_o vs target GT (scored pairs; non-empty segments). classes at precision threshold tau:")
    print("   clean (dom = o, prec >= tau) / spill (dom = o, prec < tau) / larger (dom != o, S holds >= 50 % of GT_o) / "
          "different (dom != o, < 50 %), different split into dom = other object | dom = unlabelled")
    leak = {}
    for var in R["variants"]:
        recs, recs_uc, eg_n, eg_px, n_prompted, n_empty = [], [], 0, 0, 0, 0
        for v in views:
            L = v["leak"].get(var, {})
            for o in v["objs"]:
                so = str(o)
                if so in L:
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
        n_prompted = sum(1 for v in views for o in v["objs"] if v["proj_px"][str(o)] >= 20 and v["gt_px"][str(o)])
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
            leak[var][f"{tag}|pix"] = dict(n=n, prec_pix=fr(sum(r["s_on_o"] for r in rr), spx),
                                           prec_invis_pix=fr(sum(r["s_invis_on_o"] for r in rr), sum(r["s_invis"] for r in rr)),
                                           dom_ne_o=fr(sum(r["dom"] != r["k"][3] for r in rr), n),
                                           s_over_gt_median=float(np.median([r["s_px"] / r["gt"] for r in rr])) if rr else None)
        leak[var]["empty_gt"] = dict(n_nonempty_segments=eg_n, px=eg_px)
        leak[var]["n_prompted_scored"] = n_prompted
        for tag in ("all", "uc140"):
            P = leak[var][f"{tag}|pix"]
            if not P["n"]:
                continue
            cl = {t: leak[var][f"{tag}|{t}"] for t in (0.5, 0.75, 0.9)}
            n = P["n"]

            def share(c, key):
                return fr(sum(v for k, v in c.items() if k.startswith(key)), n)
            print(f"   {var:10s} {tag:6s} n {n:5d}  clean {share(cl[0.75], 'clean'):.3f} [{share(cl[0.9], 'clean'):.3f}-{share(cl[0.5], 'clean'):.3f}]  "
                  f"spill {share(cl[0.75], 'spill'):.3f} [{share(cl[0.5], 'spill'):.3f}-{share(cl[0.9], 'spill'):.3f}]  larger {share(cl[0.75], 'larger'):.3f}  "
                  f"different {share(cl[0.75], 'different'):.3f} (obj {share(cl[0.75], 'different|obj'):.3f} / unlabelled {share(cl[0.75], 'different|bg'):.3f}); "
                  f"dom != o {P['dom_ne_o']:.3f}; pixel precision of S {P['prec_pix']:.3f}, of S & ref-invisible {P['prec_invis_pix']:.3f}; "
                  f"median |S|/|GT| {P['s_over_gt_median']:.2f}")
        print(f"   {var:10s} prompted scored pairs {n_prompted}; segments on objects absent from the target GT: {eg_n} ({eg_px:,} px)")
    # ------------------------------------------------------------------ registered primary: per scene / per under-cover object
    key = "pts|20|None"
    rows = {k: dict(zip(F, pairs[k].get(key, [0.0] * 2 + [0] * 11))) for k in keys}
    d = {k: (r["dJ"] + r["dF"]) / 2 for k, r in rows.items()}
    print(f"\nPRIMARY [{key}] per scene: n  base J&F  dJF  added px  precision  (UC pairs: dJF)")
    for sc in sorted({k[0] for k in keys}):
        sel = [k for k in keys if k[0] == sc]
        ad = sum(rows[k]["added"] for k in sel)
        u = [k for k in sel if k in UC]
        print(f"   {sc:20s} {len(sel):4d}  {np.mean([base[k]['JF'] for k in sel]):.4f}  {np.mean([d[k] for k in sel]):+.5f}  {ad:10,d}  "
              f"{fr(sum(rows[k]['on_o'] for k in sel), ad):.3f}   ({len(u)}: {np.mean([d[k] for k in u]) if u else float('nan'):+.4f})")
    print(f"\nPRIMARY [{key}] dev under-cover by (scene, cam, obj): n  base  dJF  prompted  recovered of missed  added prec")
    grp = defaultdict(list)
    for k in UCk:
        grp[(k[0], k[2], k[3])].append(k)
    for g, ks in sorted(grp.items(), key=lambda kv: -sum(base[k]["missed_px"] for k in kv[1])):
        M = sum(base[k]["missed_px"] for k in ks)
        ad = sum(rows[k]["added"] for k in ks)
        print(f"   {g[0]:18s} {g[1]:12s} {g[2]:3d} {base[ks[0]]['label'] or '-':26s} {len(ks)}  {np.mean([base[k]['JF'] for k in ks]):.3f}  "
              f"{np.mean([d[k] for k in ks]):+.4f}  {sum(f'funnel|pts' in pairs[k] for k in ks)}/{len(ks)}  "
              f"{fr(sum(rows[k]['on_o'] for k in ks), M):.3f} of {M:9,d}  {fr(sum(rows[k]['on_o'] for k in ks), ad):.3f} ({ad:,})")
    if a.json:
        json.dump(dict(summary=summ, funnel=fun, leak=leak), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S1-E13 candidate E1 (text-concept) -- POST-HOC DIAGNOSTIC report (not registered; see posthoc.py).  DEV ONLY.
  stage1/.venv/bin/python -I posthoc_report.py posthoc_rows.json [--json posthoc_summary.json]"""
import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_text-concept")
from report import DEV, F, S1, ZERO, boot, fr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    R = json.load(open(a.rows))
    dp = json.load(open(f"{DEV}/dev_pairs.json"))
    uc = json.load(open(f"{DEV}/dev_undercover.json"))["rows"]
    idx = json.load(open(f"{DEV}/index.json"))
    base = {(r["scene"], r["t"], r["cam"], r["obj"]): r for r in dp["rows"]}
    UC = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in uc}
    CLEAN = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in uc if r["precision"] is not None and r["precision"] >= 0.85}
    DOGTREE = {k for k in UC if k[0] == "Dog" and k[2] == "camera_0004" and k[3] == 3}
    LAB, UNIQ = {}, {}
    for s, S in idx["scenes"].items():
        L = {int(k): v for k, v in json.load(open(f"{S1}/Data/{s}/objects_labels.json")).items()}
        cnt = Counter(L[o]["label"] for o in S["basic"])
        for o in S["basic"]:
            LAB[(s, o)] = L[o]
            UNIQ[(s, o)] = cnt[L[o]["label"]] == 1
    pairs = {}
    for v in R["views"]:
        for o, r in v["pairs"].items():
            pairs[(v["scene"], v["t"], v["cam"], int(o))] = r
    assert set(pairs) == set(base), (len(pairs), len(base))
    keys = sorted(pairs)
    UCk = [k for k in keys if k in UC]
    CLk = [k for k in keys if k in CLEAN]
    nd = [k for k in UCk if k not in DOGTREE]
    print("POST-HOC DIAGNOSTIC (not registered; written after the registered E1 results were seen).  DEV ONLY.  PROTOCOL CHANGE "
          "(label names).  TAU 0.5, split names.  noconn = no 8-connectivity to D_o; contested pixels dropped as registered.")
    summ = {}
    for arm in R["arms"]:
        rows = {k: dict(zip(F, pairs[k].get(arm, ZERO))) for k in keys}
        d = {k: (r["dJ"] + r["dF"]) / 2 for k, r in rows.items()}
        S = {}
        for name, sel, clus in (("all", keys, lambda k: k[0]), ("uc140", UCk, lambda k: (k[0], k[2], k[3])),
                                ("clean67", CLk, lambda k: (k[0], k[2], k[3])), ("uc_wo_dogtree", nd, lambda k: (k[0], k[2], k[3])),
                                ("not_uc", [k for k in keys if k not in UC], lambda k: k[0])):
            vals = [d[k] for k in sel]
            lo, hi = boot(vals, [clus(k) for k in sel])
            add = sum(rows[k]["added"] for k in sel)
            on = sum(rows[k]["on_o"] for k in sel)
            miss = sum(base[k]["missed_px"] for k in sel)
            S[name] = dict(n=len(sel), dJF=float(np.mean(vals)), ci=[lo, hi], n_up=sum(d[k] > 0 for k in sel),
                           n_down=sum(d[k] < 0 for k in sel), sum_down=float(sum(min(0, d[k]) for k in sel)), added=add,
                           prec=fr(on, add), on_other=fr(sum(rows[k]["on_other"] for k in sel), add), recovered=fr(on, miss),
                           rec_cls={cn: fr(sum(rows[k][F[8 + c]] for k in sel), sum(base[k][f"missed_{cn}"] for k in sel))
                                    for c, cn in ((1, "outside"), (2, "occluded"), (3, "nodepth"))})
        eg = sum(sum(v["empty_gt"].get(arm, {}).values()) for v in R["views"])
        S["empty_gt_added_px"] = eg
        summ[arm] = S
        print(f"\n[{arm}]  empty-GT added px {eg:,}")
        for name in ("all", "uc140", "clean67", "uc_wo_dogtree", "not_uc"):
            x = S[name]
            print(f"   {name:13s} n {x['n']:5d}  dJF {x['dJF']:+.5f} [{x['ci'][0]:+.5f}, {x['ci'][1]:+.5f}]  up {x['n_up']} down {x['n_down']} "
                  f"(sum down {x['sum_down']:.3f})  added {x['added']:,} prec {x['prec']:.3f} (on other obj {x['on_other']:.3f})  "
                  f"recovered {x['recovered']:.4f} (out {x['rec_cls']['outside']:.4f} / occ {x['rec_cls']['occluded']:.4f} / nod {x['rec_cls']['nodepth']:.4f})")
        print("   by object type / name multiplicity (all pairs): n  dJF  added  prec | UC n  dJF  recovered")
        for gname, gf in (("type", lambda k: LAB[(k[0], k[3])].get("type")), ("name", lambda k: "scene-unique" if UNIQ[(k[0], k[3])] else "shared")):
            for g in sorted({gf(k) for k in keys}):
                ks = [k for k in keys if gf(k) == g]
                us = [k for k in ks if k in UC]
                ad = sum(rows[k]["added"] for k in ks)
                S[f"{gname}={g}"] = dict(n=len(ks), dJF=float(np.mean([d[k] for k in ks])), added=ad, prec=fr(sum(rows[k]["on_o"] for k in ks), ad),
                                         uc_n=len(us), uc_dJF=float(np.mean([d[k] for k in us])) if us else None)
                print(f"      {gname}={g:13s} {len(ks):5d}  {np.mean([d[k] for k in ks]):+.5f}  {ad:11,d}  {fr(sum(rows[k]['on_o'] for k in ks), ad):.3f} | "
                      f"{len(us):3d}  {np.mean([d[k] for k in us]) if us else float('nan'):+.5f}  "
                      f"{fr(sum(rows[k]['on_o'] for k in us), sum(base[k]['missed_px'] for k in us)):.4f}")
        print("   per scene: n  dJF  added  prec | UC n dJF")
        for sc in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == sc]
            us = [k for k in ks if k in UC]
            ad = sum(rows[k]["added"] for k in ks)
            print(f"      {sc:20s} {len(ks):4d}  {np.mean([d[k] for k in ks]):+.5f}  {ad:11,d}  {fr(sum(rows[k]['on_o'] for k in ks), ad):.3f} | "
                  f"{len(us):3d} {np.mean([d[k] for k in us]) if us else float('nan'):+.4f}")
        if arm in ("union_all|noconn|unique", "union_all|noconn|env"):
            print("   under-cover by (scene, cam, obj): label  n  base  dJF  recovered  added prec")
            grp = defaultdict(list)
            for k in UCk:
                grp[(k[0], k[2], k[3])].append(k)
            for g, ks in sorted(grp.items(), key=lambda kv: -sum(base[k]["missed_px"] for k in kv[1])):
                M = sum(base[k]["missed_px"] for k in ks)
                ad = sum(rows[k]["added"] for k in ks)
                print(f"      {g[0]:18s} {g[1]:12s} {g[2]:3d} {LAB[(g[0], g[2])]['label']:22s} {'U' if UNIQ[(g[0], g[2])] else 'S'} {len(ks)}  "
                      f"{np.mean([base[k]['JF'] for k in ks]):.3f}  {np.mean([d[k] for k in ks]):+.4f}  "
                      f"{fr(sum(rows[k]['on_o'] for k in ks), M):.3f} of {M:9,d}  {fr(sum(rows[k]['on_o'] for k in ks), ad):.3f} ({ad:,})")
            print("   most harmed pairs (dJF < -0.02): scene t cam obj label  dJF  added  prec  on-other share")
            for k in sorted([k for k in keys if d[k] < -0.02], key=lambda k: d[k])[:25]:
                r = rows[k]
                print(f"      {k[0]:18s} {k[1]:3d} {k[2]:12s} {k[3]:3d} {LAB[(k[0], k[3])]['label']:22s} {d[k]:+.4f}  {r['added']:9,d}  "
                      f"{fr(r['on_o'], r['added']):.3f}  {fr(r['on_other'], r['added']):.3f}")
    if a.json:
        json.dump(summ, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

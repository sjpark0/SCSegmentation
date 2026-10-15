#!/usr/bin/env python3
"""S1-E14 train-band: D2 - D1 and D1 - C0' on DEV-A by GT size (pair means; good = C0 JF >= 0.8).  -> by_size.json"""
import json, sys
import numpy as np
TB = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e14_train-band"
L = lambda f: {(r["scene"], r["t"], r["cam"], r["obj"]): r for r in json.load(open(f"{TB}/{f}"))["rows"]}
A = {"C0": L("score_geoset_rep_x_A.json"), "C0'": L("score_geoset_cut0_rep_x_A.json"), "D1": L("score_geoset_dec_rep_x_A.json"), "D2": L("score_geoset_decb_rep_x_A.json")}
BINS = [(1, 2000), (2000, 10000), (10000, 100000), (100000, 10 ** 9)]
out = {}
for a, b in (("C0'", "D1"), ("D1", "D2"), ("C0", "D2")):
    out[f"{b}-{a}"] = {}
    for lo, hi in BINS:
        ks = [k for k, r in A["C0"].items() if lo <= r["gt_px"] < hi and r["JF"] >= 0.8]
        d = lambda m: float(np.mean([(A[b][k][m] or 0) - (A[a][k][m] or 0) for k in ks]))
        out[f"{b}-{a}"][f"{lo}-{hi}"] = dict(n=len(ks), dJF=d("JF"), dJ=d("J"), dF=d("F"), drecall=d("recall"), dprecision=d("precision"),
                                             pred_over_gt_a=float(sum(A[a][k]["seed_px"] for k in ks) / sum(A[a][k]["gt_px"] for k in ks)),
                                             pred_over_gt_b=float(sum(A[b][k]["seed_px"] for k in ks) / sum(A[b][k]["gt_px"] for k in ks)))
        v = out[f"{b}-{a}"][f"{lo}-{hi}"]
        print(f"{b}-{a:3s} good {lo:>6}-{hi:<10} n {v['n']:4d} dJF {v['dJF']:+.4f} dJ {v['dJ']:+.4f} dF {v['dF']:+.4f} drec {v['drecall']:+.4f} dprec {v['dprecision']:+.4f} pred/gt {v['pred_over_gt_a']:.4f}->{v['pred_over_gt_b']:.4f}")
json.dump(out, open(f"{TB}/by_size.json", "w"), indent=1)

#!/usr/bin/env python3
"""S1-E14 train-band: expected DEV-A effect of a balanced band arm from the D1 -> D2 output blend (host).

Reads interp.json (per-pair J, F, recall, precision for blends a and threshold shifts of D1 / D2), gb_summary.json
(W_grad = rms ||g_base|| / rms ||g_band|| per arm) and the pair list (good = C0 JF >= 0.8).
Dose map W -> a (the blend weight that stands in for a band arm trained at weight W, D1 = W 0, D2 = W 1):
    SGD-like (displacement linear in the loss weight):          a = W
    Adam-like (AdamW + grad-norm clip 1.0: step size fixed,      a = s(W) / s(1),  s(W) = W r / (1 + W r),
               direction = gradient-norm share of the band term)  r = 1 / W_grad (band / base gradient norm, D1 on DEV-A)
Per variant: hierarchical (pair -> view -> scene) J&F and the paired scene-cluster bootstrap vs D1 (a = 0), pair means,
good-seed J&F / 1-recall / 1-precision.
    stage1/.venv/bin/python dose_estimate.py --interp ../interp.json --gb ../gb_summary.json --out ../dose_estimate.json
"""
import argparse
import json
import os
import sys

import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from mvseg2d.paired import paired_scene_bootstrap  # noqa: E402

PAIRS = os.path.join(S1, "geomem", "runs", "_e13_dev", "dev_pairs.json")
WS = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 1.0)

ap = argparse.ArgumentParser()
ap.add_argument("--interp", required=True)
ap.add_argument("--gb", required=True)
ap.add_argument("--gb-arm", default="geoset_dec_rep_x_A")
ap.add_argument("--out", required=True)
a = ap.parse_args()
I = json.load(open(a.interp))
base = {(r["scene"], int(r["t"]), r["cam"], int(r["obj"])): r["JF"] for r in json.load(open(PAIRS))["rows"]}
keys = [tuple(k[:2]) + (k[2], int(k[3])) for k in I["keys"]]
keys = [(k[0], int(k[1]), k[2], int(k[3])) for k in keys]
good = np.array([base[k] >= 0.8 for k in keys])


def rows(v):
    return [dict(scene=k[0], view=f"{k[1]}/{k[2]}", J=x[0], F=x[1], JF=(x[0] + x[1]) / 2) for k, x in zip(keys, I["variants"][v])]


ref = rows("a0.0")
out = dict(variants={}, dose={})
for v, sc in I["variants"].items():
    arr = np.array(sc)
    jf = arr[:, :2].mean(1)
    rv = rows(v)
    bs = paired_scene_bootstrap(ref, rv, key="JF", n=10_000, seed=0, view_key="view")
    bg = paired_scene_bootstrap([r for r, g in zip(ref, good) if g], [r for r, g in zip(rv, good) if g], key="JF", n=10_000, seed=0, view_key="view")
    out["variants"][v] = dict(JF_pair=float(jf.mean()), J=float(arr[:, 0].mean()), F=float(arr[:, 1].mean()),
                              d_vs_D1=bs["delta"], ci_vs_D1=list(bs["ci"]), d_good_vs_D1=bg["delta"], ci_good_vs_D1=list(bg["ci"]),
                              JF_good_pair=float(jf[good].mean()), one_minus_recall_good=float(1 - arr[good, 2].mean()),
                              one_minus_precision_good=float(1 - arr[good, 3].mean()))
    print(f"{v:10s} J&F {jf.mean():.5f} (J {arr[:, 0].mean():.5f} F {arr[:, 1].mean():.5f}) d vs D1 {bs['delta']:+.4f} [{bs['ci'][0]:+.4f},{bs['ci'][1]:+.4f}] "
          f"good {bg['delta']:+.4f} [{bg['ci'][0]:+.4f},{bg['ci'][1]:+.4f}] | good 1-rec {1 - arr[good, 2].mean():.4f} 1-prec {1 - arr[good, 3].mean():.4f}")
gb = json.load(open(a.gb))
r = 1.0 / gb[a.gb_arm]["W_grad"]
alphas = sorted(float(v[1:]) for v in I["variants"] if v.startswith("a"))
dh = np.array([out["variants"][f"a{x}"]["d_vs_D1"] for x in alphas])
dg = np.array([out["variants"][f"a{x}"]["d_good_vs_D1"] for x in alphas])
s = lambda W: W * r / (1 + W * r)
for W in WS:
    for name, al in (("sgd", W), ("adam", s(W) / s(1.0))):
        out["dose"][f"W{W}_{name}"] = dict(alpha=al, d_vs_D1=float(np.interp(al, alphas, dh)), d_good_vs_D1=float(np.interp(al, alphas, dg)))
    print(f"W {W:5.2f}: a_sgd {W:.3f} -> d {np.interp(W, alphas, dh):+.4f} (good {np.interp(W, alphas, dg):+.4f}) | "
          f"a_adam {s(W) / s(1.0):.3f} -> d {np.interp(s(W) / s(1.0), alphas, dh):+.4f} (good {np.interp(s(W) / s(1.0), alphas, dg):+.4f})")
out["r_band_over_base"] = r
json.dump(out, open(a.out, "w"), indent=1)

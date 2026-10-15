#!/usr/bin/env python3
"""S1-E14 train-band: W -> expected DEV-A effect from the token-0 blend (tok0.json) and the fallback blend (dose_estimate.json).
a(W): SGD-like a = W; Adam-like a = s(W)/s(1), s(W) = W r/(1 + W r), r = 1/W_grad of D1 on DEV-A (gb_summary.json).  -> dose_tok0.json"""
import json
import numpy as np
TB = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e14_train-band"
T = json.load(open(f"{TB}/tok0.json"))
D = json.load(open(f"{TB}/dose_estimate.json"))
r = 1 / json.load(open(f"{TB}/gb_summary.json"))["geoset_dec_rep_x_A"]["W_grad"]
al = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
c = T["contrasts"]
d = [0.0] + [c[f"tok0_a{a} - D1_tok0"]["JF"]["delta"] for a in al[1:-1]] + [c["D2_tok0 - D1_tok0"]["JF"]["delta"]]
g = [0.0] + [c[f"tok0_a{a} - D1_tok0"]["good"]["delta"] for a in al[1:-1]] + [c["D2_tok0 - D1_tok0"]["good"]["delta"]]
fb = [T["fallback_share"][f"a{a}"] for a in al]
s = lambda W: W * r / (1 + W * r)
out = dict(r=r, rows={})
for W in (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 1.0):
    row = {}
    for nm, a in (("sgd", W), ("adam", s(W) / s(1.0))):
        row[nm] = dict(alpha=a, tok0_dJF=float(np.interp(a, al, d)), tok0_dJF_good=float(np.interp(a, al, g)),
                       fallback_share=float(np.interp(a, al, fb)),
                       fallback_on_dJF=D["dose"][f"W{W}_{nm}"]["d_vs_D1"], fallback_on_dJF_good=D["dose"][f"W{W}_{nm}"]["d_good_vs_D1"])
    out["rows"][str(W)] = row
    print(f"W {W:4.2f} | SGD a {row['sgd']['alpha']:.3f}: tok0 {row['sgd']['tok0_dJF']:+.4f} (good {row['sgd']['tok0_dJF_good']:+.4f}) fb-on {row['sgd']['fallback_on_dJF']:+.4f} "
          f"| Adam a {row['adam']['alpha']:.3f}: tok0 {row['adam']['tok0_dJF']:+.4f} (good {row['adam']['tok0_dJF_good']:+.4f}) fb-on {row['adam']['fallback_on_dJF']:+.4f} fallback share {row['adam']['fallback_share']:.3f}")
json.dump(out, open(f"{TB}/dose_tok0.json", "w"), indent=1)

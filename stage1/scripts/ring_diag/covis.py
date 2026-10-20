"""Read-only, label-free: per-view co-visibility with the reference (fraction of the view's valid Pi3 pixels that
reproject inside the reference image and are not hidden behind the reference surface, tol 1% and 5%), per scene."""
import json, os, sys
import numpy as np
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points, to_reference
D = os.path.join(REPO, "stage1/Data")
out = {}
for s in sorted(os.listdir(D)):
    m = os.path.join(D, s, "meta.json")
    if s.startswith("_") or s.endswith("_B") or not os.path.isfile(m) or not os.path.isdir(os.path.join(D, s, "Cache/pi3")):
        continue
    meta = json.load(open(m)); ref = meta["ref_cam"]
    zr = np.load(os.path.join(D, s, "Cache/pi3", f"{ref}.npz")); d_r, K_r, T_r = zr["depth"], zr["K"], zr["c2w"]
    h, w = d_r.shape
    vals = {}
    for c in meta["cams"]:
        if c == ref: continue
        z = np.load(os.path.join(D, s, "Cache/pi3", f"{c}.npz"))
        valid = (z["conf"] > CONF_MIN) & (z["depth"] > 0)
        p = to_reference(local_points(z["depth"], z["K"]), z["c2w"], T_r)
        Z = p[..., 2]; Zs = np.where(np.abs(Z) < 1e-9, 1e-9, Z)
        u = K_r[0, 0] * p[..., 0] / Zs + K_r[0, 2]; v = K_r[1, 1] * p[..., 1] / Zs + K_r[1, 2]
        inside = (Z > 0) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5)
        dref = d_r[np.clip(np.round(v), 0, h - 1).astype(int), np.clip(np.round(u), 0, w - 1).astype(int)]
        r = []
        for tol in (0.01, 0.05):
            ok = inside & valid & ((dref <= 0) | (Z <= dref * (1 + tol)))
            r.append(float(ok.sum() / max(valid.sum(), 1)))
        vals[c] = r
    a = np.array(list(vals.values()))
    out[s] = vals
    print(f"{s:22s} n={len(vals):2d} covis(tol1%) min {a[:,0].min():.2f} p10 {np.percentile(a[:,0],10):.2f} median {np.median(a[:,0]):.2f} | tol5% min {a[:,1].min():.2f}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "covis.json"), "w"), indent=0)
t = out.get("T06_flag_ganggang_F", {})
print("T06 per cam (tol1%, tol5%):", {c[4:]: [round(x, 2) for x in v] for c, v in t.items()})

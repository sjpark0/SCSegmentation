# GT-free statistics of Pi3 caches on a few dev frames (no labels read): 2nd-difference of inverse depth, cross-camera agreement
import json, os, sys
import numpy as np
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/stage1")
from geomem.pointmap import local_points, to_reference
D = "/home/sjpark/Documents/SCSegmentation/stage1/Data"
for scene, t in [("FlameSteak", 4), ("Barn", 4), ("Dog", 4), ("CBABasketball", 94), ("Breakfast", 4)]:
    cdir = f"{D}/{scene}/Cache/pi3_frames/{t:06d}"
    man = json.load(open(f"{cdir}/MANIFEST.json"))
    cams = man["cams"]
    Z = {c: np.load(f"{cdir}/{c}.npz") for c in cams}
    qs = []
    for c in cams[:6]:
        d, cf = Z[c]["depth"], Z[c]["conf"]
        v = (cf > 0.1) & (d > 0)
        w = np.where(v, 1.0 / np.maximum(d, 1e-9), np.nan)
        s2 = np.abs(w[:, :-2] - 2 * w[:, 1:-1] + w[:, 2:]) / w[:, 1:-1]
        s1 = np.abs(w[:, 1:] - w[:, :-1]) / np.minimum(w[:, 1:], w[:, :-1])
        qs.append((np.nanpercentile(s2, [50, 80, 90, 95, 98, 99]), np.nanpercentile(s1, [50, 80, 90, 95, 98, 99])))
    print(scene, t, len(cams), man["work_hw"])
    print("  2nd-diff rel invdepth pct50..99:", np.round(np.mean([q[0] for q in qs], 0), 4))
    print("  1st-diff rel invdepth pct50..99:", np.round(np.mean([q[1] for q in qs], 0), 4))
    # cross-camera agreement: cams[1] -> nearest cam
    cen = {c: Z[c]["c2w"][:3, 3] for c in cams}
    for a in cams[1:4]:
        b = sorted(cams, key=lambda c: np.linalg.norm(cen[c] - cen[a]))[1]
        za, zb = Z[a], Z[b]
        va = (za["conf"] > 0.1) & (za["depth"] > 0)
        p = to_reference(local_points(za["depth"], za["K"]), za["c2w"], zb["c2w"])
        K = zb["K"]; h, w = zb["depth"].shape
        z = p[..., 2]
        u = np.round(K[0, 0] * p[..., 0] / np.where(z > 0, z, 1) + K[0, 2]); vv = np.round(K[1, 1] * p[..., 1] / np.where(z > 0, z, 1) + K[1, 2])
        ok = va & (z > 0) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        uu, vq = u[ok].astype(int), vv[ok].astype(int)
        db = zb["depth"][vq, uu]; vb = (zb["conf"][vq, uu] > 0.1) & (db > 0)
        r = (z[ok][vb] - db[vb]) / db[vb]
        print(f"  {a}->{b}: in-image {ok.mean():.3f}, rel depth diff pct: ", np.round(np.percentile(np.abs(r), [10, 25, 50, 75, 90]), 4))

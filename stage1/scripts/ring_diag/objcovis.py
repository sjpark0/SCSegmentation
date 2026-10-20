"""Read-only, uses only the INPUT (reference mask) + Pi3: per (target camera, object) fraction of the object's
reference-visible surface that is visible (in image, unoccluded within tol) from the target camera.  Descriptive
comparison with the frame-900 J of the learned stage 1 (lat) on T06 — not a tuned rule."""
import json, os, sys
import numpy as np
from PIL import Image
REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points
D = os.path.join(REPO, "stage1/Data/T06_flag_ganggang_F")
C = os.path.join(D, "Cache/pi3")
ref = "cam_06"
zr = np.load(f"{C}/{ref}.npz"); d_r, K_r, T_r = zr["depth"], zr["K"], zr["c2w"]
h, w = d_r.shape
valid_r = (zr["conf"] > CONF_MIN) & (d_r > 0)
Pr = local_points(d_r, K_r)
lab = np.array(Image.fromarray(np.array(Image.open(f"{D}/masks/{ref}.png"))).resize((w, h), Image.NEAREST))
lat = {r['camera']: r['result']['J_all'] for r in json.load(open(f"{REPO}/plenoptic/full/jf_lat.json"))}
sub = {r['camera']: r['result']['J_all'] for r in json.load(open(f"{REPO}/plenoptic/full/jf_sub.json"))}
gt_ids = {r['camera']: r['objects'] for r in json.load(open(f"{REPO}/plenoptic/full/jf_lat.json"))}
rows = []
for i in range(32):
    c = f"cam_{i:02d}"
    if c == ref: continue
    z = np.load(f"{C}/{c}.npz"); d_c, K_c, T_c = z["depth"], z["K"], z["c2w"]
    M = np.linalg.inv(T_c) @ T_r
    for k, o in enumerate(gt_ids[c]):
        sel = (lab == o) & valid_r
        X = Pr[sel] @ M[:3, :3].T + M[:3, 3]
        Z = X[:, 2]; Zs = np.where(np.abs(Z) < 1e-9, 1e-9, Z)
        u = K_c[0, 0] * X[:, 0] / Zs + K_c[0, 2]; v = K_c[1, 1] * X[:, 1] / Zs + K_c[1, 2]
        ins = (Z > 0) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5)
        dd = d_c[np.clip(np.round(v), 0, h - 1).astype(int), np.clip(np.round(u), 0, w - 1).astype(int)]
        vis = ins & ((dd <= 0) | (Z <= dd * 1.05))
        rows.append(dict(cam=c, o=o, objcovis=float(vis.mean()), Jlat=lat[c][k], Jsub=sub[c][k]))
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "objcovis.json"), "w"), indent=0)
oc = np.array([r["objcovis"] for r in rows]); jl = np.array([r["Jlat"] for r in rows]); js = np.array([r["Jsub"] for r in rows])
fail = jl < 0.5
# AUC of (1 - objcovis) for predicting lat failure
pos, neg = oc[fail], oc[~fail]
auc = np.mean([[1.0 if p < n else 0.5 if p == n else 0.0 for n in neg] for p in pos])
print(f"pairs {len(rows)}, lat failures {fail.sum()}, AUC(low obj co-visibility -> failure) {auc:.3f}")
for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.01)):
    s = (oc >= lo) & (oc < hi)
    if s.any():
        print(f"objcovis [{lo:.2f},{hi:.2f}): n {s.sum():3d}  lat J {jl[s].mean():.3f} (fail {np.sum(jl[s]<0.5)})  sub J {js[s].mean():.3f}")

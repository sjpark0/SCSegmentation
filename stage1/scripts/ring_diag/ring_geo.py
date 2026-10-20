"""Read-only diagnostic: is the far-ring identity swap explained by Pi3 geometry?

For T06_flag_ganggang_F frame 900 (32 cams, GT for all cams), lift each camera's GT person pixels with that
camera's own Pi3 depth into the reference (cam_06) frame (scale-normalised exactly as geomem.pointmap does),
and compare per-person 3D centroids with the reference view's centroids.  Also ring angles of Pi3 centres.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points, scene_scale, to_reference  # noqa: E402

CACHE = os.path.join(REPO, "stage1/Data/T06_flag_ganggang_F/Cache/pi3")
MASK = os.path.join(REPO, "Data/PlenopticFull/T06_flag_ganggang_F/Mask")
REF = "cam_06"
cams = [f"cam_{i:02d}" for i in range(32)]


def load(c):
    z = np.load(os.path.join(CACHE, f"{c}.npz"))
    return z["depth"], z["conf"], z["K"], z["c2w"]


d_r, c_r, K_r, T_r = load(REF)
valid_r = (c_r > CONF_MIN) & (d_r > 0)
p_ref = to_reference(local_points(d_r, K_r), T_r, T_r)
scale = scene_scale(p_ref, valid_r)

# ring geometry of Pi3 centres (in the reference frame)
C = {}
for c in cams:
    _, _, _, T = load(c)
    C[c] = (np.linalg.inv(T_r) @ T)[:3, 3] / scale
P = np.stack([C[c] for c in cams])
mu = P.mean(0)
U, S, Vt = np.linalg.svd(P - mu)
xy = (P - mu) @ Vt[:2].T
ang = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
order = np.argsort(ang)
ring = {}
print(f"scene scale (median ref depth, Pi3 units) {scale:.4f}; plane fit singular values {S.round(3)}")
print("ring angle by camera (deg), radius (normalised):")
for i, c in enumerate(cams):
    ring[c] = dict(angle=float(ang[i]), radius=float(np.linalg.norm(xy[i])), dist_to_ref=float(np.linalg.norm(P[i] - C[REF])))
print(" ".join(f"{c[4:]}:{ring[c]['angle']:.0f}" for c in cams))
srt = [cams[i] for i in order]
gaps = [(srt[i], srt[(i + 1) % 32], (ang[order[(i + 1) % 32]] - ang[order[i]]) % 360) for i in range(32)]
print("angular order:", " ".join(s[4:] for s in srt))
print("gaps > 20 deg:", [(a, b, round(g, 1)) for a, b, g in gaps if g > 20])
print("steps min/max:", round(min(g for _, _, g in gaps), 1), round(max(g for _, _, g in gaps), 1))

# per-camera, per-person 3D centroids (ref frame, normalised)
out = {}
for c in cams:
    d, cf, K, T = load(c)
    valid = (cf > CONF_MIN) & (d > 0)
    pts = to_reference(local_points(d, K), T, T_r) / scale
    lab = np.array(Image.open(os.path.join(MASK, c, "000900.png")))
    lab = np.array(Image.fromarray(lab).resize((d.shape[1], d.shape[0]), Image.NEAREST))
    rec = {}
    for o in range(1, 7):
        sel = (lab == o) & valid
        if sel.sum() < 20:
            rec[o] = None
            continue
        q = pts[sel]
        rec[o] = dict(n=int(sel.sum()), centroid=np.median(q, 0).tolist(), conf_mean=float(cf[lab == o].mean()))
    out[c] = rec

refc = {o: np.array(out[REF][o]["centroid"]) for o in range(1, 7) if out[REF][o]}
print("\nreference person centroids (normalised):", {o: v.round(3).tolist() for o, v in refc.items()})
print("pairwise ref person distances:")
for a in refc:
    print(a, " ".join(f"{np.linalg.norm(refc[a] - refc[b]):.3f}" for b in refc))
print("\ncam  d  | per person: nearest ref person (dist to own, dist to nearest)  conf")
swaps = 0
summary = {}
for c in cams:
    i = int(c[4:])
    dd = min(abs(i - 6), 32 - abs(i - 6))
    row = []
    for o in range(1, 7):
        r = out[c][o]
        if r is None or o not in refc:
            row.append("   --   ")
            continue
        q = np.array(r["centroid"])
        ds = {b: float(np.linalg.norm(q - refc[b])) for b in refc}
        nb = min(ds, key=ds.get)
        if nb != o:
            swaps += 1
        row.append(f"{o}->{nb}({ds[o]:.2f}/{ds[nb]:.2f})")
        summary.setdefault(c, {})[o] = dict(nearest=nb, d_own=ds[o], d_nearest=ds[nb], conf=r["conf_mean"])
    print(c, f"{dd:2d}", " ".join(row))
print("persons whose GT-lifted centroid is nearer another ref person:", swaps)
json.dump(dict(ring=ring, centroids=summary, scale=scale), open(os.path.join(os.path.dirname(__file__), "ring_geo.json"), "w"), indent=1)

"""Read-only: which GT person does each frame-900 predicted mask cover, and does an image-position prior explain it?

Inputs: GT label maps Data/PlenopticFull/T06_flag_ganggang_F/Mask/cam_KK/000900.png; predictions (NAS) <cfg>/cam_KK/900/<p>.png.
"""
import json
import os

import numpy as np
from PIL import Image

REPO = "/home/sjpark/Documents/SCSegmentation"
MASK = os.path.join(REPO, "Data/PlenopticFull/T06_flag_ganggang_F/Mask")
RES = os.path.join(REPO, "PlenopticServer1/CROOM/SegResult/T06_flag_ganggang_F")
CFG = {"sub": "SegMaskSam3XW1CGPS4MFsCs10kG2", "lat": "SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs"}
REF = "cam_06"
cams = [f"cam_{i:02d}" for i in range(32)]
S = 4  # analyse at 1/4 resolution (540x960)


def lab(c):
    m = Image.open(os.path.join(MASK, c, "000900.png"))
    return np.array(m.resize((m.width // S, m.height // S), Image.NEAREST))


def pred(cfg, c, p):
    f = os.path.join(RES, CFG[cfg], c, "900", f"{p}.png")
    if not os.path.isfile(f):
        return None
    m = Image.open(f)
    return np.array(m.resize((m.width // S, m.height // S), Image.NEAREST)) > 127


def centroid(m):
    ys, xs = np.nonzero(m)
    return (xs.mean() / m.shape[1], ys.mean() / m.shape[0]) if xs.size else None


L = {c: lab(c) for c in cams}
refxy = {p: centroid(L[REF] == p) for p in range(1, 7)}
print("ref image centroids x:", {p: round(v[0], 3) for p, v in refxy.items()})
rows = []
for c in cams:
    if c == REF:
        continue
    i = int(c[4:])
    d = min(abs(i - 6), 32 - abs(i - 6))
    gxy = {p: centroid(L[c] == p) for p in range(1, 7)}
    for cfg in CFG:
        for p in range(1, 7):
            pm = pred(cfg, c, p)
            gt = L[c] == p
            if gt.sum() == 0:
                continue
            if pm is None or pm.sum() == 0:
                cov, frac = 0, 0.0
            else:
                counts = [(pm & (L[c] == j)).sum() for j in range(7)]
                cov = int(np.argmax(counts[1:]) + 1) if sum(counts[1:]) else 0
                frac = counts[cov] / pm.sum() if cov else 0.0
            # image-position prior: the far-view person whose x-centroid is nearest the reference x-centroid of p
            cand = {j: abs(gxy[j][0] - refxy[p][0]) for j in gxy if gxy[j] is not None}
            near_x = min(cand, key=cand.get)
            mir = {j: abs(gxy[j][0] - (1 - refxy[p][0])) for j in gxy if gxy[j] is not None}
            near_mx = min(mir, key=mir.get)
            J = (pm & gt).sum() / max((pm | gt).sum(), 1) if pm is not None else 0.0
            rows.append(dict(cam=c, d=d, cfg=cfg, p=p, J=float(J), covers=cov, cover_frac=float(frac),
                             near_x=near_x, near_mirror_x=near_mx, gt_px=int(gt.sum())))

json.dump(rows, open(os.path.join(os.path.dirname(__file__), "swap_diag.json"), "w"), indent=0)
for cfg in CFG:
    R = [r for r in rows if r["cfg"] == cfg]
    for lo, hi, name in ((1, 6, "near d1-6"), (7, 16, "far d7-16")):
        Rr = [r for r in R if lo <= r["d"] <= hi]
        wrong = [r for r in Rr if r["covers"] not in (0, r["p"])]
        empty = [r for r in Rr if r["covers"] == 0]
        print(f"{cfg} {name}: pairs {len(Rr)}, meanJ {np.mean([r['J'] for r in Rr]):.3f}, "
              f"covers other person {len(wrong)}, empty {len(empty)}")
        if wrong:
            ex = sum(r["covers"] == r["near_x"] for r in wrong)
            em = sum(r["covers"] == r["near_mirror_x"] for r in wrong)
            print(f"   of the {len(wrong)} wrong-person masks: covers the x-nearest (unmirrored) person {ex}, mirrored-x-nearest {em}")
            from collections import Counter
            print("   (p -> covered) counts:", Counter((r["p"], r["covers"]) for r in wrong).most_common(8))
print("\nfar-view J<0.5 rows (lat):")
for r in rows:
    if r["cfg"] == "lat" and r["J"] < 0.5:
        print(f"  {r['cam']} d{r['d']:2d} p{r['p']} J {r['J']:.2f} covers {r['covers']} ({r['cover_frac']:.2f}) x-near {r['near_x']} mirror-x-near {r['near_mirror_x']} gt_px(1/16) {r['gt_px']}")

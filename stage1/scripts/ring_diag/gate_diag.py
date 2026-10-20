"""Read-only: how discriminative is the `reproj` gate (THETA 0.5, MIN_VALID 64, OCC_TOL 0.01) around the T06 ring?

For each non-reference camera and person o, using GT masks in place of predictions:
  correct  = overlap the gate would compute if the mask for o were exactly GT_o of that camera
  wrong_j  = overlap the gate would compute if the mask for o covered person j's GT instead (identity swap)
Work resolution (Pi3 378x672); n_valid is reported scaled to full-res pixels as run.py counts them.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = "/home/sjpark/Documents/SCSegmentation"
sys.path.insert(0, os.path.join(REPO, "stage1"))
from geomem.pointmap import CONF_MIN, local_points, to_reference  # noqa: E402

CACHE = os.path.join(REPO, "stage1/Data/T06_flag_ganggang_F/Cache/pi3")
MASK = os.path.join(REPO, "Data/PlenopticFull/T06_flag_ganggang_F/Mask")
REF = "cam_06"
OCC_TOL, THETA, MIN_VALID = 0.01, 0.5, 64
cams = [f"cam_{i:02d}" for i in range(32)]


def load(c):
    z = np.load(os.path.join(CACHE, f"{c}.npz"))
    return z["depth"], z["conf"], z["K"], z["c2w"]


d_r, c_r, K_r, T_r = load(REF)
h, w = d_r.shape
H, W = 2160, 3840
px_scale = (H / h) * (W / w)
L_ref = np.array(Image.open(os.path.join(MASK, REF, "000900.png")))
rows = []
for c in cams:
    if c == REF:
        continue
    i = int(c[4:])
    dist = min(abs(i - 6), 32 - abs(i - 6))
    d, cf, K, T = load(c)
    valid_u = (cf > CONF_MIN) & (d > 0)
    p = to_reference(local_points(d, K), T, T_r)
    z = p[..., 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K_r[0, 0] * p[..., 0] / zs + K_r[0, 2]
    v = K_r[1, 1] * p[..., 1] / zs + K_r[1, 2]
    inside = (z > 0) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5)
    col = np.clip(np.round(u), 0, w - 1).astype(int)
    row = np.clip(np.round(v), 0, h - 1).astype(int)
    dref = d_r[row, col]
    unocc = (dref <= 0) | (z <= dref * (1 + OCC_TOL))
    ok = inside & unocc & valid_u
    # reference GT label at the projected location (full-res nearest)
    U = np.clip(np.round((u + 0.5) * W / w - 0.5), 0, W - 1).astype(int)
    V = np.clip(np.round((v + 0.5) * H / h - 0.5), 0, H - 1).astype(int)
    lab_at = L_ref[V, U]
    L = np.array(Image.fromarray(np.array(Image.open(os.path.join(MASK, c, "000900.png")))).resize((w, h), Image.NEAREST))
    for o in range(1, 7):
        gt_o = (L == o) & valid_u
        if gt_o.sum() == 0 or not (L_ref == o).any():
            continue
        frac_inside = float((gt_o & inside).sum() / gt_o.sum())
        frac_ok = float((gt_o & ok).sum() / gt_o.sum())

        def overlap(mask_j):
            sel = mask_j & ok
            n = sel.sum() * px_scale
            if n < MIN_VALID:
                return float("nan"), float(n)
            return float((sel & (lab_at == o)).sum() / sel.sum()), float(n)

        corr, n_corr = overlap(gt_o)
        wrong = {}
        for j in range(1, 7):
            if j != o and ((L == j) & valid_u).sum() > 0:
                wrong[j] = overlap((L == j) & valid_u)[0]
        rows.append(dict(cam=c, d=dist, o=o, frac_inside=frac_inside, frac_unoccluded=frac_ok, n_valid_full=n_corr,
                         correct=corr, wrong=wrong))

json.dump(rows, open(os.path.join(os.path.dirname(__file__), "gate_diag.json"), "w"), indent=0)


def summ(R, name):
    corr = np.array([r["correct"] for r in R])
    nan = np.isnan(corr)
    passed = (~nan) & (corr >= THETA)
    wr = [x for r in R for x in r["wrong"].values()]
    wr = np.array(wr)
    wnan = np.isnan(wr)
    wpass = (~wnan) & (wr >= THETA)
    print(f"{name}: persons {len(R)}; mean frac of GT px visible-from-ref {np.mean([r['frac_unoccluded'] for r in R]):.2f} "
          f"(inside ref image {np.mean([r['frac_inside'] for r in R]):.2f}); "
          f"correct mask -> gate pass {passed.mean():.2f}, reject {((~nan) & (corr < THETA)).mean():.2f}, NaN->IoU rule {nan.mean():.2f}; "
          f"swapped mask -> pass {wpass.mean():.2f}, reject {((~wnan) & (wr < THETA)).mean():.2f}, NaN {wnan.mean():.2f}")


for lo, hi in ((1, 3), (4, 6), (7, 9), (10, 12), (13, 16)):
    summ([r for r in rows if lo <= r["d"] <= hi], f"d {lo:2d}-{hi:2d}")
print("\nper-camera median overlap of the correct mask / frac unoccluded:")
for c in cams:
    R = [r for r in rows if r["cam"] == c]
    if R:
        print(c, R[0]["d"], np.round(np.nanmedian([r["correct"] for r in R]) if not all(np.isnan([r['correct'] for r in R])) else np.nan, 2),
              np.round(np.mean([r["frac_unoccluded"] for r in R]), 2))

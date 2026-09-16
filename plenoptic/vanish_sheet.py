#!/usr/bin/env python3
"""Contact sheet of the (camera, person) pairs whose control mask is below 10 % of its seed area at t = 100
(score.json Q6_end_vanished): columns t = 0, first vanishing frame, t = 100; fill = control mask, for a by-eye
classification (left the field of view / occluded / tracker loss).  -> plenoptic/vanish_sheet.png"""
import json, os
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import areas, CTRL, HEAD, NFRAME, VANISH, SHARE
from make_figs import frame, pred, COLS
Ac, Ah = areas(CTRL), areas(HEAD)
pairs = sorted(ko for ko in Ac if Ac[ko][0] and Ac[ko][-1] < VANISH * Ac[ko][0])
rows = []
for k, o in pairs:
    lost = np.nonzero(Ac[(k, o)] < VANISH * Ac[(k, o)][0])[0]; t1 = int(lost[lost > 0][0]) if (lost > 0).any() else NFRAME - 1
    rows.append((k, o, t1, int(Ah[(k, o)][-1] < VANISH * Ah[(k, o)][0])))
def crop_box(cam, o, t_last, margin=1.0):
    m = pred(CTRL, cam, t_last, o)
    if m is None or not m.any(): return None
    yy, xx = np.nonzero(m); h, w = yy.max() - yy.min(), xx.max() - xx.min(); H, W = m.shape
    return (max(int(xx.min() - margin * w), 0), max(int(yy.min() - margin * h), 0), min(int(xx.max() + margin * w), W), min(int(yy.max() + margin * h), H))
fig, axs = plt.subplots(len(rows), 4, figsize=(10, 2.0 * len(rows)))
for r, (k, o, t1, hv) in enumerate(rows):
    cam = f"cam_{k:02d}"; box = crop_box(cam, o, max(t1 - 1, 0))
    for c, t in enumerate((max(t1 - 3, 0), t1, min(t1 + 6, NFRAME - 1), NFRAME - 1)):
        img = frame(cam, t).astype(float); m = pred(CTRL, cam, t, o)
        if m is not None and m.any(): img[m] = 0.4 * img[m] + 0.6 * COLS[o]
        img = img.astype(np.uint8)
        if box: img = img[box[1]:box[3], box[0]:box[2]]
        ax = axs[r, c]; ax.imshow(img); ax.axis("off")
        if c == 0: ax.set_title(f"{cam} p{o}  vanish@{t1}  headline_vanished={hv}   t={t}", fontsize=8, loc="left")
        else: ax.set_title(f"t={t}", fontsize=7)
plt.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vanish_sheet.png"), dpi=70); print("\n".join(f"cam_{k:02d} p{o} vanish@{t1} headline_vanished={hv}" for k, o, t1, hv in rows))

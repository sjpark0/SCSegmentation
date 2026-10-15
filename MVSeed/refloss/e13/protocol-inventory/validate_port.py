#!/usr/bin/env python3
"""Port check: dev_reach.py class counts (numpy) vs the torch rows of the E12 design measurement
(scratchpad/visibility-signal/train_vis.json, methods.geomem.run machinery on GPU) on the rows both have:
reference = c_ini, t != ref_frame (DEV), stride-2 frames.  Only DEV rows are compared."""
import json, sys
import numpy as np
SP = "/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad"
mine = json.load(open(sys.argv[1]))["rows"]
txt = open(f"{SP}/visibility-signal/train_vis.json").read()
tv = json.loads(txt[txt.index("{"):])["rows"]
meta = {}
import os
for r in mine:
    meta.setdefault(r["scene"], json.load(open(f"/home/sjpark/Documents/SCSegmentation/stage1/Data/{r['scene']}/meta.json")))
idx = {(r["scene"], r["t"], r["cam"], r["obj"]): r for r in mine}
n = exact = 0
tot_m = np.zeros(4); tot_t = np.zeros(4); worst = (0, None)
for r in tv:
    m = meta.get(r["scene"])
    if m is None or r["ref"] != m["ref_cam"] or r["t"] == m["ref_frame"]:
        continue
    k = (r["scene"], r["t"], r["cam"], r["obj"])
    if k not in idx or r["gt_px"] == 0:
        continue
    a = np.array(idx[k]["n"], float)
    b = np.array([r["gt_vis"], r["gt_outside"], r["gt_occluded"], r["gt_nodepth"]], float)
    assert a.sum() == b.sum() == r["gt_px"], (k, a, b)
    n += 1
    exact += int((a == b).all())
    tot_m += a; tot_t += b
    d = np.abs(a - b).sum() / max(b.sum(), 1)
    if d > worst[0]:
        worst = (d, k)
print(f"rows compared {n}, identical class counts {exact} ({exact / max(n, 1):.3f})")
print("pixel-weighted class shares numpy:", np.round(tot_m / tot_m.sum(), 5), " torch:", np.round(tot_t / tot_t.sum(), 5))
print("max per-row L1 share difference", round(worst[0], 5), worst[1])
print("total |diff| px / total px", np.abs(tot_m - tot_t).sum() / tot_t.sum())

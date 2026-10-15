#!/usr/bin/env python3
"""S1-E14 train-band: output-space dose proxy between E9 D1 (W 0) and E9 D2 (W 1) on DEV-A (host; never the keyframe).

The two arms share the training flow (same samples per step; only the band term differs), so a band arm at an
intermediate weight is approximated, to first order, by blending their round-2 low_res logits:
    low(a) = (1 - a) * low_D1 + a * low_D2      presence(a) = (1 - a) * pres_D1 + a * pres_D2
(objects absent in exactly one arm: absent if presence(a) <= 0, else the present arm's logits -- the absent arm's raw
logits were replaced by NO_OBJ_SCORE in the dump), then the adopted path (two bilinear stages, threshold 0, xcl out).
a = 0 / 1 must reproduce the D1 / D2 PNG scores exactly (checked).  Also D1 with a global threshold shift tau
(pos = L > tau; xcl argmax unchanged), and D2 likewise: the threshold curves of both arms, to separate a
global calibration shift from a change of boundary shape.
    stage1/.venv/bin/python interp.py --out interp.json [--workers 8]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(S1, "geomem", "runs", "_e14_logitdump", "code"))
import e14dump as E  # noqa: E402
from mvseg2d.io import load_labelmap  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
TB = os.path.join(S1, "geomem", "runs", "_e14_train-band")
R1, R2 = os.path.join(TB, "geoset_dec_rep_x_A"), os.path.join(TB, "geoset_decb_rep_x_A")
PAIRS = os.path.join(S1, "geomem", "runs", "_e13_dev", "dev_pairs.json")
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
TAUS = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def xcl_tau(L, tau):
    """run.exclusive_masks with pos = L > tau (tau 0 = adopted)."""
    B = L.shape[0]
    pos = L > tau
    w = L.argmax(0)
    win = w[None] == np.arange(B)[:, None, None]
    npos = pos.reshape(B, -1).sum(1)
    lost = (pos & ~win).reshape(B, -1).sum(1)
    exempt = (npos > 0) & (lost.astype(np.float32) >= np.float32(E.XCL_TAU_KEEP) * npos.astype(np.float32))
    return pos & (win | exempt[:, None, None])


def score(masks, objects, lab, rows):
    H, W = lab.shape
    bp = bound_pix(H, W)
    se = disk(bp)
    idx = {int(o): i for i, o in enumerate(objects)}
    out = []
    for r in rows:
        o = int(r["obj"])
        G = lab == o
        S = masks[idx[o]]
        ys, xs = np.nonzero(G | S)
        M = bp + 2
        y0, y1 = max(int(ys.min()) - M, 0), min(int(ys.max()) + M + 1, H)
        x0, x1 = max(int(xs.min()) - M, 0), min(int(xs.max()) + M + 1, W)
        g, s = G[y0:y1, x0:x1], S[y0:y1, x0:x1]
        J, F = jf_pair(g, s, bp, se)
        tp = int((g & s).sum())
        out.append((float(J), float(F), tp / max(int(g.sum()), 1), (tp / int(s.sum())) if s.any() else 0.0))
    return out


def view_job(args):
    scene, t, cam, rows = args
    z1, z2 = E.load(R1, scene, t, cam), E.load(R2, scene, t, cam)
    assert list(z1["objects"]) == list(z2["objects"])
    hw = tuple(int(v) for v in z1["hw"])
    lab = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    l1, l2 = z1["low_res"].astype(np.float32), z2["low_res"].astype(np.float32)
    p1, p2 = z1["presence"] > 0, z2["presence"] > 0
    res = {}
    for a in ALPHAS:
        pa = (1 - a) * z1["presence"] + a * z2["presence"]
        low = np.where((p1 & p2)[:, None, None], (1 - a) * l1 + a * l2, np.where(p1[:, None, None], l1, l2))
        low = np.where((pa > 0)[:, None, None] if 0 < a < 1 else (p1 if a == 0 else p2)[:, None, None], low, E.NO_OBJ_SCORE)
        low = low.astype(np.float32)
        m = E.masks_numpy(low, hw, xcl=True, fma=True)
        res[f"a{a}"] = score(m, z1["objects"], lab, rows)
    for nm, l in (("D1", l1), ("D2", l2)):
        L = E.logits_full_np(l, hw, fma=True)
        for tau in TAUS:
            res[f"{nm}_t{tau}"] = score(xcl_tau(L, np.float32(tau)), z1["objects"], lab, rows)
    return [(r["scene"], r["t"], r["cam"], r["obj"]) for r in rows], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = json.load(open(PAIRS))["rows"]
    by = defaultdict(list)
    for r in rows:
        by[(r["scene"], int(r["t"]), r["cam"])].append(r)
    jobs = [(s, t, c, rs) for (s, t, c), rs in sorted(by.items())]
    keys, per = [], defaultdict(list)
    with Pool(a.workers) as pool:
        for ks, res in pool.imap(view_job, jobs, chunksize=1):
            keys += ks
            for v, sc in res.items():
                per[v] += sc
    out = dict(keys=keys, variants={v: [list(x) for x in sc] for v, sc in per.items()})
    json.dump(out, open(a.out, "w"))
    for v, sc in per.items():
        arr = np.array(sc)
        print(v, "J&F %.5f J %.5f F %.5f" % (arr[:, :2].mean(1).mean(), arr[:, 0].mean(), arr[:, 1].mean()))


if __name__ == "__main__":
    main()

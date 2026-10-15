#!/usr/bin/env python3
"""S1-E14 train-band: host scoring of a dev_decode-layout mask root on the DEV-A pair list (never the keyframe).

Pairs = rows of <pairs> (default _e13_dev/dev_pairs.json, the adopted DEV-A scoring; the pair list depends on the GT
only).  Per pair: J, F, JF (mvseg2d.metrics.jf_pair, as dev_score), recall, precision, the +-2 px band oracle (seedfail
definition: band = dilate(g, ellipse 5) & ~erode(g, same); pred replaced by GT inside it), FN / FP inside that band;
`kind` (seedcat kind) and `view` copied from the base row so mvseg2d.dev_compare can read the output directly.
    stage1/.venv/bin/python score_arm.py --masks <root> --out <json> [--pairs ...] [--check]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from mvseg2d.io import load_labelmap, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
PAIRS = os.path.join(S1, "geomem", "runs", "_e13_dev", "dev_pairs.json")
K2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def view_job(args):
    masks, scene, t, cam, rows = args
    gt_lab = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    H, W = gt_lab.shape
    bp = bound_pix(H, W)
    se = disk(bp)
    M = bp + 8
    out = []
    for row in rows:
        o = int(row["obj"])
        G = gt_lab == o
        p = os.path.join(masks, scene, f"{t:06d}", cam, f"{o}.png")
        S = read_pred_mask(p) if os.path.isfile(p) else np.zeros((H, W), bool)
        ys, xs = np.nonzero(G | S)
        y0, y1 = max(int(ys.min()) - M, 0), min(int(ys.max()) + M + 1, H)
        x0, x1 = max(int(xs.min()) - M, 0), min(int(xs.max()) + M + 1, W)
        g, s = G[y0:y1, x0:x1], S[y0:y1, x0:x1]
        J, F = jf_pair(g, s, bp, se)
        g8 = g.astype(np.uint8)
        b = (cv2.dilate(g8, K2) > 0) & ~(cv2.erode(g8, K2) > 0)
        Jo, Fo = jf_pair(g, np.where(b, g, s), bp, se)
        tp = int((g & s).sum())
        out.append(dict(scene=scene, t=t, cam=cam, obj=o, view=row.get("view", f"{t}/{cam}"), kind=row.get("kind"),
                        label=row.get("label"), J=float(J), F=float(F), JF=float((J + F) / 2), JF_base=row["JF"],
                        gt_px=int(g.sum()), seed_px=int(s.sum()), recall=tp / max(int(g.sum()), 1),
                        precision=(tp / int(s.sum())) if s.any() else None, or2=float((Jo + Fo) / 2),
                        fn_in2=int((g & ~s & b).sum()), fp_in2=int((s & ~g & b).sum()), band2=int(b.sum()),
                        fn=int((g & ~s).sum()), fp=int((s & ~g).sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true", help="assert JF == the pairs file's JF (use on the base root)")
    a = ap.parse_args()
    rows = json.load(open(a.pairs))["rows"]
    by = defaultdict(list)
    for r in rows:
        by[(r["scene"], int(r["t"]), r["cam"])].append(r)
    jobs = [(a.masks, s, t, c, rs) for (s, t, c), rs in sorted(by.items())]
    with Pool(a.workers) as pool:
        res = [x for part in pool.map(view_job, jobs, chunksize=1) for x in part]
    if a.check:
        d = max(abs(r["JF"] - r["JF_base"]) for r in res)
        print("max |JF - pairs JF| =", d)
        assert d < 1e-12
    json.dump(dict(masks=a.masks, pairs=a.pairs, rows=res), open(a.out, "w"))
    jf = np.array([r["JF"] for r in res])
    print(f"{len(res)} pairs  J&F {jf.mean():.5f}  J {np.mean([r['J'] for r in res]):.5f}  F {np.mean([r['F'] for r in res]):.5f}")


if __name__ == "__main__":
    main()

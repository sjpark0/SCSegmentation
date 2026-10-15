#!/usr/bin/env python3
"""S1-E14 boundary anatomy, partition view (host; DEV-A only, never the keyframe).

The DEV GT label maps are a dense partition (label 0 is essentially absent), and every seeded object (basic set,
present in c_ini at t) is decoded in every eval view.  So per view (scene, t, cam) the predicted masks of all seeded
objects form a (partial) partition, and each error pixel of a pair has an owner on the other side:

  FN of o  (g_o & ~s_o):  "unclaimed"   no seeded mask claims the pixel (a gap in the predicted partition)
                          "neighbour"   another seeded object's mask claims it (that neighbour's FP)
  FP of o  (s_o & ~g_o):  "on seeded"   the pixel's GT owner is another seeded object (its mask may or may not claim it too)
                          "on unseeded" the GT owner is a labelled object that was not decoded (non-basic / absent from c_ini)
split inside / outside the +-2 px GT band (seedfail definition) and, for unclaimed FN, by the gap width (distance of
the pixel to the nearest claimed pixel, cv2 DIST_L2 mask 5).

GT-free probe ("fill"): every unclaimed pixel within k px of a claimed pixel is given to the mask of the nearest claimed
pixel (cv2.distanceTransformWithLabels; a multiply-claimed nearest pixel -> its lowest-id claimer); k = 1, 2, 4, 8, inf.
Ceiling ("oracle fill"): unclaimed pixels given to their GT owner when it is seeded (k = 2 and inf).
Per pair J&F of each variant; per view the pixels the fill adds to masks whose GT is empty (unscored hallucination growth).

    stage1/.venv/bin/python bd_partition.py --masks <root> --pairs <dev_pairs.json> --out <json>
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
from mvseg2d.io import load_labelmap, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
KS = (1, 2, 4, 8, None)
GAPB = (1, 2, 4, 8)          # unclaimed-FN gap width bins: <=1, (1,2], (2,4], (4,8], >8


def ell(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def jf(g, p, bp, se):
    J, F = jf_pair(g, p, bp, se)
    return float((J + F) / 2)


def view_job(args):
    masks, scene, t, cam, ref_cam, basic, rows = args
    fdir = os.path.join(DATA, scene, "frames", f"{t:06d}", "masks")
    gt_lab = load_labelmap(os.path.join(fdir, f"{cam}.png"))
    ref_lab = load_labelmap(os.path.join(fdir, f"{ref_cam}.png"))
    seeded = [o for o in basic if (ref_lab == o).any()]
    cfg = json.load(open(os.path.join(masks, scene, f"{t:06d}", "config.json")))
    assert sorted(cfg["objects"]) == sorted(seeded), (scene, t, cfg["objects"], seeded)
    H, W = gt_lab.shape
    bp = bound_pix(H, W)
    se = disk(bp)
    S = {}
    C = np.zeros((H, W), np.uint8)
    owner = np.full((H, W), -1, np.int32)
    for o in seeded:
        p = os.path.join(masks, scene, f"{t:06d}", cam, f"{o}.png")
        if os.path.isfile(p):
            m = read_pred_mask(p)
            S[o] = m
            C += m
            owner[m & (owner < 0)] = o
    gseed = np.isin(gt_lab, seeded)
    unc = C == 0
    # nearest claimed pixel for every unclaimed pixel
    if unc.all():
        dist = np.full((H, W), np.inf, np.float32)
        near = np.full((H, W), -1, np.int32)
    else:
        dist, lab_img = cv2.distanceTransformWithLabels(unc.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.full(int(lab_img.max()) + 1, -1, np.int32)
        lut[lab_img[~unc]] = owner[~unc]
        near = np.where(unc, lut[lab_img], owner)
    v = dict(scene=scene, t=t, cam=cam, H=H, W=W, n_seeded=len(seeded), px=H * W, unclaimed=int(unc.sum()),
             unclaimed_gt_seeded=int((unc & gseed).sum()), unclaimed_gt_unseeded=int((unc & ~gseed & (gt_lab != 0)).sum()),
             unclaimed_gt_0=int((unc & (gt_lab == 0)).sum()), multi=int((C >= 2).sum()),
             wrong_single_seeded=int(((C == 1) & gseed & (owner != gt_lab)).sum()),
             wrong_single_unseeded=int(((C == 1) & ~gseed).sum()),
             gt_unseeded_px=int((~gseed & (gt_lab != 0)).sum()), gt_label0_px=int((gt_lab == 0).sum()))
    lo = 0
    for i, hb in enumerate(GAPB + (None,)):
        m = unc & (dist > lo) if hb is None else unc & (dist > lo) & (dist <= hb)
        if i == 0:
            m = unc & (dist <= hb)
        tag = f"gap{lo}_{hb or 'inf'}"
        v[f"unc_{tag}"] = int(m.sum())
        v[f"unc_{tag}_seeded"] = int((m & gseed).sum())
        lo = hb if hb is not None else lo
    # fill growth of masks whose GT is empty (unscored)
    scored = {int(r["obj"]) for r in rows}
    for k in KS:
        add = unc & (dist <= (k if k is not None else np.inf))
        v[f"fill{k or 'inf'}_px"] = int(add.sum())
        v[f"fill{k or 'inf'}_correct_px"] = int((add & (near == gt_lab)).sum())
        v[f"fill{k or 'inf'}_to_unscored_px"] = int((add & ~np.isin(near, list(scored))).sum())
    out = []
    for row in rows:
        o = int(row["obj"])
        G = gt_lab == o
        s = S.get(o, np.zeros((H, W), bool))
        ys, xs = np.nonzero(G | s | (unc & (near == o)))
        M = bp + 2 + 16
        y0, y1 = max(int(ys.min()) - M, 0), min(int(ys.max()) + M + 1, H)
        x0, x1 = max(int(xs.min()) - M, 0), min(int(xs.max()) + M + 1, W)
        sl = (slice(y0, y1), slice(x0, x1))
        g, sc, u, nr, ds, lab, Cc, gs = G[sl], s[sl], unc[sl], near[sl], dist[sl], gt_lab[sl], C[sl], gseed[sl]
        k2 = ell(2)
        g8 = g.astype(np.uint8)
        band = (cv2.dilate(g8, k2) > 0) & ~(cv2.erode(g8, k2) > 0)
        fn, fp = g & ~sc, sc & ~g
        r = dict(scene=scene, t=t, cam=cam, obj=o, JF=jf(g, sc, bp, se), JF_score=row["JF"], gt_px=int(g.sum()))
        for nm, bm in (("in", band), ("out", ~band)):
            r[f"fn_unc_{nm}"] = int((fn & u & bm).sum())
            r[f"fn_nb_{nm}"] = int((fn & ~u & bm).sum())
            r[f"fp_seeded_{nm}"] = int((fp & gs & bm).sum())
            r[f"fp_seeded_both_{nm}"] = int((fp & gs & (Cc >= 2) & bm).sum())
            r[f"fp_unseeded_{nm}"] = int((fp & ~gs & bm).sum())
        lo = 0
        for i, hb in enumerate(GAPB + (None,)):
            m = fn & u & ((ds <= hb) if i == 0 else (ds > lo) if hb is None else (ds > lo) & (ds <= hb))
            r[f"fn_unc_gap{lo}_{hb or 'inf'}"] = int(m.sum())
            lo = hb if hb is not None else lo
        for k in KS:
            add = u & (ds <= (k if k is not None else np.inf)) & (nr == o)
            r[f"fill{k or 'inf'}"] = jf(g, sc | add, bp, se)
            r[f"fill{k or 'inf'}_add"] = int(add.sum())
            r[f"fill{k or 'inf'}_add_ok"] = int((add & g).sum())
        r["ofill2"] = jf(g, sc | (u & g & (ds <= 2)), bp, se)
        r["ofillinf"] = jf(g, sc | (u & g), bp, se)
        out.append(r)
    return v, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    d = json.load(open(a.pairs))
    idx = json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.pairs)), "index.json")))
    views = {}
    for r in d["rows"]:
        views.setdefault((r["scene"], r["t"], r["cam"]), []).append(r)
    jobs = [(a.masks, s, t, c, idx["scenes"][s]["ref_cam"], [int(o) for o in idx["scenes"][s]["basic"]], rows)
            for (s, t, c), rows in sorted(views.items())]
    with Pool(a.procs) as pool:
        res = list(pool.imap_unordered(view_job, jobs, chunksize=1))
    vrows = sorted([v for v, _ in res], key=lambda v: (v["scene"], v["t"], v["cam"]))
    rows = sorted([r for _, rs in res for r in rs], key=lambda r: (r["scene"], r["t"], r["cam"], r["obj"]))
    dmax = max(abs(r["JF"] - r["JF_score"]) for r in rows)
    print(f"{len(vrows)} views, {len(rows)} pairs, max |JF - JF_pairs| = {dmax:.2e}, {time.time() - t0:.0f}s", flush=True)
    assert dmax < 1e-9, dmax
    meta = dict(masks=os.path.abspath(a.masks), pairs=os.path.abspath(a.pairs), ks=[k or "inf" for k in KS], gap_bins=GAPB,
                seconds=time.time() - t0)
    json.dump(dict(meta=meta, views=vrows, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()

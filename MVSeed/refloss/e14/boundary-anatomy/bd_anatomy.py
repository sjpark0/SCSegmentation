#!/usr/bin/env python3
"""S1-E14 boundary anatomy on DEV-A (host, numpy + cv2; never the keyframe).

For every DEV-A pair of <pairs> (dev_score output; the pair list is defined by the GT, so any candidate decode in the
dev_decode layout can be scored on the same list) this recomputes J/F of the masks in <masks> and measures, per pair:

  band oracles   r = 1, 2, 4, 8: JF after replacing the prediction by the GT inside the +-r px band around the GT boundary
                 (seedfail.py definition: band = dilate(g, ellipse 2r+1) & ~erode(g, same)); for r = 1, 2, 4 also the
                 FN-only (s | g&band) and FP-only (s & ~(band & ~g)) oracles, i.e. fix only under- / only over-coverage
  pixels         FN/FP inside / outside each band; GT boundary length (seg2bmap pixels)
  neighbour      class of every pixel: outside g = its own GT label class, inside g = the class of the nearest pixel
                 outside g (cv2.distanceTransformWithLabels, DIST_L2 mask 5); classes 0 = unlabelled background
                 (label 0), 1 = another SEEDED object (basic set and present in c_ini at t = decoded in this view),
                 2 = another labelled object that was not seeded.  Two variants: rho = 0 (strict) and rho = 3 (a pixel
                 within 3 px of another object counts as that object -- tolerant to thin unlabelled annotation seams).
                 Per class: band(r=2) pixels, FN/FP in band, GT inner-boundary pixels, and the band(r=2) oracle
                 restricted to that class's part of the band.
  frame border   distance of each pixel to the nearest image edge d = min(y, x, H-1-y, W-1-x); band(r=2) pixels and
                 FN/FP by bins [0,4) [4,16) [16,64) [64,inf); the r=2 oracle restricted to d < 16 and to d >= 16;
                 errors outside the band within 4 px of the frame; GT touches the frame.

    stage1/.venv/bin/python bd_anatomy.py --masks <root> --pairs <dev_pairs.json> --out <json> [--check-jf]
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
from mvseg2d.metrics import bound_pix, disk, jf_pair, seg2bmap  # noqa: E402

DATA = os.path.join(S1, "Data")
RADII = (1, 2, 4, 8)
SIDE = (1, 2, 4)                 # FN-only / FP-only oracles
DBINS = (0, 4, 16, 64)           # frame-distance bins
NEAR = 16                        # frame-near split for the restricted oracle
RHOS = (0, 3)
CLS = ("bg", "seeded", "unseeded")


def ell(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def band_of(g8, r):
    k = ell(r)
    return (cv2.dilate(g8, k) > 0) & ~(cv2.erode(g8, k) > 0)


def classes(g, lab, o, seeded, rho):
    """Per-pixel neighbour class (0 bg, 1 seeded other, 2 unseeded labelled) of a crop; see module doc."""
    oth_s = np.isin(lab, [q for q in seeded if q != o])
    oth_u = (lab != 0) & (lab != o) & ~oth_s
    if rho > 0:
        k = ell(rho)
        oth_s = cv2.dilate(oth_s.astype(np.uint8), k) > 0
        oth_u = (cv2.dilate(oth_u.astype(np.uint8), k) > 0) & ~oth_s
    c_out = np.where(oth_s, 1, np.where(oth_u, 2, 0)).astype(np.int64)
    if (~g).sum() == 0:
        return c_out
    _, lab_img = cv2.distanceTransformWithLabels(g.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(lab_img.max()) + 1, np.int64)
    lut[lab_img[~g]] = c_out[~g]
    return np.where(g, lut[lab_img], c_out)


def jfs(g, p, bp, se):
    J, F = jf_pair(g, p, bp, se)
    return float(J), float(F)


def view_job(args):
    masks, scene, t, cam, ref_cam, basic, rows = args
    fdir = os.path.join(DATA, scene, "frames", f"{t:06d}", "masks")
    gt_lab = load_labelmap(os.path.join(fdir, f"{cam}.png"))
    ref_lab = load_labelmap(os.path.join(fdir, f"{ref_cam}.png"))
    seeded = [o for o in basic if (ref_lab == o).any()]
    H, W = gt_lab.shape
    bp = bound_pix(H, W)
    se = disk(bp)
    M = bp + 2 + 8 + 4
    out = []
    for row in rows:
        o = int(row["obj"])
        assert o in seeded, (scene, t, cam, o)
        G = gt_lab == o
        p = os.path.join(masks, scene, f"{t:06d}", cam, f"{o}.png")
        S = read_pred_mask(p) if os.path.isfile(p) else np.zeros((H, W), bool)
        assert S.shape == (H, W)
        ys, xs = np.nonzero(G | S)
        y0, y1 = max(int(ys.min()) - M, 0), min(int(ys.max()) + M + 1, H)
        x0, x1 = max(int(xs.min()) - M, 0), min(int(xs.max()) + M + 1, W)
        g, s, lab = G[y0:y1, x0:x1], S[y0:y1, x0:x1], gt_lab[y0:y1, x0:x1]
        g8 = g.astype(np.uint8)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.minimum(np.minimum(yy, xx), np.minimum(H - 1 - yy, W - 1 - xx))
        fn, fp = g & ~s, s & ~g
        J, F = jfs(g, s, bp, se)
        r = dict(scene=scene, t=t, cam=cam, obj=o, view=f"{t}/{cam}", H=H, W=W, bp=bp, J=J, F=F, JF=(J + F) / 2,
                 JF_score=row["JF"], gt_px=int(g.sum()), seed_px=int(s.sum()), fn=int(fn.sum()), fp=int(fp.sum()),
                 blen=int(seg2bmap(g).sum()), touches=bool(G[0].any() or G[-1].any() or G[:, 0].any() or G[:, -1].any()),
                 n_seeded=len(seeded), label=row.get("label"), kind=row.get("kind"))
        bands = {}
        for rad in RADII:
            b = band_of(g8, rad)
            bands[rad] = b
            Jo, Fo = jfs(g, np.where(b, g, s), bp, se)
            r[f"or{rad}"], r[f"orJ{rad}"], r[f"orF{rad}"] = (Jo + Fo) / 2, Jo, Fo
            r[f"band{rad}"] = int(b.sum())
            r[f"fn_in{rad}"], r[f"fp_in{rad}"] = int((fn & b).sum()), int((fp & b).sum())
            if rad in SIDE:
                Ja, Fa = jfs(g, s | (g & b), bp, se)
                Jb, Fb = jfs(g, s & ~(b & ~g), bp, se)
                r[f"orFN{rad}"], r[f"orFP{rad}"] = (Ja + Fa) / 2, (Jb + Fb) / 2
                r[f"orFN_J{rad}"], r[f"orFN_F{rad}"], r[f"orFP_J{rad}"], r[f"orFP_F{rad}"] = Ja, Fa, Jb, Fb
        b2 = bands[2]
        inner = g & ~(cv2.erode(g8, ell(1)) > 0)
        for rho in RHOS:
            c = classes(g, lab, o, seeded, rho)
            for k, name in enumerate(CLS):
                m = c == k
                tag = f"{name}_r{rho}"
                r[f"b2_{tag}"] = int((b2 & m).sum())
                r[f"fn2_{tag}"] = int((fn & b2 & m).sum())
                r[f"fp2_{tag}"] = int((fp & b2 & m).sum())
                r[f"bnd_{tag}"] = int((inner & m).sum())
                r[f"fpout2_{tag}"] = int((fp & ~b2 & m).sum())
                bm = b2 & m
                if bm.any() and (fn | fp)[bm].any():
                    Jo, Fo = jfs(g, np.where(bm, g, s), bp, se)
                    r[f"or2_{tag}"] = (Jo + Fo) / 2
                else:
                    r[f"or2_{tag}"] = r["JF"]
        for i, lo in enumerate(DBINS):
            hi = DBINS[i + 1] if i + 1 < len(DBINS) else 10 ** 9
            m = (dist >= lo) & (dist < hi)
            tag = f"d{lo}"
            r[f"b2_{tag}"] = int((b2 & m).sum())
            r[f"fn2_{tag}"] = int((fn & b2 & m).sum())
            r[f"fp2_{tag}"] = int((fp & b2 & m).sum())
            r[f"bnd_{tag}"] = int((inner & m).sum())
        for name, m in (("near", dist < NEAR), ("far", dist >= NEAR)):
            bm = b2 & m
            if bm.any() and (fn | fp)[bm].any():
                Jo, Fo = jfs(g, np.where(bm, g, s), bp, se)
                r[f"or2_{name}"] = (Jo + Fo) / 2
            else:
                r[f"or2_{name}"] = r["JF"]
        edge = dist < 4
        r["fn_out2_edge4"], r["fp_out2_edge4"] = int((fn & ~b2 & edge).sum()), int((fp & ~b2 & edge).sum())
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--pairs", required=True, help="dev_score json whose rows define the pair list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-jf", action="store_true", help="assert the recomputed JF equals the pairs file's JF")
    ap.add_argument("--procs", type=int, default=24)
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
        rows = [r for rs in pool.imap_unordered(view_job, jobs, chunksize=1) for r in rs]
    rows.sort(key=lambda r: (r["scene"], r["t"], r["cam"], r["obj"]))
    dmax = max(abs(r["JF"] - r["JF_score"]) for r in rows)
    print(f"{len(rows)} pairs, max |JF - JF_pairs| = {dmax:.2e}, {time.time() - t0:.0f}s", flush=True)
    if a.check_jf:
        assert dmax < 1e-9, dmax
    meta = dict(masks=os.path.abspath(a.masks), pairs=os.path.abspath(a.pairs), radii=RADII, side=SIDE, dbins=DBINS,
                near=NEAR, rhos=RHOS, classes=CLS, max_dJF_vs_pairs=dmax, seconds=time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(meta=meta, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()

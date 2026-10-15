#!/usr/bin/env python3
"""S1-E14 train-band: loss and logit-gradient balance of the E9 band term on DEV-A (host, numpy + cv2; never the keyframe).

For every eval view of a logit-dump root (<scene>/<t>/_logits/<cam>.npz: round-2 low_res [B,288,288] as decode_scene used
it) and every object of that view, with target T = (GT label map of cam at t == obj) (may be empty, as in training):
  L     = the adopted two bilinear stages 288 -> 1008 -> (h, w) (e14dump.logits_full_np, plain float32 order)
  terms = geomem/losses.py per object: 20 * sigmoid_focal (mean over pixels), dice (eps 1), band_bce (R_BAND 4, square
          9x9 max-pool band = dilate(T) & ~erode(T); mean BCE over band pixels; 0 when T has no band)
  grads = analytic d/dL of 20*focal + dice ("base"; the iou and presence terms have no gradient into the mask logits) and
          of band_bce ("band"), at full resolution, then pulled back to the 288 low_res through the transpose of the
          two bilinear stages (the gradient the decoder's mask path sees)
Objects predicted absent (presence <= 0: low_res = NO_OBJ_SCORE -1024 by torch.where) carry NO gradient into the mask logits;
their loss values are reported separately (they are the spikes of the E9 D2 band term).
    stage1/.venv/bin/python grad_balance.py --root <dump root> --out <json> [--workers 8]
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
sys.path.insert(0, os.path.join(S1, "geomem", "runs", "_e14_logitdump", "code"))
import e14dump as E  # noqa: E402
from mvseg2d.io import load_labelmap  # noqa: E402

DATA = os.path.join(S1, "Data")
W_FOCAL, ALPHA, GAMMA, R_BAND = 20.0, 0.25, 2.0, 4
KSQ = np.ones((2 * R_BAND + 1, 2 * R_BAND + 1), np.uint8)
PAIRS = os.path.join(S1, "geomem", "runs", "_e13_dev", "dev_pairs.json")


def interp_matrix(n_in, n_out):
    i0, i1, l0, l1 = E._axis(n_in, n_out, fma=False)
    M = np.zeros((n_out, n_in), np.float64)
    r = np.arange(n_out)
    np.add.at(M, (r, i0), l0.astype(np.float64))
    np.add.at(M, (r, i1), l1.astype(np.float64))
    return M


_MAT = {}


def composite(n_out):
    if n_out not in _MAT:
        _MAT[n_out] = (interp_matrix(E.IMAGE_SIZE, n_out) @ interp_matrix(288, E.IMAGE_SIZE)).astype(np.float32)   # [n_out, 288]
    return _MAT[n_out]


def softplus(x):
    return np.logaddexp(0, x)


def obj_terms(x, t):
    """x float32 [h,w] logits, t bool [h,w] -> (loss dict, g_base [h,w], g_band [h,w] or None)."""
    x = x.astype(np.float64)
    N = x.size
    tf = t.astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-x))
    logp, log1mp = -softplus(-x), -softplus(x)
    ce = np.where(t, -logp, -log1mp)
    p_t = np.where(t, p, 1 - p)
    a_t = np.where(t, ALPHA, 1 - ALPHA)
    focal = float((a_t * ce * (1 - p_t) ** GAMMA).mean())
    gf = np.where(t, ALPHA * (1 - p) ** GAMMA * (GAMMA * p * logp - (1 - p)),
                  (1 - ALPHA) * p ** GAMMA * (-GAMMA * (1 - p) * log1mp + p)) / N
    A = 2 * (p * tf).sum() + 1.0
    Bd = p.sum() + tf.sum() + 1.0
    dice = float(1 - A / Bd)
    gd = -(2 * tf * Bd - A) / Bd ** 2 * p * (1 - p)
    t8 = t.astype(np.uint8)
    band = (cv2.dilate(t8, KSQ) > 0) & ~(cv2.erode(t8, KSQ, borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0)
    nb = int(band.sum())
    if nb:
        band_bce = float(ce[band].mean())
        gb = np.where(band, p - tf, 0.0) / nb
        # confident-wrong share of the band (|x| > 4 and wrong side) and its share of the band BCE
        wrong = band & ((x > 0) != t)
        conf_wrong = wrong & (np.abs(x) > 4)
        extra = dict(band_wrong_frac=float(wrong.sum() / nb), band_confwrong_frac=float(conf_wrong.sum() / nb),
                     band_bce_from_confwrong=float(ce[conf_wrong].sum() / max(ce[band].sum(), 1e-12)),
                     band_px=nb, band_fn_frac=float((band & t & (x <= 0)).sum() / nb), band_fp_frac=float((band & ~t & (x > 0)).sum() / nb))
    else:
        band_bce, gb, extra = 0.0, None, dict(band_px=0)
    return dict(focal20=W_FOCAL * focal, dice=dice, band=band_bce, **extra), W_FOCAL * gf + gd, gb


def pull_back(g, Ah, Aw):
    return Ah.T @ g.astype(np.float32) @ Aw          # [288,288]


def view_job(args):
    root, scene, t, cam, pairs = args
    z = E.load(root, scene, t, cam)
    hw = tuple(int(v) for v in z["hw"])
    lab = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    assert lab.shape == hw
    pres = z["presence"] > 0
    L = E.logits_full_np(z["low_res"], hw, fma=False)
    Ah, Aw = composite(hw[0]), composite(hw[1])
    out = []
    for i, o in enumerate(z["objects"]):
        o = int(o)
        tgt = lab == o
        rec = dict(scene=scene, t=t, cam=cam, obj=o, present=bool(pres[i]), gt_px=int(tgt.sum()),
                   scored=(scene, t, cam, o) in pairs, token=int(z["token"][i]))
        if not pres[i]:
            # NO_OBJ_SCORE logits: loss values only (no gradient reaches the mask logits through torch.where)
            if tgt.any():
                t8 = tgt.astype(np.uint8)
                band = (cv2.dilate(t8, KSQ) > 0) & ~(cv2.erode(t8, KSQ, borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0)
                rec.update(band=float(-E.NO_OBJ_SCORE * (band & tgt).sum() / max(band.sum(), 1)),
                           focal20=float(W_FOCAL * ALPHA * -E.NO_OBJ_SCORE * tgt.mean()))
            out.append(rec)
            continue
        terms, gbase, gband = obj_terms(L[i], tgt)
        rec.update(terms)
        Gb = pull_back(gbase, Ah, Aw)
        rec["nb_full"] = float(np.linalg.norm(gbase))
        rec["nb_low"] = float(np.linalg.norm(Gb))
        if gband is not None:
            Gn = pull_back(gband, Ah, Aw)
            rec["nn_full"] = float(np.linalg.norm(gband))
            rec["nn_low"] = float(np.linalg.norm(Gn))
            den = rec["nb_low"] * rec["nn_low"]
            rec["cos_low"] = float((Gb * Gn).sum() / den) if den > 0 else None
            den = rec["nb_full"] * rec["nn_full"]
            rec["cos_full"] = float((gbase * gband).sum() / den) if den > 0 else None
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    pairs = E.dev_pairs(PAIRS)
    views = E.views(a.root)
    if a.limit:
        views = views[:a.limit]
    jobs = [(a.root, s, t, c, pairs) for s, t, c in views]
    with Pool(a.workers) as pool:
        recs = [r for part in pool.imap(view_job, jobs, chunksize=1) for r in part]
    json.dump(dict(root=a.root, n_views=len(views), objects=recs), open(a.out, "w"))
    pres = [r for r in recs if r["present"]]
    wb = [r for r in pres if r.get("nn_low") is not None]
    nb = np.array([r["nb_low"] for r in wb]); nn = np.array([r["nn_low"] for r in wb])
    print(f"views {len(views)} objects {len(recs)} present {len(pres)} with band {len(wb)}")
    print("W_grad (rms base / rms band, low-res) %.4f   median per-object ratio %.4f   mean cos %.3f" %
          (np.sqrt((nb ** 2).mean()) / np.sqrt((nn ** 2).mean()), np.median(nb / nn), np.mean([r["cos_low"] for r in wb])))
    print("loss means (present, GT non-empty): 20focal %.4f dice %.4f band %.4f" %
          tuple(np.mean([r[k] for r in wb]) for k in ("focal20", "dice", "band")))


if __name__ == "__main__":
    main()

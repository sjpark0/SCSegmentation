#!/usr/bin/env python3
"""S1-E14 threshold-upsample: host-only post-decode variants of the adopted round-2 decode on DEV-A (PREREG.md).

Per eval view of the logit dump (geomem/runs/_e14_logitdump/geoset_rep_x_A): low_res [B,288,288] of ALL B objects ->
upsampler U -> full-res logits -> pos = L > -tau -> adopted exclusivity (argmax winner, nested-mask cap 0.5) -> J, F of
every scored DEV-A pair (rows of _e13_dev/dev_pairs.json).  Never the keyframe, never DEV-B.

Upsamplers: bil2 (adopted, bit-exact host emulation), bil1 (control), bic1, jbu(ss, sr), gf(r, eps) -- see PREREG.md.
Logits are kept as per-object crops (the bbox where the object's logits > T_LO = -16 < -max tau); outside its crop an
object's value is <= T_LO, so it can be neither positive nor the argmax at a pixel where some object is positive
(exact for bil2 / bil1 / jbu, which are convex combinations of the 288 taps; bic1 crops come from the full-image result,
so also exact; gf: see gf_object).

    stage1/.venv/bin/python tu_variants.py --out-dir <dir>/parts [--procs 12] [--only Scene/t/cam ...] [--families ...]
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

S1 = "/home/sjpark/Documents/SCSegmentation/stage1"
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(S1, "geomem/runs/_e14_logitdump/code"))
import e14dump as E  # noqa: E402
from mvseg2d.io import load_labelmap  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
DUMP = os.path.join(S1, "geomem/runs/_e14_logitdump/geoset_rep_x_A")
PAIRS = os.path.join(S1, "geomem/runs/_e13_dev/dev_pairs.json")
TAUS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
T_LO = -16.0
G = 288
JBU = [(ss, sr) for ss in (0.5, 1.0) for sr in (4.0, 8.0, 16.0, 32.0, math.inf)]
GF = [(r, eps) for r in (4, 8) for eps in (1e-4, 1e-3, 1e-2)]
GF_MARGIN = 4 * max(r for r, _ in GF) + 2          # bil2 crops keep this margin so gf sees p within 4r of the active bbox
f32 = np.float32


def ujbu(ss, sr):
    return f"jbu_s{ss:g}_r{'inf' if math.isinf(sr) else f'{sr:g}'}"


def ugf(r, eps):
    return f"gf_r{r}_e{eps:g}"


def bbox(m, margin, H, W):
    rows = np.flatnonzero(m.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(m.any(0))
    return (max(int(rows[0]) - margin, 0), min(int(rows[-1]) + margin + 1, H),
            max(int(cols[0]) - margin, 0), min(int(cols[-1]) + margin + 1, W))


def crop_of(L, margin=0):
    """full-res [h,w] -> (y0, y1, x0, x1, crop) over the bbox of L > T_LO (+ margin), or None."""
    H, W = L.shape
    b = bbox(L > T_LO, margin, H, W)
    if b is None:
        return None
    y0, y1, x0, x1 = b
    return (y0, y1, x0, x1, np.ascontiguousarray(L[y0:y1, x0:x1], dtype=f32))


# ------------------------------------------------------------------------------------------------------- upsamplers
def up_bil2(lr, present, hw):
    """ADOPTED: bit-exact host emulation of 288 -> 1008 -> (h,w) bilinear; crops keep GF_MARGIN for gf."""
    out = []
    for b in range(lr.shape[0]):
        out.append(crop_of(E.logits_full_np(lr[b:b + 1], hw)[0], GF_MARGIN) if present[b] else None)
    return out


def up_cv(lr, present, hw, interp):
    H, W = hw
    return [crop_of(cv2.resize(lr[b], (W, H), interpolation=interp)) if present[b] else None
            for b in range(lr.shape[0])]


def taps(n_out):
    """4 tap cells along one axis for each output index: rows [4,n], d [4,n] (u - cell), valid [4,n]."""
    u = (np.arange(n_out, dtype=np.float64) + 0.5) * G / n_out - 0.5
    fu = np.floor(u).astype(np.int64)
    idx = fu[None] + np.arange(-1, 3)[:, None]
    valid = (idx >= 0) & (idx <= G - 1)
    d = u[None] - idx
    return np.clip(idx, 0, G - 1), d, valid


def jbu_weights(lab, lab288, ss, sr):
    """[16,h,w] float32 normalised JBU weights (tap order k*4 + l = row tap k, col tap l)."""
    H, W = lab.shape[:2]
    ri, rd, rv = taps(H)
    ci, cd, cv = taps(W)
    logw = np.empty((16, H, W), f32)
    for k in range(4):
        sy = (-(rd[k] ** 2) / (2 * ss * ss)).astype(f32)
        sy[~rv[k]] = -np.inf
        for l in range(4):
            sx = (-(cd[l] ** 2) / (2 * ss * ss)).astype(f32)
            sx[~cv[l]] = -np.inf
            e = sy[:, None] + sx[None, :]
            if not math.isinf(sr):
                q = lab288[ri[k]][:, ci[l]]                          # [H,W,3] guide colour of the tap cell
                d2 = ((lab - q) ** 2).sum(-1)
                e = e - d2 / f32(2 * sr * sr)
            logw[k * 4 + l] = e
    logw -= logw.max(0, keepdims=True)
    w = np.exp(logw, out=logw)
    w /= w.sum(0, keepdims=True)
    return w, ri, ci, rv, cv


def up_jbu(lr, present, hw, wts):
    w, ri, ci, rv, cv = wts
    out = []
    for b in range(lr.shape[0]):
        if not present[b]:
            out.append(None)
            continue
        x = lr[b]
        a = x > T_LO
        if not a.any():
            out.append(None)
            continue
        rows, cols = np.flatnonzero(a.any(1)), np.flatnonzero(a.any(0))
        i0, i1, j0, j1 = rows[0], rows[-1], cols[0], cols[-1]
        ys = np.flatnonzero(((ri >= i0) & (ri <= i1) & rv).any(0))    # hi-res rows with a tap row in the active bbox
        xs = np.flatnonzero(((ci >= j0) & (ci <= j1) & cv).any(0))
        y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
        acc = np.zeros((y1 - y0, x1 - x0), f32)
        for k in range(4):
            xr = x[ri[k, y0:y1]]                                      # [hy, 288]
            for l in range(4):
                acc += w[k * 4 + l, y0:y1, x0:x1] * xr[:, ci[l, x0:x1]]
        out.append((y0, y1, x0, x1, acc))
    return out


def boxmean(x, r, N):
    return cv2.boxFilter(x, cv2.CV_32F, (2 * r + 1, 2 * r + 1), normalize=False, borderType=cv2.BORDER_CONSTANT) / N


def gf_guide(I, r, eps):
    """Object-independent guided-filter terms on the full image: mean_I [h,w,3], inverse of Sigma + eps U as 6 entries."""
    H, W = I.shape[:2]
    N = boxmean(np.ones((H, W), f32), r, 1.0)
    mI = [boxmean(I[..., c], r, N) for c in range(3)]
    S = {}
    for a in range(3):
        for b in range(a, 3):
            S[a, b] = boxmean(I[..., a] * I[..., b], r, N) - mI[a] * mI[b] + (eps if a == b else 0.0)
    s00, s01, s02, s11, s12, s22 = (S[0, 0], S[0, 1], S[0, 2], S[1, 1], S[1, 2], S[2, 2])
    c00 = s11 * s22 - s12 * s12
    c01 = s02 * s12 - s01 * s22
    c02 = s01 * s12 - s02 * s11
    c11 = s00 * s22 - s02 * s02
    c12 = s01 * s02 - s00 * s12
    c22 = s00 * s11 - s01 * s01
    det = s00 * c00 + s01 * c01 + s02 * c02
    inv = np.stack([c00, c01, c02, c11, c12, c22], -1) / det[..., None]
    return dict(r=r, mI=np.stack(mI, -1).astype(f32), inv=inv.astype(f32))


def gf_object(I, gd, c, hw):
    """Guided filter (He et al. 2013, colour guide) of one object's bil2 logits.  c = bil2 crop (active bbox + GF_MARGIN);
    the filter runs on that crop (its box sums are exact for outputs >= 2r from a non-image crop edge) and the output
    crop is the active bbox (logits > T_LO) grown by 2r.  Outside it the object is taken as <= T_LO (all p within 2r
    are <= T_LO there; a regression overshoot of > 12 logits would be needed to make it positive)."""
    H, W = hw
    r = gd["r"]
    y0, y1, x0, x1, p = c
    act = bbox(p > T_LO, 0, *p.shape)
    if act is None:
        return None
    ay0, ay1, ax0, ax1 = act
    oy0, oy1 = max(y0 + ay0 - 2 * r, 0), min(y0 + ay1 + 2 * r, H)       # output region (image coords)
    ox0, ox1 = max(x0 + ax0 - 2 * r, 0), min(x0 + ax1 + 2 * r, W)
    # need p on output region grown by 2r; the bil2 crop has GF_MARGIN = 4 r_max + 2 beyond the active bbox
    Ic = I[y0:y1, x0:x1]
    N = boxmean(np.ones(p.shape, f32), r, 1.0)
    mp = boxmean(p, r, N)
    mI = gd["mI"][y0:y1, x0:x1]
    cov = np.stack([boxmean(Ic[..., k] * p, r, N) for k in range(3)], -1) - mI * mp[..., None]
    inv = gd["inv"][y0:y1, x0:x1]
    i00, i01, i02, i11, i12, i22 = (inv[..., k] for k in range(6))
    a0 = i00 * cov[..., 0] + i01 * cov[..., 1] + i02 * cov[..., 2]
    a1 = i01 * cov[..., 0] + i11 * cov[..., 1] + i12 * cov[..., 2]
    a2 = i02 * cov[..., 0] + i12 * cov[..., 1] + i22 * cov[..., 2]
    bb = mp - a0 * mI[..., 0] - a1 * mI[..., 1] - a2 * mI[..., 2]
    q = boxmean(a0, r, N) * Ic[..., 0] + boxmean(a1, r, N) * Ic[..., 1] + boxmean(a2, r, N) * Ic[..., 2] + boxmean(bb, r, N)
    # crop edges that are not image edges: the box sums there are wrong within 2r -> must not reach the output region
    assert (oy0 - y0 >= 2 * r or y0 == 0) and (y1 - oy1 >= 2 * r or y1 == H), (y0, y1, oy0, oy1, r)
    assert (ox0 - x0 >= 2 * r or x0 == 0) and (x1 - ox1 >= 2 * r or x1 == W), (x0, x1, ox0, ox1, r)
    return (oy0, oy1, ox0, ox1, np.ascontiguousarray(q[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0], dtype=f32))


def up_gf(crops_bil2, hw, I, gd):
    return [None if c is None else gf_object(I, gd, c, hw) for c in crops_bil2]


# ------------------------------------------------------------------------------------------------ exclusivity + score
def argmax_map(crops, hw):
    H, W = hw
    best = np.full((H, W), -np.inf, f32)
    arg = np.full((H, W), -1, np.int32)
    for b, c in enumerate(crops):
        if c is None:
            continue
        y0, y1, x0, x1, L = c
        sb, sa = best[y0:y1, x0:x1], arg[y0:y1, x0:x1]
        m = L > sb                                               # strict: exact ties keep the lowest index (torch)
        sb[m] = L[m]
        sa[m] = b
    return arg


def keep_masks(crops, arg, tau):
    """adopted exclusive_masks(L + tau, 0.5) on crops -> {b: (y0,y1,x0,x1,keep crop)}."""
    out = {}
    for b, c in enumerate(crops):
        if c is None:
            continue
        y0, y1, x0, x1, L = c
        pos = L > f32(-tau)
        npos = int(pos.sum())
        if npos == 0:
            continue
        win = arg[y0:y1, x0:x1] == b
        lost = int((pos & ~win).sum())
        exempt = f32(lost) >= f32(0.5) * f32(npos)
        out[b] = (y0, y1, x0, x1, pos if exempt else pos & win)
    return out


def view_job(args):
    scene, t, cam, rows, families, check = args
    t0 = time.time()
    d = E.load(DUMP, scene, t, cam)
    objs = [int(o) for o in d["objects"]]
    hw = (int(d["hw"][0]), int(d["hw"][1]))
    H, W = hw
    lr = d["low_res"].astype(f32)
    present = d["presence"] > 0
    fdir = os.path.join(DATA, scene, "frames", f"{t:06d}")
    gt = load_labelmap(os.path.join(fdir, "masks", f"{cam}.png"))
    assert gt.shape == hw
    bgr = cv2.imread(os.path.join(fdir, "images", f"{cam}.jpg"), cv2.IMREAD_COLOR)
    assert bgr.shape[:2] == hw
    bp = bound_pix(H, W)
    se = disk(bp)
    pairs = sorted(int(r["obj"]) for r in rows)
    pidx = {o: objs.index(o) for o in pairs}
    gts = {o: gt == o for o in pairs}
    gbox = {o: bbox(gts[o], 0, H, W) for o in pairs}
    assert all(v is not None for v in gbox.values())               # dev pairs have gt_px > 0
    res = {}

    def score(name, crops):
        arg = argmax_map(crops, hw)
        for tau in TAUS:
            km = keep_masks(crops, arg, tau)
            if check and name == "bil2" and tau == 0.0:
                pm = E.read_png_masks(DUMP, scene, t, cam, objs, hw)
                full = np.zeros_like(pm)
                for b, (y0, y1, x0, x1, k) in km.items():
                    full[b, y0:y1, x0:x1] = k
                res["check_png_diff_px"] = int((full != pm).sum())
            J, F, TP, PP = [], [], [], []
            for o in pairs:
                b = pidx[o]
                gy0, gy1, gx0, gx1 = gbox[o]
                if b in km:                                       # union bbox of GT and prediction, grown by bp + 2
                    y0, y1, x0, x1, k = km[b]
                    kb = bbox(k, 0, *k.shape)
                    if kb is not None:
                        gy0, gy1 = min(gy0, y0 + kb[0]), max(gy1, y0 + kb[1])
                        gx0, gx1 = min(gx0, x0 + kb[2]), max(gx1, x0 + kb[3])
                m = bp + 2
                cy0, cy1, cx0, cx1 = max(gy0 - m, 0), min(gy1 + m, H), max(gx0 - m, 0), min(gx1 + m, W)
                pred = np.zeros((cy1 - cy0, cx1 - cx0), bool)
                if b in km:
                    oy0, oy1 = max(y0, cy0), min(y1, cy1)
                    ox0, ox1 = max(x0, cx0), min(x1, cx1)
                    if oy1 > oy0 and ox1 > ox0:
                        pred[oy0 - cy0:oy1 - cy0, ox0 - cx0:ox1 - cx0] = k[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
                g = gts[o][cy0:cy1, cx0:cx1]
                j, f = jf_pair(g, pred, bp, se)                   # == jf_pair on the full frame (same internal crop)
                J.append(float(j))
                F.append(float(f))
                TP.append(int((pred & g).sum()))
                PP.append(int(pred.sum()))
            res[f"{name}|{tau:g}"] = dict(J=J, F=F, tp=TP, pp=PP)

    timing = {}
    t1 = time.time()
    c_bil2 = up_bil2(lr, present, hw)
    timing["bil2"] = time.time() - t1
    if "bil2" in families:
        score("bil2", c_bil2)
    if "bil1" in families:
        score("bil1", up_cv(lr, present, hw, cv2.INTER_LINEAR))
    if "bic1" in families:
        t1 = time.time()
        score("bic1", up_cv(lr, present, hw, cv2.INTER_CUBIC))
        timing["bic1"] = time.time() - t1
    if "jbu" in families:
        lab = cv2.cvtColor(bgr.astype(f32) / 255.0, cv2.COLOR_BGR2Lab)
        lab288 = cv2.resize(lab, (G, G), interpolation=cv2.INTER_AREA)
        for ss, sr in JBU:
            t1 = time.time()
            wts = jbu_weights(lab, lab288, ss, sr)
            score(ujbu(ss, sr), up_jbu(lr, present, hw, wts))
            del wts
            timing[ujbu(ss, sr)] = time.time() - t1
    if "gf" in families:
        I = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(f32) / 255.0
        for r, eps in GF:
            t1 = time.time()
            gd = gf_guide(I, r, eps)
            score(ugf(r, eps), up_gf(c_bil2, hw, I, gd))
            del gd
            timing[ugf(r, eps)] = time.time() - t1
    gt_px = [int(gts[o].sum()) for o in pairs]
    return dict(scene=scene, t=t, cam=cam, view=f"{t}/{cam}", objs=pairs, gt_px=gt_px, hw=list(hw),
                n_obj=len(objs), n_present=int(present.sum()), res=res,
                check_png_diff_px=res.pop("check_png_diff_px", None), timing=timing, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None, help="scene/t/cam keys")
    ap.add_argument("--families", nargs="*", default=["bil2", "bil1", "bic1", "jbu", "gf"])
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = json.load(open(PAIRS))["rows"]
    views = {}
    for r in rows:
        views.setdefault((r["scene"], int(r["t"]), r["cam"]), []).append(r)
    jobs = []
    for (s, t, c), rs in views.items():
        key = f"{s}/{t}/{c}"
        if a.only and key not in a.only:
            continue
        if os.path.isfile(os.path.join(a.out_dir, f"{s}_{t:06d}_{c}.json")):
            continue
        d = E.load(DUMP, s, t, c)
        jobs.append((int(d["hw"][0]) * int(d["hw"][1]) * int((d["presence"] > 0).sum()), (s, t, c, rs, set(a.families), not a.no_check)))
    jobs = [j for _, j in sorted(jobs, key=lambda x: -x[0])]
    print(f"{len(jobs)} views to do, families {a.families}", flush=True)
    t0 = time.time()
    with Pool(a.procs, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(view_job, jobs, chunksize=1)):
            p = os.path.join(a.out_dir, f"{r['scene']}_{r['t']:06d}_{r['cam']}.json")
            json.dump(r, open(p + ".tmp", "w"))
            os.replace(p + ".tmp", p)
            print(f"[{i + 1}/{len(jobs)}] {r['scene']} {r['t']} {r['cam']} {r['seconds']:.0f}s png_diff={r['check_png_diff_px']} "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

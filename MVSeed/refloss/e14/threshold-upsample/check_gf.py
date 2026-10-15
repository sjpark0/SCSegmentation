#!/usr/bin/env python3
"""Check the gf crop approximation (tu_variants.gf_object): full-image guided filter of every object vs the crop
version, compared as final keep masks (all objects, all taus) on a few DEV-A views.  Also checks bic1/jbu crop
exactness the same way (full-image computation vs crops).

    stage1/.venv/bin/python check_gf.py Scene/t/cam ...
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tu_variants as T  # noqa: E402

f32 = np.float32


def full_crops(Lfull):
    H, W = Lfull.shape[1:]
    return [(0, H, 0, W, Lfull[b]) if Lfull[b].max() > -1000 else None for b in range(Lfull.shape[0])]


def keep_full(crops, hw, tau):
    arg = T.argmax_map(crops, hw)
    km = T.keep_masks(crops, arg, tau)
    out = np.zeros((len(crops), *hw), bool)
    for b, (y0, y1, x0, x1, k) in km.items():
        out[b, y0:y1, x0:x1] = k
    return out


def gf_full(p, I, gd):
    r = gd["r"]
    H, W = p.shape
    N = T.boxmean(np.ones((H, W), f32), r, 1.0)
    mp = T.boxmean(p, r, N)
    mI = gd["mI"]
    cov = np.stack([T.boxmean(I[..., k] * p, r, N) for k in range(3)], -1) - mI * mp[..., None]
    inv = gd["inv"]
    i00, i01, i02, i11, i12, i22 = (inv[..., k] for k in range(6))
    a0 = i00 * cov[..., 0] + i01 * cov[..., 1] + i02 * cov[..., 2]
    a1 = i01 * cov[..., 0] + i11 * cov[..., 1] + i12 * cov[..., 2]
    a2 = i02 * cov[..., 0] + i12 * cov[..., 1] + i22 * cov[..., 2]
    bb = mp - a0 * mI[..., 0] - a1 * mI[..., 1] - a2 * mI[..., 2]
    return T.boxmean(a0, r, N) * I[..., 0] + T.boxmean(a1, r, N) * I[..., 1] + T.boxmean(a2, r, N) * I[..., 2] + T.boxmean(bb, r, N)


def main():
    res = {}
    for key in sys.argv[1:]:
        s, t, c = key.split("/")
        t = int(t)
        d = T.E.load(T.DUMP, s, t, c)
        hw = (int(d["hw"][0]), int(d["hw"][1]))
        lr = d["low_res"].astype(f32)
        present = d["presence"] > 0
        bgr = cv2.imread(os.path.join(T.DATA, s, "frames", f"{t:06d}", "images", f"{c}.jpg"), cv2.IMREAD_COLOR)
        I = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(f32) / 255.0
        Lb = np.stack([T.E.logits_full_np(lr[b:b + 1], hw)[0] if present[b] else np.full(hw, -1024, f32)
                       for b in range(lr.shape[0])])
        cb = T.up_bil2(lr, present, hw)
        out = {}
        for r, eps in T.GF:
            gd = T.gf_guide(I, r, eps)
            cf = T.up_gf(cb, hw, I, gd)
            ff = full_crops(np.stack([gf_full(Lb[b], I, gd) if present[b] else Lb[b] for b in range(len(Lb))]))
            out[T.ugf(r, eps)] = {f"{tau:g}": int((keep_full(cf, hw, tau) != keep_full(ff, hw, tau)).sum()) for tau in T.TAUS}
        # bic1 + one jbu setting: crops vs full image
        Lc = np.stack([cv2.resize(lr[b], (hw[1], hw[0]), interpolation=cv2.INTER_CUBIC) for b in range(lr.shape[0])])
        Lc[~present] = -1024
        out["bic1"] = {f"{tau:g}": int((keep_full(T.up_cv(lr, present, hw, cv2.INTER_CUBIC), hw, tau) != keep_full(full_crops(Lc), hw, tau)).sum()) for tau in T.TAUS}
        lab = cv2.cvtColor(bgr.astype(f32) / 255.0, cv2.COLOR_BGR2Lab)
        lab288 = cv2.resize(lab, (T.G, T.G), interpolation=cv2.INTER_AREA)
        w = T.jbu_weights(lab, lab288, 1.0, 8.0)
        cj = T.up_jbu(lr, present, hw, w)
        W_, ri, ci, rv, cv = w
        Lj = np.full((lr.shape[0], *hw), -1024, f32)
        for b in range(lr.shape[0]):
            if present[b]:
                acc = np.zeros(hw, f32)
                for k in range(4):
                    xr = lr[b][ri[k]]
                    for l in range(4):
                        acc += W_[k * 4 + l] * xr[:, ci[l]]
                Lj[b] = acc
        out["jbu_s1_r8"] = {f"{tau:g}": int((keep_full(cj, hw, tau) != keep_full(full_crops(Lj), hw, tau)).sum()) for tau in T.TAUS}
        res[key] = out
        print(key, json.dumps(out), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_crops.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

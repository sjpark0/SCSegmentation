#!/usr/bin/env python3
"""Failure-mode diagnosis of candidate D on single DEV views (host).  Recomputes the target-side growth inputs
(plab/pero/cut/acc/traversable) with grow.py's own functions, reads the saved primary grown masks (main/masks), and
reports where the missed pixels of an object stop, plus a viz panel.  DEV ONLY (asserts t != ref_frame).
    stage1/.venv/bin/python -I diag.py SCENE T CAM OBJ [SCENE T CAM OBJ ...] --out viz.jpg
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import grow as Gm  # noqa: E402
from mvseg2d.io import load_labelmap, load_meta  # noqa: E402

DATA, DEVROOT = Gm.DATA, Gm.DEVROOT


def one(scene, t, T, o, cfg):
    idx = json.load(open(os.path.join(DEVROOT, "index.json")))["scenes"][scene]
    meta = load_meta(DATA, scene)
    c = meta["ref_cam"]
    assert t != int(meta["ref_frame"])
    basic = [int(x) for x in idx["basic"]]
    cdir = os.path.join(DATA, scene, "Cache", "pi3_frames", f"{t:06d}")
    hw_w = tuple(json.load(open(os.path.join(cdir, "MANIFEST.json")))["work_hw"])
    fdir = os.path.join(DATA, scene, "frames", f"{t:06d}")
    Lc = load_labelmap(os.path.join(fdir, "masks", f"{c}.png"))
    Lc = np.where(np.isin(Lc, basic), Lc, 0).astype(np.uint8)
    H, W = Lc.shape
    objs = [x for x in basic if (Lc == x).any()]
    zc = Gm.load_cam(scene, t, c)
    Kc = Gm.full_K(zc["K"], hw_w, (H, W))
    dcf = cv2.resize(zc["depth"], (W, H), interpolation=cv2.INTER_NEAREST)
    r = int(round(cfg["R_ERODE"] * W / hw_w[1]))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    Lero = np.zeros_like(Lc)
    for q in objs:
        Lero[cv2.erode((Lc == q).astype(np.uint8), se).astype(bool)] = q
    Lc_w = cv2.resize(Lc, (hw_w[1], hw_w[0]), interpolation=cv2.INTER_NEAREST)
    bins_c, _ = Gm.color_bins(os.path.join(fdir, "images", f"{c}.jpg"), hw_w, cfg["NBINS"])
    z = Gm.load_cam(scene, t, T)
    pc = Gm.to_reference(Gm.local_points(z["depth"], z["K"]), z["c2w"], zc["c2w"]).astype(np.float32)
    visg, row, col = Gm.project(pc, Kc, (H, W), dcf)
    vis = visg & z["valid"]
    plab = np.full(hw_w, Gm.INVIS, np.uint8)
    plab[vis] = Lc[row[vis], col[vis]]
    pero = np.full(hw_w, Gm.INVIS, np.uint8)
    pero[vis] = Lero[row[vis], col[vis]]
    cut = Gm.cut_mask(z["depth"], z["valid"], cfg["TAU_2"])
    bins, _ = Gm.color_bins(os.path.join(fdir, "images", f"{T}.jpg"), hw_w, cfg["NBINS"])
    acc = Gm.accept_bins(bins_c, Lc_w, o, cfg["NBINS"], cfg["LR"])[bins]
    trav = z["valid"] & ~cut & ((plab == o) | ((plab == Gm.INVIS) & acc))
    n, lab = cv2.connectedComponents(trav.astype(np.uint8), connectivity=4)
    seeds = trav & (pero == o)
    sd = np.unique(lab[seeds])
    sd = sd[sd > 0]
    tv = np.isin(lab, sd)
    zz = np.load(os.path.join(HERE, "main", "masks", scene, f"{t:06d}", f"{T}.npz"))
    mv = np.unpackbits(zz[f"mv|{o}"])[:hw_w[0] * hw_w[1]].reshape(hw_w).astype(bool) if f"mv|{o}" in zz else np.zeros(hw_w, bool)
    gt = load_labelmap(os.path.join(fdir, "masks", f"{T}.png"))
    gtw = cv2.resize(gt, (hw_w[1], hw_w[0]), interpolation=cv2.INTER_NEAREST)
    D = Gm.rd(os.path.join(DEVROOT, scene, f"{t:06d}", T, f"{o}.png"), (H, W))
    Dw = cv2.resize(D.astype(np.uint8), (hw_w[1], hw_w[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    miss = (gtw == o) & ~Dw & ~vis
    mv_ = miss & z["valid"]
    # where do the missed, valid, ref-invisible pixels of o stop?
    st = dict(missed_inv_valid=int(mv_.sum()), acc=int((mv_ & acc).sum()), not_cut=int((mv_ & ~cut).sum()),
              trav=int((mv_ & trav).sum()), tv_grown=int((mv_ & tv).sum()), mv_grown=int((mv_ & mv).sum()))
    # largest traversable component holding missed pixels: does it touch a seed?  what borders it?
    comp_ids, cnts = np.unique(lab[mv_ & trav], return_counts=True)
    top = []
    for ci, cn in sorted(zip(comp_ids, cnts), key=lambda x: -x[1])[:3]:
        if ci == 0:
            continue
        m = lab == ci
        ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
        top.append(dict(comp=int(ci), missed_px=int(cn), comp_px=int(m.sum()), seeded=bool(ci in sd), comp_gt_o=float((gtw[m] == o).mean()),
                        ring_px=int(ring.sum()), ring_cut=float((ring & cut).mean() / max(ring.mean(), 1e-12)),
                        ring_vis_other=float((ring & (plab != Gm.INVIS) & (plab != o)).sum() / max(ring.sum(), 1)),
                        ring_vis_o=float((ring & (plab == o)).sum() / max(ring.sum(), 1)),
                        ring_inv_rejected=float((ring & (plab == Gm.INVIS) & ~acc & z["valid"]).sum() / max(ring.sum(), 1)),
                        ring_nodepth=float((ring & ~z["valid"]).sum() / max(ring.sum(), 1))))
    st["top_components"] = top
    st["seeds_T"] = int(seeds.sum())
    st["vis_o_T"] = int((plab == o).sum())
    # where the primary's grown region lands (GT label share, work res, ref-invisible)
    R = mv & ~vis
    labs, cn = np.unique(gtw[R], return_counts=True)
    st["grown_invis_px"] = int(R.sum())
    st["grown_invis_gt"] = {int(a): round(float(b) / max(1, R.sum()), 3) for a, b in sorted(zip(labs, cn), key=lambda x: -x[1])[:5]}
    # viz panel (work res)
    img = cv2.resize(cv2.imread(os.path.join(fdir, "images", f"{T}.jpg")), (hw_w[1], hw_w[0]), interpolation=cv2.INTER_AREA)
    p1 = img.copy()
    p1[gtw == o] = (0.5 * p1[gtw == o] + [0, 127, 0]).astype(np.uint8)
    p1[Dw] = (0.5 * p1[Dw] + [127, 0, 0]).astype(np.uint8)
    p2 = np.zeros_like(img)
    p2[plab == Gm.INVIS] = (90, 90, 90)
    p2[(plab != Gm.INVIS) & (plab != o)] = (0, 0, 200)
    p2[plab == o] = (0, 200, 0)
    p2[~z["valid"]] = (0, 0, 0)
    p3 = np.zeros_like(img)
    p3[trav] = (200, 200, 200)
    p3[cut] = (0, 0, 255)
    p3[tv] = (255, 200, 0)
    p3[seeds] = (0, 255, 0)
    p4 = img.copy() // 3
    p4[mv & (gtw == o)] = (0, 255, 0)
    p4[mv & (gtw != o)] = (0, 0, 255)
    panel = np.concatenate([p1, p2, p3, p4], 1)
    txt = f"{scene} t={t} {T} obj {o}: GT(green)+D(blue) | plab vis-o green, vis-other red, invisible grey, nodepth black | " \
          f"trav grey, cut red, tv-grown cyan, seed green | mv grown: on o green, off o red"
    cv2.putText(panel, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return st, panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = dict(Gm.DEFAULT)
    rows = []
    for i in range(0, len(a.cases), 4):
        s, t, T, o = a.cases[i], int(a.cases[i + 1]), a.cases[i + 2], int(a.cases[i + 3])
        st, panel = one(s, t, T, o, cfg)
        print(s, t, T, o, json.dumps(st))
        w = 2400
        rows.append(cv2.resize(panel, (w, int(panel.shape[0] * w / panel.shape[1]))))
    cv2.imwrite(a.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()

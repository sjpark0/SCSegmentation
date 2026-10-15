#!/usr/bin/env python3
"""S1-E13 candidate C (sam3-prompt-extend), host part: add-only composition of SAM 3 extent proposals with the adopted
geoset_rep_x DEV masks, and DEV scoring.  DEV ONLY (index.json frames; asserts t != ref_frame).  Design: REGISTRATION.txt.

Per view (scene, t, cam), variant v, object o (seeded = basic set present in c_ini GT at t):
  C_o   = S_o & ~vis & ~D_o & ~union_{q != o} D_q
  Apre_o = C_o pixels in 8-connected components of (C_o | D_o) holding a D_o pixel
  config (min_proj, gate): active = {o : |P_o| >= min_proj and predicted IoU(v) >= gate}
  A_o   = Apre_o minus pixels in Apre of >= 2 active objects (contested);  N_o = D_o | A_o
Per scored pair (GT non-empty): J&F of N_o (mvseg2d.metrics.jf_pair) minus J&F of D_o (re-computed, asserted equal to
dev_pairs.json), added pixels by GT label (o / other object / unlabelled 0) and by visibility class, and the raw-proposal
leak statistics.  The funnel (S & missed -> & ~vis -> & unclaimed -> connected -> uncontested) for the main config.

  PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python -I compose_score.py --out rows.json --procs 12
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

REPO = "/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from mvseg2d.io import load_labelmap, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
DEV = os.path.join(S1, "geomem", "runs", "_e13_dev")
PROPS = os.path.join(S1, "geomem", "runs", "_e13_sam3-prompt-extend", "props")
VARIANTS = ("pts", "pts_whole", "pts_neg", "box", "mask_box")
MAIN = (20, None)                                            # (min_proj, iou gate) registered
CONFIGS = [(20, None), (200, None), (2000, None), (20, 0.7), (20, 0.85)]
CLSN = {1: "outside", 2: "occluded", 3: "nodepth"}


def rd(p, hw):
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    m = read_pred_mask(p)
    assert m.shape == tuple(hw), (p, m.shape, hw)
    return m


def h(m):
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return "empty"
    return hashlib.blake2b(np.concatenate([ys, xs]).astype(np.int32).tobytes(), digest_size=16).hexdigest()


def connected_add(C, D):
    if not C.any() or not D.any():
        return np.zeros_like(C)
    U = C | D
    rows, cols = np.flatnonzero(U.any(1)), np.flatnonzero(U.any(0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    n, lab = cv2.connectedComponents(U[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    keep = np.unique(lab[D[y0:y1, x0:x1]])
    keep = keep[keep > 0]
    A = np.zeros_like(C)
    A[y0:y1, x0:x1] = np.isin(lab, keep) & C[y0:y1, x0:x1]
    return A


def view(job):
    scene, t, cam, ref_frame = job
    assert t != ref_frame
    vdir = os.path.join(PROPS, scene, f"{t:06d}", cam)
    meta = json.load(open(os.path.join(vdir, "meta.json")))
    H, W = meta["hw"]
    hw = (H, W)
    objs = [int(o) for o in meta["objs"]]
    gt = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    assert gt.shape == hw
    cls = cv2.imread(os.path.join(vdir, "cls.png"), cv2.IMREAD_UNCHANGED)
    assert cls.shape == hw and cls.max() <= 3
    invis = cls > 0
    D = {o: rd(os.path.join(DEV, scene, f"{t:06d}", cam, f"{o}.png"), hw) for o in objs}
    cnt = np.zeros(hw, np.uint8)
    for o in objs:
        cnt += D[o]
    n_dev_overlap = int((cnt > 1).sum())
    bp = bound_pix(H, W)
    se = disk(bp)
    G = {o: gt == o for o in objs}
    seeded = np.isin(gt, objs)
    gpx = {o: int(G[o].sum()) for o in objs}
    base = {}
    for o in objs:
        if gpx[o]:
            base[o] = jf_pair(G[o], D[o], bp, se)
    proj_px = {int(k): v for k, v in meta["proj_px"].items()}
    props = {int(k): v for k, v in meta["props"].items()}
    gtcls = {o: [int((G[o] & (cls == c)).sum()) for c in (0, 1, 2, 3)] for o in base}
    out = dict(scene=scene, t=t, cam=cam, objs=objs, n_dev_overlap=n_dev_overlap, base={o: list(base[o]) for o in base},
               gt_px=gpx, proj_px=proj_px, gtcls=gtcls, pairs={}, empty_gt={}, leak={})
    # leak of the raw proposals + composition per variant
    for v in VARIANTS:
        S, Apre, fun = {}, {}, {}
        for o in objs:
            if o not in props:
                continue
            S[o] = rd(os.path.join(vdir, v, f"{o}.png"), hw)
            s = S[o]
            ns = int(s.sum())
            if ns:
                labs, cn = np.unique(gt[s], return_counts=True)
                dom = int(labs[int(np.argmax(cn))])
                on = int((s & G[o]).sum())
                si = s & invis
                out["leak"].setdefault(v, {})[o] = dict(
                    s_px=ns, s_on_o=on, dom=dom, dom_px=int(cn.max()), s_on_bg=int(cn[labs == 0].sum()),
                    s_invis=int(si.sum()), s_invis_on_o=int((si & G[o]).sum()),
                    pred_iou=props[o][v]["pred_iou"])
            claimed_other = (cnt - D[o]) > 0
            C = s & invis & ~D[o] & ~claimed_other
            Apre[o] = connected_add(C, D[o])
            if gpx[o]:
                miss = G[o] & ~D[o]
                fun[o] = [int(miss.sum()), int((s & miss).sum()), int((s & miss & invis).sum()),
                          int((C & miss).sum()), int((Apre[o] & miss).sum())]
            del C
        jf_cache = {}
        for cfg in CONFIGS:
            mp, gate = cfg
            act = [o for o in Apre if proj_px[o] >= mp and (gate is None or props[o][v]["pred_iou"] >= gate)]
            ca = np.zeros(hw, np.uint8)
            for o in act:
                ca += Apre[o]
            key = f"{v}|{mp}|{gate}"
            for o in objs:
                A = Apre[o] & (ca == 1) if o in act else np.zeros(hw, bool)
                na = int(A.sum())
                if not gpx[o]:
                    if na:
                        out["empty_gt"].setdefault(key, {})[o] = na
                    continue
                r = out["pairs"].setdefault(o, {})
                if na == 0:
                    r[key] = [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    continue
                hh = h(A)
                if hh not in jf_cache.setdefault(o, {}):
                    J, F = jf_pair(G[o], D[o] | A, bp, se)
                    jf_cache[o][hh] = (J - base[o][0], F - base[o][1])
                dJ, dF = jf_cache[o][hh]
                lab_a = gt[A]
                on_o = A & G[o]
                n_on = int(on_o.sum())
                n_bg = int((lab_a == 0).sum())
                ca_cls = [int((A & (cls == c)).sum()) for c in (1, 2, 3)]
                on_cls = [int((on_o & (cls == c)).sum()) for c in (1, 2, 3)]
                # [dJ, dF, added, on_o, on_other, on_bg, add_out, add_occ, add_nod, on_out, on_occ, on_nod, on_other_seeded]
                r[key] = [float(dJ), float(dF), na, n_on, na - n_on - n_bg, n_bg] + ca_cls + on_cls + \
                    [int((A & seeded & ~G[o]).sum())]
                if cfg == MAIN:
                    fun[o].append(n_on)
            if cfg == MAIN:
                for o in fun:
                    if len(fun[o]) == 5:
                        fun[o].append(0)
                    out["pairs"].setdefault(o, {})[f"funnel|{v}"] = fun[o]
        del S, Apre
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--scenes", default="all")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(DEV, "index.json")))
    jobs = []
    for s, S in sorted(idx["scenes"].items()):
        if a.scenes != "all" and s not in a.scenes.split(","):
            continue
        for t in S["frames"]:
            for cam in S["eval_cams"]:
                assert os.path.isfile(os.path.join(PROPS, s, f"{t:06d}", cam, "DONE")), (s, t, cam)
                jobs.append((s, t, cam, S["ref_frame"]))
    t0 = time.time()
    res = []
    with Pool(a.procs) as p:
        for i, r in enumerate(p.imap_unordered(view, jobs)):
            res.append(r)
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    json.dump(dict(configs=CONFIGS, variants=VARIANTS, main=MAIN, views=res, seconds=time.time() - t0), open(a.out, "w"))
    print(f"done {len(res)} views {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

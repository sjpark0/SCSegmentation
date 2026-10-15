#!/usr/bin/env python3
"""S1-E13 candidate E1 (text-concept), host part: selection of a text-concept detection per seeded object, add-only
composition with the adopted geoset_rep_x DEV masks (IDENTICAL to candidate C), and DEV scoring.
PROTOCOL CHANGE (label names from annotation metadata).  DEV ONLY (asserts t != ref_frame).  Design: REGISTRATION.txt.

Per view (scene, t, cam) and selection (variant, TAU, name form):
  pool_o = detections (prob > TAU) of o's prompt strings;  S_o = selection (best_ov / best_iou / union_maj / union_all /
  oracle[diagnostic, target GT]);  C_o = S_o & ~vis & ~D_o & ~union_{q != o} D_q;
  Apre_o = C_o pixels in 8-connected components of (C_o | D_o) holding a D_o pixel;
  config adds MIN_PROJ_PX: active = {o : S_o exists and (variant in {union_all, oracle} or |P_o| >= MIN_PROJ_PX)};
  A_o = Apre_o minus pixels in Apre of >= 2 active objects;  N_o = D_o | A_o.
Per scored pair (GT non-empty): J&F change, added pixels by GT label and visibility class, funnel, leak of raw S_o,
detector reach (best-instance IoU, union coverage of missed px).
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
DETS = os.path.join(S1, "geomem", "runs", "_e13_text-concept", "dets")
VARIANTS = ("best_ov", "best_iou", "union_maj", "union_all", "oracle")
PROJ_VARIANTS = ("best_ov", "best_iou", "union_maj")
SELS = [(0.5, "split"), (0.3, "split"), (0.7, "split"), (0.5, "joined")]
MAIN_SEL = (0.5, "split")
MAIN_MP = 20


def configs(v):
    out = [(tau, form, MAIN_MP) for tau, form in SELS]
    if v in PROJ_VARIANTS:
        out += [(0.5, "split", 200), (0.5, "split", 2000)]
    return out


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
    """candidate C's connected_add, verbatim."""
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


def key(v, tau, form, mp):
    return f"{v}|{tau}|{form}|{mp}"


def view(job):
    scene, t, cam, ref_frame = job
    assert t != ref_frame
    pdir = os.path.join(PROPS, scene, f"{t:06d}", cam)
    ddir = os.path.join(DETS, scene, f"{t:06d}", cam)
    pmeta = json.load(open(os.path.join(pdir, "meta.json")))
    meta = json.load(open(os.path.join(ddir, "meta.json")))
    H, W = meta["hw"]
    hw = (H, W)
    assert pmeta["hw"] == meta["hw"] and pmeta["objs"] == meta["objs"] and meta["t"] == t
    objs = [int(o) for o in meta["objs"]]
    gt = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    assert gt.shape == hw
    cls = cv2.imread(os.path.join(pdir, "cls.png"), cv2.IMREAD_UNCHANGED)
    assert cls.shape == hw and cls.max() <= 3
    vis, invis = cls == 0, cls > 0
    proj_px = {int(k): v for k, v in pmeta["proj_px"].items()}
    P = {o: rd(os.path.join(pdir, "proj", f"{o}.png"), hw) for o in objs}
    for o in objs:
        assert int(P[o].sum()) == proj_px[o], (scene, t, cam, o)
    D = {o: rd(os.path.join(DEV, scene, f"{t:06d}", cam, f"{o}.png"), hw) for o in objs}
    cnt = np.zeros(hw, np.uint8)
    for o in objs:
        cnt += D[o]
    bp = bound_pix(H, W)
    se = disk(bp)
    G = {o: gt == o for o in objs}
    seeded = np.isin(gt, objs)
    gpx = {o: int(G[o].sum()) for o in objs}
    base = {o: jf_pair(G[o], D[o], bp, se) for o in objs if gpx[o]}
    gtcls = {o: [int((G[o] & (cls == c)).sum()) for c in (0, 1, 2, 3)] for o in base}
    miss = {o: G[o] & ~D[o] for o in base}

    # ---------------------------------------------------------------- detections
    z = np.load(os.path.join(ddir, "dets.npz"))
    packed, pidx, prob = z["packed"], z["prompt"], z["prob"]
    strings = meta["strings"]
    N = int(packed.shape[0])
    M = [np.unpackbits(packed[k], count=H * W).reshape(H, W).astype(bool) for k in range(N)]
    area = [int(m.sum()) for m in M]
    a_vis = [int((m & vis).sum()) for m in M]
    oprom = {int(o): v for o, v in meta["obj_prompts"].items()}
    sid = {s: i for i, s in enumerate(strings)}
    pools_all = {form: {o: [k for k in range(N) if int(pidx[k]) in {sid[s] for s in oprom[o][form]}] for o in objs}
                 for form in ("split", "joined")}
    ovP, ovG = {}, {}
    for form in ("split", "joined"):
        for o in objs:
            for k in pools_all[form][o]:
                if (k, o) not in ovP:
                    ovP[(k, o)] = int((M[k] & P[o]).sum())
                    ovG[(k, o)] = int((M[k] & G[o]).sum()) if gpx[o] else 0

    def select(v, tau, form, o):
        pool = [k for k in pools_all[form][o] if prob[k] > tau]
        if not pool:
            return None, []
        if v == "best_ov":
            if proj_px[o] == 0:
                return None, []
            k = max(pool, key=lambda k: (ovP[(k, o)], prob[k]))
            return (None, []) if ovP[(k, o)] == 0 else (M[k], [k])
        if v == "best_iou":
            if proj_px[o] == 0:
                return None, []
            iou = {k: ovP[(k, o)] / (area[k] + proj_px[o] - ovP[(k, o)]) for k in pool}
            k = max(pool, key=lambda k: (iou[k], prob[k]))
            return (None, []) if iou[k] == 0 else (M[k], [k])
        if v == "union_maj":
            ks = [k for k in pool if ovP[(k, o)] > 0 and ovP[(k, o)] > 0.5 * a_vis[k]]
        elif v == "union_all":
            ks = pool
        elif v == "oracle":
            if not gpx[o]:
                return None, []
            iou = {k: ovG[(k, o)] / (area[k] + gpx[o] - ovG[(k, o)]) for k in pool}
            k = max(pool, key=lambda k: (iou[k], prob[k]))
            ks = [] if iou[k] == 0 else [k]
        else:
            raise ValueError(v)
        if not ks:
            return None, []
        s = np.zeros(hw, bool)
        for k in ks:
            s |= M[k]
        return s, ks

    out = dict(scene=scene, t=t, cam=cam, objs=objs, base={o: list(base[o]) for o in base}, gt_px=gpx,
               proj_px=proj_px, gtcls=gtcls, n_det=N, pairs={}, empty_gt={}, leak={}, det={})
    # detector reach per scored pair at each (tau, form): pool size, best-instance IoU vs GT, union coverage of missed px
    for tau, form in SELS:
        for o in base:
            pool = [k for k in pools_all[form][o] if prob[k] > tau]
            ious = [ovG[(k, o)] / (area[k] + gpx[o] - ovG[(k, o)]) for k in pool]
            u = np.zeros(hw, bool)
            for k in pool:
                u |= M[k]
            out["det"].setdefault(o, {})[f"{tau}|{form}"] = dict(
                n_pool=len(pool), best_iou=max(ious) if ious else 0.0, union_iou=float((u & G[o]).sum() / max(1, (u | G[o]).sum())),
                miss_cov=int((u & miss[o]).sum()), miss_cov_invis=int((u & miss[o] & invis).sum()),
                miss_cov_cls=[int((u & miss[o] & (cls == c)).sum()) for c in (1, 2, 3)])
            del u
    for v in VARIANTS:
        for tau, form in SELS:
            S, Apre, fun, sel = {}, {}, {}, {}
            for o in objs:
                s, ks = select(v, tau, form, o)
                if s is None:
                    continue
                S[o], sel[o] = s, ks
                claimed_other = (cnt - D[o]) > 0
                C = s & invis & ~D[o] & ~claimed_other
                Apre[o] = connected_add(C, D[o])
                if (tau, form) == MAIN_SEL:
                    ns = int(s.sum())
                    if ns:
                        labs, cn = np.unique(gt[s], return_counts=True)
                        si = s & invis
                        out["leak"].setdefault(v, {})[o] = dict(
                            s_px=ns, s_on_o=int((s & G[o]).sum()), dom=int(labs[int(np.argmax(cn))]), dom_px=int(cn.max()),
                            s_on_bg=int(cn[labs == 0].sum()) if (labs == 0).any() else 0, s_invis=int(si.sum()),
                            s_invis_on_o=int((si & G[o]).sum()), n_sel=len(ks), prob=[float(prob[k]) for k in ks])
                    if o in base:
                        m = miss[o]
                        fun[o] = [int(m.sum()), int((s & m).sum()), int((s & m & invis).sum()), int((C & m).sum()),
                                  int((Apre[o] & m).sum())]
                del C
            jf_cache = {}
            for (ct, cf, mp) in configs(v):
                if (ct, cf) != (tau, form):
                    continue
                act = [o for o in Apre if v not in PROJ_VARIANTS or proj_px[o] >= mp]
                ca = np.zeros(hw, np.uint8)
                for o in act:
                    ca += Apre[o]
                kk = key(v, tau, form, mp)
                for o in objs:
                    A = Apre[o] & (ca == 1) if o in act else np.zeros(hw, bool)
                    na = int(A.sum())
                    if not gpx[o]:
                        if na:
                            out["empty_gt"].setdefault(kk, {})[o] = na
                        continue
                    r = out["pairs"].setdefault(o, {})
                    if na == 0:
                        r[kk] = [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
                    r[kk] = [float(dJ), float(dF), na, n_on, na - n_on - n_bg, n_bg] + ca_cls + on_cls + \
                        [int((A & seeded & ~G[o]).sum())]
                    if (tau, form, mp) == MAIN_SEL + (MAIN_MP,) and o in fun:
                        fun[o].append(n_on)
                if (tau, form, mp) == MAIN_SEL + (MAIN_MP,):
                    for o in fun:
                        if len(fun[o]) == 5:
                            fun[o].append(0)
                        out["pairs"].setdefault(o, {})[f"funnel|{v}"] = fun[o]
                    for o in sel:
                        if o in base:
                            out["pairs"].setdefault(o, {})[f"sel|{v}"] = sel[o]
            del S, Apre
    out["sel_iou_gt"] = {o: {k: ovG[(k, o)] / (area[k] + gpx[o] - ovG[(k, o)]) for k in pools_all["split"][o]} for o in base}
    out["prob"] = [float(x) for x in prob]
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
                assert os.path.isfile(os.path.join(DETS, s, f"{t:06d}", cam, "DONE")), (s, t, cam)
                jobs.append((s, t, cam, S["ref_frame"]))
    t0 = time.time()
    res = []
    with Pool(a.procs) as p:
        for i, r in enumerate(p.imap_unordered(view, jobs)):
            res.append(r)
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    cfg = {v: [list(c) for c in configs(v)] for v in VARIANTS}
    json.dump(dict(configs=cfg, variants=VARIANTS, main=list(MAIN_SEL) + [MAIN_MP], sels=SELS, views=res,
                   seconds=time.time() - t0), open(a.out, "w"))
    print(f"done {len(res)} views {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

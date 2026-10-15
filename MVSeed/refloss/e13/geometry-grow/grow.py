#!/usr/bin/env python3
"""S1-E13 candidate D (geometry-grow): multi-view geometric-continuity growth of the reference-labelled objects into
reference-invisible surface, projected into the target view, add-only composition with the adopted geoset_rep_x DEV
masks, and DEV scoring.  Design and constants: REGISTRATION.txt (fixed before any growth/score).  Host numpy/cv2 only.
DEV ONLY: frames of geomem/runs/_e13_dev/index.json; asserts t != ref_frame; the keyframe is never read.

One job = one dev frame (scene, t): every rig camera's Pi3 cache + image at t, c_ini's input mask at t; both targets.
Per target view and variant: per scored pair the same 13-field record as candidate C's compose_score.py
[dJ, dF, added, on_o, on_other, on_bg, add_out, add_occ, add_nod, on_out, on_occ, on_nod, on_other_seeded],
the raw-region leak record, and (primary) a funnel of the missed pixels.

  PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python -I grow.py --out OUTDIR [--procs 16] [--scenes A,B]
        [--set TAU_2=0.01 ...] [--variants mv,mv_nosplat,tv,mv_noapp,mv_plane] [--save-masks]
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
from geomem.pointmap import local_points, to_reference  # noqa: E402
from mvseg2d.io import load_labelmap, load_meta, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
DEVROOT = os.path.join(S1, "geomem", "runs", "_e13_dev")
CONF_MIN, OCC_TOL = 0.1, 0.01
INVIS = 255
DEFAULT = dict(TAU_2=0.02, LR=1.0, K_LINK=4, TAU_X=0.02, M_LINK=5, R_ERODE=2, EPS_PLANE=0.01, NBINS=8)
ALL_VARIANTS = ("mv", "mv_nosplat", "tv", "mv_noapp", "mv_plane")
PRIMARY = "mv"
RANSAC_N, RANSAC_MINPTS, RANSAC_FRAC, RANSAC_SUB = 256, 100, 0.85, 20000


# ----------------------------------------------------------------------------------------------- geometry helpers
def load_cam(scene, t, cam):
    z = np.load(os.path.join(DATA, scene, "Cache", "pi3_frames", f"{t:06d}", f"{cam}.npz"))
    d = z["depth"].astype(np.float32)
    return dict(depth=d, valid=(z["conf"] > CONF_MIN) & (d > 0), K=z["K"].astype(np.float64), c2w=z["c2w"].astype(np.float64))


def project(p, K, hw_ref, depth_ref=None, tol=OCC_TOL):
    """numpy port of geomem.warp_torch.reproject_coords (float32) -> (valid, row, col)."""
    h_ref, w_ref = hw_ref
    z = p[..., 2]
    zs = np.where(np.abs(z) < 1e-9, np.float32(1e-9), z)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = K[0, 0] * p[..., 0] / zs + K[0, 2]
        v = K[1, 1] * p[..., 1] / zs + K[1, 2]
        valid = (z > 0) & (u >= -0.5) & (u <= w_ref - 0.5) & (v >= -0.5) & (v <= h_ref - 0.5) & np.isfinite(u) & np.isfinite(v)
    u = np.nan_to_num(u, nan=0.0, posinf=1e9, neginf=-1e9)
    v = np.nan_to_num(v, nan=0.0, posinf=1e9, neginf=-1e9)
    col = np.clip(np.round(u), 0, w_ref - 1).astype(np.int64)
    row = np.clip(np.round(v), 0, h_ref - 1).astype(np.int64)
    if depth_ref is not None:
        d = depth_ref[row, col]
        valid = valid & ((d <= 0) | (z <= d * np.float32(1.0 + tol)))
    return valid, row, col


def full_K(K, hw_work, hw_full):
    h, w = hw_full
    sx, sy = w / hw_work[1], h / hw_work[0]
    return np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                     [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)


def cut_mask(depth, valid, tau2):
    """Depth-discontinuity pixels: |2nd difference of inverse depth| > tau2 * inverse depth (horizontal or vertical)."""
    w = np.where(valid, 1.0 / np.maximum(depth, 1e-12), 0.0).astype(np.float64)
    cut = np.zeros(depth.shape, bool)
    c = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
    cut[:, 1:-1] |= c & (np.abs(w[:, :-2] - 2 * w[:, 1:-1] + w[:, 2:]) > tau2 * w[:, 1:-1])
    c = valid[1:-1] & valid[:-2] & valid[2:]
    cut[1:-1] |= c & (np.abs(w[:-2] - 2 * w[1:-1] + w[2:]) > tau2 * w[1:-1])
    return cut


def color_bins(path, hw_work, nb):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None, path
    img = cv2.resize(img, (hw_work[1], hw_work[0]), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int32)
    sh = 8 - int(np.log2(nb))
    return ((lab[..., 0] >> sh) * nb * nb + (lab[..., 1] >> sh) * nb + (lab[..., 2] >> sh)).astype(np.int32), img.shape[:2]


def box3(h):
    nb = h.shape[0]
    p = np.pad(h, 1)
    out = np.zeros_like(h)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += p[a:a + nb, b:b + nb, c:c + nb]
    return out


def accept_bins(bins_c, lab_c, o, nb, lr):
    B = nb ** 3
    ho = box3(np.bincount(bins_c[lab_c == o], minlength=B).reshape(nb, nb, nb).astype(np.float64)).ravel()
    hb = box3(np.bincount(bins_c[lab_c != o], minlength=B).reshape(nb, nb, nb).astype(np.float64)).ravel()
    if ho.sum() == 0:
        return np.zeros(B, bool)
    ho /= ho.sum()
    hb = hb / hb.sum() if hb.sum() else hb
    return (ho >= lr * hb) & (ho > 0)


def uf_roots(n, a, b):
    """Connected-component roots (min node id) of an undirected graph on n nodes with edges (a, b); numpy hook+jump."""
    parent = np.arange(n, dtype=np.int64)
    if a.size == 0:
        return parent
    while True:
        pa, pb = parent[a], parent[b]
        ne = pa != pb
        if not ne.any():
            return parent
        lo, hi = np.minimum(pa[ne], pb[ne]), np.maximum(pa[ne], pb[ne])
        np.minimum.at(parent, hi, lo)
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp


def fit_plane(P, eps):
    """GT-free planarity of an object's reference 3D points: RANSAC (fixed seed) -> (planar, inlier_frac, n, d)."""
    n = len(P)
    if n < RANSAC_MINPTS:
        return False, None, None, None
    rng = np.random.default_rng(0)
    Q = P[rng.choice(n, RANSAC_SUB, replace=False)] if n > RANSAC_SUB else P
    nq = np.linalg.norm(Q, axis=1)
    best, bn, bd = -1, None, None
    for _ in range(RANSAC_N):
        i = rng.choice(len(Q), 3, replace=False)
        nv = np.cross(Q[i[1]] - Q[i[0]], Q[i[2]] - Q[i[0]])
        nn = np.linalg.norm(nv)
        if nn < 1e-12:
            continue
        nv /= nn
        d = -nv @ Q[i[0]]
        k = int((np.abs(Q @ nv + d) <= eps * nq).sum())
        if k > best:
            best, bn, bd = k, nv, d
    if bn is None:
        return False, None, None, None
    inl = np.abs(P @ bn + bd) <= eps * np.linalg.norm(P, axis=1)
    if inl.sum() >= 3:
        m = P[inl].mean(0)
        _, _, vt = np.linalg.svd(P[inl] - m, full_matrices=False)
        bn = vt[-1]
        bd = -bn @ m
        inl = np.abs(P @ bn + bd) <= eps * np.linalg.norm(P, axis=1)
    frac = float(inl.mean())
    return frac >= RANSAC_FRAC, frac, bn, float(bd)


# ----------------------------------------------------------------------------------------------- composition helpers
def connected_add(C, D):
    """candidate C's rule: pixels of C in 8-connected components of (C | D) that hold a D pixel."""
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


def hsh(m):
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return "empty"
    return hashlib.blake2b(np.concatenate([ys, xs]).astype(np.int32).tobytes(), digest_size=16).hexdigest()


def up(m, hw):
    return cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def rd(p, hw):
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    m = read_pred_mask(p)
    assert m.shape == tuple(hw), (p, m.shape, hw)
    return m


# ----------------------------------------------------------------------------------------------- one frame
def frame(job):
    scene, t, cfg, variants, outdir, save_masks = job
    t0 = time.time()
    idx = json.load(open(os.path.join(DEVROOT, "index.json")))["scenes"][scene]
    meta = load_meta(DATA, scene)
    ref_frame, c = int(meta["ref_frame"]), meta["ref_cam"]
    assert t != ref_frame and idx["ref_frame"] == ref_frame and idx["ref_cam"] == c, (scene, t)
    targets = list(idx["eval_cams"])
    basic = [int(o) for o in idx["basic"]]
    cdir = os.path.join(DATA, scene, "Cache", "pi3_frames", f"{t:06d}")
    man = json.load(open(os.path.join(cdir, "MANIFEST.json")))
    assert man["ref_cam"] == c
    cams = list(man["cams"])
    assert all(x in cams for x in targets)
    hw_w = tuple(man["work_hw"])
    fdir = os.path.join(DATA, scene, "frames", f"{t:06d}")
    Lc = load_labelmap(os.path.join(fdir, "masks", f"{c}.png"))
    Lc = np.where(np.isin(Lc, basic), Lc, 0).astype(np.uint8)
    assert Lc.max() < 254
    H, W = Lc.shape
    objs = [o for o in basic if (Lc == o).any()]
    nb = int(cfg["NBINS"])
    # ---- reference camera
    zc = load_cam(scene, t, c)
    assert zc["depth"].shape == hw_w
    Kc_full = full_K(zc["K"], hw_w, (H, W))
    dc_full = cv2.resize(zc["depth"], (W, H), interpolation=cv2.INTER_NEAREST)
    r = int(round(cfg["R_ERODE"] * W / hw_w[1]))
    if r > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        Lero = np.zeros_like(Lc)
        for o in objs:
            Lero[cv2.erode((Lc == o).astype(np.uint8), se).astype(bool)] = o
    else:
        Lero = Lc
    Lc_w = cv2.resize(Lc, (hw_w[1], hw_w[0]), interpolation=cv2.INTER_NEAREST)
    bins_c, _ = color_bins(os.path.join(fdir, "images", f"{c}.jpg"), hw_w, nb)
    loc_c = local_points(zc["depth"], zc["K"]).astype(np.float32)
    # ---- every other camera
    others = [x for x in cams if x != c]
    V = {}
    for x in others:
        z = load_cam(scene, t, x)
        assert z["depth"].shape == hw_w, (scene, x)
        loc64 = local_points(z["depth"], z["K"])
        pc = to_reference(loc64, z["c2w"], zc["c2w"]).astype(np.float32)      # exactly reproj_inputs' float path
        loc = loc64.astype(np.float32)
        del loc64
        visg, row, col = project(pc, Kc_full, (H, W), dc_full)
        vis = visg & z["valid"]
        ins, _, _ = project(pc, Kc_full, (H, W), None)
        ins &= z["valid"]
        plab = np.full(hw_w, INVIS, np.uint8)
        plab[vis] = Lc[row[vis], col[vis]]
        pero = np.full(hw_w, INVIS, np.uint8)
        pero[vis] = Lero[row[vis], col[vis]]
        bins, _ = color_bins(os.path.join(fdir, "images", f"{x}.jpg"), hw_w, nb)
        V[x] = dict(z=z, loc=loc, pc=pc, vis=vis, ins=ins, plab=plab, pero=pero, cut=cut_mask(z["depth"], z["valid"], cfg["TAU_2"]),
                    bins=bins, inv=z["valid"] & ~vis)
    # ---- cross-camera links (k nearest, symmetric union, c_ini excluded)
    cen = {x: V[x]["z"]["c2w"][:3, 3] for x in others}
    pairs = set()
    for x in others:
        nn = sorted((np.linalg.norm(cen[y] - cen[x]), y) for y in others if y != x)[:int(cfg["K_LINK"])]
        for _, y in nn:
            pairs.add(tuple(sorted((x, y))))
    links = []
    n_corr = 0
    for x, y in sorted(pairs):
        zx, zy = V[x]["z"], V[y]["z"]
        p = to_reference(V[x]["loc"], zx["c2w"], zy["c2w"]).astype(np.float32)
        ok, row, col = project(p, zy["K"].astype(np.float32), hw_w, None)
        ok &= zx["valid"]
        iX = np.flatnonzero(ok)
        rr, cc = row.ravel()[iX], col.ravel()[iX]
        dy = zy["depth"][rr, cc]
        zz = p[..., 2].ravel()[iX]
        good = zy["valid"][rr, cc] & (np.abs(zz - dy) <= cfg["TAU_X"] * dy)
        links.append((x, y, iX[good].astype(np.int32), (rr[good] * hw_w[1] + cc[good]).astype(np.int32)))
        n_corr += int(good.sum())
    # ---- no-depth splat into each target: nearest point of all other cameras (incl. c_ini) at T's invalid pixels
    splat = {}
    for T in targets:
        zT = V[T]["z"]
        holes = ~zT["valid"]
        pix_l, z_l, cam_l, src_l = [], [], [], []
        srcs = [c] + [x for x in others if x != T]
        for ci, x in enumerate(srcs):
            zx = zc if x == c else V[x]["z"]
            loc = loc_c if x == c else V[x]["loc"]
            p = to_reference(loc, zx["c2w"], zT["c2w"]).astype(np.float32)
            ok, row, col = project(p, zT["K"].astype(np.float32), hw_w, None)
            ok &= zx["valid"]
            ok &= holes[row, col]
            s = np.flatnonzero(ok)
            pix_l.append(row.ravel()[s] * hw_w[1] + col.ravel()[s])
            z_l.append(p[..., 2].ravel()[s])
            cam_l.append(np.full(s.size, ci, np.int16))
            src_l.append(s.astype(np.int32))
        pix, zz = np.concatenate(pix_l), np.concatenate(z_l)
        camv, srcv = np.concatenate(cam_l), np.concatenate(src_l)
        order = np.lexsort((zz, pix))
        pix, camv, srcv = pix[order], camv[order], srcv[order]
        first = np.ones(pix.size, bool)
        first[1:] = pix[1:] != pix[:-1]
        splat[T] = dict(pix=pix[first], cam=camv[first], src=srcv[first], srcs=srcs)
    t_geo = time.time() - t0
    # ---- per object growth
    grow_specs = {}
    if any(v in variants for v in ("mv", "mv_nosplat", "tv")):
        grow_specs["app"] = ("mv", "mv_nosplat", "tv")
    if "mv_noapp" in variants:
        grow_specs["noapp"] = ("mv_noapp",)
    if "mv_plane" in variants:
        grow_specs["plane"] = ("mv_plane",)
    G = {T: {v: {} for v in variants} for T in targets}            # work-res grown masks in each target
    diag = {T: {} for T in targets}
    objdiag = {}
    for o in objs:
        acc = accept_bins(bins_c, Lc_w, o, nb, cfg["LR"])
        Pc = loc_c[(Lc_w == o) & zc["valid"]].astype(np.float64)
        planar, pfrac, pn, pd = fit_plane(Pc, cfg["EPS_PLANE"]) if "mv_plane" in variants else (False, None, None, None)
        od = dict(n_seed=0, planar=bool(planar), plane_frac=pfrac, acc_bins=int(acc.sum()))
        for spec, vs in grow_specs.items():
            use_plane = spec == "plane" and planar
            if spec == "plane" and not planar:                     # non-planar objects: identical to the primary
                if all("mv" in G[T] and o in G[T]["mv"] for T in targets):
                    for T in targets:
                        G[T]["mv_plane"][o] = G[T]["mv"][o]
                    continue
            comps, off, N = {}, {}, 1
            seeded = []
            for x in others:
                X = V[x]
                inv_ok = X["plab"] == INVIS
                if spec != "noapp":
                    inv_ok = inv_ok & acc[X["bins"]]
                if use_plane:
                    pc = X["pc"]
                    inv_ok = inv_ok & (np.abs(pc @ pn.astype(np.float32) + np.float32(pd)) <= cfg["EPS_PLANE"] * np.linalg.norm(pc, axis=-1))
                trav = X["z"]["valid"] & ~X["cut"] & ((X["plab"] == o) | inv_ok)
                n, lab = cv2.connectedComponents(trav.astype(np.uint8), connectivity=4)
                comps[x], off[x] = lab, N
                sd = np.unique(lab[trav & (X["pero"] == o)])
                sd = sd[sd > 0]
                seeded.append(N + sd)
                if spec == "app":
                    od["n_seed"] += int((trav & (X["pero"] == o)).sum())
                    if x in targets:
                        diag[x].setdefault(o, {})["trav"] = trav
                        diag[x][o]["acc"] = acc[X["bins"]]
                        diag[x][o]["seeds_T"] = int((trav & (X["pero"] == o)).sum())
                N += n
            seeded = np.concatenate(seeded)
            if "mv" in vs or "mv_noapp" in vs or "mv_plane" in vs:
                ea, eb = [], []
                for x, y, iX, iY in links:
                    a = comps[x].ravel()[iX]
                    b = comps[y].ravel()[iY]
                    m = (a > 0) & (b > 0)
                    if not m.any():
                        continue
                    key = (off[x] + a[m]).astype(np.int64) * N + (off[y] + b[m])
                    u, cnt = np.unique(key, return_counts=True)
                    u = u[cnt >= cfg["M_LINK"]]
                    ea.append(u // N)
                    eb.append(u % N)
                ea = np.concatenate(ea) if ea else np.zeros(0, np.int64)
                eb = np.concatenate(eb) if eb else np.zeros(0, np.int64)
                root = uf_roots(N, ea, eb)
                is_g = np.zeros(N, bool)
                if seeded.size:
                    is_g = np.isin(root, np.unique(root[seeded]))
                is_g[[off[x] for x in others]] = False              # component 0 of every camera = not traversable
                if spec == "app":
                    od["n_links"] = int(ea.size)
                for T in targets:
                    gT = is_g[off[T] + comps[T]]
                    vmv = {"app": "mv", "noapp": "mv_noapp", "plane": "mv_plane"}[spec]
                    sp = splat[T]
                    gs = np.zeros(sp["pix"].size, bool)
                    for ci, x in enumerate(sp["srcs"]):
                        m = sp["cam"] == ci
                        if not m.any():
                            continue
                        if x == c:
                            gs[m] = Lc_w.ravel()[sp["src"][m]] == o
                        else:
                            gs[m] = is_g[off[x] + comps[x].ravel()[sp["src"][m]]]
                    gfull = gT.copy()
                    gfull.ravel()[sp["pix"][gs]] = True
                    if vmv in variants:
                        G[T][vmv][o] = gfull
                    if spec == "app" and "mv_nosplat" in variants:
                        G[T]["mv_nosplat"][o] = gT
                    if spec == "app":
                        diag[T].setdefault(o, {})["splat_src"] = np.zeros(hw_w, bool)
                        diag[T][o]["splat_src"].ravel()[sp["pix"]] = True
            if spec == "app" and "tv" in variants:
                for T in targets:
                    lab = comps[T]
                    sd = np.unique(lab[(V[T]["pero"] == o) & (lab > 0)])
                    G[T]["tv"][o] = np.isin(lab, sd[sd > 0])
            del comps
        objdiag[o] = od
    t_grow = time.time() - t0 - t_geo
    # ---- composition + scoring per target view (full resolution)
    views = []
    for T in targets:
        views.append(score_view(scene, t, T, objs, V[T], G[T], diag[T], objdiag, variants, (H, W), hw_w))
        if save_masks:
            d = os.path.join(outdir, "masks", scene, f"{t:06d}")
            os.makedirs(d, exist_ok=True)
            arr = {f"{v}|{o}": np.packbits(G[T][v][o]) for v in variants for o in objs if G[T][v].get(o) is not None}
            np.savez_compressed(os.path.join(d, f"{T}.npz"), hw=np.array(hw_w), **arr)
    out = dict(scene=scene, t=t, cfg=cfg, variants=list(variants), n_cams=len(cams), n_pairs_link=len(pairs), n_corr=n_corr,
               objdiag={str(o): v for o, v in objdiag.items()}, views=views,
               seconds=dict(geo=t_geo, grow=t_grow, total=time.time() - t0))
    with open(os.path.join(outdir, "frames", f"{scene}_{t:06d}.json"), "w") as f:
        json.dump(out, f)
    return scene, t, out["seconds"]


def score_view(scene, t, T, objs, VT, GT, dg, objdiag, variants, hw, hw_w):
    H, W = hw
    gt = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{T}.png"))
    assert gt.shape == hw
    valid = up(VT["z"]["valid"], hw)
    vis = up(VT["vis"], hw)
    ins = up(VT["ins"], hw)
    cls = np.full(hw, 3, np.uint8)
    cls[valid & ~ins] = 1
    cls[ins & ~vis] = 2
    cls[vis] = 0
    invis = ~vis
    D = {o: rd(os.path.join(DEVROOT, scene, f"{t:06d}", T, f"{o}.png"), hw) for o in objs}
    cnt = np.zeros(hw, np.uint8)
    for o in objs:
        cnt += D[o]
    bp = bound_pix(H, W)
    se = disk(bp)
    G_ = {o: gt == o for o in objs}
    seeded = np.isin(gt, objs)
    gpx = {o: int(G_[o].sum()) for o in objs}
    base = {o: jf_pair(G_[o], D[o], bp, se) for o in objs if gpx[o]}
    gtcls = {o: [int((G_[o] & (cls == k)).sum()) for k in (0, 1, 2, 3)] for o in base}
    out = dict(scene=scene, t=t, cam=T, objs=objs, base={o: list(base[o]) for o in base}, gt_px=gpx, gtcls=gtcls,
               pairs={}, empty_gt={}, leak={}, funnel={}, objinfo={})
    for o in objs:
        out["objinfo"][o] = dict(seeds_T=(dg.get(o) or {}).get("seeds_T", 0))
    for v in variants:
        Apre = {}
        for o in objs:
            g = GT[v].get(o)
            has = g is not None and bool(g.any())
            if not has and not (v == PRIMARY and gpx[o]):
                continue
            gf = up(g, hw) if has else np.zeros(hw, bool)
            raw = gf & invis
            nr = int(raw.sum())
            if nr:
                labs, cn = np.unique(gt[raw], return_counts=True)
                out["leak"].setdefault(v, {})[o] = dict(s_px=nr, s_on_o=int((raw & G_[o]).sum()), dom=int(labs[int(np.argmax(cn))]),
                                                       dom_px=int(cn.max()), s_on_bg=int(cn[labs == 0].sum()),
                                                       by_cls=[int((raw & (cls == k)).sum()) for k in (1, 2, 3)],
                                                       on_o_cls=[int((raw & G_[o] & (cls == k)).sum()) for k in (1, 2, 3)])
            claimed_other = (cnt - D[o]) > 0
            C = raw & ~D[o] & ~claimed_other
            if has:
                Apre[o] = connected_add(C, D[o])
            if v == PRIMARY and gpx[o]:
                miss = G_[o] & ~D[o]
                d = dg.get(o, {})
                trav = up(d["trav"], hw) if "trav" in d else np.zeros(hw, bool)
                acc = up(d["acc"], hw) if "acc" in d else np.zeros(hw, bool)
                ssrc = up(d["splat_src"], hw) if "splat_src" in d else np.zeros(hw, bool)
                mi = miss & invis
                out["funnel"][o] = dict(
                    missed=int(miss.sum()), missed_invis_valid=int((mi & valid).sum()), missed_nodepth=int((miss & ~valid).sum()),
                    acc=int((mi & valid & acc).sum()), trav=int((mi & trav).sum()), grown_valid=int((mi & valid & gf).sum()),
                    nod_splat_src=int((miss & ~valid & ssrc).sum()), grown_nod=int((miss & ~valid & gf).sum()),
                    in_G=int((miss & gf).sum()), invis=int((mi & gf).sum()), unclaimed=int((C & miss).sum()),
                    connected=int((Apre[o] & miss).sum()) if o in Apre else 0)
        ca = np.zeros(hw, np.uint8)
        for o in Apre:
            ca += Apre[o]
        for o in objs:
            A = Apre[o] & (ca == 1) if o in Apre else np.zeros(hw, bool)
            na = int(A.sum())
            if not gpx[o]:
                if na:
                    out["empty_gt"].setdefault(v, {})[o] = na
                continue
            if v == PRIMARY and o in out["funnel"]:
                out["funnel"][o]["uncontested"] = int((A & G_[o]).sum())
            if na == 0:
                out["pairs"].setdefault(o, {})[v] = [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                continue
            J, F = jf_pair(G_[o], D[o] | A, bp, se)
            on_o = A & G_[o]
            lab_a = gt[A]
            n_on = int(on_o.sum())
            n_bg = int((lab_a == 0).sum())
            out["pairs"].setdefault(o, {})[v] = [float(J - base[o][0]), float(F - base[o][1]), na, n_on, na - n_on - n_bg, n_bg] + \
                [int((A & (cls == k)).sum()) for k in (1, 2, 3)] + [int((on_o & (cls == k)).sum()) for k in (1, 2, 3)] + \
                [int((A & seeded & ~G_[o]).sum())]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--frames", default="all", help="comma list of t (debug)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--save-masks", action="store_true")
    a = ap.parse_args()
    cfg = dict(DEFAULT)
    for kv in a.set:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = type(DEFAULT[k])(float(v)) if isinstance(DEFAULT[k], float) else int(v)
    variants = tuple(a.variants.split(","))
    assert all(v in ALL_VARIANTS for v in variants)
    idx = json.load(open(os.path.join(DEVROOT, "index.json")))
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    json.dump(dict(cfg=cfg, variants=variants), open(os.path.join(a.out, "config.json"), "w"))
    jobs = []
    for s, S in sorted(idx["scenes"].items()):
        if a.scenes != "all" and s not in a.scenes.split(","):
            continue
        for t in S["frames"]:
            if a.frames != "all" and str(t) not in a.frames.split(","):
                continue
            assert t != S["ref_frame"]
            if os.path.isfile(os.path.join(a.out, "frames", f"{s}_{t:06d}.json")):
                continue
            jobs.append((s, t, cfg, variants, a.out, a.save_masks))
    # biggest scenes first
    ncam = {s: len(load_meta(DATA, s)["cams"]) for s in idx["scenes"]}
    jobs.sort(key=lambda j: -ncam[j[0]])
    t0 = time.time()
    with Pool(a.procs) as p:
        for i, (s, t, sec) in enumerate(p.imap_unordered(frame, jobs)):
            print(f"{i + 1}/{len(jobs)} {s} {t} geo {sec['geo']:.1f}s grow {sec['grow']:.1f}s total {sec['total']:.1f}s "
                  f"[{time.time() - t0:.0f}s]", file=sys.stderr, flush=True)
    print(f"done {len(jobs)} frames {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

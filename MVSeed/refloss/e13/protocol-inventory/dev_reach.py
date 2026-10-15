#!/usr/bin/env python3
"""Protocol inventory, part 2 (host, numpy, read-only): on the DEV set (non-keyframe frames), how much of the
target GT that the reference camera c_ini cannot see could each information source reach?

DEV rows: every scene, every train frame t != meta['ref_frame'] (20 per scene), target X = each annotated camera
other than c_ini (2 per scene), object o = basic-set id (c_ini frame-0 census, docs/raw/muvod_object_sets.json,
i.e. the protocol's object set) present in X's label map at t.  The keyframe (meta['ref_frame']) is never read:
no label map, no Pi3 cache and no prediction of it is opened (asserted).

Per target pixel of G = (L_X == o) at full resolution, the S1-E11 classes (MVSeed/refloss/e11/extent_vis.py, ported
from methods.geomem.run.reproj_inputs + project_reference_masks + geomem.warp_torch.reproject_coords, OCC_TOL 0.01,
Pi3 validity conf > 0.1 & depth > 0, reference = c_ini at the SAME frame t):
    vis / outside / occluded / nodepth          (exclusive; R = outside | occluded | nodepth = "beyond c_ini")
Reach of each source on R:
    B_vis     visible from the third annotated camera B (same machinery, B as reference)       [2nd annotated ref cam]
    B_ongt    B_vis and the nearest B label at the reprojected position is o                    [label transfer from B's GT]
    any_vis   visible from >= 1 rig camera other than c_ini and X (work resolution, nearest-upsampled)  [all cams' RGB]
    nonann_vis   same over the NON-annotated cameras only
    occ_dyn   occluded and c_ini's label at the reprojected position is a 'dynamic' object (the occluder can move)
    occ_self  occluded by the same object (self-occlusion)
    occ_tvis  occluded, object type static/env, and visible from c_ini at >= 1 OTHER DEV frame t' (Pi3 depth of
              c_ini at t', scale-aligned to frame t's Pi3 run by the median depth ratio; requires valid depth at t')
                                                                                         [c_ini RGB at other times]
    nod_tval  nodepth at t but the same X pixel has valid Pi3 depth at >= 1 other DEV frame    [multi-frame geometry]
Label-name ambiguity (computed in the summary from objects_labels.json): unique label in the scene / another object
with the same label present in X's label map at t.

    PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python -I dev_reach.py OUT.json [--procs 8] [--scenes A,B]
"""
import argparse
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
from mvseg2d.io import load_labelmap, load_meta  # noqa: E402
from geomem.pointmap import CONF_MIN, local_points, to_reference  # noqa: E402

DATA = os.path.join(S1, "Data")
OCC_TOL = 0.01
SETS = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
CLS = ("vis", "outside", "occluded", "nodepth")


def npz(scene, t, cam):
    z = np.load(os.path.join(DATA, scene, "Cache", "pi3_frames", f"{int(t):06d}", f"{cam}.npz"))
    return {k: z[k] for k in ("depth", "conf", "K", "c2w")}


def project(p, K, hw_ref, depth_ref=None, tol=OCC_TOL):
    """numpy port of geomem.warp_torch.reproject_coords (float32): -> (valid, row, col)."""
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


def reproj_full(zu, zr, hw):
    """numpy port of methods.geomem.run.reproj_inputs."""
    h, w = hw
    pts = to_reference(local_points(zu["depth"], zu["K"]), zu["c2w"], zr["c2w"]).astype(np.float32)
    valid_u = ((zu["conf"] > CONF_MIN) & (zu["depth"] > 0)).astype(np.uint8)
    pts_full = cv2.resize(pts, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_full = cv2.resize(valid_u, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    depth_ref_full = cv2.resize(zr["depth"], (w, h), interpolation=cv2.INTER_NEAREST)
    hw_work = zr["depth"].shape
    sx, sy = w / hw_work[1], h / hw_work[0]
    K = zr["K"]
    K_full = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                       [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)
    return pts_full, valid_full, depth_ref_full, K_full


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def run_scene(scene):
    t0 = time.time()
    meta = load_meta(DATA, scene)
    ref_frame, c = meta["ref_frame"], meta["ref_cam"]
    cams, ann = list(meta["cams"]), list(meta["mask_cams"])
    labels = {int(k): v for k, v in json.load(open(os.path.join(DATA, scene, "objects_labels.json"))).items()}
    basic = set(int(i) for i in SETS[scene]["seed_ids_c_ini"])
    dev = [t for t in meta["train_frames"] if t != ref_frame]
    assert ref_frame not in dev and len(dev) == 20, (scene, dev)
    targets = [x for x in ann if x != c]
    nonann = [n for n in cams if n not in ann]
    typ = np.zeros(256, np.int8)          # 0 unlabelled, 1 dynamic, 2 static, 3 env
    for k, v in labels.items():
        typ[k] = {"dynamic": 1, "static": 2, "env": 3}[v["type"]]
    # c_ini depth of every DEV frame (work resolution) for the cross-time test
    dc = {t: npz(scene, t, c) for t in dev}
    rows, pose = [], {}
    for x in targets:
        b = [a for a in ann if a not in (c, x)][0]
        rel = {}
        for t in dev:
            zc, zx, zb = dc[t], npz(scene, t, x), npz(scene, t, b)
            L_x = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{x}.png"))
            L_c = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{c}.png"))
            L_b = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{b}.png"))
            hw = L_x.shape
            assert L_c.shape == hw and L_b.shape == hw
            T = np.linalg.inv(zc["c2w"]) @ zx["c2w"]
            rel[t] = T
            objs = sorted(o for o in np.unique(L_x) if o in basic)
            if not objs:
                continue
            # ---- S1-E11 classes w.r.t. c_ini at the same frame (full resolution)
            pts, valid_full, depth_ref, K_full = reproj_full(zx, zc, hw)
            visg, row, col = project(pts, K_full, hw, depth_ref)
            vis = visg & valid_full
            ins, _, _ = project(pts, K_full, hw, None)
            ins = ins & valid_full
            cls = np.full(hw, 3, np.int8)                    # nodepth
            cls[valid_full & ~ins] = 1                       # outside
            cls[ins & ~vis] = 2                              # occluded
            cls[vis] = 0
            occ = cls == 2
            # occluder: c_ini's label where the occluded pixel lands
            occ_id = np.zeros(hw, np.int32)
            occ_id[occ] = L_c[row[occ], col[occ]]
            occ_dyn = occ & (typ[occ_id] == 1) & (occ_id != L_x)
            occ_self = occ & (occ_id == L_x)
            occ_unl = occ & (occ_id == 0)
            # ---- third annotated camera B
            ptsb, _, depth_b, K_b = reproj_full(zx, zb, hw)
            visbg, rowb, colb = project(ptsb, K_b, hw, depth_b)
            b_vis = visbg & valid_full
            b_on = np.zeros(hw, bool)
            b_on[b_vis] = L_b[rowb[b_vis], colb[b_vis]] == L_x[b_vis]
            del ptsb, depth_b, visbg, rowb, colb
            # ---- every other rig camera (work resolution, union nearest-upsampled)
            lp = local_points(zx["depth"], zx["K"])
            vu = (zx["conf"] > CONF_MIN) & (zx["depth"] > 0)
            hw_w = zx["depth"].shape
            any_w = np.zeros(hw_w, bool)
            nonann_w = np.zeros(hw_w, bool)
            for n in cams:
                if n in (c, x):
                    continue
                zn = zc if n == c else npz(scene, t, n)
                pn = to_reference(lp, zx["c2w"], zn["c2w"]).astype(np.float32)
                vn, _, _ = project(pn, zn["K"].astype(np.float32), zn["depth"].shape, zn["depth"])
                vn &= vu
                any_w |= vn
                if n in nonann:
                    nonann_w |= vn
            any_vis = cv2.resize(any_w.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            nonann_vis = cv2.resize(nonann_w.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            # ---- c_ini at the other DEV frames (cross-time visibility of occluded static/env pixels)
            idx = cv2.resize(np.arange(zc["depth"].size, dtype=np.float32).reshape(zc["depth"].shape), (hw[1], hw[0]),
                             interpolation=cv2.INTER_NEAREST).astype(np.int64)      # full-res pixel -> work-res pixel of c
            stat = occ & (typ[L_x] >= 2)
            occ_tvis = np.zeros(hw, bool)
            if stat.any():
                rr, cc = row[stat], col[stat]
                zz = pts[..., 2][stat]
                wi = idx[rr, cc]
                vc_t = (zc["conf"] > CONF_MIN) & (zc["depth"] > 0)
                hit = np.zeros(zz.shape, bool)
                for t2 in dev:
                    if t2 == t:
                        continue
                    z2 = dc[t2]
                    v2 = (z2["conf"] > CONF_MIN) & (z2["depth"] > 0)
                    both = vc_t & v2
                    if both.sum() < 100:
                        continue
                    s = float(np.median(zc["depth"][both] / z2["depth"][both]))
                    d2 = z2["depth"].ravel()[wi] * np.float32(s)
                    hit |= (d2 > 0) & (zz <= d2 * np.float32(1.0 + OCC_TOL))
                occ_tvis[stat] = hit
            # ---- nodepth at t but valid depth of the same X pixel at another DEV frame
            nod = cls == 3
            vx_other = np.zeros(hw_w, bool)
            for t2 in dev:
                if t2 == t:
                    continue
                z2 = npz(scene, t2, x)
                vx_other |= (z2["conf"] > CONF_MIN) & (z2["depth"] > 0)
            nod_tval = nod & cv2.resize(vx_other.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            # ---- forward splat of c_ini's own labelled Pi3 points into X (work res, z-buffer, nearest-upsampled):
            #      reaches target pixels WITHOUT target depth (nodepth) from the protocol input alone
            hc_w, wc_w = zc["depth"].shape
            Lc_w = cv2.resize(L_c, (wc_w, hc_w), interpolation=cv2.INTER_NEAREST)
            vcw = (zc["conf"] > CONF_MIN) & (zc["depth"] > 0) & (Lc_w > 0)
            pcx = to_reference(local_points(zc["depth"], zc["K"]), zc["c2w"], zx["c2w"]).astype(np.float32)
            okf, rrf, ccf = project(pcx, zx["K"].astype(np.float32), hw_w, None)
            okf &= vcw
            zf = pcx[..., 2][okf]
            order = np.argsort(-zf, kind="stable")                 # far first, nearest written last
            splat = np.zeros(hw_w, np.int32)
            splat[rrf[okf][order], ccf[okf][order]] = Lc_w[okf][order]
            splat_full = cv2.resize(splat.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            fwd_hit = splat_full == L_x
            fwd_any = splat_full > 0
            del pcx, okf, rrf, ccf
            # ---- per-object counts
            present_x = set(int(o) for o in np.unique(L_x) if o)
            present_c = set(int(o) for o in np.unique(L_c) if o)
            key = L_x.astype(np.int64) * 4 + cls
            def cnt(mask):
                return np.bincount(key[mask], minlength=256 * 4).reshape(256, 4)
            n_all = np.bincount(key.ravel(), minlength=256 * 4).reshape(256, 4)
            c_bvis, c_bon, c_any, c_non = cnt(b_vis), cnt(b_on), cnt(any_vis), cnt(nonann_vis)
            c_dyn, c_self, c_unl, c_tvis, c_ntv = cnt(occ_dyn), cnt(occ_self), cnt(occ_unl), cnt(occ_tvis), cnt(nod_tval)
            occ_stat_occluder = occ & ~occ_dyn & ~occ_self & ~occ_unl          # occluded by another static/env object
            stat_obj = typ[L_x] >= 2
            c_sd, c_ss, c_so = cnt(occ_dyn & stat_obj), cnt(occ_self & stat_obj), cnt(occ_stat_occluder & stat_obj)
            c_td, c_ts, c_to = cnt(occ_tvis & occ_dyn), cnt(occ_tvis & occ_self), cnt(occ_tvis & occ_stat_occluder)
            c_fh = cnt(fwd_hit)
            # splat precision: pixels of the target where the splat says o, by the target pixel's own class
            key_s = splat_full.astype(np.int64) * 4 + cls
            c_fs = np.bincount(key_s[fwd_any], minlength=256 * 4).reshape(256, 4)
            for o in objs:
                lab = labels[o]["label"]
                rows.append({
                    "scene": scene, "t": t, "cam": x, "B": b, "obj": int(o), "label": lab, "type": labels[o]["type"],
                    "in_cini_t": o in present_c,
                    "same_label_in_x": any(labels.get(p, {}).get("label") == lab for p in present_x if p != o),
                    "same_label_in_c": any(labels.get(p, {}).get("label") == lab for p in present_c if p != o),
                    "n": n_all[o].tolist(), "B_vis": c_bvis[o].tolist(), "B_ongt": c_bon[o].tolist(),
                    "any_vis": c_any[o].tolist(), "nonann_vis": c_non[o].tolist(),
                    "occ_dyn": int(c_dyn[o, 2]), "occ_self": int(c_self[o, 2]), "occ_unl": int(c_unl[o, 2]),
                    "occ_tvis": int(c_tvis[o, 2]), "nod_tval": int(c_ntv[o, 3]),
                    "so_dyn": int(c_sd[o, 2]), "so_self": int(c_ss[o, 2]), "so_stat": int(c_so[o, 2]),
                    "fwd_hit": c_fh[o].tolist(), "fwd_said": c_fs[o].tolist(),
                    "tvis_dyn": int(c_td[o, 2]), "tvis_self": int(c_ts[o, 2]), "tvis_stat": int(c_to[o, 2]),
                })
            del pts, valid_full, depth_ref, visg, row, col, vis, ins, cls, key
        # rig rigidity: relative pose c_ini -> X across the DEV frames' Pi3 runs
        ts = sorted(rel)
        R0, t0v = rel[ts[0]][:3, :3], rel[ts[0]][:3, 3]
        rot = [rot_angle(rel[t][:3, :3] @ R0.T) for t in ts]
        tdir = [float(np.degrees(np.arccos(np.clip(np.dot(rel[t][:3, 3], t0v) /
                                                    (np.linalg.norm(rel[t][:3, 3]) * np.linalg.norm(t0v) + 1e-12), -1, 1))))
                for t in ts]
        pose[x] = {"max_rot_deg": max(rot), "max_tdir_deg": max(tdir)}
    print(f"{scene}: {len(rows)} rows, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    return scene, rows, pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--scenes", default="")
    a = ap.parse_args()
    scenes = a.scenes.split(",") if a.scenes else sorted(d for d in os.listdir(DATA) if os.path.isfile(os.path.join(DATA, d, "meta.json")))
    t0 = time.time()
    with Pool(min(a.procs, len(scenes))) as pool:
        res = pool.map(run_scene, scenes, chunksize=1)
    out = {"meta": {"seconds": time.time() - t0, "occ_tol": OCC_TOL, "conf_min": CONF_MIN, "scenes": scenes},
           "rows": [r for _, rows, _ in res for r in rows], "pose": {s: p for s, _, p in res}}
    json.dump(out, open(a.out, "w"))
    print(f"done {len(out['rows'])} rows in {out['meta']['seconds']:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()

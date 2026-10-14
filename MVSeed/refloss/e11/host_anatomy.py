#!/usr/bin/env python
"""S1-E11 code-map measurements on the adopted stage-1 result (geoset_pred_ft_rep_e8 / _r1_e8), host-side (numpy, cv2).

Per scene: reference-visibility of every non-reference camera's pixels (numpy replica of geomem.warp_torch.reproject_coords
on the Pi3 work grid), the reproj gate decision of every (cam, obj) from round1_conf.json, per-view object overlaps of the
round-2 masks and four exclusivity variants scored on the 664 keyframe pairs, and the nearest-3 neighbour choice against
"sees more of the object" alternatives.  Output: JSON with per-pair records and aggregates.

    stage1/.venv/bin/python host_anatomy.py <out.json>
"""
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

REPO = "/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1")
DATA = os.path.join(S1, "Data")
sys.path.insert(0, S1)
from mvseg2d.io import labelmap_objects, load_labelmap, load_meta, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402
from geomem.pointmap import CONF_MIN, local_points, to_reference  # noqa: E402

REP, R1 = "geoset_pred_ft_rep_e8", "geoset_pred_ft_r1_e8"
THETA, TAU, K, MIN_VALID, OCC_TOL = 0.5, 0.7, 3, 64, 0.01
SUPPRESS_TAUS = (0.5, 0.7, 0.9)


def load_mask(sc, folder, cam, o, hw):
    p = os.path.join(DATA, sc, "Result", folder, cam, f"{o}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    return read_pred_mask(p)


def ref_projection(cache, cam, ref):
    """Work-resolution reprojection of `cam` into the reference: (u, v, valid) as geomem.warp_torch.reproject_coords
    (in front, inside the image, not occluded by more than OCC_TOL) & the Pi3 confidence/depth validity of `cam`."""
    zu, zr = np.load(os.path.join(cache, f"{cam}.npz")), np.load(os.path.join(cache, f"{ref}.npz"))
    pts = to_reference(local_points(zu["depth"], zu["K"]), zu["c2w"], zr["c2w"])
    valid_u = (zu["conf"] > CONF_MIN) & (zu["depth"] > 0)
    h, w = zr["depth"].shape
    z = pts[..., 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    Kr = zr["K"]
    u = Kr[0, 0] * pts[..., 0] / zs + Kr[0, 2]
    v = Kr[1, 1] * pts[..., 1] / zs + Kr[1, 2]
    valid = (z > 0) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5) & np.isfinite(u) & np.isfinite(v)
    col = np.clip(np.round(np.nan_to_num(u)), 0, w - 1).astype(int)
    row = np.clip(np.round(np.nan_to_num(v)), 0, h - 1).astype(int)
    d = zr["depth"][row, col]
    valid &= (d <= 0) | (z <= d * (1.0 + OCC_TOL))
    valid &= valid_u
    return row, col, valid


def up(a, hw):
    return cv2.resize(a.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def conf_of(rec):
    """object_conf_mode('reproj'): present & (overlap >= theta, IoU rule where overlap is None)."""
    present = rec["presence"] > 0
    if rec["overlap"] is None:
        return bool(present and rec["iou"] >= TAU)
    return bool(present and rec["overlap"] >= THETA)


def job(sc):
    meta = load_meta(DATA, sc)
    ref, cams = meta["ref_cam"], list(meta["cams"])
    cache = os.path.join(DATA, sc, "Cache", "pi3")
    ref_lab = load_labelmap(os.path.join(DATA, sc, "masks", f"{ref}.png"))
    objects = labelmap_objects(ref_lab)
    hw = ref_lab.shape
    bp = bound_pix(*hw)
    se = disk(bp)
    scored_cams = [c for c in meta["mask_cams"] if c != ref]
    gts = {c: load_labelmap(os.path.join(DATA, sc, "masks", f"{c}.png")) for c in scored_cams}
    centres = {c: np.load(os.path.join(cache, f"{c}.npz"))["c2w"][:3, 3].astype(np.float64) for c in cams}
    with open(os.path.join(DATA, sc, "Result", REP, "round1_conf.json")) as f:
        r1conf = json.load(f)
    targets = [c for c in cams if c != ref]
    # work-res reference label map for the overlap replica
    zr = np.load(os.path.join(cache, f"{ref}.npz"))
    hw_work = zr["depth"].shape
    ref_lab_work = cv2.resize(ref_lab, (hw_work[1], hw_work[0]), interpolation=cv2.INTER_NEAREST)

    per_co = {}          # (cam, obj) -> scalar features for every non-reference camera
    pair_rows = []       # scored pairs
    overlap_check = []   # (json overlap, replica overlap)
    cam_overlap = []     # per non-ref cam: contested pixel counts of the round-2 masks
    for cam in targets:
        row, col, valid_w = ref_projection(cache, cam, ref)
        vis = up(valid_w, hw)                                   # pixels of cam visible from the reference (full res)
        on_gt_w = {o: valid_w & (ref_lab_work[row, col] == o) for o in objects}
        r1m = {o: load_mask(sc, R1, cam, o, hw) for o in objects}
        repm = {o: load_mask(sc, REP, cam, o, hw) for o in objects}
        count = np.zeros(hw, np.int32)
        for o in objects:
            count += repm[o]
        contested = count >= 2
        areas = {o: int(repm[o].sum()) for o in objects}
        minarea = np.full(hw, np.iinfo(np.int64).max, np.int64)
        maxarea = np.zeros(hw, np.int64)
        for o in objects:
            if areas[o]:
                minarea = np.where(repm[o], np.minimum(minarea, areas[o]), minarea)
                maxarea = np.where(repm[o], np.maximum(maxarea, areas[o]), maxarea)
        G = gts.get(cam)
        gt_claim = np.zeros(hw, bool)
        if G is not None:
            for o in objects:
                gt_claim |= repm[o] & (G == o)
        cam_overlap.append(dict(scene=sc, cam=cam, scored=G is not None, contested_px=int(contested.sum()),
                                union_px=int((count > 0).sum()),
                                n_obj_with_overlap=int(sum((repm[o] & contested).any() for o in objects)),
                                n_obj_nonempty=int(sum(areas[o] > 0 for o in objects))))
        for o in objects:
            rec = r1conf[cam][str(o)]
            m1 = r1m[o]
            a1 = int(m1.sum())
            # replica of reproj_overlap on the work grid
            m1w = cv2.resize(m1.astype(np.uint8), (hw_work[1], hw_work[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            sel = m1w & valid_w
            n = int(sel.sum())
            ov = float((sel & on_gt_w[o]).sum() / n) if n else float("nan")
            if rec["overlap"] is not None and n:
                overlap_check.append((rec["overlap"], ov))
            inside_other = float((repm[o] & contested).sum() / areas[o]) if areas[o] else 0.0
            per_co[(cam, o)] = dict(conf=conf_of(rec), presence=rec["presence"], iou=rec["iou"], overlap=rec["overlap"],
                                    n_valid=rec["n_valid"], area_r1=a1, inv_r1=int((m1 & ~vis).sum()),
                                    area_rep=areas[o], inv_rep=int((repm[o] & ~vis).sum()), inside_other=inside_other,
                                    valid_frac_r1=float((m1 & vis).sum() / a1) if a1 else None)
            if G is None:
                continue
            g = G == o
            gp = int(g.sum())
            if gp == 0:
                continue
            gv, gi = g & vis, g & ~vis
            m2 = repm[o]
            J, F = jf_pair(g, m2, bp, se)
            variants = {}
            cands = {
                "oracle_assign": m2 & ~(contested & gt_claim & (G != o)),
                "smaller_wins": m2 & ~(contested & (minarea < areas[o])) if areas[o] else m2,
                "larger_wins": m2 & ~(contested & (maxarea > areas[o])) if areas[o] else m2,
            }
            for t in SUPPRESS_TAUS:
                cands[f"suppress_inside_{t}"] = np.zeros(hw, bool) if inside_other >= t else m2
            for name, mm in cands.items():
                if mm is m2 or bool((mm == m2).all()):
                    variants[name] = None
                else:
                    j, f = jf_pair(g, mm, bp, se)
                    variants[name] = dict(J=j, F=f, JF=0.5 * (j + f), px=int(mm.sum()))
            pair_rows.append(dict(
                scene=sc, cam=cam, obj=o, J=J, F=F, JF=0.5 * (J + F), gt_px=gp, gt_inv_px=int(gi.sum()),
                gt_inv_frac=float(gi.sum() / gp), rep_px=areas[o], r1_px=a1,
                recall_r1=float((m1 & g).sum() / gp), recall_rep=float((m2 & g).sum() / gp),
                recall_vis_r1=float((m1 & gv).sum() / gv.sum()) if gv.any() else None,
                recall_vis_rep=float((m2 & gv).sum() / gv.sum()) if gv.any() else None,
                recall_inv_r1=float((m1 & gi).sum() / gi.sum()) if gi.any() else None,
                recall_inv_rep=float((m2 & gi).sum() / gi.sum()) if gi.any() else None,
                precision_rep=float((m2 & g).sum() / areas[o]) if areas[o] else None,
                inside_other=inside_other, contested_px_in_mask=int((m2 & contested).sum()),
                own_conf=conf_of(rec), own_overlap=rec["overlap"], own_n_valid=rec["n_valid"], own_iou=rec["iou"],
                own_presence=rec["presence"], variants=variants))
        del r1m, repm, count, contested, minarea, maxarea
    # neighbour choice for every target x object (select_candidates: 3 nearest by centre, the reference dropped)
    neigh_rows = []
    for t in targets:
        others = sorted((np.linalg.norm(centres[c] - centres[t]), c) for c in cams if c != t)
        cand = [c for _, c in others[:K] if c != ref]
        dist = {c: d for d, c in others}
        for o in objects:
            used = [c for c in cand if per_co[(c, o)]["conf"]]
            pool = [c for c in targets if c != t and per_co[(c, o)]["conf"]]
            if not pool:
                neigh_rows.append(dict(scene=sc, cam=t, obj=o, n_cand=len(cand), n_used=len(used), pool=0))
                continue
            by_area = max(pool, key=lambda c: per_co[(c, o)]["area_r1"])
            by_inv = max(pool, key=lambda c: per_co[(c, o)]["inv_r1"])
            neigh_rows.append(dict(
                scene=sc, cam=t, obj=o, n_cand=len(cand), n_used=len(used), pool=len(pool),
                used_max_area=max([per_co[(c, o)]["area_r1"] for c in used], default=0),
                best_area=per_co[(by_area, o)]["area_r1"], best_area_in_used=by_area in used,
                best_area_rank=[c for _, c in others].index(by_area) + 1, best_area_dist=dist[by_area],
                used_dists=[dist[c] for c in used], cand_dists=[dist[c] for c in cand],
                used_max_inv=max([per_co[(c, o)]["inv_r1"] for c in used], default=0),
                best_inv=per_co[(by_inv, o)]["inv_r1"], best_inv_in_used=by_inv in used,
                best_inv_rank=[c for _, c in others].index(by_inv) + 1,
                own_area_r1=per_co[(t, o)]["area_r1"], own_inv_r1=per_co[(t, o)]["inv_r1"]))
    gate_rows = [dict(scene=sc, cam=c, obj=o, **{k: v for k, v in d.items()}) for (c, o), d in per_co.items()]
    return dict(scene=sc, n_objects=len(objects), n_targets=len(targets), hw=list(hw), pairs=pair_rows,
                overlap_check=overlap_check, cam_overlap=cam_overlap, neigh=neigh_rows, gate=gate_rows)


def main():
    out = sys.argv[1]
    scenes = sorted(d for d in os.listdir(DATA) if os.path.isfile(os.path.join(DATA, d, "meta.json")))
    with Pool(6) as pool:
        res = pool.map(job, scenes)
    json.dump(res, open(out, "w"))
    print("scenes", len(res), "pairs", sum(len(r["pairs"]) for r in res))


if __name__ == "__main__":
    main()

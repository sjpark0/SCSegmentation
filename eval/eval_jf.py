#!/usr/bin/env python3
"""J&F evaluation for the MVSeg results, following the official DAVIS protocol.

Layout it expects, per dataset directory:

    Mask/<cam>/<frame>.png          ground truth, greyscale, pixel value = object id
    Mask/objects_labels.json        object id -> {label, type}
    <method>/<cam>/<frame>/<id>.png predicted binary mask for one object

Each (dataset, camera) pair is treated as one DAVIS "sequence": the object set is
the set of ids that appear in that camera's ground truth, J is the mask IoU and F
the boundary F-measure (bound_th=0.008), an empty prediction on an empty ground
truth scores 1, and the sequence score is the mean over objects of the mean over
frames.  Both aggregations are reported: over every frame, and over frames
1..n-2 the way the DAVIS evaluation code drops the first and last frame.

Run it inside the scsam3 container (it needs numpy and cv2):

    docker run --rm -v /:/host -w /host/$PWD scsam3 python eval_jf.py
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METHODS = ["SegMask", "SegMask1", "SegMaskNew", "SegMaskNew1",
                   "SegMaskNew2", "SegMaskNew3"]
METHODS = list(DEFAULT_METHODS)     # replaced by --methods before the pool forks
BOUND_TH = 0.008


# ----------------------------------------------------------------- DAVIS core
def db_eval_iou(annotation, segmentation):
    inters = np.count_nonzero(segmentation & annotation)
    union = np.count_nonzero(segmentation | annotation)
    return 1.0 if union == 0 else inters / union


def _seg2bmap(seg):
    """Boundary map of a binary mask - the DAVIS toolkit's formulation."""
    seg = seg.astype(bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = (seg ^ e) | (seg ^ s) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def _disk(radius):
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= r * r).astype(np.uint8)


def db_eval_boundary(foreground_mask, gt_mask, bound_pix, se):
    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)

    n_fg = np.count_nonzero(fg_boundary)
    n_gt = np.count_nonzero(gt_boundary)
    if n_fg == 0 and n_gt == 0:
        return 1.0
    if n_fg == 0 or n_gt == 0:
        return 0.0

    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), se)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), se)

    gt_match = np.count_nonzero(gt_boundary & fg_dil.astype(bool))
    fg_match = np.count_nonzero(fg_boundary & gt_dil.astype(bool))

    precision = fg_match / n_fg
    recall = gt_match / n_gt
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------- helper
def crop_box(a, b, margin, shape):
    """Union bounding box of two masks, grown by `margin` and clipped.

    Everything outside it is background in both masks, so J and F computed on
    the crop match the full-frame values - as long as the margin is at least
    the dilation radius, which is what the caller passes in.
    """
    rows = np.flatnonzero(a.any(1) | b.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(a.any(0) | b.any(0))
    y0 = max(int(rows[0]) - margin, 0)
    y1 = min(int(rows[-1]) + margin + 1, shape[0])
    x0 = max(int(cols[0]) - margin, 0)
    x1 = min(int(cols[-1]) + margin + 1, shape[1])
    return y0, y1, x0, x1


def frame_key(name):
    stem = os.path.splitext(name)[0]
    return int(stem) if stem.lstrip("-").isdigit() else stem


def eval_camera(job):
    """One (dataset, camera, method).  Returns the per-object J and F means."""
    dataset, cam, method = job
    ds_dir = os.path.join(ROOT, dataset)
    gt_dir = os.path.join(ds_dir, "Mask", cam)

    gt_files = sorted((f for f in os.listdir(gt_dir) if f.endswith(".png")),
                      key=frame_key)
    gts, obj_ids = [], set()
    for f in gt_files:
        g = cv2.imread(os.path.join(gt_dir, f), cv2.IMREAD_UNCHANGED)
        if g.ndim == 3:
            g = g[..., 0]
        gts.append(g)
        obj_ids.update(np.unique(g).tolist())
    obj_ids.discard(0)
    obj_ids = sorted(int(o) for o in obj_ids)

    h, w = gts[0].shape[:2]
    bound_pix = int(math.ceil(BOUND_TH * np.linalg.norm((h, w))))
    se = _disk(bound_pix)

    out = {"dataset": dataset, "camera": cam, "method": method,
           "n_frames": len(gts), "objects": obj_ids}

    if True:
        m_cam = os.path.join(ds_dir, method, cam)
        if not os.path.isdir(m_cam):
            out["result"] = None
            return out
        pred_frames = sorted(os.listdir(m_cam), key=frame_key)
        if len(pred_frames) != len(gt_files):
            # align by numeric frame id when the counts disagree
            by_id = {frame_key(p): p for p in pred_frames}
            pred_frames = [by_id.get(frame_key(f)) for f in gt_files]

        # per object, per frame
        J = np.zeros((len(obj_ids), len(gts)))
        F = np.zeros((len(obj_ids), len(gts)))
        missing = 0
        for fi, gt in enumerate(gts):
            pf = pred_frames[fi]
            for oi, obj in enumerate(obj_ids):
                gm = gt == obj
                pm = None
                if pf is not None:
                    p = os.path.join(m_cam, pf, f"{obj}.png")
                    if os.path.exists(p):
                        pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if pm is None:
                    missing += 1
                    pm = np.zeros((h, w), dtype=bool)
                else:
                    if pm.shape[:2] != (h, w):
                        pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                    pm = pm > 127

                box = crop_box(gm, pm, bound_pix + 2, (h, w))
                if box is None:
                    J[oi, fi] = 1.0
                    F[oi, fi] = 1.0
                    continue
                y0, y1, x0, x1 = box
                g_c, p_c = gm[y0:y1, x0:x1], pm[y0:y1, x0:x1]
                J[oi, fi] = db_eval_iou(g_c, p_c)
                F[oi, fi] = db_eval_boundary(p_c, g_c, bound_pix, se)

        inner = slice(1, -1) if len(gts) > 2 else slice(None)
        out["result"] = {
            "J_all": J.mean(axis=1).tolist(),
            "F_all": F.mean(axis=1).tolist(),
            "J_inner": J[:, inner].mean(axis=1).tolist(),
            "F_inner": F[:, inner].mean(axis=1).tolist(),
            "missing_files": missing,
        }
    return out


def main():
    global METHODS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*",
                    help="dataset folders to score (default: all that have every method)")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help="result folder names to score")
    ap.add_argument("--out", default="jf_raw.json", help="where to write the raw scores")
    args = ap.parse_args()
    METHODS = list(args.methods)

    datasets = sorted(
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d, "Mask"))
        and all(os.path.isdir(os.path.join(ROOT, d, m)) for m in METHODS))
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]
    if not datasets:
        sys.exit(f"no dataset under {ROOT} has all of: {', '.join(METHODS)}")

    jobs, cost = [], {}
    for d in datasets:
        mask = os.path.join(ROOT, d, "Mask")
        cams = sorted(x for x in os.listdir(mask)
                      if os.path.isdir(os.path.join(mask, x)))
        lab = os.path.join(mask, "objects_labels.json")
        n = len(json.load(open(lab))) if os.path.exists(lab) else 20
        for cam in cams:
            for m in METHODS:
                jobs.append((d, cam, m))
                cost[(d, cam, m)] = n
    # heaviest first, so the long tail starts before the short jobs fill the pool
    jobs.sort(key=lambda j: -cost[j])
    print(f"{len(datasets)} datasets, {len(jobs)} jobs", flush=True)

    with Pool(min(os.cpu_count(), len(jobs))) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(eval_camera, jobs), 1):
            results.append(r)
            print(f"[{i}/{len(jobs)}] {r['dataset']}/{r['camera']}/{r['method']} "
                  f"({len(r['objects'])} objects)", flush=True)

    out = os.path.join(ROOT, args.out)
    with open(out, "w") as fh:
        json.dump(results, fh)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

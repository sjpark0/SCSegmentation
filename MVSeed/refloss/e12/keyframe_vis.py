#!/usr/bin/env python3
"""S1-E12 visibility block (docs/stage1-E12-prereg.md section 3.4): for the 664 scored keyframe pairs
(MVSeed/runs/seedfail.json rows), split every pair's GT, hits and false positives of each given stage-1 result folder
by reference visibility, and the pixels one folder ADDS / REMOVES relative to another, with their GT overlap.

Classes (as MVSeed/refloss/e11/extent_vis.py and train_vis.py, the stage-1 reprojection machinery itself:
methods.geomem.run.reproj_inputs + project_reference_masks = geomem.warp_torch.reproject_coords, OCC_TOL 0.01,
Pi3 validity conf > 0.1 & depth > 0):
    vis       the pixel's Pi3 point projects inside the reference image, in front, not behind the reference surface
    outside   valid depth, projects outside the reference image / behind the camera     -+
    occluded  inside the reference image, fails the depth test                         +-> invisible
    nodepth   no valid Pi3 depth                                                       -+

Runs inside scsam3 (torch, GPU if available):
    stage1/scripts/docker_run.sh scsam3 python /host<repo>/MVSeed/refloss/e12/keyframe_vis.py \\
        --out /host<repo>/MVSeed/runs/e12/vis.json  FOLDER [FOLDER ...]  [--pair NEW:OLD ...]
Output: {"meta", "rows": [{scene, cam, obj, gt_<cls>, "folders": {F: {pred_px, hit_<cls>, fp_<cls>}},
"pairs": {"NEW:OLD": {add_<cls>, add_tp_<cls>, rem_<cls>, rem_tp_<cls>}}}]} with <cls> in px (all) / vis / outside /
occluded / nodepth / on_refgt (visible and on the same object's projected reference GT).
"""
import argparse
import json
import os
import sys
import time

import cv2  # noqa: F401  (reproj_inputs needs it)
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from geomem.warp_torch import OCC_TOL, reproject_coords  # noqa: E402
from methods.geomem.run import project_reference_masks, reproj_inputs  # noqa: E402
from mvseg2d.io import load_labelmap, load_meta, read_pred_mask  # noqa: E402

DATA = os.path.join(S1, "Data")
ROWS = os.path.join(REPO, "MVSeed", "runs", "seedfail.json")
CLASSES = ("vis", "outside", "occluded", "nodepth")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def classes(cache, cam, ref, hw, ref_lab_np, objs, chunk=16):
    """Visibility classes [h,w] of view `cam` w.r.t. the reference, plus pm [B,h,w]: the same object's reference GT
    projected into `cam` (the 'on reference GT' class of train_vis.py; review 2026-10-15)."""
    pts, valid_full, depth_ref, K_ref = reproj_inputs(cache, cam, ref, hw, DEV)
    _, inside = reproject_coords(pts, K_ref, hw, depth_ref=None)
    inside = inside & valid_full
    ref_lab = torch.from_numpy(ref_lab_np).to(DEV)
    _, vis = project_reference_masks((ref_lab != 0)[None], pts, valid_full, depth_ref, K_ref)
    pm = torch.zeros((len(objs),) + tuple(hw), dtype=torch.bool, device=DEV)
    for i in range(0, len(objs), chunk):
        sub = objs[i:i + chunk]
        pm[i:i + len(sub)], _ = project_reference_masks(ref_lab[None] == torch.tensor(sub, device=DEV)[:, None, None],
                                                        pts, valid_full, depth_ref, K_ref)
    out = {"vis": vis, "outside": valid_full & ~inside, "occluded": inside & ~vis, "nodepth": ~valid_full, "pm": pm}
    del pts, valid_full, depth_ref, K_ref, inside
    return out


def counts(m, C):
    """m [B,h,w] bool -> {px, vis, outside, occluded, nodepth} int arrays [B]."""
    f = lambda x: x.flatten(1).sum(1).cpu().numpy().astype(np.int64)
    return {"px": f(m), **{c: f(m & C[c][None]) for c in CLASSES}, "on_refgt": f(m & C["pm"] & C["vis"][None])}


def read_mask(scene, folder, cam, obj, hw):
    p = os.path.join(DATA, scene, "Result", folder, cam, f"{obj}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    m = read_pred_mask(p)
    assert m.shape == tuple(hw), (p, m.shape, hw)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--pair", action="append", default=[], help="NEW:OLD, both among the folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default=None, help="comma-separated subset (default: all 17)")
    a = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in a.pair]
    assert all(n in a.folders and o in a.folders for n, o in pairs), pairs
    t0 = time.time()
    rows_in = json.load(open(ROWS))["rows"]
    by_scene = {}
    for r in rows_in:
        by_scene.setdefault(r["scene"], []).append(r)
    out_rows = []
    for scene in sorted(by_scene):
        if a.scenes and scene not in a.scenes.split(","):
            continue
        ts = time.time()
        srows = by_scene[scene]
        meta = load_meta(DATA, scene)
        ref = meta["ref_cam"]
        cache = os.path.join(DATA, scene, "Cache", "pi3")
        ref_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{ref}.png"))
        hw = tuple(ref_lab.shape)
        objs = sorted({r["obj"] for r in srows})
        oi = {o: i for i, o in enumerate(objs)}
        for cam in sorted({r["cam"] for r in srows}):
            gt_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{cam}.png"))
            C = classes(cache, cam, ref, hw, ref_lab, objs)
            gt = torch.from_numpy(gt_lab).to(DEV)[None] == torch.tensor(objs, device=DEV)[:, None, None]
            g = counts(gt, C)
            preds = {f: torch.from_numpy(np.stack([read_mask(scene, f, cam, o, hw) for o in objs])).to(DEV) for f in a.folders}
            per_f = {f: {"pred_px": counts(p, C)["px"], "hit": counts(p & gt, C), "fp": counts(p & ~gt, C)} for f, p in preds.items()}
            per_p = {}
            for n, o in pairs:
                add, rem = preds[n] & ~preds[o], preds[o] & ~preds[n]
                per_p[f"{n}:{o}"] = {"add": counts(add, C), "add_tp": counts(add & gt, C),
                                     "rem": counts(rem, C), "rem_tp": counts(rem & gt, C)}
            for r in srows:
                if r["cam"] != cam:
                    continue
                b = oi[r["obj"]]
                rec = {"scene": scene, "cam": cam, "obj": r["obj"], **{f"gt_{k}": int(v[b]) for k, v in g.items()}, "folders": {}, "pairs": {}}
                assert rec["gt_px"] == r["gt_px"], (scene, cam, r["obj"], rec["gt_px"], r["gt_px"])
                for f, d in per_f.items():
                    rec["folders"][f] = {"pred_px": int(d["pred_px"][b]), **{f"hit_{k}": int(v[b]) for k, v in d["hit"].items()},
                                         **{f"fp_{k}": int(v[b]) for k, v in d["fp"].items()}}
                for k2, d in per_p.items():
                    rec["pairs"][k2] = {f"{part}_{k}": int(v[b]) for part, dd in d.items() for k, v in dd.items()}
                out_rows.append(rec)
            del C, gt, preds
            if DEV == "cuda":
                torch.cuda.empty_cache()
        log(f"{scene}: {len(srows)} pairs, {time.time() - ts:.1f} s")
    meta = {"folders": a.folders, "pairs": [f"{n}:{o}" for n, o in pairs], "rows_src": ROWS, "device": DEV, "occ_tol": OCC_TOL,
            "n_rows": len(out_rows), "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"meta": meta, "rows": out_rows}, open(a.out, "w"))
    log(f"done: {len(out_rows)} rows in {time.time() - t0:.1f} s -> {a.out}")


if __name__ == "__main__":
    main()

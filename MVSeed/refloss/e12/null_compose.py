#!/usr/bin/env python3
"""S1-E12 design measurement: the ADD-ONLY composition rule fed with EXISTING, mechanism-free models (nulls).

Composition (the E12 candidate rule), per scored view and object b, with C0 = adopted seeds geoset_pred_ft_rep_x_e11
and E = another model's masks of the same view:
    INV   = pixels of the view NOT visible from the reference camera (outside | occluded | nodepth; E11 definition,
            project_reference_masks' ~vis with OCC_TOL 0.01, Pi3 conf > 0.1 & depth > 0)
    U     = pixels claimed by NO C0 mask of the view
    add_b = E_b & INV & U, kept only in 8-connected components of (C0_b | add_b) that contain a C0_b pixel
    final_b = C0_b | add_b          (C0_b empty -> final_b empty: no new masks)
Nulls: E = geoset_dec_ft_rep_e9 (E9 D1, decoder trained, same data), geoset_decb_ft_rep_e9 (E9 D2, masks expanded
outward), geoset_pred_ft_r1_x_e11 (the adopted model's own round 1).  Scores J/F with mvseg2d.metrics.jf_pair exactly
as the keyframe evaluator; C0 itself is re-scored as a check.  Runs inside scsam3:
    stage1/scripts/docker_run.sh scsam3 python -c "$(cat null_compose.py)" > null_compose.json  (log on stderr)
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

REPO = "/host/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from geomem.warp_torch import OCC_TOL, reproject_coords  # noqa: E402
from methods.geomem.run import project_reference_masks, reproj_inputs  # noqa: E402
from mvseg2d.io import load_labelmap, load_meta, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
ROWS = os.path.join(REPO, "MVSeed", "runs", "seedfail.json")
C0 = "geoset_pred_ft_rep_x_e11"
NULLS = ("geoset_dec_ft_rep_e9", "geoset_decb_ft_rep_e9", "geoset_pred_ft_r1_x_e11")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def read_seed(scene, seeds, cam, obj, hw):
    p = os.path.join(DATA, scene, "Result", seeds, cam, f"{obj}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    m = read_pred_mask(p)
    assert m.shape == tuple(hw), (p, m.shape, hw)
    return m


def inv_classes(cache, cam, ref, hw, ref_lab_np):
    pts, valid_full, depth_ref, K_ref = reproj_inputs(cache, cam, ref, hw, DEV)
    _, inside = reproject_coords(pts, K_ref, hw, depth_ref=None)
    inside = inside & valid_full
    ref_lab = torch.from_numpy(ref_lab_np).to(DEV)
    _, vis = project_reference_masks((ref_lab != 0)[None], pts, valid_full, depth_ref, K_ref)
    out = {"vis": vis, "outside": valid_full & ~inside, "occluded": inside & ~vis, "nodepth": ~valid_full}
    return {k: v.cpu().numpy() for k, v in out.items()}


def compose(c0_b, e_b, inv, unclaimed):
    """-> (final, add_kept, add_raw) bool arrays."""
    if not c0_b.any():
        return c0_b.copy(), np.zeros_like(c0_b), np.zeros_like(c0_b)
    add = e_b & inv & unclaimed
    if not add.any():
        return c0_b.copy(), add, add
    n, lab = cv2.connectedComponents((c0_b | add).astype(np.uint8), connectivity=8)
    keep_labels = np.unique(lab[c0_b])
    keep = np.isin(lab, keep_labels)
    add_kept = add & keep
    return c0_b | add_kept, add_kept, add


def main():
    t0 = time.time()
    rows_in = json.load(open(ROWS))["rows"]
    by_scene = {}
    for r in rows_in:
        by_scene.setdefault(r["scene"], []).append(r)
    out = []
    for scene in sorted(by_scene):
        ts = time.time()
        srows = by_scene[scene]
        meta = load_meta(DATA, scene)
        ref = meta["ref_cam"]
        cache = os.path.join(DATA, scene, "Cache", "pi3")
        ref_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{ref}.png"))
        hw = tuple(ref_lab.shape)
        bp = bound_pix(*hw)
        se = disk(bp)
        for cam in sorted({r["cam"] for r in srows}):
            gt_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{cam}.png"))
            V = inv_classes(cache, cam, ref, hw, ref_lab)
            inv = ~V["vis"]
            objs = sorted({r["obj"] for r in srows if r["cam"] == cam})
            c0 = {o: read_seed(scene, C0, cam, o, hw) for o in objs}
            unclaimed = ~np.logical_or.reduce([c0[o] for o in objs]) if objs else np.ones(hw, bool)
            ext = {nm: {o: read_seed(scene, nm, cam, o, hw) for o in objs} for nm in NULLS}
            for r in srows:
                if r["cam"] != cam:
                    continue
                o = r["obj"]
                gt = gt_lab == o
                rec = {"scene": scene, "cam": cam, "obj": o, "kind": r["kind"], "JF_e8": r["JF"], "gt_px": int(gt.sum()),
                       "c0_px": int(c0[o].sum())}
                J, F = jf_pair(gt, c0[o], bp, se)
                rec["C0"] = {"J": J, "F": F}
                for nm in NULLS:
                    final, add_kept, add_raw = compose(c0[o], ext[nm][o], inv, unclaimed)
                    J, F = jf_pair(gt, final, bp, se)
                    Je, Fe = jf_pair(gt, ext[nm][o], bp, se)
                    rec[nm] = {"J": J, "F": F, "raw_J": Je, "raw_F": Fe, "ext_px": int(ext[nm][o].sum()),
                               "add_raw": int(add_raw.sum()), "add": int(add_kept.sum()),
                               "add_tp": int((add_kept & gt).sum()), "add_fp": int((add_kept & ~gt).sum()),
                               "add_fp_outside": int((add_kept & ~gt & V["outside"]).sum()),
                               "add_fp_occluded": int((add_kept & ~gt & V["occluded"]).sum()),
                               "add_fp_nodepth": int((add_kept & ~gt & V["nodepth"]).sum()),
                               "add_tp_outside": int((add_kept & gt & V["outside"]).sum()),
                               "add_tp_occluded": int((add_kept & gt & V["occluded"]).sum()),
                               "add_tp_nodepth": int((add_kept & gt & V["nodepth"]).sum()),
                               "ext_extra_inv": int((ext[nm][o] & ~c0[o] & inv).sum()),      # what E has beyond C0 in INV
                               "ext_extra_vis": int((ext[nm][o] & ~c0[o] & ~inv).sum())}
                out.append(rec)
            del V
            if DEV == "cuda":
                torch.cuda.empty_cache()
        log(f"{scene}: {len(srows)} pairs, {time.time() - ts:.1f} s (elapsed {time.time() - t0:.0f} s)")
    meta = {"C0": C0, "nulls": list(NULLS), "occ_tol": OCC_TOL, "device": DEV, "seconds": time.time() - t0, "n_rows": len(out)}
    json.dump({"meta": meta, "rows": out}, sys.stdout)
    log(f"done: {len(out)} rows in {time.time() - t0:.1f} s")


main()

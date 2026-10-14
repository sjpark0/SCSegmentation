#!/usr/bin/env python3
"""S1-E11 pre-analysis (extent-visibility): where are the pixels the adopted stage-1 seeds MISS, and who can see them?

For every scored pair (scene, cam, obj) of MVSeed/runs/seedfail.json (664 pairs, seeds = Result/geoset_pred_ft_rep_e8,
GT = stage1/Data/<scene>/masks/<cam>.png) the missed set M = GT \\ seed is split with the stage-1 reprojection
machinery (methods.geomem.run.reproj_inputs + project_reference_masks, i.e. geomem.warp_torch.reproject_coords with
the occlusion test z <= d_ref * (1 + OCC_TOL), OCC_TOL = 0.01, and the Pi3 validity conf > 0.1 & depth > 0):

    vis       the target pixel's Pi3 point projects inside the reference image, in front of it, and is not hidden
              behind the reference surface (what project_reference_masks calls `vis`)          -> (a) visible
    outside   valid Pi3 depth but the point falls outside the reference image / behind the camera  -+
    occluded  inside the reference image but fails the depth test                                  +-> (b) not visible
    nodepth   the target pixel has no valid Pi3 depth (conf <= 0.1 or depth 0)                     -+
    on_refgt  vis AND the reprojected position lands on the reference GT of the SAME object (project_reference_masks)

J&F oracles: seed | (M & vis), seed | (M & ~vis), seed | M (= seed | GT), seed | (M & on_refgt).
Neighbour knowledge: every other non-reference camera's ADOPTED seed of the object is reprojected into the target
(same machinery, the neighbour playing the reference: reproj_inputs(cache, target, neighbour)) and we record which
fraction of M it covers, for each neighbour, for the union of all, and for the union of the k = 3 nearest cameras
round 2 would consider (select_candidates, reference excluded).  GT-free unions (seed | reprojected masks) are also
scored, as candidate post-processes.

Runs inside scsam3 (torch; GPU if available):
    stage1/scripts/docker_run.sh scsam3 python -c "$(cat extent_vis.py)" > extent_vis.json   (logs on stderr)
"""
import json
import math
import os
import sys
import time

import cv2  # noqa: F401  (reproj_inputs needs it)
import numpy as np
import torch

REPO = "/host/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from methods.geomem.run import camera_centres, project_reference_masks, reproj_inputs, select_candidates  # noqa: E402
from geomem.warp_torch import OCC_TOL, reproject_coords  # noqa: E402
from mvseg2d.io import load_labelmap, load_meta, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
SEEDS = "geoset_pred_ft_rep_e8"
ROWS = os.path.join(REPO, "MVSeed", "runs", "seedfail.json")
K_NEIGHBOURS = 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def read_seed(scene, cam, obj, hw):
    p = os.path.join(DATA, scene, "Result", SEEDS, cam, f"{obj}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    m = read_pred_mask(p)
    assert m.shape == tuple(hw), (p, m.shape, hw)
    return m


def jf(gt, pred, bp, se):
    J, F = jf_pair(gt, pred, bp, se)
    return {"J": float(J), "F": float(F), "JF": float((J + F) / 2)}


def main():
    t0 = time.time()
    rows = json.load(open(ROWS))["rows"]
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["scene"], []).append(r)
    out_rows = []
    only = [a for a in sys.argv[1:] if not a.startswith("-")]        # optional scene subset (python -c ... Scene1 Scene2)
    for scene in sorted(by_scene):
        if only and scene not in only:
            continue
        ts = time.time()
        srows = by_scene[scene]
        meta = load_meta(DATA, scene)
        ref = meta["ref_cam"]
        cams = list(meta["cams"])
        cache = os.path.join(DATA, scene, "Cache", "pi3")
        man = json.load(open(os.path.join(cache, "MANIFEST.json")))
        assert man["ref_cam"] == ref, (scene, man["ref_cam"], ref)
        ref_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{ref}.png"))
        hw = tuple(ref_lab.shape)
        objs = sorted({r["obj"] for r in srows})
        oi = {o: i for i, o in enumerate(objs)}
        B = len(objs)
        bp = bound_pix(*hw)
        se = disk(bp)
        ref_masks = torch.from_numpy(np.stack([ref_lab == o for o in objs])).to(DEV)      # [B,h,w]
        centres = camera_centres(cache, cams)
        targets = sorted({r["cam"] for r in srows})
        others = [c for c in cams if c != ref]
        # adopted seeds of every non-reference camera (missing PNG = empty), as [B,h,w] bool on the device
        seeds = {c: torch.from_numpy(np.stack([read_seed(scene, c, o, hw) for o in objs])).to(DEV) for c in others}
        for cam in targets:
            gt_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{cam}.png"))
            assert gt_lab.shape == hw
            gt = torch.from_numpy(np.stack([gt_lab == o for o in objs])).to(DEV)         # [B,h,w]
            seed = seeds[cam]
            missed = gt & ~seed                                                          # [B,h,w]
            n_missed = missed.flatten(1).sum(1)                                           # [B]
            # --- (a)/(b): reference visibility of the target's pixels (the stage-1 machinery)
            pts, valid_full, depth_ref, K_ref = reproj_inputs(cache, cam, ref, hw, DEV)
            pm, vis = project_reference_masks(ref_masks, pts, valid_full, depth_ref, K_ref)   # pm [B,h,w], vis [h,w]
            _, inside = reproject_coords(pts, K_ref, hw, depth_ref=None)                 # in front + inside the image
            inside = inside & valid_full
            outside = valid_full & ~inside
            occluded = inside & ~vis
            nodepth = ~valid_full
            cnt = lambda m: m.flatten(1).sum(1).cpu().numpy()                            # noqa: E731
            c_vis, c_out, c_occ, c_nod, c_ongt = (cnt(missed & vis[None]), cnt(missed & outside[None]),
                                                  cnt(missed & occluded[None]), cnt(missed & nodepth[None]), cnt(missed & pm))
            # sanity: the reference GT reprojected vs the target GT
            pm_rec = cnt(pm & gt) / np.maximum(cnt(gt), 1)
            pm_prec = cnt(pm & gt) / np.maximum(cnt(pm), 1)
            # --- neighbours: every other non-reference camera's adopted seed reprojected into this target
            cand = [c for c in select_candidates(cam, centres, K_NEIGHBOURS) if c != ref]
            union_all = torch.zeros_like(seed)
            union_k3 = torch.zeros_like(seed)
            anyvis = torch.zeros(hw, dtype=torch.bool, device=DEV)
            per_nb = {}
            for nb in others:
                if nb == cam:
                    continue
                pts_n, valid_n, depth_n, K_n = reproj_inputs(cache, cam, nb, hw, DEV)   # target pixels in nb's frame
                nm, vis_n = project_reference_masks(seeds[nb], pts_n, valid_n, depth_n, K_n)
                union_all |= nm
                if nb in cand:
                    union_k3 |= nm
                anyvis |= vis_n
                per_nb[nb] = {"cover": cnt(missed & nm), "vis": cnt(missed & vis_n[None]),
                              "nm_px": cnt(nm), "nm_on_gt": cnt(nm & gt), "nm_on_other": cnt(nm & ~gt & (torch.from_numpy(gt_lab).to(DEV) != 0)[None]),
                              "dist": float(np.linalg.norm(centres[nb] - centres[cam]))}
                del pts_n, valid_n, depth_n, nm, vis_n
            c_nb_all = cnt(missed & union_all)
            c_nb_k3 = cnt(missed & union_k3)
            c_nb_vis = cnt(missed & anyvis[None])
            # control: the same quantities on the HIT region (GT & seed) -- what the reprojection transfers when the
            # neighbours' seeds do contain the pixels (reprojection precision / occlusion baseline)
            hit = gt & seed
            c_hit = cnt(hit)
            c_hit_vis, c_hit_ongt = cnt(hit & vis[None]), cnt(hit & pm)
            c_hit_nb_all, c_hit_nb_k3, c_hit_nb_vis = cnt(hit & union_all), cnt(hit & union_k3), cnt(hit & anyvis[None])
            del hit
            # --- per-pair J&F (CPU)
            gt_np, seed_np, miss_np = gt.cpu().numpy(), seed.cpu().numpy(), missed.cpu().numpy()
            vis_np, pm_np = vis.cpu().numpy(), pm.cpu().numpy()
            ua_np, uk_np = union_all.cpu().numpy(), union_k3.cpu().numpy()
            for r in srows:
                if r["cam"] != cam:
                    continue
                b = oi[r["obj"]]
                g, s, m = gt_np[b], seed_np[b], miss_np[b]
                base = jf(g, s, bp, se)
                assert abs(base["JF"] - r["JF"]) < 2e-4, (scene, cam, r["obj"], base, r["JF"])
                nm_ = int(n_missed[b])
                rec = {"scene": scene, "cam": cam, "obj": r["obj"], "label": r["label"], "kind": r["kind"], "cost": r["cost"],
                       "JF": r["JF"], "gt_px": int(g.sum()), "seed_px": int(s.sum()), "missed_px": nm_,
                       "missed_vis": int(c_vis[b]), "missed_outside": int(c_out[b]), "missed_occluded": int(c_occ[b]),
                       "missed_nodepth": int(c_nod[b]), "missed_on_refgt": int(c_ongt[b]),
                       "missed_nb_any_vis": int(c_nb_vis[b]), "missed_nb_union_all": int(c_nb_all[b]), "missed_nb_union_k3": int(c_nb_k3[b]),
                       "refproj_recall": float(pm_rec[b]), "refproj_precision": float(pm_prec[b]),
                       "hit_px": int(c_hit[b]), "hit_vis": int(c_hit_vis[b]), "hit_on_refgt": int(c_hit_ongt[b]),
                       "hit_nb_any_vis": int(c_hit_nb_vis[b]), "hit_nb_union_all": int(c_hit_nb_all[b]), "hit_nb_union_k3": int(c_hit_nb_k3[b]),
                       "k3": cand, "failing": r["JF"] < 0.8,
                       "nb": {nb: {"cover": int(d["cover"][b]), "vis": int(d["vis"][b]), "nm_px": int(d["nm_px"][b]),
                                   "nm_on_gt": int(d["nm_on_gt"][b]), "nm_on_other": int(d["nm_on_other"][b]), "dist": d["dist"]}
                              for nb, d in per_nb.items()},
                       "jf": {"seed": base}}
                if nm_ > 0:
                    rec["jf"]["seed_or_missvis"] = jf(g, s | (m & vis_np), bp, se)            # (a) only
                    rec["jf"]["seed_or_missnovis"] = jf(g, s | (m & ~vis_np), bp, se)        # (b) only
                    rec["jf"]["seed_or_gt"] = jf(g, s | g, bp, se)                           # both
                    rec["jf"]["seed_or_miss_onrefgt"] = jf(g, s | (m & pm_np[b]), bp, se)    # (a) restricted to ref-GT-labelled
                    rec["jf"]["seed_or_miss_nb_all"] = jf(g, s | (m & ua_np[b]), bp, se)     # oracle: what any neighbour knows
                    rec["jf"]["seed_or_miss_nb_k3"] = jf(g, s | (m & uk_np[b]), bp, se)
                else:
                    for k in ("seed_or_missvis", "seed_or_missnovis", "seed_or_gt", "seed_or_miss_onrefgt",
                              "seed_or_miss_nb_all", "seed_or_miss_nb_k3"):
                        rec["jf"][k] = dict(base)
                # GT-free candidates (no GT used in the mask)
                rec["jf"]["free_seed_or_refproj"] = jf(g, s | pm_np[b], bp, se)
                rec["jf"]["free_seed_or_nb_k3"] = jf(g, s | uk_np[b], bp, se)
                rec["jf"]["free_seed_or_nb_all"] = jf(g, s | ua_np[b], bp, se)
                rec["jf"]["free_seed_or_refproj_or_nb_k3"] = jf(g, s | pm_np[b] | uk_np[b], bp, se)
                out_rows.append(rec)
            del gt, missed, pts, valid_full, depth_ref, pm, vis, inside, outside, occluded, nodepth, union_all, union_k3, anyvis
            if DEV == "cuda":
                torch.cuda.empty_cache()
        del seeds, ref_masks
        log(f"{scene}: {len(srows)} pairs, {len(targets)} targets x {len(others) - 1} neighbours, {time.time() - ts:.1f} s")
    meta_out = {"seeds": SEEDS, "rows_src": ROWS, "device": DEV, "occ_tol": OCC_TOL, "k": K_NEIGHBOURS,
                "seconds": time.time() - t0, "n_rows": len(out_rows),
                "gpu_mb": float(torch.cuda.max_memory_allocated() / 2**20) if DEV == "cuda" else None}
    json.dump({"meta": meta_out, "rows": out_rows}, sys.stdout)
    log(f"done: {len(out_rows)} rows in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

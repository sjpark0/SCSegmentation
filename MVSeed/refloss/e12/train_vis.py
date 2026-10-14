#!/usr/bin/env python3
"""S1-E12 pre-analysis (visibility signal): how much of the TARGET GT that the stage-1 trainer supervises is
INVISIBLE from the reference camera, and how much of that invisible GT the adopted model already covers.

Part `train`: the training samples of the adopted recipe exactly as geomem.train enumerates them
(geomem.data.enumerate_samples: every scene, train_frames[::2], (r, v) ordered pairs of the 3 annotated cameras, r != v),
objects = geomem.data.objects_for (the ids of r's label map at t; the trainer draws <= 16 of them uniformly, so the
all-objects population is the unbiased version of what it supervises).  For every (sample, object) the target GT
G_o = (label map of v at t == o) is split with the stage-1 reprojection machinery on the training frame's Pi3 cache
(geomem.cache.pi3_frame_cache; methods.geomem.run.reproj_inputs + project_reference_masks, i.e.
geomem.warp_torch.reproject_coords with the occlusion test z <= d_ref * (1 + OCC_TOL), OCC_TOL = 0.01, and the Pi3
validity conf > 0.1 & depth > 0):

    vis       the target pixel's Pi3 point projects inside the reference image, in front of it, not hidden behind the
              reference surface (project_reference_masks' `vis`)                                  -> visible
    outside   valid Pi3 depth, but the point falls outside the reference image / behind the camera  -+
    occluded  inside the reference image but fails the depth test                                  +-> invisible
    nodepth   no valid Pi3 depth at the target pixel (conf <= 0.1 or depth 0)                      -+
    on_refgt  vis AND the reprojected position lands on the reference GT of the SAME object (project_reference_masks)
    on_refany vis AND lands on any labelled reference pixel (label != 0); on_refother = on_refany & ~on_refgt

Part `keyframe`: the 664 scored pairs of MVSeed/runs/seedfail.json (GT = stage1/Data/<scene>/masks/<cam>.png,
reference = meta ref_cam, Pi3 cache = Cache/pi3) with the same split of the GT, and for each seed folder
(geoset_pred_ft_rep_x_e11 = adopted since S1-E11; geoset_pred_ft_rep_e8 = previous) the hits and false positives
by visibility class: recall on visible GT vs invisible GT, and where the false positives lie.

Runs inside scsam3 (torch; GPU if available).  The scratchpad is not mounted in the container, so:
    stage1/scripts/docker_run.sh scsam3 python -c "$(cat train_vis.py)" train    [Scene ...] > train_vis.json
    stage1/scripts/docker_run.sh scsam3 python -c "$(cat train_vis.py)" keyframe [Scene ...] > keyframe_vis.json
(logs on stderr).
"""
import json
import os
import sys
import time

import cv2  # noqa: F401  (reproj_inputs needs it)
import numpy as np
import torch

REPO = "/host/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from geomem.cache import pi3_frame_cache  # noqa: E402
from geomem.data import _labelmap, enumerate_samples, objects_for  # noqa: E402
from geomem.warp_torch import OCC_TOL, reproject_coords  # noqa: E402
from methods.geomem.run import project_reference_masks, reproj_inputs  # noqa: E402
from mvseg2d.io import load_labelmap, load_labels, load_meta, read_pred_mask  # noqa: E402

DATA = os.path.join(S1, "Data")
ROWS = os.path.join(REPO, "MVSeed", "runs", "seedfail.json")
SEED_SETS = ("geoset_pred_ft_rep_x_e11", "geoset_pred_ft_rep_e8")
KF_SCORES = {s: os.path.join(REPO, "MVSeed", "runs", "e11", f"kf_{s}.json") for s in SEED_SETS}
FRAME_STRIDE = 2            # the adopted recipe (geoset_pred_ft: --frame-stride 2, docs/stage1-E8-prereg.md)
OBJ_CHUNK = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def cnt(m):
    return m.flatten(1).sum(1).cpu().numpy().astype(np.int64)


def visibility(cache, cam, ref, hw, ref_lab_np, objs):
    """Reference-visibility classes of every pixel of view `cam` and the reference GT projected into it.
    Returns dict(vis, outside, occluded, nodepth [h,w] bool; pm [B,h,w] bool (same object), pm_any [h,w] bool)."""
    pts, valid_full, depth_ref, K_ref = reproj_inputs(cache, cam, ref, hw, DEV)
    _, inside = reproject_coords(pts, K_ref, hw, depth_ref=None)
    inside = inside & valid_full
    ref_lab = torch.from_numpy(ref_lab_np).to(DEV)
    pm_any, vis = project_reference_masks((ref_lab != 0)[None], pts, valid_full, depth_ref, K_ref)
    pm_any = pm_any[0]
    pm = torch.zeros((len(objs),) + tuple(hw), dtype=torch.bool, device=DEV)
    for i in range(0, len(objs), OBJ_CHUNK):          # chunked: [16,h,w] float for grid_sample
        sub = objs[i:i + OBJ_CHUNK]
        ref_masks = ref_lab[None] == torch.tensor(sub, device=DEV)[:, None, None]
        pm[i:i + len(sub)], vis2 = project_reference_masks(ref_masks, pts, valid_full, depth_ref, K_ref)
        assert bool((vis2 == vis).all())
    out = {"vis": vis, "outside": valid_full & ~inside, "occluded": inside & ~vis, "nodepth": ~valid_full,
           "pm": pm, "pm_any": pm_any}
    del pts, valid_full, depth_ref, K_ref, inside
    return out


def split_counts(masks, V, pm):
    """masks [B,h,w] bool -> per-object pixel counts by visibility class (and the same-object / any-label landing)."""
    return {"px": cnt(masks), "vis": cnt(masks & V["vis"][None]), "outside": cnt(masks & V["outside"][None]),
            "occluded": cnt(masks & V["occluded"][None]), "nodepth": cnt(masks & V["nodepth"][None]),
            "on_refgt": cnt(masks & pm), "on_refany": cnt(masks & V["pm_any"][None] & V["vis"][None])}


def run_train(only):
    t0 = time.time()
    scenes = sorted(os.listdir(DATA))
    if only:
        scenes = [s for s in scenes if s in only]
    samples = enumerate_samples(DATA, scenes, FRAME_STRIDE)       # the trainer's enumeration (all folds together)
    log(f"{len(samples)} training samples over {len(scenes)} scenes (stride {FRAME_STRIDE})")
    rows, sample_rows = [], []
    labels_cache = {}
    for i, s in enumerate(samples):
        ts = time.time()
        if s.scene not in labels_cache:
            labels_cache[s.scene] = load_labels(DATA, s.scene)
        labels = labels_cache[s.scene]
        objs = objects_for(s, DATA)                                 # ids of the reference label map at t
        ref_lab = _labelmap(DATA, s, s.ref_cam)
        tgt_lab = _labelmap(DATA, s, s.tgt_cam)
        hw = tuple(tgt_lab.shape)
        assert ref_lab.shape == hw, (s, ref_lab.shape, hw)
        cache = pi3_frame_cache(DATA, s.scene, s.t)
        V = visibility(cache, s.tgt_cam, s.ref_cam, hw, ref_lab, objs)
        lab_t = torch.from_numpy(tgt_lab).to(DEV)
        gt = lab_t[None] == torch.tensor(objs, device=DEV)[:, None, None]       # [B,h,w] the supervised target GT
        g = split_counts(gt, V, V["pm"])
        pm_px, pm_on_gt = cnt(V["pm"]), cnt(V["pm"] & gt)
        ref_px = np.array([int((ref_lab == o).sum()) for o in objs], np.int64)
        tgt_objs = set(np.unique(tgt_lab).tolist()) - {0}
        extra = sorted(tgt_objs - set(objs))                        # labelled in the target but absent from the reference map
        img_px = int(hw[0] * hw[1])
        vis_px = int(V["vis"].sum())
        for b, o in enumerate(objs):
            rows.append({"scene": s.scene, "t": s.t, "ref": s.ref_cam, "cam": s.tgt_cam, "obj": int(o),
                         "label": (labels.get(int(o)) or {}).get("label"), "type": (labels.get(int(o)) or {}).get("type"),
                         "ref_px": int(ref_px[b]), "gt_px": int(g["px"][b]), "gt_vis": int(g["vis"][b]),
                         "gt_outside": int(g["outside"][b]), "gt_occluded": int(g["occluded"][b]), "gt_nodepth": int(g["nodepth"][b]),
                         "gt_on_refgt": int(g["on_refgt"][b]), "gt_on_refany": int(g["on_refany"][b]),
                         "refproj_px": int(pm_px[b]), "refproj_on_gt": int(pm_on_gt[b]), "img_px": img_px})
        sample_rows.append({"scene": s.scene, "t": s.t, "ref": s.ref_cam, "cam": s.tgt_cam, "n_obj": len(objs),
                            "n_obj_tgt_only": len(extra), "img_px": img_px, "vis_px": vis_px,
                            "outside_px": int(V["outside"].sum()), "occluded_px": int(V["occluded"].sum()),
                            "nodepth_px": int(V["nodepth"].sum())})
        del V, lab_t, gt
        if DEV == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or i == 0:
            log(f"[{i + 1}/{len(samples)}] {s.scene} t={s.t} {s.ref_cam}->{s.tgt_cam} B={len(objs)} {time.time() - ts:.2f} s "
                f"(elapsed {time.time() - t0:.0f} s)")
    meta = {"part": "train", "frame_stride": FRAME_STRIDE, "device": DEV, "occ_tol": OCC_TOL, "n_samples": len(samples),
            "n_rows": len(rows), "seconds": time.time() - t0,
            "gpu_mb": float(torch.cuda.max_memory_allocated() / 2**20) if DEV == "cuda" else None}
    json.dump({"meta": meta, "rows": rows, "samples": sample_rows}, sys.stdout)
    log(f"done: {len(rows)} rows / {len(samples)} samples in {time.time() - t0:.1f} s")


def read_seed(scene, seeds, cam, obj, hw):
    p = os.path.join(DATA, scene, "Result", seeds, cam, f"{obj}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    m = read_pred_mask(p)
    assert m.shape == tuple(hw), (p, m.shape, hw)
    return m


def run_keyframe(only):
    t0 = time.time()
    rows_in = json.load(open(ROWS))["rows"]
    scores = {}
    for s, p in KF_SCORES.items():
        d = json.load(open(p))["scenes"]
        scores[s] = {(r["scene"], r["cam"], r["obj"]): r for sc in d.values() for r in sc["rows"]}
    by_scene = {}
    for r in rows_in:
        by_scene.setdefault(r["scene"], []).append(r)
    out_rows = []
    for scene in sorted(by_scene):
        if only and scene not in only:
            continue
        ts = time.time()
        srows = by_scene[scene]
        meta = load_meta(DATA, scene)
        ref = meta["ref_cam"]
        cache = os.path.join(DATA, scene, "Cache", "pi3")
        ref_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{ref}.png"))
        hw = tuple(ref_lab.shape)
        img_px = int(hw[0] * hw[1])
        objs = sorted({r["obj"] for r in srows})
        oi = {o: i for i, o in enumerate(objs)}
        for cam in sorted({r["cam"] for r in srows}):
            gt_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{cam}.png"))
            assert gt_lab.shape == hw
            V = visibility(cache, cam, ref, hw, ref_lab, objs)
            gt = torch.from_numpy(gt_lab).to(DEV)[None] == torch.tensor(objs, device=DEV)[:, None, None]
            g = split_counts(gt, V, V["pm"])
            per_seed = {}
            for sname in SEED_SETS:
                pred = torch.from_numpy(np.stack([read_seed(scene, sname, cam, o, hw) for o in objs])).to(DEV)
                hit = split_counts(pred & gt, V, V["pm"])
                fp = split_counts(pred & ~gt, V, V["pm"])
                per_seed[sname] = {"pred_px": cnt(pred), "hit": hit, "fp": fp}
                del pred
            for r in srows:
                if r["cam"] != cam:
                    continue
                b = oi[r["obj"]]
                rec = {"scene": scene, "cam": cam, "obj": r["obj"], "label": r["label"], "kind": r["kind"], "cost": r["cost"],
                       "JF_e8": r["JF"], "failing_e8": r["JF"] < 0.8, "img_px": img_px,
                       "ref_px": int((ref_lab == r["obj"]).sum()),
                       "gt_px": int(g["px"][b]), "gt_vis": int(g["vis"][b]), "gt_outside": int(g["outside"][b]),
                       "gt_occluded": int(g["occluded"][b]), "gt_nodepth": int(g["nodepth"][b]),
                       "gt_on_refgt": int(g["on_refgt"][b]), "gt_on_refany": int(g["on_refany"][b]), "seeds": {}}
                assert rec["gt_px"] == r["gt_px"], (scene, cam, r["obj"], rec["gt_px"], r["gt_px"])
                for sname, d in per_seed.items():
                    sc = scores[sname].get((scene, cam, r["obj"]))
                    rec["seeds"][sname] = {"pred_px": int(d["pred_px"][b]),
                                           "J": sc["J"] if sc else None, "F": sc["F"] if sc else None,
                                           "JF": (sc["J"] + sc["F"]) / 2 if sc else None,
                                           **{f"hit_{k}": int(v[b]) for k, v in d["hit"].items()},
                                           **{f"fp_{k}": int(v[b]) for k, v in d["fp"].items()}}
                out_rows.append(rec)
            del V, gt
            if DEV == "cuda":
                torch.cuda.empty_cache()
        log(f"{scene}: {len(srows)} pairs, {time.time() - ts:.1f} s")
    meta = {"part": "keyframe", "seeds": list(SEED_SETS), "rows_src": ROWS, "device": DEV, "occ_tol": OCC_TOL,
            "n_rows": len(out_rows), "seconds": time.time() - t0,
            "gpu_mb": float(torch.cuda.max_memory_allocated() / 2**20) if DEV == "cuda" else None}
    json.dump({"meta": meta, "rows": out_rows}, sys.stdout)
    log(f"done: {len(out_rows)} rows in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    part = sys.argv[1]
    only = [a for a in sys.argv[2:] if not a.startswith("-")]
    {"train": run_train, "keyframe": run_keyframe}[part](only)

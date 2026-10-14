"""S1-E12 code map: per-training-sample cost of the stage-1 reprojection machinery and the reference-visibility of
the target GT, over the exact geoset_pred_ft sample set (every scene, train frames stride 2, ordered annotated pairs).

Run inside the SAM 3 image (cwd stage1, PYTHONPATH stage1):
    stage1/scripts/docker_run.sh scsam3 python -c "$(cat vis_cost.py)" > vis_cost.out
Per sample: reproj_inputs(cache, tgt, ref, hw) [npz load + numpy + cv2 + H2D] and project_reference_masks(...) +
reproject_coords(depth_ref=None) [GPU], timed with cuda.synchronize.  Pixel classes are EXCLUSIVE here:
  nodepth  = ~valid_full (Pi3 conf <= 0.1 or depth 0 at the target pixel)
  outside  = valid_full & ~inside   (in front of the reference camera / inside its image fails)
  occluded = valid_full & inside & ~vis
  vis      = valid_full & inside & depth test passed   (what project_reference_masks calls vis)
  on_refgt = vis & the reprojected position lands on the reference GT of the SAME object (pm[b])
Objects = all ids of the reference label map (training caps at 16 per step, choose_objects); stats are over objects
PRESENT in the target GT (n_gt > 0).  Prints one JSON.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from geomem.cache import pi3_frame_cache  # noqa: E402
from geomem.data import _labelmap, enumerate_samples, objects_for  # noqa: E402
from geomem.warp_torch import reproject_coords  # noqa: E402
from methods import common  # noqa: E402
from methods.geomem.run import project_reference_masks, reproj_inputs  # noqa: E402
from mvseg2d.io import labelmap_to_mask  # noqa: E402

DATA = os.path.join(common.REPO, "Data")
scenes = common.all_scenes(DATA)
samples = enumerate_samples(DATA, scenes, 2)
print(f"scenes {len(scenes)} samples {len(samples)}", file=sys.stderr, flush=True)

per_scene = {}
t_in, t_proj, t_masks = [], [], []
obj_rows = []     # (scene, t, ref, tgt, obj, n_gt, vis, out, occ, nod, ongt, present_in_ref)
t_start = time.time()
for i, s in enumerate(samples):
    cache = pi3_frame_cache(DATA, s.scene, s.t)
    ref_lab = _labelmap(DATA, s, s.ref_cam)
    tgt_lab = _labelmap(DATA, s, s.tgt_cam)
    hw = tuple(ref_lab.shape)
    objs = objects_for(s, DATA)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pts, valid_full, depth_ref, K_ref = reproj_inputs(cache, s.tgt_cam, s.ref_cam, hw, "cuda")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ref_masks = torch.from_numpy(np.stack([labelmap_to_mask(ref_lab, o) for o in objs])).cuda()
    tgt_masks = torch.from_numpy(np.stack([labelmap_to_mask(tgt_lab, o) for o in objs])).cuda()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    pm, vis = project_reference_masks(ref_masks, pts, valid_full, depth_ref, K_ref)
    _, inside = reproject_coords(pts, K_ref, hw, depth_ref=None)
    nodepth = ~valid_full
    outside = valid_full & ~inside
    occluded = valid_full & inside & ~vis
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    t_in.append(t1 - t0); t_masks.append(t2 - t1); t_proj.append(t3 - t2)

    gt = tgt_masks
    n_gt = gt.flatten(1).sum(1)
    n_vis = (gt & vis[None]).flatten(1).sum(1)
    n_out = (gt & outside[None]).flatten(1).sum(1)
    n_occ = (gt & occluded[None]).flatten(1).sum(1)
    n_nod = (gt & nodepth[None]).flatten(1).sum(1)
    n_ongt = (gt & pm).flatten(1).sum(1)
    n_pm = pm.flatten(1).sum(1)                 # projected reference mask size (the GeoProj memory would carry this)
    n_pm_not_gt = (pm & ~gt).flatten(1).sum(1)  # projected pixels that are NOT target GT (projection error / occluder)
    for b, o in enumerate(objs):
        obj_rows.append((s.scene, s.t, s.ref_cam, s.tgt_cam, int(o), int(n_gt[b]), int(n_vis[b]), int(n_out[b]),
                         int(n_occ[b]), int(n_nod[b]), int(n_ongt[b]), int(n_pm[b]), int(n_pm_not_gt[b])))
    sc = per_scene.setdefault(s.scene, {"hw": list(hw), "work_hw": list(np.load(os.path.join(cache, f"{s.ref_cam}.npz"))["depth"].shape),
                                        "n_samples": 0, "n_objs_ref": 0,
                                        "px_vis_frac_img": [], "px_nodepth_frac_img": []})
    sc["n_samples"] += 1
    sc["n_objs_ref"] += len(objs)
    sc["px_vis_frac_img"].append(float(vis.float().mean()))
    sc["px_nodepth_frac_img"].append(float(nodepth.float().mean()))
    del pts, valid_full, depth_ref, K_ref, ref_masks, tgt_masks, pm, vis, inside, nodepth, outside, occluded
    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{len(samples)} {time.time() - t_start:.0f}s", file=sys.stderr, flush=True)

rows = np.array([r[5:] for r in obj_rows], np.int64)
present = rows[:, 0] > 0
R = rows[present]
tot = R.sum(0)
inv = R[:, 2] + R[:, 3] + R[:, 4]
frac_inv_obj = inv / R[:, 0]
summary = {
    "n_scenes": len(scenes), "n_samples": len(samples), "n_object_rows": int(len(rows)), "n_present_in_target": int(present.sum()),
    "pixel_weighted_over_present_target_gt": {
        "vis": float(tot[1] / tot[0]), "outside": float(tot[2] / tot[0]), "occluded": float(tot[3] / tot[0]),
        "nodepth": float(tot[4] / tot[0]), "invisible_total": float((tot[2] + tot[3] + tot[4]) / tot[0]),
        "on_refgt_same_object": float(tot[5] / tot[0])},
    "object_mean_invisible_frac": float(frac_inv_obj.mean()),
    "object_median_invisible_frac": float(np.median(frac_inv_obj)),
    "objects_with_invisible_frac_ge_0.2": int((frac_inv_obj >= 0.2).sum()),
    "objects_with_invisible_frac_ge_0.5": int((frac_inv_obj >= 0.5).sum()),
    "projected_ref_mask": {"px_total": int(tot[6]), "px_not_target_gt": int(tot[7]),
                           "precision_vs_target_gt": float(1 - tot[7] / max(tot[6], 1)),
                           "recall_of_target_gt": float(tot[5] / tot[0])},
    "timing_s_per_sample": {
        "reproj_inputs_mean": float(np.mean(t_in)), "reproj_inputs_median": float(np.median(t_in)),
        "masks_h2d_mean": float(np.mean(t_masks)),
        "project_reference_masks+classes_gpu_mean": float(np.mean(t_proj)), "project_gpu_median": float(np.median(t_proj)),
        "total_mean": float(np.mean(t_in) + np.mean(t_proj)), "wall_total_s": time.time() - t_start,
        "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / 2**20)},
    "per_scene": {},
}
for sc, d in per_scene.items():
    rs = np.array([r[5:] for r in obj_rows if r[0] == sc], np.int64)
    rs = rs[rs[:, 0] > 0]
    tt = rs.sum(0)
    summary["per_scene"][sc] = {
        "hw": d["hw"], "work_hw": d["work_hw"], "n_samples": d["n_samples"], "objs_per_step_mean": d["n_objs_ref"] / d["n_samples"],
        "img_vis_frac_mean": float(np.mean(d["px_vis_frac_img"])), "img_nodepth_frac_mean": float(np.mean(d["px_nodepth_frac_img"])),
        "gt_invisible_frac_pixelweighted": float((tt[2] + tt[3] + tt[4]) / tt[0]),
        "gt_outside": float(tt[2] / tt[0]), "gt_occluded": float(tt[3] / tt[0]), "gt_nodepth": float(tt[4] / tt[0]),
        "gt_on_refgt_same_object": float(tt[5] / tt[0]),
        "proj_precision": float(1 - tt[7] / max(tt[6], 1))}
print(json.dumps(summary, indent=1))

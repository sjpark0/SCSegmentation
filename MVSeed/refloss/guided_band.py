"""Host (stage1/.venv: numpy, cv2): S1-E10 control arm C3 (docs/stage1-E10-prereg.md section 2).

The adopted stage-1 masks (stage1/Data/<scene>/Result/geoset_pred_ft_rep_e8) refined by a grey-guided filter
(He et al., box filters, radius r = the band width, eps 0.01, threshold 0.5) and pasted back with the IDENTICAL
band-clipped, never-delete rule of the crop-zoom pass -- no model forward.  Writes Result/geoset_pred_ft_rep_gf_e10
(PNGs through the CLI writer, the reference GT copied, RUN.json, zoom_conf.json with per-pair verdicts).

    stage1/.venv/bin/python MVSeed/refloss/guided_band.py [--data stage1/Data] [--scenes A,B]
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from geomem import zoom  # noqa: E402
from methods import common  # noqa: E402
from mvseg2d.io import result_dir, write_mask  # noqa: E402

SRC, OUT = "geoset_pred_ft_rep_e8", "geoset_pred_ft_rep_gf_e10"
EPS, THR = 1e-2, 0.5


def guided_filter(I, p, r, eps=EPS):
    """He et al. guided filter with a (2r+1)^2 box window: guide I [H,W] float32 in [0,1], input p float32."""
    k = (2 * r + 1, 2 * r + 1)

    def box(x):
        return cv2.boxFilter(x, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_I, mean_p = box(I), box(p)
    corr_I, corr_Ip = box(I * I), box(I * p)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return box(a) * I + box(b)


def refine_scene(data, scene_name):
    scene = common.load_scene(data, scene_name)
    t0 = time.time()
    src, out = result_dir(data, scene_name, SRC), result_dir(data, scene_name, OUT)
    if not os.path.isdir(src):
        raise SystemExit(f"{scene_name}: {src} missing")
    shutil.rmtree(out, ignore_errors=True)
    r = zoom.band_px(scene.hw)
    recs, n = {}, Counter()
    for cam in scene.cams:
        if cam == scene.ref_cam:
            continue
        d = os.path.join(src, cam)
        if not os.path.isdir(d):
            continue
        I = cv2.cvtColor(cv2.imread(scene.image_paths[cam]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        assert I.shape == tuple(scene.hw), (I.shape, scene.hw)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".png"):
                continue
            full = cv2.imread(os.path.join(d, f), cv2.IMREAD_UNCHANGED) > 127
            n["candidates"] += 1
            canvas = guided_filter(I, full.astype(np.float32), r) > THR
            iou = zoom.mask_iou(canvas, full)
            final = zoom.paste(full, canvas, r)
            v = zoom.verdict(1.0, bool(canvas.any()), iou, bool(final.any()))
            if v != "accepted":
                final = full
            added, removed = int((final & ~full).sum()), int((full & ~final).sum())
            n[v] += 1
            n["run"] += 1
            if added or removed:
                n["changed"] += 1
            n["px_added"] += added
            n["px_removed"] += removed
            recs.setdefault(cam, {})[f[:-4]] = dict(verdict=v, iou_crop_vs_full=iou, px_full=int(full.sum()),
                                                     px_final=int(final.sum()), added=added, removed=removed)
            write_mask(os.path.join(out, cam, f), final)
    common.copy_reference(scene, data, OUT)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "zoom_conf.json"), "w") as fh:
        json.dump(recs, fh, indent=1)
    cfg = dict(source=SRC, band_px=r, eps=EPS, thr=THR, zoom_margin=None, n_zoom_skipped_large=0,
               **{f"n_zoom_{k}": int(n[k]) for k in ("candidates", "run", "kept_absent", "kept_iou", "kept_empty", "accepted", "changed")},
               zoom_px_added=int(n["px_added"]), zoom_px_removed=int(n["px_removed"]))
    common.write_run(data, scene_name, OUT, time.time() - t0, 0, cfg)
    print(f"{scene_name:20s} {n['candidates']:4d} masks  accepted {n['accepted']:4d} changed {n['changed']:4d} "
          f"+{n['px_added']} -{n['px_removed']} px  {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(S1, "Data"))
    ap.add_argument("--scenes", default=None, help="comma-separated (default: every scene with meta.json)")
    a = ap.parse_args()
    scenes = a.scenes.split(",") if a.scenes else sorted(s for s in os.listdir(a.data) if os.path.isfile(os.path.join(a.data, s, "meta.json")))
    for sc in scenes:
        refine_scene(a.data, sc)


if __name__ == "__main__":
    main()

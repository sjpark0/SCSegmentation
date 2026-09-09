#!/usr/bin/env python3
"""Score first-frame seed masks against the ground truth (MVSeed, stage 1 only).

Same J as eval/eval_jf.py (mask IoU; empty prediction on empty GT scores 1.0) and the
same F (DAVIS boundary F, bound_th 0.008), read on ONE frame: the scene's start_frame.
The object set is MUVOD's basic set -- the ids visible in c_ini's first frame -- so the
numbers line up with the benchmark's basic evaluation.

Two camera scopes:
  scored   the three annotated cameras (comparable to the benchmark)
  all      every loaded view that has ground truth (only the three have it, so this is
           the same set; kept for when a run writes more)

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \
        python MVSeed/score_seeds.py --runs MVSeed_baseline --out MVSeed/runs/baseline.json
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval.eval_jf import db_eval_iou, db_eval_boundary, _disk, crop_box, BOUND_TH  # noqa: E402

DATA = os.path.join(REPO, "Data", "MVSeg")
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
CENSUS = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")


def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def score_scene(scene, run, cfg, basic):
    """[(camera, obj, J, F, gt_px, pred_px)] on the scene's start_frame."""
    d = cfg[scene]
    ds = os.path.join(DATA, d["folder"])
    start = d["start_frame"]
    rows = []
    for c in sorted(d["cam_list"]):
        cam = cam_name(c, d["prefix"], d["prefix1"])
        gt_path = os.path.join(ds, "Mask", cam, f"{start:06d}.png")
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            continue
        h, w = gt_img.shape[:2]
        bound_pix = int(math.ceil(BOUND_TH * np.linalg.norm((h, w))))
        se = _disk(bound_pix)
        for obj in sorted(basic):
            gm = gt_img == obj
            p = os.path.join(ds, run, cam, str(start), f"{obj}.png")
            pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
            if pm is None:
                pm = np.zeros((h, w), dtype=bool)
            else:
                if pm.shape[:2] != (h, w):
                    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                pm = pm > 127
            box = crop_box(gm, pm, bound_pix + 2, (h, w))
            if box is None:
                J = F = 1.0
            else:
                y0, y1, x0, x1 = box
                J = db_eval_iou(gm[y0:y1, x0:x1], pm[y0:y1, x0:x1])
                F = db_eval_boundary(pm[y0:y1, x0:x1], gm[y0:y1, x0:x1], bound_pix, se)
            rows.append((cam, int(obj), float(J), float(F),
                         int(gm.sum()), int(pm.sum())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="output folder names under each scene")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    census = json.load(open(CENSUS))
    scenes = args.scenes or sorted(census)
    out = {}
    for run in args.runs:
        per_scene = {}
        for scene in scenes:
            if scene not in cfg:
                continue
            basic = set(census[scene]["seed_ids_c_ini"])
            ci = census[scene]["c_ini"]
            rows = score_scene(scene, run, cfg, basic)
            if not rows:
                continue
            non = [r for r in rows if r[0] != ci]
            per_scene[scene] = {
                "c_ini": ci, "n": len(rows), "n_nonref": len(non),
                "J_all": float(np.mean([r[2] for r in rows])),
                "F_all": float(np.mean([r[3] for r in rows])),
                "J_nonref": float(np.mean([r[2] for r in non])) if non else None,
                "fail_nonref": int(sum(1 for r in non if r[2] < 0.5)),
                "rows": [dict(cam=r[0], obj=r[1], J=r[2], F=r[3], gt=r[4], pred=r[5])
                         for r in rows],
            }
        out[run] = per_scene
        js = [v["J_nonref"] for v in per_scene.values() if v["J_nonref"] is not None]
        fails = sum(v["fail_nonref"] for v in per_scene.values())
        tot = sum(v["n_nonref"] for v in per_scene.values())
        print(f"{run:28} scenes {len(per_scene):>2}  비기준 시드 J {np.mean(js):.4f}  "
              f"실패(J<0.5) {fails}/{tot} ({100*fails/max(tot,1):.1f}%)")
        for s, v in sorted(per_scene.items(), key=lambda x: x[1]["J_nonref"] or 1):
            print(f"    {s:22}{v['J_nonref']:.3f}  실패 {v['fail_nonref']}/{v['n_nonref']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

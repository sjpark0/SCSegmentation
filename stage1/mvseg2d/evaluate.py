"""Score one method's result folders against the ground truth (spec 2.3, 4.4).

    .venv/bin/python -m mvseg2d.evaluate --data Data --method sam3_prop [--scenes A,B] --out scores/sam3_prop.json

One row per (scene, scored camera, object) pair.  Scored cameras are the annotated ones minus the
reference camera; objects are the ids in the reference camera's label map.  A pair whose target
ground truth is empty is not a row; it is counted as a hallucination when the prediction is
non-empty.  A missing PNG is an empty prediction.  A PNG whose shape differs from the ground truth
raises ShapeMismatch (nothing is resized).  A scene whose RUN.json carries an error has status
"error" and no rows; one whose RUN.json is missing or still says "running" is "incomplete"; a scene
without a result folder is "missing".
"""
import argparse
import json
import os
import warnings

import numpy as np

from .io import (ShapeMismatch, camera_order, labelmap_objects, labelmap_to_mask, load_labelmap, load_labels,
                 load_meta, read_pred_mask, result_dir, rig_of, size_bin)
from .metrics import bound_pix, disk, jf_pair


def scored_views(meta):
    return [c for c in meta["mask_cams"] if c != meta["ref_cam"]]


def view_dist(meta, cam):
    cams = camera_order(meta["cams"])
    return abs(cams.index(cam) - cams.index(meta["ref_cam"]))


def _read_run(rdir):
    p = os.path.join(rdir, "RUN.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def evaluate_scene(data, scene, method):
    meta = load_meta(data, scene)
    labels = load_labels(data, scene)
    rdir = result_dir(data, scene, method)
    out = {"scene": scene, "method": method, "status": "ok", "error": None, "rows": [],
           "hallucinations": 0, "warnings": [], "run": None}
    if not os.path.isdir(rdir):
        out["status"] = "missing"
        return out
    run = _read_run(rdir)
    out["run"] = run
    if run is not None and run.get("error"):
        out["status"] = "error"
        out["error"] = str(run["error"])
        return out
    if run is None or run.get("status") == "running":
        out["status"] = "incomplete"          # killed before RUN.json was finalised
        return out

    ref_lab = load_labelmap(os.path.join(data, scene, "masks", f"{meta['ref_cam']}.png"))
    objects = labelmap_objects(ref_lab)
    h, w = ref_lab.shape
    bp = bound_pix(h, w)
    se = disk(bp)
    rig = rig_of(scene)

    extra = sorted(d for d in os.listdir(rdir)
                   if os.path.isdir(os.path.join(rdir, d)) and d not in meta["cams"])
    for d in extra:
        out["warnings"].append(f"{scene}/{method}: result folder {d} is not a camera of this scene")
    unknown = [o for o in objects if o not in labels]
    if unknown:
        out["warnings"].append(f"{scene}: object ids {unknown} are not in objects_labels.json")

    for cam in scored_views(meta):
        gt_lab = load_labelmap(os.path.join(data, scene, "masks", f"{cam}.png"))
        if gt_lab.shape != (h, w):
            raise ShapeMismatch(f"{scene}/{cam}: ground truth {gt_lab.shape} vs reference {(h, w)}")
        vd = view_dist(meta, cam)
        for obj in objects:
            gt = labelmap_to_mask(gt_lab, obj)
            p = os.path.join(rdir, cam, f"{obj}.png")
            if os.path.isfile(p):
                pred = read_pred_mask(p)
                if pred.shape != (h, w):
                    raise ShapeMismatch(f"{p}: {pred.shape} vs ground truth {(h, w)}")
            else:
                pred = np.zeros((h, w), bool)
            gt_px = int(gt.sum())
            pred_px = int(pred.sum())
            if gt_px == 0:
                if pred_px > 0:
                    out["hallucinations"] += 1
                continue
            J, F = jf_pair(gt, pred, bp, se)
            lab = labels.get(obj, {})
            out["rows"].append({
                "method": method, "scene": scene, "cam": cam, "obj": obj,
                "label": lab.get("label", "unknown"), "type": lab.get("type", "unknown"),
                "rig": rig, "gt_px": gt_px, "pred_px": pred_px, "view_dist": vd,
                "size_bin": size_bin(gt_px), "J": float(J), "F": float(F)})
    return out


def evaluate_method(data, method, scenes):
    return {"method": method, "scenes": {s: evaluate_scene(data, s, method) for s in scenes}}


def all_scenes(data):
    return sorted(d for d in os.listdir(data) if os.path.isfile(os.path.join(data, d, "meta.json")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="Data")
    ap.add_argument("--method", required=True)
    ap.add_argument("--scenes", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--out", default=None, help="default scores/<method>.json")
    a = ap.parse_args(argv)
    scenes = a.scenes.split(",") if a.scenes else all_scenes(a.data)
    res = evaluate_method(a.data, a.method, scenes)
    out = a.out or os.path.join("scores", f"{a.method}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    n_rows = sum(len(s["rows"]) for s in res["scenes"].values())
    st = {k: sum(1 for s in res["scenes"].values() if s["status"] == k)
          for k in ("ok", "error", "incomplete", "missing")}
    for s in res["scenes"].values():
        for wmsg in s["warnings"]:
            warnings.warn(wmsg)
    print(f"{a.method}: {n_rows} pairs, scenes ok {st['ok']} error {st['error']} incomplete {st['incomplete']} missing {st['missing']} -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S1-E17 (docs/stage1-E17-prereg.md §2): the input-only angle cache of every training sample.

    scripts/e17/run_e17.sh angles            (scsam3 on the CPU; numpy + the Pi3 frame caches)

For every (scene, t, r) of geomem.data.enumerate_samples over the 17 fold scenes at frame stride 2 (the training recipe),
a(r -> c) for every camera c of the scene = run.relay_geometry's object-anchor viewpoint change (E15 §3.1) from r's full
label map at t (every object of the reference) and the frame's Pi3 cache.  No prediction is read.  Entries that also
exist in scripts/e15/runs/angles.json (the same reference, DEV basic set) are compared and the largest difference reported.
-> scripts/e17/runs/train_angles.json {prereg, inputs, frame_stride, n_entries, n_samples, missing, compare, entries: [...]}
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, S1)
import numpy as np  # noqa: E402

from geomem.cache import pi3_frame_cache  # noqa: E402
from geomem.data import enumerate_samples  # noqa: E402
from geomem.folds import load_folds  # noqa: E402
from methods import common  # noqa: E402
from mvseg2d.io import labelmap_objects, load_labelmap, load_meta  # noqa: E402

DATA = os.path.join(S1, "Data")
OUT = os.path.join(HERE, "runs", "train_angles.json")
E15_ANGLES = os.path.join(S1, "scripts", "e15", "runs", "angles.json")


def entry(scene, t, ref):
    from methods.geomem import run as R
    cams = list(load_meta(DATA, scene)["cams"])
    lab = load_labelmap(os.path.join(common.frame_dir(DATA, scene, t), "masks", f"{ref}.png"))
    objects = labelmap_objects(lab)
    masks = np.stack([lab == o for o in objects]) if objects else np.zeros((0,) + lab.shape, bool)
    g = R.relay_geometry(pi3_frame_cache(DATA, scene, t), cams, ref, masks, math.inf)
    return dict(scene=scene, t=int(t), ref=ref, kind=g["kind"], fallback=g["fallback"], n_objects=len(objects),
                all={c: float(a) for c, a in g["angles"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    scenes = sorted(load_folds())
    samples = enumerate_samples(DATA, scenes, a.frame_stride)
    keys = sorted({(s.scene, s.t, s.ref_cam) for s in samples})
    entries, missing = [], []
    for i, (scene, t, ref) in enumerate(keys):
        e = entry(scene, t, ref)
        entries.append(e)
        cams = [c for c in load_meta(DATA, scene)["cams"] if c != ref]
        lack = [c for c in cams if c not in e["all"]]
        if lack:
            missing.append(dict(scene=scene, t=t, ref=ref, fallback=e["fallback"], cams=lack))
        if i % 50 == 0:
            print(f"{i}/{len(keys)} {scene} t={t} {ref} kind={e['kind']}", flush=True)
    comp = dict(n_common=0, max_abs_diff=0.0, worst=None)
    if os.path.isfile(E15_ANGLES):
        old = {(e["scene"], e["t"], e["ref"]): e for e in json.load(open(E15_ANGLES))["entries"]}
        for e in entries:
            o = old.get((e["scene"], e["t"], e["ref"]))
            if o is None:
                continue
            comp["n_common"] += 1
            for c, v in e["all"].items():
                if c in o["all"]:
                    d = abs(v - o["all"][c])
                    if d > comp["max_abs_diff"]:
                        comp.update(max_abs_diff=d, worst=dict(scene=e["scene"], t=e["t"], ref=e["ref"], cam=c, train=v, e15=o["all"][c]))
    vals = np.array([e["all"][s.tgt_cam] for s in samples for e in [next(x for x in entries if (x["scene"], x["t"], x["ref"]) == (s.scene, s.t, s.ref_cam))]])
    res = dict(prereg="docs/stage1-E17-prereg.md §2 (C-ov)", inputs="reference label map (all objects) + Pi3 frame cache only",
               frame_stride=a.frame_stride, n_entries=len(entries), n_samples=len(samples), missing=missing, compare=comp,
               sample_angles=dict(n=int(vals.size), frac_over_45=float((vals > 45).mean()),
                                  quantiles={q: float(np.percentile(vals, q)) for q in (50, 90, 99, 100)}),
               entries=entries)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{len(entries)} entries, {len(samples)} samples, missing {len(missing)}, compare {comp}, "
          f"frac>45 {res['sample_angles']['frac_over_45']:.4f} -> {a.out}")


if __name__ == "__main__":
    main()

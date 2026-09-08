#!/usr/bin/env python3
"""Per-scene object sets under MUVOD's protocol, and the one check its tables give us.

MUVOD reports two numbers per scene: "basic" (only the objects visible in c_ini's
reference frame) and "complete" (every labelled object).  The two are equal exactly
when c_ini's reference frame already holds every labelled object.  That turns the
published tables into a test of our c_ini: for each scene, whether c_ini holds every
object must match whether the published basic and complete scores are equal.

Writes docs/raw/muvod_object_sets.json:

    <scene>: c_ini, per-camera seed counts, the scene's object universe,
             c_ini_holds_all, published_basic_equals_complete, agree

Needs numpy and cv2, so run it in the container:

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \
        python eval/muvod_census.py
"""
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval.report_jf import MUVOD_BASELINE, MUVOD_ORDER      # noqa: E402

CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
DATA = os.path.join(REPO, "Data", "MVSeg")
OUT = os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")


def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def ids_in(path):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        sys.exit(f"missing ground truth: {path}")
    return {int(x) for x in np.unique(a).tolist() if x}


def main():
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    out, disagree = {}, []
    for scene in MUVOD_ORDER:
        c = cfg[scene]
        ds = os.path.join(DATA, c["folder"])
        if not os.path.isdir(ds):
            print(f"{scene}: not installed, skipped", file=sys.stderr)
            continue
        names = [cam_name(x, c["prefix"], c["prefix1"]) for x in sorted(c["cam_list"])]
        c_ini = cam_name(c["c_ini"], c["prefix"], c["prefix1"])
        seed, universe = {}, set()
        for n in names:
            d = os.path.join(ds, "Mask", n)
            for f in sorted(os.listdir(d)):
                if f.endswith(".png"):
                    universe |= ids_in(os.path.join(d, f))
            seed[n] = sorted(ids_in(os.path.join(d, f"{c['start_frame']:06d}.png")))
        labels = json.load(open(os.path.join(ds, "Mask", "objects_labels.json")))
        holds = set(seed[c_ini]) >= universe
        equal = MUVOD_BASELINE["basic"][scene] == MUVOD_BASELINE["complete"][scene]
        out[scene] = {
            "c_ini": c_ini, "cameras": names,
            "seed_counts": {n: len(seed[n]) for n in names},
            "seed_ids_c_ini": seed[c_ini],
            "n_objects": len(universe), "n_labels": len(labels),
            "cameras_holding_all": [n for n in names if set(seed[n]) >= universe],
            "c_ini_holds_all": holds,
            "published_basic": MUVOD_BASELINE["basic"][scene],
            "published_complete": MUVOD_BASELINE["complete"][scene],
            "published_basic_equals_complete": equal,
            "agree": holds == equal,
        }
        if not out[scene]["agree"]:
            disagree.append(scene)
        print(f"{scene:22} c_ini {c_ini:12} seed {len(seed[c_ini]):>3}/{len(universe):<3} "
              f"holds_all {str(holds):5} published_equal {str(equal):5} "
              f"{'OK' if holds == equal else '*** DISAGREE ***'}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"\nwrote {OUT}: {len(out)} scenes, {len(disagree)} disagreements"
          + (f" ({', '.join(disagree)})" if disagree else ""))
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())

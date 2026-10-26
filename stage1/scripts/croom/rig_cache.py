"""Rig calibration cache (input only): from a joint Pi3X run Cache/pi3_multi_<tag>/ whose anchor is a calibration instant
of the SAME fixed rig (every anchor camera = the same physical camera as the target camera with that name), write
Cache/<out>/ where each target camera keeps its own depth / conf / K (local pointmap, pose-independent) but takes the
POSE (c2w) of the anchor camera of the same name.  Joint run => one frame and scale for anchors and targets.

    python stage1/scripts/croom/rig_cache.py --scene CROOM_T04_001000_r00 --tag t06 [--out pi3_rig_t06]
"""
import argparse
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = os.path.join(a.data, a.scene, "Cache", f"pi3_multi_{a.tag}")
    out = os.path.join(a.data, a.scene, "Cache", a.out or f"pi3_rig_{a.tag}")
    m = json.load(open(os.path.join(src, "MANIFEST.json")))
    anchors = {n.split("/")[-1]: np.asarray(v, np.float64) for n, v in m["anchors_c2w"].items()}
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    moved = {}
    for c in m["cams"]:
        z = dict(np.load(os.path.join(src, f"{c}.npz")))
        own = z["c2w"]
        z["c2w"] = anchors[c]
        moved[c] = round(float(np.linalg.norm(own[:3, 3] - anchors[c][:3, 3])), 4)
        np.savez(os.path.join(out, f"{c}.npz"), **z)
    m2 = {k: v for k, v in m.items() if k != "anchors_c2w"}
    m2.update(rig_pose_source=f"pi3_multi_{a.tag} anchors (same-name camera)", own_to_anchor_centre_dist=moved)
    json.dump(m2, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    print(f"wrote {out}: {len(moved)} cams, centre moved median {np.median(list(moved.values())):.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Pi3X on every training frame of a scene, all cameras, reference camera first (spec 4.1).

    scripts/docker_run.sh scgeo:pi3 python -m geomem.pi3_frames --scene <name|all> [--data Data]

Writes Data/<scene>/Cache/pi3_frames/<t>/ with the same layout as Cache/pi3/ (methods/pi3_reproj/geometry.py).
Frames with a MANIFEST.json are skipped.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.cache import missing_pi3_frames, pi3_frame_cache  # noqa: E402
from methods import common  # noqa: E402
from methods.pi3_reproj.geometry import PIXEL_LIMIT, run_pi3  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=os.path.join(common.REPO, "Data"))
    a = ap.parse_args()
    scenes = common.all_scenes(a.data) if a.scene == "all" else [a.scene]
    for name in scenes:
        todo = missing_pi3_frames(a.data, name)
        if not todo:
            print(f"{name:20s} cached", flush=True)
            continue
        t0 = time.time()
        for t in todo:
            scene = common.load_scene(a.data, name, frame=t)
            run_pi3(scene, pi3_frame_cache(a.data, name, t), PIXEL_LIMIT)
        print(f"{name:20s} {len(todo)} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

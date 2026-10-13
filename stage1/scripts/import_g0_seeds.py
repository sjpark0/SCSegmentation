"""Import the parent repository's stage-1 seeds (G0, MVSeed_e0_index) as the benchmark method `g0_seed`.

    .venv/bin/python scripts/import_g0_seeds.py [--src ../SCSegmentation/Data/MVSeg] [--data Data] [--out g0_seed]

G0 is the SAM 3 pseudo-video chain in the rig's camera order (numeric, `mvseg2d.io.camera_order`); it is the stage-1
baseline SCSegmentation actually ships.  Our `sam3_prop` before the camera-order fix chained meta.json's lexicographic
order and differs from it in six scenes.  Seeds live at <src>/<scene>/MVSeed_e0_index/<cam>/<ref_frame>/<obj>.png and
are copied to Data/<scene>/Result/<out>/<cam>/<obj>.png with a RUN.json; the reference frame must match meta.json.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mvseg2d.io import load_meta, result_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data", "MVSeg"))
    ap.add_argument("--data", default="Data")
    ap.add_argument("--out", default="g0_seed")
    a = ap.parse_args()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    for scene in sorted(os.listdir(a.data)):
        meta = load_meta(a.data, scene)
        src = os.path.join(a.src, scene, "MVSeed_e0_index")
        with open(os.path.join(src, "SEED_MANIFEST.json"), encoding="utf-8") as f:
            man = json.load(f)
        assert man["start_frame"] == meta["ref_frame"] and man["reference"] == meta["ref_cam"], scene
        dst = result_dir(a.data, scene, a.out)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        n = 0
        for cam in meta["cams"]:
            if cam == meta["ref_cam"]:
                continue                      # reference camera: ground truth, as for every method
            d = os.path.join(src, cam, str(meta["ref_frame"]))
            if not os.path.isdir(d):
                continue
            os.makedirs(os.path.join(dst, cam), exist_ok=True)
            for name in os.listdir(d):
                if name.endswith(".png"):
                    shutil.copyfile(os.path.join(d, name), os.path.join(dst, cam, name))
                    n += 1
        run = {"method": a.out, "scene": scene, "config": {"source": src, "order": man["order"],
               "sequence": man["sequence"], "masks_written": n}, "git": git,
               "date": datetime.datetime.now().isoformat(timespec="seconds"), "error": None, "status": "done"}
        with open(os.path.join(dst, "RUN.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, indent=1)
        print(f"{scene:20s} {n:5d} masks")


if __name__ == "__main__":
    main()

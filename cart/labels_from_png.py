#!/usr/bin/env python3
"""Label volume for FreeTimeGS++ masks from the exported per-frame label PNGs (cart/export_cart.py <out>/labels).

    python cart/labels_from_png.py <labels png dir> <out dir> --start 120 --stop 420 [--workers 8]

Rebuilds what export_cart.py --npy writes, for a frame range, from the PNGs alone (the run masks are on the NAS now):
  <out>/labels.npy   uint8 (F, 24, 1080, 1920) = PNG label with the BAND_PX ignore band of export_cart.banded
  <out>/frames.npy   source frame of each row
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_cart import banded  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("png", help="<export>/labels with camNNNN/<frame>.png")
ap.add_argument("out")
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--stop", type=int, required=True, help="exclusive")
ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()
cams = sorted(d for d in os.listdir(A.png) if d.startswith("cam"))
frames = list(range(A.start, A.stop))
os.makedirs(A.out, exist_ok=True)
vol_path = os.path.join(A.out, "labels.npy")
np.lib.format.open_memmap(vol_path, mode="w+", dtype=np.uint8, shape=(len(frames), len(cams), 1080, 1920)).flush()
np.save(os.path.join(A.out, "frames.npy"), np.array(frames, np.int64))


def job(ci):
    vol = np.load(vol_path, mmap_mode="r+")
    missing = 0
    for i, f in enumerate(frames):
        p = os.path.join(A.png, cams[ci], f"{f:06d}.png")
        lab = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if lab is None:
            missing += 1
            lab = np.zeros((1080, 1920), np.uint8)
        assert lab.shape == (1080, 1920), (p, lab.shape)
        vol[i, ci] = banded(lab)
    vol.flush()
    return cams[ci], missing


with ProcessPoolExecutor(A.workers) as ex:
    for cam, miss in ex.map(job, range(len(cams))):
        print(cam, "missing", miss, flush=True)
print("done", vol_path, (len(frames), len(cams)))

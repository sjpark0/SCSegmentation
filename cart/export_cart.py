#!/usr/bin/env python3
"""Per-frame label and foreground images from a cart run (docs/cart.md), named by source frame number.

    export_cart.py <scene dir> <method> <out dir> [--start 120] [--workers 16]

  labels/camNNNN/<src frame:06d>.png   8-bit grey, pixel = object id (0 background, 1..17); where two objects' masks
                                        overlap, the later id in PAINT wins (the order the reference prompt was painted:
                                        table, cart, sign, laptops, chairs, people, books)
  fg/camNNNN/<src frame:06d>.png       8-bit grey 0/255, union of all objects (what is kept when the background is
                                        replaced)
The per-object binary masks the pipeline writes (<method>/camNNNN/<i>/<id>.png, 0/255) stay as they are.
Host python (stage1/.venv: cv2, numpy).
"""
import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

PAINT = [7, 6, 13, 11, 12, 14, 15, 16, 17, 1, 2, 3, 4, 5, 8, 9, 10]
A = None


def one_cam(cam):
    src = os.path.join(A.scene, A.method, cam)
    ld, fd = os.path.join(A.out, "labels", cam), os.path.join(A.out, "fg", cam)
    os.makedirs(ld, exist_ok=True)
    os.makedirs(fd, exist_ok=True)
    frames = sorted(int(t) for t in os.listdir(src) if t.isdigit())
    stats = {"frames": len(frames), "overlap_px": 0, "fg_px": 0}
    for i in frames:
        lab = None
        for o in PAINT:
            p = os.path.join(src, str(i), f"{o}.png")
            if not os.path.exists(p):
                continue
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
            if lab is None:
                lab = np.zeros(m.shape, np.uint8)
            stats["overlap_px"] += int((m & (lab > 0)).sum())
            lab[m] = o
        if lab is None:
            lab = np.zeros((1080, 1920), np.uint8)
        name = f"{A.start + i:06d}.png"
        cv2.imwrite(os.path.join(ld, name), lab)
        cv2.imwrite(os.path.join(fd, name), (lab > 0).astype(np.uint8) * 255)
        stats["fg_px"] += int((lab > 0).sum())
    return cam, stats


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("method")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--workers", type=int, default=16)
    A = ap.parse_args()
    cams = sorted(c for c in os.listdir(os.path.join(A.scene, A.method)) if c.startswith("cam"))
    with Pool(min(A.workers, len(cams))) as pool:
        rec = dict(pool.map(one_cam, cams))
    labels = json.load(open(os.path.join(A.scene, "Mask", "objects_labels.json")))
    json.dump(dict(scene=A.scene, method=A.method, source_frame_of_index_0=A.start, paint_order=PAINT, objects=labels,
                   per_cam=rec), open(os.path.join(A.out, "export.json"), "w"), indent=1)
    print(json.dumps({c: r["frames"] for c, r in rec.items()})[:300])


if __name__ == "__main__":
    main()

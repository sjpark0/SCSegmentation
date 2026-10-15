#!/usr/bin/env python3
"""Visual check of candidate C on DEV pairs: per pair one row of tiles (1/4 res):
 image+GT_o | D_o (dev mask) | P_o (projection) + ref-invisible tint | S_o per variant | A_o (added, registered config).
stage1/.venv/bin/python -I viz.py out.jpg scene:t:cam:obj [...]"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_score as CS  # noqa: E402
from mvseg2d.io import load_labelmap  # noqa: E402

SC = 4


def tile(img, m, col, tint=None, label=""):
    ov = img.copy()
    if tint is not None:
        ov[tint] = (0.6 * ov[tint] + 0.4 * np.array([90, 90, 90])).astype(np.uint8)
    ov[m] = (0.45 * ov[m] + 0.55 * np.array(col)).astype(np.uint8)
    cv2.putText(ov, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return ov


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    rows = []
    for sp in specs:
        scene, t, cam, o = sp.split(":")
        t, o = int(t), int(o)
        vdir = os.path.join(CS.PROPS, scene, f"{t:06d}", cam)
        meta = json.load(open(os.path.join(vdir, "meta.json")))
        hw = tuple(meta["hw"])
        H, W = hw
        img = cv2.imread(os.path.join(CS.DATA, scene, "frames", f"{t:06d}", "images", f"{cam}.jpg"))
        gt = load_labelmap(os.path.join(CS.DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png")) == o
        cls = cv2.imread(os.path.join(vdir, "cls.png"), cv2.IMREAD_UNCHANGED)
        D = CS.rd(os.path.join(CS.DEV, scene, f"{t:06d}", cam, f"{o}.png"), hw)
        P = CS.rd(os.path.join(vdir, "proj", f"{o}.png"), hw)
        small = (W // SC, H // SC)
        r = lambda m: cv2.resize(m.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
        im = cv2.resize(img, small)
        inv = r(cls > 0)
        tiles = [tile(im, r(gt), (0, 255, 0), label=f"{scene} {cam} t{t} o{o} GT"),
                 tile(im, r(D), (255, 0, 0), label="dev mask D"),
                 tile(im, r(P), (0, 255, 255), tint=inv, label="proj P (grey=ref-invis)")]
        for v in CS.VARIANTS:
            S = CS.rd(os.path.join(vdir, v, f"{o}.png"), hw)
            tiles.append(tile(im, r(S), (0, 0, 255), label=f"S {v} {meta['props'].get(str(o), {}).get(v, {}).get('pred_iou', 0):.2f}"))
        rows.append(np.concatenate(tiles, 1))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()

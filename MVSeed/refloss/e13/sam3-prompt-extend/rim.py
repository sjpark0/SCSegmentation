#!/usr/bin/env python3
"""Where do the primary variant's added pixels lie relative to the dev mask?  Distance of each added pixel to D_o
(cv2.distanceTransform of ~D_o, L2) in bands <= 3 px, (3, bound_pix], > bound_pix; added px and precision per band;
scored pairs only.  DEV ONLY.  stage1/.venv/bin/python -I rim.py --variant pts"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_score as CS  # noqa: E402
from mvseg2d.io import load_labelmap  # noqa: E402
from mvseg2d.metrics import bound_pix  # noqa: E402


def one(job, variant):
    scene, t, cam, ref_frame = job
    assert t != ref_frame
    vdir = os.path.join(CS.PROPS, scene, f"{t:06d}", cam)
    meta = json.load(open(os.path.join(vdir, "meta.json")))
    hw = tuple(meta["hw"])
    bp = bound_pix(*hw)
    objs = [int(o) for o in meta["objs"]]
    props = {int(k) for k in meta["props"]}
    gt = load_labelmap(os.path.join(CS.DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    invis = cv2.imread(os.path.join(vdir, "cls.png"), cv2.IMREAD_UNCHANGED) > 0
    D = {o: CS.rd(os.path.join(CS.DEV, scene, f"{t:06d}", cam, f"{o}.png"), hw) for o in objs}
    cnt = np.zeros(hw, np.uint8)
    for o in objs:
        cnt += D[o]
    Apre = {}
    for o in objs:
        if o in props and meta["proj_px"][str(o)] >= CS.MAIN[0]:
            s = CS.rd(os.path.join(vdir, variant, f"{o}.png"), hw)
            Apre[o] = CS.connected_add(s & invis & ~D[o] & ~((cnt - D[o]) > 0), D[o])
    ca = np.zeros(hw, np.uint8)
    for o in Apre:
        ca += Apre[o]
    acc = np.zeros((3, 2), np.int64)
    for o in Apre:
        A = Apre[o] & (ca == 1)
        if not A.any() or not (gt == o).any():
            continue
        dist = cv2.distanceTransform((~D[o]).astype(np.uint8), cv2.DIST_L2, 5)
        on = A & (gt == o)
        for i, m in enumerate((dist <= 3, (dist > 3) & (dist <= bp), dist > bp)):
            acc[i] += [int((A & m).sum()), int((on & m).sum())]
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pts")
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(CS.DEV, "index.json")))
    jobs = [(s, t, cam, S["ref_frame"]) for s, S in sorted(idx["scenes"].items()) for t in S["frames"] for cam in S["eval_cams"]]
    with Pool(a.procs) as p:
        acc = sum(p.map(partial(one, variant=a.variant), jobs))
    tot = acc[:, 0].sum()
    for name, (n, on) in zip(("<= 3 px", "(3, bound_pix]", "> bound_pix"), acc):
        print(f"{a.variant} added px at distance {name:15s} from D_o: {n:10,d} ({n / tot:.3f} of added), precision {on / max(1, n):.3f}")


if __name__ == "__main__":
    main()

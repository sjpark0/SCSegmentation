#!/usr/bin/env python3
"""Write the composed masks N_o = D_o | A_o of ONE variant at the registered config (min_proj 20, no gate) in the dev-route
layout <root>/<scene>/<t:06d>/<cam>/<o>.png (absent = empty), so that dev_score.py --masks <root> scores it independently.
DEV ONLY.   stage1/.venv/bin/python -I compose_write.py --variant pts --root <dir> --procs 12"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_score as CS  # noqa: E402
from mvseg2d.io import write_mask  # noqa: E402


def one(job, variant, root):
    scene, t, cam, ref_frame = job
    assert t != ref_frame
    vdir = os.path.join(CS.PROPS, scene, f"{t:06d}", cam)
    meta = json.load(open(os.path.join(vdir, "meta.json")))
    hw = tuple(meta["hw"])
    objs = [int(o) for o in meta["objs"]]
    props = {int(k): v for k, v in meta["props"].items()}
    import cv2
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
    n_add = 0
    for o in objs:
        N = D[o] | (Apre[o] & (ca == 1)) if o in Apre else D[o]
        n_add += int(N.sum() - D[o].sum())
        if N.any():
            write_mask(os.path.join(root, scene, f"{t:06d}", cam, f"{o}.png"), N)
    return n_add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pts")
    ap.add_argument("--root", required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(CS.DEV, "index.json")))
    jobs = [(s, t, cam, S["ref_frame"]) for s, S in sorted(idx["scenes"].items()) for t in S["frames"] for cam in S["eval_cams"]]
    with Pool(a.procs) as p:
        n = sum(p.map(partial(one, variant=a.variant, root=a.root), jobs))
    print(f"wrote {len(jobs)} views, {n:,} added px (all seeded objects, incl. empty-GT ones)")


if __name__ == "__main__":
    main()

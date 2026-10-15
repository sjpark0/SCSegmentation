#!/usr/bin/env python3
"""S1-E13 candidate E1 (text-concept) -- POST-HOC DIAGNOSTIC, NOT REGISTERED, written AFTER the registered results were seen
(the Dog camera_0004 tree: its missed overhanging branches are detected by "tree" as a separate instance that neither
overlaps P_o nor is 8-connected to D_o, so the registered composition cannot add it).  DEV ONLY.  Nothing here is selected;
it only measures how much of the registered loss is due to the composition's connectivity step.
Composition 'noconn': A_o = C_o (no 8-connectivity to D_o), contested pixels (>= 2 active objects) dropped, N_o = D_o | A_o.
Arms (TAU 0.5, split names):
  union_all|noconn          every seeded object, S_o = union of all its name's detections
  union_all|noconn|unique   only objects whose name is scene-unique among the basic set (metadata), others inactive
  union_all|conn|unique     registered composition (8-connected), unique-name objects only
  union_all|noconn|env      only objects of metadata type "env" (objects_labels.json; all env names are scene-unique), added
                            after the first post-hoc report showed static/dynamic additions at precision 0.01-0.15
  oracle|noconn             S_o = GT-best detection (target GT; bound)
  PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python -I posthoc.py --out posthoc_rows.json --procs 14
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np

HERE = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_text-concept"
sys.path.insert(0, HERE)
from compose_score import DATA, DETS, DEV, PROPS, S1, bound_pix, connected_add, disk, h, jf_pair, load_labelmap, rd  # noqa: E402

ARMS = ("union_all|noconn", "union_all|noconn|unique", "union_all|conn|unique", "oracle|noconn", "union_all|noconn|env")
TAU = 0.5


def view(job):
    scene, t, cam, ref_frame, uniq, env = job
    assert t != ref_frame
    pdir = os.path.join(PROPS, scene, f"{t:06d}", cam)
    ddir = os.path.join(DETS, scene, f"{t:06d}", cam)
    meta = json.load(open(os.path.join(ddir, "meta.json")))
    H, W = meta["hw"]
    hw = (H, W)
    objs = [int(o) for o in meta["objs"]]
    gt = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    cls = cv2.imread(os.path.join(pdir, "cls.png"), cv2.IMREAD_UNCHANGED)
    invis = cls > 0
    D = {o: rd(os.path.join(DEV, scene, f"{t:06d}", cam, f"{o}.png"), hw) for o in objs}
    cnt = np.zeros(hw, np.uint8)
    for o in objs:
        cnt += D[o]
    bp = bound_pix(H, W)
    se = disk(bp)
    G = {o: gt == o for o in objs}
    seeded = np.isin(gt, objs)
    gpx = {o: int(G[o].sum()) for o in objs}
    base = {o: jf_pair(G[o], D[o], bp, se) for o in objs if gpx[o]}
    z = np.load(os.path.join(ddir, "dets.npz"))
    packed, pidx, prob = z["packed"], z["prompt"], z["prob"]
    sid = {s: i for i, s in enumerate(meta["strings"])}
    oprom = {int(o): v for o, v in meta["obj_prompts"].items()}
    M = {}

    def m(k):
        if k not in M:
            M[k] = np.unpackbits(packed[k], count=H * W).reshape(H, W).astype(bool)
        return M[k]
    pool = {o: [k for k in range(len(prob)) if int(pidx[k]) in {sid[s] for s in oprom[o]["split"]} and prob[k] > TAU] for o in objs}
    out = dict(scene=scene, t=t, cam=cam, pairs={}, empty_gt={})
    for arm in ARMS:
        sel, conn = arm.split("|")[0], arm.split("|")[1] == "conn"
        only_u = arm.endswith("|unique")
        only_env = arm.endswith("|env")
        Apre = {}
        for o in objs:
            if only_u and not uniq[o]:
                continue
            if only_env and not env[o]:
                continue
            ks = pool[o]
            if sel == "oracle":
                if not gpx[o] or not ks:
                    continue
                iou = {k: (m(k) & G[o]).sum() / max(1, (m(k) | G[o]).sum()) for k in ks}
                kb = max(ks, key=lambda k: (iou[k], prob[k]))
                ks = [kb] if iou[kb] > 0 else []
            if not ks:
                continue
            s = np.zeros(hw, bool)
            for k in ks:
                s |= m(k)
            C = s & invis & ~D[o] & ~((cnt - D[o]) > 0)
            Apre[o] = connected_add(C, D[o]) if conn else C
        ca = np.zeros(hw, np.uint8)
        for o in Apre:
            ca += Apre[o]
        for o in objs:
            A = Apre[o] & (ca == 1) if o in Apre else np.zeros(hw, bool)
            na = int(A.sum())
            if not gpx[o]:
                if na:
                    out["empty_gt"].setdefault(arm, {})[o] = na
                continue
            r = out["pairs"].setdefault(o, {})
            if na == 0:
                r[arm] = [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                continue
            J, F = jf_pair(G[o], D[o] | A, bp, se)
            on_o = A & G[o]
            n_on = int(on_o.sum())
            n_bg = int((gt[A] == 0).sum())
            r[arm] = [float(J - base[o][0]), float(F - base[o][1]), na, n_on, na - n_on - n_bg, n_bg] + \
                [int((A & (cls == c)).sum()) for c in (1, 2, 3)] + [int((on_o & (cls == c)).sum()) for c in (1, 2, 3)] + \
                [int((A & seeded & ~G[o]).sum())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--scenes", default="all")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(DEV, "index.json")))
    jobs = []
    for s, S in sorted(idx["scenes"].items()):
        if a.scenes != "all" and s not in a.scenes.split(","):
            continue
        L = {int(k): v for k, v in json.load(open(os.path.join(S1, "Data", s, "objects_labels.json"))).items()}
        cnt = Counter(L[o]["label"] for o in S["basic"])
        uniq = {o: cnt[L[o]["label"]] == 1 for o in S["basic"]}
        env = {o: L[o].get("type") == "env" for o in S["basic"]}
        for t in S["frames"]:
            for cam in S["eval_cams"]:
                jobs.append((s, t, cam, S["ref_frame"], uniq, env))
    t0 = time.time()
    res = []
    with Pool(a.procs) as p:
        for i, r in enumerate(p.imap_unordered(view, jobs)):
            res.append(r)
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    json.dump(dict(arms=ARMS, views=res, seconds=time.time() - t0), open(a.out, "w"))
    print(f"done {len(res)} views {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

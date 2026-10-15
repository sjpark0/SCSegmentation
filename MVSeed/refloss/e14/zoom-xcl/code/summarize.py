#!/usr/bin/env python3
"""S1-E14 zoom-xcl: aggregate the per-frame config.json of the zx_A root (checks, cost, zoom / xcl counts) and the
adopted decode's seconds (_e13_dev) -- host, numpy only.
    stage1/.venv/bin/python summarize.py --dir stage1/geomem/runs/_e14_zoom-xcl --out <json>
"""
import argparse
import json
import os
from collections import Counter

import numpy as np


def frames(root):
    for s in sorted(os.listdir(root)):
        sd = os.path.join(root, s)
        if not os.path.isdir(sd):
            continue
        for t in sorted(os.listdir(sd)):
            p = os.path.join(sd, t, "config.json")
            if os.path.isfile(p):
                yield s, int(t), json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = a.ref or os.path.join(os.path.dirname(os.path.abspath(a.dir)), "_e13_dev")
    zx = os.path.join(a.dir, "zx_A")
    chk, zc, xc = Counter(), Counter(), {arm: Counter() for arm in ("x", "zx", "zxb")}
    sec, zsec, post, peak, n_fr, per_scene = 0.0, 0.0, 0.0, 0.0, 0, {}
    r1conf_bad, done_missing = [], []
    for s, t, c in frames(zx):
        n_fr += 1
        if not os.path.isfile(os.path.join(zx, s, f"{t:06d}", "E14DONE")):
            done_missing.append(f"{s}/{t}")
        for k, v in c["checks"].items():
            if k == "round1_conf_equal":
                if not v:
                    r1conf_bad.append(f"{s}/{t}")
            else:
                chk[k] += int(v)
        zc.update({k: v for k, v in c["zoom_counts_eval"].items()})
        for arm in xc:
            for cam, d in c["xcl_counts"][arm].items():
                xc[arm].update(d)
        sec += c["seconds"]
        zsec += c["zoom_counts_eval"].get("seconds", 0.0)
        post += c["post_seconds"]
        peak = max(peak, c["peak_gpu_gb"])
        ps = per_scene.setdefault(s, dict(seconds=0.0, zoom_seconds=0.0, crops=0, accepted=0))
        ps["seconds"] += c["seconds"]
        ps["zoom_seconds"] += c["zoom_counts_eval"].get("seconds", 0.0)
        ps["crops"] += int(c["zoom_counts_eval"].get("run", 0))
        ps["accepted"] += int(c["zoom_counts_eval"].get("accepted", 0))
    ref_sec = 0.0
    ref_ps = {}
    for s, t, c in frames(ref):
        ref_sec += c["seconds"]
        ref_ps[s] = ref_ps.get(s, 0.0) + c["seconds"]
    for s in per_scene:
        per_scene[s]["adopted_seconds"] = ref_ps.get(s)
    res = dict(frames=n_fr, done_missing=done_missing, checks=dict(chk), round1_conf_mismatch=r1conf_bad,
               zoom_counts=dict(zc), xcl_counts={k: dict(v) for k, v in xc.items()},
               decode_seconds=sec, zoom_seconds=zsec, post_seconds=post, decode_minus_zoom_post=sec - zsec - post,
               adopted_decode_seconds_e13=ref_sec, peak_gpu_gb_max=peak,
               zoom_seconds_per_crop=zsec / max(1, zc.get("run", 0)), per_scene=per_scene)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_scene"}, indent=1))


if __name__ == "__main__":
    main()

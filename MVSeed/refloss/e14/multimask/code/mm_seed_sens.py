#!/usr/bin/env python3
"""Sensitivity (not part of the pre-registered rule): the paired scene bootstrap CI of the qualifying arms under seeds 0..9."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, S1)
from mvseg2d.paired import paired_scene_bootstrap  # noqa: E402
OUT = os.path.join(S1, "geomem", "runs", "_e14_multimask")
rows = [r for p in sorted(glob.glob(os.path.join(OUT, "parts", "*.json"))) for r in json.load(open(p))["rows"]]
res = {}
for arm in ("I3", "I4"):
    for sub in ("all", "good"):
        rs = [r for r in rows if sub == "all" or (r["base_J"] + r["base_F"]) / 2 >= 0.8]
        b = [dict(scene=r["scene"], view=r["view"], JF=(r["base_J"] + r["base_F"]) / 2) for r in rs]
        c = [dict(scene=r["scene"], view=r["view"], JF=sum(r["arms"][arm]) / 2) for r in rs]
        cis = [paired_scene_bootstrap(b, c, key="JF", n=10_000, seed=s, view_key="view")["ci"] for s in range(10)]
        res[f"{arm}/{sub}"] = dict(lo=[float(x[0]) for x in cis], hi=[float(x[1]) for x in cis])
        print(arm, sub, "CI lo over seeds 0..9:", " ".join(f"{x[0]:+.5f}" for x in cis), "| lo>0 in", sum(x[0] > 0 for x in cis), "/10")
json.dump(res, open(os.path.join(OUT, "seed_sens.json"), "w"), indent=1)

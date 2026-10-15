#!/usr/bin/env python3
"""Extra DEV tables from rows.json: per variant x config, UC140 delta without the one object that carries the gain
(CBABasketball v06 obj 18), object-cluster bootstrap; and the sensitivity ranges per variant over the registered grid."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/sam3-prompt-extend")
from report import boot  # noqa: E402

DEV = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_dev"
R = json.load(open("rows.json"))
S = json.load(open("summary.json"))["summary"]
base = {(r["scene"], r["t"], r["cam"], r["obj"]): r for r in json.load(open(f"{DEV}/dev_pairs.json"))["rows"]}
uc = [(r["scene"], r["t"], r["cam"], r["obj"]) for r in json.load(open(f"{DEV}/dev_undercover.json"))["rows"]]
BB = ("CBABasketball", "v06", 18)
d = defaultdict(dict)
for v in R["views"]:
    for o in v["base"]:
        k = (v["scene"], v["t"], v["cam"], int(o))
        for key, x in v["pairs"].get(o, {}).items():
            if not key.startswith("funnel"):
                d[key][k] = (x[0] + x[1]) / 2
for var in R["variants"]:
    print(f"== {var}")
    for mp, g in R["configs"]:
        key = f"{var}|{mp}|{g}"
        sel = [k for k in uc if (k[0], k[2], k[3]) != BB]
        vals = [d[key].get(k, 0.0) for k in sel]
        lo, hi = boot(vals, [(k[0], k[2], k[3]) for k in sel])
        a = S[key]["all"]
        print(f"   {key:22s} all {a['dJF']:+.5f} [{a['ci'][0]:+.5f},{a['ci'][1]:+.5f}]  UC140 {S[key]['uc140']['dJF']:+.5f}  "
              f"UC135 w/o CBA v06 billboard {np.mean(vals):+.5f} [{lo:+.5f},{hi:+.5f}]  clean67 {S[key]['clean67']['dJF']:+.5f}  "
              f"added prec {a['prec_pooled']:.3f}  UC prec {S[key]['uc140']['prec_pooled']:.3f}")
    ks = [f"{var}|{mp}|{g}" for mp, g in R["configs"]]
    rng = lambda f: (min(f(S[k]) for k in ks), max(f(S[k]) for k in ks))  # noqa: E731
    print("   RANGE over the 5 registered configs: all dJF [%+.5f, %+.5f]; UC140 dJF [%+.5f, %+.5f]; added precision [%.3f, %.3f]" % (
        *rng(lambda s: s["all"]["dJF"]), *rng(lambda s: s["uc140"]["dJF"]), *rng(lambda s: s["all"]["prec_pooled"])))

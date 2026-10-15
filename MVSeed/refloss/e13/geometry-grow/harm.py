# Where the primary's added pixels go (DEV only): by (scene, object label, type); and split of wrong pixels
import glob, json, sys, os
from collections import defaultdict
import numpy as np
run = sys.argv[1]; var = sys.argv[2] if len(sys.argv) > 2 else "mv"
DATA = "/home/sjpark/Documents/SCSegmentation/stage1/Data"
dp = json.load(open("/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_dev/dev_pairs.json"))
base = {(r["scene"], r["t"], r["cam"], r["obj"]): r for r in dp["rows"]}
agg = defaultdict(lambda: np.zeros(5))
tot = np.zeros(5)
for f in glob.glob(run + "/frames/*.json"):
    F = json.load(open(f))
    lab = json.load(open(os.path.join(DATA, F["scene"], "objects_labels.json")))
    for v in F["views"]:
        for o, p in v["pairs"].items():
            r = p.get(var)
            if not r or r[2] == 0: continue
            k = (v["scene"], v["t"], v["cam"], int(o))
            dJF = (r[0] + r[1]) / 2
            row = np.array([r[2], r[3], r[12], r[4] - r[12], dJF])
            g = (v["scene"], v["cam"], int(o), lab[o]["label"], lab[o]["type"])
            agg[g] += row; tot += row
print(f"[{var}] added {tot[0]:,.0f}: on o {tot[1]/tot[0]:.3f}, on other SEEDED object {tot[2]/tot[0]:.3f}, on other NOT-seeded object (absent from c_ini's input at t, or outside the basic set) {tot[3]/tot[0]:.3f}")
print("top 20 (scene, cam, obj) by wrong added px: added, prec, wrong->seeded, wrong->unseeded, sum dJF over its pairs")
for g, a in sorted(agg.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:20]:
    print(f"   {g[0]:18s} {g[1]:12s} {g[2]:3d} {g[3]:26s} {g[4]:8s} added {a[0]:10,.0f} prec {a[1]/a[0]:.3f} seeded {a[2]/a[0]:.3f} unseeded {a[3]/a[0]:.3f} sum dJF {a[4]:+.3f}")
w = sorted(agg.values(), key=lambda a: -(a[0] - a[1]))
wr = np.array([a[0] - a[1] for a in w])
print(f"wrong px concentration: top 10 objects {wr[:10].sum()/wr.sum():.3f}, top 30 {wr[:30].sum()/wr.sum():.3f} of {len(wr)} (scene,cam,obj) with additions")

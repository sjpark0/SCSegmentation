import json, numpy as np
from collections import defaultdict
T = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_text-concept"
DEV = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs/_e13_dev"
rows = json.load(open(T + "/posthoc_rows.json"))
uc = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in json.load(open(DEV + "/dev_undercover.json"))["rows"]} if "rows" in json.load(open(DEV + "/dev_undercover.json")) else None
dp = json.load(open(DEV + "/dev_pairs.json"))["rows"]
allk = [(r["scene"], r["t"], r["cam"], r["obj"]) for r in dp]
arm = "union_all|noconn|env"
d = {}
for v in rows["views"]:
    for o, rec in v["pairs"].items():
        x = rec.get(arm)
        d[(v["scene"], v["t"], v["cam"], int(o))] = (x[0] + x[1]) / 2 if x else 0.0
assert set(d) == set(allk), (len(d), len(allk))
vals = np.array([d[k] for k in allk])
print("all", vals.mean(), "sum", vals.sum())
big = defaultdict(float)
for k in allk: big[(k[0], k[2], k[3])] += d[k]
top = sorted(big.items(), key=lambda kv: -abs(kv[1]))[:8]
for g, s in top: print("  group", g, "sum %+.3f share %.2f" % (s, s / vals.sum()))
excl = {("Dog", "camera_0004", 3), ("FlameSteak", "cam01", 32)}
v2 = np.array([d[k] for k in allk if (k[0], k[2], k[3]) not in excl])
print("all w/o Dog tree & FlameSteak cam01 wall: n", len(v2), "mean %+.5f" % v2.mean())
v3 = np.array([d[k] for k in allk if k[0] != "Dog"])
print("all w/o Dog scene: mean %+.5f" % v3.mean())
# scene bootstrap of all-pairs, and a leave-one-scene-out min
by = defaultdict(list)
for k in allk: by[k[0]].append(d[k])
sc = sorted(by); sums = np.array([np.sum(by[s]) for s in sc]); cnts = np.array([len(by[s]) for s in sc])
loo = {s: (sums.sum() - sums[i]) / (cnts.sum() - cnts[i]) for i, s in enumerate(sc)}
print("leave-one-scene-out min", min(loo.items(), key=lambda kv: kv[1]))
rng = np.random.default_rng(1)
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(sc), (10000, len(sc)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    print("seed", seed, "CI", np.percentile(m, 2.5), np.percentile(m, 97.5), "P(m<=0)", (m <= 0).mean())

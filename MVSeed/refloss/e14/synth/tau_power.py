"""Synthesis checks for S1-E14 (DEV-A only): tau grid from tu_variants parts, robustness and pseudo-keyframe power."""
import json, glob, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/home/sjpark/Documents/SCSegmentation/stage1")
from mvseg2d.paired import paired_scene_bootstrap, scene_means

R = "/home/sjpark/Documents/SCSegmentation/stage1/geomem/runs"
base = json.load(open(f"{R}/_e13_dev/dev_pairs.json"))["rows"]
for r in base:
    r["view"] = f"{r['t']}/{r['cam']}"
key = lambda r: (r["scene"], int(r["t"]), r["cam"], int(r["obj"]))
kb = {key(r): r for r in base}
taus = ["0", "0.25", "0.5", "1", "1.5"]
arms = {t: {} for t in taus}
for p in glob.glob(f"{R}/_e14_threshold-upsample/parts/*.json"):
    v = json.load(open(p))
    for t in taus:
        res = v["res"][f"bil2|{t}"]
        for i, o in enumerate(v["objs"]):
            k = (v["scene"], int(v["t"]), v["cam"], int(o))
            if k in kb:
                arms[t][k] = (res["J"][i], res["F"][i])
mk = lambda t: [dict(kb[k], J=arms[t][k][0], F=arms[t][k][1], JF=0.5 * (arms[t][k][0] + arms[t][k][1])) for k in kb]
rows = {t: mk(t) for t in taus}
assert max(abs(a["JF"] - b["JF"]) for a, b in zip(rows["0"], base)) < 1e-12
good = {k for k in kb if kb[k]["JF"] >= 0.8}
out = {}
def boot(a, b, sub=None, m="JF", view_key="view", seed=0):
    if sub is not None:
        a = [r for r in a if key(r) in sub]; b = [r for r in b if key(r) in sub]
    return paired_scene_bootstrap(a, b, key=m, n=10_000, seed=seed, view_key=view_key)
for t in taus[1:]:
    A, B = base, rows[t]
    al, g, f = boot(A, B), boot(A, B, good), boot(A, B, set(kb) - good)
    ps = al["per_scene"]
    d = np.array([B[i]["JF"] - A[i]["JF"] for i in range(len(A))])
    noW = np.mean([v for s, v in ps.items() if s != "Welder"])
    # pseudo-keyframes: one DEV-A frame offset per scene (same structure as the keyframe: 1 frame x 2 cams), view = cam
    ts = defaultdict(list)
    for r in A:
        ts[r["scene"]].append(int(r["t"]))
    offs = []
    for j in range(5):
        sel = set()
        for s, tl in ts.items():
            tj = sorted(set(tl))[j]
            sel |= {key(r) for r in A if r["scene"] == s and int(r["t"]) == tj}
        bj = boot(A, B, sel, view_key="cam")
        offs.append(dict(j=j, delta=bj["delta"], ci=bj["ci"], worst=min(bj["per_scene"].values()),
                         g1=bool(bj["ci"][0] > 0), g2=bool(min(bj["per_scene"].values()) >= -0.01)))
    # halves of DEV-A frames (offsets {0,2,4} vs {1,3})
    halves = []
    for js in ([0, 2, 4], [1, 3]):
        sel = set()
        for s, tl in ts.items():
            tt = sorted(set(tl))
            sel |= {key(r) for r in A if r["scene"] == s and int(r["t"]) in {tt[j] for j in js}}
        bh = boot(A, B, sel)
        halves.append(dict(js=js, delta=bh["delta"], ci=bh["ci"], worst=min(bh["per_scene"].values())))
    out[t] = dict(all=dict(d=al["delta"], ci=al["ci"]), good=dict(d=g["delta"], ci=g["ci"]), fail=dict(d=f["delta"], ci=f["ci"]),
                  J=boot(A, B, m="J")["delta"], F=boot(A, B, m="F")["delta"],
                  scenes_down=sum(v < 0 for v in ps.values()), worst=min(ps.items(), key=lambda x: x[1]),
                  best=max(ps.items(), key=lambda x: x[1]), median_scene=float(np.median(list(ps.values()))),
                  mean_noWelder=float(noW), pair_mean=float(d.mean()), down01=int((d < -0.01).sum()), up01=int((d > 0.01).sum()),
                  down01_frac=float((d < -0.01).mean()),
                  good_scene_worst=min(g["per_scene"].items(), key=lambda x: x[1]),
                  F_scene_worst=min(boot(A, B, m="F")["per_scene"].items(), key=lambda x: x[1]),
                  pseudo_kf=offs, halves=halves)
    print(t, json.dumps({k: v for k, v in out[t].items() if k not in ("pseudo_kf", "halves")}))
    for o in offs: print("   pkf", o)
    for h in halves: print("   half", h)
# scene sd and view-level noise
json.dump(out, open(os.path.join(os.path.dirname(__file__), "tau_power.json"), "w"), indent=1)

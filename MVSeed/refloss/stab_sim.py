"""Host: read stab_H.json (stab.py) -> basic-score change of each GT-free stabilisation rule on the
non-reference cameras, per scene, plus a leave-one-scene-out pick per rule family (docs/stage2-D-nonref.md section 6)."""
import json, collections
R = json.load(open("stab_H.json"))
cams = collections.defaultdict(set); objs = collections.defaultdict(set)
for r in R: cams[r["scene"]].add(r["cam"]); objs[(r["scene"], r["cam"])].add(r["obj"])
W = lambda r: 100 / (17 * len(cams[r["scene"]]) * len(objs[(r["scene"], r["cam"])])) / r["n"]
non = [r for r in R if not r["ref"]]
print("nonref loss (base) %.3f, all cams %.3f" % (sum(W(r) * sum(1 - x for x in r["JF"]["base"]) for r in non),
                                                  sum(W(r) * sum(1 - x for x in r["JF"]["base"]) for r in R)))
rules = [k for k in non[0]["JF"] if k != "base"]
G = {}
for k in rules:
    g = collections.Counter({s: 0.0 for s in cams})
    for r in non: g[r["scene"]] += W(r) * sum(a - b for a, b in zip(r["JF"][k], r["JF"]["base"]))
    G[k] = g
print("rule              total    worst(x17)  scenes down")
for k in rules:
    v = G[k]; print(f"{k:16s} {sum(v.values()):+.4f}  {17 * min(v.values()):+.3f}  {sum(x < -1e-9 for x in v.values())}")
for fam in ("hold", "cons", "med", None):
    grid = [k for k in rules if fam is None or k.startswith(fam)]
    lo, picks = 0.0, collections.Counter()
    for s in cams:
        best = max(grid, key=lambda k: sum(v for sc, v in G[k].items() if sc != s)); picks[best] += 1; lo += G[best][s]
    print(f"LOSO {fam or 'all'}: {lo:+.4f}  picks {dict(picks)}")
# oracle per frame: best of base and any rule
orc = 0.0
for r in non:
    for t in range(r["n"]):
        orc += W(r) * max(0.0, max(r["JF"][k][t] for k in rules) - r["JF"]["base"][t])
print("per-frame oracle over all rules %+.4f" % orc)

import json, collections, statistics
A = {(r["scene"], r["cam"], r["obj"], r["t"]): r for r in json.load(open("anat2_H.json"))}
S = json.load(open("snap2_H.json"))
P = collections.defaultdict(list)
for r in A.values(): P[(r["scene"], r["cam"], r["obj"])].append(r)
cams = collections.defaultdict(set); objs = collections.defaultdict(set)
for (s, c, o) in P: cams[s].add(c); objs[(s, c)].add(o)
W = {k: 100 / (17 * len(cams[k[0]]) * len(objs[(k[0], k[1])])) / len(v) for k, v in P.items()}
def sim(rule, which):
    gain = collections.Counter({s: 0.0 for s in cams})
    for f in S:
        if which == "ref" and not f["ref"]: continue
        if which == "non" and f["ref"]: continue
        k = (f["scene"], f["cam"], f["obj"]); a = A[k + (f["t"],)]
        if rule(f): gain[f["scene"]] += W[k] * (0.5 * (f["Js"] + f["Fs"]) - 0.5 * (a["J"] + a["F"]))
    v = list(gain.values()); return sum(v), min(v), sum(1 for x in v if x < -1e-9), gain
for which in ("ref", "non", "all"):
    orc = sum(W[(f["scene"], f["cam"], f["obj"])] * max(0, 0.5*(f["Js"]+f["Fs"]) - 0.5*(A[(f["scene"], f["cam"], f["obj"], f["t"])]["J"]+A[(f["scene"], f["cam"], f["obj"], f["t"])]["F"])) for f in S if which=="all" or (f["ref"] == (which=="ref")))
    print(f"== {which}: per-frame oracle {orc:.3f}")
    for tau in (0.85, 0.9, 0.95):
        for dl in (3, 4, 6):
            t, worst, neg, _ = sim(lambda f: f["iou_p_g0"] >= tau and f["imgdiff"] <= dl, which)
            print(f"   tau {tau} d {dl}: {t:+.3f} worst {worst:+.3f} neg {neg}")
# leave-one-scene-out selection over grid on 'all' to estimate honest gain
grid = [(t, d) for t in (0.8, 0.85, 0.9, 0.93, 0.95, 0.97) for d in (2, 3, 4, 5, 6, 8, 10)]
G = {g: sim(lambda f, g=g: f["iou_p_g0"] >= g[0] and f["imgdiff"] <= g[1], "all")[3] for g in grid}
lo = 0
for s in cams:
    best = max(grid, key=lambda g: sum(v for k, v in G[g].items() if k != s))
    lo += G[best][s]
print(f"LOSO honest gain (all cams): {lo:+.3f}")
G = {g: sim(lambda f, g=g: f["iou_p_g0"] >= g[0] and f["imgdiff"] <= g[1], "ref")[3] for g in grid}
lo = 0; picks = []
for s in cams:
    best = max(grid, key=lambda g: sum(v for k, v in G[g].items() if k != s)); picks.append(best)
    lo += G[best][s]
print(f"LOSO honest gain (ref only): {lo:+.3f}  picks {collections.Counter(picks)}")

# the diagnosis section-5 table (contribution to the 17-scene mean; '*' = some scene's own Δ < -0.17)
TAUS, DELTAS = (0.6, 0.7, 0.8, 0.9), (2, 3, 4, 5, 6, 8)
print("tau\\d " + " ".join(f"{d:>8}" for d in DELTAS))
for tau in TAUS:
    print(f"{tau:5}  " + " ".join(f"{(r := sim(lambda f: f['iou_p_g0'] >= tau and f['imgdiff'] <= d, 'ref'))[0]:+.4f}{'*' if r[1] * 17 < -0.17 else ' '}" for d in DELTAS))
grid2 = [(t, d) for t in TAUS for d in DELTAS]
G2 = {g: sim(lambda f, g=g: f["iou_p_g0"] >= g[0] and f["imgdiff"] <= g[1], "ref")[3] for g in grid2}
lo2 = 0; picks2 = []
for s in cams:
    best = max(grid2, key=lambda g: sum(v for k, v in G2[g].items() if k != s)); picks2.append(best); lo2 += G2[best][s]
print(f"LOSO on the section-5 grid (ref only): {lo2:+.4f}  picks {collections.Counter(picks2)}")
t, worst, neg, g = sim(lambda f: f["iou_p_g0"] >= 0.8 and f["imgdiff"] <= 4, "ref")
print("registered (0.8, 4): %+.4f; per-scene Δ (x17):" % t, {k: round(17 * v, 3) for k, v in sorted(g.items())})

import json, statistics, collections
R = json.load(open("refmap.json"))["ref_objects"]
G = {(r["scene"], r["cam"], r["obj"]): r for r in json.load(open("/home/sjpark/Documents/SCSegmentation/MVSeed/runs/grid_ceiling.json"))["rows"] if r["is_ref"]}
tot = cl = 0; bysize = collections.defaultdict(lambda: [0, 0, 0]); excess = []
for r in R:
    k = (r["scene"], r["cam"], r["obj"]); g = G[k]
    c = r["w"] * (1 - g["JF_288"]) * 20 / 21
    c576 = r["w"] * (1 - g["JF_576"]) * 20 / 21
    tot += r["loss"]; cl += c
    b = bysize[g["size_bin"]]; b[0] += r["loss"]; b[1] += c; b[2] += 1
    excess.append((r["loss"] - c, k, r["loss"], c, g["size_bin"], g["gt_px_median"]))
print(f"ref loss {tot:.3f}; 288-grid ceiling share {cl:.3f} ({cl/tot:.0%}); trackable residual {tot-cl:.3f}")
for s, (l, c, n) in bysize.items(): print(f"  {s:12s} n={n:3d} loss {l:.3f} ceiling {c:.3f} residual {l-c:.3f}")
excess.sort(reverse=True)
ex = [e[0] for e in excess]; T = sum(x for x in ex if x > 0)
print(f"positive residual total {T:.3f}; negative (beats ceiling) {sum(x for x in ex if x<0):.3f}")
for q in (5, 10, 20, 40): print(f"  top {q}: {sum(ex[:q])/T:.0%}")
for e in excess[:30]: print(f"  {e[1][0]:20s} {e[1][1]:12s} o{e[1][2]:<3} resid {e[0]:.4f} loss {e[2]:.4f} ceil {e[3]:.4f} {e[4]} px {e[5]}")

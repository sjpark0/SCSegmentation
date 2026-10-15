#!/usr/bin/env python3
"""S1-E13 candidate E2: where the dev under-cover (UC) gain of the two-reference decode comes from (host numpy, DEV only).

    stage1/.venv/bin/python breakdown_2ref.py   (reads ../_e13_dev/dev_pairs.json, both/ r2/ dev_pairs.json, cover_B.json)
"""
import json
import os
from collections import defaultdict

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
KEY = ("scene", "t", "cam", "obj")


def load(p):
    return {tuple(r[k] for k in KEY): r for r in json.load(open(p))["rows"]}


base = load(os.path.join(H, "..", "_e13_dev", "dev_pairs.json"))
arms = {n: load(os.path.join(H, n, "dev_pairs.json")) for n in ("both", "r2")}
cov = load(os.path.join(H, "cover_B.json"))
UC = [k for k, r in base.items() if r["JF"] < 0.8 and r["gt_invis_frac"] >= 0.5 and r["recall"] < 0.85]
CLEAN = {k for k in UC if base[k]["precision"] is not None and base[k]["precision"] >= 0.85}
tot = {n: sum(a[k]["JF"] - base[k]["JF"] for k in UC) for n, a in arms.items()}
print(f"UC {len(UC)} pairs; sum dJ&F both {tot['both']:+.2f} r2 {tot['r2']:+.2f} (mean {tot['both'] / len(UC):+.4f} / {tot['r2'] / len(UC):+.4f})")

print("\n(1) UC by the adopted pair's seedcat kind: n  J&F base / both / r2   share of the UC sum dJ&F (both, r2)")
by = defaultdict(list)
for k in UC:
    by[base[k]["kind"]].append(k)
for kind, ks in sorted(by.items(), key=lambda kv: -len(kv[1])):
    s = {n: sum(a[k]["JF"] - base[k]["JF"] for k in ks) for n, a in arms.items()}
    print(f"   {kind:28s} {len(ks):3d}  {np.mean([base[k]['JF'] for k in ks]):.3f} / {np.mean([arms['both'][k]['JF'] for k in ks]):.3f} / "
          f"{np.mean([arms['r2'][k]['JF'] for k in ks]):.3f}   {s['both'] / tot['both']:.2f}, {s['r2'] / tot['r2']:.2f}")

print("\n(2) UC by whether B's GT holds the object at all (B_gt_px > 0) and by the oracle reach of the missed pixels")
for name, sel in (("object absent from B", [k for k in UC if cov[k]["B_gt_px"] == 0]),
                  ("object present in B", [k for k in UC if cov[k]["B_gt_px"] > 0]),
                  ("  present, missed on B GT >= 0.25", [k for k in UC if cov[k]["B_gt_px"] > 0 and cov[k]["missed_onBgt"] >= 0.25 * max(1, cov[k]["missed_px"])]),
                  ("  present, missed on B GT < 0.25", [k for k in UC if cov[k]["B_gt_px"] > 0 and cov[k]["missed_onBgt"] < 0.25 * max(1, cov[k]["missed_px"])])):
    if not sel:
        continue
    print(f"   {name:36s} n={len(sel):3d}  J&F base {np.mean([base[k]['JF'] for k in sel]):.3f}  both {np.mean([arms['both'][k]['JF'] for k in sel]):.3f}"
          f"  r2 {np.mean([arms['r2'][k]['JF'] for k in sel]):.3f}  recall base {np.mean([base[k]['recall'] for k in sel]):.3f} both "
          f"{np.mean([arms['both'][k]['recall'] for k in sel]):.3f}")

print("\n(3) UC object groups (scene, cam, obj): share of the UC sum dJ&F (both), largest first; and the count of groups by outcome")
grp = defaultdict(list)
for k in UC:
    grp[(k[0], k[2], k[3])].append(k)
contrib = sorted(((sum(arms["both"][k]["JF"] - base[k]["JF"] for k in ks), g, ks) for g, ks in grp.items()), reverse=True)
for s, g, ks in contrib[:8]:
    print(f"   {g[0]:16s} {g[1]:12s} {g[2]:3d} {base[ks[0]]['label']:24s} {len(ks)} pairs  sum dJ&F {s:+.2f} = {s / tot['both']:.2f} of UC  "
          f"kinds {sorted({base[k]['kind'] for k in ks})}")
for n, a in arms.items():
    d = [np.mean([a[k]["JF"] - base[k]["JF"] for k in ks]) for ks in grp.values()]
    print(f"   {n}: {len(d)} groups; mean dJ&F > +0.05: {sum(x > 0.05 for x in d)}, |d| <= 0.05: {sum(abs(x) <= 0.05 for x in d)}, "
          f"< -0.05: {sum(x < -0.05 for x in d)}")
top = [c for c in contrib if c[0] > 0][:4]
rest = [k for k in UC if (k[0], k[2], k[3]) not in {c[1] for c in top}]
print(f"   UC without the 4 largest-gain groups ({', '.join(f'{c[1][0]} {c[1][1]} {base[c[2][0]]['label']}' for c in top)}): "
      f"n={len(rest)}  dJ&F both {np.mean([arms['both'][k]['JF'] - base[k]['JF'] for k in rest]):+.4f}  "
      f"r2 {np.mean([arms['r2'][k]['JF'] - base[k]['JF'] for k in rest]):+.4f}")

print("\n(4) CLEAN UC by c_ini class of the missed pixels: which dominant class recovers")
for c in ("outside", "occluded", "nodepth", "vis"):
    ks = [k for k in CLEAN if max(("vis", "outside", "occluded", "nodepth"), key=lambda q: base[k][f"missed_{q}"]) == c]
    if not ks:
        continue
    M = sum(base[k]["missed_px"] for k in ks)
    print(f"   dominant {c:9s} n={len(ks):3d}  dJ&F both {np.mean([arms['both'][k]['JF'] - base[k]['JF'] for k in ks]):+.4f}  "
          f"r2 {np.mean([arms['r2'][k]['JF'] - base[k]['JF'] for k in ks]):+.4f}  missed px {M:,} -> both "
          f"{sum(arms['both'][k]['missed_px'] for k in ks):,}  oracle onBgt {sum(cov[k]['missed_onBgt'] for k in ks) / max(1, M):.3f}")

print("\n(5) CLEAN UC without the Dog camera_0004 tree")
ks = [k for k in CLEAN if (k[0], k[2], k[3]) != ("Dog", "camera_0004", 3)]
Mb = sum(base[k]["missed_px"] for k in ks)
for n, a in arms.items():
    d = [a[k]["JF"] - base[k]["JF"] for k in ks]
    Ma = sum(a[k]["missed_px"] for k in ks)
    print(f"   {n}: n={len(ks)}  J&F {np.mean([base[k]['JF'] for k in ks]):.4f} -> {np.mean([a[k]['JF'] for k in ks]):.4f}  d {np.mean(d):+.4f}"
          f"  recall {np.mean([base[k]['recall'] for k in ks]):.3f} -> {np.mean([a[k]['recall'] for k in ks]):.3f}"
          f"  missed px {Mb:,} -> {Ma:,} (net {1 - Ma / Mb:.3f})  groups {len({(k[0], k[2], k[3]) for k in ks})}")

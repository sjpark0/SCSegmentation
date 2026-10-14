#!/usr/bin/env python
"""Aggregate host_anatomy.py's JSON with MVSeed/runs/seedfail.json (failure kinds, stage-2 cost per pair).

    stage1/.venv/bin/python aggregate.py anatomy.json
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = "/home/sjpark/Documents/SCSegmentation"
res = json.load(open(sys.argv[1]))
sf = {(r["scene"], r["cam"], r["obj"]): r for r in json.load(open(f"{REPO}/MVSeed/runs/seedfail.json"))["rows"]}
pairs = [p for r in res for p in r["pairs"]]
key = lambda p: (p["scene"], p["cam"], p["obj"])
print("pairs", len(pairs), "seedfail rows", len(sf), "joined", sum(key(p) in sf for p in pairs))
for p in pairs:
    s = sf[key(p)]
    p["kind"] = s["kind"] if s["JF"] < 0.8 else "good"
    p["cost"] = s["cost"]
    p["JF_sf"] = s["JF"]
    p["vis_ratio"] = s["vis_ratio"]
print("max |JF - seedfail JF| =", max(abs(p["JF"] - p["JF_sf"]) for p in pairs))

def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None

def scene_mean_JF(get):
    """mvseg2d.paired.scene_means: scene -> mean over cams of mean over pairs; then the mean over scenes."""
    by = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        by[p["scene"]][p["cam"]].append(get(p))
    return float(np.mean([np.mean([np.mean(v) for v in cams.values()]) for cams in by.values()]))

base_scene = scene_mean_JF(lambda p: p["JF"])
print(f"\nC0 keyframe J&F: pair mean {mean(p['JF'] for p in pairs):.4f}, scene mean {base_scene:.4f}")

# ---------------------------------------------------------------- (5) gate + overlap replica
oc = [x for r in res for x in r["overlap_check"]]
d = np.array([a - b for a, b in oc])
print(f"\n[5] overlap replica (numpy work-grid) vs round1_conf.json: n={len(oc)} mean|d|={np.abs(d).mean():.4f} "
      f"p95|d|={np.percentile(np.abs(d), 95):.4f} max|d|={np.abs(d).max():.4f}; decision (>=0.5) agrees "
      f"{np.mean([(a >= 0.5) == (b >= 0.5) for a, b in oc]):.4f}")
gate = [g for r in res for g in r["gate"]]
print(f"gate rows (non-ref cam x obj) {len(gate)}: present {sum(g['presence'] > 0 for g in gate)}, overlap None (n_valid<64) "
      f"{sum(g['overlap'] is None for g in gate)}, conf True {sum(g['conf'] for g in gate)} ({np.mean([g['conf'] for g in gate]):.3f})")
pres = [g for g in gate if g["presence"] > 0]
print(f"  among present: overlap None {sum(g['overlap'] is None for g in pres)} (of which iou>=0.7 {sum(g['overlap'] is None and g['iou'] >= 0.7 for g in pres)}), "
      f"overlap<0.5 {sum(g['overlap'] is not None and g['overlap'] < 0.5 for g in pres)}, conf {sum(g['conf'] for g in pres)}")
vf = [g["valid_frac_r1"] for g in gate if g["valid_frac_r1"] is not None]
print(f"  round-1 mask pixels that reproject validly into the reference (valid_frac): mean {np.mean(vf):.3f}, median {np.median(vf):.3f}, "
      f"<0.5 in {np.mean(np.array(vf) < 0.5):.3f} of masks")
kinds = ["under-cover(beyond c_ini)", "under-cover", "wrong-object", "boundary/mixed", "spill-onto-other", "spill-onto-other(twin)", "miss", "good"]
print("\n  own round-1 mask of the scored pair, by kind: n | conf True | overlap None | mean overlap | mean valid_frac | mean n_valid")
for k in kinds:
    ps = [p for p in pairs if p["kind"] == k]
    if not ps: continue
    gk = {(g["scene"], g["cam"], g["obj"]): g for g in gate}
    gs = [gk[key(p)] for p in ps]
    print(f"  {k:28s} {len(ps):3d} | {sum(g['conf'] for g in gs):3d} | {sum(g['overlap'] is None for g in gs):3d} | "
          f"{mean(g['overlap'] for g in gs if g['overlap'] is not None)!s:8.8s} | {mean(g['valid_frac_r1'] for g in gs)!s:8.8s} | {mean(g['n_valid'] for g in gs)!s:10.10s}")

# ---------------------------------------------------------------- (2) extent
print("\n[2] reference-visible vs -invisible part of the GT (vis = pixel's Pi3 point reprojects validly into the reference)")
print("  kind | n | gt_inv_frac | recall_vis r1->rep | recall_inv r1->rep | recall r1->rep | cost")
for k in kinds:
    ps = [p for p in pairs if p["kind"] == k]
    if not ps: continue
    print(f"  {k:28s} {len(ps):3d} | {mean(p['gt_inv_frac'] for p in ps):.3f} | {mean(p['recall_vis_r1'] for p in ps):.3f}->{mean(p['recall_vis_rep'] for p in ps):.3f} | "
          f"{mean(p['recall_inv_r1'] for p in ps):.3f}->{mean(p['recall_inv_rep'] for p in ps):.3f} | {mean(p['recall_r1'] for p in ps):.3f}->{mean(p['recall_rep'] for p in ps):.3f} | {sum(p['cost'] for p in ps):.3f}")
inv = [p for p in pairs if p["gt_inv_px"] >= 0.2 * p["gt_px"]]
print(f"  pairs with >=20% of GT reference-invisible: {len(inv)}, cost {sum(p['cost'] for p in inv):.3f}; their recall_inv r1 {mean(p['recall_inv_r1'] for p in inv):.3f} -> rep {mean(p['recall_inv_rep'] for p in inv):.3f}, "
      f"recall_vis {mean(p['recall_vis_r1'] for p in inv):.3f} -> {mean(p['recall_vis_rep'] for p in inv):.3f}")
up_ = sum(1 for p in inv if p["recall_inv_rep"] is not None and p["recall_inv_r1"] is not None and p["recall_inv_rep"] - p["recall_inv_r1"] > 0.05)
dn_ = sum(1 for p in inv if p["recall_inv_rep"] is not None and p["recall_inv_r1"] is not None and p["recall_inv_rep"] - p["recall_inv_r1"] < -0.05)
print(f"  of those, round 2 raised recall_inv by >0.05 in {up_}, lowered in {dn_}")
bc = [p for p in pairs if p["kind"] == "under-cover(beyond c_ini)"]
print("  the 9 beyond-c_ini pairs:")
for p in sorted(bc, key=lambda p: -p["cost"]):
    print(f"    {p['scene']:20s} {p['cam']:12s} obj {p['obj']:3d} gt {p['gt_px']:7d} inv_frac {p['gt_inv_frac']:.2f} recall_vis {p['recall_vis_r1']!s:5.5s}->{p['recall_vis_rep']!s:5.5s} "
          f"recall_inv {p['recall_inv_r1']!s:5.5s}->{p['recall_inv_rep']!s:5.5s} own_conf {p['own_conf']} own_overlap {p['own_overlap']!s:5.5s} cost {p['cost']:.3f}")

# ---------------------------------------------------------------- (3) overlaps + exclusivity variants
co = [c for r in res for c in r["cam_overlap"]]
print(f"\n[3] round-2 masks: non-ref views {len(co)}, views with any contested pixel {sum(c['contested_px'] > 0 for c in co)}, "
      f"contested px / union px = {sum(c['contested_px'] for c in co) / sum(c['union_px'] for c in co):.4f}, "
      f"(view,obj) masks touching a contested region {sum(c['n_obj_with_overlap'] for c in co)} of {sum(c['n_obj_nonempty'] for c in co)} non-empty")
io = [g["inside_other"] for g in gate if g["area_rep"] > 0]
print(f"  inside_other (fraction of a mask lying under another object's mask), all non-empty masks: >0 {np.mean(np.array(io) > 0):.3f}, >=0.5 {np.sum(np.array(io) >= 0.5)}, >=0.7 {np.sum(np.array(io) >= 0.7)}, >=0.9 {np.sum(np.array(io) >= 0.9)} of {len(io)}")
print("  scored pairs by kind: n | with contested px | mean inside_other | sum cost")
for k in kinds:
    ps = [p for p in pairs if p["kind"] == k]
    if not ps: continue
    print(f"  {k:28s} {len(ps):3d} | {sum(p['contested_px_in_mask'] > 0 for p in ps):3d} | {mean(p['inside_other'] for p in ps):.3f} | {sum(p['cost'] for p in ps):.3f}")
names = ["oracle_assign", "smaller_wins", "larger_wins", "suppress_inside_0.5", "suppress_inside_0.7", "suppress_inside_0.9"]
print("  variant | changed pairs | up(>0.01)/down(<-0.01) | sum dJF | pair-mean dJF | scene-mean dJF | dJF on failing 59 | on good 605")
for nme in names:
    def jf_v(p):
        v = p["variants"].get(nme)
        return v["JF"] if v else p["JF"]
    dj = {key(p): jf_v(p) - p["JF"] for p in pairs}
    ch = [p for p in pairs if p["variants"].get(nme)]
    print(f"  {nme:20s} {len(ch):3d} | {sum(d > 0.01 for d in dj.values()):3d}/{sum(d < -0.01 for d in dj.values()):3d} | {sum(dj.values()):+.3f} | {np.mean(list(dj.values())):+.5f} | "
          f"{scene_mean_JF(jf_v) - base_scene:+.5f} | {mean(dj[key(p)] for p in pairs if p['kind'] != 'good'):+.5f} | {mean(dj[key(p)] for p in pairs if p['kind'] == 'good'):+.5f}")
    worst = sorted(((dj[key(p)], key(p)) for p in pairs), key=lambda x: x[0])[:3]
    best = sorted(((dj[key(p)], key(p)) for p in pairs), key=lambda x: -x[0])[:3]
    print(f"      worst {[(round(a, 3), b) for a, b in worst]}  best {[(round(a, 3), b) for a, b in best]}")

# ---------------------------------------------------------------- (4) neighbour selection
nb = [n for r in res for n in r["neigh"]]
nb_pool = [n for n in nb if n["pool"] > 0]
print(f"\n[4] target x object (all non-ref targets): {len(nb)}; candidates after dropping the reference: 2 in {sum(n['n_cand'] == 2 for n in nb)}, 3 in {sum(n['n_cand'] == 3 for n in nb)}; "
      f"n_used 0/1/2/3: {Counter(n['n_used'] for n in nb)}; confident pool non-empty {len(nb_pool)}")
print(f"  largest round-1 mask (area) among confident cams is among the used neighbours in {np.mean([n['best_area_in_used'] for n in nb_pool]):.3f}; "
      f"its rank by camera distance: median {np.median([n['best_area_rank'] for n in nb_pool]):.0f}, mean {np.mean([n['best_area_rank'] for n in nb_pool]):.1f}; "
      f"used max area / best area: {np.mean([n['used_max_area'] / n['best_area'] for n in nb_pool if n['best_area'] > 0]):.3f}")
print(f"  largest reference-INVISIBLE round-1 area among confident cams is among the used in {np.mean([n['best_inv_in_used'] for n in nb_pool]):.3f}; "
      f"used max inv / best inv: {np.mean([n['used_max_inv'] / n['best_inv'] for n in nb_pool if n['best_inv'] > 0]):.3f}; pairs with best_inv>0: {sum(n['best_inv'] > 0 for n in nb_pool)}")
nbk = {(n["scene"], n["cam"], n["obj"]): n for n in nb}
print("  scored pairs by kind: n | n_used mean | best_area_in_used | used_max_area/best | used_max_inv/best_inv | own_inv_r1/own_area")
for k in kinds:
    ps = [p for p in pairs if p["kind"] == k]
    if not ps: continue
    ns = [nbk[key(p)] for p in ps]
    nsp = [n for n in ns if n["pool"] > 0]
    print(f"  {k:28s} {len(ps):3d} | {mean(n['n_used'] for n in ns):.2f} | {mean(n['best_area_in_used'] for n in nsp)!s:5.5s} | "
          f"{mean(n['used_max_area'] / n['best_area'] for n in nsp if n['best_area'] > 0)!s:5.5s} | {mean(n['used_max_inv'] / n['best_inv'] for n in nsp if n['best_inv'] > 0)!s:5.5s} | "
          f"{mean(n['own_inv_r1'] / n['own_area_r1'] for n in ns if n.get('own_area_r1', 0) > 0)!s:5.5s}")
print("  the 9 beyond-c_ini pairs: used neighbours' max round-1 area vs best confident, and invisible area")
for p in sorted(bc, key=lambda p: -p["cost"]):
    n = nbk[key(p)]
    print(f"    {p['scene']:20s} {p['cam']:12s} obj {p['obj']:3d} n_used {n['n_used']} used_max_area {n.get('used_max_area')} best_area {n.get('best_area')} (rank {n.get('best_area_rank')}) "
          f"used_max_inv {n.get('used_max_inv')} best_inv {n.get('best_inv')} (rank {n.get('best_inv_rank')}) own r1 area {n['own_area_r1']} inv {n['own_inv_r1']} gt {p['gt_px']}")

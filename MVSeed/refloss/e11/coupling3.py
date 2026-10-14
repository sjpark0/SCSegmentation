"""Third pass.  (1) For the 22 under-cover pairs: is the missed part covered by the round-1 mask of ANY stage-1
camera (ceiling of a neighbour re-selection, decode-only) or by the frame-0 GT of another annotated camera warped
into the target (does any camera we can check see it at all)?  (2) Over all 664 pairs: in-view seed overlap between
basic objects (objects are decoded as independent memories) and the keyframe J&F oracle of giving every overlapped
pixel to its GT owner when that owner is among the claimants (ceiling of an in-view exclusivity rule).  Host, read-only.

    stage1/.venv/bin/python coupling3.py <dir with coupling.json>
"""
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling2 import warp_mask, read_gt, read_mask, scene_cfg, frac, OBJ, S1, SEED, R1, REPO   # noqa: E402
sys.path.insert(0, os.path.join(REPO, "eval"))
import eval_jf as E   # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "."
C = json.load(open(os.path.join(D, "coupling.json")))
SF = json.load(open(os.path.join(REPO, "MVSeed/runs/seedfail.json")))["rows"]


def jf_pair(g, p):
    h, w = g.shape
    bp = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w))))
    se = E._disk(bp)
    box = E.crop_box(g, p, bp + 2, g.shape)
    if box is None:
        return 1.0
    y0, y1, x0, x1 = box
    return 0.5 * (E.db_eval_iou(g[y0:y1, x0:x1], p[y0:y1, x0:x1]) + E.db_eval_boundary(p[y0:y1, x0:x1], g[y0:y1, x0:x1], bp, se))


# ---------------------------------------------------------------- (1) under-cover: any camera?
print("(1) under-cover pairs: missed part covered by round-1 masks of any stage-1 camera / by warped GT of other annotated cams")
out1 = []
for p in C["pairs"]:
    if p["group"] != "under":
        continue
    sc, cam, o = p["scene"], p["cam"], p["obj"]
    folder, st = scene_cfg(sc)
    meta = json.load(open(os.path.join(S1, sc, "meta.json")))
    GT = read_gt(folder, cam, st)
    hw = GT.shape
    g = GT == o
    s = read_mask(sc, SEED, cam, o, hw)
    missed = g & ~s
    best, union = (None, 0.0), np.zeros(hw, bool)
    per = {}
    for c in meta["cams"]:
        if c == cam:
            continue
        m1 = read_mask(sc, R1, c, o, hw)
        if not m1.any():
            per[c] = 0.0
            continue
        wm, _ = warp_mask(sc, c, cam, hw, m1)
        f = frac(missed, wm) or 0.0
        per[c] = f
        union |= wm
        if f > best[1]:
            best = (c, f)
    gt_cov = {}
    for c in OBJ[sc]["cameras"]:
        if c == cam:
            continue
        gc = read_gt(folder, c, st) == o
        if not gc.any():
            gt_cov[c] = 0.0
            continue
        wm, _ = warp_mask(sc, c, cam, hw, gc)
        gt_cov[c] = frac(missed, wm) or 0.0
    row = dict(scene=sc, cam=cam, obj=o, kind=p["kind"], cost=p["cost"], missed_px=int(missed.sum()),
               best_r1_cam=best[0], best_r1_cov=best[1], union_r1_cov=frac(missed, union) or 0.0,
               n_cams=len(meta["cams"]) - 1, gt_cov_annotated=gt_cov, nearest3=[n["cam"] for n in p["neighbours"]])
    out1.append(row)
    print(f"   {sc[:12]:12s} {cam:12s} {o:3d} {p['kind'][:22]:22s} cost {p['cost']:+.4f} missed {row['missed_px']:7d}: best single cam r1 {best[0]} {100 * best[1]:4.0f}%  "
          f"union of all {row['n_cams']} cams r1 {100 * row['union_r1_cov']:4.0f}%  warped GT of annotated cams: "
          + ", ".join(f"{c} {100 * v:3.0f}%" for c, v in gt_cov.items()))
tot = sum(r["cost"] for r in out1)
for k in ("best_r1_cov", "union_r1_cov"):
    print(f"   SUM {k:14s} cost-weighted mean {sum(r['cost'] * r[k] for r in out1) / tot:.3f}; pairs >= 50%: {sum(r[k] >= 0.5 for r in out1)}/{len(out1)} "
          f"cost {sum(r['cost'] for r in out1 if r[k] >= 0.5):+.4f}")
mx = [max(r["gt_cov_annotated"].values()) if r["gt_cov_annotated"] else 0.0 for r in out1]
print(f"   SUM warped GT of the best other annotated cam: cost-weighted mean {sum(r['cost'] * m for r, m in zip(out1, mx)) / tot:.3f}; "
      f"pairs >= 50%: {sum(m >= 0.5 for m in mx)}/{len(out1)} cost {sum(r['cost'] for r, m in zip(out1, mx) if m >= 0.5):+.4f}")

# ---------------------------------------------------------------- (2) in-view overlap over the 664 pairs
print("(2) in-view seed overlap between basic objects, all 664 non-reference pairs")
rows_by = {(r["scene"], r["cam"], r["obj"]): r for r in SF}
n_pairs = n_touched = 0
ov_px = 0
d_jf, d_jf_fail, d_jf_good = [], [], []
cost_touched = cost_touched_fail = 0.0
per_scene = {}
for sc in sorted(OBJ):
    folder, st = scene_cfg(sc)
    ref = OBJ[sc]["c_ini"]
    basic = OBJ[sc]["seed_ids_c_ini"]
    for cam in OBJ[sc]["cameras"]:
        if cam == ref:
            continue
        GT = read_gt(folder, cam, st)
        hw = GT.shape
        seeds = {o: read_mask(sc, SEED, cam, o, hw) for o in basic}
        count = np.zeros(hw, np.int32)
        for m in seeds.values():
            count += m
        multi = count >= 2
        ov_px += int(multi.sum())
        # oracle: a multi-claimed pixel goes to its GT owner when that owner is among the claimants
        # (oc = the GT owner's own seed covers the pixel); otherwise every claimant keeps it
        oc = np.zeros(hw, bool)
        for q, mq in seeds.items():
            oc |= (GT == q) & mq
        fixed = {o: m & ~(multi & oc & (GT != o)) for o, m in seeds.items()}
        for o in basic:
            k = (sc, cam, o)
            if k not in rows_by:
                continue
            r = rows_by[k]
            g = GT == o
            if not g.any():
                continue
            n_pairs += 1
            if (fixed[o] != seeds[o]).any():
                n_touched += 1
                cost_touched += r["cost"]
                if r["JF"] < 0.8:
                    cost_touched_fail += r["cost"]
                d = jf_pair(g, fixed[o]) - jf_pair(g, seeds[o])
            else:
                d = 0.0
            d_jf.append(d)
            (d_jf_fail if r["JF"] < 0.8 else d_jf_good).append(d)
            per_scene.setdefault(sc, []).append(d)
print(f"   pairs {n_pairs}; multi-claimed seed px total {ov_px}; pairs whose seed changes under the oracle {n_touched} "
      f"(stage-2 cost of those pairs {cost_touched:+.4f}, of which failing seeds {cost_touched_fail:+.4f})")
print(f"   oracle keyframe dJ&F mean over 664: {np.mean(d_jf):+.5f}  (fail {np.mean(d_jf_fail):+.5f} n={len(d_jf_fail)}; good {np.mean(d_jf_good):+.5f} n={len(d_jf_good)}); "
      f"pairs up/down > 0.01: {sum(x > 0.01 for x in d_jf)}/{sum(x < -0.01 for x in d_jf)}")
print("   per scene mean dJ&F: " + ", ".join(f"{s} {np.mean(v):+.4f}" for s, v in sorted(per_scene.items(), key=lambda x: -np.mean(x[1]))))
json.dump(dict(under=out1, overlap=dict(n_pairs=n_pairs, ov_px=ov_px, n_touched=n_touched, cost_touched=cost_touched,
                                        cost_touched_fail=cost_touched_fail, d_jf_mean=float(np.mean(d_jf)),
                                        d_jf_fail=float(np.mean(d_jf_fail)), d_jf_good=float(np.mean(d_jf_good)),
                                        per_scene={s: float(np.mean(v)) for s, v in per_scene.items()})),
          open(os.path.join(D, "coupling3.json"), "w"), indent=1)

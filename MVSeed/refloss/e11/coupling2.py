"""Second pass on the 59 failing pairs (reads coupling.json): is the missing / wrong part of the seed already
in the round-2 memory set?  For every pair, the three candidate neighbours' round-1 masks (geoset_pred_ft_r1_e8)
are forward-warped into the target view with the Pi3 geometry (occlusion-checked against the target depth,
nearest splat + small dilation to close splat holes) and compared with the missed GT pixels (under-cover) or
the wrong seed pixels (spill / wrong object).  Also: how much of the wrong part is claimed by another object's
seed in the same view (an in-view exclusivity rule could resolve it).  Host, read-only.

    stage1/.venv/bin/python coupling2.py <dir with coupling.json>
"""
import collections
import json
import math
import os
import sys

import cv2
import numpy as np

REPO = "/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1/Data")
MV = os.path.join(REPO, "Data/MVSeg")
CFG = json.load(open(os.path.join(REPO, "SCSam3/demo/MVSeg.json")))
OBJ = json.load(open(os.path.join(REPO, "docs/raw/muvod_object_sets.json")))
SEED, R1 = "geoset_pred_ft_rep_e8", "geoset_pred_ft_r1_e8"
CONF_MIN, OCC_TOL = 0.1, 0.01
D = sys.argv[1] if len(sys.argv) > 1 else "."
C = json.load(open(os.path.join(D, "coupling.json")))


def scene_cfg(sc):
    c = CFG.get(sc) or next(v for v in CFG.values() if v.get("folder") == sc)
    return c["folder"], c["start_frame"]


def read_gt(folder, cam, st):
    g = cv2.imread(os.path.join(MV, folder, "Mask", cam, "%06d.png" % st), cv2.IMREAD_UNCHANGED)
    return g[..., 0] if g.ndim == 3 else g


def read_mask(sc, res, cam, o, hw):
    p = os.path.join(S1, sc, "Result", res, cam, f"{o}.png")
    s = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
    if s is None:
        return np.zeros(hw, bool)
    if s.shape != hw:
        s = cv2.resize(s, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return s > 127


def warp_mask(sc, src, dst, hw, mask_src):
    """Forward-warp a bool mask of view `src` (full res) into view `dst`: Pi3 points of src -> dst frame -> dst pixels,
    occlusion-checked against dst's Pi3 depth (as geomem.warp_torch.reproject_coords with the roles swapped).
    Returns a bool [h,w] map in dst (dilated by the work->full resolution ratio to close splat holes)."""
    cache = os.path.join(S1, sc, "Cache", "pi3")
    h, w = hw
    zu = np.load(os.path.join(cache, f"{src}.npz"))
    zr = np.load(os.path.join(cache, f"{dst}.npz"))
    v, u = np.mgrid[0:zu["depth"].shape[0], 0:zu["depth"].shape[1]].astype(np.float64)
    z = zu["depth"].astype(np.float64)
    Ku = zu["K"]
    p = np.stack([(u - Ku[0, 2]) / Ku[0, 0] * z, (v - Ku[1, 2]) / Ku[1, 1] * z, z], -1)
    p[zu["depth"] <= 0] = 0
    Tm = np.linalg.inv(zr["c2w"].astype(np.float64)) @ zu["c2w"].astype(np.float64)
    pts = (p @ Tm[:3, :3].T + Tm[:3, 3]).astype(np.float32)
    valid_u = ((zu["conf"] > CONF_MIN) & (zu["depth"] > 0)).astype(np.uint8)
    pts_full = cv2.resize(pts, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_full = cv2.resize(valid_u, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    depth_dst = cv2.resize(zr["depth"], (w, h), interpolation=cv2.INTER_NEAREST)
    hw_work = zr["depth"].shape
    sx, sy = w / hw_work[1], h / hw_work[0]
    Kr = zr["K"]
    fx, fy, cx, cy = Kr[0, 0] * sx, Kr[1, 1] * sy, (Kr[0, 2] + 0.5) * sx - 0.5, (Kr[1, 2] + 0.5) * sy - 0.5
    zz = pts_full[..., 2].astype(np.float64)
    zs = np.where(np.abs(zz) < 1e-9, 1e-9, zz)
    uu = fx * pts_full[..., 0] / zs + cx
    vv = fy * pts_full[..., 1] / zs + cy
    valid = (zz > 0) & (uu >= -0.5) & (uu <= w - 0.5) & (vv >= -0.5) & (vv <= h - 0.5) & np.isfinite(uu) & np.isfinite(vv)
    col = np.clip(np.round(uu), 0, w - 1).astype(np.int64)
    row = np.clip(np.round(vv), 0, h - 1).astype(np.int64)
    dd = depth_dst[row, col]
    valid &= (dd <= 0) | (zz <= dd * (1.0 + OCC_TOL))
    valid &= valid_full
    sel = valid & mask_src
    out = np.zeros(hw, bool)
    out[row[sel], col[sel]] = True
    k = int(math.ceil(max(h / hw_work[0], w / hw_work[1]))) + 1
    out = cv2.dilate(out.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
    return out, int(sel.sum())


def frac(mask, sel):
    n = int(mask.sum())
    return (float((mask & sel).sum()) / n) if n else None


res = []
for p in C["pairs"]:
    if p["group"] == "rest" and p["kind"] not in ("wrong-object", "miss"):
        continue
    sc, cam, o = p["scene"], p["cam"], p["obj"]
    folder, st = scene_cfg(sc)
    GT = read_gt(folder, cam, st)
    hw = GT.shape
    g = GT == o
    s = read_mask(sc, SEED, cam, o, hw)
    missed, extra = g & ~s, s & ~g
    basic = OBJ[sc]["seed_ids_c_ini"]
    others = np.zeros(hw, bool)
    other_seed = np.zeros(hw, bool)
    for q in basic:
        if q == o:
            continue
        m = read_mask(sc, SEED, cam, q, hw)
        others |= m
        if q == p["other_obj"]:
            other_seed = m
    cov = []
    adm_union, all_union = np.zeros(hw, bool), np.zeros(hw, bool)
    for n in p["neighbours"]:
        m1 = read_mask(sc, R1, n["cam"], o, hw)
        wm, n_valid = warp_mask(sc, n["cam"], cam, hw, m1)
        cov.append(dict(cam=n["cam"], admitted=n["confident"], r1_px=int(m1.sum()), warped_px=int(wm.sum()), n_valid=n_valid,
                        cov_gt=frac(g, wm), cov_missed=frac(missed, wm), cov_extra=frac(extra, wm)))
        all_union |= wm
        if n["confident"]:
            adm_union |= wm
    # the reference GT of this object and of the spilled-onto object, projected into the target
    ref = OBJ[sc]["c_ini"]
    G0 = read_gt(folder, ref, st)
    ref_own, _ = warp_mask(sc, ref, cam, hw, G0 == o)
    ref_other, _ = warp_mask(sc, ref, cam, hw, (G0 == p["other_obj"]) if p["other_obj"] else np.zeros(G0.shape, bool))
    ref_any_other, _ = warp_mask(sc, ref, cam, hw, (G0 != o) & (G0 != 0))
    row = dict(scene=sc, cam=cam, obj=o, kind=p["kind"], group=p["group"], cost=p["cost"], gt_px=int(g.sum()), seed_px=int(s.sum()),
               missed_px=int(missed.sum()), extra_px=int(extra.sum()),
               missed_in_ref_proj=frac(missed, ref_own), missed_in_admitted_nb=frac(missed, adm_union), missed_in_any_nb=frac(missed, all_union),
               missed_in_ref_or_admitted=frac(missed, ref_own | adm_union),
               extra_on_other_seed_same_view=frac(extra, others), extra_on_spilled_obj_seed=frac(extra, other_seed),
               extra_on_ref_other_proj=frac(extra, ref_any_other), extra_on_ref_own_proj=frac(extra, ref_own),
               extra_in_admitted_nb=frac(extra, adm_union), extra_in_any_nb=frac(extra, all_union),
               gt_in_ref_proj=frac(g, ref_own), gt_in_admitted_nb=frac(g, adm_union),
               neighbours=cov, n_admitted=sum(n["confident"] for n in p["neighbours"]))
    res.append(row)
    f = lambda x: "  -- " if x is None else f"{100 * x:5.0f}"
    if p["group"] == "under" or p["kind"] == "miss":
        print(f"{sc[:12]:12s} {cam:12s} {o:3d} {p['kind'][:22]:22s} cost {p['cost']:+.4f} missed {row['missed_px']:7d} px: in projected ref GT {f(row['missed_in_ref_proj'])}%  "
              f"in admitted nb r1 {f(row['missed_in_admitted_nb'])}%  in any nb r1 {f(row['missed_in_any_nb'])}%  ref|adm {f(row['missed_in_ref_or_admitted'])}%  "
              f"(nb: " + ", ".join(f"{c['cam']}{'A' if c['admitted'] else 'x'} {f(c['cov_missed'])}" for c in cov) + ")")
    else:
        print(f"{sc[:12]:12s} {cam:12s} {o:3d} {p['kind'][:22]:22s} cost {p['cost']:+.4f} extra {row['extra_px']:7d} px: on another obj's seed (same view) {f(row['extra_on_other_seed_same_view'])}% "
              f"(spilled obj {p['other_obj']}: {f(row['extra_on_spilled_obj_seed'])}%)  on projected ref GT of other objs {f(row['extra_on_ref_other_proj'])}%  own {f(row['extra_on_ref_own_proj'])}%  "
              f"in admitted nb r1 {f(row['extra_in_admitted_nb'])}%  any nb {f(row['extra_in_any_nb'])}%")
json.dump(res, open(os.path.join(D, "coupling2.json"), "w"), indent=1)
# cost-weighted summaries
for grp_name, sel in (("under", lambda r: r["group"] == "under"), ("spill+wrong", lambda r: r["group"] == "spill" or r["kind"] == "wrong-object")):
    rs = [r for r in res if sel(r)]
    tot = sum(r["cost"] for r in rs)
    if grp_name == "under":
        for key_ in ("missed_in_ref_proj", "missed_in_admitted_nb", "missed_in_any_nb", "missed_in_ref_or_admitted"):
            wsum = sum(r["cost"] * (r[key_] or 0) for r in rs)
            n_half = sum(1 for r in rs if (r[key_] or 0) >= 0.5)
            print(f"SUM {grp_name:12s} {key_:28s} cost-weighted mean {wsum / tot:.3f} (cost {wsum:+.4f} of {tot:+.4f}); pairs >= 50%: {n_half}/{len(rs)} "
                  f"cost {sum(r['cost'] for r in rs if (r[key_] or 0) >= 0.5):+.4f}")
    else:
        for key_ in ("extra_on_other_seed_same_view", "extra_on_spilled_obj_seed", "extra_on_ref_other_proj", "extra_on_ref_own_proj", "extra_in_admitted_nb", "extra_in_any_nb"):
            wsum = sum(r["cost"] * (r[key_] or 0) for r in rs)
            n_half = sum(1 for r in rs if (r[key_] or 0) >= 0.5)
            print(f"SUM {grp_name:12s} {key_:28s} cost-weighted mean {wsum / tot:.3f} (cost {wsum:+.4f} of {tot:+.4f}); pairs >= 50%: {n_half}/{len(rs)} "
                  f"cost {sum(r['cost'] for r in rs if (r[key_] or 0) >= 0.5):+.4f}")

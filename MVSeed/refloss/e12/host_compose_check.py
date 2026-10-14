"""Judge check (host, numpy only): independent re-implementation of the E12 design-2 composition null.

INV = ~vis with vis re-derived in numpy from the Pi3 caches (same formulas as run.reproj_inputs +
warp_torch.reproject_coords with depth_ref, OCC_TOL 0.01, CONF_MIN 0.1), composition rule as registered, J/F by
mvseg2d.metrics.jf_pair.  Compares every row with null_compose.json, then measures two things the design-time script
did not: (a) U computed from ALL C0 PNGs of the view (the registered rule: hallucinated masks included) instead of
the scored rows' objects only; (b) pixels added to >= 2 objects (the 'no overlap added' invariant).
    PYTHONDONTWRITEBYTECODE=1 stage1/.venv/bin/python -I host_compose_check.py <null_compose.json>
"""
import json
import os
import sys
import time

import cv2
import numpy as np

REPO = "/home/sjpark/Documents/SCSegmentation"
S1 = os.path.join(REPO, "stage1")
sys.path.insert(0, S1)
from geomem.pointmap import CONF_MIN, local_points, to_reference  # noqa: E402
from mvseg2d.io import load_labelmap, load_meta, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402
from mvseg2d.paired import paired_scene_bootstrap  # noqa: E402

DATA = os.path.join(S1, "Data")
C0 = "geoset_pred_ft_rep_x_e11"
NULLS = ("geoset_dec_ft_rep_e9", "geoset_decb_ft_rep_e9", "geoset_pred_ft_r1_x_e11")
OCC_TOL = 0.01


def vis_np(cache, cam, ref, hw):
    h, w = hw
    zu = np.load(os.path.join(cache, f"{cam}.npz"))
    zr = np.load(os.path.join(cache, f"{ref}.npz"))
    pts = to_reference(local_points(zu["depth"], zu["K"]), zu["c2w"], zr["c2w"]).astype(np.float32)
    valid_u = ((zu["conf"] > CONF_MIN) & (zu["depth"] > 0)).astype(np.uint8)
    pts_full = cv2.resize(pts, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_full = cv2.resize(valid_u, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    depth_ref = cv2.resize(zr["depth"], (w, h), interpolation=cv2.INTER_NEAREST)
    hw_work = zr["depth"].shape
    sx, sy = w / hw_work[1], h / hw_work[0]
    K = zr["K"]
    Kf = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                   [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)
    x, y, z = pts_full[..., 0], pts_full[..., 1], pts_full[..., 2]
    zs = np.where(np.abs(z) < 1e-9, np.float32(1e-9), z)
    with np.errstate(all="ignore"):
        u = Kf[0, 0] * x / zs + Kf[0, 2]
        v = Kf[1, 1] * y / zs + Kf[1, 2]
        valid = (z > 0) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5) & np.isfinite(u) & np.isfinite(v)
        col = np.clip(np.round(np.nan_to_num(u)), 0, w - 1).astype(np.int64)
        row = np.clip(np.round(np.nan_to_num(v)), 0, h - 1).astype(np.int64)
    d = depth_ref[row, col]
    valid = valid & ((d <= 0) | (z <= d * np.float32(1.0 + OCC_TOL)))
    return valid & valid_full


def read_seed(scene, seeds, cam, obj, hw):
    p = os.path.join(DATA, scene, "Result", seeds, cam, f"{obj}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    return read_pred_mask(p)


def compose(c0_b, e_b, inv, unclaimed):
    if not c0_b.any():
        return c0_b.copy(), np.zeros_like(c0_b)
    add = e_b & inv & unclaimed
    if not add.any():
        return c0_b.copy(), add
    _, lab = cv2.connectedComponents((c0_b | add).astype(np.uint8), connectivity=8)
    keep = np.isin(lab, np.unique(lab[c0_b]))
    add_kept = add & keep
    return c0_b | add_kept, add_kept


def main():
    t0 = time.time()
    ref_json = json.loads(open(sys.argv[1]).read()[open(sys.argv[1]).read().index("{"):])
    refrows = {(r["scene"], r["cam"], r["obj"]): r for r in ref_json["rows"]}
    rows_in = json.load(open(os.path.join(REPO, "MVSeed", "runs", "seedfail.json")))["rows"]
    by_scene = {}
    for r in rows_in:
        by_scene.setdefault(r["scene"], []).append(r)
    out = []
    mism = {nm: {"add": 0, "add_tp": 0, "J": 0, "F": 0, "max_add_diff": 0, "max_J_diff": 0.0} for nm in NULLS}
    vis_diff_px = 0
    overlap = {nm: 0 for nm in NULLS}
    overlap_full = {nm: 0 for nm in NULLS}
    for scene in sorted(by_scene):
        srows = by_scene[scene]
        meta = load_meta(DATA, scene)
        ref = meta["ref_cam"]
        cache = os.path.join(DATA, scene, "Cache", "pi3")
        ref_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{ref}.png"))
        hw = tuple(ref_lab.shape)
        bp = bound_pix(*hw)
        se = disk(bp)
        for cam in sorted({r["cam"] for r in srows}):
            gt_lab = load_labelmap(os.path.join(DATA, scene, "masks", f"{cam}.png"))
            inv = ~vis_np(cache, cam, ref, hw)
            objs = sorted({r["obj"] for r in srows if r["cam"] == cam})
            folder = os.path.join(DATA, scene, "Result", C0, cam)
            all_objs = sorted(int(f[:-4]) for f in os.listdir(folder) if f.endswith(".png"))
            c0 = {o: read_seed(scene, C0, cam, o, hw) for o in set(objs) | set(all_objs)}
            u_rows = ~np.logical_or.reduce([c0[o] for o in objs])
            u_all = ~np.logical_or.reduce([c0[o] for o in all_objs]) if all_objs else np.ones(hw, bool)
            for nm in NULLS:
                ext = {o: read_seed(scene, nm, cam, o, hw) for o in objs}
                cnt = np.zeros(hw, np.uint8)
                cnt2 = np.zeros(hw, np.uint8)
                for r in [r for r in srows if r["cam"] == cam]:
                    o = r["obj"]
                    gt = gt_lab == o
                    final, add_kept = compose(c0[o], ext[o], inv, u_rows)
                    final2, add_kept2 = compose(c0[o], ext[o], inv, u_all)
                    cnt += add_kept.astype(np.uint8)
                    cnt2 += add_kept2.astype(np.uint8)
                    J, F = jf_pair(gt, final, bp, se)
                    J2, F2 = jf_pair(gt, final2, bp, se)
                    key = (scene, cam, o)
                    x = refrows[key][nm]
                    a, atp = int(add_kept.sum()), int((add_kept & gt).sum())
                    m = mism[nm]
                    m["add"] += a != x["add"]
                    m["add_tp"] += atp != x["add_tp"]
                    m["J"] += abs(J - x["J"]) > 1e-9
                    m["F"] += abs(F - x["F"]) > 1e-9
                    m["max_add_diff"] = max(m["max_add_diff"], abs(a - x["add"]))
                    m["max_J_diff"] = max(m["max_J_diff"], abs(J - x["J"]))
                    out.append(dict(scene=scene, cam=cam, obj=o, null=nm, J=J, F=F, J_allU=J2, F_allU=F2,
                                    add=a, add_allU=int(add_kept2.sum()), C0J=refrows[key]["C0"]["J"], C0F=refrows[key]["C0"]["F"]))
                overlap[nm] += int((cnt >= 2).sum())
                overlap_full[nm] += int((cnt2 >= 2).sum())
        print(f"{scene} done {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    print("row mismatches vs null_compose.json (counts of 664 rows):", json.dumps(mism, default=float))
    print("pixels added to >= 2 scored objects (U from rows):", overlap, " (U from all C0 PNGs):", overlap_full)
    for nm in NULLS:
        rs = [r for r in out if r["null"] == nm]
        base = [dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], JF=0.5 * (r["C0J"] + r["C0F"])) for r in rs]
        comp = [dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], JF=0.5 * (r["J"] + r["F"])) for r in rs]
        comp2 = [dict(scene=r["scene"], cam=r["cam"], obj=r["obj"], JF=0.5 * (r["J_allU"] + r["F_allU"])) for r in rs]
        b1 = paired_scene_bootstrap(base, comp, key="JF")
        b2 = paired_scene_bootstrap(base, comp2, key="JF")
        print(f"{nm}: recomputed composed dJ&F {b1['delta']:+.4f} [{b1['ci'][0]:+.4f}, {b1['ci'][1]:+.4f}]  | registered U (all C0 PNGs) "
              f"{b2['delta']:+.4f} [{b2['ci'][0]:+.4f}, {b2['ci'][1]:+.4f}]  added px rows-U {sum(r['add'] for r in rs):,} allU {sum(r['add_allU'] for r in rs):,}"
              f"  rows changed by allU {sum(r['add'] != r['add_allU'] for r in rs)}")
    json.dump(out, open("host_compose_rows.json", "w"), default=float)
    print(f"seconds {time.time() - t0:.0f}")


main()

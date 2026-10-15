#!/usr/bin/env python3
"""S1-E13 DEV scorer: per-pair J&F and reference-visibility anatomy of stage-1 masks on the DEV set (never the keyframe).

Pairs = (scene, t, cam, obj) for t in <dev>/index.json frames (non-ref train frames ref+4..ref+20), cam in the 2 annotated
non-reference cameras, obj in basic set (docs/raw/muvod_object_sets.json) present in c_ini's GT at t, GT of obj in cam at t
non-empty (as the 664 keyframe pairs of MVSeed/runs/seedfail.json: gt_px > 0).  Masks <masks>/<scene>/<t:06d>/<cam>/<obj>.png
(absent = empty), the layout dev_decode.py writes; any candidate in that layout can be scored with --masks.

Per pair: J, F, JF (mvseg2d.metrics.jf_pair, bound_pix of the frame), recall, precision, frac_other, vis_ratio, twin, label,
the seedcat kind (MVSeed/refloss/seedcat.py rules), and the pixel classes of MVSeed/refloss/e11/extent_vis.py with the frame's
Pi3 (Cache/pi3_frames/<t>; methods.geomem.run.reproj_inputs + project_reference_masks, OCC_TOL 0.01, conf > 0.1 & depth > 0):
    vis (visible from c_ini) / outside (valid depth, outside c_ini image or behind) / occluded (inside, fails depth test) /
    nodepth (no valid Pi3 depth) / on_refgt (vis and lands on c_ini GT of the same object)
for the GT (gt_*) and for the missed set GT \\ mask (missed_*); with --r1, also the round-1 J&F of the same cam (<t>/_r1/<cam>)
and the missed pixels covered by the k=3 candidates' ROUND-1 masks reprojected into the cam (missed_nb_k3_r1).

    flock <lock> stage1/scripts/docker_run.sh scsam3 python geomem/runs/_e13_dev-route/dev_score.py \
        --masks geomem/runs/_e13_dev --out geomem/runs/_e13_dev/dev_pairs.json [--r1]
"""
import argparse
import json
import os
import sys
import time

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, S1)
import cv2  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from geomem.cache import pi3_frame_cache  # noqa: E402
from geomem.warp_torch import OCC_TOL, reproject_coords  # noqa: E402
from methods.geomem.run import camera_centres, project_reference_masks, reproj_inputs, select_candidates  # noqa: E402
from mvseg2d.io import load_labelmap, load_labels, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")
K = 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def kind(r):
    """MVSeed/refloss/seedcat.py kind() verbatim (diagnostic, uses the GT)."""
    if r["seed_px"] == 0 or (r["recall"] or 0) < 0.05:
        return "miss" if r["seed_px"] == 0 or r["gt_px"] < 50 else "wrong-object"
    if (r["precision"] or 0) < 0.3:
        return "wrong-object"
    if r["frac_other"] >= 0.2:
        return "spill-onto-other" + ("(twin)" if r["twin"] else "")
    if r["recall"] < 0.85 and r["precision"] >= 0.85:
        return "under-cover" + ("(beyond c_ini)" if r["vis_ratio"] > 1.2 else "")
    return "boundary/mixed"


def read(root, scene, t, cam, o, hw):
    p = os.path.join(root, scene, f"{t:06d}", cam, f"{o}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    m = read_pred_mask(p)
    assert m.shape == tuple(hw), (p, m.shape, hw)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--index", default=os.path.join(S1, "geomem", "runs", "_e13_dev", "index.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--r1", action="store_true", help="also score <masks>/<scene>/<t>/_r1 and the k=3 round-1 neighbour coverage")
    ap.add_argument("--scenes", default="all")
    a = ap.parse_args()
    idx = json.load(open(a.index))
    t0 = time.time()
    rows, empty_gt = [], []
    scenes = sorted(idx["scenes"]) if a.scenes == "all" else a.scenes.split(",")
    for scene in scenes:
        S = idx["scenes"][scene]
        ref, basic = S["ref_cam"], [int(o) for o in S["basic"]]
        labels = load_labels(DATA, scene)
        lab_of = {o: (labels.get(o) or {}).get("label") for o in basic}
        twin = {o: lab_of[o] is not None and any(lab_of[q] == lab_of[o] for q in basic if q != o) for o in basic}
        ts = time.time()
        for t in S["frames"]:
            assert t != S["ref_frame"], (scene, t)
            fdir = os.path.join(DATA, scene, "frames", f"{t:06d}")
            ref_lab = load_labelmap(os.path.join(fdir, "masks", f"{ref}.png"))
            hw = tuple(ref_lab.shape)
            objs = [o for o in basic if (ref_lab == o).any()]
            oi = {o: i for i, o in enumerate(objs)}
            bp = bound_pix(*hw)
            se = disk(bp)
            cache = pi3_frame_cache(DATA, scene, t)
            assert json.load(open(os.path.join(cache, "MANIFEST.json")))["ref_cam"] == ref
            ref_masks = torch.from_numpy(np.stack([ref_lab == o for o in objs])).to(DEV)
            ref_px = {o: int((ref_lab == o).sum()) for o in objs}
            centres = camera_centres(cache, sorted(f[:-4] for f in os.listdir(cache) if f.endswith(".npz")))
            for cam in S["eval_cams"]:
                gt_lab = load_labelmap(os.path.join(fdir, "masks", f"{cam}.png"))
                assert gt_lab.shape == hw
                gt = torch.from_numpy(np.stack([gt_lab == o for o in objs])).to(DEV)
                seed_np = np.stack([read(a.masks, scene, t, cam, o, hw) for o in objs])
                seed = torch.from_numpy(seed_np).to(DEV)
                missed = gt & ~seed
                pts, valid_full, depth_ref, K_ref = reproj_inputs(cache, cam, ref, hw, DEV)
                pm, vis = project_reference_masks(ref_masks, pts, valid_full, depth_ref, K_ref)
                _, inside = reproject_coords(pts, K_ref, hw, depth_ref=None)
                inside = inside & valid_full
                cls = dict(vis=vis, outside=valid_full & ~inside, occluded=inside & ~vis, nodepth=~valid_full)

                def cnt(m):
                    return m.flatten(1).sum(1).cpu().numpy()
                c_gt = {k: cnt(gt & v[None]) for k, v in cls.items()}
                c_gt["on_refgt"] = cnt(gt & pm)
                c_ms = {k: cnt(missed & v[None]) for k, v in cls.items()}
                c_ms["on_refgt"] = cnt(missed & pm)
                cand = [c for c in select_candidates(cam, centres, K) if c != ref]
                c_nb = None
                r1_np = None
                if a.r1:
                    r1_np = np.stack([_read_r1(a.masks, scene, t, cam, o, hw) for o in objs])
                    union = torch.zeros_like(seed)
                    for nb in cand:
                        nbm = torch.from_numpy(np.stack([_read_r1(a.masks, scene, t, nb, o, hw) for o in objs])).to(DEV)
                        pn, vn, dn, kn = reproj_inputs(cache, cam, nb, hw, DEV)
                        nm, _ = project_reference_masks(nbm, pn, vn, dn, kn)
                        union |= nm
                        del nbm, pn, vn, dn, kn, nm
                    c_nb = cnt(missed & union)
                    del union
                gt_np = gt.cpu().numpy()
                n_gt, n_seed, n_miss = cnt(gt), seed_np.reshape(len(objs), -1).sum(1), cnt(missed)
                for o in objs:
                    b = oi[o]
                    g, s = gt_np[b], seed_np[b]
                    gp, spx = int(n_gt[b]), int(n_seed[b])
                    if gp == 0:
                        empty_gt.append(dict(scene=scene, t=t, cam=cam, obj=o, seed_px=spx))
                        continue
                    J, F = jf_pair(g, s, bp, se)
                    inter = int((g & s).sum())
                    other = {int(x): int(c) for x, c in zip(*np.unique(gt_lab[s & ~g], return_counts=True)) if x != 0}
                    oo = max(other, key=other.get) if other else None
                    r = dict(scene=scene, t=t, cam=cam, obj=o, label=lab_of[o], J=float(J), F=float(F), JF=float((J + F) / 2),
                             gt_px=gp, seed_px=spx, recall=inter / gp, precision=inter / spx if spx else None,
                             frac_other=(other[oo] / spx) if oo else 0.0, other_obj=oo, vis_ratio=gp / max(1, ref_px[o]),
                             twin=twin[o], missed_px=int(n_miss[b]), k3=cand)
                    for k2 in ("vis", "outside", "occluded", "nodepth", "on_refgt"):
                        r[f"gt_{k2}"] = int(c_gt[k2][b])
                        r[f"missed_{k2}"] = int(c_ms[k2][b])
                    r["gt_invis_frac"] = 1.0 - r["gt_vis"] / gp
                    r["kind"] = kind(r)
                    if a.r1:
                        J1, F1 = jf_pair(g, r1_np[b], bp, se)
                        r.update(JF_r1=float((J1 + F1) / 2), missed_nb_k3_r1=int(c_nb[b]))
                    rows.append(r)
                del gt, seed, missed, pts, valid_full, depth_ref, pm, vis, inside, cls
                if DEV == "cuda":
                    torch.cuda.empty_cache()
        print(f"{scene:20s} {sum(r['scene'] == scene for r in rows)} pairs  {time.time() - ts:.1f}s", file=sys.stderr, flush=True)
    meta = dict(masks=os.path.abspath(a.masks), index=os.path.abspath(a.index), occ_tol=OCC_TOL, k=K, device=DEV,
                n_rows=len(rows), n_empty_gt=len(empty_gt), seconds=time.time() - t0, r1=bool(a.r1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(meta=meta, rows=rows, empty_gt=empty_gt), f)
    print(f"done {len(rows)} rows, {len(empty_gt)} empty-GT pairs excluded, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


def _read_r1(root, scene, t, cam, o, hw):
    p = os.path.join(root, scene, f"{t:06d}", "_r1", cam, f"{o}.png")
    if not os.path.isfile(p):
        return np.zeros(hw, bool)
    m = read_pred_mask(p)
    assert m.shape == tuple(hw), (p, m.shape, hw)
    return m


if __name__ == "__main__":
    main()

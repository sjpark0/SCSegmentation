#!/usr/bin/env python3
"""Score stage-1 masks on the standard DEV set (geomem/devset.py, docs/stage1-dev-protocol.md): per-pair J&F and the
reference-visibility anatomy -- never the keyframe.  From the S1-E13 scratch scorer (MVSeed/refloss/e13/dev-route).

Pairs = (scene, t, cam, obj) for t in <masks>/index.json frames, cam in the 2 annotated non-reference cameras, obj in the
basic set present in c_ini's GT at t with non-empty GT in cam.  Masks <masks>/<scene>/<t:06d>/<cam>/<obj>.png (absent =
empty), the layout methods.geomem.dev_decode writes; any candidate in that layout can be scored.

Per pair: J, F, JF (mvseg2d.metrics.jf_pair), recall, precision, frac_other, vis_ratio, twin, label, the seedcat kind, a
`view` key "<t>/<cam>" (the DEV view for mvseg2d.paired with view_key="view"), `covers` = the GT object the predicted
mask overlaps most at 1/4 resolution (geomem.id3d.covers = the ring score.py rule; 0 = empty / no object) and
`other_obj_mask` = covers not in {0, obj} (S1-E15 B4), and the pixel classes of
MVSeed/refloss/e11/extent_vis.py with the frame's Pi3 (vis / outside / occluded / nodepth / on_refgt) for the GT (gt_*) and
the missed set (missed_*); with --r1, also the round-1 J&F and the k=3 round-1 neighbour coverage of the missed pixels.
Reference rotation (docs/stage1-dev-protocol.md §4, DEV-A' / DEV-B'): an index.json with `rot` names the rotated input
camera as ref_cam (the Pi3 cache keeps the original c_ini, orig_ref_cam); rows then carry `ref` and the view key
"<t>/<ref>/<cam>" (pair key (scene, t, ref, cam, obj)).
Runs inside scsam3:
    scripts/docker_run.sh scsam3 python -m methods.geomem.dev_score --masks Data/_dev/geoset_rep_x_A --out <json> [--r1]
"""
import argparse
import json
import os
import sys
import time

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, S1)
import cv2  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from geomem.cache import pi3_frame_cache  # noqa: E402
from geomem.id3d import covers as covers_q  # noqa: E402
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
    ap.add_argument("--index", default=None, help="default <masks>/index.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r1", action="store_true", help="also score <masks>/<scene>/<t>/_r1 and the k=3 round-1 neighbour coverage")
    ap.add_argument("--scenes", default="all")
    a = ap.parse_args()
    a.index = a.index or os.path.join(a.masks, "index.json")
    idx = json.load(open(a.index))
    t0 = time.time()
    rows, empty_gt = [], []
    scenes = sorted(idx["scenes"]) if a.scenes == "all" else a.scenes.split(",")
    for scene in scenes:
        S = idx["scenes"][scene]
        ref, basic = S["ref_cam"], [int(o) for o in S["basic"]]
        rot = int(idx.get("rot", 0))
        cache_ref = S.get("orig_ref_cam", ref) if rot else ref           # the frame cache's own c_ini
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
            assert json.load(open(os.path.join(cache, "MANIFEST.json")))["ref_cam"] == cache_ref
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
                        empty_gt.append(dict(scene=scene, t=t, cam=cam, obj=o, seed_px=spx, **({"ref": ref} if rot else {})))
                        continue
                    J, F = jf_pair(g, s, bp, se)
                    inter = int((g & s).sum())
                    other = {int(x): int(c) for x, c in zip(*np.unique(gt_lab[s & ~g], return_counts=True)) if x != 0}
                    oo = max(other, key=other.get) if other else None
                    r = dict(scene=scene, t=t, cam=cam, obj=o, label=lab_of[o], J=float(J), F=float(F), JF=float((J + F) / 2),
                             gt_px=gp, seed_px=spx, recall=inter / gp, precision=inter / spx if spx else None,
                             frac_other=(other[oo] / spx) if oo else 0.0, other_obj=oo, vis_ratio=gp / max(1, ref_px[o]),
                             twin=twin[o], missed_px=int(n_miss[b]), k3=cand, view=f"{t}/{ref}/{cam}" if rot else f"{t}/{cam}")
                    if rot:
                        r["ref"] = ref
                    r["covers"] = covers_q(s, gt_lab)
                    r["other_obj_mask"] = r["covers"] not in (0, o)
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
    if idx.get("rot"):
        meta["rot"] = int(idx["rot"])
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

#!/usr/bin/env python3
"""S1-E13 candidate E2, geometric oracle: how much of the adopted preset's missed pixels a SECOND annotated reference B
could reach by Pi3 reprojection (DEV frames only; the keyframe is never read).

For every DEV pair (scene, t, X, obj) of geomem/runs/_e13_dev/dev_pairs.json (the adopted geoset_rep_x decode), B = the other
annotated non-reference camera.  Every pixel of X is reprojected into B with the frame's Pi3 (methods.geomem.run.reproj_inputs
+ project_reference_masks, the same occlusion test OCC_TOL 0.01 used for c_ini) ->
    visB   : X pixel has valid Pi3 depth, lands inside B and passes B's depth test (B sees that 3D point)
    onBgt  : visB and lands on B's GT of the same object (the label a B reference would carry there)
counted on GT (gt_*) and on the adopted missed set GT \\ mask (missed_*), split by the c_ini class of the pixel
(vis / outside / occluded / nodepth, the dev_score.py classes).

    flock <lock> stage1/scripts/docker_run.sh scsam3 python geomem/runs/_e13_second-reference/dev_cover_B.py \
        --out geomem/runs/_e13_second-reference/cover_B.json [--arm both=geomem/runs/_e13_second-reference/both ...]
    with --arm: rec_<name>_* = adopted missed pixels that the arm's masks cover, split by visB / onBgt / c_ini class
"""
import argparse
import json
import os
import sys
import time

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, S1)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from geomem.cache import pi3_frame_cache  # noqa: E402
from geomem.warp_torch import reproject_coords  # noqa: E402
from methods.geomem.run import project_reference_masks, reproj_inputs  # noqa: E402
from mvseg2d.io import load_labelmap, read_pred_mask  # noqa: E402

DATA = os.path.join(S1, "Data")
BASE = os.path.join(S1, "geomem", "runs", "_e13_dev")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLS = ("vis", "outside", "occluded", "nodepth")


def read(root, scene, t, cam, o, hw):
    p = os.path.join(root, scene, f"{t:06d}", cam, f"{o}.png")
    return read_pred_mask(p) if os.path.isfile(p) else np.zeros(hw, bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=os.path.join(BASE, "dev_pairs.json"))
    ap.add_argument("--masks", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", default=[], help="name=mask root: also count the adopted missed pixels the "
                                                               "arm's masks now cover (rec_<name>_*), split as above")
    a = ap.parse_args()
    arms = [s.split("=", 1) for s in a.arm]
    idx = json.load(open(os.path.join(BASE, "index.json")))
    rows = json.load(open(a.pairs))["rows"]
    by = {}
    for r in rows:
        by.setdefault((r["scene"], r["t"], r["cam"]), []).append(r)
    t0 = time.time()
    out = []
    for (scene, t, X), rs in sorted(by.items()):
        S = idx["scenes"][scene]
        assert t != S["ref_frame"]
        ref = S["ref_cam"]
        Bcam = [c for c in S["eval_cams"] if c != X][0]
        fdir = os.path.join(DATA, scene, "frames", f"{t:06d}", "masks")
        gtX = load_labelmap(os.path.join(fdir, f"{X}.png"))
        gtB = load_labelmap(os.path.join(fdir, f"{Bcam}.png"))
        hw = gtX.shape
        assert gtB.shape == hw
        objs = [r["obj"] for r in rs]
        cache = pi3_frame_cache(DATA, scene, t)
        gt = torch.from_numpy(np.stack([gtX == o for o in objs])).to(DEV)
        seed = torch.from_numpy(np.stack([read(a.masks, scene, t, X, o, hw) for o in objs])).to(DEV)
        missed = gt & ~seed
        # c_ini classes (dev_score.py)
        ref_lab = load_labelmap(os.path.join(fdir, f"{ref}.png"))
        ref_masks = torch.from_numpy(np.stack([ref_lab == o for o in objs])).to(DEV)
        pts, valid_full, depth_ref, K_ref = reproj_inputs(cache, X, ref, hw, DEV)
        pm_ref, vis = project_reference_masks(ref_masks, pts, valid_full, depth_ref, K_ref)
        _, inside = reproject_coords(pts, K_ref, hw, depth_ref=None)
        inside = inside & valid_full
        cls = dict(vis=vis, outside=valid_full & ~inside, occluded=inside & ~vis, nodepth=~valid_full)
        del pts, depth_ref, ref_masks
        # B
        B_masks = torch.from_numpy(np.stack([gtB == o for o in objs])).to(DEV)
        pB, vB, dB, KB = reproj_inputs(cache, X, Bcam, hw, DEV)
        onB, visB = project_reference_masks(B_masks, pB, vB, dB, KB)
        del pB, vB, dB, KB

        def cnt(m):
            return m.flatten(1).sum(1).cpu().numpy()
        res = {}
        for name, base in (("gt", gt), ("missed", missed)):
            res[f"{name}_px"] = cnt(base)
            res[f"{name}_visB"] = cnt(base & visB[None])
            res[f"{name}_onBgt"] = cnt(base & onB)
            res[f"{name}_onBgt_or_refgt"] = cnt(base & (onB | pm_ref))
            for k, v in cls.items():
                res[f"{name}_{k}"] = cnt(base & v[None])
                res[f"{name}_{k}_visB"] = cnt(base & v[None] & visB[None])
                res[f"{name}_{k}_onBgt"] = cnt(base & v[None] & onB)
        for an, aroot in arms:
            am = torch.from_numpy(np.stack([read(aroot, scene, t, X, o, hw) for o in objs])).to(DEV)
            rec = missed & am
            res[f"rec_{an}"] = cnt(rec)
            res[f"rec_{an}_visB"] = cnt(rec & visB[None])
            res[f"rec_{an}_onBgt"] = cnt(rec & onB)
            for k, v in cls.items():
                res[f"rec_{an}_{k}"] = cnt(rec & v[None])
                res[f"rec_{an}_{k}_onBgt"] = cnt(rec & v[None] & onB)
            del am, rec
        Bpx = (B_masks.flatten(1).sum(1)).cpu().numpy()
        for i, r in enumerate(rs):
            d = dict(scene=scene, t=t, cam=X, obj=r["obj"], label=r["label"], B=Bcam, B_gt_px=int(Bpx[i]))
            d.update({k: int(v[i]) for k, v in res.items()})
            assert d["gt_px"] == r["gt_px"] and d["missed_px"] == r["missed_px"], (scene, t, X, r["obj"])
            for k in CLS:
                assert d[f"missed_{k}"] == r[f"missed_{k}"], (scene, t, X, r["obj"], k)
            out.append(d)
        del gt, seed, missed, cls, vis, inside, valid_full, onB, visB, B_masks, pm_ref
        if DEV == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(dict(meta=dict(pairs=os.path.abspath(a.pairs), masks=os.path.abspath(a.masks), n=len(out),
                                 seconds=time.time() - t0, device=DEV), rows=out), f)
    print(f"done {len(out)} rows {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

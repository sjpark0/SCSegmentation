#!/usr/bin/env python3
"""Verify the S1-E14 logit dump against a PNG root, in a fresh process, from the files on disk only.

  --mode torch  (scsam3, GPU):  e14dump.masks_torch  (the adopted upsampling + run.exclusive_masks)
  --mode numpy  (host):         e14dump.masks_numpy  (float32 emulation)
For every dumped eval view and every object: re-thresholded mask vs the reference PNG (absent = empty); also the
candidate consistency low_res == where(presence > 0, mm_low_res[token], -1024), ious == mm_ious[token] (fp16 domain).
Summary over all object-views and over the scored DEV pairs (dev_pairs.json rows) -> <root>/verify_<mode>.json.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, S1)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import e14dump as E  # noqa: E402


def check_one(job):
    """One dumped view -> (rows, candidate_ok, npz bytes, mm npz bytes)."""
    root, ref_root, mode, mm, pairs, (scene, t, cam) = job
    z = E.load(root, scene, t, cam, mm=mm)
    objs, hw = [int(o) for o in z["objects"]], tuple(int(v) for v in z["hw"])
    keep = E.masks_torch(z["low_res"], hw) if mode == "torch" else E.masks_numpy(z["low_res"], hw)
    ref = E.read_png_masks(ref_root, scene, t, cam, objs, hw)
    ok = True
    if mm:
        B = len(objs)
        tok = z["token"].astype(np.int64)
        sel = z["mm_low_res"][np.arange(B), tok]
        recon = np.where((z["presence"] > 0)[:, None, None], sel, np.float16(E.NO_OBJ_SCORE))
        ok = bool(np.array_equal(recon, z["low_res"]) and np.array_equal(z["mm_ious"][np.arange(B), tok], z["ious"]))
    rows = [dict(scene=scene, t=t, cam=cam, obj=o, diff_px=int((keep[i] != ref[i]).sum()), fg_px=int(ref[i].sum()),
                 pair=(scene, t, cam, o) in pairs) for i, o in enumerate(objs)]
    return rows, ok, os.path.getsize(E.logit_path(root, scene, t, cam)), os.path.getsize(E.logit_path(root, scene, t, cam, mm=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(S1, "geomem", "runs", "_e14_logitdump", "geoset_rep_x_A"))
    ap.add_argument("--ref-root", default=os.path.join(S1, "geomem", "runs", "_e13_dev"))
    ap.add_argument("--pairs", default=os.path.join(S1, "geomem", "runs", "_e13_dev", "dev_pairs.json"))
    ap.add_argument("--mode", choices=("torch", "numpy"), required=True)
    ap.add_argument("--no-mm", action="store_true", help="skip the candidate-consistency check (faster)")
    ap.add_argument("--workers", type=int, default=1, help="numpy mode: parallel processes over views")
    a = ap.parse_args()
    pairs = E.dev_pairs(a.pairs)
    t0 = time.time()
    jobs = [(a.root, a.ref_root, a.mode, not a.no_mm, pairs, v) for v in E.views(a.root)]
    if a.mode == "numpy" and a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            res = pool.map(check_one, jobs, chunksize=1)
    else:
        res = [check_one(j) for j in jobs]
    rows = [r for rs, _, _, _ in res for r in rs]
    mm_bad = [f"{j[-1][0]}/{j[-1][1]}/{j[-1][2]}" for j, (_, ok, _, _) in zip(jobs, res) if not ok]
    nview = len(res)
    bytes_lr = sum(x[2] for x in res)
    bytes_mm = sum(x[3] for x in res)
    in_pairs = [r for r in rows if r["pair"]]
    found = {(r["scene"], r["t"], r["cam"], r["obj"]) for r in in_pairs}
    out = dict(mode=a.mode, root=a.root, ref_root=a.ref_root, seconds=time.time() - t0, views=nview,
               object_views=len(rows), object_views_identical=sum(r["diff_px"] == 0 for r in rows),
               object_views_diff_px_total=sum(r["diff_px"] for r in rows),
               dev_pairs=len(pairs), dev_pairs_found=len(found), dev_pairs_identical=sum(r["diff_px"] == 0 for r in in_pairs),
               dev_pairs_diff_px_total=sum(r["diff_px"] for r in in_pairs),
               dev_pairs_missing=sorted(map(list, pairs - found))[:20],
               candidate_consistency_checked=not a.no_mm, candidate_consistency_bad=mm_bad,
               bytes_low_res_npz=bytes_lr, bytes_mm_npz=bytes_mm,
               diffs=[r for r in rows if r["diff_px"]][:200])
    if a.mode == "torch":
        import torch
        out.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    with open(os.path.join(a.root, f"verify_{a.mode}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "diffs"}, indent=1))
    print("first diffs", out["diffs"][:10])


if __name__ == "__main__":
    main()

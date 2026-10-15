#!/usr/bin/env python3
"""S1-E14 boundary anatomy: 288 / 576 grid round-trip ceiling of the DEV-A GT (scsam3, CPU torch; never the keyframe).

The round trip is MVSeed/refloss/seedgrid.py rt() verbatim: GT -> +-1 -> bilinear to 1008x1008 -> bilinear antialias to
n x n -> bilinear to 1008 -> bilinear to the image -> > 0 (n = 288 = the decoder's low_res grid; 576 = a 2x grid).
Per DEV-A pair of <pairs> (dev_score rows): J/F/JF of rt288(g) and rt576(g) against g (mvseg2d.metrics.jf_pair, the
same DAVIS arithmetic as eval_jf), the round trip's FN/FP pixels, and the overlap of the prediction's error pixels
(<masks>) with the 288 round trip's error pixels (same direction).  GPU hidden (CUDA_VISIBLE_DEVICES="").
    flock <gpu.lock> scripts/docker_run.sh scsam3 python geomem/runs/_e14_boundary-anatomy/bd_grid.py \
        --pairs <dev_pairs.json> --masks <root> --out <json>
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, S1)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mvseg2d.io import load_labelmap, read_pred_mask  # noqa: E402
from mvseg2d.metrics import bound_pix, disk, jf_pair  # noqa: E402

DATA = os.path.join(S1, "Data")


def rt(g, n):
    t = torch.from_numpy(g.astype(np.float32))[None, None] * 2 - 1
    lo = F.interpolate(F.interpolate(t, size=(1008, 1008), mode="bilinear", align_corners=False), size=(n, n),
                       mode="bilinear", align_corners=False, antialias=True)
    hi = F.interpolate(F.interpolate(lo, size=(1008, 1008), mode="bilinear", align_corners=False), size=g.shape,
                       mode="bilinear", align_corners=False)
    return hi[0, 0].numpy() > 0


def job(args):
    torch.set_num_threads(2)
    masks, scene, t, cam, rows = args
    gt_lab = load_labelmap(os.path.join(DATA, scene, "frames", f"{t:06d}", "masks", f"{cam}.png"))
    H, W = gt_lab.shape
    bp = bound_pix(H, W)
    se = disk(bp)
    out = []
    for row in rows:
        o = int(row["obj"])
        g = gt_lab == o
        p = os.path.join(masks, scene, f"{t:06d}", cam, f"{o}.png")
        s = read_pred_mask(p) if os.path.isfile(p) else np.zeros((H, W), bool)
        r = dict(scene=scene, t=t, cam=cam, obj=o)
        for n in (288, 576):
            q = rt(g, n)
            J, Fm = jf_pair(g, q, bp, se)
            r[f"g{n}"], r[f"g{n}_J"], r[f"g{n}_F"] = float((J + Fm) / 2), float(J), float(Fm)
            if n == 288:
                fn_q, fp_q = g & ~q, q & ~g
                fn_s, fp_s = g & ~s, s & ~g
                r.update(fn288=int(fn_q.sum()), fp288=int(fp_q.sum()), fn_both=int((fn_s & fn_q).sum()),
                         fp_both=int((fp_s & fp_q).sum()), fn_s=int(fn_s.sum()), fp_s=int(fp_s.sum()))
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    views = {}
    for r in json.load(open(a.pairs))["rows"]:
        views.setdefault((r["scene"], r["t"], r["cam"]), []).append(r)
    jobs = [(a.masks, s, t, c, rows) for (s, t, c), rows in sorted(views.items())]
    with Pool(a.procs) as pool:
        rows = [r for rs in pool.imap_unordered(job, jobs, chunksize=1) for r in rs]
    rows.sort(key=lambda r: (r["scene"], r["t"], r["cam"], r["obj"]))
    meta = dict(pairs=os.path.abspath(a.pairs), masks=os.path.abspath(a.masks), torch=torch.__version__,
                seconds=time.time() - t0, n=len(rows))
    json.dump(dict(meta=meta, rows=rows), open(a.out, "w"))
    print(f"done {len(rows)} rows {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

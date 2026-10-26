#!/usr/bin/env python3
"""Score stage-1 masks of one CROOM scene against the annotated label maps, the way scripts/ring_diag/d1_d5/score.py
scores T06 (docs/croom-annotation.md 6):

    python scripts/croom/score.py --scene CROOM_T04_001000_r00 --out <json> <name>=<mask_dir> [...]

<mask_dir>/cam_KK/<obj>.png (the run.py layout; a missing PNG = empty prediction).  For every non-reference camera and
every object of meta.basic_objects whose GT in that camera is non-empty at 1/S resolution: J at 1/S (nearest resize)
and at full resolution, `covers` = the GT object the prediction overlaps most (0: empty / overlaps no object),
wrong = covers not in (0, obj).  Ring distance d = min(|i - r|, n - |i - r|); near = d 1..6, far = d 7..16 (32 cams).
Writes {"scene", "ref_cam", "objects", "arms": {name: summary}, "rows": {name: rows}}; re-runs add / replace arms.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, S1)
from annot.labels import read_label  # noqa: E402

NEAR_MAX = 6


def ring_distance(i, r, n):
    return min(abs(i - r), n - abs(i - r))


def load_gt(annot_dir, cams, scale):
    full, q = {}, {}
    for c in cams:
        lab = read_label(os.path.join(annot_dir, "labels", f"{int(c[4:])}.png"))
        im = Image.fromarray(lab)
        full[c] = lab
        q[c] = np.array(im.resize((im.width // scale, im.height // scale), Image.NEAREST))
    return full, q


def load_pred(d, c, p, scale):
    f = os.path.join(d, c, f"{p}.png")
    if not os.path.isfile(f):
        return None, None
    m = Image.open(f)
    return np.array(m) > 127, np.array(m.resize((m.width // scale, m.height // scale), Image.NEAREST)) > 127


def score_dir(d, cams, ref, objects, GTF, GTQ, scale):
    n, r = len(cams), cams.index(ref)
    rows = []
    for i, c in enumerate(cams):
        if c == ref:
            continue
        dist = ring_distance(i, r, n)
        for p in objects:
            gtq = GTQ[c] == p
            if gtq.sum() == 0:
                continue
            pf, pq = load_pred(d, c, p, scale)
            if pq is None or pq.sum() == 0:
                cov, frac = 0, 0.0
            else:
                counts = {j: int((pq & (GTQ[c] == j)).sum()) for j in objects}
                cov = max(counts, key=counts.get) if sum(counts.values()) else 0
                frac = counts[cov] / pq.sum() if cov else 0.0
            Jq = float((pq & gtq).sum() / max((pq | gtq).sum(), 1)) if pq is not None else 0.0
            gtf = GTF[c] == p
            Jf = float((pf & gtf).sum() / max((pf | gtf).sum(), 1)) if pf is not None else 0.0
            rows.append(dict(cam=c, d=dist, p=p, J=Jq, J_full=Jf, covers=cov, cover_frac=float(frac),
                             pred_px_q=int(pq.sum()) if pq is not None else 0))
    return rows


def summarise(rows, n_cams=32):
    d_max = n_cams // 2

    def sub(lo, hi):
        return [r for r in rows if lo <= r["d"] <= hi]
    out = {"n_pairs": len(rows)}
    for name, (lo, hi) in (("all", (1, d_max)), ("near", (1, NEAR_MAX)), ("far", (NEAR_MAX + 1, d_max))):
        R = sub(lo, hi)
        out[f"J_{name}"] = float(np.mean([r["J"] for r in R])) if R else None
        out[f"Jfull_{name}"] = float(np.mean([r["J_full"] for r in R])) if R else None
        out[f"wrong_{name}"] = sum(r["covers"] not in (0, r["p"]) for r in R)
        out[f"empty_{name}"] = sum(r["covers"] == 0 for r in R)
        out[f"fail_{name}"] = sum(r["J"] < 0.5 for r in R)
        out[f"n_{name}"] = len(R)
    out["J_by_d"] = {str(k): (float(np.mean([r["J"] for r in rows if r["d"] == k])) if any(r["d"] == k for r in rows) else None)
                     for k in range(1, d_max + 1)}
    out["J_by_obj"] = {str(p): float(np.mean([r["J"] for r in rows if r["p"] == p])) for p in sorted({r["p"] for r in rows})}
    out["swap_pattern"] = [f"{a}->{b} x{n}" for (a, b), n in
                           Counter((r["p"], r["covers"]) for r in rows if r["covers"] not in (0, r["p"])).most_common()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=os.path.join(S1, "Data"))
    ap.add_argument("--annot-root", default=os.path.join(S1, "Data", "_annot", "CROOM"), dest="annot_root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("arms", nargs="+", help="<name>=<mask_dir>")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.data, a.scene, "meta.json")))
    an = meta["annot"]
    annot_dir = os.path.join(a.annot_root, an["take"], f"{an['frame']:06d}", "annot")
    cams, ref, objects = list(meta["cams"]), meta["ref_cam"], [int(o) for o in meta["basic_objects"]]
    GTF, GTQ = load_gt(annot_dir, cams, a.scale)
    res = json.load(open(a.out)) if os.path.isfile(a.out) else {"arms": {}, "rows": {}}
    res.update(scene=a.scene, ref_cam=ref, objects=objects, scale=a.scale, annot_dir=annot_dir)
    for spec in a.arms:
        name, d = spec.split("=", 1)
        assert os.path.isdir(d), d
        rows = score_dir(d, cams, ref, objects, GTF, GTQ, a.scale)
        res["arms"][name] = dict(summarise(rows, len(cams)), mask_dir=d)
        res["rows"][name] = rows
        s = res["arms"][name]
        print(f"{name:16s} n {s['n_pairs']} J {s['J_all']:.4f} (full {s['Jfull_all']:.4f})  near {s['J_near']:.3f}  "
              f"far {s['J_far']:.3f}  wrong near/far {s['wrong_near']}/{s['wrong_far']}  empty near/far "
              f"{s['empty_near']}/{s['empty_far']}  J<0.5 {s['fail_all']}  swaps {s['swap_pattern'][:4]}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

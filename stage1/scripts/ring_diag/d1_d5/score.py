"""Score T06 frame-900 stage-1 arms exactly as stage1/scripts/ring_diag/swap_diag.py does, plus full-resolution J.

    score.py <out.json> <arm>=<mask_dir> [...]

<mask_dir>/cam_KK/<p>.png (the stage-1 CLI layout; a missing PNG = empty prediction).  For each non-reference camera
(cam_06 excluded) and each person p in 1..6 whose GT is non-empty at 1/4 resolution: J at 1/4 resolution (nearest
resize, swap_diag's rule), J at full 3840x2160 resolution, `covers` = the GT person with the largest overlap with the
predicted mask (0 when the mask is empty or overlaps no person), wrong = covers not in (0, p), empty = covers == 0.
Ring distance d = min(|i-6|, 32-|i-6|); near = d 1-6, far = d 7-16.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

REPO = "/home/sjpark/Documents/SCSegmentation"
MASK = os.path.join(REPO, "Data/PlenopticFull/T06_flag_ganggang_F/Mask")
REF = "cam_06"
CAMS = [f"cam_{i:02d}" for i in range(32)]
S = 4


def load_gt():
    full, q = {}, {}
    for c in CAMS:
        m = Image.open(os.path.join(MASK, c, "000900.png"))
        full[c] = np.array(m)
        q[c] = np.array(m.resize((m.width // S, m.height // S), Image.NEAREST))
    return full, q


def load_pred(d, c, p):
    f = os.path.join(d, c, f"{p}.png")
    if not os.path.isfile(f):
        return None, None
    m = Image.open(f)
    return np.array(m) > 127, np.array(m.resize((m.width // S, m.height // S), Image.NEAREST)) > 127


def score_dir(d, GTF, GTQ):
    rows = []
    for c in CAMS:
        if c == REF:
            continue
        i = int(c[4:])
        dist = min(abs(i - 6), 32 - abs(i - 6))
        for p in range(1, 7):
            gtq = GTQ[c] == p
            if gtq.sum() == 0:
                continue
            pf, pq = load_pred(d, c, p)
            if pq is None or pq.sum() == 0:
                cov, frac = 0, 0.0
            else:
                counts = [(pq & (GTQ[c] == j)).sum() for j in range(7)]
                cov = int(np.argmax(counts[1:]) + 1) if sum(counts[1:]) else 0
                frac = counts[cov] / pq.sum() if cov else 0.0
            Jq = float((pq & gtq).sum() / max((pq | gtq).sum(), 1)) if pq is not None else 0.0
            gtf = GTF[c] == p
            Jf = float((pf & gtf).sum() / max((pf | gtf).sum(), 1)) if pf is not None else 0.0
            rows.append(dict(cam=c, d=dist, p=p, J=Jq, J_full=Jf, covers=cov, cover_frac=float(frac),
                             pred_px_q=int(pq.sum()) if pq is not None else 0))
    return rows


def summarise(rows):
    def sub(lo, hi):
        return [r for r in rows if lo <= r["d"] <= hi]
    out = {"n_pairs": len(rows)}
    for name, (lo, hi) in (("all", (1, 16)), ("near", (1, 6)), ("far", (7, 16))):
        R = sub(lo, hi)
        out[f"J_{name}"] = float(np.mean([r["J"] for r in R]))
        out[f"Jfull_{name}"] = float(np.mean([r["J_full"] for r in R]))
        out[f"wrong_{name}"] = sum(r["covers"] not in (0, r["p"]) for r in R)
        out[f"empty_{name}"] = sum(r["covers"] == 0 for r in R)
        out[f"fail_{name}"] = sum(r["J"] < 0.5 for r in R)
        out[f"n_{name}"] = len(R)
    out["J_by_d"] = {str(k): float(np.mean([r["J"] for r in rows if r["d"] == k])) for k in range(1, 17)}
    out["swap_pattern"] = [f"p{a}->p{b} x{n}" for (a, b), n in
                           Counter((r["p"], r["covers"]) for r in rows if r["covers"] not in (0, r["p"])).most_common()]
    return out


def main():
    out_json = sys.argv[1]
    arms = [a.split("=", 1) for a in sys.argv[2:]]
    GTF, GTQ = load_gt()
    res = json.load(open(out_json)) if os.path.isfile(out_json) else {"arms": {}, "rows": {}}
    for name, d in arms:
        assert os.path.isdir(d), d
        rows = score_dir(d, GTF, GTQ)
        res["arms"][name] = dict(summarise(rows), mask_dir=d)
        res["rows"][name] = rows
        s = res["arms"][name]
        print(f"{name:22s} n {s['n_pairs']} J {s['J_all']:.4f} (full {s['Jfull_all']:.4f})  near {s['J_near']:.3f}  "
              f"far {s['J_far']:.3f}  wrong near/far {s['wrong_near']}/{s['wrong_far']}  empty near/far "
              f"{s['empty_near']}/{s['empty_far']}  J<0.5 {s['fail_all']}", flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
